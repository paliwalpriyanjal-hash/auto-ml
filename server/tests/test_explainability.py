import io
import json
import os
import sys
import unittest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression

# Matplotlib headless config
import matplotlib
matplotlib.use("Agg")

# Server root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from app.main import app
from app.services.model_registry_service import (
    register_model,
    delete_model,
    init_model_registry,
    load_model,
)

client = TestClient(app)


class TestExplainabilityEngine(unittest.TestCase):
    """
    Automated Test Suite for Phase 2: Model Explainability Engine.
    Tests capabilities, permutation importance, SHAP attributions,
    schema contract enforcement, and artifact immutability.
    """

    @classmethod
    def setUpClass(cls):
        init_model_registry()

        # 1. Register a Classification Tree Model (Random Forest)
        cls.rf_clf = Pipeline([
            ("scaler", StandardScaler()),
            ("estimator", RandomForestClassifier(n_estimators=10, random_state=42))
        ])
        cls.X_clf = pd.DataFrame({
            "age": [25, 45, 35, 50, 23, 60, 48, 33, 55, 40],
            "income": [30000, 70000, 50000, 90000, 25000, 110000, 85000, 48000, 95000, 60000],
        })
        cls.y_clf = pd.Series([0, 1, 0, 1, 0, 1, 1, 0, 1, 0], name="approved")
        cls.rf_clf.fit(cls.X_clf, cls.y_clf)

        reg_clf = register_model(
            pipeline=cls.rf_clf,
            model_name="TestRF_Classifier",
            task_type="classification",
            model_type="RandomForestClassifier",
            dataset_name="credit.csv",
            target_column="approved",
            feature_columns=["age", "income"],
            metrics={"accuracy": 1.0}
        )
        cls.clf_model_id = reg_clf["id"]

        # 2. Register a Regression Model (Linear Regression)
        cls.lin_reg = Pipeline([
            ("scaler", StandardScaler()),
            ("estimator", LinearRegression())
        ])
        cls.X_reg = pd.DataFrame({
            "sqft": [1000, 1500, 2000, 2500, 3000, 1200, 1800, 2200],
            "bedrooms": [2, 3, 3, 4, 4, 2, 3, 4],
        })
        cls.y_reg = pd.Series([200000, 300000, 400000, 500000, 600000, 240000, 360000, 440000], name="price")
        cls.lin_reg.fit(cls.X_reg, cls.y_reg)

        reg_reg = register_model(
            pipeline=cls.lin_reg,
            model_name="TestLinear_Regressor",
            task_type="regression",
            model_type="LinearRegression",
            dataset_name="housing.csv",
            target_column="price",
            feature_columns=["sqft", "bedrooms"],
            metrics={"r2": 1.0}
        )
        cls.reg_model_id = reg_reg["id"]

    @classmethod
    def tearDownClass(cls):
        delete_model(cls.clf_model_id)
        delete_model(cls.reg_model_id)

    # -------------------------------------------------------------
    # 1. Capabilities Endpoint
    # -------------------------------------------------------------
    def test_01_explainability_capabilities(self):
        """GET /models/{id}/explainability/capabilities returns supported methods and metadata."""
        res = client.get(f"/models/{self.clf_model_id}/explainability/capabilities")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["model_id"], self.clf_model_id)
        self.assertEqual(data["task_type"], "classification")
        self.assertEqual(data["feature_columns"], ["age", "income"])
        methods = [m["method"] for m in data["supported_methods"]]
        self.assertIn("permutation", methods)
        self.assertIn("shap", methods)

    def test_02_unknown_model_capabilities_returns_404(self):
        """Requesting capabilities for non-existent model ID returns 404."""
        res = client.get("/models/999999/explainability/capabilities")
        self.assertEqual(res.status_code, 404)

    # -------------------------------------------------------------
    # 2. Permutation Importance
    # -------------------------------------------------------------
    def test_03_permutation_importance_classification(self):
        """Permutation importance executes on evaluation data for classification."""
        eval_df = pd.DataFrame({
            "age": [28, 48, 38, 52, 25, 62],
            "income": [32000, 72000, 52000, 92000, 27000, 112000],
            "approved": [0, 1, 0, 1, 0, 1]
        })
        payload = {
            "rows": eval_df.to_dict(orient="records"),
            "target_column": "approved",
            "n_repeats": 3,
            "scoring": "f1_macro"
        }
        res = client.post(f"/models/{self.clf_model_id}/explainability/permutation", json=payload)
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()

        self.assertEqual(data["method"], "permutation")
        self.assertEqual(data["scoring_metric"], "f1_macro")
        self.assertEqual(len(data["features"]), 2)
        feat_names = [f["feature"] for f in data["features"]]
        self.assertIn("age", feat_names)
        self.assertIn("income", feat_names)
        for f in data["features"]:
            self.assertIn("importance_mean", f)
            self.assertIn("importance_std", f)
            self.assertIn("rank", f)

    def test_04_permutation_importance_regression(self):
        """Permutation importance executes on evaluation data for regression."""
        eval_df = pd.DataFrame({
            "sqft": [1100, 1600, 2100, 2600, 3100, 1300],
            "bedrooms": [2, 3, 3, 4, 4, 2],
            "price": [220000, 320000, 420000, 520000, 620000, 260000]
        })
        payload = {
            "rows": eval_df.to_dict(orient="records"),
            "target_column": "price",
            "n_repeats": 3,
            "scoring": "r2"
        }
        res = client.post(f"/models/{self.reg_model_id}/explainability/permutation", json=payload)
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()

        self.assertEqual(data["task_type"], "regression")
        self.assertEqual(data["scoring_metric"], "r2")
        self.assertEqual(len(data["features"]), 2)

    def test_05_permutation_missing_target_returns_400(self):
        """Permutation importance without target column in dataset returns clear 400."""
        payload = {
            "rows": [
                {"age": 30, "income": 40000},
                {"age": 40, "income": 60000},
                {"age": 50, "income": 80000},
                {"age": 60, "income": 100000},
                {"age": 70, "income": 120000},
            ],
            "target_column": "missing_approved"
        }
        res = client.post(f"/models/{self.clf_model_id}/explainability/permutation", json=payload)
        self.assertEqual(res.status_code, 400)
        self.assertIn("target", res.json()["detail"].lower())

    def test_06_permutation_missing_required_features_returns_400(self):
        """Permutation importance with missing required features returns 400."""
        payload = {
            "rows": [
                {"age": 30, "approved": 0},
                {"age": 40, "approved": 1},
                {"age": 50, "approved": 1},
                {"age": 60, "approved": 1},
                {"age": 70, "approved": 1},
            ],
            "target_column": "approved"
        }
        res = client.post(f"/models/{self.clf_model_id}/explainability/permutation", json=payload)
        self.assertEqual(res.status_code, 400)
        self.assertIn("missing_columns", res.json()["detail"])

    def test_07_permutation_extra_features_returns_400(self):
        """Permutation importance with unexpected extra features returns 400."""
        payload = {
            "rows": [
                {"age": 30, "income": 40000, "extra_col": 999, "approved": 0},
                {"age": 40, "income": 60000, "extra_col": 999, "approved": 1},
                {"age": 50, "income": 80000, "extra_col": 999, "approved": 1},
                {"age": 60, "income": 100000, "extra_col": 999, "approved": 1},
                {"age": 70, "income": 120000, "extra_col": 999, "approved": 1},
            ],
            "target_column": "approved"
        }
        res = client.post(f"/models/{self.clf_model_id}/explainability/permutation", json=payload)
        self.assertEqual(res.status_code, 400)
        self.assertIn("extra_columns", res.json()["detail"])

    def test_08_permutation_csv_file_upload(self):
        """Permutation importance accepts multipart CSV file upload."""
        csv_text = "age,income,approved\n25,30000,0\n45,70000,1\n35,50000,0\n55,90000,1\n60,100000,1\n"
        files = {"file": ("eval.csv", io.BytesIO(csv_text.encode("utf-8")), "text/csv")}
        res = client.post(
            f"/models/{self.clf_model_id}/explainability/permutation",
            data={"target_column": "approved", "n_repeats": 3},
            files=files
        )
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()
        self.assertEqual(data["rows_evaluated"], 5)

    # -------------------------------------------------------------
    # 3. SHAP Explanations
    # -------------------------------------------------------------
    def test_09_shap_tree_model_explanation(self):
        """SHAP explanations generate global importance and local row attributions for tree models."""
        eval_df = pd.DataFrame({
            "age": [28, 48, 38],
            "income": [32000, 72000, 52000],
            "approved": [0, 1, 0]
        })
        payload = {
            "rows": eval_df.to_dict(orient="records"),
            "target_column": "approved",
            "max_rows": 3
        }
        res = client.post(f"/models/{self.clf_model_id}/explainability/shap", json=payload)
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()

        self.assertEqual(data["method"], "shap")
        self.assertEqual(data["explainer_type"], "TreeExplainer")
        self.assertIn("global_importance", data)
        self.assertEqual(len(data["global_importance"]), 2)

        # Local explanations check
        self.assertIn("local_explanations", data)
        self.assertEqual(len(data["local_explanations"]), 3)
        loc = data["local_explanations"][0]
        self.assertIn("base_value", loc)
        self.assertIn("prediction", loc)
        self.assertIn("contributions", loc)
        self.assertEqual(len(loc["contributions"]), 2)
        contrib_names = [c["feature"] for c in loc["contributions"]]
        self.assertIn("age", contrib_names)
        self.assertIn("income", contrib_names)

    def test_10_shap_linear_model_explanation(self):
        """SHAP explanations work for linear models via LinearExplainer."""
        eval_df = pd.DataFrame({
            "sqft": [1100, 1600, 2100],
            "bedrooms": [2, 3, 3],
        })
        payload = {
            "rows": eval_df.to_dict(orient="records"),
            "max_rows": 3
        }
        res = client.post(f"/models/{self.reg_model_id}/explainability/shap", json=payload)
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()
        self.assertEqual(data["method"], "shap")
        self.assertEqual(data["explainer_type"], "LinearExplainer")
        self.assertEqual(len(data["local_explanations"]), 3)

    # -------------------------------------------------------------
    # 4. Model Immutability & Safety
    # -------------------------------------------------------------
    def test_11_model_artifact_not_mutated_during_explanation(self):
        """Requesting explanations does NOT refit or alter the saved model weights."""
        test_row = pd.DataFrame([{"age": 42, "income": 65000}])

        # Baseline prediction
        pipe_before, _ = load_model(self.clf_model_id)
        pred_before = pipe_before.predict(test_row)

        # Run permutation importance and SHAP
        eval_df = pd.DataFrame({
            "age": [20, 30, 40, 50, 60],
            "income": [20000, 40000, 60000, 80000, 100000],
            "approved": [0, 0, 1, 1, 1]
        })
        client.post(
            f"/models/{self.clf_model_id}/explainability/permutation",
            json={"rows": eval_df.to_dict(orient="records"), "target_column": "approved"}
        )
        client.post(
            f"/models/{self.clf_model_id}/explainability/shap",
            json={"rows": eval_df.to_dict(orient="records")}
        )

        # Prediction after explanations
        pipe_after, _ = load_model(self.clf_model_id)
        pred_after = pipe_after.predict(test_row)

        np.testing.assert_array_equal(pred_before, pred_after)

    def test_12_json_safety_no_nan_or_inf(self):
        """All responses must be strictly JSON compliant with no unhandled NaN or Inf values."""
        eval_df = pd.DataFrame({
            "age": [20, 30, 40, 50, 60],
            "income": [20000, 40000, 60000, 80000, 100000],
            "approved": [0, 0, 1, 1, 1]
        })
        res = client.post(
            f"/models/{self.clf_model_id}/explainability/permutation",
            json={"rows": eval_df.to_dict(orient="records"), "target_column": "approved"}
        )
        self.assertEqual(res.status_code, 200)
        # json.loads ensures valid RFC 8259 JSON parsing
        parsed = json.loads(res.text)
        self.assertIsInstance(parsed, dict)


if __name__ == "__main__":
    unittest.main()
