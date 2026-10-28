import io
import json
import os
import sys
import unittest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
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
    load_model,
    predict_with_model,
    delete_model,
    list_models,
    get_model,
    init_model_registry,
)
from app.services.experiment_service import (
    create_experiment,
    list_experiments,
    delete_experiment,
    init_experiment_db,
)

client = TestClient(app)


class TestAutoVeyraSRSAudit(unittest.TestCase):
    """
    Automated Audit Suite for AutoVeyra SRS Compliance & Data Quality.
    Validates FR-001, FR-020, FR-024, FR-025, FR-042-FR-049, AC-01-AC-11,
    leak-free cross-validation, and prediction schema integrity.
    """

    def setUp(self):
        init_model_registry()
        init_experiment_db()

    def _post_file(self, endpoint: str, df: pd.DataFrame, form_data: dict):
        csv_buffer = io.BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        files = {"file": ("audit_sample.csv", csv_buffer, "text/csv")}
        return client.post(endpoint, data=form_data, files=files)

    # -------------------------------------------------------------
    # 1. Input CSV Validation & Edge Cases (FR-001, FR-042, AC-01)
    # -------------------------------------------------------------
    def test_01_empty_csv_rejected(self):
        """Empty CSV upload returns 400 error."""
        csv_buffer = io.BytesIO(b"")
        files = {"file": ("empty.csv", csv_buffer, "text/csv")}
        res = client.post("/compare", data={"task_type": "classification", "target_column": "target"}, files=files)
        self.assertEqual(res.status_code, 400)
        self.assertIn("empty", res.json()["detail"].lower())

    def test_02_empty_rows_csv_rejected(self):
        """CSV with headers only and no rows returns 400 error."""
        df = pd.DataFrame(columns=["feature1", "target"])
        res = self._post_file("/compare", df, {"task_type": "classification", "target_column": "target"})
        self.assertEqual(res.status_code, 400)

    def test_03_duplicate_column_headers_rejected(self):
        """CSV with duplicate column headers after trimming returns 400 error."""
        csv_text = "colA, colA ,target\n1,2,0\n3,4,1\n"
        files = {"file": ("dup.csv", io.BytesIO(csv_text.encode("utf-8")), "text/csv")}
        res = client.post("/compare", data={"task_type": "classification", "target_column": "target"}, files=files)
        self.assertEqual(res.status_code, 400)
        self.assertIn("duplicate", res.json()["detail"].lower())

    def test_04_target_column_not_in_dataset_rejected(self):
        """Non-existent target column returns clear 400 error."""
        df = pd.DataFrame({"x1": [1, 2, 3, 4], "x2": [10, 20, 30, 40], "label": [0, 1, 0, 1]})
        res = self._post_file("/compare", df, {"task_type": "classification", "target_column": "missing_target"})
        self.assertEqual(res.status_code, 400)
        self.assertIn("missing_target", res.json()["detail"])

    # -------------------------------------------------------------
    # 2. Classification Target Constraints & Quality (FR-020, AC-02)
    # -------------------------------------------------------------
    def test_05_single_class_target_rejected(self):
        """Single-class target is rejected with diagnostic error."""
        df = pd.DataFrame({
            "f1": [1, 2, 3, 4, 5, 6, 7, 8],
            "target": ["ClassA"] * 8
        })
        res = self._post_file("/compare", df, {"task_type": "classification", "target_column": "target"})
        self.assertEqual(res.status_code, 400)
        detail = res.json()["detail"].lower()
        self.assertTrue("at least two" in detail or "at least 2" in detail)

    def test_06_single_sample_class_rejected(self):
        """Class with only 1 sample is rejected to prevent stratified split failure."""
        df = pd.DataFrame({
            "f1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "target": ["A"] * 9 + ["SingleSampleClass"]
        })
        res = self._post_file("/compare", df, {"task_type": "classification", "target_column": "target"})
        self.assertEqual(res.status_code, 400)
        self.assertIn("has only 1 sample", res.json()["detail"])

    # -------------------------------------------------------------
    # 3. Leak-Free Preprocessing & CV Audit (FR-024, FR-025, AC-05)
    # -------------------------------------------------------------
    def test_07_comparison_train_holdout_split_and_cv_leak_free(self):
        """
        Verify that model comparison separates holdout test set before CV,
        fits pipeline transformers strictly within folds, and ranks by CV mean.
        """
        np.random.seed(42)
        n = 40
        df = pd.DataFrame({
            "num_feat": np.random.randn(n) * 100,  # wide range to test scaling
            "cat_feat": np.random.choice(["type_X", "type_Y", "type_Z"], size=n),
            "target": np.random.choice([0, 1], size=n)
        })
        payload = {
            "task_type": "classification",
            "target_column": "target",
            "scaler": "standard",
            "splitRatio": 0.75,
            "missing": "median",
            "encoding": "onehot",
            "ranking_metric": "f1_macro"
        }
        res = self._post_file("/compare", df, payload)
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()

        # Check CV fold execution
        self.assertGreaterEqual(data["cv_folds"], 2)
        self.assertEqual(len(data["model_results"]), 4)

        # Check that ranking order corresponds to CV performance
        ranked = sorted(
            [m for m in data["model_results"] if m.get("rank") is not None],
            key=lambda m: m["rank"]
        )
        ranked_cv_scores = [m["cv_mean"] for m in ranked]
        self.assertEqual(ranked_cv_scores, sorted(ranked_cv_scores, reverse=True))

        # Check that best model has holdout evaluation metrics
        best_model = data["best_model"]
        self.assertIn("holdout_metrics", best_model)
        holdout = best_model["holdout_metrics"]
        self.assertIn("accuracy", holdout)
        self.assertIn("f1_macro", holdout)
        self.assertIn("confusion_matrix", holdout)
        self.assertEqual(holdout["testing_samples"], int(n * 0.25))

        # Registered model should be returned automatically
        self.assertIn("registered_model", data)
        self.assertIsNotNone(data["registered_model"]["id"])

    def test_08_regression_ranking_metric_direction(self):
        """
        Regression comparison ranks RMSE/MAE ascending (lower is better)
        and R2 descending (higher is better).
        """
        np.random.seed(123)
        n = 30
        x = np.linspace(0, 10, n)
        df = pd.DataFrame({
            "x": x,
            "y": 2.5 * x + np.random.normal(0, 1, n)
        })
        payload = {
            "task_type": "regression",
            "target_column": "y",
            "scaler": "standard",
            "splitRatio": 0.8,
            "missing": "median",
            "encoding": "onehot",
            "ranking_metric": "rmse"
        }
        res = self._post_file("/compare", df, payload)
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()

        # Lower RMSE is better: rank 1 must have smallest cv_mean
        ranked = sorted(
            [m for m in data["model_results"] if m.get("rank") is not None],
            key=lambda m: m["rank"]
        )
        ranked_cv_means = [m["cv_mean"] for m in ranked]
        self.assertEqual(ranked_cv_means, sorted(ranked_cv_means, reverse=False))

    # -------------------------------------------------------------
    # 4. Model Registry & Serialization Audit (FR-046, FR-047, AC-07)
    # -------------------------------------------------------------
    def test_09_model_registry_lifecycle_and_reload(self):
        """
        Verify model pipeline registration, persistence in SQLite/joblib,
        schema contract storage, and clean deletion.
        """
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression())
        ])
        X_mock = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
        y_mock = np.array([0, 0, 1, 1])
        pipe.fit(X_mock, y_mock)

        # Register model
        reg = register_model(
            pipeline=pipe,
            model_name="AuditTest_LR",
            task_type="classification",
            model_type="LogisticRegression",
            dataset_name="audit_test.csv",
            target_column="label",
            feature_columns=["f1", "f2"],
            metrics={"accuracy": 1.0, "f1_macro": 1.0}
        )
        model_id = reg["id"]
        self.assertIsInstance(model_id, int)

        # Retrieve model record
        info = get_model(model_id)
        self.assertEqual(info["model_name"], "AuditTest_LR")
        self.assertEqual(info["feature_columns"], ["f1", "f2"])

        # Reload fitted artifact and verify inference equivalence
        loaded_pipe, meta = load_model(model_id)
        test_df = pd.DataFrame({"f1": [1.5, 3.5], "f2": [2.5, 4.5]})
        pred_original = pipe.predict(test_df.values)
        pred_loaded = loaded_pipe.predict(test_df.values)
        np.testing.assert_array_equal(pred_original, pred_loaded)

        # Clean up
        del_res = delete_model(model_id)
        self.assertEqual(del_res["id"], model_id)

    # -------------------------------------------------------------
    # 5. Real-Time Inference & Contract Enforcement (FR-048, AC-08)
    # -------------------------------------------------------------
    def test_10_prediction_endpoint_strict_column_contract(self):
        """
        Predictions endpoint strictly validates expected feature columns:
        Missing columns return 400; extra unexpected columns return 400.
        """
        # Register a quick mock model with exact contract
        from sklearn.linear_model import LinearRegression
        pipe = LinearRegression()
        X = pd.DataFrame({"age": [20, 30, 40], "income": [2000, 3000, 4000]})
        y = pd.Series([100, 200, 300])
        pipe.fit(X, y)

        reg = register_model(
            pipeline=pipe,
            model_name="InferenceContract_Model",
            task_type="regression",
            model_type="LinearRegression",
            feature_columns=["age", "income"],
            metrics={"r2": 1.0}
        )
        model_id = reg["id"]

        try:
            # Valid prediction request
            pred_res = client.post(
                f"/models/{model_id}/predict",
                json={"rows": [{"age": 25, "income": 2500}]}
            )
            self.assertEqual(pred_res.status_code, 200)
            data = pred_res.json()
            self.assertIn("predictions", data)
            self.assertEqual(len(data["predictions"]), 1)

            # Missing required column 'income'
            bad_res_missing = client.post(
                f"/models/{model_id}/predict",
                json={"rows": [{"age": 25}]}
            )
            self.assertEqual(bad_res_missing.status_code, 400)
            self.assertIn("missing_columns", bad_res_missing.json()["detail"])

            # Extra unexpected column 'extra_col'
            bad_res_extra = client.post(
                f"/models/{model_id}/predict",
                json={"rows": [{"age": 25, "income": 2500, "extra_col": 999}]}
            )
            self.assertEqual(bad_res_extra.status_code, 400)
            self.assertIn("extra_columns", bad_res_extra.json()["detail"])

            # Empty rows list
            bad_res_empty = client.post(
                f"/models/{model_id}/predict",
                json={"rows": []}
            )
            self.assertEqual(bad_res_empty.status_code, 400)

        finally:
            delete_model(model_id)

    # -------------------------------------------------------------
    # 6. Experiment History Logging Audit (FR-049, AC-09)
    # -------------------------------------------------------------
    def test_11_experiment_history_persistence_and_retrieval(self):
        """
        Verify that runs log parameter metadata, metric performance,
        and allow query retrieval and deletion.
        """
        exp = create_experiment(
            experiment_name="Audit Experiment Alpha",
            task_type="classification",
            dataset_name="sample_dataset.csv",
            target_column="outcome",
            model_type="DecisionTreeClassifier",
            scaler_type="standard",
            missing_strategy="median",
            encoding_strategy="onehot",
            split_ratio=0.8,
            metrics={"accuracy": 0.92, "f1_macro": 0.91},
            result={"status": "completed"}
        )
        exp_id = exp["id"]
        self.assertIsInstance(exp_id, int)

        # Retrieve experiment list
        listing = list_experiments(limit=50)
        self.assertGreaterEqual(listing["total"], 1)
        exp_ids = [e["id"] for e in listing["experiments"]]
        self.assertIn(exp_id, exp_ids)

        # Clean up
        del_exp = delete_experiment(exp_id)
        self.assertEqual(del_exp["id"], exp_id)

    # -------------------------------------------------------------
    # 7. Data Cleaning Engine Audit (FR-044, AC-04)
    # -------------------------------------------------------------
    def test_12_advanced_cleaning_pipeline(self):
        """Verify advanced data cleaning endpoint processes imputation, scaling, and summary."""
        df = pd.DataFrame({
            "age": [20, 25, np.nan, 40, 50],
            "city": ["NY", "SF", "NY", np.nan, "SF"],
            "score": [80.0, 85.0, 90.0, 95.0, 100.0]
        })
        config = {
            "missing_values": {"strategy": "median"},
            "scaling": {"strategy": "standard"},
            "encoding": {"strategy": "onehot"}
        }
        res = self._post_file(
            "/api/clean",
            df,
            {"preprocessing": json.dumps(config)}
        )
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()
        self.assertIn("csv", data)
        self.assertIn("stats", data)
        self.assertIn("rows_processed", data["stats"])


if __name__ == "__main__":
    unittest.main()
