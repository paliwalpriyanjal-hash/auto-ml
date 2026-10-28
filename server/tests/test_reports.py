# tests/test_reports.py

import io
import os
import sys
import unittest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor

# Add server root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import HTTPException
from fastapi.testclient import TestClient
from app.main import app
from app.services.report_service import init_report_db, get_report
from app.services.model_registry_service import init_model_registry, register_model, delete_model, get_model


class TestModelReporting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_model_registry()
        init_report_db()
        cls.client = TestClient(app)

        # Create sample classification model
        X_clf = pd.DataFrame({
            "feature_1": [1.0, 2.5, 3.2, 4.8, 5.1, 6.0],
            "feature_2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        })
        y_clf = pd.Series([0, 0, 1, 1, 1, 0])
        clf_pipe = Pipeline([("scaler", StandardScaler()), ("estimator", RandomForestClassifier(n_estimators=5, random_state=42, n_jobs=1))])
        clf_pipe.fit(X_clf, y_clf)
        cls.clf_reg = register_model(
            pipeline=clf_pipe,
            model_name="Reporting_Test_Classifier",
            task_type="classification",
            model_type="RandomForestClassifier",
            dataset_name="churn.csv",
            target_column="churned",
            feature_columns=["feature_1", "feature_2"],
            metrics={"accuracy": 0.85, "f1_macro": 0.83, "precision_macro": 0.84, "recall_macro": 0.82},
        )
        cls.clf_id = cls.clf_reg["id"]

        # Create sample regression model
        X_reg = pd.DataFrame({"sqft": [1000.0, 1500.0, 2000.0], "rooms": [2.0, 3.0, 4.0]})
        y_reg = pd.Series([200.0, 300.0, 400.0])
        reg_pipe = Pipeline([("scaler", StandardScaler()), ("estimator", DecisionTreeRegressor(random_state=42))])
        reg_pipe.fit(X_reg, y_reg)
        cls.reg_reg = register_model(
            pipeline=reg_pipe,
            model_name="Reporting_Test_Regressor",
            task_type="regression",
            model_type="DecisionTree",
            dataset_name="housing.csv",
            target_column="price",
            feature_columns=["sqft", "rooms"],
            metrics={"rmse": 12.5, "mae": 8.2, "r2": 0.91},
        )
        cls.reg_id = cls.reg_reg["id"]

    def test_01_generate_classification_report(self):
        """Test report generation for registered classification model."""
        res = self.client.post(
            "/reports/generate",
            json={
                "model_id": self.clf_id,
                "report_title": "Q4 Churn Model Evaluation",
                "model_card": {
                    "intended_use": "Customer churn risk assessment in telecom ops",
                    "out_of_scope_use": "Autonomous credit scoring",
                },
            },
        )
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()

        self.assertIn("report_id", data)
        self.assertEqual(data["model_id"], self.clf_id)
        self.assertEqual(data["task_type"], "classification")
        self.assertEqual(data["report_title"], "Q4 Churn Model Evaluation")

        # Verify sections
        self.assertIn("executive_summary", data)
        self.assertIn("dataset_profile", data)
        self.assertIn("evaluation_methodology", data)
        self.assertIn("performance_metrics", data)
        self.assertIn("model_card", data)
        self.assertIn("provenance", data)
        self.assertIn("limitations", data)

        card = data["model_card"]
        self.assertEqual(card["intended_use"], "Customer churn risk assessment in telecom ops")

    def test_02_generate_regression_report(self):
        """Test report generation for registered regression model."""
        res = self.client.post("/reports/generate", json={"model_id": self.reg_id})
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()

        self.assertEqual(data["task_type"], "regression")
        metrics = data["performance_metrics"]["metrics"]
        self.assertIn("rmse", metrics)
        self.assertEqual(metrics["rmse"], 12.5)

    def test_03_report_with_tuning_evidence(self):
        """Test that tuning and feature engineering evidence are gathered when available."""
        # Execute small tuning run which links registered_model_id
        csv_data = "x1,x2,target\n1,2,0\n3,4,1\n5,6,0\n7,8,1\n2,3,0\n6,7,1\n4,5,0\n8,9,1\n"
        tune_res = self.client.post(
            "/tuning/start",
            files={"file": ("test_tune.csv", io.BytesIO(csv_data.encode()), "text/csv")},
            data={
                "task_type": "classification",
                "target_column": "target",
                "model_type": "DecisionTreeClassifier",
                "n_trials": 2,
                "n_splits": 2,
            },
        )
        self.assertEqual(tune_res.status_code, 200)
        tuned_model_id = tune_res.json()["registered_model_id"]

        # Generate report for tuned model
        rep_res = self.client.post("/reports/generate", json={"model_id": tuned_model_id})
        self.assertEqual(rep_res.status_code, 200)
        rep_data = rep_res.json()

        tuning_sec = rep_data.get("tuning_history", {})
        self.assertEqual(tuning_sec.get("status"), "available")
        self.assertIn("best_params", tuning_sec)

    def test_04_missing_optional_metadata_graceful(self):
        """Verify report generates cleanly when tuning or feature history is absent."""
        res = self.client.post("/reports/generate", json={"model_id": self.clf_id})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["tuning_history"]["status"], "unavailable")
        self.assertIn("No hyperparameter tuning run was linked", data["tuning_history"]["message"])

    def test_05_model_card_editable_fields(self):
        """Verify PATCH /reports/{id}/model-card updates narrative fields without modifying measured metrics."""
        rep = self.client.post("/reports/generate", json={"model_id": self.clf_id}).json()
        rep_id = rep["report_id"]

        patch_res = self.client.patch(
            f"/reports/{rep_id}/model-card",
            json={
                "intended_use": "Strictly clinical research assistance",
                "known_limitations": "Not tested on pediatric cohorts",
            },
        )
        self.assertEqual(patch_res.status_code, 200)
        patched = patch_res.json()

        # Check narrative fields updated
        self.assertEqual(patched["model_card"]["intended_use"], "Strictly clinical research assistance")
        self.assertEqual(patched["model_card"]["known_limitations"], "Not tested on pediatric cohorts")

        # Verify quantitative metrics were untampered
        self.assertEqual(patched["performance_metrics"]["metrics"]["accuracy"], 0.85)

    def test_06_snapshot_immutability_on_deleted_model(self):
        """Verify historical report remains accessible even if the underlying model is deleted."""
        # Create temporary model
        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        y = pd.Series([0, 1, 0])
        pipe = Pipeline([("scaler", StandardScaler()), ("estimator", DecisionTreeRegressor())])
        pipe.fit(X, y)
        temp_model = register_model(
            pipeline=pipe,
            model_name="Temp_Model_ToDelete",
            task_type="regression",
            model_type="DecisionTree",
            dataset_name="temp.csv",
            target_column="target",
            feature_columns=["a", "b"],
            metrics={"r2": 0.88},
        )
        t_id = temp_model["id"]

        # Generate report
        rep = self.client.post("/reports/generate", json={"model_id": t_id}).json()
        rep_id = rep["report_id"]

        # Delete model from registry
        delete_model(t_id)
        with self.assertRaises(HTTPException):
            get_model(t_id)

        # Retrieve report - should succeed from immutable snapshot
        rep_get = self.client.get(f"/reports/{rep_id}")
        self.assertEqual(rep_get.status_code, 200)
        self.assertEqual(rep_get.json()["model_id"], t_id)
        self.assertEqual(rep_get.json()["performance_metrics"]["metrics"]["r2"], 0.88)

    def test_07_pdf_download_structure(self):
        """Test GET /reports/{id}/download?format=pdf produces valid binary PDF."""
        rep = self.client.post("/reports/generate", json={"model_id": self.clf_id}).json()
        rep_id = rep["report_id"]

        pdf_res = self.client.get(f"/reports/{rep_id}/download?format=pdf")
        self.assertEqual(pdf_res.status_code, 200)
        self.assertEqual(pdf_res.headers["content-type"], "application/pdf")
        self.assertTrue(pdf_res.content.startswith(b"%PDF-"))
        self.assertTrue(len(pdf_res.content) > 1000)

    def test_08_html_download_and_xss_sanitization(self):
        """Test HTML export and assert script injection characters are escaped."""
        rep = self.client.post("/reports/generate", json={"model_id": self.clf_id}).json()
        rep_id = rep["report_id"]

        # Patch with malicious script payload
        self.client.patch(
            f"/reports/{rep_id}/model-card",
            json={"intended_use": "<script>alert('xss_attack')</script>Normal intended use."},
        )

        html_res = self.client.get(f"/reports/{rep_id}/download?format=html")
        self.assertEqual(html_res.status_code, 200)
        self.assertIn("text/html", html_res.headers["content-type"])
        html_text = html_res.text

        # Must not contain unescaped script tag
        self.assertNotIn("<script>alert('xss_attack')</script>", html_text)
        self.assertIn("&lt;script&gt;alert(&#x27;xss_attack&#x27;)&lt;/script&gt;", html_text)

    def test_09_json_export_schema(self):
        """Test structured JSON report download."""
        rep = self.client.post("/reports/generate", json={"model_id": self.clf_id}).json()
        rep_id = rep["report_id"]

        json_res = self.client.get(f"/reports/{rep_id}/download?format=json")
        self.assertEqual(json_res.status_code, 200)
        self.assertEqual(json_res.headers["content-type"], "application/json")
        data = json_res.json()
        self.assertEqual(data["report_id"], rep_id)
        self.assertIn("model_card", data)

    def test_10_pagination_and_filtering(self):
        """Test GET /reports pagination and model_id filtering."""
        list_res = self.client.get(f"/reports?limit=2&offset=0&model_id={self.clf_id}")
        self.assertEqual(list_res.status_code, 200)
        data = list_res.json()
        self.assertIn("items", data)
        self.assertIn("total", data)
        self.assertEqual(data["limit"], 2)

    def test_11_report_deletion_preserves_model(self):
        """Test deleting a report record removes the report but does not delete the model."""
        rep = self.client.post("/reports/generate", json={"model_id": self.clf_id}).json()
        rep_id = rep["report_id"]

        del_res = self.client.delete(f"/reports/{rep_id}")
        self.assertEqual(del_res.status_code, 200)

        # Report should be 404
        self.assertEqual(self.client.get(f"/reports/{rep_id}").status_code, 404)

        # Underlying model in registry must remain intact!
        model = get_model(self.clf_id)
        self.assertIsNotNone(model)
        self.assertEqual(model["id"], self.clf_id)

    def test_12_invalid_identifiers(self):
        """Verify error handling on non-existent or invalid model/report IDs."""
        res1 = self.client.post("/reports/generate", json={"model_id": 999999})
        self.assertEqual(res1.status_code, 404)

        res2 = self.client.get("/reports/non_existent_report_id_123")
        self.assertEqual(res2.status_code, 404)

        res3 = self.client.get(f"/reports/{self.clf_id}/download?format=invalid_format")
        self.assertEqual(res3.status_code, 400)

    def test_13_provenance_and_schema_version(self):
        """Verify provenance records schema version and execution environment."""
        rep = self.client.post("/reports/generate", json={"model_id": self.clf_id}).json()
        prov = rep.get("provenance", {})
        self.assertEqual(prov["report_schema_version"], "1.0.0")
        self.assertIn("python_version", prov)
        self.assertIn("report_generated_at", prov)

    def test_14_bounded_resource_large_feature_list(self):
        """Verify report, PDF, and HTML generation remain bounded with 100+ feature columns."""
        feat_cols = [f"sensor_metric_{i}" for i in range(120)]
        X_large = pd.DataFrame({col: np.random.randn(10) for col in feat_cols})
        y_large = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        large_pipe = Pipeline([("scaler", StandardScaler()), ("estimator", DecisionTreeRegressor())])
        large_pipe.fit(X_large, y_large)
        large_model = register_model(
            pipeline=large_pipe,
            model_name="Large_Feature_Model",
            task_type="regression",
            model_type="DecisionTree",
            dataset_name="telemetry_large.csv",
            target_column="power",
            feature_columns=feat_cols,
            metrics={"rmse": 4.2},
        )
        l_id = large_model["id"]

        rep = self.client.post("/reports/generate", json={"model_id": l_id}).json()
        rep_id = rep["report_id"]

        # PDF generation should not crash or overflow
        pdf_res = self.client.get(f"/reports/{rep_id}/download?format=pdf")
        self.assertEqual(pdf_res.status_code, 200)

        # HTML generation should not crash
        html_res = self.client.get(f"/reports/{rep_id}/download?format=html")
        self.assertEqual(html_res.status_code, 200)


if __name__ == "__main__":
    unittest.main()
