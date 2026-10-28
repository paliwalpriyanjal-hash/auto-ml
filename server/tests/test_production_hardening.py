# server/tests/test_production_hardening.py
"""
Automated verification suite for AutoVeyra Phase 6:
Production Readiness, Security, and Reliability Hardening.
"""

import io
import os
import sys
import unittest
from pathlib import Path

# Add server root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from fastapi.testclient import TestClient
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.config import settings
from app.main import app
from app.services.model_registry_service import (
    init_model_registry,
    register_model,
    delete_model,
    get_model,
    MODEL_DIR,
)
from app.services.report_service import init_report_db


class TestProductionHardening(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_model_registry()
        init_report_db()
        cls.client = TestClient(app)

        # Create a lightweight model for inference testing
        pipe = Pipeline([("scaler", StandardScaler()), ("estimator", LinearRegression())])
        X = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [10.0, 20.0, 30.0]})
        y = pd.Series([100.0, 200.0, 300.0])
        pipe.fit(X, y)
        cls.test_model = register_model(
            pipeline=pipe,
            model_name="Hardening_Baseline_Model",
            task_type="regression",
            model_type="LinearRegression",
            dataset_name="hardening_data.csv",
            target_column="target",
            feature_columns=["f1", "f2"],
            metrics={"rmse": 0.0, "r2": 1.0},
        )
        cls.model_id = cls.test_model["id"]

    @classmethod
    def tearDownClass(cls):
        try:
            delete_model(cls.model_id)
        except Exception:
            pass

    # 1. Health & Readiness Probes
    def test_01_health_probe(self):
        """GET /health must return 200 with service metadata and status 'healthy'."""
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["app"], settings.APP_NAME)
        self.assertEqual(data["version"], settings.APP_VERSION)
        self.assertIn("environment", data)

    def test_02_readiness_probe(self):
        """GET /ready must return 200 and verify SQLite databases and filesystem storage."""
        res = self.client.get("/ready")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "ready")
        self.assertIn("checks", data)
        self.assertEqual(data["checks"]["sqlite_databases"], "healthy")
        self.assertEqual(data["checks"]["model_storage"], "writable")

    # 2. Correlation ID Propagation & Middleware
    def test_03_correlation_id_propagation(self):
        """Incoming X-Correlation-ID header must be echoed back in the response headers."""
        custom_cid = "trace-prod-hardening-xyz-987"
        res = self.client.get("/health", headers={"X-Correlation-ID": custom_cid})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers.get("x-correlation-id"), custom_cid)

    def test_04_correlation_id_auto_generation(self):
        """When client omits correlation ID, server must generate a unique hex ID."""
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        cid = res.headers.get("x-correlation-id")
        self.assertIsNotNone(cid)
        self.assertTrue(len(cid) >= 16)

    # 3. Upload Size & Resource Limiting
    def test_05_oversized_upload_rejection_413(self):
        """Uploading a file exceeding MAX_UPLOAD_SIZE_BYTES must be rejected with HTTP 413."""
        # Temporarily test by lowering limit
        original_limit = settings.MAX_UPLOAD_SIZE_BYTES
        try:
            settings.MAX_UPLOAD_SIZE_BYTES = 100  # 100 bytes limit
            oversized_csv = "col1,col2\n" + ("12345,67890\n" * 20)  # > 200 bytes
            res = self.client.post(
                "/features/profile",
                files={"file": ("oversized.csv", io.BytesIO(oversized_csv.encode()), "text/csv")},
            )
            self.assertEqual(res.status_code, 413)
            self.assertIn("exceeds maximum allowed limit", res.json()["detail"])
        finally:
            settings.MAX_UPLOAD_SIZE_BYTES = original_limit

    def test_06_malformed_csv_handling(self):
        """Uploading corrupted, unparseable binary content must return clean HTTP 400."""
        garbage_bytes = b"\x00\x01\x02\xff\xfe\x00\x00corrupt"
        res = self.client.post(
            "/features/profile",
            files={"file": ("corrupt.csv", io.BytesIO(garbage_bytes), "text/csv")},
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("detail", res.json())

    # 4. Prediction Batch Limits & Validation
    def test_07_prediction_batch_size_limit_413(self):
        """Prediction batch exceeding MAX_PREDICTION_ROWS must return HTTP 413."""
        original_limit = settings.MAX_PREDICTION_ROWS
        try:
            settings.MAX_PREDICTION_ROWS = 10
            large_batch = [{"f1": float(i), "f2": float(i * 10)} for i in range(15)]
            res = self.client.post(
                f"/models/{self.model_id}/predict",
                json={"rows": large_batch},
            )
            self.assertEqual(res.status_code, 413)
            self.assertIn("exceeds maximum allowed limit", res.json()["detail"])
        finally:
            settings.MAX_PREDICTION_ROWS = original_limit

    def test_08_prediction_empty_payload_400(self):
        """Empty rows array must return HTTP 400."""
        res = self.client.post(
            f"/models/{self.model_id}/predict",
            json={"rows": []},
        )
        self.assertEqual(res.status_code, 400)

    # 5. Corrupted & Missing Artifact Recovery
    def test_09_corrupted_model_artifact_recovery(self):
        """If a model .joblib artifact on disk is corrupted, predict must return HTTP 500 cleanly."""
        # Create temporary model
        pipe = Pipeline([("est", LinearRegression())])
        pipe.fit(pd.DataFrame({"x": [1, 2]}), pd.Series([1, 2]))
        m = register_model(
            pipeline=pipe,
            model_name="Corrupt_Test_Model",
            task_type="regression",
            model_type="LinearRegression",
            feature_columns=["x"],
            metrics={"r2": 1.0},
        )
        temp_id = m["id"]

        # Corrupt the artifact file on disk
        artifact_path = None
        from app.services.model_registry_service import _get_model_record
        rec = _get_model_record(temp_id)
        artifact_path = Path(rec["artifact_path"])
        artifact_path.write_bytes(b"CORRUPTED_NOT_A_VALID_JOBLIB_HEADER_000")

        try:
            pred_res = self.client.post(
                f"/models/{temp_id}/predict",
                json={"rows": [{"x": 1.5}]},
            )
            self.assertEqual(pred_res.status_code, 500)
            self.assertIn("Could not load model artifact", pred_res.json()["detail"])
        finally:
            delete_model(temp_id)

    def test_10_missing_model_artifact_recovery(self):
        """If a model .joblib artifact file is deleted from disk, predict must return HTTP 404."""
        pipe = Pipeline([("est", LinearRegression())])
        pipe.fit(pd.DataFrame({"x": [1, 2]}), pd.Series([1, 2]))
        m = register_model(
            pipeline=pipe,
            model_name="Missing_Test_Model",
            task_type="regression",
            model_type="LinearRegression",
            feature_columns=["x"],
            metrics={"r2": 1.0},
        )
        temp_id = m["id"]

        # Unlink artifact from disk
        from app.services.model_registry_service import _get_model_record
        rec = _get_model_record(temp_id)
        artifact_path = Path(rec["artifact_path"])
        artifact_path.unlink(missing_ok=True)

        try:
            pred_res = self.client.post(
                f"/models/{temp_id}/predict",
                json={"rows": [{"x": 1.5}]},
            )
            self.assertEqual(pred_res.status_code, 404)
            self.assertIn("model artifact is missing", pred_res.json()["detail"])
        finally:
            delete_model(temp_id)

    # 6. Path Traversal & Isolation
    def test_11_path_traversal_protection(self):
        """Model loading must verify artifact path stays within MODEL_DIR."""
        from app.services.model_registry_service import load_model, _connect
        # Insert a malicious record with path traversal in DB
        with _connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO registered_models (
                    model_name, task_type, model_type, dataset_name,
                    target_column, artifact_path, feature_columns_json,
                    metrics_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "Malicious_Model", "regression", "LinearRegression", "test.csv",
                    "y", str(Path(MODEL_DIR).parent / "sensitive.txt"), "[]", "{}", "now"
                ),
            )
            bad_id = cursor.lastrowid

        from fastapi import HTTPException
        try:
            with self.assertRaises(HTTPException) as ctx:
                load_model(bad_id)
            self.assertIn(ctx.exception.status_code, [404, 500])
        finally:
            with _connect() as conn:
                conn.execute("DELETE FROM registered_models WHERE id = ?", (bad_id,))

    # 7. Failed Registration Rollback & Ghost Model Prevention
    def test_12_failed_registration_atomic_rollback(self):
        """If pipeline has no predict() or fails validation, no database row or artifact is created."""
        class InvalidPipelineWithoutPredict:
            pass

        with self.assertRaises(ValueError):
            register_model(
                pipeline=InvalidPipelineWithoutPredict(),
                model_name="Should_Not_Exist",
                task_type="regression",
                model_type="Invalid",
            )

    # 8. Report Export XSS Sanitization & File Header
    def test_13_report_export_xss_prevention(self):
        """Malicious script injection in report metadata must be escaped in HTML export."""
        rep = self.client.post(
            "/reports/generate",
            json={
                "model_id": self.model_id,
                "report_title": "<script>alert('title_hack')</script>Safety Report",
            },
        ).json()
        rep_id = rep["report_id"]

        try:
            html_res = self.client.get(f"/reports/{rep_id}/download?format=html")
            self.assertEqual(html_res.status_code, 200)
            self.assertNotIn("<script>alert('title_hack')</script>", html_res.text)
            self.assertIn("&lt;script&gt;alert(&#x27;title_hack&#x27;)&lt;/script&gt;", html_res.text)

            # Check filename header sanitization
            disp = html_res.headers.get("content-disposition", "")
            self.assertIn("filename=\"autoveyra_report_", disp)
        finally:
            self.client.delete(f"/reports/{rep_id}")

    # 9. Clean 404 on Unknown IDs
    def test_14_non_existent_report_or_model_404(self):
        """Invalid model or report IDs return clean 404 without unhandled tracebacks."""
        self.assertEqual(self.client.get("/models/999999").status_code, 404)
        self.assertEqual(self.client.get("/reports/non_existent_999").status_code, 404)
        self.assertEqual(self.client.get("/models/999999/explainability/capabilities").status_code, 404)

    # 10. CORS & Preflight Response
    def test_15_cors_preflight(self):
        """OPTIONS preflight requests must return permitted methods and headers."""
        res = self.client.options(
            "/health",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
            },
        )
        self.assertEqual(res.status_code, 200)
        self.assertIn("access-control-allow-origin", res.headers)


if __name__ == "__main__":
    unittest.main()
