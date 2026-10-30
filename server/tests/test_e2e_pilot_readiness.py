# server/tests/test_e2e_pilot_readiness.py
# AutoVeyra Phase 8: End-to-End Pilot Readiness & Release Candidate Smoke Test Suite

import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Add server root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from app.main import app
from app.config import settings
from app.services.model_registry_service import register_model, delete_model, init_model_registry
from app.services.report_service import init_report_db, generate_model_report
from app.services.monitoring_service import init_monitoring_db, get_monitoring_stats
from app.services.auth_service import init_auth_db, register_user, authenticate_user, create_access_token


class TestE2EPilotReadiness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_auth_db()
        init_model_registry()
        init_report_db()
        init_monitoring_db()
        cls.client = TestClient(app)

        # Load deterministic synthetic demo dataset
        repo_root = Path(__file__).resolve().parents[2]
        cls.demo_csv_path = repo_root / "demo_data" / "customer_churn_synthetic.csv"
        if not cls.demo_csv_path.exists():
            raise FileNotFoundError(f"Demo dataset not found at {cls.demo_csv_path}")

        cls.demo_df = pd.read_csv(cls.demo_csv_path)

    def test_01_health_and_readiness_probes(self):
        """Verify operational health and database readiness probes."""
        res_health = self.client.get("/health")
        self.assertEqual(res_health.status_code, 200)
        self.assertEqual(res_health.json().get("status"), "healthy")

        res_ready = self.client.get("/ready")
        self.assertEqual(res_ready.status_code, 200)
        ready_data = res_ready.json()
        self.assertEqual(ready_data.get("status"), "ready")
        self.assertIn("checks", ready_data)

    def test_02_data_profiling_and_cleaning(self):
        """Verify CSV upload, data profiling and dataset ingestion."""
        csv_bytes = self.demo_df.to_csv(index=False).encode("utf-8")
        files = {"file": ("churn.csv", io.BytesIO(csv_bytes), "text/csv")}
        
        # 1. Dataset Profiling
        prof_res = self.client.post("/features/profile", files=files, data={"target_column": "Churn"})
        self.assertEqual(prof_res.status_code, 200)
        prof_data = prof_res.json()
        self.assertIn("dataset_summary", prof_data)
        self.assertIn("columns", prof_data)

        # 2. Advanced Data Cleaning
        files_clean = {"file": ("churn.csv", io.BytesIO(csv_bytes), "text/csv")}
        clean_res = self.client.post("/clean", files=files_clean, data={"preprocessing": "{}"})
        self.assertEqual(clean_res.status_code, 200)
        clean_data = clean_res.json()
        self.assertIn("stats", clean_data)
        self.assertIn("csv", clean_data)

    def test_03_feature_generation_and_selection(self):
        """Verify automated feature engineering preview and strategy evaluation."""
        csv_bytes = self.demo_df.to_csv(index=False).encode("utf-8")
        
        # 1. Preview Transformations
        files_prev = {"file": ("churn.csv", io.BytesIO(csv_bytes), "text/csv")}
        prev_res = self.client.post(
            "/features/preview",
            files=files_prev,
            data={"clip_outliers": "true", "group_rare_categories": "true"}
        )
        self.assertEqual(prev_res.status_code, 200)
        prev_data = prev_res.json()
        self.assertIn("transformed_columns", prev_data)
        self.assertIn("sample_preview", prev_data)

        # 2. Evaluate Feature Strategies
        files_eval = {"file": ("churn.csv", io.BytesIO(csv_bytes), "text/csv")}
        eval_res = self.client.post(
            "/features/evaluate",
            files=files_eval,
            data={
                "task_type": "classification",
                "target_column": "Churn",
                "model_type": "RandomForestClassifier",
                "n_splits": 2,
                "selection_method": "variance"
            }
        )
        self.assertEqual(eval_res.status_code, 200)
        eval_data = eval_res.json()
        self.assertIn("winning_strategy", eval_data)
        self.assertIn("strategy_results", eval_data)
        self.assertIn("baseline_holdout_metrics", eval_data)

    def test_04_training_and_comparison_pipeline(self):
        """Verify model training and multi-algorithm cross-validation comparison."""
        csv_bytes = self.demo_df.to_csv(index=False).encode("utf-8")

        # 1. Single Model Training
        files_train = {"file": ("churn.csv", io.BytesIO(csv_bytes), "text/csv")}
        train_res = self.client.post(
            "/train",
            files=files_train,
            data={
                "task_type": "classification",
                "target_column": "Churn",
                "model": "RandomForestClassifier",
                "scaler": "standard",
                "splitRatio": 0.8,
                "missing": "mean",
                "encoding": "onehot"
            }
        )
        self.assertEqual(train_res.status_code, 200)
        train_data = train_res.json()
        self.assertIn("metrics", train_data)
        self.assertIn("accuracy", train_data["metrics"])

        # 2. Multi-Model Comparison
        files_comp = {"file": ("churn.csv", io.BytesIO(csv_bytes), "text/csv")}
        comp_res = self.client.post(
            "/compare",
            files=files_comp,
            data={
                "task_type": "classification",
                "target_column": "Churn",
                "scaler": "standard",
                "splitRatio": 0.8,
                "missing": "median",
                "encoding": "onehot"
            }
        )
        self.assertEqual(comp_res.status_code, 200)
        comp_data = comp_res.json()
        self.assertIn("model_results", comp_data)
        self.assertIn("best_model", comp_data)

    def test_05_optuna_hyperparameter_tuning(self):
        """Verify Optuna hyperparameter optimization execution and trial logging."""
        csv_bytes = self.demo_df.to_csv(index=False).encode("utf-8")
        files = {"file": ("churn.csv", io.BytesIO(csv_bytes), "text/csv")}

        tune_res = self.client.post(
            "/tuning/start",
            files=files,
            data={
                "target_column": "Churn",
                "task_type": "classification",
                "model_type": "RandomForestClassifier",
                "n_trials": 2,
                "timeout_seconds": 30
            }
        )
        self.assertEqual(tune_res.status_code, 200)
        tune_data = tune_res.json()
        self.assertIn("run_id", tune_data)
        self.assertIn("tuned_metrics", tune_data)
        run_id = tune_data["run_id"]

        # Verify trial history query
        trials_res = self.client.get(f"/tuning/{run_id}/trials")
        self.assertEqual(trials_res.status_code, 200)
        trials_data = trials_res.json()
        self.assertIn("trials", trials_data)
        self.assertGreaterEqual(len(trials_data["trials"]), 1)

    def test_06_model_registry_served_inference_and_cache(self):
        """Verify model registration, LRU serving cache, latency, and schema enforcement."""
        features = ["Age", "TenureMonths", "MonthlyCharges", "SupportCalls"]
        X = self.demo_df[features]
        y = self.demo_df["Churn"]

        clf = RandomForestClassifier(n_estimators=5, random_state=42)
        clf.fit(X, y)

        reg_info = register_model(
            pipeline=clf,
            model_name="E2E_ChurnModel",
            task_type="classification",
            model_type="RandomForestClassifier",
            metrics={"accuracy": 0.88},
            feature_columns=features
        )
        model_id = reg_info["id"]

        try:
            # First inference -> cold load (cached=False)
            row = {"Age": 45, "TenureMonths": 12, "MonthlyCharges": 70.0, "SupportCalls": 2}
            res1 = self.client.post(f"/models/{model_id}/predict", json={"rows": [row]})
            self.assertEqual(res1.status_code, 200)
            data1 = res1.json()
            self.assertIn("predictions", data1)
            self.assertFalse(data1["cached"])

            # Second inference -> warm cache hit (cached=True)
            res2 = self.client.post(f"/models/{model_id}/predict", json={"rows": [row]})
            self.assertEqual(res2.status_code, 200)
            data2 = res2.json()
            self.assertTrue(data2["cached"])
            self.assertLess(data2["latency_ms"], 50.0)

            # Strict schema mismatch rejection
            bad_row = {"Age": 45, "InvalidColumn": 999}
            res_bad = self.client.post(f"/models/{model_id}/predict", json={"rows": [bad_row]})
            self.assertEqual(res_bad.status_code, 400)

            # Explainability: Permutation Importance
            csv_bytes = self.demo_df[features + ["Churn"]].to_csv(index=False).encode("utf-8")
            files = {"file": ("eval.csv", io.BytesIO(csv_bytes), "text/csv")}
            perm_res = self.client.post(
                f"/models/{model_id}/explainability/permutation",
                files=files,
                data={"target_column": "Churn", "n_repeats": 2}
            )
            self.assertEqual(perm_res.status_code, 200)
            perm_data = perm_res.json()
            self.assertIn("features", perm_data)
            self.assertIn("top_features", perm_data)

            # Explainability: SHAP Waterfall
            files_shap = {"file": ("eval.csv", io.BytesIO(csv_bytes), "text/csv")}
            shap_res = self.client.post(
                f"/models/{model_id}/explainability/shap",
                files=files_shap,
                data={"target_column": "Churn", "max_rows": 10}
            )
            self.assertEqual(shap_res.status_code, 200)
            shap_data = shap_res.json()
            self.assertIn("global_importance", shap_data)
            self.assertIn("local_explanations", shap_data)

            # Reporting: Generate PDF, HTML, JSON evaluation report
            rep_res = self.client.post(
                "/reports/generate",
                json={"model_id": model_id, "report_title": "E2E Churn Report"}
            )
            self.assertEqual(rep_res.status_code, 200)
            rep_data = rep_res.json()
            report_id = rep_data["report_id"]

            # Download PDF report
            pdf_res = self.client.get(f"/reports/{report_id}/download?format=pdf")
            self.assertEqual(pdf_res.status_code, 200)
            self.assertTrue(pdf_res.content.startswith(b"%PDF"))

            # Download HTML report
            html_res = self.client.get(f"/reports/{report_id}/download?format=html")
            self.assertEqual(html_res.status_code, 200)
            self.assertIn("<html", html_res.text.lower())

            # Download JSON report
            json_res = self.client.get(f"/reports/{report_id}/download?format=json")
            self.assertEqual(json_res.status_code, 200)
            self.assertEqual(json_res.json()["report_id"], report_id)

            # Monitoring: Verify stats and drift
            mon_res = self.client.get("/monitoring/stats")
            self.assertEqual(mon_res.status_code, 200)
            self.assertGreaterEqual(mon_res.json()["total_inferences"], 2)

            drift_res = self.client.get(f"/monitoring/drift/{model_id}")
            self.assertEqual(drift_res.status_code, 200)
            self.assertEqual(drift_res.json()["model_id"], model_id)

        finally:
            delete_model(model_id)

    def test_07_isolated_backup_and_restoration(self):
        """Verify zero-downtime backup creation and restoration in an isolated sandbox directory."""
        try:
            from scripts.backup_databases import backup_databases
        except ImportError:
            from server.scripts.backup_databases import backup_databases

        temp_dir = tempfile.mkdtemp()
        try:
            backup_databases(Path(temp_dir))
            
            # Find the created snapshot directory
            snapshots = list(Path(temp_dir).glob("autoveyra_backup_*"))
            self.assertGreaterEqual(len(snapshots), 1)
            snapshot_dir = snapshots[0]
            
            # Check databases directory
            db_dir = snapshot_dir / "databases"
            self.assertTrue(db_dir.exists())
            self.assertTrue((db_dir / "model_registry.db").exists())
            self.assertTrue((db_dir / "auth.db").exists())
            self.assertTrue((db_dir / "monitoring.db").exists())
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_08_authentication_mode_toggle_and_unauthorized_checks(self):
        """Verify behavior in permissive local dev mode vs. strict JWT auth mode."""
        # 1. Dev Mode (AUTH_ENABLED=False) -> Access allowed
        with patch.object(settings, "AUTH_ENABLED", False):
            res_dev = self.client.get("/auth/me")
            self.assertEqual(res_dev.status_code, 200)
            self.assertEqual(res_dev.json().get("username"), "local_admin")

        # 2. Strict Auth Mode (AUTH_ENABLED=True)
        with patch.object(settings, "AUTH_ENABLED", True):
            # Missing header -> 401
            res_unauth = self.client.get("/auth/me")
            self.assertEqual(res_unauth.status_code, 401)

            # Invalid token -> 401
            res_bad = self.client.get("/auth/me", headers={"Authorization": "Bearer fake.token"})
            self.assertEqual(res_bad.status_code, 401)

            # Register and login a real user in auth.db
            unique_name = f"pilot_user_{abs(hash(self)) % 100000}"
            self.client.post("/auth/register", json={"username": unique_name, "password": "PilotPassword123!"})
            login_res = self.client.post("/auth/login", json={"username": unique_name, "password": "PilotPassword123!"})
            user_token = login_res.json()["access_token"]
            
            res_auth_ok = self.client.get("/auth/me", headers={"Authorization": f"Bearer {user_token}"})
            self.assertEqual(res_auth_ok.status_code, 200)
            self.assertEqual(res_auth_ok.json()["username"], unique_name)


if __name__ == "__main__":
    unittest.main()
