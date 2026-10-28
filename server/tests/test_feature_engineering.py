# tests/test_feature_engineering.py

import io
import os
import sys
import unittest
import numpy as np
import pandas as pd

# Add server root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from app.main import app
from app.services.feature_service import init_feature_db


class TestFeatureEngineering(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_feature_db()
        cls.client = TestClient(app)

    def _make_sample_csv(self, n_rows=80):
        np.random.seed(42)
        dates = pd.date_range("2025-01-01", periods=n_rows, freq="D").astype(str)
        feature_num1 = np.random.randn(n_rows) * 10
        feature_num2 = feature_num1 * 0.98 + np.random.randn(n_rows) * 0.01  # highly collinear
        feature_nonneg = np.random.exponential(scale=5.0, size=n_rows)
        feature_const = [42.0] * n_rows
        feature_cat = np.random.choice(["cat_A", "cat_B", "rare_X"], p=[0.7, 0.25, 0.05], size=n_rows)
        target = (feature_num1 * 1.5 + feature_nonneg * 0.5 > 10).astype(int)

        df = pd.DataFrame({
            "record_date": dates,
            "feature_num1": feature_num1,
            "feature_collinear": feature_num2,
            "feature_nonneg": feature_nonneg,
            "feature_constant": feature_const,
            "category": feature_cat,
            "target": target,
        })
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        return buf

    def _make_regression_csv(self, n_rows=80):
        np.random.seed(42)
        x1 = np.random.randn(n_rows) * 5
        x2 = np.random.randn(n_rows) * 10
        x3_const = [1.0] * n_rows
        target = 3.0 * x1 - 1.2 * x2 + np.random.randn(n_rows) * 0.2
        df = pd.DataFrame({
            "x1": x1,
            "x2": x2,
            "x_constant": x3_const,
            "cat": np.random.choice(["low", "med", "high"], size=n_rows),
            "target": target,
        })
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        return buf

    def test_01_profile_dataset(self):
        """Test dataset profiling: column types, quality alerts, collinear redundant pairs."""
        csv_file = self._make_sample_csv()
        res = self.client.post(
            "/features/profile",
            files={"file": ("profile_test.csv", csv_file, "text/csv")},
            data={"target_column": "target", "correlation_threshold": 0.85},
        )
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()

        self.assertIn("dataset_summary", data)
        self.assertIn("columns", data)
        self.assertIn("quality_alerts", data)
        self.assertIn("redundant_pairs", data)
        self.assertIn("record_date", data.get("datetime_candidates", []))
        self.assertIn("feature_constant", data.get("constant_columns", []))

        # Check collinear redundant pair detection
        collinear_pairs = data["redundant_pairs"]
        self.assertTrue(len(collinear_pairs) >= 1)
        pair = collinear_pairs[0]
        self.assertIn("feature_num1", [pair["feature_a"], pair["feature_b"]])

    def test_02_preview_transformations(self):
        """Test transformation preview on sample rows."""
        csv_file = self._make_sample_csv()
        res = self.client.post(
            "/features/preview",
            files={"file": ("preview_test.csv", csv_file, "text/csv")},
            data={
                "datetime_cols": '["record_date"]',
                "log_cols": '["feature_nonneg"]',
                "clip_outliers": True,
                "group_rare_categories": True,
            },
        )
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()

        self.assertIn("created_columns", data)
        self.assertIn("dropped_columns", data)
        self.assertIn("record_date_year", data["created_columns"])
        self.assertIn("record_date", data["dropped_columns"])
        self.assertTrue(len(data["sample_preview"]) > 0)

    def test_03_evaluate_classification_feature_strategies(self):
        """Test multi-strategy comparison on classification task."""
        csv_file = self._make_sample_csv()
        res = self.client.post(
            "/features/evaluate",
            files={"file": ("eval_cls.csv", csv_file, "text/csv")},
            data={
                "task_type": "classification",
                "target_column": "target",
                "model_type": "RandomForestClassifier",
                "ranking_metric": "f1_macro",
                "n_splits": 2,
                "split_ratio": 0.8,
                "datetime_cols": '["record_date"]',
                "log_cols": '["feature_nonneg"]',
                "selection_method": "model_based",
                "k_best_count": 5,
            },
        )
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()

        self.assertIn("winning_strategy", data)
        self.assertEqual(len(data["strategy_results"]), 4)  # baseline, engineered, selected, combined
        self.assertIn("baseline_holdout_metrics", data)
        self.assertIn("winning_holdout_metrics", data)
        self.assertIn("delta_metrics", data)
        self.assertIsNotNone(data["registered_model_id"])

    def test_04_evaluate_regression_variance_threshold(self):
        """Test variance threshold selection on regression task."""
        csv_file = self._make_regression_csv()
        res = self.client.post(
            "/features/evaluate",
            files={"file": ("eval_reg.csv", csv_file, "text/csv")},
            data={
                "task_type": "regression",
                "target_column": "target",
                "model_type": "DecisionTree",
                "ranking_metric": "rmse",
                "n_splits": 2,
                "selection_method": "variance",
                "variance_threshold": 0.001,
            },
        )
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()

        self.assertEqual(data["task_type"], "regression")
        self.assertIn("rmse", data["winning_holdout_metrics"])

    def test_05_evaluate_k_best_selection(self):
        """Test K-Best feature selection strategy."""
        csv_file = self._make_sample_csv()
        res = self.client.post(
            "/features/evaluate",
            files={"file": ("eval_kbest.csv", csv_file, "text/csv")},
            data={
                "task_type": "classification",
                "target_column": "target",
                "model_type": "DecisionTreeClassifier",
                "selection_method": "k_best",
                "k_best_count": 3,
                "n_splits": 2,
            },
        )
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()
        self.assertIn("registered_model_id", data)

    def test_06_evaluate_correlation_filter(self):
        """Test correlation filter feature selection."""
        csv_file = self._make_sample_csv()
        res = self.client.post(
            "/features/evaluate",
            files={"file": ("eval_corr.csv", csv_file, "text/csv")},
            data={
                "task_type": "classification",
                "target_column": "target",
                "model_type": "LogisticRegression",
                "selection_method": "correlation",
                "correlation_threshold": 0.85,
                "n_splits": 2,
            },
        )
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()
        self.assertIn("winning_strategy", data)

    def test_07_holdout_leakage_isolation(self):
        """Verify strict sample count partitioning and holdout isolation."""
        csv_file = self._make_sample_csv(n_rows=100)
        res = self.client.post(
            "/features/evaluate",
            files={"file": ("eval_leak.csv", csv_file, "text/csv")},
            data={
                "task_type": "classification",
                "target_column": "target",
                "model_type": "DecisionTreeClassifier",
                "split_ratio": 0.75,
                "n_splits": 2,
            },
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        dq = data["data_quality"]
        self.assertEqual(dq["total_rows"], 100)
        self.assertEqual(dq["train_samples"], 75)
        self.assertEqual(dq["test_samples"], 25)

    def test_08_registered_model_inference(self):
        """Verify that a feature-engineered registered pipeline can be queried for inference."""
        csv_file = self._make_sample_csv()
        res = self.client.post(
            "/features/evaluate",
            files={"file": ("eval_pred.csv", csv_file, "text/csv")},
            data={
                "task_type": "classification",
                "target_column": "target",
                "model_type": "RandomForestClassifier",
                "datetime_cols": '["record_date"]',
                "log_cols": '["feature_nonneg"]',
                "n_splits": 2,
            },
        )
        self.assertEqual(res.status_code, 200)
        reg_id = res.json()["registered_model_id"]
        self.assertIsNotNone(reg_id)

        # Call prediction endpoint with raw original schema
        pred_res = self.client.post(
            f"/models/{reg_id}/predict",
            json={
                "rows": [
                    {
                        "record_date": "2025-06-01",
                        "feature_num1": 1.2,
                        "feature_collinear": 1.18,
                        "feature_nonneg": 4.5,
                        "feature_constant": 42.0,
                        "category": "cat_A",
                    },
                    {
                        "record_date": "2025-06-02",
                        "feature_num1": -2.0,
                        "feature_collinear": -1.95,
                        "feature_nonneg": 0.5,
                        "feature_constant": 42.0,
                        "category": "cat_B",
                    },
                ]
            },
        )
        self.assertEqual(pred_res.status_code, 200, pred_res.text)
        pred_data = pred_res.json()
        self.assertEqual(len(pred_data["predictions"]), 2)

    def test_09_registered_model_explainability(self):
        """Verify explainability capabilities on registered feature pipeline."""
        csv_file = self._make_sample_csv()
        res = self.client.post(
            "/features/evaluate",
            files={"file": ("eval_exp.csv", csv_file, "text/csv")},
            data={
                "task_type": "classification",
                "target_column": "target",
                "model_type": "DecisionTreeClassifier",
                "n_splits": 2,
            },
        )
        self.assertEqual(res.status_code, 200)
        reg_id = res.json()["registered_model_id"]

        cap_res = self.client.get(f"/models/{reg_id}/explainability/capabilities")
        self.assertEqual(cap_res.status_code, 200)
        methods = [m["method"] for m in cap_res.json().get("supported_methods", [])]
        self.assertIn("permutation", methods)

    def test_10_history_listing_and_deletion(self):
        """Test GET /features/history, GET /features/history/{id}, and DELETE."""
        csv_file = self._make_sample_csv()
        eval_res = self.client.post(
            "/features/evaluate",
            files={"file": ("eval_hist.csv", csv_file, "text/csv")},
            data={
                "task_type": "classification",
                "target_column": "target",
                "model_type": "LogisticRegression",
                "n_splits": 2,
            },
        )
        self.assertEqual(eval_res.status_code, 200)
        run_id = eval_res.json()["run_id"]

        # List
        list_res = self.client.get("/features/history")
        self.assertEqual(list_res.status_code, 200)
        self.assertIn(run_id, [item["run_id"] for item in list_res.json()["items"]])

        # Detail
        detail_res = self.client.get(f"/features/history/{run_id}")
        self.assertEqual(detail_res.status_code, 200)
        self.assertEqual(detail_res.json()["run_id"], run_id)

        # Delete
        del_res = self.client.delete(f"/features/history/{run_id}")
        self.assertEqual(del_res.status_code, 200)
        self.assertEqual(self.client.get(f"/features/history/{run_id}").status_code, 404)

    def test_11_invalid_task_and_inputs(self):
        """Verify error handling on invalid tasks and parameters."""
        csv_file = self._make_sample_csv()
        res = self.client.post(
            "/features/evaluate",
            files={"file": ("invalid.csv", csv_file, "text/csv")},
            data={
                "task_type": "invalid_task",
                "target_column": "target",
                "model_type": "DecisionTreeClassifier",
            },
        )
        self.assertEqual(res.status_code, 400)

    def test_12_feature_pipeline_tuning_compatibility(self):
        """Verify that a dataset configured for feature modeling can also be tuned seamlessly via Optuna."""
        csv_file = self._make_regression_csv()
        tune_res = self.client.post(
            "/tuning/start",
            files={"file": ("tune_compat.csv", csv_file, "text/csv")},
            data={
                "task_type": "regression",
                "target_column": "target",
                "model_type": "DecisionTree",
                "metric": "rmse",
                "n_trials": 2,
                "n_splits": 2,
            },
        )
        self.assertEqual(tune_res.status_code, 200, tune_res.text)
        self.assertIn("best_params", tune_res.json())


if __name__ == "__main__":
    unittest.main()
