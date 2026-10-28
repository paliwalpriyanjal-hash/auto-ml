# tests/test_tuning_engine.py

import io
import os
import sys
import unittest
import numpy as np
import pandas as pd

# Server root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient

from app.main import app
from app.services.tuning_service import (
    init_tuning_db,
    suggest_model_params,
    build_model_from_params,
    _calculate_delta_metrics,
    cancel_tuning_run,
)


class TestTuningEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_tuning_db()
        cls.client = TestClient(app)

    def _make_classification_csv(self, n_rows=60):
        np.random.seed(42)
        df = pd.DataFrame({
            "feature_num1": np.random.randn(n_rows),
            "feature_num2": np.random.rand(n_rows) * 100,
            "feature_cat": np.random.choice(["type_A", "type_B", "type_C"], size=n_rows),
            "target": np.random.choice([0, 1], size=n_rows),
        })
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        return buf

    def _make_regression_csv(self, n_rows=60):
        np.random.seed(42)
        x1 = np.random.randn(n_rows)
        x2 = np.random.rand(n_rows) * 50
        target = 2.5 * x1 - 0.5 * x2 + np.random.randn(n_rows) * 0.1
        df = pd.DataFrame({
            "feature_x1": x1,
            "feature_x2": x2,
            "category": np.random.choice(["low", "medium", "high"], size=n_rows),
            "target": target,
        })
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        return buf

    def test_tuning_classification_rf(self):
        """Test classification tuning with RandomForestClassifier."""
        csv_file = self._make_classification_csv()
        response = self.client.post(
            "/tuning/start",
            files={"file": ("test_cls.csv", csv_file, "text/csv")},
            data={
                "task_type": "classification",
                "target_column": "target",
                "model_type": "RandomForestClassifier",
                "metric": "f1_macro",
                "n_trials": 3,
                "n_splits": 2,
                "scaler": "standard",
                "missing": "median",
                "splitRatio": 0.8,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()

        self.assertIn("run_id", data)
        self.assertEqual(data["task_type"], "classification")
        self.assertEqual(data["model_type"], "RandomForestClassifier")
        self.assertEqual(data["metric"], "f1_macro")
        self.assertEqual(data["direction"], "maximize")
        self.assertIn("best_score", data)
        self.assertIn("best_params", data)
        self.assertIn("baseline_metrics", data)
        self.assertIn("tuned_metrics", data)
        self.assertIn("delta_metrics", data)
        self.assertIsNotNone(data["registered_model_id"])
        self.assertTrue(len(data["trials"]) >= 1)

    def test_tuning_regression_dt(self):
        """Test regression tuning with DecisionTree and RMSE minimization."""
        csv_file = self._make_regression_csv()
        response = self.client.post(
            "/tuning/start",
            files={"file": ("test_reg.csv", csv_file, "text/csv")},
            data={
                "task_type": "regression",
                "target_column": "target",
                "model_type": "DecisionTree",
                "metric": "rmse",
                "n_trials": 3,
                "n_splits": 2,
                "scaler": "minmax",
                "missing": "mean",
                "splitRatio": 0.8,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()

        self.assertEqual(data["task_type"], "regression")
        self.assertEqual(data["direction"], "minimize")
        self.assertIn("rmse", data["baseline_metrics"])
        self.assertIn("rmse", data["tuned_metrics"])
        self.assertIn("rmse", data["delta_metrics"])
        self.assertEqual(data["delta_metrics"]["rmse"]["direction"], "minimize")

    def test_tuning_linear_regression_diagnostic(self):
        """Verify LinearRegression raises clean diagnostic error as non-tunable."""
        csv_file = self._make_regression_csv()
        response = self.client.post(
            "/tuning/start",
            files={"file": ("test_reg.csv", csv_file, "text/csv")},
            data={
                "task_type": "regression",
                "target_column": "target",
                "model_type": "LinearRegression",
                "n_trials": 2,
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("tunable hyperparameters", response.json()["detail"])

    def test_tuning_invalid_model_or_metric(self):
        """Verify invalid model or metric returns 400 Bad Request."""
        csv_file = self._make_classification_csv()
        response = self.client.post(
            "/tuning/start",
            files={"file": ("test.csv", csv_file, "text/csv")},
            data={
                "task_type": "classification",
                "target_column": "target",
                "model_type": "UnknownModel123",
            },
        )
        self.assertEqual(response.status_code, 400)

        csv_file2 = self._make_classification_csv()
        response2 = self.client.post(
            "/tuning/start",
            files={"file": ("test.csv", csv_file2, "text/csv")},
            data={
                "task_type": "classification",
                "target_column": "target",
                "model_type": "DecisionTreeClassifier",
                "metric": "invalid_metric_xyz",
            },
        )
        self.assertEqual(response2.status_code, 400)

    def test_tuning_history_and_retrieval(self):
        """Test GET /tuning and GET /tuning/{run_id}."""
        csv_file = self._make_classification_csv()
        res_post = self.client.post(
            "/tuning/start",
            files={"file": ("test_cls.csv", csv_file, "text/csv")},
            data={
                "task_type": "classification",
                "target_column": "target",
                "model_type": "LogisticRegression",
                "metric": "accuracy",
                "n_trials": 2,
                "n_splits": 2,
            },
        )
        self.assertEqual(res_post.status_code, 200)
        run_id = res_post.json()["run_id"]

        # List runs
        list_res = self.client.get("/tuning")
        self.assertEqual(list_res.status_code, 200)
        list_data = list_res.json()
        self.assertIn("items", list_data)
        run_ids = [item["run_id"] for item in list_data["items"]]
        self.assertIn(run_id, run_ids)

        # Get run detail
        detail_res = self.client.get(f"/tuning/{run_id}")
        self.assertEqual(detail_res.status_code, 200)
        detail_data = detail_res.json()
        self.assertEqual(detail_data["run_id"], run_id)
        self.assertEqual(detail_data["model_type"], "LogisticRegression")
        self.assertIn("best_params", detail_data)
        self.assertIn("trials", detail_data)

        # Get trials leaderboard
        trials_res = self.client.get(f"/tuning/{run_id}/trials")
        self.assertEqual(trials_res.status_code, 200)
        self.assertEqual(trials_res.json()["run_id"], run_id)

    def test_registered_model_prediction_and_explainability(self):
        """Verify that a tuned registered model can be predicted on and explained."""
        csv_file = self._make_classification_csv()
        res_post = self.client.post(
            "/tuning/start",
            files={"file": ("test_cls.csv", csv_file, "text/csv")},
            data={
                "task_type": "classification",
                "target_column": "target",
                "model_type": "DecisionTreeClassifier",
                "metric": "f1_macro",
                "n_trials": 2,
                "n_splits": 2,
            },
        )
        self.assertEqual(res_post.status_code, 200)
        reg_model_id = res_post.json()["registered_model_id"]
        self.assertIsNotNone(reg_model_id)

        # Predict
        pred_res = self.client.post(
            f"/models/{reg_model_id}/predict",
            json={
                "rows": [
                    {"feature_num1": 0.5, "feature_num2": 25.0, "feature_cat": "type_A"},
                    {"feature_num1": -1.2, "feature_num2": 80.0, "feature_cat": "type_B"},
                ]
            },
        )
        self.assertEqual(pred_res.status_code, 200)
        pred_data = pred_res.json()
        self.assertEqual(len(pred_data["predictions"]), 2)

        # Explainability capabilities
        cap_res = self.client.get(f"/models/{reg_model_id}/explainability/capabilities")
        self.assertEqual(cap_res.status_code, 200)
        cap_data = cap_res.json()
        methods = [m["method"] for m in cap_data.get("supported_methods", [])]
        self.assertIn("permutation", methods)

    def test_tuning_delete_run(self):
        """Test DELETE /tuning/{run_id}."""
        csv_file = self._make_regression_csv()
        res_post = self.client.post(
            "/tuning/start",
            files={"file": ("test_del.csv", csv_file, "text/csv")},
            data={
                "task_type": "regression",
                "target_column": "target",
                "model_type": "RandomForest",
                "metric": "mae",
                "n_trials": 2,
                "n_splits": 2,
            },
        )
        self.assertEqual(res_post.status_code, 200)
        run_id = res_post.json()["run_id"]

        del_res = self.client.delete(f"/tuning/{run_id}")
        self.assertEqual(del_res.status_code, 200)

        # Verify 404 after deletion
        get_res = self.client.get(f"/tuning/{run_id}")
        self.assertEqual(get_res.status_code, 404)

    def test_cancel_endpoint(self):
        """Test POST /tuning/{run_id}/cancel endpoint."""
        cancel_res = self.client.post("/tuning/dummy_run_123/cancel")
        self.assertEqual(cancel_res.status_code, 200)
        self.assertIn("Cancellation requested", cancel_res.json()["message"])

    def test_tuning_extra_trees_classifier(self):
        """Test ExtraTreesClassifier hyperparameter search."""
        csv_file = self._make_classification_csv()
        response = self.client.post(
            "/tuning/start",
            files={"file": ("test_et.csv", csv_file, "text/csv")},
            data={
                "task_type": "classification",
                "target_column": "target",
                "model_type": "ExtraTreesClassifier",
                "metric": "accuracy",
                "n_trials": 2,
                "n_splits": 2,
            },
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["model_type"], "ExtraTreesClassifier")
        self.assertIn("n_estimators", data["best_params"])

    def test_tuning_random_forest_regressor(self):
        """Test RandomForest regressor tuning with R2 maximization."""
        csv_file = self._make_regression_csv()
        response = self.client.post(
            "/tuning/start",
            files={"file": ("test_rf_reg.csv", csv_file, "text/csv")},
            data={
                "task_type": "regression",
                "target_column": "target",
                "model_type": "RandomForest",
                "metric": "r2",
                "n_trials": 2,
                "n_splits": 2,
            },
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["model_type"], "RandomForest")
        self.assertEqual(data["direction"], "maximize")
        self.assertIn("r2", data["tuned_metrics"])

    def test_leakage_free_holdout_split(self):
        """Verify that holdout split ratio is respected and testing samples match expectation."""
        csv_file = self._make_classification_csv(n_rows=100)
        response = self.client.post(
            "/tuning/start",
            files={"file": ("test_leakage.csv", csv_file, "text/csv")},
            data={
                "task_type": "classification",
                "target_column": "target",
                "model_type": "LogisticRegression",
                "metric": "f1_macro",
                "n_trials": 2,
                "n_splits": 2,
                "splitRatio": 0.75,
            },
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        dq = data["data_quality"]
        self.assertEqual(dq["testing_samples"], 25)
        self.assertEqual(dq["training_samples"], 75)
        self.assertEqual(data["baseline_metrics"]["testing_samples"], 25)
        self.assertEqual(data["tuned_metrics"]["testing_samples"], 25)

    def test_pagination_in_tuning_history(self):
        """Verify limit and offset query parameters in GET /tuning."""
        res = self.client.get("/tuning?limit=2&offset=0")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["limit"], 2)
        self.assertEqual(data["offset"], 0)
        self.assertTrue(len(data["items"]) <= 2)


if __name__ == "__main__":
    unittest.main()
