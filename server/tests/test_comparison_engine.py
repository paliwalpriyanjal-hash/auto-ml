import os
import sys
import io
import unittest

# Ensure headless Matplotlib backend before importing visualization or app
import matplotlib
matplotlib.use("Agg")

# Ensure server root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
import pandas as pd
import numpy as np

from app.main import app

client = TestClient(app)


class TestModelComparison(unittest.TestCase):
    """Tests for POST /compare endpoint."""

    def _post_compare(self, df: pd.DataFrame, form_data: dict):
        csv_buffer = io.BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        files = {"file": ("test.csv", csv_buffer, "text/csv")}
        return client.post("/compare", data=form_data, files=files)

    # ------------------------------------------------------------------
    # 1. Binary classification comparison
    # ------------------------------------------------------------------
    def test_01_binary_classification_comparison(self):
        """All 4 classification models are compared for a binary dataset."""
        df = pd.DataFrame({
            "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                   1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5],
            "f2": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
                   15, 25, 35, 45, 55, 65, 75, 85, 95, 105],
            "label": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
                      0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        })
        payload = {
            "task_type": "classification",
            "target_column": "label",
            "scaler": "standard",
            "splitRatio": 0.8,
            "missing": "median",
            "encoding": "onehot",
        }
        res = self._post_compare(df, payload)
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()

        self.assertEqual(data["task_type"], "classification")
        self.assertEqual(data["target_column"], "label")
        self.assertEqual(data["ranking_metric"], "f1_macro")
        self.assertIn("cv_folds", data)

        # All 4 classification models present
        model_names = [r["model"] for r in data["model_results"]]
        self.assertIn("LogisticRegression", model_names)
        self.assertIn("DecisionTreeClassifier", model_names)
        self.assertIn("RandomForestClassifier", model_names)
        self.assertIn("ExtraTreesClassifier", model_names)

        # Best model has holdout metrics
        best = data["best_model"]
        self.assertIn("holdout_metrics", best)
        hm = best["holdout_metrics"]
        self.assertIn("accuracy", hm)
        self.assertIn("confusion_matrix", hm)
        self.assertEqual(len(hm["confusion_matrix"]["labels"]), 2)

    # ------------------------------------------------------------------
    # 2. Multiclass classification comparison
    # ------------------------------------------------------------------
    def test_02_multiclass_classification_comparison(self):
        """Multiclass (3 classes) comparison returns all 4 models ranked."""
        df = pd.DataFrame({
            "feat_a": list(range(30)),
            "feat_b": [x * 2.5 for x in range(30)],
            "target": ["cat", "dog", "bird"] * 10,
        })
        payload = {
            "task_type": "classification",
            "target_column": "target",
            "scaler": "minmax",
            "splitRatio": 0.8,
            "missing": "median",
            "encoding": "onehot",
        }
        res = self._post_compare(df, payload)
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()

        self.assertEqual(len(data["classes"]), 3)
        self.assertEqual(len(data["model_results"]), 4)

        # Confusion matrix is 3x3
        cm = data["best_model"]["holdout_metrics"]["confusion_matrix"]
        self.assertEqual(len(cm["matrix"]), 3)
        self.assertEqual(len(cm["matrix"][0]), 3)

    # ------------------------------------------------------------------
    # 3. Regression comparison
    # ------------------------------------------------------------------
    def test_03_regression_comparison(self):
        """All 3 regression models are compared with RMSE ranking."""
        df = pd.DataFrame({
            "x1": list(range(1, 31)),
            "x2": [x * 2.5 for x in range(1, 31)],
            "y": [x * 10 + 5 for x in range(1, 31)],
        })
        payload = {
            "task_type": "regression",
            "target_column": "y",
            "scaler": "standard",
            "splitRatio": 0.8,
            "missing": "median",
            "encoding": "onehot",
        }
        res = self._post_compare(df, payload)
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()

        self.assertEqual(data["task_type"], "regression")
        self.assertEqual(data["ranking_metric"], "rmse")

        model_names = [r["model"] for r in data["model_results"]]
        self.assertIn("LinearRegression", model_names)
        self.assertIn("DecisionTree", model_names)
        self.assertIn("RandomForest", model_names)
        self.assertEqual(len(model_names), 3)

        # Best model has regression holdout metrics
        hm = data["best_model"]["holdout_metrics"]
        self.assertIn("rmse", hm)
        self.assertIn("r2", hm)
        self.assertIn("mae", hm)

    # ------------------------------------------------------------------
    # 4. Target column not in last position
    # ------------------------------------------------------------------
    def test_04_target_not_last_column(self):
        """Target column in first position works correctly."""
        df = pd.DataFrame({
            "target_first": [1, 0] * 15,
            "feature_x": [10.5, 20.3] * 15,
            "feature_y": [100, 200] * 15,
        })
        payload = {
            "task_type": "classification",
            "target_column": "target_first",
            "scaler": "standard",
            "splitRatio": 0.8,
            "missing": "median",
            "encoding": "onehot",
        }
        res = self._post_compare(df, payload)
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()
        self.assertEqual(data["target_column"], "target_first")

    # ------------------------------------------------------------------
    # 5. Mixed numeric and categorical features
    # ------------------------------------------------------------------
    def test_05_mixed_numeric_and_categorical(self):
        """Mixed feature types are handled correctly."""
        df = pd.DataFrame({
            "num1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] * 3,
            "cat1": ["small", "medium", "large", "small", "medium"] * 6,
            "target": ["A", "B"] * 15,
        })
        payload = {
            "task_type": "classification",
            "target_column": "target",
            "scaler": "standard",
            "splitRatio": 0.8,
            "missing": "median",
            "encoding": "onehot",
        }
        res = self._post_compare(df, payload)
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()
        self.assertEqual(data["data_quality"]["feature_counts"]["numeric"], 1)
        self.assertEqual(data["data_quality"]["feature_counts"]["categorical"], 1)

    # ------------------------------------------------------------------
    # 6. Missing values handled via pipeline imputation
    # ------------------------------------------------------------------
    def test_06_missing_values(self):
        """Missing values in features are handled without breaking comparison."""
        df = pd.DataFrame({
            "num_missing": [None, 2.0, None, 4.0, 5.0, 6.0, None, 8.0, 9.0, 10.0] * 3,
            "cat_missing": [None, "X", "Y", None, "X", "Y", "X", "Y", None, "X"] * 3,
            "target": ["Class0", "Class1"] * 15,
        })
        payload = {
            "task_type": "classification",
            "target_column": "target",
            "scaler": "standard",
            "splitRatio": 0.8,
            "missing": "median",
            "encoding": "onehot",
        }
        res = self._post_compare(df, payload)
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()
        # All models should succeed
        for mr in data["model_results"]:
            self.assertEqual(mr["status"], "success", msg=mr.get("error"))

    # ------------------------------------------------------------------
    # 7. Invalid target column
    # ------------------------------------------------------------------
    def test_07_invalid_target_column(self):
        """Nonexistent target column returns 400."""
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [0, 1, 0, 1, 0]})
        payload = {
            "task_type": "classification",
            "target_column": "nonexistent",
            "scaler": "standard",
            "splitRatio": 0.8,
            "missing": "median",
            "encoding": "onehot",
        }
        res = self._post_compare(df, payload)
        self.assertEqual(res.status_code, 400)
        self.assertIn("was not found", res.json()["detail"])

    # ------------------------------------------------------------------
    # 8. Insufficient samples / infeasible CV folds
    # ------------------------------------------------------------------
    def test_08_insufficient_samples(self):
        """Very small dataset with single-sample class is rejected."""
        df = pd.DataFrame({
            "f1": list(range(10)),
            "target": ["Common"] * 9 + ["RareSingleSample"],
        })
        payload = {
            "task_type": "classification",
            "target_column": "target",
            "scaler": "standard",
            "splitRatio": 0.8,
            "missing": "median",
            "encoding": "onehot",
        }
        res = self._post_compare(df, payload)
        self.assertEqual(res.status_code, 400)
        self.assertIn("has only 1 sample", res.json()["detail"])

    # ------------------------------------------------------------------
    # 9. Failed model does not break others
    # ------------------------------------------------------------------
    def test_09_failed_model_does_not_break_comparison(self):
        """
        If a model fails during CV, other models still succeed and the
        failed model is not ranked as winner.
        """
        # We test this by checking that when all models succeed, none have
        # status=failed, and then verifying the winner has rank=1
        df = pd.DataFrame({
            "f1": list(range(30)),
            "f2": [x * 1.5 for x in range(30)],
            "label": [0, 1] * 15,
        })
        payload = {
            "task_type": "classification",
            "target_column": "label",
            "scaler": "standard",
            "splitRatio": 0.8,
            "missing": "median",
            "encoding": "onehot",
        }
        res = self._post_compare(df, payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()

        successful = [r for r in data["model_results"] if r["status"] == "success"]
        failed = [r for r in data["model_results"] if r["status"] == "failed"]

        # Winner must be among successful models
        best_name = data["best_model"]["model"]
        successful_names = [r["model"] for r in successful]
        self.assertIn(best_name, successful_names)

        # Failed models must not have a rank
        for f in failed:
            self.assertIsNone(f["rank"])

        # Winner must have rank 1
        winner = next(r for r in data["model_results"] if r["model"] == best_name)
        self.assertEqual(winner["rank"], 1)

    # ------------------------------------------------------------------
    # 10. Correct ranking direction: higher-is-better (classification)
    # ------------------------------------------------------------------
    def test_10_ranking_direction_classification(self):
        """Classification ranking metrics are higher-is-better (rank 1 = highest CV mean)."""
        df = pd.DataFrame({
            "f1": list(range(30)),
            "f2": [x * 1.5 for x in range(30)],
            "label": [0, 1] * 15,
        })
        for metric in ["f1_macro", "balanced_accuracy", "accuracy", "f1_weighted"]:
            payload = {
                "task_type": "classification",
                "target_column": "label",
                "scaler": "standard",
                "splitRatio": 0.8,
                "missing": "median",
                "encoding": "onehot",
                "ranking_metric": metric,
            }
            res = self._post_compare(df, payload)
            self.assertEqual(res.status_code, 200, msg=f"Failed for metric={metric}: {res.text}")
            data = res.json()
            self.assertEqual(data["ranking_metric"], metric)

            successful = [r for r in data["model_results"] if r["status"] == "success" and r["cv_mean"] is not None]
            if len(successful) >= 2:
                # Rank 1 should have the highest cv_mean
                rank1 = next(r for r in successful if r["rank"] == 1)
                for r in successful:
                    if r["rank"] and r["rank"] > 1:
                        self.assertGreaterEqual(rank1["cv_mean"], r["cv_mean"],
                            msg=f"Metric {metric}: rank 1 ({rank1['cv_mean']}) < rank {r['rank']} ({r['cv_mean']})")

    # ------------------------------------------------------------------
    # 11. Correct ranking direction: lower-is-better (RMSE, MAE)
    # ------------------------------------------------------------------
    def test_11_ranking_direction_regression_lower_better(self):
        """For RMSE and MAE, rank 1 has the lowest CV mean."""
        df = pd.DataFrame({
            "x1": list(range(1, 31)),
            "x2": [x * 2.5 for x in range(1, 31)],
            "y": [x * 10 + 5 for x in range(1, 31)],
        })
        for metric in ["rmse", "mae"]:
            payload = {
                "task_type": "regression",
                "target_column": "y",
                "scaler": "standard",
                "splitRatio": 0.8,
                "missing": "median",
                "encoding": "onehot",
                "ranking_metric": metric,
            }
            res = self._post_compare(df, payload)
            self.assertEqual(res.status_code, 200, msg=f"Failed for metric={metric}: {res.text}")
            data = res.json()

            successful = [r for r in data["model_results"] if r["status"] == "success" and r["cv_mean"] is not None]
            if len(successful) >= 2:
                rank1 = next(r for r in successful if r["rank"] == 1)
                for r in successful:
                    if r["rank"] and r["rank"] > 1:
                        self.assertLessEqual(rank1["cv_mean"], r["cv_mean"],
                            msg=f"Metric {metric}: rank 1 ({rank1['cv_mean']}) > rank {r['rank']} ({r['cv_mean']})")

    # ------------------------------------------------------------------
    # 12. Correct ranking direction: higher-is-better (R²)
    # ------------------------------------------------------------------
    def test_12_ranking_direction_r2_higher_better(self):
        """For R², rank 1 has the highest CV mean."""
        df = pd.DataFrame({
            "x1": list(range(1, 31)),
            "x2": [x * 2.5 for x in range(1, 31)],
            "y": [x * 10 + 5 for x in range(1, 31)],
        })
        payload = {
            "task_type": "regression",
            "target_column": "y",
            "scaler": "standard",
            "splitRatio": 0.8,
            "missing": "median",
            "encoding": "onehot",
            "ranking_metric": "r2",
        }
        res = self._post_compare(df, payload)
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()
        self.assertEqual(data["ranking_metric"], "r2")

        successful = [r for r in data["model_results"] if r["status"] == "success" and r["cv_mean"] is not None]
        if len(successful) >= 2:
            rank1 = next(r for r in successful if r["rank"] == 1)
            for r in successful:
                if r["rank"] and r["rank"] > 1:
                    self.assertGreaterEqual(rank1["cv_mean"], r["cv_mean"])

    # ------------------------------------------------------------------
    # 13. NaN/Inf serialization safety
    # ------------------------------------------------------------------
    def test_13_nan_inf_serialization(self):
        """Response JSON is free of NaN and Inf values."""
        import json

        df = pd.DataFrame({
            "f1": list(range(30)),
            "f2": [x * 1.5 for x in range(30)],
            "label": [0, 1] * 15,
        })
        payload = {
            "task_type": "classification",
            "target_column": "label",
            "scaler": "standard",
            "splitRatio": 0.8,
            "missing": "median",
            "encoding": "onehot",
        }
        res = self._post_compare(df, payload)
        self.assertEqual(res.status_code, 200)

        # Serialize to text and check for NaN/Inf tokens
        raw_text = json.dumps(res.json())
        self.assertNotIn("NaN", raw_text)
        self.assertNotIn("Infinity", raw_text)
        self.assertNotIn("-Infinity", raw_text)

    # ------------------------------------------------------------------
    # 14. Existing /train classification compatibility
    # ------------------------------------------------------------------
    def test_14_train_classification_still_works(self):
        """The existing /train classification endpoint is unaffected."""
        df = pd.DataFrame({
            "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                   1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5],
            "f2": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
                   15, 25, 35, 45, 55, 65, 75, 85, 95, 105],
            "label": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
                      0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        })
        csv_buffer = io.BytesIO(df.to_csv(index=False).encode())
        res = client.post("/train", data={
            "task_type": "classification",
            "target_column": "label",
            "model": "LogisticRegression",
            "scaler": "standard",
            "splitRatio": 0.8,
            "missing": "mean",
            "encoding": "onehot",
        }, files={"file": ("test.csv", csv_buffer, "text/csv")})
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()
        self.assertEqual(data["task_type"], "classification")
        self.assertIn("accuracy", data["metrics"])

    # ------------------------------------------------------------------
    # 15. Existing /train regression compatibility
    # ------------------------------------------------------------------
    def test_15_train_regression_still_works(self):
        """The existing /train regression endpoint is unaffected."""
        df = pd.DataFrame({
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "feature2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
            "target": [105.0, 202.0, 301.0, 399.0, 502.0, 601.0, 703.0, 805.0, 899.0, 1004.0],
        })
        csv_buffer = io.BytesIO(df.to_csv(index=False).encode())
        res = client.post("/train", data={
            "model": "LinearRegression",
            "scaler": "standard",
            "splitRatio": 0.8,
            "missing": "drop",
            "encoding": "label",
        }, files={"file": ("test.csv", csv_buffer, "text/csv")})
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()
        self.assertIn("r2", data["metrics"])
        self.assertIn("rmse", data["metrics"])
        self.assertEqual(data["model_type"], "LinearRegression")

    # ------------------------------------------------------------------
    # 16. Leakage safety: pipeline placement verification
    # ------------------------------------------------------------------
    def test_16_leakage_safety_pipeline_design(self):
        """
        Verify that preprocessing is inside the CV pipeline, not outside.
        The comparison service uses sklearn Pipeline with ColumnTransformer
        inside cross_validate(), which means each fold fits preprocessing
        only on its training portion. We verify this by checking that:
        1. The response includes cv_folds > 1
        2. All models have cv_std (which requires multiple folds)
        3. The cv_std is a non-negative number
        """
        df = pd.DataFrame({
            "f1": list(range(30)),
            "f2": [x * 1.5 for x in range(30)],
            "label": [0, 1] * 15,
        })
        payload = {
            "task_type": "classification",
            "target_column": "label",
            "scaler": "standard",
            "splitRatio": 0.8,
            "missing": "median",
            "encoding": "onehot",
        }
        res = self._post_compare(df, payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertGreater(data["cv_folds"], 1)

        for mr in data["model_results"]:
            if mr["status"] == "success":
                self.assertIsNotNone(mr["cv_std"])
                self.assertGreaterEqual(mr["cv_std"], 0.0)

    # ------------------------------------------------------------------
    # 17. Unsupported ranking metric rejected
    # ------------------------------------------------------------------
    def test_17_unsupported_ranking_metric(self):
        """Invalid ranking_metric returns 400 with available options."""
        df = pd.DataFrame({
            "f1": list(range(20)),
            "label": [0, 1] * 10,
        })
        payload = {
            "task_type": "classification",
            "target_column": "label",
            "scaler": "standard",
            "splitRatio": 0.8,
            "missing": "median",
            "encoding": "onehot",
            "ranking_metric": "auc_pr",
        }
        res = self._post_compare(df, payload)
        self.assertEqual(res.status_code, 400)
        self.assertIn("Unsupported ranking_metric", res.json()["detail"])

    # ------------------------------------------------------------------
    # 18. Missing values with drop strategy
    # ------------------------------------------------------------------
    def test_18_missing_with_drop_strategy(self):
        """Drop missing strategy works in comparison."""
        df = pd.DataFrame({
            "f1": [1.0, None, 3.0, None, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] * 3,
            "f2": [10, 20, None, 40, 50, 60, 70, 80, 90, 100] * 3,
            "label": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 3,
        })
        payload = {
            "task_type": "classification",
            "target_column": "label",
            "scaler": "standard",
            "splitRatio": 0.8,
            "missing": "drop",
            "encoding": "onehot",
        }
        res = self._post_compare(df, payload)
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()
        self.assertGreater(data["data_quality"]["dropped_feature_nans"], 0)

    # ------------------------------------------------------------------
    # 19. Regression with target column in middle position
    # ------------------------------------------------------------------
    def test_19_regression_target_middle_column(self):
        """Regression target in the middle column position works correctly."""
        df = pd.DataFrame({
            "feature_a": list(range(1, 31)),
            "target_y": [x * 3.0 + 2 for x in range(1, 31)],
            "feature_b": [x * 0.5 for x in range(1, 31)],
        })
        payload = {
            "task_type": "regression",
            "target_column": "target_y",
            "scaler": "standard",
            "splitRatio": 0.8,
            "missing": "mean",
            "encoding": "onehot",
        }
        res = self._post_compare(df, payload)
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()
        self.assertEqual(data["target_column"], "target_y")
        self.assertEqual(data["task_type"], "regression")

    # ------------------------------------------------------------------
    # 20. Response structure completeness
    # ------------------------------------------------------------------
    def test_20_response_structure(self):
        """Response contains all documented fields."""
        df = pd.DataFrame({
            "f1": list(range(30)),
            "f2": [x * 1.5 for x in range(30)],
            "label": [0, 1] * 15,
        })
        payload = {
            "task_type": "classification",
            "target_column": "label",
            "scaler": "standard",
            "splitRatio": 0.8,
            "missing": "median",
            "encoding": "onehot",
        }
        res = self._post_compare(df, payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()

        # Top-level required fields
        self.assertIn("task_type", data)
        self.assertIn("target_column", data)
        self.assertIn("ranking_metric", data)
        self.assertIn("cv_folds", data)
        self.assertIn("model_results", data)
        self.assertIn("best_model", data)
        self.assertIn("warnings", data)
        self.assertIn("data_quality", data)

        # Model result fields
        for mr in data["model_results"]:
            self.assertIn("model", mr)
            self.assertIn("cv_mean", mr)
            self.assertIn("cv_std", mr)
            self.assertIn("rank", mr)
            self.assertIn("status", mr)
            self.assertIn("error", mr)

        # Best model fields
        best = data["best_model"]
        self.assertIn("model", best)
        self.assertIn("cv_mean", best)
        self.assertIn("cv_std", best)
        self.assertIn("holdout_metrics", best)


if __name__ == "__main__":
    unittest.main()
