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

    # ------------------------------------------------------------------
    # Phase 3 Advanced Feature Engineering Tests
    # ------------------------------------------------------------------

    def test_13_cyclical_and_temporal_features(self):
        """Verify cyclical sine/cosine encoding and calendar component extraction."""
        csv_file = self._make_sample_csv()
        res = self.client.post(
            "/features/preview",
            files={"file": ("cyclical_test.csv", csv_file, "text/csv")},
            data={
                "datetime_cols": '["record_date"]',
                "cyclical_datetime": True,
            },
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        created = data["created_columns"]
        self.assertIn("record_date_month_sin", created)
        self.assertIn("record_date_month_cos", created)
        self.assertIn("record_date_dayofweek_sin", created)
        self.assertIn("record_date_is_weekend", created)
        self.assertIn("record_date_quarter", created)

    def test_14_timeseries_past_only_lags_and_rolling(self):
        """Verify past-only lag and rolling statistics generation with zero future leakage."""
        from app.services.feature_service import TabularFeatureEngineer
        df = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=10, freq="D"),
            "demand": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        })
        engineer = TabularFeatureEngineer(
            time_series_col="timestamp",
            lag_features={"demand": [1, 2]},
            rolling_features={"demand": [3]},
        )
        transformed = engineer.fit_transform(df)
        self.assertIn("demand_lag_1", transformed.columns)
        self.assertIn("demand_lag_2", transformed.columns)
        self.assertIn("demand_rolling_mean_3", transformed.columns)

        # Value at index 2 (t=3, value=30): lag_1 should be 20.0, lag_2 should be 10.0
        self.assertEqual(float(transformed.loc[2, "demand_lag_1"]), 20.0)
        self.assertEqual(float(transformed.loc[2, "demand_lag_2"]), 10.0)

    def test_15_yeo_johnson_and_safe_interactions(self):
        """Verify Yeo-Johnson power transform and zero-division-protected ratio interactions."""
        csv_file = self._make_sample_csv()
        res = self.client.post(
            "/features/preview",
            files={"file": ("yeo_test.csv", csv_file, "text/csv")},
            data={
                "yeo_johnson_cols": '["feature_num1"]',
                "sqrt_cols": '["feature_nonneg"]',
                "interaction_pairs": '[["feature_num1", "feature_nonneg", "multiply"], ["feature_num1", "feature_nonneg", "divide"]]',
            },
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        created = data["created_columns"]
        self.assertIn("feature_nonneg_sqrt", created)
        self.assertIn("feature_num1_x_feature_nonneg", created)
        self.assertIn("feature_num1_div_feature_nonneg", created)

        # Ensure no infinite or NaN values in sample preview
        for row in data["sample_preview"]:
            ratio_val = row.get("feature_num1_div_feature_nonneg")
            self.assertIsNotNone(ratio_val)
            self.assertTrue(np.isfinite(ratio_val))

    def test_16_text_feature_extraction(self):
        """Verify lightweight text feature extraction."""
        from app.services.feature_service import TabularFeatureEngineer
        df = pd.DataFrame({
            "notes": ["Customer reported latency issue on checkout", "Short feedback", "", None],
            "val": [1, 2, 3, 4],
        })
        engineer = TabularFeatureEngineer(text_cols=["notes"], drop_raw_text=True)
        transformed = engineer.fit_transform(df)
        self.assertIn("notes_char_count", transformed.columns)
        self.assertIn("notes_word_count", transformed.columns)
        self.assertIn("notes_is_empty", transformed.columns)
        self.assertEqual(int(transformed.loc[2, "notes_is_empty"]), 1)
        self.assertEqual(int(transformed.loc[0, "notes_word_count"]), 6)

    def test_17_mutual_info_and_rfe_selection(self):
        """Verify Mutual Information and Recursive Feature Elimination selectors."""
        csv_file = self._make_sample_csv(n_rows=60)
        # Test mutual_info
        mi_res = self.client.post(
            "/features/evaluate",
            files={"file": ("mi_test.csv", csv_file, "text/csv")},
            data={
                "task_type": "classification",
                "target_column": "target",
                "model_type": "DecisionTreeClassifier",
                "selection_method": "mutual_info",
                "k_best_count": 3,
                "n_splits": 2,
            },
        )
        self.assertEqual(mi_res.status_code, 200, mi_res.text)
        self.assertIn("registered_model_id", mi_res.json())

        # Test rfe
        csv_file.seek(0)
        rfe_res = self.client.post(
            "/features/evaluate",
            files={"file": ("rfe_test.csv", csv_file, "text/csv")},
            data={
                "task_type": "classification",
                "target_column": "target",
                "model_type": "DecisionTreeClassifier",
                "selection_method": "rfe",
                "k_best_count": 3,
                "n_splits": 2,
            },
        )
        self.assertEqual(rfe_res.status_code, 200, rfe_res.text)
        self.assertIn("registered_model_id", rfe_res.json())

    def test_18_recommendations_endpoint(self):
        """Test POST /features/recommendations adaptive strategy selector."""
        csv_file = self._make_sample_csv()
        res = self.client.post(
            "/features/recommendations",
            files={"file": ("recs_test.csv", csv_file, "text/csv")},
            data={"target_column": "target"},
        )
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()
        self.assertIn("presets", data)
        self.assertIn("conservative", data["presets"])
        self.assertIn("balanced", data["presets"])
        self.assertIn("advanced", data["presets"])
        self.assertIn("active_profiles", data)
        self.assertIn("recommendation_explanations", data)

    def test_19_compare_strategies_endpoint(self):
        """Test POST /features/compare-strategies benchmarking Baseline vs Conservative vs Balanced vs Advanced."""
        csv_file = self._make_sample_csv(n_rows=70)
        res = self.client.post(
            "/features/compare-strategies",
            files={"file": ("compare_test.csv", csv_file, "text/csv")},
            data={
                "task_type": "classification",
                "target_column": "target",
                "model_type": "RandomForestClassifier",
                "ranking_metric": "f1_macro",
                "n_splits": 2,
            },
        )
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()
        self.assertIn("winning_strategy", data)
        self.assertIn("comparison_matrix", data)
        self.assertIn("statistical_verdict", data)
        self.assertIn("registered_model_id", data)
        self.assertEqual(len(data["comparison_matrix"]), 4)  # baseline, conservative, balanced, advanced

    def test_20_feature_lineage_endpoint(self):
        """Test GET /features/lineage/{run_id} retrieval."""
        csv_file = self._make_sample_csv(n_rows=60)
        eval_res = self.client.post(
            "/features/evaluate",
            files={"file": ("lineage_test.csv", csv_file, "text/csv")},
            data={
                "task_type": "classification",
                "target_column": "target",
                "model_type": "DecisionTreeClassifier",
                "datetime_cols": '["record_date"]',
                "cyclical_datetime": True,
                "n_splits": 2,
            },
        )
        self.assertEqual(eval_res.status_code, 200)
        run_id = eval_res.json()["run_id"]

        lin_res = self.client.get(f"/features/lineage/{run_id}")
        self.assertEqual(lin_res.status_code, 200)
        data = lin_res.json()
        self.assertEqual(data["run_id"], run_id)
        self.assertIn("feature_lineage", data)
        self.assertTrue(len(data["feature_lineage"]) > 0)


if __name__ == "__main__":
    unittest.main()
