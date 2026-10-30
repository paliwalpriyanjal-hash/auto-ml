import os
import sys
import io
import unittest
import numpy as np
import pandas as pd

# Headless matplotlib
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from app.main import app
from app.services.ml_intelligence_service import (
    analyze_dataset_intelligence,
    build_dataset_aware_preprocessor,
    generate_training_diagnostics,
)

client = TestClient(app)


class TestMLIntelligenceEngine(unittest.TestCase):
    """
    Test suite for AutoVeyra Phase 1:
    - Pre-training Dataset Intelligence & Leakage Detection
    - Leak-free Preprocessing and Evaluation
    - Expanded Classification and Regression Estimators
    - Training Cross-Validation, Generalization Gaps & Failure Diagnostics
    - Model Registry Integration for Regression Models
    """

    def _post_train(self, df: pd.DataFrame, form_data: dict):
        csv_buffer = io.BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        files = {"file": ("dataset.csv", csv_buffer, "text/csv")}
        return client.post("/train", data=form_data, files=files)

    def _post_analyze(self, df: pd.DataFrame, form_data: dict = None):
        csv_buffer = io.BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        files = {"file": ("dataset.csv", csv_buffer, "text/csv")}
        return client.post("/train/analyze", data=form_data or {}, files=files)

    # ------------------------------------------------------------------
    # 1. Dataset Intelligence Analysis Unit Tests
    # ------------------------------------------------------------------
    def test_01_dataset_intelligence_binary_and_imbalance(self):
        """Intelligent dataset analysis detects binary classification, class imbalance and missing values."""
        df = pd.DataFrame({
            "age": [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75],
            "income": [2000, 2500, None, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500],
            "status": ["active"] * 10 + ["churned"] * 2  # 5:1 imbalance ratio
        })

        analysis = analyze_dataset_intelligence(df, target_column="status")
        self.assertEqual(analysis["dimensions"]["rows"], 12)
        self.assertEqual(analysis["dimensions"]["columns"], 3)
        self.assertEqual(analysis["target_analysis"]["inferred_task"], "binary_classification")
        self.assertEqual(analysis["target_analysis"]["inferred_target"], "status")
        self.assertTrue(analysis["target_analysis"]["is_imbalanced"])
        self.assertEqual(analysis["target_analysis"]["imbalance_severity"], "moderate")
        self.assertGreaterEqual(analysis["target_analysis"]["imbalance_ratio"], 5.0)
        self.assertGreaterEqual(len(analysis["recommendations"]), 1)

    def test_02_dataset_intelligence_regression_and_collinearity(self):
        """Intelligent dataset analysis detects continuous regression and collinear features."""
        x = np.linspace(1, 100, 30)
        df = pd.DataFrame({
            "feature_x": x,
            "feature_collinear": x * 2.0 + 0.001,  # Near-perfect correlation
            "target_num": x * 4.5 + np.random.normal(0, 1, 30),
        })

        analysis = analyze_dataset_intelligence(df, target_column="target_num")
        self.assertEqual(analysis["target_analysis"]["inferred_task"], "regression")
        # Collinear pair detection
        collinear_pairs = analysis["collinear_features"]
        self.assertTrue(any("feature_x" in p["feature_a"] or "feature_collinear" in p["feature_a"] for p in collinear_pairs))

    # ------------------------------------------------------------------
    # 2. Pre-training Dataset Analysis API Endpoints
    # ------------------------------------------------------------------
    def test_03_analyze_endpoint(self):
        """POST /train/analyze and /api/analyze-dataset respond with dataset profiling and recommendations."""
        df = pd.DataFrame({
            "feat1": [1.0, 2.0, 3.0, 4.0, 5.0] * 4,
            "feat2": ["cat", "dog"] * 10,
            "target": [0, 1] * 10,
        })
        res = self._post_analyze(df, {"target_column": "target"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["dimensions"]["rows"], 20)
        self.assertIn("recommendations", data)
        self.assertIn("target_analysis", data)

    # ------------------------------------------------------------------
    # 3. Expanded Classification Estimators & Diagnostics
    # ------------------------------------------------------------------
    def test_04_hist_gradient_boosting_classifier(self):
        """HistGradientBoostingClassifier trains with training CV, holdout metrics and diagnostics."""
        df = pd.DataFrame({
            "a": [1.0, 2.5, 3.1, 4.8, 5.2, 6.7, 7.1, 8.4, 9.0, 10.5] * 3,
            "b": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] * 3,
            "y": ["alpha", "beta"] * 15,
        })
        payload = {
            "task_type": "classification",
            "target_column": "y",
            "model": "HistGradientBoostingClassifier",
            "scaler": "standard",
            "splitRatio": 0.8,
            "missing": "median",
            "encoding": "onehot",
        }
        res = self._post_train(df, payload)
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()
        self.assertEqual(data["model_type"], "HistGradientBoostingClassifier")
        self.assertIn("cv_metrics", data)
        self.assertIn("accuracy_mean", data["cv_metrics"])
        self.assertIn("training_metrics", data)
        self.assertIn("overfitting_gap", data)
        self.assertIn("diagnostics", data)
        self.assertIn("registered_model", data)

    def test_05_gradient_boosting_and_knn_classifiers(self):
        """GradientBoostingClassifier and KNeighborsClassifier are supported."""
        df = pd.DataFrame({
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] * 2,
            "x2": [10, 12, 14, 16, 18, 20, 22, 24, 26, 28] * 2,
            "label": ["no", "yes"] * 10,
        })
        for model_name in ["GradientBoostingClassifier", "KNeighborsClassifier"]:
            payload = {
                "task_type": "classification",
                "target_column": "label",
                "model": model_name,
                "scaler": "standard",
                "splitRatio": 0.8,
                "missing": "mean",
                "encoding": "onehot",
            }
            res = self._post_train(df, payload)
            self.assertEqual(res.status_code, 200, f"{model_name} failed: {res.text}")
            self.assertIn("metrics", res.json())

    # ------------------------------------------------------------------
    # 4. Expanded Regression Estimators & Model Registration
    # ------------------------------------------------------------------
    def test_06_expanded_regression_models_and_registration(self):
        """Expanded regression models (Ridge, Lasso, HistGradientBoosting) register models and return CV metrics."""
        x = np.linspace(1, 30, 25)
        df = pd.DataFrame({
            "feat1": x,
            "feat2": x * 1.5,
            "price": x * 10 + 2.5,
        })
        for reg_model in ["Ridge", "HistGradientBoosting", "ExtraTrees"]:
            payload = {
                "task_type": "regression",
                "target_column": "price",
                "model": reg_model,
                "scaler": "standard",
                "splitRatio": 0.8,
                "missing": "median",
                "encoding": "onehot",
            }
            res = self._post_train(df, payload)
            self.assertEqual(res.status_code, 200, f"{reg_model} failed: {res.text}")
            data = res.json()
            self.assertEqual(data["model_type"], reg_model)
            self.assertIn("metrics", data)
            self.assertIn("rmse", data["metrics"])
            self.assertIn("r2", data["metrics"])
            self.assertIn("cv_metrics", data)
            self.assertIn("r2_mean", data["cv_metrics"])
            self.assertIn("registered_model", data)
            self.assertIsNotNone(data["registered_model"]["id"])

    # ------------------------------------------------------------------
    # 5. Model Comparison with Extended Model Suite
    # ------------------------------------------------------------------
    def test_07_compare_extended_suite(self):
        """POST /compare with suite='extended' compares expanded candidate models and includes ranking explanation."""
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] * 3,
            "b": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] * 3,
            "target": ["classA", "classB"] * 15,
        })
        csv_buffer = io.BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        payload = {
            "task_type": "classification",
            "target_column": "target",
            "scaler": "standard",
            "splitRatio": 0.8,
            "missing": "median",
            "encoding": "onehot",
            "suite": "extended",
        }
        files = {"file": ("test.csv", csv_buffer, "text/csv")}
        res = client.post("/compare", data=payload, files=files)
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()
        model_names = [m["model"] for m in data["model_results"]]
        self.assertGreater(len(model_names), 4)
        self.assertIn("HistGradientBoostingClassifier", model_names)
        self.assertIn("ranking_explanation", data)
        self.assertIn("recommendation_reason", data["best_model"])


if __name__ == "__main__":
    unittest.main()
