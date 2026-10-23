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

from app.main import app

client = TestClient(app)

class TestClassificationAndRegressionEngine(unittest.TestCase):

    def _post_train(self, df: pd.DataFrame, form_data: dict):
        csv_buffer = io.BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        files = {"file": ("test.csv", csv_buffer, "text/csv")}
        return client.post("/train", data=form_data, files=files)

    def test_1_binary_classification_numeric(self):
        """1. Binary classification with numeric features."""
        df = pd.DataFrame({
            "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                   1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5],
            "f2": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
                   15, 25, 35, 45, 55, 65, 75, 85, 95, 105],
            "label": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
                      0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        })
        payload = {
            "task_type": "classification",
            "target_column": "label",
            "model": "LogisticRegression",
            "scaler": "standard",
            "splitRatio": 0.8,
            "missing": "mean",
            "encoding": "onehot"
        }
        res = self._post_train(df, payload)
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()
        self.assertEqual(data["task_type"], "classification")
        self.assertEqual(data["target_column"], "label")
        self.assertEqual(data["classes"], ["0", "1"])
        self.assertIn("accuracy", data["metrics"])
        self.assertIn("balanced_accuracy", data["metrics"])
        self.assertIn("roc_auc", data["metrics"])
        self.assertEqual(len(data["confusion_matrix"]["matrix"]), 2)
        self.assertEqual(len(data["confusion_matrix"]["matrix"][0]), 2)
        self.assertEqual(data["confusion_matrix"]["labels"], ["0", "1"])

    def test_2_multiclass_classification(self):
        """2. Multiclass classification (3 classes)."""
        df = pd.DataFrame({
            "feat_a": list(range(30)),
            "feat_b": [x * 2.5 for x in range(30)],
            "target": ["cat", "dog", "bird"] * 10
        })
        payload = {
            "task_type": "classification",
            "target_column": "target",
            "model": "RandomForestClassifier",
            "scaler": "minmax",
            "splitRatio": 0.8,
            "missing": "median",
            "encoding": "onehot"
        }
        res = self._post_train(df, payload)
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()
        self.assertEqual(data["task_type"], "classification")
        self.assertEqual(len(data["classes"]), 3)
        self.assertEqual(len(data["confusion_matrix"]["matrix"]), 3)
        self.assertEqual(len(data["confusion_matrix"]["matrix"][0]), 3)
        self.assertIn("f1_macro", data["metrics"])

    def test_3_mixed_numeric_and_categorical_features(self):
        """3. Mixed numeric and categorical features."""
        df = pd.DataFrame({
            "num1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] * 2,
            "cat1": ["small", "medium", "large", "small", "medium"] * 4,
            "target": ["A", "B"] * 10
        })
        payload = {
            "task_type": "classification",
            "target_column": "target",
            "model": "ExtraTreesClassifier",
            "scaler": "standard",
            "splitRatio": 0.8,
            "missing": "median",
            "encoding": "onehot"
        }
        res = self._post_train(df, payload)
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()
        self.assertIn("f1_weighted", data["metrics"])
        self.assertEqual(data["data_quality"]["feature_counts"]["numeric"], 1)
        self.assertEqual(data["data_quality"]["feature_counts"]["categorical"], 1)

    def test_4_missing_feature_values(self):
        """4. Missing feature values handled with median imputation without leakage."""
        df = pd.DataFrame({
            "num_missing": [None, 2.0, None, 4.0, 5.0, 6.0, None, 8.0, 9.0, 10.0] * 2,
            "cat_missing": [None, "X", "Y", None, "X", "Y", "X", "Y", None, "X"] * 2,
            "target": ["Class0", "Class1"] * 10
        })
        payload = {
            "task_type": "classification",
            "target_column": "target",
            "model": "DecisionTreeClassifier",
            "scaler": "standard",
            "splitRatio": 0.8,
            "missing": "median",
            "encoding": "onehot"
        }
        res = self._post_train(df, payload)
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()
        self.assertIsNotNone(data["metrics"]["accuracy"])

    def test_5_unseen_categorical_in_holdout(self):
        """5. Unseen categorical value in holdout set does not crash pipeline."""
        # 18 rows with cat "A" and "B", 2 rows with "NEW_UNSEEN"
        df = pd.DataFrame({
            "val": list(range(20)),
            "cat": ["A"] * 9 + ["B"] * 9 + ["NEW_UNSEEN"] * 2,
            "target": [0, 1] * 10
        })
        payload = {
            "task_type": "classification",
            "target_column": "target",
            "model": "LogisticRegression",
            "scaler": "standard",
            "splitRatio": 0.8,
            "missing": "median",
            "encoding": "onehot"
        }
        res = self._post_train(df, payload)
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()
        self.assertIn("accuracy", data["metrics"])

    def test_6_target_in_first_or_middle_column(self):
        """6. Target column in first or middle column, not just last."""
        df = pd.DataFrame({
            "target_first": [1, 0] * 10,
            "feature_x": [10.5, 20.3] * 10,
            "feature_y": [100, 200] * 10
        })
        payload = {
            "task_type": "classification",
            "target_column": "target_first",
            "model": "RandomForestClassifier",
            "scaler": "standard",
            "splitRatio": 0.8,
            "missing": "median",
            "encoding": "onehot"
        }
        res = self._post_train(df, payload)
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()
        self.assertEqual(data["target_column"], "target_first")
        self.assertNotIn("target_first", data["columns"])
        self.assertIn("feature_x", data["columns"])
        self.assertIn("feature_y", data["columns"])

    def test_7_single_class_target_rejected(self):
        """7. Single-class target is rejected with clear 400 error."""
        df = pd.DataFrame({
            "f1": [1, 2, 3, 4, 5],
            "target": ["OnlyClass"] * 5
        })
        payload = {
            "task_type": "classification",
            "target_column": "target",
            "model": "LogisticRegression",
            "scaler": "standard",
            "splitRatio": 0.8,
            "missing": "mean",
            "encoding": "onehot"
        }
        res = self._post_train(df, payload)
        self.assertEqual(res.status_code, 400)
        self.assertIn("at least 2 distinct classes", res.json()["detail"])

    def test_8_class_with_only_one_sample_rejected(self):
        """8. Class with only one sample is rejected with 400 error."""
        df = pd.DataFrame({
            "f1": list(range(10)),
            "target": ["Common"] * 9 + ["RareSingleSample"]
        })
        payload = {
            "task_type": "classification",
            "target_column": "target",
            "model": "LogisticRegression",
            "scaler": "standard",
            "splitRatio": 0.8,
            "missing": "mean",
            "encoding": "onehot"
        }
        res = self._post_train(df, payload)
        self.assertEqual(res.status_code, 400)
        self.assertIn("has only 1 sample", res.json()["detail"])

    def test_9_unsuitable_high_cardinality_target(self):
        """9. Unsuitable high-cardinality target rejected with 400 error."""
        # 30 distinct classes in target (> 20 limit)
        df = pd.DataFrame({
            "f1": list(range(60)),
            "target_id": [f"ID_{i}" for i in range(30)] * 2
        })
        payload = {
            "task_type": "classification",
            "target_column": "target_id",
            "model": "LogisticRegression",
            "scaler": "standard",
            "splitRatio": 0.8,
            "missing": "mean",
            "encoding": "onehot"
        }
        res = self._post_train(df, payload)
        self.assertEqual(res.status_code, 400)
        self.assertIn("exceeding the initial supported limit", res.json()["detail"])

    def test_10_unsupported_free_text_column_diagnostic(self):
        """10. Unsupported free-text column produces clear diagnostic instead of silent drop."""
        df = pd.DataFrame({
            "f1": list(range(20)),
            "user_review": [
                "This is a very long descriptive product review written by a user"
            ] * 20,
            "target": [0, 1] * 10
        })
        payload = {
            "task_type": "classification",
            "target_column": "target",
            "model": "RandomForestClassifier",
            "scaler": "standard",
            "splitRatio": 0.8,
            "missing": "mean",
            "encoding": "onehot"
        }
        res = self._post_train(df, payload)
        self.assertEqual(res.status_code, 400)
        self.assertIn("free-form text", res.json()["detail"])

    def test_11_invalid_task_target_and_split_parameters(self):
        """11. Invalid task type, target name, and split ratio return 400."""
        df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [0, 1, 0, 1]})

        # Invalid task_type
        res1 = self._post_train(df, {
            "task_type": "reinforcement_learning",
            "target_column": "b",
            "model": "LogisticRegression",
            "scaler": "standard",
            "splitRatio": 0.8,
            "missing": "mean",
            "encoding": "onehot"
        })
        self.assertEqual(res1.status_code, 400)
        self.assertIn("Unsupported task_type", res1.json()["detail"])

        # Missing target_column for classification
        res2 = self._post_train(df, {
            "task_type": "classification",
            "model": "LogisticRegression",
            "scaler": "standard",
            "splitRatio": 0.8,
            "missing": "mean",
            "encoding": "onehot"
        })
        self.assertEqual(res2.status_code, 400)
        self.assertIn("target_column", res2.json()["detail"])

        # Nonexistent target_column
        res3 = self._post_train(df, {
            "task_type": "classification",
            "target_column": "non_existent_column",
            "model": "LogisticRegression",
            "scaler": "standard",
            "splitRatio": 0.8,
            "missing": "mean",
            "encoding": "onehot"
        })
        self.assertEqual(res3.status_code, 400)
        self.assertIn("was not found in dataset columns", res3.json()["detail"])

        # Invalid splitRatio <= 0 or >= 1
        res4 = self._post_train(df, {
            "task_type": "classification",
            "target_column": "b",
            "model": "LogisticRegression",
            "scaler": "standard",
            "splitRatio": 1.5,
            "missing": "mean",
            "encoding": "onehot"
        })
        self.assertEqual(res4.status_code, 400)
        self.assertIn("splitRatio", res4.json()["detail"])

    def test_12_regression_backward_compatibility(self):
        """12. Regression requests that omit task_type retain existing flow and response structure."""
        df = pd.DataFrame({
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "feature2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
            "target": [105.0, 202.0, 301.0, 399.0, 502.0, 601.0, 703.0, 805.0, 899.0, 1004.0]
        })
        payload = {
            "model": "LinearRegression",
            "scaler": "standard",
            "splitRatio": 0.8,
            "missing": "drop",
            "encoding": "label"
        }
        res = self._post_train(df, payload)
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()
        self.assertIn("metrics", data)
        self.assertIn("rmse", data["metrics"])
        self.assertIn("r2", data["metrics"])
        self.assertIn("mae", data["metrics"])
        self.assertIn("visualization", data)
        self.assertIn("prediction_plot", data)
        self.assertEqual(data["model_type"], "LinearRegression")


if __name__ == "__main__":
    unittest.main()
