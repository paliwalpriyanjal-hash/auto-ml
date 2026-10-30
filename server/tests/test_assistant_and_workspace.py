# server/tests/test_assistant_and_workspace.py
"""
AutoVeyra Phase 6: Guided Workspace & Intelligent User Assistance Test Suite
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from app.main import app
from app.services.assistant_service import (
    ML_GLOSSARY,
    ALGORITHM_PROFILES,
    recommend_next_action,
    explain_metric_result,
    explain_generalization_gap,
    validate_preflight,
    create_or_update_workspace_project,
    get_workspace_project,
    list_workspace_projects,
    delete_workspace_project,
)


class TestAssistantAndWorkspace(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_01_glossary_retrieval(self):
        # 1. Full glossary
        res = self.client.get("/assistant/glossary")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertGreaterEqual(data["total"], 30)
        self.assertTrue(any(t["term"] == "Accuracy" for t in data["terms"]))
        self.assertTrue(any(t["term"] == "F1 Score" for t in data["terms"]))

        # 2. Category filter
        res_cat = self.client.get("/assistant/glossary?category=Evaluation%20Metrics")
        self.assertEqual(res_cat.status_code, 200)
        data_cat = res_cat.json()
        self.assertTrue(all(t["category"] == "Evaluation Metrics" for t in data_cat["terms"]))

        # 3. Query search
        res_q = self.client.get("/assistant/glossary?q=overfitting")
        self.assertEqual(res_q.status_code, 200)
        data_q = res_q.json()
        self.assertGreaterEqual(data_q["total"], 1)
        self.assertEqual(data_q["terms"][0]["term"], "Overfitting")

        # 4. Single term lookup
        res_single = self.client.get("/assistant/glossary/shap")
        self.assertEqual(res_single.status_code, 200)
        self.assertIn("SHAP", res_single.json()["term"])

        # 5. Non-existent term
        res_404 = self.client.get("/assistant/glossary/non_existent_ml_jargon_xyz")
        self.assertEqual(res_404.status_code, 404)

    def test_02_algorithm_profiles(self):
        res = self.client.get("/assistant/algorithms")
        self.assertEqual(res.status_code, 200)
        algos = res.json()["algorithms"]
        self.assertIn("randomforestclassifier", algos)
        self.assertIn("logisticregression", algos)
        self.assertIn("gradientboostingclassifier", algos)
        rf = algos["randomforestclassifier"]
        self.assertIn("simple_analogy", rf)
        self.assertIn("strengths", rf)
        self.assertIn("limitations", rf)

    def test_03_next_action_recommendation(self):
        # Case A: Nothing started -> upload
        rec_empty = recommend_next_action({})
        self.assertEqual(rec_empty["step"], 1)

        # Case B: Dataset uploaded, no target -> define objective
        rec_target = recommend_next_action({"dataset_name": "data.csv"})
        self.assertEqual(rec_target["step"], 4)

        # Case C: Target selected, no EDA -> run EDA / inspect quality
        rec_eda = recommend_next_action({
            "dataset_name": "data.csv",
            "target_column": "label",
            "completed_steps": [1, 2, 4],
        })
        self.assertEqual(rec_eda["step"], 3)

        # Case D: Data issues detected, clean not done -> clean
        rec_clean = recommend_next_action({
            "dataset_name": "data.csv",
            "target_column": "label",
            "completed_steps": [1, 2, 3, 4],
            "has_data_issues": True,
        })
        self.assertEqual(rec_clean["step"], 5)

        # Case E: Data clean, no models -> train
        rec_train = recommend_next_action({
            "dataset_name": "data.csv",
            "target_column": "label",
            "completed_steps": [1, 2, 3, 4, 5, 6],
        })
        self.assertEqual(rec_train["step"], 7)

        # Case F: Models trained, no compare -> compare
        rec_compare = recommend_next_action({
            "dataset_name": "data.csv",
            "target_column": "label",
            "completed_steps": [1, 2, 3, 4, 5, 6, 7],
        })
        self.assertEqual(rec_compare["step"], 8)

        # Case G: Compared, no explain -> explain
        rec_explain = recommend_next_action({
            "dataset_name": "data.csv",
            "target_column": "label",
            "completed_steps": [1, 2, 3, 4, 5, 6, 7, 8],
        })
        self.assertEqual(rec_explain["step"], 9)

        # Case H: Explained, no report -> report
        rec_report = recommend_next_action({
            "dataset_name": "data.csv",
            "target_column": "label",
            "completed_steps": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        })
        self.assertEqual(rec_report["step"], 10)

    def test_04_context_explanations(self):
        # 1. Metric: Accuracy
        acc_exp = explain_metric_result("accuracy", 0.94)
        self.assertEqual(acc_exp["quality"], "Excellent")
        self.assertIn("94.0%", acc_exp["value"])

        # 2. Metric: F1 Score
        f1_exp = explain_metric_result("f1", 0.88)
        self.assertEqual(f1_exp["quality"], "Strong")

        # 3. Metric: R2
        r2_exp = explain_metric_result("r2", 0.76)
        self.assertEqual(r2_exp["quality"], "Moderate")

        # 4. Generalization gap: Overfitting
        gap_overfit = explain_generalization_gap(0.95, 0.70)
        self.assertEqual(gap_overfit["status"], "Possible Overfitting")

        # 5. Generalization gap: Well Generalized
        gap_good = explain_generalization_gap(0.85, 0.84)
        self.assertEqual(gap_good["status"], "Well Generalized")

        # 6. REST API endpoint
        res = self.client.post(
            "/assistant/explain-context",
            json={"type": "metric", "metric_name": "f1", "value": 0.82},
        )
        self.assertEqual(res.status_code, 200)
        self.assertIn("F1 Score", res.json()["metric"])

    def test_05_preflight_validation(self):
        # 1. Upload non-CSV
        val_bad_ext = validate_preflight("upload", {"filename": "data.json", "file_size": 100})
        self.assertFalse(val_bad_ext["is_valid"])
        self.assertTrue(any("CSV" in e for e in val_bad_ext["errors"]))

        # 2. Training without target
        val_no_target = validate_preflight("train", {"target_column": "", "columns": ["a", "b"]})
        self.assertFalse(val_no_target["is_valid"])
        self.assertTrue(any("No target column" in e for e in val_no_target["errors"]))

        # 3. Training with target not in columns
        val_missing_col = validate_preflight("train", {"target_column": "z", "columns": ["a", "b"]})
        self.assertFalse(val_missing_col["is_valid"])

        # 4. Training with single-class target
        val_single_class = validate_preflight(
            "train",
            {
                "target_column": "label",
                "columns": ["a", "label"],
                "row_count": 50,
                "class_distribution": {"1": 50},
            },
        )
        self.assertFalse(val_single_class["is_valid"])
        self.assertTrue(any("at least 2 classes" in e for e in val_single_class["errors"]))

        # 5. Valid training preflight
        val_valid = validate_preflight(
            "train",
            {
                "target_column": "label",
                "columns": ["feature_1", "label"],
                "row_count": 100,
                "class_distribution": {"0": 50, "1": 50},
            },
        )
        self.assertTrue(val_valid["is_valid"])
        self.assertEqual(len(val_valid["errors"]), 0)

    def test_06_workspace_projects_crud(self):
        # Create
        create_res = self.client.post(
            "/assistant/projects",
            json={
                "name": "E-Commerce Churn Project",
                "dataset_name": "customers.csv",
                "target_column": "churn",
                "task_type": "classification",
                "experience_mode": "guided",
                "current_step": 3,
                "completed_steps": [1, 2],
            },
        )
        self.assertEqual(create_res.status_code, 200)
        proj = create_res.json()
        pid = proj["project_id"]
        self.assertEqual(proj["name"], "E-Commerce Churn Project")
        self.assertEqual(proj["current_step"], 3)

        # Get
        get_res = self.client.get(f"/assistant/projects/{pid}")
        self.assertEqual(get_res.status_code, 200)
        self.assertEqual(get_res.json()["project_id"], pid)

        # List
        list_res = self.client.get("/assistant/projects")
        self.assertEqual(list_res.status_code, 200)
        self.assertGreaterEqual(list_res.json()["total"], 1)

        # Update
        update_res = self.client.post(
            "/assistant/projects",
            json={
                "project_id": pid,
                "name": "E-Commerce Churn Project (Updated)",
                "current_step": 5,
                "completed_steps": [1, 2, 3, 4],
            },
        )
        self.assertEqual(update_res.status_code, 200)
        self.assertEqual(update_res.json()["current_step"], 5)
        self.assertEqual(len(update_res.json()["completed_steps"]), 4)

        # Delete
        del_res = self.client.delete(f"/assistant/projects/{pid}")
        self.assertEqual(del_res.status_code, 200)
        self.assertTrue(del_res.json()["deleted"])

        # Verify 404
        get_404 = self.client.get(f"/assistant/projects/{pid}")
        self.assertEqual(get_404.status_code, 404)


if __name__ == "__main__":
    unittest.main()
