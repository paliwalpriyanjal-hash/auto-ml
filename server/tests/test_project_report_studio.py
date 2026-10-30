# server/tests/test_project_report_studio.py
"""
AutoVeyra Phase 5 Test Suite:
Comprehensive unit and integration tests for Unified AI/ML Project Report Studio,
Experiment Compatibility Validation, Chart Rendering, ReportLab PDF, and ZIP Project Bundling.
"""

import io
import json
import os
import sys
import unittest
import zipfile
import numpy as np
import pandas as pd

# Add server root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from app.main import app
from app.services.experiment_service import (
    create_experiment_run,
    create_workbench_experiment,
    init_experiment_db,
    register_dataset_from_df,
)
from app.services.project_report_service import (
    create_project_report,
    delete_project_report,
    generate_project_html,
    generate_project_pdf,
    generate_project_zip_bundle,
    get_project_report,
    init_project_report_db,
    list_project_reports,
    render_project_charts,
    validate_project_runs_compatibility,
)


class TestProjectReportStudio(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_experiment_db()
        init_project_report_db()
        cls.client = TestClient(app)

        # Create a deterministic synthetic classification dataset
        np.random.seed(42)
        n = 80
        cls.df = pd.DataFrame({
            "age": np.random.randint(18, 70, n),
            "income": np.random.normal(50000, 15000, n),
            "credit_score": np.random.normal(650, 50, n),
            "churned": np.random.choice([0, 1], n),
        })
        # Add a couple NaNs for missingness chart verification
        cls.df.loc[0:3, "credit_score"] = np.nan

        reg_meta = register_dataset_from_df(cls.df, filename="customer_churn.csv")
        cls.dataset_id = reg_meta["dataset_id"]
        cls.dataset_hash = reg_meta["file_hash"]

        # Create sample runs across modules for compilation
        exp = create_workbench_experiment(
            name="Churn Pipeline Study",
            module="training",
            dataset_id=cls.dataset_id,
        )
        cls.exp_id = exp["experiment_id"]

        run_train = create_experiment_run(
            experiment_id=cls.exp_id,
            run_name="Logistic Regression Baseline",
            parameters={"model_type": "logistic", "target_column": "churned", "task_type": "classification"},
        )
        cls.train_run_id = run_train["run_id"]

        run_eda = create_experiment_run(
            experiment_id=cls.exp_id,
            run_name="EDA Profiling Stage",
            module="eda",
            parameters={"target_column": "churned"},
        )
        cls.eda_run_id = run_eda["run_id"]

    def test_01_compatibility_validation_success(self):
        """Verify that runs with matching dataset ancestry and target column pass validation."""
        val = validate_project_runs_compatibility(
            dataset_id=self.dataset_id,
            selected_runs={
                "eda": self.eda_run_id,
                "training": self.train_run_id,
            },
        )
        self.assertTrue(val["is_compatible"])
        self.assertEqual(len(val["conflicts"]), 0)
        self.assertEqual(val["resolved_target"], "churned")
        self.assertEqual(val["dataset_hash"], self.dataset_hash)

    def test_02_compatibility_validation_conflicts(self):
        """Verify that non-existent runs or mismatched configurations produce explicit conflicts."""
        val = validate_project_runs_compatibility(
            dataset_id=self.dataset_id,
            selected_runs={"training": "run_non_existent_999"},
        )
        self.assertFalse(val["is_compatible"])
        self.assertGreaterEqual(len(val["conflicts"]), 1)
        self.assertIn("does not exist", val["conflicts"][0]["reason"])

    def test_03_create_project_report(self):
        """Verify creating a unified project report compiling multi-module results."""
        payload = {
            "project_name": "Customer Churn Prediction System",
            "project_description": "Final academic year technical project report.",
            "author_name": "Jane Doe",
            "institution": "Institute of Technology",
            "report_template": "academic",
            "original_dataset_id": self.dataset_id,
            "selected_runs": {
                "eda": self.eda_run_id,
                "training": self.train_run_id,
            },
            "include_original_dataset": True,
        }

        report = create_project_report(payload)
        self.assertIn("report_id", report)
        self.assertEqual(report["project_name"], "Customer Churn Prediction System")
        self.assertEqual(report["original_dataset_id"], self.dataset_id)
        self.assertEqual(report["dataset_sha256"], self.dataset_hash)

        # Verify 15 sections assembly
        rep_data = report["report_data"]
        self.assertIn("cover_page", rep_data)
        self.assertIn("executive_summary", rep_data)
        self.assertIn("dataset_information", rep_data)
        self.assertIn("exploratory_data_analysis", rep_data)
        self.assertIn("model_training", rep_data)
        self.assertIn("workflow_flowchart", rep_data)
        self.assertIn("conclusion_limitations", rep_data)

    def test_04_chart_rendering_engine(self):
        """Verify that high-resolution Matplotlib charts are rendered as valid PNG bytes."""
        payload = {
            "project_name": "Chart Test Report",
            "report_template": "professional",
            "original_dataset_id": self.dataset_id,
            "selected_runs": {"training": self.train_run_id},
        }
        report = create_project_report(payload)
        charts = render_project_charts(self.df, report["report_data"])

        self.assertIn("missingness", charts)
        self.assertIn("correlation", charts)
        self.assertIn("workflow_flowchart", charts)

        # PNG magic number verification: \x89PNG\r\n\x1a\n
        for c_key, c_val in charts.items():
            self.assertTrue(c_val["bytes"].startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertTrue(c_val["base64"].startswith("data:image/png;base64,"))

    def test_05_pdf_and_html_generation(self):
        """Verify ReportLab PDF and responsive HTML generation."""
        payload = {
            "project_name": "PDF/HTML Export Verification",
            "report_template": "technical",
            "original_dataset_id": self.dataset_id,
            "selected_runs": {"training": self.train_run_id},
        }
        report = create_project_report(payload)
        charts = render_project_charts(self.df, report["report_data"])

        # PDF
        pdf_bytes = generate_project_pdf(report["report_data"], charts)
        self.assertTrue(len(pdf_bytes) > 1000)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))

        # HTML
        html_str = generate_project_html(report["report_data"], charts)
        self.assertIn("PDF/HTML Export Verification", html_str)
        self.assertIn("customer_churn.csv", html_str)
        self.assertIn("<html", html_str.lower())

    def test_06_project_zip_bundle_integrity(self):
        """
        Verify that the downloadable ZIP package preserves the exact original dataset,
        PDF/HTML reports, charts, and a cryptographic MANIFEST.json.
        """
        payload = {
            "project_name": "ZIP Bundle Verification Project",
            "report_template": "complete",
            "original_dataset_id": self.dataset_id,
            "selected_runs": {"training": self.train_run_id},
            "include_original_dataset": True,
        }
        report = create_project_report(payload)

        zip_bytes = generate_project_zip_bundle(report["report_id"])
        self.assertTrue(len(zip_bytes) > 2000)

        # Open and inspect ZIP contents
        with zipfile.ZipFile(io.BytesIO(zip_bytes), mode="r") as zf:
            file_list = zf.namelist()
            self.assertIn("Report/final_project_report.pdf", file_list)
            self.assertIn("Report/final_project_report.html", file_list)
            self.assertIn("Report/final_project_report.json", file_list)
            self.assertIn("Dataset/customer_churn.csv", file_list)
            self.assertIn("Workflow/workflow_lineage.json", file_list)
            self.assertIn("MANIFEST.json", file_list)

            # Verify packaged original dataset SHA-256 checksum
            packaged_csv_bytes = zf.read("Dataset/customer_churn.csv")
            import hashlib
            computed_csv_hash = hashlib.sha256(packaged_csv_bytes).hexdigest()
            self.assertEqual(computed_csv_hash, self.dataset_hash)

            # Verify MANIFEST.json
            manifest = json.loads(zf.read("MANIFEST.json").decode("utf-8"))
            self.assertEqual(manifest["report_id"], report["report_id"])
            self.assertTrue(any(f["path"] == "Dataset/customer_churn.csv" and f["sha256"] == self.dataset_hash for f in manifest["files"]))

    def test_07_rest_api_endpoints(self):
        """Verify REST API routes for validation, creation, listing, downloads, and deletion."""
        # 1. Validation route
        r_val = self.client.post(
            "/reports/project/validate",
            json={
                "dataset_id": self.dataset_id,
                "selected_runs": {"training": self.train_run_id},
            },
        )
        self.assertEqual(r_val.status_code, 200)
        self.assertTrue(r_val.json()["is_compatible"])

        # 2. Generation route
        r_gen = self.client.post(
            "/reports/project/generate",
            json={
                "project_name": "API Automated Report",
                "original_dataset_id": self.dataset_id,
                "selected_runs": {"training": self.train_run_id},
            },
        )
        self.assertEqual(r_gen.status_code, 200)
        rep_id = r_gen.json()["report_id"]

        # 3. List route
        r_list = self.client.get("/reports/project")
        self.assertEqual(r_list.status_code, 200)
        self.assertTrue(any(r["report_id"] == rep_id for r in r_list.json()["reports"]))

        # 4. Details route
        r_detail = self.client.get(f"/reports/project/{rep_id}")
        self.assertEqual(r_detail.status_code, 200)

        # 5. Downloads: PDF
        r_pdf = self.client.get(f"/reports/project/{rep_id}/download?format=pdf")
        self.assertEqual(r_pdf.status_code, 200)
        self.assertEqual(r_pdf.headers["content-type"], "application/pdf")
        self.assertTrue(r_pdf.content.startswith(b"%PDF"))

        # 6. Downloads: HTML
        r_html = self.client.get(f"/reports/project/{rep_id}/download?format=html")
        self.assertEqual(r_html.status_code, 200)
        self.assertIn("text/html", r_html.headers["content-type"])

        # 7. Downloads: JSON
        r_json = self.client.get(f"/reports/project/{rep_id}/download?format=json")
        self.assertEqual(r_json.status_code, 200)
        self.assertEqual(r_json.headers["content-type"], "application/json")

        # 8. Downloads: ZIP bundle
        r_zip = self.client.get(f"/reports/project/{rep_id}/download?format=zip")
        self.assertEqual(r_zip.status_code, 200)
        self.assertEqual(r_zip.headers["content-type"], "application/zip")
        self.assertTrue(len(r_zip.content) > 1000)

        # 9. Delete route
        r_del = self.client.delete(f"/reports/project/{rep_id}")
        self.assertEqual(r_del.status_code, 200)


if __name__ == "__main__":
    unittest.main()
