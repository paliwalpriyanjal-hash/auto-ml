import os
import sys
import io
import asyncio
import unittest
import numpy as np
import pandas as pd

# Set headless matplotlib
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from app.main import app
from app.services.eda_service import (
    profile_dataset,
    detect_eda_strategy,
    analyze_numerical_features,
    analyze_categorical_features,
    analyze_target_relationships,
    analyze_data_quality,
    generate_eda_insights,
    generate_eda_flowchart,
    generate_eda_pdf_report,
    generate_eda_html_report,
    run_full_eda_pipeline,
    get_eda_analysis_from_db,
    list_eda_history_from_db,
)

client = TestClient(app)


class TestEDAEngine(unittest.TestCase):
    """
    Comprehensive test suite for AutoVeyra Phase 2:
    Advanced Adaptive EDA Intelligence & Automated Analysis Report Engine.
    """

    def setUp(self):
        # Create deterministic synthetic test datasets
        np.random.seed(42)
        n = 120
        self.mixed_df = pd.DataFrame({
            "age": np.random.randint(18, 70, size=n),
            "income": np.random.exponential(scale=50000, size=n),
            "experience": np.random.randint(0, 30, size=n),
            "department": np.random.choice(["Sales", "Engineering", "Marketing", "HR"], size=n),
            "city": np.random.choice(["New York", "San Francisco", "Austin", "Chicago"], size=n),
            "performance_score": np.random.uniform(1.0, 5.0, size=n),
            "promoted": np.random.choice([0, 1], size=n, p=[0.7, 0.3]),
        })

    def _post_eda_file(self, endpoint: str, df: pd.DataFrame, form_data: dict = None):
        csv_buffer = io.BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        files = {"file": ("test_dataset.csv", csv_buffer, "text/csv")}
        return client.post(endpoint, data=form_data or {}, files=files)

    # ------------------------------------------------------------------
    # 1. Dataset Profiling Unit Tests
    # ------------------------------------------------------------------
    def test_01_profile_dataset_structure(self):
        """Dataset profiling returns structural dimensions, column profiles, and semantic types."""
        profile = profile_dataset(self.mixed_df)
        self.assertIn("dataset_summary", profile)
        self.assertIn("column_profiles", profile)

        summary = profile["dataset_summary"]
        self.assertEqual(summary["row_count"], 120)
        self.assertEqual(summary["column_count"], 7)
        self.assertEqual(summary["numeric_columns_count"], 4)
        self.assertEqual(summary["boolean_columns_count"], 1)
        self.assertEqual(summary["categorical_columns_count"], 2)

        col_profiles = profile["column_profiles"]
        self.assertEqual(len(col_profiles), 7)

        # Verify column metadata
        age_col = col_profiles["age"]
        self.assertIn("min", age_col)
        self.assertIn("max", age_col)
        self.assertIn("mean", age_col)
        self.assertIn("std", age_col)
        self.assertIn("skewness", age_col)
        self.assertIn("kurtosis", age_col)

        dept_col = col_profiles["department"]
        self.assertIn("top_categories", dept_col)
        self.assertGreaterEqual(len(dept_col["top_categories"]), 1)

    # ------------------------------------------------------------------
    # 2. Adaptive Strategy Engine Unit Tests
    # ------------------------------------------------------------------
    def test_02_detect_eda_strategy_profiles(self):
        """Strategy engine adapts correctly across different dataset shapes and target types."""
        profile = profile_dataset(self.mixed_df)
        strategy = detect_eda_strategy(self.mixed_df, profile, target_column="promoted")

        self.assertEqual(strategy["target_column"], "promoted")
        self.assertEqual(strategy["task_type"], "binary_classification")
        self.assertIn("classification_target", strategy["active_profiles"])
        self.assertIn("mixed_tabular", strategy["active_profiles"])
        self.assertGreater(len(strategy["strategy_reasons"]), 0)

        # Test numerical-heavy profile
        num_df = pd.DataFrame(np.random.randn(50, 10), columns=[f"col_{i}" for i in range(10)])
        num_profile = profile_dataset(num_df)
        num_strategy = detect_eda_strategy(num_df, num_profile)
        self.assertIn("numerical_heavy", num_strategy["active_profiles"])

        # Test small dataset profile
        small_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        small_profile = profile_dataset(small_df)
        small_strategy = detect_eda_strategy(small_df, small_profile)
        self.assertIn("small_dataset", small_strategy["active_profiles"])

        # Test regression task profile
        reg_strategy = detect_eda_strategy(self.mixed_df, profile, target_column="performance_score")
        self.assertEqual(reg_strategy["task_type"], "regression")
        self.assertIn("regression_target", reg_strategy["active_profiles"])

    # ------------------------------------------------------------------
    # 3. Numerical Analysis Unit Tests
    # ------------------------------------------------------------------
    def test_03_analyze_numerical_features(self):
        """Numerical analysis calculates moments, outliers, and collinear pairs."""
        collinear_df = pd.DataFrame({
            "var_a": np.linspace(1, 100, 100),
            "var_b": np.linspace(1, 100, 100) + np.random.normal(0, 0.01, 100), # highly collinear
            "var_c": np.random.normal(0, 1, 100),
        })
        num_cols = ["var_a", "var_b", "var_c"]
        num_analysis = analyze_numerical_features(collinear_df, numeric_columns=num_cols)

        self.assertIn("summary_table", num_analysis)
        self.assertIn("pearson_matrix", num_analysis)
        self.assertIn("spearman_matrix", num_analysis)
        self.assertIn("highly_correlated_pairs", num_analysis)

        # Check moments
        summary_table = {item["column"]: item for item in num_analysis["summary_table"]}
        stats_a = summary_table["var_a"]
        self.assertIn("skewness", stats_a)
        self.assertIn("kurtosis", stats_a)
        self.assertIn("outlier_count", stats_a)

        # Check collinearity detection
        collinear_pairs = num_analysis["highly_correlated_pairs"]
        self.assertGreaterEqual(len(collinear_pairs), 1)
        self.assertEqual(collinear_pairs[0]["feature_1"], "var_a")
        self.assertEqual(collinear_pairs[0]["feature_2"], "var_b")
        self.assertGreaterEqual(abs(collinear_pairs[0]["pearson"]), 0.99)

    # ------------------------------------------------------------------
    # 4. Categorical Analysis Unit Tests
    # ------------------------------------------------------------------
    def test_04_analyze_categorical_features(self):
        """Categorical analysis computes proportions, rare categories, and Chi-Square / Cramer's V."""
        cat_df = pd.DataFrame({
            "segment": ["Enterprise"] * 80 + ["SMB"] * 35 + ["Micro"] * 5,
            "region": ["East"] * 60 + ["West"] * 60,
        })
        cat_cols = ["segment", "region"]
        cat_analysis = analyze_categorical_features(cat_df, categorical_columns=cat_cols)

        self.assertIn("frequency_distributions", cat_analysis)
        self.assertIn("rare_category_alerts", cat_analysis)
        self.assertIn("associations", cat_analysis)

        # Check rare categories detection (Micro is 5/120 < 5%)
        rare_alerts = cat_analysis["rare_category_alerts"]
        self.assertTrue(any(a["column"] == "segment" for a in rare_alerts))

        # Check association between segment and region
        assoc = cat_analysis["associations"]
        self.assertGreaterEqual(len(assoc), 1)
        self.assertIn("cramers_v", assoc[0])
        self.assertIn("p_value", assoc[0])

    # ------------------------------------------------------------------
    # 5. Target-Aware Analysis Unit Tests
    # ------------------------------------------------------------------
    def test_05_analyze_target_relationships(self):
        """Target-aware analysis generates class-stratified metrics and correlation rankings."""
        profile = profile_dataset(self.mixed_df)
        num_cols = profile["dataset_summary"]["numeric_columns"]
        cat_cols = profile["dataset_summary"]["categorical_columns"]

        # Classification target
        cls_target_analysis = analyze_target_relationships(
            df=self.mixed_df,
            target_column="promoted",
            task_type="binary_classification",
            numeric_columns=num_cols,
            categorical_columns=cat_cols,
        )
        self.assertEqual(cls_target_analysis["task_type"], "classification")
        self.assertIn("class_proportions", cls_target_analysis)
        self.assertIn("class_numerical_profiles", cls_target_analysis)
        self.assertIn("categorical_associations", cls_target_analysis)

        # Regression target
        reg_target_analysis = analyze_target_relationships(
            df=self.mixed_df,
            target_column="performance_score",
            task_type="regression",
            numeric_columns=num_cols,
            categorical_columns=cat_cols,
        )
        self.assertEqual(reg_target_analysis["task_type"], "regression")
        self.assertIn("target_statistics", reg_target_analysis)
        self.assertIn("correlation_ranking", reg_target_analysis)

    # ------------------------------------------------------------------
    # 6. Data Quality Diagnostics Unit Tests
    # ------------------------------------------------------------------
    def test_06_analyze_data_quality(self):
        """Data quality detects missingness, duplicates, constant columns, and leakage."""
        dirty_df = pd.DataFrame({
            "id": range(10),
            "constant_col": [42] * 10,
            "missing_col": [1, None, 3, None, 5, 6, 7, 8, 9, 10],
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "leaky_col": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1], # 1.0 correlation with target
        })
        quality = analyze_data_quality(dirty_df, target_column="target")
        self.assertIn("columns_with_missing", quality)
        self.assertTrue(any(item["column"] == "missing_col" for item in quality["columns_with_missing"]))
        self.assertEqual(quality["total_missing_cells"], 2)
        self.assertTrue(any(item["column"] == "leaky_col" for item in quality["leakage_candidates"]))

    # ------------------------------------------------------------------
    # 7. Insights Engine & Flowchart Unit Tests
    # ------------------------------------------------------------------
    def test_07_insights_and_flowchart_generation(self):
        """Generates evidence-backed structured findings and dynamic workflow flowchart."""
        pipeline_result = asyncio.run(run_full_eda_pipeline(self.mixed_df, target_column="promoted"))
        insights = pipeline_result["insights"]
        flowchart = pipeline_result["flowchart"]

        self.assertIn("critical", insights)
        self.assertIn("warnings", insights)
        self.assertIn("info", insights)
        self.assertIn("recommendations", insights)
        self.assertIn("total_insights_count", insights)

        self.assertIn("total_steps", flowchart)
        self.assertIn("steps", flowchart)
        self.assertGreaterEqual(flowchart["completed_steps"], 5)

        for step in flowchart["steps"]:
            self.assertIn(step["status"], ["completed", "skipped"])
            self.assertIn("details", step)

    # ------------------------------------------------------------------
    # 8. Report Generation Unit Tests (PDF, HTML, JSON)
    # ------------------------------------------------------------------
    def test_08_pdf_and_html_report_generation(self):
        """PDF generator creates valid PDF document and HTML generator outputs standalone markup."""
        pipeline_result = asyncio.run(run_full_eda_pipeline(self.mixed_df, target_column="promoted"))

        # Test PDF generator
        pdf_bytes = generate_eda_pdf_report(pipeline_result)
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertTrue(pdf_bytes.startswith(b"%PDF-"), "Generated PDF does not start with %PDF- header")
        self.assertGreater(len(pdf_bytes), 1000)

        # Test HTML generator
        html_markup = generate_eda_html_report(pipeline_result)
        self.assertIsInstance(html_markup, str)
        self.assertIn("<!DOCTYPE html>", html_markup)
        self.assertIn("AutoVeyra", html_markup)
        self.assertIn("AutoVeyra Automated EDA Report", html_markup)

    # ------------------------------------------------------------------
    # 9. FastAPI Route Integration Tests
    # ------------------------------------------------------------------
    def test_09_api_profile_endpoint(self):
        """POST /eda/profile successfully returns dataset profiling summary."""
        res = self._post_eda_file("/eda/profile", self.mixed_df)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("profile", data)
        self.assertIn("strategy", data)
        self.assertEqual(data["profile"]["dataset_summary"]["row_count"], 120)
        self.assertEqual(data["profile"]["dataset_summary"]["column_count"], 7)

    def test_10_api_analyze_and_retrieval_endpoints(self):
        """POST /eda/analyze runs full pipeline and subsequent GET endpoints return artifacts."""
        # 1. Run full analysis
        res = self._post_eda_file("/eda/analyze", self.mixed_df, {"target_column": "promoted"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("analysis_id", data)
        analysis_id = data["analysis_id"]
        self.assertTrue(analysis_id.startswith("eda_"))

        # 2. GET /eda/{id}
        res_get = client.get(f"/eda/{analysis_id}")
        self.assertEqual(res_get.status_code, 200)
        retrieved = res_get.json()
        self.assertEqual(retrieved["analysis_id"], analysis_id)

        # 3. GET /eda/{id}/insights
        res_ins = client.get(f"/eda/{analysis_id}/insights")
        self.assertEqual(res_ins.status_code, 200)
        self.assertIn("total_insights_count", res_ins.json())

        # 4. GET /eda/{id}/flowchart
        res_fl = client.get(f"/eda/{analysis_id}/flowchart")
        self.assertEqual(res_fl.status_code, 200)
        self.assertIn("steps", res_fl.json())

        # 5. GET /eda/{id}/report/pdf
        res_pdf = client.get(f"/eda/{analysis_id}/report/pdf")
        self.assertEqual(res_pdf.status_code, 200)
        self.assertEqual(res_pdf.headers.get("content-type"), "application/pdf")
        self.assertTrue(res_pdf.content.startswith(b"%PDF-"))

        # 6. GET /eda/{id}/report/html
        res_html = client.get(f"/eda/{analysis_id}/report/html")
        self.assertEqual(res_html.status_code, 200)
        self.assertIn("text/html", res_html.headers.get("content-type"))
        self.assertIn("<!DOCTYPE html>", res_html.text)

        # 7. GET /eda/{id}/report/json
        res_json = client.get(f"/eda/{analysis_id}/report/json")
        self.assertEqual(res_json.status_code, 200)
        self.assertEqual(res_json.json()["analysis_id"], analysis_id)

        # 8. GET /eda/history
        res_hist = client.get("/eda/history")
        self.assertEqual(res_hist.status_code, 200)
        history = res_hist.json()
        self.assertTrue(isinstance(history, list))
        self.assertGreaterEqual(len(history), 1)

    def test_11_api_error_handling(self):
        """API properly validates inputs and returns 404 for nonexistent records."""
        # Non-existent analysis ID
        res_404 = client.get("/eda/nonexistent_id_12345")
        self.assertEqual(res_404.status_code, 404)

        # Empty CSV upload
        empty_buffer = io.BytesIO(b"")
        files = {"file": ("empty.csv", empty_buffer, "text/csv")}
        res_bad = client.post("/eda/analyze", files=files)
        self.assertEqual(res_bad.status_code, 400)


if __name__ == "__main__":
    unittest.main()
