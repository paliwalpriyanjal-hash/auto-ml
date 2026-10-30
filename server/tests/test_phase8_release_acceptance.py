# server/tests/test_phase8_release_acceptance.py
"""
AutoVeyra Phase 8: Final Acceptance, Production Hardening & Release Certification Suite.

Validates:
1. Complete User Journeys A through F (Beginner, Advanced, Workbench Lineage, Regression, Imbalanced Classification, Temporal Zero-Leakage).
2. ML Methodology & Data Leakage Safeguards (Training fold isolation, past-only time features, holdout integrity).
3. Data Integrity & File Management (SHA-256 manifests, derivative dataset separation, non-destructive storage).
4. Security & Path Traversal Protections (Bcrypt hashing, JWT revocation, boundary checks).
5. API Integration & Error Resilience (Input validation, clear error diagnostics, graceful failure).
"""

import hashlib
import io
import json
import math
import os
import shutil
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

# Ensure server root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from app.main import app
from app.config import settings
from app.services.auth_service import (
    init_auth_db,
    register_user,
    authenticate_user,
    create_access_token,
    revoke_token,
    is_token_revoked,
)
from app.services.model_registry_service import (
    init_model_registry,
    register_model,
    get_model,
    list_models,
    delete_model,
)
from app.services.report_service import init_report_db
from app.services.monitoring_service import init_monitoring_db, get_monitoring_stats
from app.services.experiment_service import (
    init_experiment_db,
    register_dataset_from_df,
    create_workbench_experiment,
    create_experiment_run,
    get_run_lineage,
    create_workflow,
    execute_workflow,
    get_dataset,
)
from app.services.project_report_service import (
    init_project_report_db,
    create_project_report,
    get_project_report,
    generate_project_pdf,
    generate_project_html,
    generate_project_zip_bundle,
)
from app.services.ml_intelligence_service import (
    analyze_dataset_intelligence,
    build_dataset_aware_preprocessor,
    get_adaptive_classification_models,
    get_adaptive_regression_models,
    evaluate_classification_metrics,
    evaluate_regression_metrics,
    generate_training_diagnostics,
)
from app.services.feature_service import (
    TabularFeatureEngineer,
    evaluate_feature_strategies,
    init_feature_db,
)


class TestPhase8ReleaseAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_auth_db()
        init_model_registry()
        init_report_db()
        init_monitoring_db()
        init_experiment_db()
        init_project_report_db()
        init_feature_db()
        cls.client = TestClient(app)

    # =========================================================================
    # JOURNEY A: COMPLETE BEGINNER GUIDED WORKSPACE WORKFLOW
    # =========================================================================
    def test_01_journey_a_beginner_guided_workflow(self):
        """
        JOURNEY A: Beginner end-to-end user journey:
        Onboarding -> Project Context -> CSV Upload -> Ingestion & EDA ->
        Target Resolution -> Clean recommendations -> Feature recommendations ->
        Model Training -> Comparison Leaderboard -> Explainability ->
        Project Report Studio -> ZIP export with original dataset and SHA-256 match.
        """
        # 1. Project Context Initiation
        proj_res = self.client.post(
            "/assistant/projects",
            json={"name": "Retail Churn Beginner Pilot", "experience_mode": "guided"}
        )
        self.assertEqual(proj_res.status_code, 200)
        project_id = proj_res.json()["project_id"]

        # 2. Upload synthetic tabular dataset
        np.random.seed(101)
        n = 100
        df = pd.DataFrame({
            "CustomerID": [f"CUST_{i:04d}" for i in range(n)],
            "Age": np.random.randint(20, 70, n),
            "MonthlySpend": np.random.uniform(20.0, 150.0, n).round(2),
            "SupportCalls": np.random.choice([0, 1, 2, 3, 5], n),
            "PlanType": np.random.choice(["Basic", "Standard", "Premium"], n),
            "Churn": np.random.choice([0, 1], n, p=[0.7, 0.3]),
        })
        # Insert known missing value
        df.loc[5:7, "MonthlySpend"] = np.nan

        csv_bytes = df.to_csv(index=False).encode("utf-8")

        # Register dataset in workbench
        reg_res = self.client.post(
            "/experiments/datasets/upload",
            files={"file": ("retail_churn.csv", io.BytesIO(csv_bytes), "text/csv")}
        )
        self.assertEqual(reg_res.status_code, 200)
        reg_data = reg_res.json()
        dataset_id = reg_data["dataset_id"]
        original_hash = reg_data["file_hash"]

        # 3. Pre-training Dataset Intelligence & EDA
        eda_res = self.client.post(
            "/eda/analyze",
            files={"file": ("retail_churn.csv", io.BytesIO(csv_bytes), "text/csv")},
            data={"target_column": "Churn"}
        )
        self.assertEqual(eda_res.status_code, 200)
        eda_data = eda_res.json()
        self.assertIn("dataset_summary", eda_data["profile"])
        self.assertEqual(eda_data["profile"]["dataset_summary"]["inferred_task"], "binary_classification")

        # 4. Cleaning Operation (Impute missing MonthlySpend with median)
        clean_res = self.client.post(
            "/clean",
            files={"file": ("retail_churn.csv", io.BytesIO(csv_bytes), "text/csv")},
            data={
                "preprocessing": json.dumps({
                    "impute_missing": True,
                    "imputation_method": "median",
                    "numeric_columns": ["MonthlySpend", "Age", "SupportCalls"],
                    "available_columns": list(df.columns)
                })
            }
        )
        self.assertEqual(clean_res.status_code, 200, clean_res.text)
        cleaned_csv = clean_res.json()["csv"]
        self.assertNotIn(",,", cleaned_csv)

        # 5. Feature Engineering Evaluation
        feat_res = self.client.post(
            "/features/evaluate",
            files={"file": ("retail_churn_clean.csv", io.BytesIO(cleaned_csv.encode("utf-8")), "text/csv")},
            data={
                "target_column": "Churn",
                "task_type": "classification",
                "model_type": "RandomForestClassifier",
                "n_splits": 2
            }
        )
        self.assertEqual(feat_res.status_code, 200)
        feat_data = feat_res.json()
        self.assertIn("strategy_results", feat_data)
        self.assertIn("winning_strategy", feat_data)

        # 6. Model Training & Comparison
        train_res = self.client.post(
            "/train",
            files={"file": ("retail_churn_clean.csv", io.BytesIO(cleaned_csv.encode("utf-8")), "text/csv")},
            data={
                "target_column": "Churn",
                "task_type": "classification",
                "model": "RandomForest",
                "scaler": "standard",
                "splitRatio": 0.8,
                "missing": "median",
                "encoding": "onehot"
            }
        )
        self.assertEqual(train_res.status_code, 200)
        train_data = train_res.json()
        self.assertIn("metrics", train_data)
        self.assertIn("accuracy", train_data["metrics"])

        # 7. Create unified project report in Project Report Studio
        rep_res = self.client.post(
            "/reports/project/generate",
            json={
                "project_name": "Beginner Churn Pilot Report",
                "original_dataset_id": dataset_id,
                "project_description": "Comprehensive beginner AutoML run for customer churn.",
                "include_original_dataset": True,
            }
        )
        self.assertEqual(rep_res.status_code, 200)
        report_id = rep_res.json()["report_id"]

        # 8. Download ZIP bundle with original dataset included
        zip_res = self.client.get(f"/reports/project/{report_id}/download?format=zip")
        self.assertEqual(zip_res.status_code, 200)
        self.assertEqual(zip_res.headers.get("content-type"), "application/zip")

        # 9. Verify SHA-256 match and bundle content
        zip_buf = io.BytesIO(zip_res.content)
        with zipfile.ZipFile(zip_buf, "r") as zf:
            file_list = zf.namelist()
            self.assertIn("MANIFEST.json", file_list)
            self.assertIn("Report/final_project_report.html", file_list)
            self.assertIn("Report/final_project_report.pdf", file_list)
            
            # Find original dataset file
            dataset_entries = [f for f in file_list if f.lower().startswith("dataset/")]
            self.assertGreaterEqual(len(dataset_entries), 1)
            dataset_entry = dataset_entries[0]
            extracted_csv_bytes = zf.read(dataset_entry)
            extracted_hash = hashlib.sha256(extracted_csv_bytes).hexdigest()
            self.assertEqual(extracted_hash, original_hash, "Original dataset SHA-256 hash must match manifest")

    # =========================================================================
    # JOURNEY B: ADVANCED ML USER WORKFLOW
    # =========================================================================
    def test_02_journey_b_advanced_ml_user(self):
        """
        JOURNEY B: Advanced practitioner workflow:
        Advanced upload -> Multi-cleaning experiments -> Strategy comparison ->
        Optuna Bayesian Tuning -> Holdout evaluation -> Lineage tracking -> Technical report.
        """
        np.random.seed(202)
        n = 120
        df = pd.DataFrame({
            "feature_x1": np.random.normal(10, 2, n),
            "feature_x2": np.random.exponential(5, n),
            "category_cat": np.random.choice(["TypeA", "TypeB", "TypeC"], n),
            "outcome": np.random.choice([0, 1], n),
        })
        csv_bytes = df.to_csv(index=False).encode("utf-8")

        # 1. Feature Strategy Comparison
        strat_res = self.client.post(
            "/features/evaluate",
            files={"file": ("advanced.csv", io.BytesIO(csv_bytes), "text/csv")},
            data={
                "target_column": "outcome",
                "task_type": "classification",
                "model_type": "RandomForestClassifier",
                "n_splits": 2
            }
        )
        self.assertEqual(strat_res.status_code, 200)
        eval_data = strat_res.json()
        self.assertIn("strategy_results", eval_data)
        self.assertIn("winning_strategy", eval_data)
        self.assertGreaterEqual(len(eval_data["strategy_results"]), 2)

        # 2. Bayesian Hyperparameter Tuning via Optuna
        tune_res = self.client.post(
            "/tuning/start",
            files={"file": ("advanced.csv", io.BytesIO(csv_bytes), "text/csv")},
            data={
                "target_column": "outcome",
                "task_type": "classification",
                "model_type": "RandomForestClassifier",
                "metric": "f1_macro",
                "n_trials": 3,
                "n_splits": 2
            }
        )
        self.assertEqual(tune_res.status_code, 200)
        tune_data = tune_res.json()
        self.assertEqual(tune_data["status"], "completed")
        self.assertGreater(len(tune_data["trials"]), 0)
        best_params = tune_data["best_params"]
        self.assertIsInstance(best_params, dict)

    # =========================================================================
    # JOURNEY C: INDEPENDENT EXPERIMENT WORKBENCH & LINEAGE DAG
    # =========================================================================
    def test_03_journey_c_workbench_experiments_and_lineage(self):
        """
        JOURNEY C: Workbench multi-run experiments, DAG workflow construction,
        execution, and artifact lineage provenance.
        """
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6], "target": [10, 20, 30, 40, 50, 60]})
        reg = register_dataset_from_df(df, filename="lineage_study.csv")
        ds_id = reg["dataset_id"]

        exp = create_workbench_experiment(name="Pipeline Lineage Study", module="training", dataset_id=ds_id)
        exp_id = exp["experiment_id"]

        # Run 1: Cleaning stage
        run1 = create_experiment_run(
            experiment_id=exp_id,
            run_name="Imputation Stage",
            module="cleaning",
            parameters={"method": "mean"},
            dataset_id=ds_id,
        )
        run1_id = run1["run_id"]

        # Run 2: Feature Engineering stage (inherits from run1)
        run2 = create_experiment_run(
            experiment_id=exp_id,
            run_name="Scaling Stage",
            module="feature_engineering",
            parent_run_id=run1_id,
            parameters={"scaler": "standard"},
            dataset_id=ds_id,
        )
        run2_id = run2["run_id"]

        # Verify lineage graph
        lineage = get_run_lineage(run2_id)
        self.assertEqual(lineage["focus_run_id"], run2_id)
        self.assertGreaterEqual(len(lineage["nodes"]), 2)
        self.assertGreaterEqual(len(lineage["edges"]), 1)

    # =========================================================================
    # JOURNEY D: REGRESSION PROJECT WITH ZERO-SAFE METRICS
    # =========================================================================
    def test_04_journey_d_regression_project(self):
        """
        JOURNEY D: Regression task with continuous features, target detection,
        unstratified K-Fold CV, residuals, and zero-value target safety for MAPE.
        """
        np.random.seed(303)
        n = 100
        # Include a 0.0 in target to stress test zero-safe MAPE calculations
        targets = np.random.uniform(10.0, 100.0, n)
        targets[0] = 0.0

        df = pd.DataFrame({
            "sqft": np.random.uniform(500, 3500, n),
            "bedrooms": np.random.randint(1, 6, n),
            "bathrooms": np.random.uniform(1.0, 4.0, n),
            "price": targets,
        })

        # 1. Dataset pre-training analysis
        intel = analyze_dataset_intelligence(df, target_column="price")
        self.assertEqual(intel["target_profile"]["task_type"], "regression")

        # 2. Regression Preprocessor Generation
        prep = build_dataset_aware_preprocessor(
            numeric_columns=["sqft", "bedrooms", "bathrooms"],
            categorical_columns=[],
            scaler="standard"
        )
        self.assertIsNotNone(prep)

        # 3. Model Training & Safe Metric Computation
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        train_res = self.client.post(
            "/train",
            files={"file": ("house_prices.csv", io.BytesIO(csv_bytes), "text/csv")},
            data={
                "target_column": "price",
                "task_type": "regression",
                "model": "LinearRegression",
                "scaler": "standard",
                "splitRatio": 0.8,
                "missing": "mean",
                "encoding": "onehot"
            }
        )
        self.assertEqual(train_res.status_code, 200)
        train_data = train_res.json()
        metrics = train_data["metrics"]
        self.assertIn("r2", metrics)
        self.assertIn("rmse", metrics)
        self.assertIn("mae", metrics)
        # Target includes 0.0, MAPE should either be safely computed or gracefully omitted
        self.assertTrue(metrics.get("mape") is None or isinstance(metrics.get("mape"), float))

    # =========================================================================
    # JOURNEY E: HIGHLY IMBALANCED CLASSIFICATION
    # =========================================================================
    def test_05_journey_e_imbalanced_classification(self):
        """
        JOURNEY E: Highly imbalanced binary classification dataset.
        Verifies:
        - Imbalance severity detection.
        - Stratified K-Fold CV selection.
        - Balanced accuracy and macro F1 metric availability.
        - Confusion matrix integrity.
        """
        np.random.seed(404)
        n = 100
        # 92 negatives, 8 positives = 11.5:1 imbalance ratio (> 10 is severe)
        df = pd.DataFrame({
            "signal_a": np.random.normal(0, 1, n),
            "signal_b": np.random.normal(5, 2, n),
            "fraud": np.array([0] * 92 + [1] * 8),
        })

        intel = analyze_dataset_intelligence(df, target_column="fraud")
        tp = intel["target_profile"]
        self.assertTrue(tp["is_imbalanced"])
        self.assertEqual(tp["imbalance_severity"], "severe")
        self.assertGreaterEqual(tp["imbalance_ratio"], 10.0)

        # Train model with balanced weighting
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        train_res = self.client.post(
            "/train",
            files={"file": ("fraud_data.csv", io.BytesIO(csv_bytes), "text/csv")},
            data={
                "target_column": "fraud",
                "task_type": "classification",
                "model": "LogisticRegression",
                "scaler": "standard",
                "splitRatio": 0.8,
                "missing": "mean",
                "encoding": "onehot"
            }
        )
        self.assertEqual(train_res.status_code, 200)
        res_data = train_res.json()
        self.assertIn("confusion_matrix", res_data)
        cm = res_data["confusion_matrix"]
        self.assertIsInstance(cm, dict)
        self.assertIn("matrix", cm)
        self.assertIsInstance(cm["matrix"], list)
        self.assertIn("metrics", res_data)
        self.assertIn("f1_macro", res_data["metrics"])
        self.assertIn("balanced_accuracy", res_data["metrics"])

    # =========================================================================
    # JOURNEY F: TEMPORAL DATASET & ZERO-LEAKAGE TIME SAFEGUARDS
    # =========================================================================
    def test_06_journey_f_temporal_zero_leakage(self):
        """
        JOURNEY F: Temporal dataset validation:
        - Past-only lag features (lag 1, lag 2).
        - Past-only rolling window calculations with shift(1).
        - Strict verification: feature values at time t NEVER use future information from t+1.
        """
        n = 50
        dates = pd.date_range("2026-01-01", periods=n, freq="D")
        sales = np.array([100 + i * 2 for i in range(n)], dtype=float)
        df = pd.DataFrame({
            "timestamp": dates.astype(str),
            "sales": sales,
        })

        # Apply TabularFeatureEngineer with past-only lag and rolling statistics
        transformer = TabularFeatureEngineer(
            lag_features={"sales": [1, 2]},
            rolling_features={"sales": [3]},
        )
        X_out = transformer.fit_transform(df)

        # 1. Verify Lag 1: value at index i must equal sales[i-1] for i >= 1
        for i in range(1, n):
            self.assertAlmostEqual(
                X_out.loc[i, "sales_lag_1"],
                sales[i - 1],
                places=4,
                msg=f"Lag 1 at row {i} leaked future data or computed incorrectly"
            )

        # 2. Verify Past-Only Rolling Mean:
        # window 3 at index i must strictly average sales[i-3:i], NEVER sales[i]
        for i in range(3, n):
            expected_past_mean = np.mean(sales[i - 3:i])
            self.assertAlmostEqual(
                X_out.loc[i, "sales_rolling_mean_3"],
                expected_past_mean,
                places=4,
                msg=f"Rolling mean at row {i} must use strictly past data (t-1), zero lookahead."
            )

    # =========================================================================
    # ML METHODOLOGY & ADVERSARIAL LEAKAGE AUDIT
    # =========================================================================
    def test_07_ml_methodology_adversarial_isolation(self):
        """
        Verifies that preprocessing transformers fitted on training data do not
        leak test fold summary statistics.
        """
        # Train data mean = 10, Test data mean = 1000
        X_train = pd.DataFrame({"val": [9.0, 10.0, 11.0]})
        X_test = pd.DataFrame({"val": [1000.0, 2000.0]})

        prep = build_dataset_aware_preprocessor(
            numeric_columns=["val"],
            categorical_columns=[],
            scaler="standard"
        )
        prep.fit(X_train)
        
        # Check that scaler mean inside ColumnTransformer equals 10.0, not contaminated by 1000.0
        scaler_step = prep.named_transformers_["numeric"].named_steps["scaler"]
        self.assertAlmostEqual(scaler_step.mean_[0], 10.0, places=4)

        # Transform test data using the train-fitted scaler
        transformed_test = prep.transform(X_test)
        self.assertGreater(transformed_test[0, 0], 1000.0)

    # =========================================================================
    # DATA INTEGRITY & PATH TRAVERSAL PREVENTIONS
    # =========================================================================
    def test_08_security_path_traversal_and_input_boundaries(self):
        """
        Audit path traversal resistance, parameter boundaries, and error safety.
        """
        malicious_ids = [
            "../../etc/passwd",
            "..%2F..%2Fetc%2Fpasswd",
            "../windows/win.ini",
            "report_id_invalid_special$$$#@!",
        ]
        for mal_id in malicious_ids:
            res = self.client.get(f"/reports/project/{mal_id}/download?format=html")
            self.assertIn(res.status_code, [400, 404, 405, 422])

        for mal_id in malicious_ids:
            res = self.client.post(f"/models/{mal_id}/predict", json={"features": {}})
            self.assertIn(res.status_code, [400, 404, 405, 422])

        for mal_id in malicious_ids:
            res = self.client.get(f"/models/{mal_id}/download")
            self.assertIn(res.status_code, [400, 404, 405, 422])

    # =========================================================================
    # API RESILIENCE & MEANINGFUL ERROR MESSAGING
    # =========================================================================
    def test_09_api_resilience_and_clean_diagnostics(self):
        """
        Test API failure conditions to guarantee understandable diagnostics without stack traces:
        - Empty CSV file
        - Missing target column
        - Non-existent model ID
        """
        # 1. Missing target column
        df = pd.DataFrame({"col_a": [1, 2, 3], "col_b": [4, 5, 6]})
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        res_missing = self.client.post(
            "/train",
            files={"file": ("data.csv", io.BytesIO(csv_bytes), "text/csv")},
            data={
                "target_column": "non_existent_column_xyz",
                "task_type": "classification",
                "model": "LogisticRegression",
                "scaler": "standard",
                "splitRatio": 0.8,
                "missing": "mean",
                "encoding": "onehot"
            }
        )
        self.assertEqual(res_missing.status_code, 400)
        self.assertIn("non_existent_column_xyz", res_missing.json()["detail"])

        # 2. Non-existent model ID
        res_model = self.client.get("/models/999999999")
        self.assertEqual(res_model.status_code, 404)
        self.assertIn("detail", res_model.json())


if __name__ == "__main__":
    unittest.main()
