# server/tests/test_experiments_workbench.py
"""
AutoVeyra Phase 4 Test Suite:
Comprehensive unit and integration tests for Independent Module-Wise Experiment Workbench,
Experiment Tracking, Comparison Engine, Lineage DAG, and Reusable Workflows.
"""

import io
import json
import os
import sys
import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd

# Add server root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient

from app.main import app
from app.services.experiment_service import (
    init_experiment_db,
    register_dataset_from_bytes,
    register_dataset_from_df,
    get_dataset,
    list_datasets,
    load_dataset_df,
    create_workbench_experiment,
    list_workbench_experiments,
    get_workbench_experiment,
    delete_workbench_experiment,
    create_experiment_run,
    list_experiment_runs,
    get_run,
    delete_run,
    cancel_run,
    execute_module_run,
    compare_experiment_runs,
    get_run_lineage,
    create_workflow,
    list_workflows,
    get_workflow,
    delete_workflow,
    execute_workflow,
    get_workflow_execution,
    create_experiment,
    list_experiments,
    get_experiment,
    delete_experiment,
)


class TestExperimentWorkbench(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        init_experiment_db()

        # Create a deterministic synthetic classification dataset
        np.random.seed(42)
        n_samples = 60
        cls.df_class = pd.DataFrame({
            "feature1": np.random.normal(10, 2, n_samples),
            "feature2": np.random.normal(5, 1, n_samples),
            "category": np.random.choice(["A", "B"], n_samples),
            "target": np.random.choice([0, 1], n_samples),
        })

        # Register dataset once for suite
        reg_res = register_dataset_from_df(cls.df_class, filename="synthetic_class.csv")
        cls.class_dataset_id = reg_res["dataset_id"]

    def test_01_dataset_registration_and_integrity(self):
        """Verify dataset registration, SHA-256 calculation, and schema extraction."""
        csv_bytes = b"colA,colB,colC\n1,2.5,cat\n3,4.5,dog\n5,6.5,cat\n"
        meta = register_dataset_from_bytes(csv_bytes, filename="sample.csv")

        self.assertIn("dataset_id", meta)
        self.assertEqual(meta["filename"], "sample.csv")
        self.assertEqual(meta["row_count"], 3)
        self.assertEqual(meta["column_count"], 3)
        self.assertIsInstance(meta["schema"], list)
        self.assertEqual(len(meta["schema"]), 3)

        # Idempotence: registering identical bytes should return same record
        meta2 = register_dataset_from_bytes(csv_bytes, filename="sample.csv")
        self.assertEqual(meta["dataset_id"], meta2["dataset_id"])
        self.assertEqual(meta["file_hash"], meta2["file_hash"])

        # Load back via DataFrame
        df_loaded = load_dataset_df(meta["dataset_id"])
        self.assertEqual(len(df_loaded), 3)
        self.assertListEqual(list(df_loaded.columns), ["colA", "colB", "colC"])

    def test_02_workbench_experiment_crud(self):
        """Verify creating, listing, getting, and deleting workbench experiments."""
        exp = create_workbench_experiment(
            name="Test EDA Exploration",
            module="eda",
            description="Testing EDA experiment creation",
            dataset_id=self.class_dataset_id,
            tags=["unit-test", "eda"],
        )
        exp_id = exp["experiment_id"]
        self.assertTrue(exp_id.startswith("exp_"))
        self.assertEqual(exp["module"], "eda")
        self.assertEqual(exp["tags"], ["unit-test", "eda"])

        # Fetch
        fetched = get_workbench_experiment(exp_id)
        self.assertEqual(fetched["name"], "Test EDA Exploration")

        # List
        lst = list_workbench_experiments(module="eda", search="Test EDA")
        self.assertTrue(any(e["experiment_id"] == exp_id for e in lst["experiments"]))

        # Delete
        del_res = delete_workbench_experiment(exp_id)
        self.assertEqual(del_res["experiment_id"], exp_id)

    def test_03_run_lifecycle_and_cancellation(self):
        """Verify run drafting, querying, and user cancellation."""
        exp = create_workbench_experiment(
            name="Lifecycle Test Exp",
            module="training",
            dataset_id=self.class_dataset_id,
        )
        exp_id = exp["experiment_id"]

        run = create_experiment_run(
            experiment_id=exp_id,
            run_name="Run Alpha",
            parameters={"model": "logistic", "scaler": "standard"},
        )
        run_id = run["run_id"]
        self.assertEqual(run["status"], "draft")
        self.assertEqual(run["experiment_id"], exp_id)

        # Cancel run
        cancelled = cancel_run(run_id)
        self.assertEqual(cancelled["status"], "cancelled")

        # Cleanup
        delete_run(run_id)
        delete_workbench_experiment(exp_id)

    def test_04_eda_module_execution(self):
        """Execute independent EDA module run and verify summary profile & metrics."""
        import asyncio

        exp = create_workbench_experiment(
            name="Independent EDA Experiment",
            module="eda",
            dataset_id=self.class_dataset_id,
        )
        run = create_experiment_run(
            experiment_id=exp["experiment_id"],
            run_name="EDA Profiling Run",
            parameters={"target_column": "target"},
        )

        completed_run = asyncio.run(execute_module_run(run["run_id"]))
        self.assertEqual(completed_run["status"], "completed")
        self.assertGreater(completed_run["duration_seconds"], 0.0)

        metrics = completed_run["metrics"]
        self.assertIn("rows", metrics)
        self.assertIn("columns", metrics)
        self.assertIn("primary_profile", metrics)

        artifacts = completed_run["artifacts"]
        self.assertTrue(len(artifacts) >= 1)
        self.assertEqual(artifacts[0]["type"], "eda_profile")

    def test_05_cleaning_module_execution(self):
        """Execute independent Data Cleaning run and verify derived cleaned dataset artifact."""
        import asyncio

        # Create dirty dataset with some missing values
        df_dirty = self.df_class.copy()
        df_dirty.loc[0:4, "feature1"] = np.nan
        df_dirty = pd.concat([df_dirty, df_dirty.iloc[[0, 1]]], ignore_index=True)  # duplicates

        dirty_meta = register_dataset_from_df(df_dirty, filename="dirty_test.csv")

        exp = create_workbench_experiment(
            name="Cleaning Experiment",
            module="cleaning",
            dataset_id=dirty_meta["dataset_id"],
        )
        run = create_experiment_run(
            experiment_id=exp["experiment_id"],
            run_name="Clean Missing & Duplicates",
            parameters={
                "impute_missing": True,
                "missing_strategy": "mean",
                "remove_duplicates": True,
            },
        )

        completed_run = asyncio.run(execute_module_run(run["run_id"]))
        self.assertEqual(completed_run["status"], "completed")

        metrics = completed_run["metrics"]
        self.assertIn("rows_processed", metrics)
        self.assertGreaterEqual(metrics["duplicates_removed"], 2)

        # Verify derived dataset artifact was registered
        artifacts = completed_run["artifacts"]
        self.assertTrue(len(artifacts) >= 1)
        self.assertEqual(artifacts[0]["type"], "dataset")
        derived_ds_id = artifacts[0]["dataset_id"]

        # Ensure derived dataset can be retrieved and loaded
        derived_ds = get_dataset(derived_ds_id)
        self.assertEqual(derived_ds["dataset_id"], derived_ds_id)

    def test_06_feature_engineering_module_execution(self):
        """Execute independent Feature Engineering run and verify feature expansion."""
        import asyncio

        exp = create_workbench_experiment(
            name="Feature Engineering Exp",
            module="features",
            dataset_id=self.class_dataset_id,
        )
        run = create_experiment_run(
            experiment_id=exp["experiment_id"],
            run_name="Polynomial & Interactions Run",
            parameters={
                "target_column": "target",
                "enable_polynomial": True,
                "enable_interactions": True,
                "max_interactions": 5,
            },
        )

        completed_run = asyncio.run(execute_module_run(run["run_id"]))
        self.assertEqual(completed_run["status"], "completed")

        metrics = completed_run["metrics"]
        self.assertIn("original_features", metrics)
        self.assertIn("engineered_features", metrics)
        self.assertGreaterEqual(metrics["engineered_features"], metrics["original_features"])

    def test_07_training_module_execution(self):
        """Execute independent Model Training run and verify holdout evaluation metrics."""
        import asyncio

        exp = create_workbench_experiment(
            name="Model Training Exp",
            module="training",
            dataset_id=self.class_dataset_id,
        )
        run = create_experiment_run(
            experiment_id=exp["experiment_id"],
            run_name="Logistic Regression Standard Scaler",
            parameters={
                "task_type": "classification",
                "target_column": "target",
                "model_type": "logistic",
                "scaler": "standard",
                "split_ratio": 0.8,
            },
        )

        completed_run = asyncio.run(execute_module_run(run["run_id"]))
        self.assertEqual(completed_run["status"], "completed")

        metrics = completed_run["metrics"]
        self.assertEqual(metrics["task_type"], "classification")
        self.assertIn("accuracy", metrics)
        self.assertIn("f1", metrics)

    def test_07b_tuning_module_execution(self):
        """Execute independent Hyperparameter Tuning run and verify trial optimization."""
        import asyncio

        exp = create_workbench_experiment(
            name="Optuna Tuning Exp",
            module="tuning",
            dataset_id=self.class_dataset_id,
        )
        run = create_experiment_run(
            experiment_id=exp["experiment_id"],
            run_name="Tuning Random Forest",
            parameters={
                "task_type": "classification",
                "target_column": "target",
                "model_type": "random_forest",
                "n_trials": 2,
                "n_splits": 2,
            },
        )

        completed_run = asyncio.run(execute_module_run(run["run_id"]))
        self.assertEqual(completed_run["status"], "completed")

        metrics = completed_run["metrics"]
        self.assertIn("best_score", metrics)
        self.assertIn("trials_evaluated", metrics)
        self.assertGreaterEqual(metrics["trials_evaluated"], 1)

    def test_07c_evaluation_module_execution(self):
        """Execute independent Model Evaluation run and verify statistical diagnostics."""
        import asyncio

        exp = create_workbench_experiment(
            name="Evaluation Exp",
            module="evaluation",
            dataset_id=self.class_dataset_id,
        )
        run = create_experiment_run(
            experiment_id=exp["experiment_id"],
            run_name="Target Class Distribution Eval",
            parameters={"target_column": "target"},
        )

        completed_run = asyncio.run(execute_module_run(run["run_id"]))
        self.assertEqual(completed_run["status"], "completed")

        metrics = completed_run["metrics"]
        self.assertIn("sample_size", metrics)
        self.assertIn("classes_count", metrics)
        self.assertEqual(metrics["classes_count"], 2)

    def test_08_multi_run_comparison_engine(self):
        """Verify side-by-side comparison across runs of the same module with parameter & metric diffs."""
        import asyncio

        exp = create_workbench_experiment(
            name="Comparison Training Exp",
            module="training",
            dataset_id=self.class_dataset_id,
        )

        run1 = create_experiment_run(
            experiment_id=exp["experiment_id"],
            run_name="Run 1 - Logistic Standard",
            parameters={"model_type": "logistic", "scaler": "standard", "target_column": "target"},
        )
        run2 = create_experiment_run(
            experiment_id=exp["experiment_id"],
            run_name="Run 2 - Logistic MinMax",
            parameters={"model_type": "logistic", "scaler": "minmax", "target_column": "target"},
        )

        asyncio.run(execute_module_run(run1["run_id"]))
        asyncio.run(execute_module_run(run2["run_id"]))

        comparison = compare_experiment_runs([run1["run_id"], run2["run_id"]])
        self.assertEqual(comparison["module"], "training")
        self.assertEqual(comparison["runs_count"], 2)
        self.assertIn("accuracy", [cm["metric"] for cm in comparison["comparison_metrics"]])
        self.assertIn("varying_parameters", comparison)
        self.assertIn("scaler", comparison["varying_parameters"])
        self.assertIn("chart_series", comparison)
        self.assertEqual(len(comparison["chart_series"]), 2)

    def test_09_multi_run_comparison_incompatible_module_guard(self):
        """Verify that comparing runs from mismatched modules raises HTTP 400."""
        exp_eda = create_workbench_experiment(name="EDA Exp", module="eda", dataset_id=self.class_dataset_id)
        exp_train = create_workbench_experiment(name="Train Exp", module="training", dataset_id=self.class_dataset_id)

        run_eda = create_experiment_run(experiment_id=exp_eda["experiment_id"], run_name="EDA")
        run_train = create_experiment_run(experiment_id=exp_train["experiment_id"], run_name="Train")

        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as cm:
            compare_experiment_runs([run_eda["run_id"], run_train["run_id"]])

        self.assertEqual(cm.exception.status_code, 400)
        self.assertIn("Incompatible run comparison", cm.exception.detail)

    def test_10_lineage_and_dependency_dag(self):
        """Verify lineage resolution from root dataset through parent runs to output artifacts."""
        exp = create_workbench_experiment(name="Lineage Exp", module="cleaning", dataset_id=self.class_dataset_id)
        run_parent = create_experiment_run(experiment_id=exp["experiment_id"], run_name="Clean Stage")
        run_child = create_experiment_run(
            experiment_id=exp["experiment_id"],
            run_name="Downstream Stage",
            parent_run_id=run_parent["run_id"],
        )

        lineage = get_run_lineage(run_child["run_id"])
        self.assertEqual(lineage["focus_run_id"], run_child["run_id"])

        node_ids = [n["id"] for n in lineage["nodes"]]
        self.assertIn(run_child["run_id"], node_ids)
        self.assertIn(run_parent["run_id"], node_ids)
        self.assertIn(self.class_dataset_id, node_ids)

        edge_targets = [e["to"] for e in lineage["edges"]]
        self.assertIn(run_child["run_id"], edge_targets)

    def test_11_reusable_workflow_orchestration(self):
        """Verify defining a multi-stage workflow and executing it sequentially."""
        import asyncio

        wf = create_workflow(
            name="Two Stage Clean and Profile Workflow",
            description="Run cleaning then EDA profiling on cleaned result",
            stages=[
                {"name": "Clean Step", "module": "cleaning", "parameters": {"impute_missing": True}},
                {"name": "EDA Step", "module": "eda", "parameters": {"target_column": "target"}},
            ],
        )
        self.assertTrue(wf["workflow_id"].startswith("wf_"))
        self.assertEqual(len(wf["stages"]), 2)

        # Execute workflow
        exec_res = asyncio.run(execute_workflow(workflow_id=wf["workflow_id"], dataset_id=self.class_dataset_id))
        self.assertEqual(exec_res["status"], "completed")
        self.assertEqual(len(exec_res["stage_runs"]), 2)
        self.assertEqual(exec_res["stage_runs"][0]["status"], "completed")
        self.assertEqual(exec_res["stage_runs"][1]["status"], "completed")

        # Verify get and list workflows
        wfs = list_workflows()
        self.assertTrue(any(w["workflow_id"] == wf["workflow_id"] for w in wfs["workflows"]))

        # Cleanup
        delete_workflow(wf["workflow_id"])

    def test_12_fastapi_rest_api_endpoints(self):
        """Verify REST API routes for datasets, workbench, runs, comparison, and legacy endpoints."""
        # Datasets list
        r_ds = self.client.get("/experiments/datasets")
        self.assertEqual(r_ds.status_code, 200)
        self.assertIn("datasets", r_ds.json())

        # Workbench create
        r_exp = self.client.post(
            "/experiments/workbench",
            json={
                "name": "API Test Experiment",
                "module": "training",
                "dataset_id": self.class_dataset_id,
            },
        )
        self.assertEqual(r_exp.status_code, 200)
        exp_id = r_exp.json()["experiment_id"]

        # Run create
        r_run = self.client.post(
            f"/experiments/workbench/{exp_id}/runs",
            json={
                "run_name": "API Run Test",
                "parameters": {"model_type": "logistic"},
            },
        )
        self.assertEqual(r_run.status_code, 200)
        run_id = r_run.json()["run_id"]

        # Run execute via API
        r_exec = self.client.post(f"/experiments/runs/{run_id}/execute")
        self.assertEqual(r_exec.status_code, 200)
        self.assertEqual(r_exec.json()["status"], "completed")

        # Lineage via API
        r_lineage = self.client.get(f"/experiments/runs/{run_id}/lineage")
        self.assertEqual(r_lineage.status_code, 200)
        self.assertIn("nodes", r_lineage.json())

        # Legacy backward compatibility endpoints
        r_legacy = self.client.get("/experiments")
        self.assertEqual(r_legacy.status_code, 200)
        self.assertIn("experiments", r_legacy.json())


if __name__ == "__main__":
    unittest.main()
