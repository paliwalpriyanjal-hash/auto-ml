# AUTOVEYRA PHASE 4: INDEPENDENT MODULE-WISE EXPERIMENT WORKBENCH, EXPERIMENT TRACKING & WORKFLOW ORCHESTRATION

## 1. Executive Summary & Architectural Audit

The AutoVeyra platform has developed robust capabilities across:
- **Phase 1 ML Intelligence & Adaptive Training**: Model training, multi-model evaluation, and model registry (`train_service.py`, `classification_service.py`, `comparison_service.py`, `ml_intelligence_service.py`).
- **Phase 2 Adaptive EDA Intelligence**: Multi-profile exploratory data analysis, data quality audits, correlation matrices, and downloadable reports (`eda_service.py`, `eda_routes.py`).
- **Phase 3 Feature Engineering & Feature Selection**: Adaptive feature generation, multi-strategy candidate ranking, feature importances, and transformation pipelines (`feature_service.py`, `feature_routes.py`).
- **Phase 4 & Existing Experiment Tracking**: A single flat table `experiments` in `server/data/experiments.db` that recorded training and model comparison outcomes.

### Gap Analysis Against Phase 4 Requirements:
1. **Monolithic Coupling**: Prior to Phase 4, experiments were tightly coupled to full-model training or comparison. Users could not execute or track standalone Data Cleaning, standalone EDA profiling, standalone Feature Selection, or standalone Optuna Hyperparameter Tuning experiments without running the entire AutoML pipeline.
2. **Missing Entity Hierarchy**: ML systems require separation between:
   - **Dataset**: Immutable source data with cryptographic SHA-256 hash, dimensions, and column signatures.
   - **Experiment**: Logical hypothesis or goal (e.g., "Feature Selection for Churn" or "Tree Model Exploration").
   - **Run**: Execution instance with unique UUID, status lifecycle (`draft`, `queued`, `running`, `completed`, `failed`, `cancelled`), hyperparameters/configs, metrics, execution latency, and artifact lineage.
   - **Artifact**: Produced outputs (fitted models, transformed datasets, HTML/PDF reports, feature importance graphs) referencing parents.
   - **Workflow**: Composable, multi-stage DAG chaining module runs (e.g., Dataset -> Clean -> Engineer Features -> Train -> Evaluate).
3. **Run Comparison & Compatibility**: Comparing experiments required manual eyeballing. A structured comparison engine is needed to compare metrics side-by-side, enforce task/module compatibility guards, and calculate relative delta percentages.
4. **Lineage & Provenance**: Need a DAG dependency graph linking raw datasets -> intermediate cleaning/features -> trained model artifacts -> evaluation reports.

---

## 2. Multi-Entity Database Schema (`experiments.db`)

To ensure **100% backward compatibility**, the existing `experiments` table remains untouched. We introduce normalized tables in `server/data/experiments.db`:

### A. `datasets_registry`
- `dataset_id` (TEXT PRIMARY KEY) - UUID or SHA256 prefix
- `filename` (TEXT NOT NULL)
- `file_hash` (TEXT NOT NULL UNIQUE) - SHA-256 of file contents
- `file_size` (INTEGER NOT NULL)
- `row_count` (INTEGER NOT NULL)
- `column_count` (INTEGER NOT NULL)
- `schema_json` (TEXT NOT NULL) - Column names, types, missing counts
- `storage_path` (TEXT NOT NULL)
- `created_at` (TEXT NOT NULL)

### B. `workbench_experiments`
- `experiment_id` (TEXT PRIMARY KEY) - UUID
- `name` (TEXT NOT NULL)
- `module` (TEXT NOT NULL) - `eda`, `cleaning`, `features`, `training`, `tuning`, `evaluation`, `pipeline`
- `description` (TEXT)
- `dataset_id` (TEXT, FK -> datasets_registry)
- `tags_json` (TEXT)
- `created_at` (TEXT NOT NULL)
- `updated_at` (TEXT NOT NULL)

### C. `experiment_runs`
- `run_id` (TEXT PRIMARY KEY) - UUID
- `experiment_id` (TEXT NOT NULL, FK -> workbench_experiments)
- `run_name` (TEXT NOT NULL)
- `module` (TEXT NOT NULL)
- `status` (TEXT NOT NULL) - `draft`, `queued`, `running`, `completed`, `failed`, `cancelled`
- `parameters_json` (TEXT NOT NULL) - Preprocessing, model parameters, tuning trials, etc.
- `metrics_json` (TEXT NOT NULL) - Metric dictionary
- `artifacts_json` (TEXT NOT NULL) - Artifact references (file paths, hashes, sizes)
- `parent_run_id` (TEXT) - Run lineage parent
- `dataset_id` (TEXT NOT NULL)
- `duration_seconds` (REAL DEFAULT 0.0)
- `error_message` (TEXT)
- `created_at` (TEXT NOT NULL)
- `completed_at` (TEXT)

### D. `workflow_definitions`
- `workflow_id` (TEXT PRIMARY KEY) - UUID
- `name` (TEXT NOT NULL)
- `description` (TEXT)
- `stages_json` (TEXT NOT NULL) - Ordered list of module configurations
- `created_at` (TEXT NOT NULL)
- `updated_at` (TEXT NOT NULL)

### E. `workflow_executions`
- `execution_id` (TEXT PRIMARY KEY) - UUID
- `workflow_id` (TEXT NOT NULL, FK -> workflow_definitions)
- `status` (TEXT NOT NULL) - `running`, `completed`, `failed`, `cancelled`
- `stage_runs_json` (TEXT NOT NULL) - Map of stage_name -> run_id
- `duration_seconds` (REAL DEFAULT 0.0)
- `created_at` (TEXT NOT NULL)
- `completed_at` (TEXT)

---

## 3. Independent Module Execution Architecture

Each module can be executed in isolation via `execute_module_run()`:
1. **EDA Module (`eda`)**:
   - Takes: `dataset_id`, optional `target_column`.
   - Executes: Phase 2 `profile_dataset()` & `detect_eda_strategy()`.
   - Records: Quality metrics (missing cell %, duplicate rows, high correlations count), outputs summary metrics.
2. **Data Cleaning Module (`cleaning`)**:
   - Takes: `dataset_id`, cleaning configuration (`missing_strategy`, `outlier_method`, `duplicate_removal`, `scaling`).
   - Executes: `AdvancedDataCleaner`.
   - Generates: Cleaned dataset artifact in `server/data/cleaned_datasets/{run_id}.csv`, registers derivative dataset hash, records missing cells removed, outlier counts.
3. **Feature Engineering Module (`features`)**:
   - Takes: `dataset_id`, `target_column`, feature config (`strategy`, polynomial/interaction/selection flags).
   - Executes: `TabularFeatureEngineer`.
   - Generates: Feature dataset artifact, records feature count before/after, top important features, score improvements.
4. **Model Training Module (`training`)**:
   - Takes: `dataset_id` (raw or cleaned/features), `task_type`, `target_column`, `model_type`, hyperparameters.
   - Executes: `train_model_service()` or `compare_models_service()`.
   - Generates: Fitted model artifact (registered in Model Registry or run artifacts), holdout test metrics, confusion matrix / residuals.
5. **Hyperparameter Tuning Module (`tuning`)**:
   - Takes: `dataset_id`, `task_type`, `target_column`, `model_type`, `metric`, `n_trials`, `n_splits`.
   - Executes: Optuna study from `tuning_service.py`.
   - Records: Best trial params, baseline vs tuned score delta, trial history.
6. **Model Evaluation Module (`evaluation`)**:
   - Takes: `dataset_id`, `model_id` or `run_id`, evaluation metrics config.
   - Executes: Out-of-sample evaluation, calibration, fairness/bias checks, report generation via `report_service.py`.
   - Generates: Evaluation report snapshot artifact.

---

## 4. Multi-Run Comparison Engine & Lineage DAG

- **Comparison Rules**:
  - Enforces compatible module type (`module == module`).
  - Enforces identical target column & task type where applicable.
  - Generates parameter diffs (side-by-side highlighting changed hyperparameters).
  - Generates metric diffs with delta % relative to baseline run.
  - Highlights winning run per metric (e.g. higher accuracy/F1/R2, lower RMSE/MAE).
- **Lineage DAG**:
  - Traverses `parent_run_id` recursively upwards to raw dataset root.
  - Traverses child runs downwards to downstream models and evaluation reports.
  - Emits JSON graph format `{ nodes: [...], edges: [...] }` easily visualized by Cytoscape / Mermaid / React Flow / custom SVG.

---

## 5. Implementation Roadmap
1. `server/app/services/experiment_service.py`: Implement database schema migrations, dataset registration & hashing, workbench experiments, runs lifecycle, module executors, comparisons, lineage, and workflow orchestration.
2. `server/app/routes/experiment_routes.py`: Expose REST API endpoints with robust parameter validation.
3. `server/tests/test_experiments_workbench.py`: Comprehensive test suite verifying all module executions, comparisons, lineage, workflows, and edge cases.
4. `client/src/pages/Experiments.jsx`: Upgrade UI into a modern, 6-tab SaaS Experiment Workbench.
5. Verification: Run backend tests (`160+ tests passing`) and frontend build (`npm run build`).
