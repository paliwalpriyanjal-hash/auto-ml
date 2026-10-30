# AUTOVEYRA PHASE 5: UNIFIED AI/ML PROJECT REPORT STUDIO, COMPLETE TECHNICAL DOCUMENTATION & ORIGINAL DATASET EXPORT

## 1. Executive Summary & Architectural Audit

The AutoVeyra platform has developed comprehensive, modular capabilities:
- **Phase 1 ML Intelligence & Adaptive Training**: Preprocessing pipelines, classification/regression models, holdout evaluation, metrics calculation.
- **Phase 2 Adaptive EDA Intelligence**: Dataset profiling, statistical audits, correlation matrices, high-resolution visualizations, HTML/PDF EDA report generation (`eda_service.py`).
- **Phase 3 Feature Engineering & Feature Selection**: Adaptive feature generation (polynomial, interaction, binning, text, rare categories), multi-strategy ranking, feature lineage (`feature_service.py`).
- **Phase 4 Independent Module Experiment Workbench & Tracking**: Normalized entities (`datasets_registry`, `workbench_experiments`, `experiment_runs`, `workflow_definitions`, `workflow_executions`), SHA-256 dataset immutability, run lifecycle, multi-run comparison matrix, and lineage DAG (`experiment_service.py`).
- **Phase 5 Existing Model Reporting**: Model card generation and holdout evaluation snapshots for individual registered models (`report_service.py`).

### Current Gap Analysis for Phase 5:
1. **Single-Model vs Full-Project Coupling**: Existing `model_reports` in `report_service.py` only summarize a single registered model ID. There is no unified mechanism to compile an entire end-to-end data science project spanning raw dataset profiling, cleaning runs, feature engineering, candidate model comparison, hyperparameter optimization, explainability, and workflow lineage into one coherent report.
2. **Missing Dataset Preservation & Packaging**: No export mechanism packages the immutable original uploaded CSV (verified via SHA-256) together with the comprehensive report (PDF, HTML, JSON), visual charts, workflow flowchart, and metadata in a structured ZIP bundle.
3. **Template Diversity**: Users need tailored documentation templates (Academic/University thesis, Professional Data Science report, Technical ML engineering report, and Complete AutoVeyra Project Report).
4. **Compatibility Validation**: Need rigorous validation before compilation to verify that selected runs share the same dataset/lineage, consistent target column, and task type.

---

## 2. Unified Project Report Architecture

### Database Schema Extension (`reports.db`)
In addition to the existing `model_reports` table, we add the `project_reports` table:

```sql
CREATE TABLE IF NOT EXISTS project_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id TEXT UNIQUE NOT NULL,
    project_name TEXT NOT NULL,
    project_description TEXT,
    author_name TEXT,
    institution TEXT,
    report_template TEXT NOT NULL,      -- 'academic', 'professional', 'technical', 'complete'
    original_dataset_id TEXT NOT NULL,
    original_filename TEXT NOT NULL,
    dataset_sha256 TEXT NOT NULL,
    dataset_dimensions TEXT NOT NULL,
    target_column TEXT,
    task_type TEXT,
    selected_workflow_id TEXT,
    selected_workflow_execution_id TEXT,
    selected_runs_json TEXT NOT NULL,             -- Map of module -> run_id
    compatibility_validation_json TEXT NOT NULL,  -- Validation results & status
    included_sections_json TEXT NOT NULL,         -- Active sections list
    bundle_options_json TEXT NOT NULL,            -- Export toggles
    report_data_json TEXT NOT NULL,               -- Full 15-section document tree
    visualizations_json TEXT NOT NULL,            -- Base64 / paths of rendered charts
    flowchart_json TEXT NOT NULL,                 -- Workflow DAG representation
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_proj_reports_created_at ON project_reports(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_proj_reports_dataset_id ON project_reports(original_dataset_id);
```

---

## 3. Original Dataset Preservation & Integrity Controls

1. **Immutability Guarantee**: The original dataset in `server/data/datasets_store/{file_hash}.csv` is never overwritten, modified, or truncated.
2. **Pre-Packaging Verification**: Before ZIP bundling, the file's SHA-256 checksum is recalculated and verified against `datasets_registry.file_hash`.
3. **Derivative Separation**: Original raw dataset is stored in `Dataset/original_dataset.csv`. If selected, cleaned dataset is stored separately in `Dataset/cleaned_dataset.csv`.
4. **Opt-in Privacy Toggles**:
   - `include_original_dataset` (default: true, opt-out available).
   - `include_cleaned_dataset` (default: false, opt-in).
   - `include_model_artifact` (default: false, opt-in with size note).
5. **PDF Presentation**: The PDF includes exact metadata (filename, dimensions, SHA-256 hash, size, target column, feature types) rather than inlining thousands of rows into the PDF.

---

## 4. Intelligent Project Report Builder & Compatibility Engine

### 9-Step Guided Creation Workflow:
1. **Select Dataset**: Pick from registered datasets library or upload new CSV.
2. **Select Scope**:
   - Option A: Select a completed Reusable Workflow execution (auto-resolves all stage runs).
   - Option B: Independently pick runs from EDA, Cleaning, Features, Training, Tuning, Evaluation, Explainability.
3. **Validate Compatibility**: Check dataset identity, derived lineage, target column, and task type.
4. **Select Report Sections**: Choose from standard 15 sections.
5. **Choose Template**: Academic, Professional, Technical, or Complete AutoVeyra.
6. **Configure Inclusions**: Toggle original dataset, cleaned CSV, and model file.
7. **Preview Structure**: Review section list and metadata.
8. **Compile & Generate**: Assemble multi-module JSON document, render charts, build PDF/HTML.
9. **Export & Download**: Download PDF, HTML, JSON, or complete ZIP project bundle with `MANIFEST.json`.

---

## 5. Report Content Structure (15 Standard Sections)

1. **Cover Page**: Title, subtitle, branding, author, institution, dataset, date.
2. **Executive Summary**: Objectives, primary metrics, winning model, key limitations.
3. **Project Overview**: Problem statement, prediction task, technology stack.
4. **Original Dataset Information**: Filename, ID, SHA-256 hash, dimensions, schema.
5. **Exploratory Data Analysis**: Profiling, distributions, missingness, correlation heatmap.
6. **Data Cleaning & Preprocessing**: Transformation decisions, rows retained, imputation.
7. **Feature Engineering**: Original vs generated vs selected features, transformations.
8. **Model Training**: Candidate algorithms, train/holdout splits, latency.
9. **Model Comparison**: Metric matrix, rankings, winning model selection reasoning.
10. **Hyperparameter Tuning**: Optuna configuration, trial count, best parameters, score delta.
11. **Final Model Evaluation**: Accuracy, F1, Precision, Recall, Confusion Matrix (or R², RMSE, MAE, residuals).
12. **Model Explainability**: Global feature importances, SHAP/permutation summary.
13. **Complete Workflow Flowchart**: Dynamic lineage DAG graphic.
14. **Experiment History & Reproducibility**: Run IDs, configs, seeds, reproduction commands.
15. **Conclusion & Limitations**: Honest summary of achievements and technical boundaries.

---

## 6. Implementation Roadmap
1. `server/app/services/project_report_service.py`: Project report assembly, compatibility checking, chart rendering via Matplotlib, ReportLab PDF generation, HTML generation, and ZIP packaging with `MANIFEST.json`.
2. `server/app/routes/report_routes.py`: Extended REST endpoints (`/reports/project/...`) maintaining backward compatibility with `/reports` single-model endpoints.
3. `server/tests/test_project_report_studio.py`: Unit and integration test suite validating compatibility checks, project report creation, PDF/HTML generation, SHA-256 verification, and ZIP packaging.
4. `client/src/pages/Reports.jsx`: Upgraded frontend studio with unified navigation between the Project Report Studio and Model Governance Cards.
5. Verification: Run full backend test suite (`python -m unittest discover`) and frontend build (`npm run build`).
