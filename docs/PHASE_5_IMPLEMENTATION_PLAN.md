# AutoVeyra Phase 5 Implementation Plan: Model Evaluation Reports, Model Cards & Exports

**Document Version:** 1.0  
**Phase:** Phase 5 — Production-Grade Model Evaluation Reports, Model Cards & Exports  
**Author:** ML Engineering & Systems Architecture  
**Status:** Approved & Ready for Implementation  

---

## 1. Baseline Audit & Existing Interfaces

AutoVeyra currently persists and serves modular machine learning assets across SQLite databases and filesystem joblib artifacts:
1. **Model Registry (`server/data/model_registry.db` + `models/*.joblib`):**
   - Stores `id`, `model_name`, `task_type`, `model_type`, `dataset_name`, `target_column`, `feature_columns_json`, `metrics_json`, `created_at`.
   - Reused via `model_registry_service.get_model(model_id)` and `load_model(model_id)`.
2. **Hyperparameter Tuning History (`server/data/tuning_history.db`):**
   - Tables: `tuning_runs` (keyed by `run_id` and indexed by `registered_model_id`), `tuning_trials`.
   - Contains: best trial number, best parameters, baseline vs. tuned holdout metrics, delta metrics, trial exploration history.
3. **Feature Engineering History (`server/data/feature_history.db`):**
   - Table: `feature_runs` (keyed by `run_id` and `registered_model_id`).
   - Contains: strategy comparison results (`baseline`, `feature_engineered`, `feature_selected`, `combined`), selected features, removed features, transformation configurations.
4. **Experiment Tracking (`server/data/experiments.db`):**
   - Table: `experiments`. Correlates training configurations, metrics, and parameters.
5. **Explainability Service (`server/app/services/explainability_service.py`):**
   - Provides capabilities (`/models/{id}/explainability/capabilities`) and permutation importance.

---

## 2. Gaps Identified

1. **No Centralized Reporting Snapshot:** Live models store scattered metrics in separate databases; there is no consolidated snapshot of evidence suitable for stakeholders, auditors, or academic portfolios.
2. **Missing Model Cards:** Current registered models do not record intended use, out-of-scope applications, ethical constraints, known failure modes, or provenance.
3. **No Multi-Format Document Export:** Neither PDF nor printable standalone HTML reports exist.
4. **Narrative vs. Measured Integrity:** Users need to annotate reports with business context and operational limitations without compromising or fabricating measured quantitative scores.

---

## 3. Planned Architecture & Schema

### 3.1. Database: `server/data/reports.db`
```sql
CREATE TABLE IF NOT EXISTS model_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id TEXT UNIQUE NOT NULL,
    model_id INTEGER NOT NULL,
    model_name TEXT NOT NULL,
    task_type TEXT NOT NULL,
    report_title TEXT NOT NULL,
    executive_summary_json TEXT NOT NULL,
    dataset_profile_json TEXT NOT NULL,
    evaluation_methodology_json TEXT NOT NULL,
    performance_metrics_json TEXT NOT NULL,
    model_card_json TEXT NOT NULL,
    feature_engineering_json TEXT NOT NULL,
    tuning_history_json TEXT NOT NULL,
    explainability_summary_json TEXT NOT NULL,
    provenance_json TEXT NOT NULL,
    limitations_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_reports_model_id ON model_reports(model_id);
CREATE INDEX IF NOT EXISTS idx_reports_created_at ON model_reports(created_at DESC);
```

### 3.2. Dedicated Services & Endpoints
- `server/app/services/report_service.py`:
  - `generate_model_report(model_id, options)`: Gathers evidence from registry, tuning, features, and experiments into an immutable snapshot.
  - `update_model_card(report_id, narrative_updates)`: Updates editable narrative fields (`intended_use`, `out_of_scope_use`, `known_limitations`, `ethical_considerations`, `author_notes`).
  - `render_html_report(report_id)`: Generates self-contained HTML document with inline CSS matching AutoVeyra design tokens; sanitized against XSS.
  - `render_pdf_report(report_id)`: Generates professional multi-page PDF via `reportlab` with AutoVeyra palette, metric tables, callout boxes, and running footers with page numbers.
  - `delete_report(report_id)`: Safely deletes report without affecting registered models or `.joblib` artifacts.
- `server/app/routes/report_routes.py`:
  - `POST /reports/generate`
  - `GET /reports` (paginated, filterable by `model_id`)
  - `GET /reports/{report_id}`
  - `PATCH /reports/{report_id}/model-card`
  - `GET /reports/{report_id}/download?format=pdf|html|json`
  - `DELETE /reports/{report_id}`

### 3.3. Frontend Workspace: `client/src/pages/Reports.jsx`
- Model selector with instant report generation.
- Full report preview with navigation sidebar (Executive Summary, Dataset Profile, Evaluation Methodology, Performance Metrics, Model Card, Feature Engineering, Tuning History, Explainability, Provenance, Limitations).
- Editable Model Card modal for narrative annotations.
- One-click downloads for PDF, HTML, and JSON.
- Report history drawer.
- Navigation links to Saved Models, Predictions, Explainability, Tuning, and Features.

---

## 4. Test Strategy

Create `server/tests/test_reports.py` with 15 test cases:
1. `test_01_generate_classification_report`
2. `test_02_generate_regression_report`
3. `test_03_report_with_tuning_and_features`
4. `test_04_missing_optional_metadata_graceful`
5. `test_05_model_card_editable_fields`
6. `test_06_snapshot_immutability_on_deleted_model`
7. `test_07_pdf_download_structure`
8. `test_08_html_download_and_xss_sanitization`
9. `test_09_json_export_schema`
10. `test_10_pagination_and_filtering`
11. `test_11_report_deletion_preserves_model`
12. `test_12_invalid_identifiers_return_404_or_400`
13. `test_13_conservative_claims_no_hyperbole`
14. `test_14_bounded_resource_large_feature_list`
15. `test_15_provenance_and_versioning`

Verify regression across all existing 80 tests (Phase 1–4) and run the frontend production build.
