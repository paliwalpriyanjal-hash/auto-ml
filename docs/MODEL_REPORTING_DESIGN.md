# AutoVeyra — Model Reporting & Model Card Architecture Design (Phase 5)

## 1. Architectural Overview & Design Philosophy

The Model Reporting & Model Cards subsystem provides enterprise-grade, reproducible, audit-ready evaluation reports for registered machine learning models in AutoVeyra.

### Core Principles
1. **Immutable Snapshot Guarantee**: When a report is generated, all measured holdout metrics, feature column definitions, dataset profiles, and baseline provenance are deep-copied into a versioned JSON document in `reports.db`. Deleting or modifying the registered model or pipeline artifact never corrupts or alters past audit reports.
2. **Standardized Model Cards**: Based on the seminal framework by Mitchell et al. (2019), model cards structure operational intent, out-of-scope/prohibited usage, known constraints, subgroup fairness/disparities, and auditor sign-off.
3. **Strict Separation of Measured vs. Narrative State**: Quantitative performance metrics (accuracy, F1 macro, ROC-AUC, RMSE, MAE, R²) and preprocessing parameters remain immutable. Only narrative governance fields (`intended_use`, `out_of_scope_use`, `known_limitations`, `ethical_considerations`, `author_notes`) can be updated via `PATCH /reports/{id}/model-card`.
4. **Deterministic Evidence Aggregation**: Reports automatically cross-reference Optuna hyperparameter optimization records (`tuning_history.db`), feature engineering strategies (`feature_history.db`), experiment runs (`experiments.db`), and explainability capabilities (`explainability_service`) without live coupling.
5. **Multi-Format Export**: Reports can be downloaded in three formats:
   - **PDF**: Generated on the fly using `reportlab`, with custom page geometry, branded headers/footers, dynamic page counts via `NumberedCanvas`, clean typography, and zero disk leakage.
   - **HTML**: Self-contained, printable, responsive HTML sanitized with `html.escape` to prevent XSS script injection.
   - **JSON**: Machine-readable, structured audit payload complying with Schema v1.0.0.

---

## 2. Data Schema & Persistence

### SQLite Database: `server/data/reports.db`

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
```

### Connection Lifecycle Management
All database access is managed through `@contextmanager` using `contextlib` and `try/finally` blocks, ensuring that every connection commits its transaction and calls `.close()`. This eliminates `ResourceWarning: unclosed database in <sqlite3.Connection>` completely.

---

## 3. API Contract Specifications

| Method | Endpoint | Description | Payload / Query |
| :--- | :--- | :--- | :--- |
| `POST` | `/reports/generate` | Generate snapshot report for a registered model | `{"model_id": int, "report_title": str, "model_card": dict}` |
| `GET` | `/reports` | List historical reports with pagination and filter | `?limit=50&offset=0&model_id=123` |
| `GET` | `/reports/{report_id}` | Retrieve full report document | None |
| `PATCH` | `/reports/{report_id}/model-card` | Update narrative fields of Model Card | `{"intended_use": str, "known_limitations": str, ...}` |
| `GET` | `/reports/{report_id}/download` | Export report artifact | `?format=pdf\|html\|json` |
| `DELETE` | `/reports/{report_id}` | Remove report snapshot without affecting model | None |

---

## 4. Frontend Workspace (`Reports.jsx`)

The frontend workspace provides:
- Seamless model selection from registered pipelines (`GET /models`).
- One-click snapshot generation with loading indicator and error recovery.
- Responsive history drawer listing all past audit records with timestamps and quick deletion.
- Tabular metric rendering with 4-decimal formatting and holdout confusion matrix.
- Feature column chip browser with real-time text filter.
- Inline modal for editing Model Card narrative metadata with live feedback.
- Instant download buttons for PDF, HTML, and JSON formats.
- Cross-workspace action shortcuts to `/predict`, `/explain`, `/tune`, and `/features`.
