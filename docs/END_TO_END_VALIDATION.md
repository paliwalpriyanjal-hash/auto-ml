# AutoVeyra — End-to-End System Validation Report

## 1. Scope & Methodology

This document establishes the end-to-end validation results for **AutoVeyra 2.0-PRO (Release Candidate 1)**. Every layer—from raw CSV ingestion and data profiling to model serving, explainability, reports, operational telemetry, and containerized deployment—has been audited against verifiable automated tests and empirical executions.

---

## 2. Platform Tier Validation

### Tier 1: Ingestion, Profiling & Preprocessing
- **Validation Track**: Uploading CSV payloads with mixed data types, missing indicators (`NA`, `null`, `?`, `-`), and categorical features.
- **Components Verified**: `AdvancedDataCleaner`, `profile_dataset`, `safe_read_csv`.
- **Observed Behavior**:
  - Rejects files exceeding `AUTOVEYRA_MAX_UPLOAD_BYTES` (50MB) and files with non-CSV MIME types.
  - Correctly infers numeric vs categorical vs datetime columns.
  - Imputes missing numerical values via median/mean/KNN and categorical values via mode.
  - Drops target-level NaNs strictly without leaking test distributions into training sets.

### Tier 2: Model Training & Multi-Algorithm Comparison
- **Validation Track**: Automated cross-validation and holdout evaluation across classification and regression tasks.
- **Components Verified**: `train_service.py`, `classification_service.py`, `comparison_service.py`.
- **Observed Behavior**:
  - Regression: Trains and scores `LinearRegression`, `DecisionTreeRegressor`, `RandomForestRegressor`, `ExtraTreesRegressor`.
  - Classification: Trains and scores `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`, `ExtraTreesClassifier`.
  - Computes comprehensive metrics: $F_1$-Macro, ROC-AUC, Accuracy, Precision, Recall, Confusion Matrix, and Log Loss.
  - Comparison engine automatically crowns the top-performing pipeline by primary metric and registers the winning model artifact to disk.

### Tier 3: Hyperparameter Optimization (Optuna)
- **Validation Track**: Automated hyperparameter exploration using Bayesian Tree-structured Parzen Estimator (TPE).
- **Components Verified**: `tuning_service.py`, `tuning_routes.py`, `tuning_history.db`.
- **Observed Behavior**:
  - Bounded concurrency protects CPU/RAM from thrashing.
  - Each trial logs parameters, cross-validation score, and duration.
  - Computes delta metrics comparing baseline pipeline against the tuned configuration.
  - Supports trial session persistence, pagination, and session deletion.

### Tier 4: Model Explainability (XAI)
- **Validation Track**: Interpretability and contribution analysis on trained pipelines.
- **Components Verified**: `explainability_service.py`, `model_routes.py`.
- **Observed Behavior**:
  - **Permutation Feature Importance**: Shuffles features across out-of-sample evaluation rows and measures score degradation over repeated trials.
  - **TreeExplainer SHAP**: Computes global mean absolute SHAP values and row-level local waterfall attributions.
  - Efficiently aggregates transformed one-hot sub-features back into original dataset columns for clean human interpretability.

### Tier 5: Model Evaluation Reports & Governance Model Cards
- **Validation Track**: Generating immutable evaluation snapshots with standardized Model Cards and exporting to PDF, HTML, and JSON.
- **Components Verified**: `report_service.py`, `report_routes.py`, `reports.db`.
- **Observed Behavior**:
  - Captures dataset metadata, train/test splits, comprehensive metric matrices, and editable Model Card sections (intended use, out-of-scope use, risk assessment).
  - Export PDF: Generates valid ReportLab PDF documents with charts, tables, and branding.
  - Export HTML: Self-contained CSS-styled HTML reports.
  - Export JSON: Full structured machine-readable snapshot.

### Tier 6: High-Throughput Model Serving & LRU Cache
- **Validation Track**: Real-time batch and single-row inference via `/models/{id}/predict`.
- **Components Verified**: `serving_service.py`, `ModelServingCache`.
- **Observed Behavior**:
  - Enforces strict input schema validation: rejects payloads with missing or extra columns with HTTP 400.
  - First prediction request performs trusted deserialization (`cached: false`, ~15ms).
  - Subsequent prediction requests serve directly from process memory (`cached: true`, **< 1ms latency**).
  - Model deletion (`DELETE /models/{id}`) deterministically invalidates and evicts the model from cache memory.

### Tier 7: Telemetry, Latency Percentiles & Covariate Drift
- **Validation Track**: Operational performance tracking and distribution shift monitoring.
- **Components Verified**: `monitoring_service.py`, `monitoring_routes.py`, `monitoring.db`.
- **Observed Behavior**:
  - Telemetry is captured asynchronously without blocking client responses.
  - Evaluates operational $p50, p90, p99$ latency percentiles dynamically.
  - **Privacy Guarantee**: Zero raw user rows or feature vectors are saved to telemetry storage; only column-level summary statistics ($\mu, \min, \max, N_{\text{null}}$) are stored.
  - Detects covariate input shifts comparing incoming inference batches against baseline training statistics.

### Tier 8: Authentication, Authorization & Session Revocation
- **Validation Track**: User registration, login, role-based access, and token lifecycle.
- **Components Verified**: `auth_service.py`, `auth_routes.py`, `auth.db`.
- **Observed Behavior**:
  - Passwords hashed with salted `bcrypt`.
  - Signed HS256 JWT bearer tokens with configurable expiration claims.
  - Token revocation blocklist invalidates tokens upon `POST /auth/logout`.
  - Seamless environment toggle: frictionless local dev mode (`AUTH_ENABLED=False`) vs strict production security mode (`AUTH_ENABLED=True`).

### Tier 9: Reliability, Backups & Container Deployment
- **Validation Track**: Process crashes, hot SQLite snapshots, non-root Docker execution.
- **Components Verified**: `server/scripts/backup_databases.py`, `server/Dockerfile`, `docker-compose.yml`.
- **Observed Behavior**:
  - Online zero-downtime hot backups using Python's `sqlite3.Connection.backup()`.
  - Non-root user `autoveyra` (`UID 10001:GID 10001`) prevents privilege escalation.
  - Health and readiness probes (`/health`, `/ready`) support automated Kubernetes / Docker health restarts.
