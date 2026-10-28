# AutoVeyra Phase 6 — Production Readiness, Security & Reliability Hardening Plan

**Date**: October 29, 2026  
**Status**: IN PROGRESS  
**Target Milestone**: Phase 6 — Complete Production Readiness & Hardening

---

## 1. Executive Summary & Pre-Flight Baseline Audit

Before making any modifications for Phase 6, a comprehensive audit of the baseline platform was conducted:

### Pre-Flight Verification Results
- **Backend Test Suite**: 94 tests across all modules (`test_srs_audit.py`, `test_classification_engine.py`, `test_comparison_engine.py`, `test_explainability.py`, `test_tuning_engine.py`, `test_feature_engineering.py`, `test_reports.py`).
  - Total Tests: **94**
  - Failures: **0**
  - Errors: **0**
  - Execution Time: **62.307s**
  - Exit Code: **0 (OK)**
- **Warnings Baseline**:
  - Pydantic protected namespace warnings: **0** (eliminated in Phase 5).
  - SQLite unclosed database ResourceWarnings: **0** (eliminated in Phase 5).
  - SciPy/scikit-learn deprecation warnings: 32 expected upstream warnings from `sklearn.linear_model._logistic.py`.
- **Frontend Production Build**: `npm run build --prefix client` succeeded in **15.90s** with 3364 modules transformed and **0 compilation errors**.

---

## 2. Hardening Objectives & Threat Model

Phase 6 hardens AutoVeyra against denial-of-service, unhandled exceptions, corrupted state, resource starvation, and unexpected input payloads.

### Key Risk Areas & Mitigations

1. **Unbounded File Uploads (Denial of Service / OOM)**:
   - *Risk*: A malicious or accidental upload of a multi-gigabyte file could crash the FastAPI worker process or exhaust memory during `pd.read_csv`.
   - *Control*: Introduce `safe_read_csv()` enforcing a strict byte-size ceiling (`MAX_UPLOAD_SIZE_BYTES`, default 25MB), maximum row count (`MAX_DATASET_ROWS`, default 100,000), and maximum column count (`MAX_DATASET_COLS`, default 500). Validate encoding with UTF-8 and Latin-1 fallbacks.

2. **Prediction Payload Flooding**:
   - *Risk*: Sending an unbounded JSON array (e.g. 100,000 rows) to `POST /models/{id}/predict` could stall the worker thread during pipeline `.predict()`.
   - *Control*: Enforce `MAX_PREDICTION_ROWS` (default 5,000 rows per batch). Reject oversized arrays with HTTP 413 (Payload Too Large).

3. **Joblib / Pickle Deserialization Trust Boundary**:
   - *Risk*: `joblib.load()` executes arbitrary Python code during unpickling if an attacker replaces or points to an arbitrary file.
   - *Control*: Ensure model paths are never taken from unverified user parameters. Restrict loads strictly to files located inside `server/data/models/` verifying `path.relative_to(MODEL_DIR)`. Verify file integrity, check for corrupted or empty artifact files, and raise clean HTTP 404/500 errors instead of unhandled crashes.

4. **Partial Registration & Data Integrity**:
   - *Risk*: If artifact serialization fails after database insertion (or vice versa), the system enters an inconsistent state with "ghost" models or orphan files.
   - *Control*: Two-phase atomic commit:
     1. Write artifact to a temporary file (`.model_{uuid}.tmp`).
     2. Insert metadata record in `registered_models` inside an active transaction.
     3. Atomically move `.tmp` file to final destination.
     4. If the database insert fails, immediately unlink both `.tmp` and final artifact.
     5. If file write fails, never execute the database insert.

5. **Resource Limits on Compute-Heavy Workloads**:
   - *Risk*: Setting `n_trials=1000` or `n_splits=50` in Optuna tuning or feature evaluation causes CPU exhaustion.
   - *Control*: Enforce bounded parameters: `n_trials <= MAX_TUNING_TRIALS` (default 100), `n_splits <= MAX_CV_SPLITS` (default 10), `samples <= MAX_EXPLAIN_SAMPLES` (default 500).

6. **Observability, Health & Correlation**:
   - *Risk*: Production monitoring lacks standard liveness/readiness probes, and error logs cannot be correlated to client requests.
   - *Control*:
     - Middleware attaching `X-Correlation-ID` to all requests and response headers.
     - `GET /health`: Shallow liveness probe.
     - `GET /ready`: Deep readiness probe verifying disk write access and SQLite database connectivity.
     - Structured logging of method, path, status, and duration without logging sensitive tabular payloads.

7. **Security Architecture Documentation**:
   - *Boundary*: AutoVeyra is a self-hosted single-tenant / local AutoML platform. Document clearly that it does not provide multi-tenant authentication/RBAC, and specify the production deployment prerequisites (reverse proxy, TLS, network isolation).

---

## 3. Implementation Step-by-Step

### Phase 6A: Central Configuration & Limits (`server/app/config.py`)
- Define `Settings` class with sensible defaults and environment variable overrides (`AUTOVEYRA_MAX_UPLOAD_MB`, `AUTOVEYRA_MAX_ROWS`, etc.).
- Expose global `settings` instance.

### Phase 6B: Input Validation & Safe Ingestion Utilities (`server/app/utils/security.py`)
- `safe_read_csv(file, max_bytes, max_rows, max_cols)`: Read in chunks, enforce size ceiling, handle encoding, validate dimensions.
- `validate_prediction_payload(rows, max_rows)`: Validate type, reject oversized batches.
- Integrate `safe_read_csv` across `data_routes.py`, `feature_routes.py`, `model_routes.py`, and services.

### Phase 6C: Model Registry Hardening & Recovery (`model_registry_service.py`)
- Atomic rollback on failed registration.
- Corrupted artifact detection and graceful error reporting.
- Path traversal protection verification.

### Phase 6D: Observability, Middleware & Health Probes (`main.py`)
- Correlation ID middleware (`X-Correlation-ID`).
- `GET /health` and `GET /ready`.
- Global exception handlers for standard JSON error schemas (`{"detail": "...", "correlation_id": "..."}`).

### Phase 6E: Automated Verification Suite (`server/tests/test_production_hardening.py`)
- 15+ test cases covering:
  - Oversized uploads (HTTP 413).
  - Malformed CSVs (HTTP 400).
  - Prediction batch limit enforcement.
  - Corrupted joblib recovery.
  - Ghost model prevention (rollback on failed artifact).
  - Health and readiness endpoints.
  - Correlation ID propagation.
  - Path traversal rejection.

### Phase 6F: Documentation & SRS Matrix Update
- `docs/PRODUCTION_READINESS.md`
- `docs/PHASE_6_TEST_REPORT.md`
- Update `docs/SRS_COMPLIANCE_MATRIX.md` with verified Phase 6 controls.
