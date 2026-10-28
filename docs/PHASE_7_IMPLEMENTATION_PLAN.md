# AutoVeyra Phase 7 — Secure Deployment, Model Serving & Monitoring Plan

**Date**: October 29, 2026  
**Status**: IN PROGRESS  
**Target Milestone**: Phase 7 — Enterprise-Ready Deployment, Model Serving & Operational Monitoring

---

## 1. Executive Summary & Pre-Flight Baseline Audit

Before beginning Phase 7, a comprehensive baseline audit of the repository was completed:

### Baseline Test & Build Status
- **Backend Test Suite (All 8 Modules)**:
  - Total Tests: **109** across `test_srs_audit.py`, `test_classification_engine.py`, `test_comparison_engine.py`, `test_explainability.py`, `test_tuning_engine.py`, `test_feature_engineering.py`, `test_reports.py`, `test_production_hardening.py`.
  - Pass Count: **109 / 109** (100% pass rate).
  - Failures / Errors: **0**.
  - Warnings: **0** Pydantic protected namespace warnings, **0** unclosed SQLite connection ResourceWarnings.
- **Frontend Production Build**: `npm run build --prefix client` succeeded with **0 compilation errors** in 31.01s.

---

## 2. Phase 7 Architecture & Core Modules

Phase 7 elevates AutoVeyra from a hardened local tool to an enterprise-grade production ML platform with built-in authentication, high-throughput model serving, operational diagnostics, and reproducible deployment artifacts.

### A. Authentication & Authorization Subsystem (`server/app/services/auth_service.py`)
1. **Configurable Authentication Mode**:
   - Controlled via `AUTOVEYRA_AUTH_ENABLED` (defaults to `True` for production, `False` for local dev mode).
   - Secret key managed via `AUTOVEYRA_JWT_SECRET` (auto-generates a secure 32-byte secret if omitted).
2. **Password Security**:
   - Direct `bcrypt` with salt generation and key verification (`bcrypt.gensalt()` and `bcrypt.hashpw()`).
3. **Token Management**:
   - JWT tokens signed with `python-jose` (`HS256`) containing expiration (`exp`), user ID (`sub`), and role (`role`).
   - SQLite table `user_sessions` supporting active token tracking and instant revocation upon logout.
4. **Idempotent Data Ownership Migration**:
   - Adds `owner_id` / `owner_username` columns to `registered_models`, `experiments`, and `model_reports` without deleting or corrupting existing pre-Phase 7 models.
   - Pre-existing models default to system-accessible/admin ownership.

### B. High-Performance Model Serving (`server/app/services/serving_service.py`)
1. **In-Memory Thread-Safe LRU Model Cache**:
   - Bounded cache (default 8 models) holding pre-loaded, deserialized `sklearn.pipeline.Pipeline` objects.
   - Eliminates repeated disk I/O and joblib unpickling overhead for high-frequency prediction requests.
   - Instant cache eviction when a model is deleted via `DELETE /models/{id}`.
2. **Inference Latency & Outcome Metadata**:
   - Measures end-to-end inference latency in milliseconds.
   - Injects `X-Inference-Time-Ms` into HTTP response headers.
3. **Structured Inference Response**:
   - Returns prediction outputs alongside model version (`v1.0`), model ID, row count, timestamp, and request correlation ID.

### C. Operational Monitoring & Diagnostics (`server/app/services/monitoring_service.py`)
1. **Lightweight Monitoring Database (`server/data/monitoring.db`)**:
   - `inference_logs`: Records timestamp, correlation ID, model ID, latency ms, row count, status (`success`/`error`), error message, and lightweight feature distribution summary (means, min, max, null rates).
   - Never logs raw customer dataset records or PII.
2. **Aggregated Diagnostics API (`server/app/routes/monitoring_routes.py`)**:
   - `GET /monitoring/stats`: Request volume, error rate, p50/p90/p99 latency percentiles, requests by model.
   - `GET /monitoring/logs`: Paginated recent prediction logs with filter by model ID and status.
   - `GET /monitoring/models`: Active serving inventory, cache status, and average latency per model.
   - `GET /monitoring/drift/{model_id}`: Statistical drift indicator comparing recent incoming feature means against baseline training metadata.
3. **Bounded Storage Retention**:
   - Automated pruning keeping at most 50,000 log entries to prevent disk exhaustion.

### D. Production Deployment Architecture
1. **Containerization**:
   - `server/Dockerfile`: Multi-stage, non-root user execution, Uvicorn production server.
   - `docker-compose.yml`: Coordinates backend with persistent named volume for `server/data/`.
2. **Database Backup & Restoration**:
   - Automated script `server/scripts/backup_databases.py` creating consistent timestamped SQLite database snapshots.
3. **Concurrency & Workload Safety**:
   - `HeavyWorkloadLock`: Thread-safe execution lock preventing multiple massive Optuna tuning or multi-model comparisons from starving CPU cores concurrently.

### E. Frontend Operations & Monitoring Workspace (`client/src/pages/Monitoring.jsx`)
1. **Operational Monitoring Dashboard**:
   - Real-time stat cards (Total Inferences, Success Rate, Average Latency, P95 Latency, Active Cached Models).
   - Model Serving Inventory table with active cache indicators and per-model latency.
   - Recent Prediction Logs feed with status pills, error diagnostics, and request IDs.
   - Basic Feature Drift Inspector.
2. **Authentication Flow (`client/src/pages/Login.jsx`)**:
   - Login & Register modal/page with JWT token storage in `localStorage`.
   - Dynamic user profile and Logout action in Navbar.

---

## 3. Implementation Sequence

1. **Step 1**: Implement `auth_service.py` and `auth_routes.py` with SQLite user storage, bcrypt password hashing, and JWT handling.
2. **Step 2**: Implement `serving_service.py` with in-memory LRU model caching and version metadata.
3. **Step 3**: Implement `monitoring_service.py` and `monitoring_routes.py` with inference logging, telemetry aggregations, and feature drift heuristics.
4. **Step 4**: Integrate serving and monitoring into `model_routes.py` and `main.py`.
5. **Step 5**: Create Docker deployment configuration (`Dockerfile`, `docker-compose.yml`) and backup script (`backup_databases.py`).
6. **Step 6**: Build frontend `Monitoring.jsx` and `Login.jsx`, update navigation and routing.
7. **Step 7**: Write targeted tests (`test_auth_and_security.py`, `test_model_serving_and_monitoring.py`).
8. **Step 8**: Run full platform regression tests (all modules) and frontend production build.
9. **Step 9**: Deliver documentation (`DEPLOYMENT_GUIDE.md`, `MODEL_SERVING_AND_MONITORING.md`, `PHASE_7_TEST_REPORT.md`, updated `SRS_COMPLIANCE_MATRIX.md`).
