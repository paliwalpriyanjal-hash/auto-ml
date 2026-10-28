# AutoVeyra — Production Readiness, Security & Reliability Guide

## 1. System Architecture & Threat Model

AutoVeyra is an Automated Machine Learning (AutoML) platform designed for structured tabular datasets.

### Trust Boundaries & Tenancy
- **Architecture**: Self-hosted, single-tenant tabular AutoML service.
- **Trust Boundary**: The platform assumes a trusted local developer or internal enterprise network environment.
- **Authentication & Authorization**: The application currently has **no user authentication (AuthN) or role-based access control (RBAC)**. All endpoints are open to network clients that can reach the FastAPI port.
- **Deployment Implication**: In production environments, AutoVeyra **must** be deployed behind a secure reverse proxy (e.g., NGINX, Cloudflare, Traefik, or AWS ALB) enforcing:
  1. TLS 1.3 encryption.
  2. Authentication (OIDC, OAuth2, mTLS, or Basic Auth).
  3. Network isolation / VPC firewalls preventing public Internet access to the raw API.

---

## 2. Hardened Security Controls

### A. Ingestion & CSV Upload Security
- **File Size Ceiling**: Enforced by `safe_read_csv` reading incoming streams in 64 KB chunks. Requests exceeding `MAX_UPLOAD_SIZE_BYTES` (default 25 MB) are terminated early with **HTTP 413 (Payload Too Large)** before buffering in memory.
- **Dimension Clamping**: Datasets exceeding `MAX_DATASET_ROWS` (default 100,000) or `MAX_DATASET_COLS` (default 500) are rejected with clear diagnostic HTTP 400 responses.
- **Encoding Resilience**: Dual-tier decoding (UTF-8 primary with Latin-1/ISO-8859-1 fallback) prevents server crashes on accented or legacy character sets.
- **Header Integrity**: Column headers are stripped, normalized, and validated for duplicates and blank names.

### B. Serialization & Deserialization Security (`joblib` / `pickle`)
- **No User-Supplied Deserialization**: The application **never** accepts serialized model files from users.
- **Path Traversal Isolation**: Model artifacts are stored strictly within `server/data/models/`. During retrieval and inference (`load_model`), artifact paths are resolved and asserted against `MODEL_DIR.resolve()`. Any path outside this root is rejected with HTTP 500/404.
- **Missing / Corrupted Artifact Recovery**: Missing `.joblib` files return HTTP 404; corrupted or truncated artifacts return clean HTTP 500 errors without terminating the FastAPI worker process.

### C. Inference & Batch Size Limits
- **Batch Size Ceiling**: `predict_with_model` enforces `MAX_PREDICTION_ROWS` (default 5,000 rows). Batches exceeding this limit are rejected with **HTTP 413**.
- **Strict Column Schema**: Inference strictly verifies that the input payload matches the registered feature contract (rejects missing columns with 400, rejects extra unexpected columns with 400).

### D. Report Generation & XSS Sanitization
- All user-editable narrative fields (`intended_use`, `out_of_scope_use`, `known_limitations`, etc.), report titles, model names, and feature column names are sanitized using `html.escape()` prior to HTML rendering.
- Content-Disposition attachment headers strip non-alphanumeric characters to prevent header injection.
- PDF generation uses memory-bounded flowables via ReportLab.

### E. SQLite Concurrency & Lifecycle
- Every database interaction uses `@contextmanager` with guaranteed transaction commit and `.close()` in a `finally:` block.
- Zero connection leaks or unclosed database ResourceWarnings.

---

## 3. Configuration & Resource Limits

All resource limits are centrally managed in `server/app/config.py` and can be customized via environment variables:

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `AUTOVEYRA_ENV` | `production` | Environment mode (`production` or `development`) |
| `AUTOVEYRA_MAX_UPLOAD_BYTES` | `26214400` (25 MB) | Maximum upload file size |
| `AUTOVEYRA_MAX_DATASET_ROWS` | `100000` | Maximum rows in uploaded CSV |
| `AUTOVEYRA_MAX_DATASET_COLS` | `500` | Maximum feature columns |
| `AUTOVEYRA_MAX_PREDICTION_ROWS` | `5000` | Maximum batch prediction rows per request |
| `AUTOVEYRA_MAX_TUNING_TRIALS` | `100` | Upper limit for Optuna optimization trials |
| `AUTOVEYRA_MAX_CV_SPLITS` | `10` | Upper limit for cross-validation folds |
| `AUTOVEYRA_MAX_EXPLAIN_SAMPLES` | `500` | Upper limit for SHAP / Permutation evaluations |
| `AUTOVEYRA_ALLOWED_ORIGINS` | `localhost:5173,...` | Comma-separated CORS allowed origins |

---

## 4. Operations, Observability & Health Probes

### Health Probes
- **Liveness Probe**: `GET /health`
  - Returns `{"status": "healthy", "app": "...", "version": "2.0-PRO", "environment": "production"}` with HTTP 200.
  - Used by Kubernetes or container orchestrators to detect process lockup.
- **Readiness Probe**: `GET /ready`
  - Performs live read/write checks against SQLite databases and filesystem artifact storage.
  - Returns `{"status": "ready", "checks": {"sqlite_databases": "healthy", "model_storage": "writable"}}` with HTTP 200.
  - Returns HTTP 503 if persistent storage or databases are unavailable.

### Request Correlation & Logging
- Every request is tagged with an `X-Correlation-ID` header (extracted from client or auto-generated as a UUID hex).
- Response headers include `X-Correlation-ID`.
- Error logs include `[CID=...]` tags without logging customer dataset rows.

---

## 5. Local Development & Production Run Instructions

### Backend Startup
```bash
# Set environment (optional)
export AUTOVEYRA_ENV=production
export AUTOVEYRA_MAX_UPLOAD_BYTES=26214400

# Start server using Uvicorn
cd server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

### Frontend Production Build & Preview
```bash
cd client
npm run build
npm run preview -- --port 5173
```
