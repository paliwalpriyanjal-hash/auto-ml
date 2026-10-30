# AutoVeyra 2.0-PRO — Known System Limitations & Boundary Constraints

This document explicitly defines the operational boundaries, architectural design choices, and technical limitations of AutoVeyra 2.0-PRO.

---

## 1. Multi-Tenancy & Data Isolation Boundaries

> [!WARNING]
> **Single-Tenant Deployment Model**  
> AutoVeyra 2.0-PRO is architected as a **single-tenant** platform (suitable for dedicated organizational deployments, single research labs, or single team instances). While user registration, login authentication, and role checking exist in `auth.db`, the underlying data storage (`experiments.db`, `model_registry.db`, `reports.db`, and `server/data/models/`) is shared across all registered users on that instance. Row-level tenant isolation is **not** enforced. Do not deploy a single AutoVeyra instance across mutually untrusted organizations without separate isolated container instances.

---

## 2. Storage & Concurrency Architecture

1. **SQLite Database Concurrency**:
   - The platform uses embedded SQLite databases with Write-Ahead Logging (`PRAGMA journal_mode = WAL`) and 30-second busy timeout locks.
   - While SQLite WAL allows concurrent readers alongside a single writer, high-frequency simultaneous write operations (e.g., hundreds of concurrent training runs writing to `experiments.db`) may encounter file-lock contention (`database is locked`).
   - For this reason, the production container is intentionally configured with a single Uvicorn worker process (`--workers 1`).
2. **Local Filesystem Dependency**:
   - Model artifacts (`.joblib`), registered raw datasets, and generated reports are stored on the local persistent filesystem under `server/data/`.
   - Multi-node horizontally scaled deployments require shared network-attached storage (NFS/EFS) or an object store adapter (S3/GCS), which is not bundled in the default local distribution.

---

## 3. Dataset Scale & Ingestion Boundaries

AutoVeyra is tuned for typical tabular machine learning tasks on developer and student laptops:

| Metric | Configured Limit | Technical Reason |
|:---|:---:|:---|
| **Max File Upload** | 25 MB (`AUTOVEYRA_MAX_UPLOAD_BYTES`) | Prevents backend memory exhaustion during multipart HTTP streaming. |
| **Max Dataset Rows** | 100,000 rows (`AUTOVEYRA_MAX_DATASET_ROWS`) | Bound for in-memory pandas operations, cross-validation, and Optuna search on typical 8GB–16GB RAM machines. |
| **Max Feature Columns** | 500 columns (`AUTOVEYRA_MAX_DATASET_COLS`) | Prevents $O(N^2)$ explosion in correlation matrix and pairwise feature interaction generation. |
| **Max Inference Batch** | 5,000 rows (`AUTOVEYRA_MAX_PREDICTION_ROWS`) | Bounds real-time inference request latency. |

---

## 4. Computational & ML Algorithm Limitations

1. **Explainability Sampling**:
   - Kernel and Tree SHAP calculations are computationally expensive ($O(M \times 2^{|F|})$). To prevent browser timeouts and CPU lockups, background explainability runs are capped at a maximum background sample of 500 records (`AUTOVEYRA_MAX_EXPLAIN_SAMPLES`).
2. **Text Feature Extraction**:
   - Free-text processing is supported via lightweight statistical indicators (character count, word count, emptiness) and bounded TF-IDF (`max_features=15`). Large-scale NLP embeddings (BERT/transformers) are out of scope for this tabular AutoML release.
3. **Time-Series Horizon Forecasting**:
   - Feature engineering provides past-only lag and rolling window features for tabular time-series regression. Direct recursive autoregressive multi-step forecasting (e.g., ARIMA/Prophet) is not natively automated.
4. **Model Serving LRU Cache**:
   - In-memory serving pipelines are cached using an LRU policy with a default capacity of 8 active models (`AUTOVEYRA_MODEL_CACHE_SIZE`). When serving more than 8 distinct models concurrently, cache misses result in disk joblib re-reads (~5ms–15ms load latency).

---

## 5. Deployment Environment Dependencies

1. **Docker Daemon**:
   - Containerized deployment requires a running Docker engine. In environments where Docker is not available, AutoVeyra must be launched using standard Python (`uvicorn app.main:app`) and Node.js (`npm run build && npm run preview`) runtime processes.
2. **Browser Execution Context**:
   - LocalStorage is utilized for client-side JWT token storage. Running the client inside restricted third-party iframes with strict cross-origin cookie/storage partitioning may require same-origin reverse-proxy hosting.
