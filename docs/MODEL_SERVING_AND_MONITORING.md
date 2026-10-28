# AutoVeyra — Production Model Serving & Monitoring Architecture

## 1. Executive Summary

Phase 7 introduces high-performance, resilient model serving and comprehensive operational telemetry to AutoVeyra. The architecture transitions model prediction from ad-hoc on-demand file loading to a thread-safe, bounded in-memory LRU model cache with sub-millisecond warm prediction throughput, non-blocking telemetry logging, and statistical feature drift detection without compromising privacy.

---

## 2. Model Serving Pipeline & LRU Cache Architecture

### Overview
In previous phases, inference required deserializing joblib artifacts from disk on every single HTTP POST request. For high-throughput workloads, disk I/O and unpickling created noticeable latency overhead.

Phase 7 implements `ModelServingCache` inside `server/app/services/serving_service.py`:

```
Client Inference Request
         │
         ▼
[POST /models/{id}/predict]
         │
         ▼
[Input Schema Validation] ──► Checks missing/extra columns, aligns ordering
         │
         ▼
  [Cache Hit?]
   ├── YES ──► Retrieve cached (pipeline, metadata) instantly (< 0.1ms)
   └── NO  ──► Thread-safe disk load from trusted registry, store in LRU cache
         │
         ▼
[Execute pipeline.predict()]
         │
         ▼
[Record Telemetry Event]  ──► Asynchronously log latency, count, summary stats
         │
         ▼
Return Response (predictions, probabilities, latency_ms, cached: bool)
```

### Key Technical Attributes
1. **Thread-Safe LRU Strategy**: Backed by `collections.OrderedDict` protected by a `threading.Lock`.
2. **Configurable Capacity**: Controlled via `AUTOVEYRA_MODEL_CACHE_SIZE` (default 8). When cache exceeds capacity, least-recently used models are safely evicted.
3. **Cache Invalidation & Eviction**: When a model is updated or deleted via `DELETE /models/{id}`, the entry is immediately evicted from memory via `serving_cache.evict(model_id)`.
4. **Latency Profiling**: Measures exact elapsed time using high-resolution monotonic timer `time.perf_counter()`.

---

## 3. Telemetry & Operational Metrics (`monitoring.db`)

AutoVeyra records telemetry in a dedicated SQLite database (`server/data/monitoring.db`), decoupled from experiment and model metadata.

### Schema Definition
```sql
CREATE TABLE IF NOT EXISTS inference_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    correlation_id TEXT NOT NULL,
    model_id INTEGER NOT NULL,
    model_name TEXT NOT NULL,
    task_type TEXT NOT NULL,
    latency_ms REAL NOT NULL,
    row_count INTEGER NOT NULL,
    status TEXT NOT NULL,
    error_message TEXT,
    features_summary_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_inf_model ON inference_logs(model_id);
CREATE INDEX IF NOT EXISTS idx_inf_timestamp ON inference_logs(timestamp);
```

### Metrics Computed
- **Total Requests**: Cumulative count of served inference requests.
- **Success & Failure Counts**: Segregated by HTTP outcome.
- **Error Rate**: Ratio of failed inferences to total volume.
- **Latency Percentiles**: Real-time `p50`, `p90`, and `p99` percentiles computed using `numpy.percentile()`.
- **Per-Model Utilization**: Distribution of requests, average latency, and errors per model.

---

## 4. Privacy-Preserving Input Feature Drift Detection

### Zero Raw Data Retention
To comply with GDPR, HIPAA, and enterprise data privacy policies, AutoVeyra **never logs raw input feature rows** into telemetry databases.

Instead, during served inference, `execute_served_inference` extracts lightweight aggregated statistical moments for each numeric feature:
- Column mean ($\mu$)
- Column minimum & maximum ($[\min, \max]$)
- Missing / null count ($N_{\text{null}}$)

### Covariate Shift Indicator
When analyzing a model via `GET /monitoring/drift/{model_id}`:
1. The engine aggregates recent inference batches (last 25 requests).
2. It compares the observed inference feature means against training baseline distribution statistics stored in registered model metadata.
3. If the relative shift $|\mu_{\text{inf}} - \mu_{\text{train}}| / (\sigma_{\text{train}} + \epsilon)$ exceeds threshold (default 25%), the feature is flagged as having input drift.

> [!NOTE]
> This indicates **covariate feature shift**. It does not constitute true concept drift, which requires ground truth labels that are unavailable at inference time.

---

## 5. Heavy Workload Concurrency Protection

Training, hyperparameter optimization, and SHAP explanations are CPU-intensive operations. In high-concurrency environments, unbounded concurrent jobs can exhaust system RAM and cause OOM crashes.

Phase 7 introduces `WorkloadConcurrencyManager` (`server/app/utils/workload_lock.py`):
- Bounded counting semaphore (`settings.MAX_CONCURRENT_WORKLOADS`, default 2).
- Non-blocking acquisition with HTTP 503 (`Heavy computation workload limit reached. Please retry in a few moments.`).
- Protects AutoML training, Optuna sweeps, and SHAP explainability runs.

---

## 6. Frontend Monitoring Dashboard

The AutoVeyra UI features a dedicated **Operational Monitoring Workspace** accessible at `/monitoring`:
- **Live Health & Status Indicator**: Real-time latency, request volume, and error rates.
- **Serving Inventory Grid**: Live inspection of models currently held in in-memory LRU cache.
- **Audit Log Feed**: Filterable table of past inference calls showing correlation IDs, latencies, and statuses.
- **Covariate Drift Inspector**: Interactive tool allowing operators to inspect statistical feature drift against training baselines.
