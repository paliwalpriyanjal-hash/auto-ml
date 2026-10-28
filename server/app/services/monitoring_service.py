# server/app/services/monitoring_service.py
"""
Lightweight operational telemetry and monitoring service for AutoVeyra.
Captures inference events, request outcomes, latency percentiles, error rates,
and basic input feature drift heuristics without logging private dataset rows.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from app.config import settings

SERVER_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = SERVER_DIR / "data"
MONITORING_DB_PATH = DATA_DIR / "monitoring.db"


@contextmanager
def _connect():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(MONITORING_DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def init_monitoring_db() -> None:
    """Initializes telemetry logging database tables and indexes."""
    with _connect() as conn:
        conn.execute(
            """
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
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_inf_model ON inference_logs(model_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_inf_timestamp ON inference_logs(timestamp)"
        )


def record_inference_event(
    model_id: int,
    model_name: str,
    task_type: str,
    latency_ms: float,
    row_count: int,
    status: str = "success",
    error_message: Optional[str] = None,
    correlation_id: str = "unknown",
    features_summary: Optional[Dict[str, Any]] = None,
) -> None:
    """Records inference outcome telemetry and enforces bounded log retention."""
    init_monitoring_db()
    now_iso = datetime.now(timezone.utc).isoformat()
    summary_json = json.dumps(features_summary or {})

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO inference_logs (
                timestamp, correlation_id, model_id, model_name, task_type,
                latency_ms, row_count, status, error_message, features_summary_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now_iso,
                correlation_id,
                model_id,
                model_name,
                task_type,
                round(float(latency_ms), 3),
                int(row_count),
                status,
                error_message,
                summary_json,
            ),
        )

        # Bounded retention pruning: keep newest MAX_MONITORING_RECORDS
        cursor = conn.execute("SELECT COUNT(*) FROM inference_logs")
        total = cursor.fetchone()[0]
        if total > settings.MAX_MONITORING_RECORDS + 500:
            excess = total - settings.MAX_MONITORING_RECORDS
            conn.execute(
                f"DELETE FROM inference_logs WHERE id IN (SELECT id FROM inference_logs ORDER BY id ASC LIMIT {excess})"
            )


def get_monitoring_stats() -> Dict[str, Any]:
    """Computes operational performance summary: volume, error rate, and latency percentiles."""
    init_monitoring_db()

    with _connect() as conn:
        total = conn.execute("SELECT COUNT(*) FROM inference_logs").fetchone()[0]
        if total == 0:
            return {
                "total_inferences": 0,
                "successful_inferences": 0,
                "failed_inferences": 0,
                "error_rate": 0.0,
                "avg_latency_ms": 0.0,
                "p50_latency_ms": 0.0,
                "p90_latency_ms": 0.0,
                "p99_latency_ms": 0.0,
                "requests_by_model": [],
                "serving_inventory": [],
            }

        success_count = conn.execute(
            "SELECT COUNT(*) FROM inference_logs WHERE status = 'success'"
        ).fetchone()[0]
        error_count = conn.execute(
            "SELECT COUNT(*) FROM inference_logs WHERE status = 'error'"
        ).fetchone()[0]

        # Latency percentiles
        cursor = conn.execute(
            "SELECT latency_ms FROM inference_logs WHERE status = 'success' ORDER BY latency_ms ASC"
        )
        latencies = [r[0] for r in cursor.fetchall()]

        if latencies:
            avg_lat = float(np.mean(latencies))
            p50 = float(np.percentile(latencies, 50))
            p90 = float(np.percentile(latencies, 90))
            p99 = float(np.percentile(latencies, 99))
        else:
            avg_lat = p50 = p90 = p99 = 0.0

        # Model breakdown
        m_cursor = conn.execute(
            """
            SELECT model_id, model_name, task_type,
                   COUNT(*) as request_count,
                   AVG(latency_ms) as avg_latency,
                   SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as errors
            FROM inference_logs
            GROUP BY model_id, model_name, task_type
            ORDER BY request_count DESC
            LIMIT 20
            """
        )
        requests_by_model = [
            {
                "model_id": r["model_id"],
                "model_name": r["model_name"],
                "task_type": r["task_type"],
                "request_count": r["request_count"],
                "avg_latency_ms": round(r["avg_latency"] or 0.0, 2),
                "error_count": r["errors"],
            }
            for r in m_cursor.fetchall()
        ]

    from app.services.serving_service import serving_cache
    inventory = serving_cache.get_inventory()

    return {
        "total_inferences": total,
        "successful_inferences": success_count,
        "failed_inferences": error_count,
        "error_rate": round(error_count / total, 4) if total > 0 else 0.0,
        "avg_latency_ms": round(avg_lat, 2),
        "p50_latency_ms": round(p50, 2),
        "p90_latency_ms": round(p90, 2),
        "p99_latency_ms": round(p99, 2),
        "requests_by_model": requests_by_model,
        "serving_inventory": inventory,
    }


def get_monitoring_logs(
    limit: int = 50,
    offset: int = 0,
    model_id: Optional[int] = None,
    status: Optional[str] = None,
) -> Dict[str, Any]:
    """Retrieves paginated operational inference event logs."""
    init_monitoring_db()
    limit = max(1, min(int(limit), 100))
    offset = max(0, int(offset))

    query = "SELECT id, timestamp, correlation_id, model_id, model_name, task_type, latency_ms, row_count, status, error_message FROM inference_logs WHERE 1=1"
    count_query = "SELECT COUNT(*) FROM inference_logs WHERE 1=1"
    params = []

    if model_id is not None:
        query += " AND model_id = ?"
        count_query += " AND model_id = ?"
        params.append(model_id)

    if status:
        query += " AND status = ?"
        count_query += " AND status = ?"
        params.append(status.strip().lower())

    query += " ORDER BY id DESC LIMIT ? OFFSET ?"

    with _connect() as conn:
        total = conn.execute(count_query, params).fetchone()[0]
        cursor = conn.execute(query, params + [limit, offset])
        items = [dict(r) for r in cursor.fetchall()]

    return {"items": items, "total": total, "limit": limit, "offset": offset}


def get_model_drift_indicator(model_id: int) -> Dict[str, Any]:
    """
    Computes statistical feature drift indicators comparing recent incoming inference batches.
    Note: Indicates input distribution shifts only; does not prove concept drift without true labels.
    """
    init_monitoring_db()

    with _connect() as conn:
        cursor = conn.execute(
            """
            SELECT features_summary_json FROM inference_logs
            WHERE model_id = ? AND status = 'success' AND features_summary_json IS NOT NULL
            ORDER BY id DESC LIMIT 25
            """,
            (model_id,),
        )
        rows = cursor.fetchall()

    if len(rows) < 3:
        return {
            "model_id": model_id,
            "sample_batches": len(rows),
            "status": "insufficient_data",
            "message": "Fewer than 3 inference batches recorded. Submit additional prediction requests to evaluate feature stability.",
            "feature_indicators": [],
        }

    # Aggregate numeric column means across recent batches
    feature_means: Dict[str, List[float]] = {}
    for r in rows:
        summary = json.loads(r[0] or "{}")
        for col, stats in summary.items():
            if "mean" in stats:
                feature_means.setdefault(col, []).append(stats["mean"])

    indicators = []
    has_warning = False

    for col, values in feature_means.items():
        if len(values) >= 3:
            std = float(np.std(values))
            mean = float(np.mean(values))
            cv = abs(std / (mean + 1e-6))  # Coefficient of variation
            status = "stable"
            if cv > 0.5:
                status = "shift_detected"
                has_warning = True

            indicators.append({
                "feature": col,
                "recent_mean": round(mean, 4),
                "std_deviation": round(std, 4),
                "variance_ratio": round(cv, 4),
                "status": status,
            })

    return {
        "model_id": model_id,
        "sample_batches": len(rows),
        "status": "shift_warning" if has_warning else "stable",
        "message": (
            "Potential covariate shift detected in one or more input features across recent batches."
            if has_warning
            else "Input feature distributions are statistically stable across recent prediction batches."
        ),
        "feature_indicators": indicators,
    }
