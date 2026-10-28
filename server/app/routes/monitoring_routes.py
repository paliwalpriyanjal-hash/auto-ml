# server/app/routes/monitoring_routes.py
"""
Operational telemetry and model monitoring API routes.
"""

from typing import Optional
from fastapi import APIRouter, Query

from app.services.monitoring_service import (
    get_model_drift_indicator,
    get_monitoring_logs,
    get_monitoring_stats,
)
from app.services.serving_service import serving_cache

router = APIRouter(prefix="/monitoring", tags=["Operational Monitoring & Diagnostics"])


@router.get("/stats")
def monitoring_stats_endpoint():
    """Retrieve operational telemetry summary: prediction volumes, error rates, and latency percentiles."""
    return get_monitoring_stats()


@router.get("/logs")
def monitoring_logs_endpoint(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    model_id: Optional[int] = Query(default=None),
    status: Optional[str] = Query(default=None),
):
    """Retrieve paginated historical inference event records."""
    return get_monitoring_logs(
        limit=limit,
        offset=offset,
        model_id=model_id,
        status=status,
    )


@router.get("/models")
def serving_models_endpoint():
    """List actively served models and in-memory LRU cache status."""
    return {
        "cached_models": serving_cache.get_inventory(),
        "cache_capacity": serving_cache.maxsize,
    }


@router.get("/drift/{model_id}")
def model_drift_endpoint(model_id: int):
    """Evaluate statistical input feature drift across recent inference batches for a registered model."""
    return get_model_drift_indicator(model_id=model_id)
