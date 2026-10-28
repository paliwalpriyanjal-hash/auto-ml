# server/app/services/serving_service.py
"""
High-performance Model Serving engine for AutoVeyra.
Implements in-memory thread-safe LRU pipeline caching, model versioning,
batch validation, inference latency profiling, and outcome logging.
"""

import collections
import json
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import HTTPException

from app.config import settings
from app.services.model_registry_service import load_model, _get_model_record
from app.utils.security import validate_prediction_payload


class ModelServingCache:
    """Thread-safe bounded LRU cache for deserialized model pipelines."""

    def __init__(self, maxsize: int = 8):
        self.maxsize = maxsize
        self._cache: collections.OrderedDict[int, Tuple[Any, Dict[str, Any], float]] = collections.OrderedDict()
        self._lock = threading.Lock()

    def get(self, model_id: int) -> Tuple[Optional[Tuple[Any, Dict[str, Any]]], bool]:
        """Returns (pipeline, metadata), was_cached flag."""
        with self._lock:
            if model_id in self._cache:
                self._cache.move_to_end(model_id)
                pipeline, metadata, _ = self._cache[model_id]
                return (pipeline, metadata), True

        # Load from disk outside lock to prevent blocking concurrent served queries
        pipeline, metadata = load_model(model_id)

        with self._lock:
            self._cache[model_id] = (pipeline, metadata, time.time())
            self._cache.move_to_end(model_id)
            if len(self._cache) > self.maxsize:
                self._cache.popitem(last=False)

        return (pipeline, metadata), False

    def evict(self, model_id: int) -> bool:
        """Evicts a model pipeline from memory upon deletion or update."""
        with self._lock:
            if model_id in self._cache:
                del self._cache[model_id]
                return True
        return False

    def clear(self) -> None:
        """Evicts all models from cache."""
        with self._lock:
            self._cache.clear()

    def get_inventory(self) -> List[Dict[str, Any]]:
        """Returns currently cached models and their load timestamps."""
        with self._lock:
            items = []
            for m_id, (_, meta, loaded_at) in self._cache.items():
                items.append({
                    "model_id": m_id,
                    "model_name": meta.get("model_name"),
                    "task_type": meta.get("task_type"),
                    "version": meta.get("version", "v1.0"),
                    "loaded_at_epoch": loaded_at,
                    "serving_status": "in_memory",
                })
            return items


serving_cache = ModelServingCache(maxsize=settings.MODEL_CACHE_SIZE)


def execute_served_inference(
    model_id: int,
    rows: Any,
    correlation_id: str = "unknown",
) -> Dict[str, Any]:
    """
    Executes high-throughput batch inference with input validation, cached model loading,
    latency tracking, and asynchronous/bounded telemetry capture.
    """
    start_time = time.perf_counter()
    rows = validate_prediction_payload(rows)

    # 1. Fetch pre-loaded pipeline or load from trusted registry
    (pipeline, metadata), was_cached = serving_cache.get(model_id)
    expected_columns = json.loads(metadata.get("feature_columns_json") or "[]")

    # 2. Schema Validation
    input_df = pd.DataFrame(rows)
    if expected_columns:
        missing_columns = [col for col in expected_columns if col not in input_df.columns]
        extra_columns = [col for col in input_df.columns if col not in expected_columns]

        if missing_columns or extra_columns:
            error_details = []
            if missing_columns:
                error_details.append(f"missing_columns: {missing_columns}")
            if extra_columns:
                error_details.append(f"extra_columns: {extra_columns}")
            raise HTTPException(
                status_code=400,
                detail=f"Feature contract mismatch: {'; '.join(error_details)}.",
            )

        # Align columns to exact expected training order
        input_df = input_df[expected_columns]

    # 3. Execute Model Predictions
    try:
        raw_predictions = pipeline.predict(input_df)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Inference execution failed on pipeline: {exc}",
        )

    # Format predictions cleanly
    if hasattr(raw_predictions, "tolist"):
        predictions = raw_predictions.tolist()
    else:
        predictions = list(raw_predictions)

    # Compute classification probabilities if available
    probabilities = None
    if hasattr(pipeline, "predict_proba"):
        try:
            raw_proba = pipeline.predict_proba(input_df)
            if hasattr(raw_proba, "tolist"):
                probabilities = raw_proba.tolist()
            else:
                probabilities = list(raw_proba)
        except Exception:
            probabilities = None

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    # 4. Record Telemetry Event (without logging raw row payloads!)
    try:
        from app.services.monitoring_service import record_inference_event
        # Compute lightweight column summary stats for drift detection
        summary_stats = {}
        for col in input_df.columns:
            if pd.api.types.is_numeric_dtype(input_df[col]):
                summary_stats[col] = {
                    "mean": float(input_df[col].mean()),
                    "min": float(input_df[col].min()),
                    "max": float(input_df[col].max()),
                    "null_count": int(input_df[col].isna().sum()),
                }
            else:
                summary_stats[col] = {
                    "null_count": int(input_df[col].isna().sum()),
                }

        record_inference_event(
            model_id=model_id,
            model_name=metadata.get("model_name", "Unknown"),
            task_type=metadata.get("task_type", "unknown"),
            latency_ms=elapsed_ms,
            row_count=len(rows),
            status="success",
            correlation_id=correlation_id,
            features_summary=summary_stats,
        )
    except Exception:
        pass  # Telemetry failure must never fail user inference

    return {
        "model_id": model_id,
        "model_name": metadata.get("model_name"),
        "task_type": metadata.get("task_type"),
        "version": metadata.get("version", "v1.0"),
        "serving_status": "active",
        "cached": was_cached,
        "latency_ms": round(elapsed_ms, 2),
        "predictions_count": len(predictions),
        "predictions": predictions,
        "probabilities": probabilities,
        "correlation_id": correlation_id,
    }
