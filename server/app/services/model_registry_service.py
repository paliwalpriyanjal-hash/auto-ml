from contextlib import contextmanager

# app/services/model_registry_service.py

import json
import math
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import joblib
import pandas as pd
from fastapi import HTTPException
from app.config import settings


SERVER_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = SERVER_DIR / "data"
MODEL_DIR = DATA_DIR / "models"
DB_PATH = DATA_DIR / "model_registry.db"


@contextmanager
def _connect():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(str(DB_PATH), timeout=10)
    connection.row_factory = sqlite3.Row
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def init_model_registry():
    with _connect() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS registered_models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                task_type TEXT NOT NULL,
                model_type TEXT NOT NULL,
                dataset_name TEXT,
                target_column TEXT,
                artifact_path TEXT NOT NULL,
                feature_columns_json TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)


def _json_safe(value: Any):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]

    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (ValueError, TypeError):
            pass

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if value is None or isinstance(value, (str, int, bool)):
        return value

    return str(value)


def register_model(
    *,
    pipeline: Any,
    model_name: str,
    task_type: str,
    model_type: str,
    dataset_name: Optional[str] = None,
    target_column: Optional[str] = None,
    feature_columns: Optional[list] = None,
    metrics: Optional[dict] = None,
) -> dict:
    """Persist a trusted, fitted sklearn pipeline and its metadata."""

    init_model_registry()

    if task_type not in {"regression", "classification"}:
        raise ValueError("task_type must be regression or classification")

    if not hasattr(pipeline, "predict"):
        raise ValueError("The supplied model must support predict().")

    model_key = uuid.uuid4().hex
    artifact_name = f"model_{model_key}.joblib"
    artifact_path = MODEL_DIR / artifact_name
    temporary_path = MODEL_DIR / f".{artifact_name}.tmp"

    metadata = {
        "model_name": model_name,
        "task_type": task_type,
        "model_type": model_type,
        "dataset_name": dataset_name,
        "target_column": target_column,
        "artifact_path": str(artifact_path.resolve()),
        "feature_columns_json": json.dumps(
            _json_safe(feature_columns or []),
            allow_nan=False,
        ),
        "metrics_json": json.dumps(
            _json_safe(metrics or {}),
            allow_nan=False,
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        # Write to a temporary file before replacing the final artifact.
        joblib.dump(pipeline, temporary_path)
        temporary_path.replace(artifact_path)

        with _connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO registered_models (
                    model_name, task_type, model_type, dataset_name,
                    target_column, artifact_path, feature_columns_json,
                    metrics_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    metadata["model_name"],
                    metadata["task_type"],
                    metadata["model_type"],
                    metadata["dataset_name"],
                    metadata["target_column"],
                    metadata["artifact_path"],
                    metadata["feature_columns_json"],
                    metadata["metrics_json"],
                    metadata["created_at"],
                ),
            )
            model_id = cursor.lastrowid

        return get_model(model_id)

    except Exception:
        temporary_path.unlink(missing_ok=True)
        artifact_path.unlink(missing_ok=True)
        raise


def _serialize(row: sqlite3.Row) -> dict:
    item = dict(row)
    item["feature_columns"] = json.loads(
        item.pop("feature_columns_json")
    )
    item["metrics"] = json.loads(item.pop("metrics_json"))
    # Do not expose the local artifact path in public API responses.
    item.pop("artifact_path", None)
    return item


def list_models(
    limit: int = 50,
    offset: int = 0,
    task_type: Optional[str] = None,
) -> dict:
    init_model_registry()
    limit = max(1, min(int(limit), 100))
    offset = max(0, int(offset))

    query = "SELECT * FROM registered_models"
    params = []

    if task_type:
        query += " WHERE task_type = ?"
        params.append(task_type.strip().lower())

    query += " ORDER BY id DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    count_query = "SELECT COUNT(*) FROM registered_models"
    count_params = []
    if task_type:
        count_query += " WHERE task_type = ?"
        count_params.append(task_type.strip().lower())

    with _connect() as connection:
        total = connection.execute(
            count_query, count_params
        ).fetchone()[0]
        rows = connection.execute(query, params).fetchall()

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "models": [_serialize(row) for row in rows],
    }


def get_model(model_id: int) -> dict:
    init_model_registry()

    with _connect() as connection:
        row = connection.execute(
            "SELECT * FROM registered_models WHERE id = ?",
            (model_id,),
        ).fetchone()

    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"Registered model {model_id} was not found.",
        )

    return _serialize(row)


def _get_model_record(model_id: int) -> sqlite3.Row:
    init_model_registry()

    with _connect() as connection:
        row = connection.execute(
            "SELECT * FROM registered_models WHERE id = ?",
            (model_id,),
        ).fetchone()

    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"Registered model {model_id} was not found.",
        )

    return row


def load_model(model_id: int):
    """Load a pipeline created by this application's trusted registry."""
    row = _get_model_record(model_id)
    artifact_path = Path(row["artifact_path"]).resolve()

    # Prevent metadata from pointing outside the model artifact directory.
    try:
        artifact_path.relative_to(MODEL_DIR.resolve())
    except ValueError:
        raise HTTPException(
            status_code=500,
            detail="Invalid model artifact path.",
        )

    if not artifact_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="The registered model artifact is missing.",
        )

    try:
        # Only load artifacts created and controlled by this application.
        pipeline = joblib.load(artifact_path)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not load model artifact: {exc}",
        )

    return pipeline, dict(row)


def predict_with_model(model_id: int, rows: list) -> dict:
    if not rows:
        raise HTTPException(
            status_code=400,
            detail="Provide at least one row for prediction.",
        )

    pipeline, metadata = load_model(model_id)
    expected_columns = json.loads(metadata["feature_columns_json"])

    if not isinstance(rows, list) or not all(
        isinstance(row, dict) for row in rows
    ):
        raise HTTPException(
            status_code=400,
            detail="rows must be a list of JSON objects.",
        )

    if len(rows) > settings.MAX_PREDICTION_ROWS:
        raise HTTPException(
            status_code=413,
            detail=f"Batch prediction exceeds maximum allowed limit of {settings.MAX_PREDICTION_ROWS:,} rows (received {len(rows):,} rows).",
        )

    input_df = pd.DataFrame(rows)

    if expected_columns:
        missing_columns = [
            column for column in expected_columns
            if column not in input_df.columns
        ]
        extra_columns = [
            column for column in input_df.columns
            if column not in expected_columns
        ]

        if missing_columns or extra_columns:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Input columns do not match the registered model.",
                    "missing_columns": missing_columns,
                    "extra_columns": extra_columns,
                    "expected_columns": expected_columns,
                },
            )

        input_df = input_df[expected_columns]

    try:
        predictions = pipeline.predict(input_df)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {exc}",
        )

    return {
        "model_id": model_id,
        "task_type": metadata["task_type"],
        "model_type": metadata["model_type"],
        "predictions": _json_safe(predictions.tolist()),
    }


def delete_model(model_id: int) -> dict:
    row = _get_model_record(model_id)
    artifact_path = Path(row["artifact_path"]).resolve()

    try:
        artifact_path.relative_to(MODEL_DIR.resolve())
    except ValueError:
        raise HTTPException(
            status_code=500,
            detail="Invalid model artifact path.",
        )

    with _connect() as connection:
        connection.execute(
            "DELETE FROM registered_models WHERE id = ?",
            (model_id,),
        )

    artifact_path.unlink(missing_ok=True)

    return {
        "message": "Registered model deleted successfully.",
        "id": model_id,
    }
