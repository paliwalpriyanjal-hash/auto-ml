from contextlib import contextmanager

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import HTTPException


# Keep the database inside the server folder.
SERVER_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = SERVER_DIR / "data"
DB_PATH = DATA_DIR / "experiments.db"


@contextmanager
def _connect():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(
        str(DB_PATH),
        timeout=10,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def _json_safe(value: Any) -> Any:
    """Convert nested data to JSON-safe Python values."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (ValueError, TypeError):
            pass

    if isinstance(value, float):
        import math
        return value if math.isfinite(value) else None

    if value is None or isinstance(value, (str, int, bool)):
        return value

    return str(value)


def _dump(value: Any) -> str:
    return json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        allow_nan=False,
    )


def init_experiment_db() -> None:
    """Create the experiment table if it does not exist."""
    with _connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_name TEXT NOT NULL,
                task_type TEXT NOT NULL,
                dataset_name TEXT,
                target_column TEXT,
                model_type TEXT NOT NULL,
                scaler_type TEXT,
                missing_strategy TEXT,
                encoding_strategy TEXT,
                split_ratio REAL,
                metrics_json TEXT NOT NULL,
                result_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_experiments_created_at
            ON experiments(created_at DESC)
            """
        )


def create_experiment(
    *,
    experiment_name: Optional[str],
    task_type: str,
    dataset_name: Optional[str],
    target_column: Optional[str],
    model_type: str,
    scaler_type: Optional[str],
    missing_strategy: Optional[str],
    encoding_strategy: Optional[str],
    split_ratio: Optional[float],
    metrics: dict,
    result: dict,
) -> dict:
    """Save a successful training/comparison result."""
    init_experiment_db()

    safe_name = (experiment_name or "").strip()
    if not safe_name:
        safe_name = (
            f"{model_type} - "
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
        )

    created_at = datetime.now(timezone.utc).isoformat()

    with _connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO experiments (
                experiment_name, task_type, dataset_name,
                target_column, model_type, scaler_type,
                missing_strategy, encoding_strategy, split_ratio,
                metrics_json, result_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                safe_name,
                task_type,
                dataset_name,
                target_column,
                model_type,
                scaler_type,
                missing_strategy,
                encoding_strategy,
                split_ratio,
                _dump(metrics),
                _dump(result),
                created_at,
            ),
        )

        experiment_id = cursor.lastrowid

    return get_experiment(experiment_id)


def _serialize_row(row: sqlite3.Row, include_result: bool = False) -> dict:
    item = dict(row)
    item["metrics"] = json.loads(item.pop("metrics_json"))
    result_json = item.pop("result_json", None)

    if include_result and result_json is not None:
        item["result"] = json.loads(result_json)

    return item


def list_experiments(
    limit: int = 50,
    offset: int = 0,
    task_type: Optional[str] = None,
) -> dict:
    init_experiment_db()

    limit = max(1, min(int(limit), 100))
    offset = max(0, int(offset))

    query = """
        SELECT id, experiment_name, task_type, dataset_name,
               target_column, model_type, scaler_type,
               missing_strategy, encoding_strategy, split_ratio,
               metrics_json, created_at
        FROM experiments
    """
    params = []

    if task_type:
        query += " WHERE task_type = ?"
        params.append(task_type.strip().lower())

    query += " ORDER BY id DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    count_query = "SELECT COUNT(*) FROM experiments"
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
        "experiments": [
            _serialize_row(row) for row in rows
        ],
    }


def get_experiment(experiment_id: int) -> dict:
    init_experiment_db()

    with _connect() as connection:
        row = connection.execute(
            "SELECT * FROM experiments WHERE id = ?",
            (experiment_id,),
        ).fetchone()

    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"Experiment {experiment_id} was not found.",
        )

    return _serialize_row(row, include_result=True)


def delete_experiment(experiment_id: int) -> dict:
    init_experiment_db()

    with _connect() as connection:
        cursor = connection.execute(
            "DELETE FROM experiments WHERE id = ?",
            (experiment_id,),
        )

        deleted = cursor.rowcount

    if deleted == 0:
        raise HTTPException(
            status_code=404,
            detail=f"Experiment {experiment_id} was not found.",
        )

    return {
        "message": "Experiment deleted successfully.",
        "id": experiment_id,
    }
