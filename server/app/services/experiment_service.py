# server/app/services/experiment_service.py
"""
AutoVeyra Phase 4: Independent Module-Wise Experiment Workbench,
Experiment Tracking & Workflow Orchestration Service.
"""

from contextlib import contextmanager
import hashlib
import io
import json
import os
import re
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from fastapi import HTTPException, UploadFile
import numpy as np
import pandas as pd

# Keep database and storage inside the server folder.
SERVER_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = SERVER_DIR / "data"
DB_PATH = DATA_DIR / "experiments.db"
DATASETS_STORE_DIR = DATA_DIR / "datasets_store"
CLEANED_DATASETS_DIR = DATA_DIR / "cleaned_datasets"
FEATURE_DATASETS_DIR = DATA_DIR / "feature_datasets"

# Allowed modules for independent experiments
VALID_MODULES = {"eda", "cleaning", "features", "training", "tuning", "evaluation", "pipeline"}
RUN_STATUSES = {"draft", "queued", "running", "completed", "failed", "cancelled"}

# In-memory set of cancelled run IDs
_CANCELLED_WORKBENCH_RUNS = set()


@contextmanager
def _connect():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DATASETS_STORE_DIR.mkdir(parents=True, exist_ok=True)
    CLEANED_DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    FEATURE_DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(
        str(DB_PATH),
        timeout=20,
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

    if isinstance(value, (np.integer, np.int64, np.int32)):
        return int(value)

    if isinstance(value, (np.floating, np.float64, np.float32)):
        import math
        f_val = float(value)
        return f_val if math.isfinite(f_val) else None

    if isinstance(value, np.ndarray):
        return [_json_safe(x) for x in value.tolist()]

    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()

    if value is None or isinstance(value, (str, int, bool)):
        return value

    return str(value)


def _dump(value: Any) -> str:
    return json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        allow_nan=False,
    )


# ----------------------------------------------------------------------
# In-Memory UploadFile Wrapper for Service Interop
# ----------------------------------------------------------------------

class VirtualUploadFile:
    """FastAPI UploadFile compatible object created from bytes or disk file."""
    def __init__(self, filename: str, content: bytes):
        self.filename = filename
        self._content = content
        self._file = io.BytesIO(content)
        self.file = self._file

    async def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            return self._file.read()
        return self._file.read(size)

    async def seek(self, offset: int = 0) -> None:
        self._file.seek(offset)

    async def close(self) -> None:
        pass


# ----------------------------------------------------------------------
# Schema Initialization & Migrations
# ----------------------------------------------------------------------

def init_experiment_db() -> None:
    """Initialize database tables for legacy experiments and new Phase 4 workbench entities."""
    with _connect() as conn:
        # Legacy Phase 1-3 experiments table (strictly preserved for backward compatibility)
        conn.execute(
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
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_experiments_created_at
            ON experiments(created_at DESC)
            """
        )

        # Datasets registry
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS datasets_registry (
                dataset_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                file_hash TEXT NOT NULL UNIQUE,
                file_size INTEGER NOT NULL,
                row_count INTEGER NOT NULL,
                column_count INTEGER NOT NULL,
                schema_json TEXT NOT NULL,
                storage_path TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_datasets_hash
            ON datasets_registry(file_hash)
            """
        )

        # Workbench experiments
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS workbench_experiments (
                experiment_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                module TEXT NOT NULL,
                description TEXT,
                dataset_id TEXT,
                tags_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_wb_exp_created_at
            ON workbench_experiments(created_at DESC)
            """
        )

        # Experiment runs
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS experiment_runs (
                run_id TEXT PRIMARY KEY,
                experiment_id TEXT NOT NULL,
                run_name TEXT NOT NULL,
                module TEXT NOT NULL,
                status TEXT NOT NULL,
                parameters_json TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                artifacts_json TEXT NOT NULL,
                parent_run_id TEXT,
                dataset_id TEXT NOT NULL,
                duration_seconds REAL DEFAULT 0.0,
                error_message TEXT,
                created_at TEXT NOT NULL,
                completed_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_runs_exp_id
            ON experiment_runs(experiment_id)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_runs_created_at
            ON experiment_runs(created_at DESC)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_runs_module
            ON experiment_runs(module)
            """
        )

        # Reusable workflow definitions
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS workflow_definitions (
                workflow_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                stages_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )

        # Reusable workflow executions
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS workflow_executions (
                execution_id TEXT PRIMARY KEY,
                workflow_id TEXT NOT NULL,
                status TEXT NOT NULL,
                stage_runs_json TEXT NOT NULL,
                duration_seconds REAL DEFAULT 0.0,
                created_at TEXT NOT NULL,
                completed_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_wf_exec_created_at
            ON workflow_executions(created_at DESC)
            """
        )


# ----------------------------------------------------------------------
# Dataset Registry Management
# ----------------------------------------------------------------------

def register_dataset_from_df(
    df: pd.DataFrame,
    filename: str = "dataset.csv",
) -> Dict[str, Any]:
    """Register an in-memory DataFrame into the immutable dataset registry."""
    init_experiment_db()
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    return register_dataset_from_bytes(csv_bytes, filename=filename)


def register_dataset_from_bytes(
    content: bytes,
    filename: str = "dataset.csv",
) -> Dict[str, Any]:
    """Register CSV bytes into the immutable dataset registry with SHA-256 integrity."""
    init_experiment_db()
    file_hash = hashlib.sha256(content).hexdigest()
    dataset_id = f"ds_{file_hash[:12]}"
    file_size = len(content)

    # Check if already registered
    with _connect() as conn:
        existing = conn.execute(
            "SELECT * FROM datasets_registry WHERE file_hash = ?",
            (file_hash,),
        ).fetchone()
        if existing:
            item = dict(existing)
            item["schema"] = json.loads(item.pop("schema_json"))
            return item

    # Save to disk store
    storage_path = DATASETS_STORE_DIR / f"{file_hash}.csv"
    with open(storage_path, "wb") as f:
        f.write(content)

    # Parse metadata and schema
    df = pd.read_csv(io.BytesIO(content))
    row_count = int(len(df))
    column_count = int(len(df.columns))

    schema = []
    for col in df.columns:
        s = df[col]
        null_count = int(s.isnull().sum())
        schema.append({
            "name": str(col),
            "dtype": str(s.dtype),
            "null_count": null_count,
            "null_pct": round((null_count / max(1, row_count)) * 100, 2),
            "sample_values": [str(x) for x in s.dropna().head(3).tolist()],
        })

    created_at = datetime.now(timezone.utc).isoformat()

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO datasets_registry (
                dataset_id, filename, file_hash, file_size,
                row_count, column_count, schema_json, storage_path, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                dataset_id,
                filename,
                file_hash,
                file_size,
                row_count,
                column_count,
                _dump(schema),
                str(storage_path),
                created_at,
            ),
        )

    return {
        "dataset_id": dataset_id,
        "filename": filename,
        "file_hash": file_hash,
        "file_size": file_size,
        "row_count": row_count,
        "column_count": column_count,
        "schema": schema,
        "storage_path": str(storage_path),
        "created_at": created_at,
    }


def get_dataset(dataset_id: str) -> Dict[str, Any]:
    """Retrieve registered dataset record by ID."""
    init_experiment_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM datasets_registry WHERE dataset_id = ?",
            (dataset_id,),
        ).fetchone()

    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset with ID '{dataset_id}' not found.",
        )

    item = dict(row)
    item["schema"] = json.loads(item.pop("schema_json"))
    return item


def list_datasets(limit: int = 50, offset: int = 0) -> Dict[str, Any]:
    """List registered datasets with pagination."""
    init_experiment_db()
    limit = max(1, min(int(limit), 100))
    offset = max(0, int(offset))

    with _connect() as conn:
        total = conn.execute("SELECT COUNT(*) FROM datasets_registry").fetchone()[0]
        rows = conn.execute(
            """
            SELECT * FROM datasets_registry
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()

    datasets = []
    for r in rows:
        item = dict(r)
        item["schema"] = json.loads(item.pop("schema_json"))
        datasets.append(item)

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "datasets": datasets,
    }


def load_dataset_df(dataset_id: str) -> pd.DataFrame:
    """Load a registered dataset as a pandas DataFrame."""
    record = get_dataset(dataset_id)
    path = record["storage_path"]
    if not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail=f"Underlying dataset file for '{dataset_id}' is missing from storage.",
        )
    return pd.read_csv(path)


# ----------------------------------------------------------------------
# Workbench Experiment Management
# ----------------------------------------------------------------------

def create_workbench_experiment(
    name: str,
    module: str,
    description: Optional[str] = None,
    dataset_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Create a new workbench experiment."""
    init_experiment_db()
    clean_name = (name or "").strip()
    if not clean_name:
        raise HTTPException(status_code=400, detail="Experiment name is required.")

    clean_module = (module or "").strip().lower()
    if clean_module not in VALID_MODULES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid module '{clean_module}'. Must be one of {sorted(list(VALID_MODULES))}.",
        )

    if dataset_id:
        get_dataset(dataset_id)

    experiment_id = f"exp_{uuid.uuid4().hex[:10]}"
    now = datetime.now(timezone.utc).isoformat()
    tags_list = tags if isinstance(tags, list) else []

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO workbench_experiments (
                experiment_id, name, module, description,
                dataset_id, tags_json, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experiment_id,
                clean_name,
                clean_module,
                (description or "").strip() or None,
                dataset_id,
                _dump(tags_list),
                now,
                now,
            ),
        )

    return get_workbench_experiment(experiment_id)


def list_workbench_experiments(
    limit: int = 50,
    offset: int = 0,
    module: Optional[str] = None,
    search: Optional[str] = None,
) -> Dict[str, Any]:
    """List workbench experiments with module and text filters."""
    init_experiment_db()
    limit = max(1, min(int(limit), 100))
    offset = max(0, int(offset))

    query = "SELECT * FROM workbench_experiments WHERE 1=1"
    params = []

    if module:
        query += " AND module = ?"
        params.append(module.strip().lower())

    if search:
        query += " AND (name LIKE ? OR description LIKE ?)"
        s = f"%{search.strip()}%"
        params.extend([s, s])

    count_query = query.replace("SELECT *", "SELECT COUNT(*)")

    query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    with _connect() as conn:
        total = conn.execute(count_query, params[:-2]).fetchone()[0]
        rows = conn.execute(query, params).fetchall()

        # Gather run counts per experiment
        exp_ids = [r["experiment_id"] for r in rows]
        run_counts = {}
        if exp_ids:
            placeholders = ",".join(["?"] * len(exp_ids))
            counts = conn.execute(
                f"""
                SELECT experiment_id, COUNT(*) as cnt,
                       SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END) as completed_cnt
                FROM experiment_runs
                WHERE experiment_id IN ({placeholders})
                GROUP BY experiment_id
                """,
                exp_ids,
            ).fetchall()
            for c in counts:
                run_counts[c["experiment_id"]] = {
                    "total_runs": c["cnt"],
                    "completed_runs": c["completed_cnt"],
                }

    experiments = []
    for r in rows:
        item = dict(r)
        item["tags"] = json.loads(item.pop("tags_json"))
        counts = run_counts.get(item["experiment_id"], {"total_runs": 0, "completed_runs": 0})
        item["runs_count"] = counts["total_runs"]
        item["completed_runs_count"] = counts["completed_runs"]
        experiments.append(item)

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "experiments": experiments,
    }


def get_workbench_experiment(experiment_id: str) -> Dict[str, Any]:
    """Get single workbench experiment details and associated runs."""
    init_experiment_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM workbench_experiments WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchone()

        if not row:
            raise HTTPException(
                status_code=404,
                detail=f"Workbench experiment '{experiment_id}' not found.",
            )

        runs = conn.execute(
            """
            SELECT * FROM experiment_runs
            WHERE experiment_id = ?
            ORDER BY created_at DESC
            """,
            (experiment_id,),
        ).fetchall()

    item = dict(row)
    item["tags"] = json.loads(item.pop("tags_json"))

    serialized_runs = []
    for r in runs:
        run_item = dict(r)
        run_item["parameters"] = json.loads(run_item.pop("parameters_json"))
        run_item["metrics"] = json.loads(run_item.pop("metrics_json"))
        run_item["artifacts"] = json.loads(run_item.pop("artifacts_json"))
        serialized_runs.append(run_item)

    item["runs"] = serialized_runs
    item["runs_count"] = len(serialized_runs)
    return item


def delete_workbench_experiment(experiment_id: str) -> Dict[str, Any]:
    """Delete workbench experiment and cascade delete associated runs."""
    init_experiment_db()
    with _connect() as conn:
        conn.execute("DELETE FROM experiment_runs WHERE experiment_id = ?", (experiment_id,))
        cur = conn.execute("DELETE FROM workbench_experiments WHERE experiment_id = ?", (experiment_id,))
        if cur.rowcount == 0:
            raise HTTPException(
                status_code=404,
                detail=f"Workbench experiment '{experiment_id}' not found.",
            )

    return {"message": "Experiment and associated runs deleted successfully.", "experiment_id": experiment_id}


# ----------------------------------------------------------------------
# Experiment Run Management & Execution
# ----------------------------------------------------------------------

def create_experiment_run(
    experiment_id: str,
    run_name: Optional[str] = None,
    module: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None,
    parent_run_id: Optional[str] = None,
    dataset_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a new run entry in draft/queued state."""
    init_experiment_db()
    exp = get_workbench_experiment(experiment_id)

    target_dataset_id = dataset_id or exp.get("dataset_id")
    if not target_dataset_id:
        raise HTTPException(
            status_code=400,
            detail="A valid dataset_id is required for an experiment run.",
        )

    # Validate dataset existence
    get_dataset(target_dataset_id)

    run_module = (module or exp["module"]).strip().lower()
    run_id = f"run_{uuid.uuid4().hex[:10]}"
    name = (run_name or "").strip() or f"{exp['name']} - Run {exp['runs_count'] + 1}"
    params = parameters or {}
    now = datetime.now(timezone.utc).isoformat()

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO experiment_runs (
                run_id, experiment_id, run_name, module, status,
                parameters_json, metrics_json, artifacts_json,
                parent_run_id, dataset_id, duration_seconds,
                error_message, created_at, completed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0.0, NULL, ?, NULL)
            """,
            (
                run_id,
                experiment_id,
                name,
                run_module,
                "draft",
                _dump(params),
                _dump({}),
                _dump([]),
                parent_run_id,
                target_dataset_id,
                now,
            ),
        )

    return get_run(run_id)


def get_run(run_id: str) -> Dict[str, Any]:
    """Get run details by run_id."""
    init_experiment_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM experiment_runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")

    item = dict(row)
    item["parameters"] = json.loads(item.pop("parameters_json"))
    item["metrics"] = json.loads(item.pop("metrics_json"))
    item["artifacts"] = json.loads(item.pop("artifacts_json"))
    return item


def list_experiment_runs(
    limit: int = 50,
    offset: int = 0,
    module: Optional[str] = None,
    status: Optional[str] = None,
    experiment_id: Optional[str] = None,
    search: Optional[str] = None,
) -> Dict[str, Any]:
    """List runs across experiments with filtering."""
    init_experiment_db()
    limit = max(1, min(int(limit), 100))
    offset = max(0, int(offset))

    query = "SELECT * FROM experiment_runs WHERE 1=1"
    params = []

    if module:
        query += " AND module = ?"
        params.append(module.strip().lower())

    if status:
        query += " AND status = ?"
        params.append(status.strip().lower())

    if experiment_id:
        query += " AND experiment_id = ?"
        params.append(experiment_id.strip())

    if search:
        query += " AND (run_name LIKE ? OR run_id LIKE ?)"
        s = f"%{search.strip()}%"
        params.extend([s, s])

    count_query = query.replace("SELECT *", "SELECT COUNT(*)")

    query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    with _connect() as conn:
        total = conn.execute(count_query, params[:-2]).fetchone()[0]
        rows = conn.execute(query, params).fetchall()

    runs = []
    for r in rows:
        item = dict(r)
        item["parameters"] = json.loads(item.pop("parameters_json"))
        item["metrics"] = json.loads(item.pop("metrics_json"))
        item["artifacts"] = json.loads(item.pop("artifacts_json"))
        runs.append(item)

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "runs": runs,
    }


def delete_run(run_id: str) -> Dict[str, Any]:
    """Delete a single experiment run."""
    init_experiment_db()
    with _connect() as conn:
        cur = conn.execute("DELETE FROM experiment_runs WHERE run_id = ?", (run_id,))
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")

    return {"message": "Run deleted successfully.", "run_id": run_id}


def cancel_run(run_id: str) -> Dict[str, Any]:
    """Mark a running or queued run as cancelled."""
    init_experiment_db()
    _CANCELLED_WORKBENCH_RUNS.add(run_id)
    with _connect() as conn:
        conn.execute(
            """
            UPDATE experiment_runs
            SET status = 'cancelled', error_message = 'Cancelled by user.'
            WHERE run_id = ? AND status IN ('draft', 'queued', 'running')
            """,
            (run_id,),
        )
    return get_run(run_id)


# ----------------------------------------------------------------------
# Module Execution Dispatcher
# ----------------------------------------------------------------------

async def execute_module_run(run_id: str) -> Dict[str, Any]:
    """Execute the independent module workflow for the specified run."""
    init_experiment_db()
    run = get_run(run_id)

    # Update state to running
    with _connect() as conn:
        conn.execute(
            "UPDATE experiment_runs SET status = 'running', error_message = NULL WHERE run_id = ?",
            (run_id,),
        )

    start_time = time.time()
    module = run["module"]
    params = run["parameters"] or {}
    dataset_id = run["dataset_id"]

    # Lineage check: If parent_run_id is specified and produced a dataset artifact, prefer that!
    if run.get("parent_run_id"):
        try:
            parent_run = get_run(run["parent_run_id"])
            parent_artifacts = parent_run.get("artifacts") or []
            for art in parent_artifacts:
                if art.get("type") == "dataset" and art.get("dataset_id"):
                    dataset_id = art["dataset_id"]
                    break
        except Exception:
            pass

    try:
        # Load dataset
        df = load_dataset_df(dataset_id)
        dataset_meta = get_dataset(dataset_id)
        filename = dataset_meta.get("filename", "dataset.csv")

        metrics: Dict[str, Any] = {}
        artifacts: List[Dict[str, Any]] = []

        if module == "eda":
            metrics, artifacts = await _execute_eda_module(df, filename, params, run_id)
        elif module == "cleaning":
            metrics, artifacts = await _execute_cleaning_module(df, filename, params, run_id)
        elif module == "features":
            metrics, artifacts = await _execute_features_module(df, filename, params, run_id)
        elif module == "training":
            metrics, artifacts = await _execute_training_module(df, filename, params, run_id)
        elif module == "tuning":
            metrics, artifacts = await _execute_tuning_module(df, filename, params, run_id)
        elif module == "evaluation":
            metrics, artifacts = await _execute_evaluation_module(df, filename, params, run_id)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported module execution: '{module}'.")

        # Check for cancellation
        if run_id in _CANCELLED_WORKBENCH_RUNS:
            _CANCELLED_WORKBENCH_RUNS.remove(run_id)
            with _connect() as conn:
                conn.execute(
                    "UPDATE experiment_runs SET status = 'cancelled' WHERE run_id = ?",
                    (run_id,),
                )
            return get_run(run_id)

        duration = round(time.time() - start_time, 3)
        completed_at = datetime.now(timezone.utc).isoformat()

        with _connect() as conn:
            conn.execute(
                """
                UPDATE experiment_runs
                SET status = 'completed',
                    metrics_json = ?,
                    artifacts_json = ?,
                    duration_seconds = ?,
                    completed_at = ?
                WHERE run_id = ?
                """,
                (
                    _dump(metrics),
                    _dump(artifacts),
                    duration,
                    completed_at,
                    run_id,
                ),
            )

        return get_run(run_id)

    except Exception as exc:
        duration = round(time.time() - start_time, 3)
        completed_at = datetime.now(timezone.utc).isoformat()
        err_msg = str(exc)

        with _connect() as conn:
            conn.execute(
                """
                UPDATE experiment_runs
                SET status = 'failed',
                    error_message = ?,
                    duration_seconds = ?,
                    completed_at = ?
                WHERE run_id = ?
                """,
                (
                    err_msg,
                    duration,
                    completed_at,
                    run_id,
                ),
            )
        raise HTTPException(
            status_code=500,
            detail=f"Run '{run_id}' failed: {err_msg}",
        )


# ----------------------------------------------------------------------
# Module-Specific Executors
# ----------------------------------------------------------------------

async def _execute_eda_module(
    df: pd.DataFrame,
    filename: str,
    params: Dict[str, Any],
    run_id: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    from app.services.eda_service import profile_dataset, detect_eda_strategy

    target_column = params.get("target_column")
    profile = profile_dataset(df=df, filename=filename, target_column=target_column)
    strategy = detect_eda_strategy(df=df, profile_data=profile, target_column=target_column)

    summary_stats = profile.get("summary", {})
    metrics = {
        "rows": summary_stats.get("rows", len(df)),
        "columns": summary_stats.get("columns", len(df.columns)),
        "missing_cells_pct": summary_stats.get("missing_percentage", 0.0),
        "duplicate_rows": summary_stats.get("duplicate_rows", 0),
        "primary_profile": strategy.get("primary_profile", "general"),
        "complexity_tier": strategy.get("complexity_tier", "standard"),
        "high_correlations_count": len(profile.get("high_correlations", [])),
    }

    # Save artifact profile
    artifact_path = DATA_DIR / "eda_artifacts" / f"{run_id}_profile.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with open(artifact_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe({"profile": profile, "strategy": strategy}), f, indent=2)

    artifacts = [
        {
            "type": "eda_profile",
            "name": f"{filename} - EDA Profile",
            "path": str(artifact_path),
            "size_bytes": os.path.getsize(artifact_path),
        }
    ]
    return metrics, artifacts


async def _execute_cleaning_module(
    df: pd.DataFrame,
    filename: str,
    params: Dict[str, Any],
    run_id: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    from app.services.adanceCleaning import AdvancedDataCleaner

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    virtual_file = VirtualUploadFile(filename=filename, content=csv_bytes)

    # Format preprocessing config for AdvancedDataCleaner
    preprocessing_config = {
        "available_columns": list(df.columns),
        "impute_missing": params.get("impute_missing", True),
        "missing_strategy": params.get("missing_strategy", "mean"),
        "remove_outliers": params.get("remove_outliers", False),
        "outlier_method": params.get("outlier_method", "zscore"),
        "outlier_threshold": float(params.get("outlier_threshold", 3.0)),
        "remove_duplicates": params.get("remove_duplicates", True),
        "feature_scaling": params.get("feature_scaling", False),
        "scaling_method": params.get("scaling_method", "standard"),
        "encode_categorical": params.get("encode_categorical", False),
        "numeric_columns": [c for c in df.select_dtypes(include=[np.number]).columns],
        "categorical_columns": [c for c in df.select_dtypes(exclude=[np.number]).columns],
    }

    cleaner = AdvancedDataCleaner(virtual_file, preprocessing_config)
    clean_result = await cleaner.clean_data()
    clean_csv = clean_result.get("csv", "")
    stats = clean_result.get("stats", {})

    # Register the cleaned CSV as a new immutable dataset artifact
    clean_filename = f"cleaned_{filename}"
    derived_dataset = register_dataset_from_bytes(clean_csv.encode("utf-8"), filename=clean_filename)

    metrics = {
        "rows_processed": stats.get("rows_processed", len(df)),
        "rows_removed": stats.get("rows_removed", 0),
        "duplicates_removed": stats.get("duplicates_removed", 0),
        "outliers_removed": stats.get("outliers_removed", 0),
        "columns_processed": stats.get("columns_processed", len(df.columns)),
        "data_retention_pct": round((derived_dataset["row_count"] / max(1, len(df))) * 100, 2),
    }

    artifacts = [
        {
            "type": "dataset",
            "dataset_id": derived_dataset["dataset_id"],
            "name": clean_filename,
            "path": derived_dataset["storage_path"],
            "rows": derived_dataset["row_count"],
            "cols": derived_dataset["column_count"],
        }
    ]
    return metrics, artifacts


async def _execute_features_module(
    df: pd.DataFrame,
    filename: str,
    params: Dict[str, Any],
    run_id: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    from app.services.feature_service import TabularFeatureEngineer

    target_column = params.get("target_column")
    if not target_column or target_column not in df.columns:
        # Default to last column
        target_column = df.columns[-1]

    X = df.drop(columns=[target_column])
    y = df[target_column]

    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    poly_cols = num_cols[:3] if params.get("enable_polynomial", True) else []
    interaction_pairs = []
    if params.get("enable_interactions", True) and len(num_cols) >= 2:
        for i in range(min(3, len(num_cols) - 1)):
            interaction_pairs.append((num_cols[i], num_cols[i+1], "*"))

    engineer = TabularFeatureEngineer(
        polynomial_cols=poly_cols,
        interaction_pairs=interaction_pairs,
        log_cols=params.get("log_cols", []),
        clip_outliers=params.get("clip_outliers", False),
    )

    engineer.fit(X, y)
    X_trans = engineer.transform(X)

    # Recombine with target
    df_engineered = X_trans.copy()
    df_engineered[target_column] = y.values

    # Register the engineered dataset artifact
    feat_filename = f"features_{filename}"
    csv_bytes = df_engineered.to_csv(index=False).encode("utf-8")
    derived_dataset = register_dataset_from_bytes(csv_bytes, filename=feat_filename)

    metrics = {
        "original_features": len(X.columns),
        "engineered_features": len(X_trans.columns),
        "features_added": len(X_trans.columns) - len(X.columns),
        "lineage_steps_count": len(engineer.get_lineage()),
        "target_column": target_column,
    }

    artifacts = [
        {
            "type": "dataset",
            "dataset_id": derived_dataset["dataset_id"],
            "name": feat_filename,
            "path": derived_dataset["storage_path"],
            "rows": derived_dataset["row_count"],
            "cols": derived_dataset["column_count"],
        }
    ]
    return metrics, artifacts


async def _execute_training_module(
    df: pd.DataFrame,
    filename: str,
    params: Dict[str, Any],
    run_id: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    from app.services.train_service import train_model_service
    from app.services.classification_service import train_classification_service

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    virtual_file = VirtualUploadFile(filename=filename, content=csv_bytes)

    target_column = params.get("target_column") or df.columns[-1]
    task_type = params.get("task_type")

    # Auto-detect task type if not provided
    if not task_type:
        target_series = df[target_column].dropna()
        if pd.api.types.is_numeric_dtype(target_series) and target_series.nunique() > 15:
            task_type = "regression"
        else:
            task_type = "classification"

    task_type = task_type.strip().lower()
    model_name = params.get("model", params.get("model_type", "linear" if task_type == "regression" else "logistic"))
    scaler = params.get("scaler", "standard")
    split_ratio = float(params.get("split_ratio", params.get("splitRatio", 0.8)))
    missing = params.get("missing", "median")
    encoding = params.get("encoding", "onehot")

    if task_type == "regression":
        result = await train_model_service(
            file=virtual_file,
            model=model_name,
            scaler=scaler,
            split_ratio=split_ratio,
            missing=missing,
            encoding=encoding,
            target_column=target_column,
            return_pipeline=False,
        )
    else:
        result = await train_classification_service(
            file=virtual_file,
            model=model_name,
            scaler=scaler,
            split_ratio=split_ratio,
            missing=missing,
            encoding=encoding,
            target_column=target_column,
            return_pipeline=False,
        )

    # Extract clean metrics
    eval_metrics = result.get("metrics") or result.get("holdout_metrics") or {}
    metrics = {
        "task_type": task_type,
        "model_type": model_name,
        "target_column": target_column,
        **eval_metrics,
    }
    if "f1" not in metrics:
        metrics["f1"] = metrics.get("f1_weighted") or metrics.get("f1_macro") or 0.0
    if "precision" not in metrics:
        metrics["precision"] = metrics.get("precision_weighted") or metrics.get("precision_macro") or 0.0
    if "recall" not in metrics:
        metrics["recall"] = metrics.get("recall_weighted") or metrics.get("recall_macro") or 0.0

    # Save trained model summary as artifact
    model_art_path = DATA_DIR / "model_artifacts" / f"{run_id}_train_result.json"
    model_art_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_art_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(result), f, indent=2)

    artifacts = [
        {
            "type": "model_result",
            "name": f"{model_name} Training Result",
            "path": str(model_art_path),
            "size_bytes": os.path.getsize(model_art_path),
        }
    ]
    return metrics, artifacts


async def _execute_tuning_module(
    df: pd.DataFrame,
    filename: str,
    params: Dict[str, Any],
    run_id: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    from app.services.tuning_service import run_hyperparameter_tuning

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    virtual_file = VirtualUploadFile(filename=filename, content=csv_bytes)

    target_column = params.get("target_column") or df.columns[-1]
    task_type = params.get("task_type")
    if not task_type:
        target_series = df[target_column].dropna()
        if pd.api.types.is_numeric_dtype(target_series) and target_series.nunique() > 15:
            task_type = "regression"
        else:
            task_type = "classification"

    model_type = params.get("model_type") or params.get("model") or ("RandomForestClassifier" if task_type == "classification" else "RandomForestRegressor")
    # Normalize friendly model names to tuning service names
    model_name_map = {
        "random_forest": "RandomForestClassifier" if task_type == "classification" else "RandomForestRegressor",
        "rf": "RandomForestClassifier" if task_type == "classification" else "RandomForestRegressor",
        "logistic": "LogisticRegression",
        "decision_tree": "DecisionTreeClassifier" if task_type == "classification" else "DecisionTreeRegressor",
        "dt": "DecisionTreeClassifier" if task_type == "classification" else "DecisionTreeRegressor",
        "extra_trees": "ExtraTreesClassifier",
        "linear": "LinearRegression",
    }
    model_type = model_name_map.get(str(model_type).lower(), model_type)
    n_trials = int(params.get("n_trials", 10))
    n_splits = int(params.get("n_splits", 3))

    tuning_res = await run_hyperparameter_tuning(
        file=virtual_file,
        task_type=task_type,
        target_column=target_column,
        model_type=model_type,
        metric=params.get("metric"),
        n_trials=n_trials,
        n_splits=n_splits,
        scaler=params.get("scaler", "standard"),
        missing=params.get("missing", "median"),
        split_ratio=float(params.get("split_ratio", 0.8)),
    )

    baseline_metrics = tuning_res.get("baseline_metrics") or {}
    tuned_metrics = tuning_res.get("tuned_metrics") or {}

    primary_metric = tuning_res.get("metric") or ("accuracy" if task_type == "classification" else "r2")
    baseline_val = baseline_metrics.get(primary_metric, 0.0)
    tuned_val = tuned_metrics.get(primary_metric, 0.0)

    delta_pct = 0.0
    if baseline_val and baseline_val != 0:
        delta_pct = round(((tuned_val - baseline_val) / abs(baseline_val)) * 100, 2)

    trials_list = tuning_res.get("trials") or tuning_res.get("trial_history") or []
    best_params = tuning_res.get("best_params") or tuning_res.get("best_parameters") or {}
    best_score = tuning_res.get("best_score") or tuned_val

    metrics = {
        "primary_metric": primary_metric,
        "baseline_score": baseline_val,
        "tuned_score": tuned_val,
        "best_score": best_score,
        "score_improvement_pct": delta_pct,
        "best_trial_number": tuning_res.get("best_trial_number"),
        "best_parameters": best_params,
        "trials_evaluated": len(trials_list),
    }

    # Save tuning summary artifact
    art_path = DATA_DIR / "tuning_artifacts" / f"{run_id}_tuning.json"
    art_path.parent.mkdir(parents=True, exist_ok=True)
    with open(art_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(tuning_res), f, indent=2)

    artifacts = [
        {
            "type": "tuning_report",
            "name": f"{model_type} Optuna Optimization",
            "path": str(art_path),
            "tuning_run_id": tuning_res.get("run_id"),
        }
    ]
    return metrics, artifacts


async def _execute_evaluation_module(
    df: pd.DataFrame,
    filename: str,
    params: Dict[str, Any],
    run_id: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    # Comprehensive evaluation and statistical diagnostics
    target_column = params.get("target_column") or df.columns[-1]
    y = df[target_column].dropna()

    is_numeric = pd.api.types.is_numeric_dtype(y)
    metrics: Dict[str, Any] = {
        "sample_size": len(y),
        "target_column": target_column,
        "is_continuous": is_numeric and y.nunique() > 15,
    }

    if is_numeric and y.nunique() > 15:
        metrics.update({
            "mean": round(float(y.mean()), 4),
            "std": round(float(y.std()), 4),
            "skewness": round(float(y.skew()), 4),
            "kurtosis": round(float(y.kurtosis()), 4),
            "min": round(float(y.min()), 4),
            "max": round(float(y.max()), 4),
        })
    else:
        val_counts = y.value_counts(normalize=True).to_dict()
        metrics["class_distribution"] = {str(k): round(float(v), 4) for k, v in val_counts.items()}
        metrics["classes_count"] = y.nunique()

    # Save evaluation artifact
    art_path = DATA_DIR / "eval_artifacts" / f"{run_id}_eval.json"
    art_path.parent.mkdir(parents=True, exist_ok=True)
    with open(art_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(metrics), f, indent=2)

    artifacts = [
        {
            "type": "evaluation_summary",
            "name": f"{filename} Target Evaluation",
            "path": str(art_path),
        }
    ]
    return metrics, artifacts


# ----------------------------------------------------------------------
# Multi-Run Experiment Comparison Engine
# ----------------------------------------------------------------------

HIGHER_IS_BETTER_METRICS = {
    "accuracy", "f1", "precision", "recall", "r2", "roc_auc", "auc",
    "balanced_accuracy", "explained_variance", "tuned_score", "baseline_score",
    "score_improvement_pct", "data_retention_pct",
}
LOWER_IS_BETTER_METRICS = {
    "rmse", "mae", "mse", "log_loss", "loss", "max_error",
    "missing_cells_pct", "duplicate_rows", "outliers_removed",
}


def compare_experiment_runs(run_ids: List[str]) -> Dict[str, Any]:
    """
    Compare multiple experiment runs side-by-side with strict compatibility checks,
    parameter diffing, metric deltas, and visual comparison charts data.
    """
    init_experiment_db()
    if not run_ids or len(run_ids) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least 2 run IDs are required for side-by-side comparison.",
        )

    runs = [get_run(r_id) for r_id in run_ids]

    # Validate all runs belong to compatible modules
    modules = set(r["module"] for r in runs)
    if len(modules) > 1:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Incompatible run comparison: cannot compare runs across different modules. "
                f"Found modules: {sorted(list(modules))}. Please select runs from the same module."
            ),
        )

    module = list(modules)[0]

    # Extract Parameter Diff
    all_param_keys = set()
    for r in runs:
        all_param_keys.update(r.get("parameters", {}).keys())

    common_parameters = {}
    varying_parameters = {}

    for k in sorted(all_param_keys):
        vals = [r.get("parameters", {}).get(k) for r in runs]
        # Check if all values match
        first_val = vals[0]
        if all(v == first_val for v in vals):
            common_parameters[k] = first_val
        else:
            varying_parameters[k] = {r["run_id"]: r.get("parameters", {}).get(k) for r in runs}

    # Extract Metric Matrix
    all_metric_keys = set()
    for r in runs:
        all_metric_keys.update(r.get("metrics", {}).keys())

    # Filter out non-numeric / structural metrics for charts
    comparison_metrics = []
    baseline_run = runs[0]

    for m_key in sorted(all_metric_keys):
        numeric_vals = []
        is_numeric_metric = True

        for r in runs:
            val = r.get("metrics", {}).get(m_key)
            if val is not None and isinstance(val, (int, float)) and not isinstance(val, bool):
                numeric_vals.append(val)
            else:
                is_numeric_metric = False

        if not is_numeric_metric or not numeric_vals:
            continue

        direction = "higher" if m_key.lower() in HIGHER_IS_BETTER_METRICS else (
            "lower" if m_key.lower() in LOWER_IS_BETTER_METRICS else "neutral"
        )

        # Determine winning run
        best_run_id = None
        if direction == "higher":
            max_val = max(numeric_vals)
            for r in runs:
                if r.get("metrics", {}).get(m_key) == max_val:
                    best_run_id = r["run_id"]
                    break
        elif direction == "lower":
            min_val = min(numeric_vals)
            for r in runs:
                if r.get("metrics", {}).get(m_key) == min_val:
                    best_run_id = r["run_id"]
                    break

        baseline_metric_val = baseline_run.get("metrics", {}).get(m_key)

        metric_row = {
            "metric": m_key,
            "direction": direction,
            "best_run_id": best_run_id,
            "values": {},
            "deltas": {},
        }

        for r in runs:
            r_val = r.get("metrics", {}).get(m_key)
            metric_row["values"][r["run_id"]] = r_val
            if r_val is not None and baseline_metric_val is not None and baseline_metric_val != 0:
                delta = round(((r_val - baseline_metric_val) / abs(baseline_metric_val)) * 100, 2)
                metric_row["deltas"][r["run_id"]] = delta
            else:
                metric_row["deltas"][r["run_id"]] = 0.0

        comparison_metrics.append(metric_row)

    # Build chart friendly comparison payload
    chart_series = []
    for r in runs:
        chart_series.append({
            "run_id": r["run_id"],
            "run_name": r["run_name"],
            "duration_seconds": r.get("duration_seconds", 0.0),
            "status": r.get("status"),
            "metrics": {
                cm["metric"]: cm["values"].get(r["run_id"]) for cm in comparison_metrics
            },
        })

    return {
        "module": module,
        "runs_count": len(runs),
        "baseline_run_id": baseline_run["run_id"],
        "runs": [
            {
                "run_id": r["run_id"],
                "run_name": r["run_name"],
                "status": r["status"],
                "duration_seconds": r["duration_seconds"],
                "created_at": r["created_at"],
            }
            for r in runs
        ],
        "common_parameters": common_parameters,
        "varying_parameters": varying_parameters,
        "comparison_metrics": comparison_metrics,
        "chart_series": chart_series,
    }


# ----------------------------------------------------------------------
# Lineage & Dependency Graph
# ----------------------------------------------------------------------

def get_run_lineage(run_id: str) -> Dict[str, Any]:
    """
    Construct the full DAG lineage graph for a given run ID,
    including ancestor datasets, parent runs, and downstream child runs.
    """
    init_experiment_db()
    current_run = get_run(run_id)

    nodes = {}
    edges = []

    def add_node(node_id: str, label: str, node_type: str, details: dict):
        if node_id not in nodes:
            nodes[node_id] = {
                "id": node_id,
                "label": label,
                "type": node_type,
                "details": details,
            }

    # Add current run node
    add_node(
        current_run["run_id"],
        current_run["run_name"],
        f"run_{current_run['module']}",
        {
            "status": current_run["status"],
            "duration": current_run["duration_seconds"],
            "created_at": current_run["created_at"],
        },
    )

    # 1. Trace Ancestry Upwards
    curr = current_run
    while curr.get("parent_run_id"):
        parent_id = curr["parent_run_id"]
        try:
            parent_run = get_run(parent_id)
            add_node(
                parent_run["run_id"],
                parent_run["run_name"],
                f"run_{parent_run['module']}",
                {
                    "status": parent_run["status"],
                    "duration": parent_run["duration_seconds"],
                    "created_at": parent_run["created_at"],
                },
            )
            edges.append({
                "from": parent_run["run_id"],
                "to": curr["run_id"],
                "label": "derived_from",
            })
            curr = parent_run
        except Exception:
            break

    # Add root dataset node
    root_dataset_id = curr.get("dataset_id")
    if root_dataset_id:
        try:
            ds = get_dataset(root_dataset_id)
            add_node(
                ds["dataset_id"],
                ds["filename"],
                "dataset_root",
                {"rows": ds["row_count"], "cols": ds["column_count"], "hash": ds["file_hash"][:8]},
            )
            edges.append({
                "from": ds["dataset_id"],
                "to": curr["run_id"],
                "label": "inputs_to",
            })
        except Exception:
            pass

    # 2. Trace Children Downwards
    with _connect() as conn:
        child_rows = conn.execute(
            "SELECT * FROM experiment_runs WHERE parent_run_id = ?",
            (run_id,),
        ).fetchall()

    for r in child_rows:
        child_run = dict(r)
        add_node(
            child_run["run_id"],
            child_run["run_name"],
            f"run_{child_run['module']}",
            {"status": child_run["status"], "duration": child_run["duration_seconds"]},
        )
        edges.append({
            "from": current_run["run_id"],
            "to": child_run["run_id"],
            "label": "feeds_into",
        })

    # Add Output Artifact Nodes
    for art in current_run.get("artifacts", []):
        art_id = f"art_{art.get('name', 'artifact')}_{uuid.uuid4().hex[:6]}"
        add_node(
            art_id,
            art.get("name", "Artifact"),
            "artifact",
            art,
        )
        edges.append({
            "from": current_run["run_id"],
            "to": art_id,
            "label": "produces",
        })

    return {
        "focus_run_id": run_id,
        "nodes": list(nodes.values()),
        "edges": edges,
    }


# ----------------------------------------------------------------------
# Reusable Workflow Orchestration
# ----------------------------------------------------------------------

def create_workflow(
    name: str,
    stages: List[Dict[str, Any]],
    description: Optional[str] = None,
) -> Dict[str, Any]:
    """Define a composable multi-stage workflow pipeline."""
    init_experiment_db()
    clean_name = (name or "").strip()
    if not clean_name:
        raise HTTPException(status_code=400, detail="Workflow name is required.")

    if not stages or not isinstance(stages, list):
        raise HTTPException(status_code=400, detail="Workflow must define at least 1 stage.")

    for i, st in enumerate(stages):
        if "module" not in st or st["module"] not in VALID_MODULES:
            raise HTTPException(
                status_code=400,
                detail=f"Stage {i+1} has invalid module '{st.get('module')}'. Must be one of {sorted(list(VALID_MODULES))}.",
            )

    workflow_id = f"wf_{uuid.uuid4().hex[:10]}"
    now = datetime.now(timezone.utc).isoformat()

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO workflow_definitions (
                workflow_id, name, description, stages_json, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                workflow_id,
                clean_name,
                (description or "").strip() or None,
                _dump(stages),
                now,
                now,
            ),
        )

    return get_workflow(workflow_id)


def list_workflows(limit: int = 50, offset: int = 0) -> Dict[str, Any]:
    """List defined workflow pipelines."""
    init_experiment_db()
    limit = max(1, min(int(limit), 100))
    offset = max(0, int(offset))

    with _connect() as conn:
        total = conn.execute("SELECT COUNT(*) FROM workflow_definitions").fetchone()[0]
        rows = conn.execute(
            "SELECT * FROM workflow_definitions ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()

    workflows = []
    for r in rows:
        item = dict(r)
        item["stages"] = json.loads(item.pop("stages_json"))
        workflows.append(item)

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "workflows": workflows,
    }


def get_workflow(workflow_id: str) -> Dict[str, Any]:
    """Get single workflow definition."""
    init_experiment_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM workflow_definitions WHERE workflow_id = ?",
            (workflow_id,),
        ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found.")

    item = dict(row)
    item["stages"] = json.loads(item.pop("stages_json"))
    return item


def delete_workflow(workflow_id: str) -> Dict[str, Any]:
    """Delete a workflow definition."""
    init_experiment_db()
    with _connect() as conn:
        cur = conn.execute("DELETE FROM workflow_definitions WHERE workflow_id = ?", (workflow_id,))
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found.")

    return {"message": "Workflow deleted successfully.", "workflow_id": workflow_id}


async def execute_workflow(
    workflow_id: str,
    dataset_id: str,
) -> Dict[str, Any]:
    """
    Execute a multi-stage workflow sequentially, feeding derivative dataset
    artifacts from upstream stages into downstream stages.
    """
    init_experiment_db()
    wf = get_workflow(workflow_id)
    get_dataset(dataset_id)

    execution_id = f"exec_{uuid.uuid4().hex[:10]}"
    now = datetime.now(timezone.utc).isoformat()
    start_time = time.time()

    # Record initial execution state
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO workflow_executions (
                execution_id, workflow_id, status, stage_runs_json,
                duration_seconds, created_at, completed_at
            )
            VALUES (?, ?, 'running', ?, 0.0, ?, NULL)
            """,
            (
                execution_id,
                workflow_id,
                _dump({}),
                now,
            ),
        )

    # Create parent experiment container for this workflow execution
    wf_exp = create_workbench_experiment(
        name=f"Workflow: {wf['name']} ({execution_id})",
        module="pipeline",
        description=f"Automated execution of workflow '{wf['name']}'",
        dataset_id=dataset_id,
    )

    stage_runs = []
    current_dataset_id = dataset_id
    parent_run_id = None
    exec_status = "completed"

    try:
        for idx, stage in enumerate(wf["stages"]):
            stage_module = stage["module"]
            stage_name = stage.get("name", f"Stage {idx+1}: {stage_module.title()}")
            stage_params = stage.get("parameters", {})

            # Create run in container experiment
            run_record = create_experiment_run(
                experiment_id=wf_exp["experiment_id"],
                run_name=f"{stage_name}",
                module=stage_module,
                parameters=stage_params,
                parent_run_id=parent_run_id,
                dataset_id=current_dataset_id,
            )

            # Execute run
            executed_run = await execute_module_run(run_record["run_id"])
            stage_runs.append({
                "stage_index": idx + 1,
                "stage_name": stage_name,
                "run_id": executed_run["run_id"],
                "status": executed_run["status"],
                "duration_seconds": executed_run["duration_seconds"],
                "metrics": executed_run.get("metrics"),
            })

            parent_run_id = executed_run["run_id"]

            # If stage produced a dataset artifact, propagate it
            for art in executed_run.get("artifacts", []):
                if art.get("type") == "dataset" and art.get("dataset_id"):
                    current_dataset_id = art["dataset_id"]
                    break

            if executed_run["status"] != "completed":
                exec_status = "failed"
                break

    except Exception:
        exec_status = "failed"

    duration = round(time.time() - start_time, 3)
    completed_at = datetime.now(timezone.utc).isoformat()

    with _connect() as conn:
        conn.execute(
            """
            UPDATE workflow_executions
            SET status = ?,
                stage_runs_json = ?,
                duration_seconds = ?,
                completed_at = ?
            WHERE execution_id = ?
            """,
            (
                exec_status,
                _dump(stage_runs),
                duration,
                completed_at,
                execution_id,
            ),
        )

    return get_workflow_execution(execution_id)


def get_workflow_execution(execution_id: str) -> Dict[str, Any]:
    """Retrieve details for a single workflow execution."""
    init_experiment_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM workflow_executions WHERE execution_id = ?",
            (execution_id,),
        ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail=f"Workflow execution '{execution_id}' not found.")

    item = dict(row)
    item["stage_runs"] = json.loads(item.pop("stage_runs_json"))
    return item


def list_workflow_executions(
    workflow_id: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> Dict[str, Any]:
    """List workflow executions."""
    init_experiment_db()
    limit = max(1, min(int(limit), 100))
    offset = max(0, int(offset))

    query = "SELECT * FROM workflow_executions"
    params = []

    if workflow_id:
        query += " WHERE workflow_id = ?"
        params.append(workflow_id.strip())

    query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    with _connect() as conn:
        rows = conn.execute(query, params).fetchall()

    executions = []
    for r in rows:
        item = dict(r)
        item["stage_runs"] = json.loads(item.pop("stage_runs_json"))
        executions.append(item)

    return {"executions": executions}


# ----------------------------------------------------------------------
# Backward Compatibility Helpers (Preserving Existing Contracts)
# ----------------------------------------------------------------------

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
    """Save a successful training/comparison result (Legacy Phase 1-3 support)."""
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
    """List legacy experiments with pagination."""
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
    """Get legacy experiment details."""
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
    """Delete legacy experiment."""
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
