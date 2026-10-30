# server/app/routes/experiment_routes.py
"""
AutoVeyra Phase 4: Independent Module-Wise Experiment Workbench,
Experiment Tracking & Workflow Orchestration API Router.
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Body, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from app.services.experiment_service import (
    # Legacy Phase 1-3
    list_experiments,
    get_experiment,
    delete_experiment,
    # Phase 4 Datasets
    register_dataset_from_bytes,
    get_dataset,
    list_datasets,
    # Phase 4 Workbench
    create_workbench_experiment,
    list_workbench_experiments,
    get_workbench_experiment,
    delete_workbench_experiment,
    # Phase 4 Runs
    create_experiment_run,
    list_experiment_runs,
    get_run,
    delete_run,
    cancel_run,
    execute_module_run,
    # Phase 4 Comparison & Lineage
    compare_experiment_runs,
    get_run_lineage,
    # Phase 4 Workflows
    create_workflow,
    list_workflows,
    get_workflow,
    delete_workflow,
    execute_workflow,
    get_workflow_execution,
    list_workflow_executions,
)

router = APIRouter(
    prefix="/experiments",
    tags=["Experiment Workbench & Tracking"],
)


# ======================================================================
# 1. Dataset Registry Endpoints
# ======================================================================

@router.post("/datasets/upload")
async def upload_dataset_endpoint(
    file: UploadFile = File(...),
):
    """Upload and register a dataset with cryptographic SHA-256 integrity."""
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files can be registered.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    meta = register_dataset_from_bytes(content=content, filename=file.filename)
    return JSONResponse(content=meta)


@router.get("/datasets")
def list_datasets_endpoint(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """List registered immutable datasets."""
    return list_datasets(limit=limit, offset=offset)


@router.get("/datasets/{dataset_id}")
def get_dataset_endpoint(dataset_id: str):
    """Retrieve details and schema of a registered dataset."""
    return get_dataset(dataset_id)


# ======================================================================
# 2. Workbench Experiment Endpoints
# ======================================================================

@router.post("/workbench")
def create_workbench_experiment_endpoint(
    payload: Dict[str, Any] = Body(
        ...,
        example={
            "name": "Decision Tree Depth Exploration",
            "module": "training",
            "description": "Evaluate classification performance across max_depth values.",
            "dataset_id": "ds_123456789abc",
            "tags": ["tree", "classification"],
        },
    ),
):
    """Create a new workbench experiment."""
    name = payload.get("name")
    module = payload.get("module")
    if not name or not module:
        raise HTTPException(status_code=400, detail="Both 'name' and 'module' are required.")

    return create_workbench_experiment(
        name=name,
        module=module,
        description=payload.get("description"),
        dataset_id=payload.get("dataset_id"),
        tags=payload.get("tags"),
    )


@router.get("/workbench")
def list_workbench_experiments_endpoint(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    module: Optional[str] = Query(default=None),
    search: Optional[str] = Query(default=None),
):
    """List workbench experiments with module and search filters."""
    return list_workbench_experiments(
        limit=limit,
        offset=offset,
        module=module,
        search=search,
    )


@router.get("/workbench/{experiment_id}")
def get_workbench_experiment_endpoint(experiment_id: str):
    """Get single workbench experiment details and its attached runs."""
    return get_workbench_experiment(experiment_id)


@router.delete("/workbench/{experiment_id}")
def delete_workbench_experiment_endpoint(experiment_id: str):
    """Delete a workbench experiment and cascade delete its runs."""
    return delete_workbench_experiment(experiment_id)


# ======================================================================
# 3. Experiment Runs Endpoints
# ======================================================================

@router.post("/workbench/{experiment_id}/runs")
def create_run_endpoint(
    experiment_id: str,
    payload: Dict[str, Any] = Body(
        default={},
        example={
            "run_name": "StandardScaler + LogisticRegression",
            "module": "training",
            "parameters": {"scaler": "standard", "model_type": "logistic"},
            "parent_run_id": None,
            "dataset_id": None,
        },
    ),
):
    """Create a new run within a workbench experiment."""
    return create_experiment_run(
        experiment_id=experiment_id,
        run_name=payload.get("run_name"),
        module=payload.get("module"),
        parameters=payload.get("parameters"),
        parent_run_id=payload.get("parent_run_id"),
        dataset_id=payload.get("dataset_id"),
    )


@router.get("/runs")
def list_runs_endpoint(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    module: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
    experiment_id: Optional[str] = Query(default=None),
    search: Optional[str] = Query(default=None),
):
    """List runs across all experiments."""
    return list_experiment_runs(
        limit=limit,
        offset=offset,
        module=module,
        status=status,
        experiment_id=experiment_id,
        search=search,
    )


@router.get("/runs/{run_id}")
def get_run_endpoint(run_id: str):
    """Retrieve details for a single run."""
    return get_run(run_id)


@router.delete("/runs/{run_id}")
def delete_run_endpoint(run_id: str):
    """Delete a single experiment run."""
    return delete_run(run_id)


@router.post("/runs/{run_id}/execute")
async def execute_run_endpoint(run_id: str):
    """Trigger execution of an independent module run."""
    return await execute_module_run(run_id)


@router.post("/runs/{run_id}/cancel")
def cancel_run_endpoint(run_id: str):
    """Cancel an active or queued run."""
    return cancel_run(run_id)


@router.get("/runs/{run_id}/lineage")
def get_run_lineage_endpoint(run_id: str):
    """Retrieve full dependency and lineage DAG for a run."""
    return get_run_lineage(run_id)


# ======================================================================
# 4. Multi-Run Comparison Endpoint
# ======================================================================

@router.post("/compare")
def compare_runs_endpoint(
    payload: Dict[str, Any] = Body(
        ...,
        example={"run_ids": ["run_1234", "run_5678"]},
    ),
):
    """Compare multiple experiment runs side-by-side with diffs and winning metrics."""
    run_ids = payload.get("run_ids")
    if not run_ids or not isinstance(run_ids, list):
        raise HTTPException(
            status_code=400,
            detail="Payload must contain a 'run_ids' list with at least 2 run IDs.",
        )

    return compare_experiment_runs(run_ids)


# ======================================================================
# 5. Reusable Workflow Endpoints
# ======================================================================

@router.post("/workflows")
def create_workflow_endpoint(
    payload: Dict[str, Any] = Body(
        ...,
        example={
            "name": "Standard ML Pipeline",
            "description": "Cleaning -> Features -> Training",
            "stages": [
                {"name": "Clean Data", "module": "cleaning", "parameters": {"impute_missing": True}},
                {"name": "Feature Engineering", "module": "features", "parameters": {"strategy": "balanced"}},
                {"name": "Model Training", "module": "training", "parameters": {"model_type": "random_forest"}},
            ],
        },
    ),
):
    """Define a reusable multi-stage workflow pipeline."""
    name = payload.get("name")
    stages = payload.get("stages")
    if not name or not stages:
        raise HTTPException(status_code=400, detail="Workflow 'name' and 'stages' are required.")

    return create_workflow(
        name=name,
        stages=stages,
        description=payload.get("description"),
    )


@router.get("/workflows")
def list_workflows_endpoint(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """List defined reusable workflows."""
    return list_workflows(limit=limit, offset=offset)


@router.get("/workflows/{workflow_id}")
def get_workflow_endpoint(workflow_id: str):
    """Retrieve details for a single workflow definition."""
    return get_workflow(workflow_id)


@router.delete("/workflows/{workflow_id}")
def delete_workflow_endpoint(workflow_id: str):
    """Delete a workflow definition."""
    return delete_workflow(workflow_id)


@router.post("/workflows/{workflow_id}/execute")
async def execute_workflow_endpoint(
    workflow_id: str,
    payload: Dict[str, Any] = Body(
        ...,
        example={"dataset_id": "ds_123456789abc"},
    ),
):
    """Execute a reusable multi-stage workflow on a dataset."""
    dataset_id = payload.get("dataset_id")
    if not dataset_id:
        raise HTTPException(status_code=400, detail="'dataset_id' is required to execute a workflow.")

    return await execute_workflow(workflow_id=workflow_id, dataset_id=dataset_id)


@router.get("/workflows/{workflow_id}/executions")
def list_workflow_executions_endpoint(
    workflow_id: str,
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """List execution history for a workflow."""
    return list_workflow_executions(workflow_id=workflow_id, limit=limit, offset=offset)


@router.get("/executions/{execution_id}")
def get_execution_endpoint(execution_id: str):
    """Retrieve details of a single workflow execution."""
    return get_workflow_execution(execution_id)


# ======================================================================
# 6. Legacy Experiments Endpoints (Preserving Backward Compatibility)
# ======================================================================

@router.get("")
def get_legacy_experiments(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    task_type: Optional[str] = Query(default=None),
):
    """List legacy AutoML experiment runs (Phase 1-3)."""
    return list_experiments(
        limit=limit,
        offset=offset,
        task_type=task_type,
    )


@router.get("/{experiment_id:int}")
def get_legacy_experiment_details(experiment_id: int):
    """Retrieve details of a single legacy experiment."""
    return get_experiment(experiment_id)


@router.delete("/{experiment_id:int}")
def remove_legacy_experiment(experiment_id: int):
    """Delete a single legacy experiment."""
    return delete_experiment(experiment_id)
