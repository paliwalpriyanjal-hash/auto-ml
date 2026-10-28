# app/routes/tuning_routes.py

from typing import Optional
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from app.services.tuning_service import (
    cancel_tuning_run,
    delete_tuning_run,
    get_tuning_run,
    list_tuning_runs,
    run_hyperparameter_tuning,
)

router = APIRouter(prefix="/tuning", tags=["Hyperparameter Tuning"])


@router.post("/start")
@router.post("")
async def start_tuning(
    file: UploadFile = File(...),
    task_type: str = Form(...),
    target_column: str = Form(...),
    model_type: str = Form(...),
    metric: Optional[str] = Form(None),
    n_trials: int = Form(20),
    n_splits: int = Form(3),
    scaler: str = Form("standard"),
    missing: str = Form("median"),
    splitRatio: float = Form(0.8),
):
    """
    Launch Optuna hyperparameter optimization for a candidate model.
    Runs CV over the training partition and evaluates baseline vs tuned pipeline
    on the isolated holdout test split.
    """
    return await run_hyperparameter_tuning(
        file=file,
        task_type=task_type,
        target_column=target_column,
        model_type=model_type,
        metric=metric,
        n_trials=n_trials,
        n_splits=n_splits,
        scaler=scaler,
        missing=missing,
        split_ratio=splitRatio,
    )


@router.get("")
def get_tuning_history(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """List historical hyperparameter tuning runs with pagination."""
    return list_tuning_runs(limit=limit, offset=offset)


@router.get("/{run_id}")
def get_tuning_run_detail(run_id: str):
    """Retrieve full details for a tuning run, including baseline vs tuned deltas and trials."""
    return get_tuning_run(run_id)


@router.get("/{run_id}/trials")
def get_tuning_run_trials(run_id: str):
    """Retrieve individual trials and leaderboard for a tuning run."""
    run = get_tuning_run(run_id)
    return {
        "run_id": run_id,
        "best_trial_number": run.get("best_trial_number"),
        "best_score": run.get("best_score"),
        "trials": run.get("trials", []),
    }


@router.delete("/{run_id}")
def remove_tuning_run(
    run_id: str,
    delete_registered_model: bool = Query(default=False),
):
    """Delete a tuning run and optionally its registered model artifact."""
    return delete_tuning_run(
        run_id=run_id,
        delete_registered_model=delete_registered_model,
    )


@router.post("/{run_id}/cancel")
def stop_tuning_run(run_id: str):
    """Request early cancellation of an active tuning run."""
    return cancel_tuning_run(run_id)
