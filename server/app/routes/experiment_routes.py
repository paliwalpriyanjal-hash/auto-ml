
from typing import Optional

from fastapi import APIRouter, Query

from app.services.experiment_service import (
    list_experiments,
    get_experiment,
    delete_experiment,
)

router = APIRouter(
    prefix="/experiments",
    tags=["Experiments"],
)


@router.get("")
def get_experiments(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    task_type: Optional[str] = Query(default=None),
):
    return list_experiments(
        limit=limit,
        offset=offset,
        task_type=task_type,
    )


@router.get("/{experiment_id}")
def get_experiment_details(experiment_id: int):
    return get_experiment(experiment_id)


@router.delete("/{experiment_id}")
def remove_experiment(experiment_id: int):
    return delete_experiment(experiment_id)
