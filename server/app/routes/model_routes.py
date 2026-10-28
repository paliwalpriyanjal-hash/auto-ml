# app/routes/model_routes.py

import io
from typing import Optional, List, Union
import pandas as pd
from fastapi import APIRouter, Body, Query, File, UploadFile, Form, Request, HTTPException
from app.utils.security import safe_read_csv

from app.services.serving_service import execute_served_inference, serving_cache
from app.services.model_registry_service import (
    list_models,
    get_model,
    predict_with_model,
    delete_model,
)
from app.services.explainability_service import (
    get_explainability_capabilities,
    compute_permutation_importance,
    compute_shap_explanations,
)

router = APIRouter(prefix="/models", tags=["Model Registry"])


@router.get("")
def get_registered_models(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    task_type: Optional[str] = Query(default=None),
):
    return list_models(
        limit=limit,
        offset=offset,
        task_type=task_type,
    )


@router.get("/{model_id}")
def get_registered_model(model_id: int):
    return get_model(model_id)


@router.post("/{model_id}/predict")
def predict(
    model_id: int,
    request: Request,
    payload: dict = Body(
        ...,
        example={
            "rows": [
                {"feature_1": 10, "feature_2": "example"}
            ]
        },
    ),
):
    rows = payload.get("rows")

    if not isinstance(rows, list):
        raise HTTPException(
            status_code=400,
            detail="Request body must contain a 'rows' list.",
        )

    cid = getattr(request.state, "correlation_id", "unknown") if hasattr(request, "state") else "unknown"
    return execute_served_inference(model_id, rows, correlation_id=cid)


@router.delete("/{model_id}")
def remove_registered_model(model_id: int):
    serving_cache.evict(model_id)
    return delete_model(model_id)


# -------------------------------------------------------------
# Explainability Endpoints
# -------------------------------------------------------------

@router.get("/{model_id}/explainability/capabilities")
def explainability_capabilities(model_id: int):
    """
    Get explainability capabilities, supported methods, task type,
    feature metadata, and known limitations for a registered model.
    """
    return get_explainability_capabilities(model_id)


async def _parse_evaluation_dataset(
    request: Request,
    file: Optional[UploadFile],
    target_column_form: Optional[str],
    dataset_source_form: Optional[str],
):
    """Parse evaluation dataset from either multipart CSV upload or JSON body."""
    content_type = request.headers.get("content-type", "")

    if file is not None:
        df = await safe_read_csv(file)
        if len(set(df.columns)) != len(df.columns):
            raise HTTPException(
                status_code=400,
                detail="Evaluation CSV contains duplicate column headers.",
            )
        return df, target_column_form, dataset_source_form or "evaluation_data", {}

    if "application/json" in content_type:
        try:
            payload = await request.json()
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Invalid JSON payload.",
            )
        rows = payload.get("rows")
        if not rows or not isinstance(rows, list):
            raise HTTPException(
                status_code=400,
                detail="JSON payload must contain a non-empty 'rows' list of objects.",
            )
        df = pd.DataFrame(rows)
        if df.empty:
            raise HTTPException(
                status_code=400,
                detail="Supplied 'rows' list contains no valid data.",
            )
        target = payload.get("target_column") or target_column_form
        source = payload.get("dataset_source") or dataset_source_form or "evaluation_data"
        return df, target, source, payload

    raise HTTPException(
        status_code=400,
        detail="Please provide an evaluation dataset via CSV file upload or JSON payload.",
    )


@router.post("/{model_id}/explainability/permutation")
async def explain_permutation(
    model_id: int,
    request: Request,
    file: Optional[UploadFile] = File(None),
    target_column: Optional[str] = Form(None),
    n_repeats: int = Form(5),
    scoring: Optional[str] = Form(None),
    dataset_source: str = Form("evaluation_data"),
):
    """
    Calculate permutation feature importance for a registered model.
    Accepts evaluation dataset via multipart CSV upload or JSON body.
    """
    df, resolved_target, source, json_payload = await _parse_evaluation_dataset(
        request, file, target_column, dataset_source
    )
    repeats = json_payload.get("n_repeats", n_repeats)
    scorer = json_payload.get("scoring", scoring)

    return compute_permutation_importance(
        model_id=model_id,
        df=df,
        target_column=resolved_target,
        n_repeats=repeats,
        scoring=scorer,
        dataset_source=source,
    )


@router.post("/{model_id}/explainability/shap")
async def explain_shap(
    model_id: int,
    request: Request,
    file: Optional[UploadFile] = File(None),
    target_column: Optional[str] = Form(None),
    max_rows: int = Form(50),
    dataset_source: str = Form("evaluation_data"),
):
    """
    Generate SHAP feature attributions and waterfall contributions for a registered model.
    Accepts evaluation dataset via multipart CSV upload or JSON body.
    """
    df, resolved_target, source, json_payload = await _parse_evaluation_dataset(
        request, file, target_column, dataset_source
    )
    rows_limit = json_payload.get("max_rows", max_rows)
    row_indices = json_payload.get("row_indices")
    selected_class = json_payload.get("selected_class")

    return compute_shap_explanations(
        model_id=model_id,
        df=df,
        target_column=resolved_target,
        row_indices=row_indices,
        max_rows=rows_limit,
        selected_class=selected_class,
        dataset_source=source,
    )
