# app/routes/feature_routes.py

import io
import json
from typing import Optional, List
import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from app.utils.security import safe_read_csv

from app.services.feature_service import (
    TabularFeatureEngineer,
    delete_feature_run,
    evaluate_feature_strategies,
    get_feature_run,
    list_feature_runs,
    profile_dataset,
)

router = APIRouter(prefix="/features", tags=["Feature Engineering & Selection"])


@router.post("/profile")
async def profile_dataset_endpoint(
    file: UploadFile = File(...),
    target_column: Optional[str] = Form(None),
    correlation_threshold: float = Form(0.85),
):
    """
    Profile a tabular CSV dataset for column distributions, data quality alerts,
    collinear pairs, high-cardinality, and potential identifier columns.
    """
    df = await safe_read_csv(file)

    return profile_dataset(
        df=df,
        target_column=target_column,
        correlation_threshold=correlation_threshold,
    )


@router.post("/preview")
async def preview_transformation_endpoint(
    file: UploadFile = File(...),
    datetime_cols: str = Form("[]"),
    log_cols: str = Form("[]"),
    clip_outliers: bool = Form(False),
    group_rare_categories: bool = Form(False),
):
    """
    Preview feature transformations on top sample rows without mutating original data.
    """
    df = await safe_read_csv(file)

    try:
        parsed_dt = json.loads(datetime_cols) if isinstance(datetime_cols, str) else datetime_cols
        parsed_log = json.loads(log_cols) if isinstance(log_cols, str) else log_cols
    except Exception:
        parsed_dt, parsed_log = [], []

    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if c not in num_cols]

    engineer = TabularFeatureEngineer(
        datetime_cols=parsed_dt,
        drop_original_datetime=True,
        log_cols=parsed_log,
        clip_outliers=clip_outliers,
        outlier_cols=num_cols,
        group_rare_categories=group_rare_categories,
        rare_cat_cols=cat_cols,
    )

    transformed = engineer.fit_transform(df.head(10))
    sample_records = transformed.head(5).to_dict(orient="records")

    return {
        "original_columns": list(df.columns),
        "transformed_columns": list(transformed.columns),
        "created_columns": [c for c in transformed.columns if c not in df.columns],
        "dropped_columns": [c for c in df.columns if c not in transformed.columns],
        "sample_preview": sample_records,
    }


@router.post("/evaluate")
async def evaluate_features_endpoint(
    file: UploadFile = File(...),
    task_type: str = Form(...),
    target_column: str = Form(...),
    model_type: str = Form(...),
    ranking_metric: Optional[str] = Form(None),
    n_splits: int = Form(3),
    split_ratio: float = Form(0.8),
    scaler: str = Form("standard"),
    missing: str = Form("median"),
    datetime_cols: str = Form("[]"),
    log_cols: str = Form("[]"),
    clip_outliers: bool = Form(False),
    group_rare_categories: bool = Form(False),
    selection_method: str = Form("none"),
    variance_threshold: float = Form(0.01),
    correlation_threshold: float = Form(0.85),
    k_best_count: int = Form(10),
):
    """
    Compare baseline vs feature-engineered vs feature-selected pipelines using
    training-only CV and holdout validation. Registers winning model artifact.
    """
    try:
        parsed_dt = json.loads(datetime_cols) if isinstance(datetime_cols, str) else []
        parsed_log = json.loads(log_cols) if isinstance(log_cols, str) else []
    except Exception:
        parsed_dt, parsed_log = [], []

    return await evaluate_feature_strategies(
        file=file,
        task_type=task_type,
        target_column=target_column,
        model_type=model_type,
        ranking_metric=ranking_metric,
        n_splits=n_splits,
        split_ratio=split_ratio,
        scaler=scaler,
        missing=missing,
        datetime_cols=parsed_dt,
        log_cols=parsed_log,
        clip_outliers=clip_outliers,
        group_rare_categories=group_rare_categories,
        selection_method=selection_method,
        variance_threshold=variance_threshold,
        correlation_threshold=correlation_threshold,
        k_best_count=k_best_count,
    )


@router.get("/history")
def get_feature_history(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """List historical feature engineering runs."""
    return list_feature_runs(limit=limit, offset=offset)


@router.get("/history/{run_id}")
def get_feature_run_detail(run_id: str):
    """Retrieve full details of a specific feature engineering run."""
    return get_feature_run(run_id)


@router.delete("/history/{run_id}")
def delete_feature_run_endpoint(run_id: str):
    """Delete a feature engineering run record."""
    return delete_feature_run(run_id)
