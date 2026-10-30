# app/routes/feature_routes.py

import io
import json
from typing import Optional, List, Any
import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from app.utils.security import safe_read_csv

from app.services.feature_service import (
    TabularFeatureEngineer,
    compare_feature_strategies,
    delete_feature_run,
    evaluate_feature_strategies,
    get_eda_feature_recommendations,
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
    eda_analysis_id: Optional[str] = Form(None),
):
    """
    Profile a tabular CSV dataset for column distributions, data quality alerts,
    collinear pairs, high-cardinality, and potential identifier columns.
    Enriched with Phase 2 EDA cross-referencing where eda_analysis_id is provided.
    """
    df = await safe_read_csv(file)

    return profile_dataset(
        df=df,
        target_column=target_column,
        correlation_threshold=correlation_threshold,
        eda_analysis_id=eda_analysis_id,
    )


@router.post("/recommendations")
async def recommendations_endpoint(
    file: UploadFile = File(...),
    target_column: Optional[str] = Form(None),
    eda_analysis_id: Optional[str] = Form(None),
):
    """
    Returns dataset-adaptive feature engineering recommendations and presets
    (Conservative, Balanced, Advanced) tailored to the 11 dataset profiles.
    """
    df = await safe_read_csv(file)
    return get_eda_feature_recommendations(
        df=df,
        target_column=target_column,
        eda_analysis_id=eda_analysis_id,
    )


@router.post("/preview")
async def preview_transformation_endpoint(
    file: UploadFile = File(...),
    datetime_cols: str = Form("[]"),
    cyclical_datetime: bool = Form(False),
    log_cols: str = Form("[]"),
    yeo_johnson_cols: str = Form("[]"),
    sqrt_cols: str = Form("[]"),
    clip_outliers: bool = Form(False),
    group_rare_categories: bool = Form(False),
    frequency_encode_cols: str = Form("[]"),
    interaction_pairs: str = Form("[]"),
    polynomial_cols: str = Form("[]"),
    text_cols: str = Form("[]"),
):
    """
    Preview feature transformations on top sample rows without mutating original data.
    """
    df = await safe_read_csv(file)

    try:
        parsed_dt = json.loads(datetime_cols) if isinstance(datetime_cols, str) else datetime_cols
        parsed_log = json.loads(log_cols) if isinstance(log_cols, str) else log_cols
        parsed_yeo = json.loads(yeo_johnson_cols) if isinstance(yeo_johnson_cols, str) else yeo_johnson_cols
        parsed_sqrt = json.loads(sqrt_cols) if isinstance(sqrt_cols, str) else sqrt_cols
        parsed_freq = json.loads(frequency_encode_cols) if isinstance(frequency_encode_cols, str) else frequency_encode_cols
        parsed_inter = json.loads(interaction_pairs) if isinstance(interaction_pairs, str) else interaction_pairs
        parsed_poly = json.loads(polynomial_cols) if isinstance(polynomial_cols, str) else polynomial_cols
        parsed_text = json.loads(text_cols) if isinstance(text_cols, str) else text_cols
    except Exception:
        parsed_dt, parsed_log, parsed_yeo, parsed_sqrt = [], [], [], []
        parsed_freq, parsed_inter, parsed_poly, parsed_text = [], [], [], []

    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if c not in num_cols]

    engineer = TabularFeatureEngineer(
        datetime_cols=parsed_dt,
        drop_original_datetime=True,
        cyclical_datetime=cyclical_datetime,
        log_cols=parsed_log,
        yeo_johnson_cols=parsed_yeo,
        sqrt_cols=parsed_sqrt,
        clip_outliers=clip_outliers,
        outlier_cols=num_cols,
        group_rare_categories=group_rare_categories,
        rare_cat_cols=cat_cols,
        frequency_encode_cols=parsed_freq,
        interaction_pairs=parsed_inter,
        polynomial_cols=parsed_poly,
        text_cols=parsed_text,
    )

    sample_size = min(len(df), 20)
    transformed = engineer.fit_transform(df.head(sample_size))
    sample_records = transformed.head(5).to_dict(orient="records")

    return {
        "original_columns": list(df.columns),
        "transformed_columns": list(transformed.columns),
        "created_columns": [c for c in transformed.columns if c not in df.columns],
        "dropped_columns": [c for c in df.columns if c not in transformed.columns],
        "sample_preview": sample_records,
        "feature_lineage": engineer.get_lineage(),
    }


@router.post("/compare-strategies")
async def compare_strategies_endpoint(
    file: UploadFile = File(...),
    task_type: str = Form(...),
    target_column: str = Form(...),
    model_type: str = Form(...),
    ranking_metric: Optional[str] = Form(None),
    n_splits: int = Form(3),
    split_ratio: float = Form(0.8),
    eda_analysis_id: Optional[str] = Form(None),
):
    """
    Benchmark Original Features vs Conservative vs Balanced vs Advanced presets
    under strictly identical CV splits. Evaluates winning preset on untouched holdout.
    """
    return await compare_feature_strategies(
        file=file,
        task_type=task_type,
        target_column=target_column,
        model_type=model_type,
        ranking_metric=ranking_metric,
        n_splits=n_splits,
        split_ratio=split_ratio,
        eda_analysis_id=eda_analysis_id,
    )


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
    cyclical_datetime: bool = Form(False),
    log_cols: str = Form("[]"),
    yeo_johnson_cols: str = Form("[]"),
    sqrt_cols: str = Form("[]"),
    clip_outliers: bool = Form(False),
    group_rare_categories: bool = Form(False),
    frequency_encode_cols: str = Form("[]"),
    interaction_pairs: str = Form("[]"),
    polynomial_cols: str = Form("[]"),
    target_encode_cols: str = Form("[]"),
    text_cols: str = Form("[]"),
    selection_method: str = Form("none"),
    variance_threshold: float = Form(0.01),
    correlation_threshold: float = Form(0.85),
    k_best_count: int = Form(10),
    eda_analysis_id: Optional[str] = Form(None),
):
    """
    Compare baseline vs feature-engineered vs feature-selected pipelines using
    training-only CV and holdout validation. Registers winning model artifact.
    """
    try:
        parsed_dt = json.loads(datetime_cols) if isinstance(datetime_cols, str) else []
        parsed_log = json.loads(log_cols) if isinstance(log_cols, str) else []
        parsed_yeo = json.loads(yeo_johnson_cols) if isinstance(yeo_johnson_cols, str) else []
        parsed_sqrt = json.loads(sqrt_cols) if isinstance(sqrt_cols, str) else []
        parsed_freq = json.loads(frequency_encode_cols) if isinstance(frequency_encode_cols, str) else []
        parsed_inter = json.loads(interaction_pairs) if isinstance(interaction_pairs, str) else []
        parsed_poly = json.loads(polynomial_cols) if isinstance(polynomial_cols, str) else []
        parsed_te = json.loads(target_encode_cols) if isinstance(target_encode_cols, str) else []
        parsed_text = json.loads(text_cols) if isinstance(text_cols, str) else []
    except Exception:
        parsed_dt, parsed_log, parsed_yeo, parsed_sqrt = [], [], [], []
        parsed_freq, parsed_inter, parsed_poly, parsed_te, parsed_text = [], [], [], [], []

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
        cyclical_datetime=cyclical_datetime,
        log_cols=parsed_log,
        yeo_johnson_cols=parsed_yeo,
        sqrt_cols=parsed_sqrt,
        clip_outliers=clip_outliers,
        group_rare_categories=group_rare_categories,
        frequency_encode_cols=parsed_freq,
        interaction_pairs=parsed_inter,
        polynomial_cols=parsed_poly,
        target_encode_cols=parsed_te,
        text_cols=parsed_text,
        selection_method=selection_method,
        variance_threshold=variance_threshold,
        correlation_threshold=correlation_threshold,
        k_best_count=k_best_count,
        eda_analysis_id=eda_analysis_id,
    )


@router.get("/lineage/{run_id}")
def get_feature_lineage_endpoint(run_id: str):
    """Retrieve full feature lineage graph mapping raw columns to transformed features."""
    record = get_feature_run(run_id)
    return {
        "run_id": run_id,
        "dataset_name": record.get("dataset_name"),
        "winning_strategy": record.get("winning_strategy"),
        "feature_lineage": record.get("feature_lineage", []),
        "selected_features": record.get("selected_features", []),
        "removed_features": record.get("removed_features", []),
    }


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
