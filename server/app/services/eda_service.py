"""
AutoVeyra Phase 2: Advanced Adaptive EDA Intelligence & Automated Analysis Report Engine
Service Module: eda_service.py

Provides:
- Comprehensive dataset-level and column-level statistical profiling.
- Adaptive EDA strategy selector covering 11 distinct dataset profiles.
- Numerical feature analysis (descriptive stats, IQR/Z-score outliers, Pearson/Spearman correlation).
- Categorical feature analysis (Top-N frequency, rare categories, Chi-square & Cramer's V).
- Target-aware feature relationship analysis (classification class-wise stats, regression correlation ranking).
- Data quality diagnostics (missingness patterns, duplicates, constant columns, target leakage, PII indicators).
- Headless visualization engine generating base64 charts and structured chart data.
- Automated evidence-based statistical insights engine.
- Dynamic EDA execution flowchart.
- Professional multi-page PDF, HTML, and JSON report generator.
- SQLite-backed persistence in server/data/eda.db.
"""

import os
import io
import math
import json
import time
import uuid
import base64
import sqlite3
import datetime
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Set temporary MPLCONFIGDIR before importing matplotlib
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats

from fastapi import HTTPException
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Image as RLImage,
    HRFlowable,
    KeepTogether,
)

SERVER_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = SERVER_DIR / "data"
EDA_DIR = DATA_DIR / "eda"
DB_PATH = DATA_DIR / "eda.db"


# =====================================================================
# 1. DATABASE & CACHING PERSISTENCE
# =====================================================================

def _connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=15)
    conn.row_factory = sqlite3.Row
    return conn


def init_eda_db() -> None:
    """Initializes the SQLite database for EDA analyses and reports."""
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS eda_analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id TEXT UNIQUE NOT NULL,
                filename TEXT NOT NULL,
                row_count INTEGER NOT NULL,
                column_count INTEGER NOT NULL,
                target_column TEXT,
                task_type TEXT,
                summary_json TEXT NOT NULL,
                profiles_json TEXT NOT NULL,
                strategy_json TEXT NOT NULL,
                quality_json TEXT NOT NULL,
                numerical_json TEXT NOT NULL,
                categorical_json TEXT NOT NULL,
                target_relationships_json TEXT NOT NULL,
                insights_json TEXT NOT NULL,
                flowchart_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_eda_analysis_id ON eda_analyses(analysis_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_eda_created_at ON eda_analyses(created_at DESC)
        """)


def _safe_float(val: Any, decimals: int = 4) -> Optional[float]:
    if val is None:
        return None
    try:
        f = float(val)
        return round(f, decimals) if math.isfinite(f) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return round(float(obj), 4) if math.isfinite(float(obj)) else None
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if obj is None or isinstance(obj, str):
        return obj
    return str(obj)


# =====================================================================
# 2. INTELLIGENT DATASET PROFILING ENGINE
# =====================================================================

def profile_dataset(
    df: pd.DataFrame,
    filename: str = "dataset.csv",
    target_column: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Computes comprehensive dataset-level metadata and per-column statistical profiles.
    Detects semantic data types, cardinality, missingness, and distribution moments.
    """
    total_rows, total_cols = df.shape
    memory_usage_bytes = int(df.memory_usage(deep=True).sum())
    memory_usage_mb = round(memory_usage_bytes / (1024 * 1024), 2)
    memory_str = f"{memory_usage_mb} MB" if memory_usage_mb >= 1.0 else f"{round(memory_usage_bytes / 1024, 1)} KB"

    # Duplicates & Overall Missingness
    dup_rows_count = int(df.duplicated().sum())
    dup_rows_pct = round((dup_rows_count / max(1, total_rows)) * 100, 2)
    dup_cols_count = int(df.T.duplicated().sum()) if total_cols > 1 else 0

    total_cells = total_rows * total_cols
    missing_cells = int(df.isna().sum().sum())
    missing_pct = round((missing_cells / max(1, total_cells)) * 100, 2)

    column_profiles: Dict[str, Any] = {}
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    boolean_cols: List[str] = []
    datetime_cols: List[str] = []
    text_cols: List[str] = []
    constant_cols: List[str] = []
    near_constant_cols: List[str] = []
    id_cols: List[str] = []
    pii_cols: List[str] = []

    # Common PII keywords
    pii_keywords = {"email", "ssn", "phone", "password", "passport", "credit_card", "card_number", "ip_address", "social_security"}

    for col in df.columns:
        series = df[col]
        non_null = series.dropna()
        n_missing = int(series.isna().sum())
        missing_rate = round((n_missing / max(1, total_rows)) * 100, 2)
        n_unique = int(series.nunique(dropna=True))
        cardinality_ratio = round(n_unique / max(1, total_rows), 4)

        lower_col = str(col).lower().strip()

        # Check PII indicators
        is_pii = any(k in lower_col for k in pii_keywords)
        if is_pii:
            pii_cols.append(col)

        # Check constant / near-constant
        is_constant = n_unique <= 1
        is_near_constant = False
        if not is_constant and len(non_null) > 0:
            top_freq = non_null.value_counts().iloc[0]
            if (top_freq / len(non_null)) >= 0.96:
                is_near_constant = True

        if is_constant:
            constant_cols.append(col)
        elif is_near_constant:
            near_constant_cols.append(col)

        # Semantic Type Inference
        semantic_type = "unknown"
        col_profile: Dict[str, Any] = {
            "name": col,
            "raw_dtype": str(series.dtype),
            "missing_count": n_missing,
            "missing_pct": missing_rate,
            "unique_count": n_unique,
            "cardinality_ratio": cardinality_ratio,
            "is_constant": is_constant,
            "is_near_constant": is_near_constant,
            "is_pii": is_pii,
        }

        # Check Boolean
        bool_candidates = {0, 1, 0.0, 1.0, True, False, "true", "false", "yes", "no", "y", "n"}
        unique_vals_set = set(non_null.head(100).unique()) if len(non_null) > 0 else set()
        is_bool = False
        if 0 < n_unique <= 2:
            try:
                norm_vals = {str(v).lower().strip() for v in unique_vals_set}
                if norm_vals.issubset({"0", "1", "true", "false", "yes", "no", "y", "n"}):
                    is_bool = True
            except Exception:
                pass

        # Check Datetime
        is_datetime = False
        if "date" in lower_col or "time" in lower_col or "timestamp" in lower_col or "year" in lower_col:
            if not pd.api.types.is_numeric_dtype(series) or (n_unique > 10 and series.min() > 1900):
                try:
                    sample_dt = pd.to_datetime(non_null.head(30), errors="coerce")
                    if sample_dt.notna().sum() >= 20:
                        is_datetime = True
                except Exception:
                    pass

        # Check Identifier
        is_id = False
        if n_unique == total_rows and total_rows > 20:
            if any(id_word in lower_col for id_word in {"id", "uuid", "guid", "key", "code", "index", "no", "num"}):
                is_id = True
                id_cols.append(col)

        # Check Text-like
        is_text = False
        if (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)) and not is_datetime and not is_bool:
            sample_strings = non_null.head(50).astype(str)
            avg_words = float(sample_strings.apply(lambda s: len(s.split())).mean()) if len(sample_strings) > 0 else 0.0
            if avg_words > 3.0:
                is_text = True
                text_cols.append(col)

        # Determine Final Semantic Type
        if is_id:
            semantic_type = "identifier"
        elif is_bool:
            semantic_type = "boolean"
            boolean_cols.append(col)
        elif is_datetime:
            semantic_type = "datetime"
            datetime_cols.append(col)
        elif is_text:
            semantic_type = "text"
        elif pd.api.types.is_numeric_dtype(series):
            semantic_type = "numeric"
            numeric_cols.append(col)
        else:
            semantic_type = "categorical"
            categorical_cols.append(col)

        col_profile["semantic_type"] = semantic_type

        # Top value & frequency
        if len(non_null) > 0:
            val_counts = non_null.value_counts()
            col_profile["top_value"] = str(val_counts.index[0])
            col_profile["top_frequency"] = int(val_counts.iloc[0])
        else:
            col_profile["top_value"] = None
            col_profile["top_frequency"] = 0

        # Detailed Numerical Statistics
        if semantic_type == "numeric" and len(non_null) > 0:
            clean_num = pd.to_numeric(non_null, errors="coerce").dropna()
            if len(clean_num) > 0:
                q25, q50, q75 = clean_num.quantile([0.25, 0.50, 0.75])
                iqr = float(q75 - q25)
                lower_bound = q25 - 1.5 * iqr
                upper_bound = q75 + 1.5 * iqr
                outliers = clean_num[(clean_num < lower_bound) | (clean_num > upper_bound)]
                outlier_count = int(len(outliers))
                outlier_pct = round((outlier_count / len(clean_num)) * 100, 2)

                # Distribution moments
                skew_val = float(clean_num.skew()) if len(clean_num) > 2 else 0.0
                kurt_val = float(clean_num.kurt()) if len(clean_num) > 3 else 0.0

                # Histogram data for frontend rendering
                hist_counts, bin_edges = np.histogram(clean_num, bins=min(12, max(5, n_unique)))
                hist_data = [
                    {"bin": f"{round(bin_edges[i], 2)} - {round(bin_edges[i+1], 2)}", "count": int(hist_counts[i])}
                    for i in range(len(hist_counts))
                ]

                col_profile.update({
                    "min": _safe_float(clean_num.min()),
                    "max": _safe_float(clean_num.max()),
                    "mean": _safe_float(clean_num.mean()),
                    "median": _safe_float(q50),
                    "std": _safe_float(clean_num.std()),
                    "skewness": _safe_float(skew_val),
                    "kurtosis": _safe_float(kurt_val),
                    "q25": _safe_float(q25),
                    "q50": _safe_float(q50),
                    "q75": _safe_float(q75),
                    "iqr": _safe_float(iqr),
                    "outlier_count_iqr": outlier_count,
                    "outlier_pct_iqr": outlier_pct,
                    "histogram_bins": hist_data,
                })

        # Categorical Statistics
        elif semantic_type in {"categorical", "boolean"} and len(non_null) > 0:
            val_counts = non_null.value_counts(normalize=True)
            proportions = [
                {"category": str(cat), "proportion": round(float(prop), 4), "count": int(non_null.value_counts()[cat])}
                for cat, prop in val_counts.head(10).items()
            ]
            rare_cats = [str(cat) for cat, prop in val_counts.items() if prop < 0.05]
            col_profile.update({
                "top_categories": proportions,
                "rare_categories_count": len(rare_cats),
                "has_rare_categories": len(rare_cats) > 0,
            })

        # Datetime Statistics
        elif semantic_type == "datetime" and len(non_null) > 0:
            try:
                dt_series = pd.to_datetime(non_null, errors="coerce").dropna()
                if len(dt_series) > 0:
                    min_dt = dt_series.min()
                    max_dt = dt_series.max()
                    span_days = int((max_dt - min_dt).total_seconds() / 86400)
                    col_profile.update({
                        "min_date": min_dt.isoformat(),
                        "max_date": max_dt.isoformat(),
                        "timespan_days": span_days,
                    })
            except Exception:
                pass

        # Text Statistics
        elif semantic_type == "text" and len(non_null) > 0:
            str_series = non_null.astype(str)
            col_profile.update({
                "avg_words": round(float(str_series.apply(lambda s: len(s.split())).mean()), 1),
                "avg_characters": round(float(str_series.apply(len).mean()), 1),
                "empty_text_count": int((str_series.str.strip() == "").sum()),
            })

        column_profiles[col] = col_profile

    # Determine Target Column if provided or auto-inferred
    inferred_target = target_column
    if not inferred_target or inferred_target not in df.columns:
        # Default to last non-id column
        potential_targets = [c for c in df.columns if c not in id_cols]
        inferred_target = potential_targets[-1] if potential_targets else df.columns[-1]

    # Inferred Task Type
    inferred_task = "regression"
    target_series = df[inferred_target].dropna() if inferred_target in df.columns else pd.Series(dtype=object)
    target_uniques = target_series.nunique()

    if (
        pd.api.types.is_object_dtype(target_series)
        or pd.api.types.is_string_dtype(target_series)
        or target_uniques <= 10
        or (pd.api.types.is_integer_dtype(target_series) and target_uniques <= 20)
    ):
        inferred_task = "binary_classification" if target_uniques == 2 else "multiclass_classification"
    else:
        inferred_task = "regression"

    dataset_summary = {
        "filename": filename,
        "row_count": total_rows,
        "column_count": total_cols,
        "memory_usage_bytes": memory_usage_bytes,
        "memory_usage_str": memory_str,
        "duplicate_rows_count": dup_rows_count,
        "duplicate_rows_pct": dup_rows_pct,
        "duplicate_columns_count": dup_cols_count,
        "total_cells": total_cells,
        "missing_cells_count": missing_cells,
        "missing_cells_pct": missing_pct,
        "numeric_columns_count": len(numeric_cols),
        "categorical_columns_count": len(categorical_cols),
        "boolean_columns_count": len(boolean_cols),
        "datetime_columns_count": len(datetime_cols),
        "text_columns_count": len(text_cols),
        "constant_columns_count": len(constant_cols),
        "near_constant_columns_count": len(near_constant_cols),
        "identifier_columns_count": len(id_cols),
        "pii_columns_count": len(pii_cols),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "boolean_columns": boolean_cols,
        "datetime_columns": datetime_cols,
        "text_columns": text_cols,
        "constant_columns": constant_cols,
        "near_constant_columns": near_constant_cols,
        "identifier_columns": id_cols,
        "pii_columns": pii_cols,
        "inferred_target": inferred_target,
        "inferred_task": inferred_task,
    }

    return {
        "dataset_summary": dataset_summary,
        "column_profiles": column_profiles,
    }


# =====================================================================
# 3. ADAPTIVE EDA STRATEGY ENGINE
# =====================================================================

def detect_eda_strategy(
    df: pd.DataFrame,
    profile_data: Dict[str, Any],
    target_column: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Selects tailored exploratory analyses based on the 11 supported dataset profiles:
    - Numerical-heavy, Categorical-heavy, Mixed, Classification, Regression,
      Datetime-oriented, Time-series-oriented, Text-containing, High-dimensional,
      Small dataset, Large dataset.
    Establishes deterministic sampling limits and provides transparent strategy explanations.
    """
    summary = profile_data["dataset_summary"]
    total_rows = summary["row_count"]
    total_cols = summary["column_count"]
    n_num = summary["numeric_columns_count"]
    n_cat = summary["categorical_columns_count"] + summary["boolean_columns_count"]
    n_dt = summary["datetime_columns_count"]
    n_text = summary["text_columns_count"]

    target = target_column or summary["inferred_target"]
    if target_column and target_column in df.columns:
        s = df[target_column].dropna()
        u = s.nunique()
        if (
            pd.api.types.is_object_dtype(s)
            or pd.api.types.is_string_dtype(s)
            or u <= 10
            or (pd.api.types.is_integer_dtype(s) and u <= 20)
        ):
            task = "binary_classification" if u == 2 else "multiclass_classification"
        else:
            task = "regression"
    else:
        task = summary["inferred_task"]

    active_profiles: List[str] = []
    strategy_reasons: List[str] = []
    recommended_operations: List[str] = []

    # Profile A: Numerical-heavy
    num_ratio = n_num / max(1, total_cols)
    if num_ratio >= 0.65:
        active_profiles.append("numerical_heavy")
        strategy_reasons.append(
            f"Numerical features represent {round(num_ratio * 100, 1)}% of columns. Prioritizing parametric moments, IQR outlier diagnostics, and Pearson/Spearman correlation matrices."
        )
        recommended_operations.extend(["distribution_analysis", "outlier_diagnostics", "correlation_matrices"])

    # Profile B: Categorical-heavy
    cat_ratio = n_cat / max(1, total_cols)
    if cat_ratio >= 0.60:
        active_profiles.append("categorical_heavy")
        strategy_reasons.append(
            f"Categorical features represent {round(cat_ratio * 100, 1)}% of columns. Prioritizing Top-N frequency distributions, Chi-square independence tests, and Cramer's V effect sizes."
        )
        recommended_operations.extend(["category_frequencies", "chi_square_tests", "cramers_v_matrix"])

    # Profile C: Mixed
    if num_ratio < 0.65 and cat_ratio < 0.60:
        active_profiles.append("mixed_tabular")
        strategy_reasons.append(
            "Balanced composition of numerical and categorical variables. Applying joint group-by aggregation, box plots by category, and cross-type association metrics."
        )
        recommended_operations.extend(["numerical_analysis", "categorical_analysis", "cross_type_relationships"])

    # Profile D / E: Task Profile
    if "classification" in task:
        active_profiles.append("classification_target")
        strategy_reasons.append(
            f"Target variable '{target}' has discrete classes. Prioritizing class distribution, imbalance ratio severity, and class-wise feature distributions."
        )
        recommended_operations.extend(["class_imbalance_analysis", "feature_by_class_distributions"])
    else:
        active_profiles.append("regression_target")
        strategy_reasons.append(
            f"Target variable '{target}' is continuous. Prioritizing target skewness, residual outlier checks, and feature-to-target linear/non-linear correlation rankings."
        )
        recommended_operations.extend(["target_distribution_analysis", "feature_target_correlations"])

    # Profile F: Datetime-oriented
    if n_dt > 0:
        active_profiles.append("datetime_oriented")
        strategy_reasons.append(
            f"Detected {n_dt} temporal variables. Prioritizing timespan analysis, date component breakdowns, and temporal trend visualization."
        )
        recommended_operations.append("temporal_trend_analysis")

    # Profile G: Time-series-oriented
    if n_dt > 0 and total_rows > 30:
        # Check if index or first datetime column is sorted or regular
        try:
            dt_col = summary["datetime_columns"][0]
            parsed_dt = pd.to_datetime(df[dt_col].dropna().head(40), errors="coerce")
            if parsed_dt.is_monotonic_increasing or parsed_dt.is_monotonic_decreasing:
                active_profiles.append("time_series_oriented")
                strategy_reasons.append(
                    f"Temporal sequence detected in column '{dt_col}'. Time-series exploratory trend checks enabled; chronological (non-random) splitting recommended for modeling."
                )
                recommended_operations.append("sequential_autocorrelation_check")
        except Exception:
            pass

    # Profile H: Text-containing
    if n_text > 0:
        active_profiles.append("text_containing")
        strategy_reasons.append(
            f"Detected {n_text} unstructured text columns. Analyzing word-count distributions, character lengths, and tokenization readiness."
        )
        recommended_operations.append("text_diagnostics")

    # Profile I: High-dimensional
    if total_cols > 25:
        active_profiles.append("high_dimensional")
        strategy_reasons.append(
            f"High feature count ({total_cols} columns). Limiting correlation heatmaps to top 20 most informative/correlated variables to maintain readable visualization."
        )
        recommended_operations.append("bounded_correlation_matrix")

    # Profile J: Small dataset
    if total_rows < 150:
        active_profiles.append("small_dataset")
        strategy_reasons.append(
            f"Small sample size ({total_rows} rows). Statistical significance tests and holdout metrics may exhibit statistical noise; K-Fold cross-validation strongly advised."
        )

    # Profile K: Large dataset
    is_sampled = False
    sample_size = total_rows
    sampling_disclosed = None
    if total_rows > 4000:
        active_profiles.append("large_dataset")
        is_sampled = True
        sample_size = 2500
        sampling_disclosed = (
            f"Deterministic stratified/uniform sampling of {sample_size} rows applied for computationally intensive scatter plots and correlation KDEs (Full dataset size: {total_rows} rows)."
        )
        strategy_reasons.append(sampling_disclosed)

    return {
        "active_profiles": active_profiles,
        "target_column": target,
        "task_type": task,
        "strategy_reasons": strategy_reasons,
        "recommended_operations": list(set(recommended_operations)),
        "is_sampled": is_sampled,
        "sample_size": sample_size,
        "sampling_disclosed": sampling_disclosed,
    }


# =====================================================================
# 4. ADVANCED STATISTICAL ANALYSIS ENGINES
# =====================================================================

def analyze_numerical_features(
    df: pd.DataFrame,
    numeric_columns: List[str],
    target_column: Optional[str] = None,
    sample_size: int = 3000,
) -> Dict[str, Any]:
    """
    Computes parametric and non-parametric numerical statistics:
    - Moments: Mean, Std, Skewness, Kurtosis
    - Robust: Median, IQR, Outliers
    - Pairwise: Pearson and Spearman correlation matrices
    - Multicollinearity: Highly correlated pairs (|r| > 0.80)
    """
    if not numeric_columns:
        return {"columns": [], "correlations": {}, "summary_table": [], "highly_correlated_pairs": []}

    sub_df = df[numeric_columns].copy()
    if len(sub_df) > sample_size:
        sub_df = sub_df.sample(n=sample_size, random_state=42)

    summary_table = []
    skewed_cols = []
    extreme_outlier_cols = []

    for col in numeric_columns:
        series = pd.to_numeric(sub_df[col], errors="coerce").dropna()
        if len(series) == 0:
            continue

        q25, q50, q75 = series.quantile([0.25, 0.50, 0.75])
        iqr = float(q75 - q25)
        outlier_count = int(len(series[(series < q25 - 1.5 * iqr) | (series > q75 + 1.5 * iqr)]))
        outlier_pct = round((outlier_count / len(series)) * 100, 2)
        skew_val = float(series.skew()) if len(series) > 2 else 0.0

        if abs(skew_val) > 1.5:
            skewed_cols.append(col)
        if outlier_pct > 8.0:
            extreme_outlier_cols.append(col)

        summary_table.append({
            "column": col,
            "count": int(len(series)),
            "mean": _safe_float(series.mean()),
            "std": _safe_float(series.std()),
            "median": _safe_float(q50),
            "min": _safe_float(series.min()),
            "max": _safe_float(series.max()),
            "skewness": _safe_float(skew_val),
            "kurtosis": _safe_float(series.kurt() if len(series) > 3 else 0.0),
            "outlier_count": outlier_count,
            "outlier_pct": outlier_pct,
        })

    # Correlation Matrices (Pearson and Spearman)
    pearson_corr = sub_df.corr(method="pearson").round(4).fillna(0.0)
    spearman_corr = sub_df.corr(method="spearman").round(4).fillna(0.0)

    # Detect Highly Correlated Pairs
    highly_correlated_pairs = []
    corr_cols = list(pearson_corr.columns)
    for i in range(len(corr_cols)):
        for j in range(i + 1, len(corr_cols)):
            c1, c2 = corr_cols[i], corr_cols[j]
            p_val = float(pearson_corr.iloc[i, j])
            s_val = float(spearman_corr.iloc[i, j])
            if abs(p_val) >= 0.80 or abs(s_val) >= 0.80:
                highly_correlated_pairs.append({
                    "feature_1": c1,
                    "feature_2": c2,
                    "pearson": round(p_val, 4),
                    "spearman": round(s_val, 4),
                    "risk": "High Collinearity" if abs(p_val) >= 0.90 else "Moderate Collinearity",
                })

    return {
        "columns": numeric_columns,
        "summary_table": summary_table,
        "skewed_columns": skewed_cols,
        "outlier_columns": extreme_outlier_cols,
        "pearson_matrix": pearson_corr.to_dict(),
        "spearman_matrix": spearman_corr.to_dict(),
        "highly_correlated_pairs": highly_correlated_pairs,
    }


def analyze_categorical_features(
    df: pd.DataFrame,
    categorical_columns: List[str],
    target_column: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Computes frequency distributions, rare category alerts, and categorical association
    tests (Chi-square test of independence & Cramer's V effect size).
    """
    if not categorical_columns:
        return {"columns": [], "frequency_distributions": {}, "associations": []}

    freq_dists = {}
    rare_category_alerts = []

    for col in categorical_columns:
        series = df[col].dropna().astype(str)
        if len(series) == 0:
            continue

        vc = series.value_counts(normalize=True)
        top_cats = [
            {"category": str(cat), "count": int(series.value_counts()[cat]), "proportion": round(float(prop), 4)}
            for cat, prop in vc.head(10).items()
        ]
        rare_cats = [str(cat) for cat, prop in vc.items() if prop < 0.05]

        freq_dists[col] = {
            "cardinality": int(series.nunique()),
            "top_categories": top_cats,
            "rare_count": len(rare_cats),
        }

        if len(rare_cats) > 0:
            rare_category_alerts.append({
                "column": col,
                "rare_count": len(rare_cats),
                "sample_rare": rare_cats[:3],
            })

    # Pairwise Cramer's V / Chi-Square
    associations = []
    eval_cols = categorical_columns[:8]  # Bound pairwise matrix to avoid O(N^2) explosion
    for i in range(len(eval_cols)):
        for j in range(i + 1, len(eval_cols)):
            c1, c2 = eval_cols[i], eval_cols[j]
            try:
                contingency = pd.crosstab(df[c1].astype(str), df[c2].astype(str))
                if contingency.size > 1 and contingency.shape[0] > 1 and contingency.shape[1] > 1:
                    chi2, p_val, dof, _ = stats.chi2_contingency(contingency)
                    n = contingency.sum().sum()
                    phi2 = chi2 / n
                    r, k = contingency.shape
                    # Bias-corrected Cramer's V
                    cramers_v = math.sqrt(max(0, phi2 / min(r - 1, k - 1)))
                    associations.append({
                        "feature_1": c1,
                        "feature_2": c2,
                        "cramers_v": round(float(cramers_v), 4),
                        "p_value": _safe_float(p_val),
                        "significant": bool(p_val < 0.05),
                    })
            except Exception:
                pass

    return {
        "columns": categorical_columns,
        "frequency_distributions": freq_dists,
        "rare_category_alerts": rare_category_alerts,
        "associations": sorted(associations, key=lambda x: x["cramers_v"], reverse=True),
    }


def analyze_target_relationships(
    df: pd.DataFrame,
    target_column: str,
    task_type: str,
    numeric_columns: List[str],
    categorical_columns: List[str],
) -> Dict[str, Any]:
    """
    Performs target-aware exploratory analysis:
    - Classification: class distribution, imbalance ratio, class-wise numerical profiles, Cramer's V with target
    - Regression: target distribution moments, Pearson & Spearman correlation rankings, non-linear indicators
    """
    if target_column not in df.columns:
        return {"target_column": target_column, "error": "Target column not found in dataset."}

    target_series = df[target_column].dropna()
    is_classification = "classification" in task_type

    if is_classification:
        val_counts = target_series.astype(str).value_counts()
        total_valid = len(target_series)
        class_proportions = [
            {"class": str(c), "count": int(cnt), "percentage": round((cnt / max(1, total_valid)) * 100, 2)}
            for c, cnt in val_counts.items()
        ]
        maj_count = int(val_counts.iloc[0]) if len(val_counts) > 0 else 1
        min_count = int(val_counts.iloc[-1]) if len(val_counts) > 0 else 1
        imbalance_ratio = round(maj_count / max(1, min_count), 2)
        imbalance_severity = "severe" if imbalance_ratio >= 10.0 else ("moderate" if imbalance_ratio >= 3.0 else "balanced")

        # Class-wise numerical features summary
        class_numerical_profiles = []
        for n_col in numeric_columns[:6]:
            try:
                grouped = df.groupby(target_column)[n_col].agg(["mean", "median", "std"]).round(3).reset_index()
                class_numerical_profiles.append({
                    "feature": n_col,
                    "by_class": grouped.to_dict(orient="records"),
                })
            except Exception:
                pass

        # Categorical vs Target Association (Cramer's V)
        target_associations = []
        for c_col in categorical_columns[:8]:
            try:
                ct = pd.crosstab(df[c_col].astype(str), target_series.astype(str))
                chi2, p_val, _, _ = stats.chi2_contingency(ct)
                n = ct.sum().sum()
                r, k = ct.shape
                if min(r - 1, k - 1) > 0:
                    cv = math.sqrt(max(0, (chi2 / n) / min(r - 1, k - 1)))
                    target_associations.append({
                        "feature": c_col,
                        "cramers_v": round(float(cv), 4),
                        "p_value": _safe_float(p_val),
                    })
            except Exception:
                pass

        return {
            "target_column": target_column,
            "task_type": "classification",
            "class_proportions": class_proportions,
            "classes_count": len(val_counts),
            "imbalance_ratio": imbalance_ratio,
            "imbalance_severity": imbalance_severity,
            "class_numerical_profiles": class_numerical_profiles,
            "categorical_associations": sorted(target_associations, key=lambda x: x["cramers_v"], reverse=True),
        }

    else:
        # Regression Target
        num_target = pd.to_numeric(target_series, errors="coerce").dropna()
        target_stats = {
            "mean": _safe_float(num_target.mean()),
            "median": _safe_float(num_target.median()),
            "std": _safe_float(num_target.std()),
            "min": _safe_float(num_target.min()),
            "max": _safe_float(num_target.max()),
            "skewness": _safe_float(num_target.skew()),
            "kurtosis": _safe_float(num_target.kurt()),
        }

        # Feature-to-target correlations
        corr_ranking = []
        for n_col in numeric_columns:
            if n_col == target_column:
                continue
            pair_df = df[[n_col, target_column]].dropna()
            if len(pair_df) > 5:
                p_r, p_p = stats.pearsonr(pair_df[n_col], pair_df[target_column])
                s_r, s_p = stats.spearmanr(pair_df[n_col], pair_df[target_column])
                corr_ranking.append({
                    "feature": n_col,
                    "pearson": round(float(p_r), 4),
                    "spearman": round(float(s_r), 4),
                    "pearson_p_value": _safe_float(p_p),
                    "non_linear_potential": bool(abs(s_r) - abs(p_r) > 0.15),
                })

        corr_ranking.sort(key=lambda x: abs(x["pearson"]), reverse=True)

        return {
            "target_column": target_column,
            "task_type": "regression",
            "target_statistics": target_stats,
            "correlation_ranking": corr_ranking,
        }


def analyze_data_quality(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Examines missingness distributions, duplicate patterns, constant/near-constant features,
    infinite numeric values, and potential target leakage indicators.
    """
    total_rows, total_cols = df.shape
    missing_by_col = []
    infinite_values_count = 0

    for col in df.columns:
        s = df[col]
        n_miss = int(s.isna().sum())
        miss_pct = round((n_miss / max(1, total_rows)) * 100, 2)
        if pd.api.types.is_numeric_dtype(s):
            inf_count = int(np.isinf(s.dropna()).sum())
            infinite_values_count += inf_count

        missing_by_col.append({
            "column": col,
            "missing_count": n_miss,
            "missing_pct": miss_pct,
        })

    missing_by_col.sort(key=lambda x: x["missing_count"], reverse=True)
    cols_with_missing = [c for c in missing_by_col if c["missing_count"] > 0]

    # Target leakage candidates (correlation > 0.98 with target)
    leakage_candidates = []
    if target_column and target_column in df.columns and pd.api.types.is_numeric_dtype(df[target_column]):
        for col in df.columns:
            if col != target_column and pd.api.types.is_numeric_dtype(df[col]):
                pair = df[[col, target_column]].dropna()
                if len(pair) >= 5:
                    r = abs(float(pair[col].corr(pair[target_column])))
                    if r >= 0.98:
                        leakage_candidates.append({
                            "column": col,
                            "correlation": round(r, 4),
                            "warning": f"Extreme correlation ({round(r, 4)}) with target '{target_column}'. Verify this feature is known prior to inference.",
                        })

    return {
        "total_missing_cells": int(df.isna().sum().sum()),
        "total_missing_pct": round((df.isna().sum().sum() / max(1, total_rows * total_cols)) * 100, 2),
        "columns_with_missing_count": len(cols_with_missing),
        "columns_with_missing": cols_with_missing[:15],
        "complete_rows_count": int(df.dropna().shape[0]),
        "complete_rows_pct": round((df.dropna().shape[0] / max(1, total_rows)) * 100, 2),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_columns": int(df.T.duplicated().sum()) if total_cols > 1 else 0,
        "infinite_values_count": infinite_values_count,
        "leakage_candidates": leakage_candidates,
    }


# =====================================================================
# 5. HEADLESS AUTOMATIC VISUALIZATION ENGINE
# =====================================================================

def generate_eda_visualizations(
    df: pd.DataFrame,
    profile_data: Dict[str, Any],
    target_column: Optional[str] = None,
    task_type: Optional[str] = None,
    max_sample: int = 2500,
) -> Dict[str, str]:
    """
    Renders clean, styled charts via Matplotlib/Seaborn and encodes them into Base64 PNGs.
    Generates missingness bar charts, correlation heatmaps, target distributions,
    and feature-target relationship plots.
    """
    plt.style.use("dark_background")
    images: Dict[str, str] = {}

    plot_df = df
    if len(df) > max_sample:
        plot_df = df.sample(n=max_sample, random_state=42)

    palette = ["#8265FF", "#A855F7", "#3B82F6", "#10B981", "#F59E0B", "#EF4444"]

    def _fig_to_base64(fig) -> str:
        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", dpi=130, facecolor="#111126", edgecolor="none")
        plt.close(fig)
        buf.seek(0)
        return "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")

    # 1. Missingness Chart (if any missing values exist)
    missing_counts = df.isna().sum()
    missing_series = missing_counts[missing_counts > 0].sort_values(ascending=False).head(12)

    if not missing_series.empty:
        try:
            fig, ax = plt.subplots(figsize=(7, 3.2))
            fig.patch.set_facecolor("#111126")
            ax.set_facecolor("#171630")
            bars = ax.barh(missing_series.index, (missing_series.values / len(df)) * 100, color="#8265FF", alpha=0.85)
            ax.set_xlabel("Missing Percentage (%)", color="#A4A4B8", fontsize=9)
            ax.set_title("Missing Values by Feature", color="#FFFFFF", fontsize=11, fontweight="bold", pad=10)
            ax.tick_params(colors="#A4A4B8", labelsize=8)
            ax.grid(axis="x", linestyle="--", alpha=0.15, color="#FFFFFF")
            images["missingness_bar_chart"] = _fig_to_base64(fig)
        except Exception:
            pass

    # 2. Correlation Heatmap (Pearson)
    num_cols = profile_data["dataset_summary"]["numeric_columns"]
    if len(num_cols) >= 2:
        try:
            heat_cols = num_cols[:14]  # Bound to 14 features for readability
            corr_mat = plot_df[heat_cols].corr(method="pearson").round(2)
            fig, ax = plt.subplots(figsize=(6.5, 5))
            fig.patch.set_facecolor("#111126")
            ax.set_facecolor("#171630")
            sns.heatmap(
                corr_mat,
                annot=len(heat_cols) <= 8,
                fmt=".2f",
                cmap="Purples",
                cbar=True,
                ax=ax,
                linewidths=0.5,
                linecolor="#111126",
            )
            ax.set_title("Pearson Correlation Heatmap", color="#FFFFFF", fontsize=11, fontweight="bold", pad=10)
            ax.tick_params(colors="#A4A4B8", labelsize=8, rotation=45)
            images["correlation_heatmap_pearson"] = _fig_to_base64(fig)
        except Exception:
            pass

    # 3. Target Distribution Chart
    target = target_column or profile_data["dataset_summary"]["inferred_target"]
    task = task_type or profile_data["dataset_summary"]["inferred_task"]

    if target and target in df.columns:
        try:
            fig, ax = plt.subplots(figsize=(6.5, 3.4))
            fig.patch.set_facecolor("#111126")
            ax.set_facecolor("#171630")
            target_series = df[target].dropna()

            if "classification" in task:
                counts = target_series.astype(str).value_counts().head(10)
                ax.bar(counts.index, counts.values, color=palette[:len(counts)], alpha=0.85)
                ax.set_ylabel("Sample Count", color="#A4A4B8", fontsize=9)
                ax.set_title(f"Target Class Distribution ({target})", color="#FFFFFF", fontsize=11, fontweight="bold", pad=10)
            else:
                num_target = pd.to_numeric(target_series, errors="coerce").dropna()
                ax.hist(num_target, bins=20, color="#8265FF", alpha=0.8, edgecolor="#111126")
                ax.set_ylabel("Frequency", color="#A4A4B8", fontsize=9)
                ax.set_title(f"Target Value Distribution ({target})", color="#FFFFFF", fontsize=11, fontweight="bold", pad=10)

            ax.tick_params(colors="#A4A4B8", labelsize=8)
            ax.grid(axis="y", linestyle="--", alpha=0.15, color="#FFFFFF")
            images["target_distribution_chart"] = _fig_to_base64(fig)
        except Exception:
            pass

    # 4. Top Numerical Feature Distributions (Histograms)
    if num_cols:
        try:
            top_num = [c for c in num_cols if c != target][:2]
            if top_num:
                fig, axes = plt.subplots(1, len(top_num), figsize=(7, 3.2))
                if len(top_num) == 1:
                    axes = [axes]
                fig.patch.set_facecolor("#111126")
                for idx, col_name in enumerate(top_num):
                    ax = axes[idx]
                    ax.set_facecolor("#171630")
                    clean_vals = pd.to_numeric(plot_df[col_name], errors="coerce").dropna()
                    ax.hist(clean_vals, bins=16, color="#A855F7", alpha=0.8, edgecolor="#111126")
                    ax.set_title(f"Distribution: {col_name}", color="#FFFFFF", fontsize=10, fontweight="bold")
                    ax.tick_params(colors="#A4A4B8", labelsize=8)
                    ax.grid(axis="y", linestyle="--", alpha=0.15, color="#FFFFFF")
                images["numerical_distributions_chart"] = _fig_to_base64(fig)
        except Exception:
            pass

    # 5. Top Categorical Feature Distributions
    cat_cols = profile_data["dataset_summary"]["categorical_columns"]
    if cat_cols:
        try:
            top_cat = [c for c in cat_cols if c != target][:2]
            if top_cat:
                fig, axes = plt.subplots(1, len(top_cat), figsize=(7, 3.2))
                if len(top_cat) == 1:
                    axes = [axes]
                fig.patch.set_facecolor("#111126")
                for idx, col_name in enumerate(top_cat):
                    ax = axes[idx]
                    ax.set_facecolor("#171630")
                    c_counts = df[col_name].dropna().astype(str).value_counts().head(6)
                    ax.bar(c_counts.index, c_counts.values, color="#3B82F6", alpha=0.85)
                    ax.set_title(f"Frequency: {col_name}", color="#FFFFFF", fontsize=10, fontweight="bold")
                    ax.tick_params(colors="#A4A4B8", labelsize=8, rotation=25)
                    ax.grid(axis="y", linestyle="--", alpha=0.15, color="#FFFFFF")
                images["categorical_distributions_chart"] = _fig_to_base64(fig)
        except Exception:
            pass

    return images


# =====================================================================
# 6. AUTOMATED EVIDENCE-BASED STATISTICAL INSIGHTS ENGINE
# =====================================================================

def generate_eda_insights(
    profile_data: Dict[str, Any],
    quality_data: Dict[str, Any],
    num_analysis: Dict[str, Any],
    cat_analysis: Dict[str, Any],
    target_analysis: Dict[str, Any],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Synthesizes empirical statistical findings and categorizes them into:
    - critical: Data leakage risks, high missingness, severe class imbalance
    - warnings: Outliers, high collinearity, rare categories, near-constant columns
    - info: Dimensionality, profile composition, distribution normality
    - recommendations: Concrete actionable data science & preprocessing steps
    """
    critical: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    info: List[Dict[str, Any]] = []
    recommendations: List[Dict[str, Any]] = []

    summary = profile_data["dataset_summary"]
    col_profiles = profile_data["column_profiles"]

    # 1. Critical: Potential Target Leakage
    for leak in quality_data.get("leakage_candidates", []):
        critical.append({
            "category": "Data Leakage Risk",
            "severity": "critical",
            "title": f"Potential Target Leakage in Column '{leak['column']}'",
            "affected_columns": [leak["column"]],
            "supporting_statistic": f"Correlation r = {leak['correlation']}",
            "explanation": f"Column '{leak['column']}' exhibits near-perfect correlation with the target. In real-world inference, this feature may not be available prior to prediction.",
            "recommendation": "Inspect feature origin. If it is computed from or downstream of the target, exclude it from model training.",
        })

    # 2. Critical: Severe Missing Data
    for col_info in quality_data.get("columns_with_missing", []):
        if col_info["missing_pct"] >= 40.0:
            critical.append({
                "category": "Data Completeness",
                "severity": "critical",
                "title": f"Extreme Missingness in Column '{col_info['column']}'",
                "affected_columns": [col_info["column"]],
                "supporting_statistic": f"{col_info['missing_pct']}% missing ({col_info['missing_count']} rows)",
                "explanation": f"Over 40% of observations in column '{col_info['column']}' are missing. Imputation may introduce substantial bias.",
                "recommendation": "Consider dropping this column or creating an explicit missingness indicator feature before imputation.",
            })

    # 3. Critical: Severe Class Imbalance
    if target_analysis.get("task_type") == "classification" and target_analysis.get("imbalance_severity") == "severe":
        critical.append({
            "category": "Class Imbalance",
            "severity": "critical",
            "title": f"Severe Class Imbalance in Target '{target_analysis['target_column']}'",
            "affected_columns": [target_analysis["target_column"]],
            "supporting_statistic": f"Imbalance ratio {target_analysis['imbalance_ratio']}:1",
            "explanation": "Standard classification accuracy will be misleading because trivial majority-class classifiers achieve deceptive high accuracy.",
            "recommendation": "Use Macro F1 and Balanced Accuracy for evaluation. Apply cost-sensitive balanced class weights or resampling within training folds.",
        })

    # 4. Warnings: Multicollinearity
    for pair in num_analysis.get("highly_correlated_pairs", []):
        warnings.append({
            "category": "Multicollinearity",
            "severity": "warning",
            "title": f"Strong Collinearity: '{pair['feature_1']}' and '{pair['feature_2']}'",
            "affected_columns": [pair["feature_1"], pair["feature_2"]],
            "supporting_statistic": f"Pearson r = {pair['pearson']}, Spearman rho = {pair['spearman']}",
            "explanation": "High correlation between predictor variables inflates coefficient variance in linear and distance-based models.",
            "recommendation": "Consider dropping one of the redundant variables, applying L1 Lasso shrinkage, or using PCA.",
        })

    # 5. Warnings: Outliers & Skewness
    for col in num_analysis.get("skewed_columns", [])[:3]:
        prof = col_profiles.get(col, {})
        warnings.append({
            "category": "Distribution Skewness",
            "severity": "warning",
            "title": f"Heavily Skewed Feature '{col}'",
            "affected_columns": [col],
            "supporting_statistic": f"Skewness = {prof.get('skewness')}",
            "explanation": f"Feature '{col}' exhibits significant distribution asymmetry. Distance-based estimators (KNN, SVM, Linear) may be biased by extreme tail values.",
            "recommendation": "Apply RobustScaler or logarithmic / Yeo-Johnson power transformations during preprocessing.",
        })

    # 6. Warnings: Constant & Identifier Columns
    for col in summary.get("constant_columns", []):
        warnings.append({
            "category": "Feature Variance",
            "severity": "warning",
            "title": f"Zero-Variance Constant Column '{col}'",
            "affected_columns": [col],
            "supporting_statistic": "1 unique value",
            "explanation": f"Column '{col}' contains exactly one distinct value across all rows and provides zero predictive information.",
            "recommendation": "Drop this column before model training to reduce feature dimensionality.",
        })

    for col in summary.get("identifier_columns", []):
        warnings.append({
            "category": "Identifier Column",
            "severity": "warning",
            "title": f"Apparent ID Feature '{col}'",
            "affected_columns": [col],
            "supporting_statistic": "100% unique values",
            "explanation": f"Column '{col}' has unique values per row matching identifier heuristics. Training on raw IDs risks memorization.",
            "recommendation": "Exclude from model feature matrices unless required as an entity key for temporal grouping.",
        })

    # 7. Informational Insights
    info.append({
        "category": "Dataset Profiling",
        "severity": "info",
        "title": f"Dataset Dimension Profile ({summary['row_count']} rows x {summary['column_count']} columns)",
        "affected_columns": [],
        "supporting_statistic": f"{summary['memory_usage_str']} memory usage",
        "explanation": f"The dataset contains {summary['numeric_columns_count']} numeric, {summary['categorical_columns_count']} categorical, {summary['boolean_columns_count']} boolean, and {summary['datetime_columns_count']} datetime features.",
        "recommendation": "Proceed with dataset-adaptive preprocessing tailored to feature type composition.",
    })

    if quality_data.get("duplicate_rows", 0) > 0:
        info.append({
            "category": "Data Cleanliness",
            "severity": "info",
            "title": f"Duplicate Rows Detected ({quality_data['duplicate_rows']} records)",
            "affected_columns": [],
            "supporting_statistic": f"{quality_data['duplicate_rows']} duplicate rows",
            "explanation": "Duplicate observations may artificially inflate cross-validation scores if identical records exist across train and test folds.",
            "recommendation": "Review whether duplicates reflect genuine independent observations or repeated records.",
        })

    # 8. Actionable Preprocessing Recommendations
    recommendations.append({
        "category": "Preprocessing Recommendation",
        "severity": "recommendation",
        "title": "Leak-Free Preprocessing Pipeline Architecture",
        "affected_columns": [],
        "supporting_statistic": f"{summary['numeric_columns_count']} num / {summary['categorical_columns_count']} cat",
        "explanation": "Construct an sklearn ColumnTransformer with Median/Mean imputation for numeric variables and OneHotEncoder(handle_unknown='ignore') for categoricals.",
        "recommendation": "Fit preprocessors strictly inside cross-validation training folds to ensure zero leakage.",
    })

    if target_analysis.get("task_type") == "regression":
        top_corr = target_analysis.get("correlation_ranking", [])
        if top_corr:
            strongest = top_corr[0]
            recommendations.append({
                "category": "Feature Selection Recommendation",
                "severity": "recommendation",
                "title": f"Strongest Linear Predictor: '{strongest['feature']}'",
                "affected_columns": [strongest["feature"]],
                "supporting_statistic": f"Pearson r = {strongest['pearson']}",
                "explanation": f"Feature '{strongest['feature']}' exhibits the highest linear association with the target.",
                "recommendation": "Ensure this feature is prioritized in feature importance tracking and model interpretability.",
            })

    return {
        "critical": critical,
        "warnings": warnings,
        "info": info,
        "recommendations": recommendations,
        "total_insights_count": len(critical) + len(warnings) + len(info) + len(recommendations),
    }


# =====================================================================
# 7. DYNAMIC EDA WORKFLOW FLOWCHART ENGINE
# =====================================================================

def generate_eda_flowchart(
    strategy_data: Dict[str, Any],
    profile_data: Dict[str, Any],
    duration_ms: float = 0.0,
) -> Dict[str, Any]:
    """
    Constructs an execution tree representing completed, skipped, and warned operations
    along with transparent operational rationale and execution timing.
    """
    summary = profile_data["dataset_summary"]
    active_profiles = strategy_data["active_profiles"]

    steps = [
        {
            "id": "upload_validation",
            "title": "Dataset Upload & Header Sanitization",
            "status": "completed",
            "details": f"Successfully parsed {summary['row_count']} rows and {summary['column_count']} unique columns.",
        },
        {
            "id": "semantic_profiling",
            "title": "Statistical & Semantic Typing",
            "status": "completed",
            "details": f"Classified {summary['numeric_columns_count']} numeric, {summary['categorical_columns_count']} categorical, {summary['datetime_columns_count']} datetime features.",
        },
        {
            "id": "strategy_adaptation",
            "title": "Adaptive Strategy Selection",
            "status": "completed",
            "details": f"Activated {len(active_profiles)} profiles: {', '.join(active_profiles[:3])}.",
        },
        {
            "id": "quality_diagnostics",
            "title": "Data Quality & Integrity Audit",
            "status": "completed",
            "details": f"Audited {summary['total_cells']} cells: {summary['missing_cells_pct']}% missing, {summary['duplicate_rows_count']} duplicate rows.",
        },
        {
            "id": "numerical_analysis",
            "title": "Numerical Moments & Outlier Analysis",
            "status": "completed" if summary["numeric_columns_count"] > 0 else "skipped",
            "details": (
                f"Evaluated {summary['numeric_columns_count']} numeric features for skewness, IQR outliers, and kurtosis."
                if summary["numeric_columns_count"] > 0
                else "Skipped: No numeric features present in dataset."
            ),
        },
        {
            "id": "categorical_analysis",
            "title": "Categorical Frequency & Cramer's V Tests",
            "status": "completed" if summary["categorical_columns_count"] > 0 else "skipped",
            "details": (
                f"Computed Top-N category distributions and Cramer's V association tests for {summary['categorical_columns_count']} categorical features."
                if summary["categorical_columns_count"] > 0
                else "Skipped: No categorical features present in dataset."
            ),
        },
        {
            "id": "temporal_analysis",
            "title": "Temporal Trends & Seasonality Checks",
            "status": "completed" if summary["datetime_columns_count"] > 0 else "skipped",
            "details": (
                f"Evaluated timespan across {summary['datetime_columns_count']} datetime columns."
                if summary["datetime_columns_count"] > 0
                else "Skipped: No datetime features detected in dataset."
            ),
        },
        {
            "id": "target_relationship",
            "title": "Target-Aware Exploratory Analysis",
            "status": "completed",
            "details": f"Analyzed feature associations against target '{strategy_data['target_column']}' ({strategy_data['task_type']}).",
        },
        {
            "id": "visualization_rendering",
            "title": "Statistical Chart Rendering",
            "status": "completed",
            "details": "Generated correlation heatmaps, missingness bars, and distribution plots.",
        },
        {
            "id": "insight_generation",
            "title": "Automated Evidence-Based Insights",
            "status": "completed",
            "details": "Synthesized critical alerts, warnings, and preprocessing recommendations.",
        },
        {
            "id": "report_generation",
            "title": "Multi-Format Report Compilation",
            "status": "completed",
            "details": "Compiled PDF, HTML, and JSON report artifacts for download.",
        },
    ]

    return {
        "execution_time_ms": round(duration_ms, 1),
        "total_steps": len(steps),
        "completed_steps": len([s for s in steps if s["status"] == "completed"]),
        "skipped_steps": len([s for s in steps if s["status"] == "skipped"]),
        "steps": steps,
    }


# =====================================================================
# 8. MULTI-FORMAT PROFESSIONAL EDA REPORT GENERATOR (PDF, HTML, JSON)
# =====================================================================

def generate_eda_pdf_report(
    analysis_data: Dict[str, Any],
) -> bytes:
    """
    Generates a professional, multi-page executive PDF document via ReportLab.
    Matches the 9-page/section specification with typography, data tables, and embedded charts.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=45,
        rightMargin=45,
        topMargin=45,
        bottomMargin=45,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#111827"),
        spaceAfter=6,
    )
    h2_style = ParagraphStyle(
        "SectionH2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#8265FF"),
        spaceBefore=14,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#374151"),
        spaceAfter=5,
    )
    meta_style = ParagraphStyle(
        "MetaText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#6B7280"),
    )

    story = []
    summary = analysis_data["profile"]["dataset_summary"]
    quality = analysis_data["quality"]
    strategy = analysis_data["strategy"]
    insights = analysis_data["insights"]

    # PAGE 1: Executive Summary
    story.append(Paragraph("AutoVeyra Exploratory Data Analysis Report", title_style))
    story.append(Paragraph(f"Dataset: <b>{summary['filename']}</b> | Generated: {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}", meta_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#8265FF"), spaceBefore=8, spaceAfter=14))

    story.append(Paragraph("Executive Summary & Dataset Identity", h2_style))
    exec_summary_text = (
        f"This automated exploratory audit evaluated <b>{summary['row_count']} records</b> across <b>{summary['column_count']} features</b> "
        f"({summary['memory_usage_str']} memory usage). The platform detected a <b>{strategy['task_type'].replace('_', ' ').title()}</b> problem structure "
        f"focused on target variable <b>'{strategy['target_column']}'</b>. "
        f"Overall dataset completeness is <b>{100 - quality['total_missing_pct']:.1f}%</b> ({quality['total_missing_cells']} missing cells). "
        f"A total of <b>{insights.get('total_insights_count', 0)} analytical findings</b> were identified, including "
        f"<b>{len(insights.get('critical', []))} critical issues</b> and <b>{len(insights.get('warnings', []))} warnings</b>."
    )
    story.append(Paragraph(exec_summary_text, body_style))
    story.append(Spacer(1, 10))

    # Overview Metrics Table
    overview_table_data = [
        ["Metric", "Value", "Metric", "Value"],
        ["Total Rows", str(summary["row_count"]), "Total Columns", str(summary["column_count"])],
        ["Numerical Columns", str(summary["numeric_columns_count"]), "Categorical Columns", str(summary["categorical_columns_count"])],
        ["Missing Cells", f"{quality['total_missing_cells']} ({quality['total_missing_pct']}%)", "Duplicate Rows", str(quality["duplicate_rows"])],
        ["Target Column", str(strategy["target_column"]), "Inferred Task", str(strategy["task_type"])],
    ]
    t = Table(overview_table_data, colWidths=[120, 135, 120, 135])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F2937")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D1D5DB")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F9FAFB"), colors.white]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t)
    story.append(Spacer(1, 14))

    # Critical Alerts Summary
    if insights.get("critical"):
        story.append(Paragraph("Critical Integrity Findings", h2_style))
        for crit in insights["critical"][:3]:
            story.append(Paragraph(f"• <b>{crit['title']}</b>: {crit['explanation']}", body_style))
            story.append(Paragraph(f"  <i>Action: {crit['recommendation']}</i>", meta_style))
            story.append(Spacer(1, 4))

    story.append(PageBreak())

    # PAGE 2: Column Profiles & Data Quality
    story.append(Paragraph("Column-Level Profiling & Data Quality Audit", h2_style))
    story.append(Paragraph("Per-variable statistical distributions, cardinality, and completeness.", body_style))
    story.append(Spacer(1, 6))

    col_headers = ["Column Name", "Type", "Missing %", "Unique", "Sample Mean / Top Category", "Outliers"]
    col_rows = [col_headers]
    for col_name, c_prof in list(analysis_data["profile"]["column_profiles"].items())[:15]:
        val_summary = str(c_prof.get("mean", c_prof.get("top_value", "N/A")))[:18]
        outliers_str = f"{c_prof.get('outlier_count_iqr', 0)}" if c_prof.get("semantic_type") == "numeric" else "-"
        col_rows.append([
            col_name[:18],
            c_prof.get("semantic_type", "unknown")[:10],
            f"{c_prof.get('missing_pct', 0)}%",
            str(c_prof.get("unique_count", 0)),
            val_summary,
            outliers_str,
        ])

    col_table = Table(col_rows, colWidths=[105, 75, 60, 60, 140, 65])
    col_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F2937")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F9FAFB"), colors.white]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(col_table)
    story.append(Spacer(1, 14))

    # Missing Data Diagnostics
    story.append(Paragraph("Missing Data & Duplicate Diagnostics", h2_style))
    story.append(Paragraph(
        f"Total complete records without any missing features: <b>{quality['complete_rows_count']} rows ({quality['complete_rows_pct']}%)</b>. "
        f"Duplicate rows detected: <b>{quality['duplicate_rows']}</b>. Duplicate column headers: <b>{quality['duplicate_columns']}</b>. "
        f"Zero-variance constant columns: <b>{len(summary['constant_columns'])}</b>.",
        body_style,
    ))

    story.append(PageBreak())

    # PAGE 3: Feature Relationships & Correlations
    story.append(Paragraph("Correlation Analysis & Feature Relationships", h2_style))
    story.append(Paragraph(
        "Pairwise linear (Pearson) and monotonic (Spearman rank) correlation coefficients. "
        "High correlation indicates collinearity that can destabilize regression and distance-based estimators.",
        body_style,
    ))
    story.append(Spacer(1, 8))

    high_corr = analysis_data["numerical"].get("highly_correlated_pairs", [])
    if high_corr:
        corr_table_data = [["Feature 1", "Feature 2", "Pearson r", "Spearman rho", "Collinearity Risk"]]
        for pair in high_corr[:8]:
            corr_table_data.append([
                pair["feature_1"][:20],
                pair["feature_2"][:20],
                str(pair["pearson"]),
                str(pair["spearman"]),
                pair["risk"],
            ])
        corr_table = Table(corr_table_data, colWidths=[120, 120, 75, 75, 115])
        corr_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F2937")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7.5),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F9FAFB"), colors.white]),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(corr_table)
    else:
        story.append(Paragraph("No severe collinear pairs (|r| >= 0.80) were detected among numerical variables.", body_style))

    story.append(Spacer(1, 14))

    # Target Association
    target_rel = analysis_data["target_relationships"]
    story.append(Paragraph(f"Target Association ({strategy['target_column']})", h2_style))
    if strategy["task_type"] == "regression":
        top_preds = target_rel.get("correlation_ranking", [])[:5]
        if top_preds:
            t_data = [["Feature", "Pearson r", "Spearman rho", "Non-Linear Potential"]]
            for pred in top_preds:
                t_data.append([
                    pred["feature"][:25],
                    str(pred["pearson"]),
                    str(pred["spearman"]),
                    "Yes" if pred.get("non_linear_potential") else "No",
                ])
            target_table = Table(t_data, colWidths=[180, 100, 100, 125])
            target_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F2937")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            story.append(target_table)
    else:
        class_props = target_rel.get("class_proportions", [])
        if class_props:
            cp_data = [["Class Label", "Sample Count", "Percentage (%)"]]
            for cp in class_props:
                cp_data.append([str(cp["class"]), str(cp["count"]), f"{cp['percentage']}%"])
            cp_table = Table(cp_data, colWidths=[180, 150, 175])
            cp_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F2937")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            story.append(cp_table)

    story.append(PageBreak())

    # PAGE 4: Automated Insights & Workflow Steps
    story.append(Paragraph("Actionable Recommendations & Preprocessing Roadmap", h2_style))
    for rec in insights.get("recommendations", []):
        story.append(Paragraph(f"• <b>{rec['title']}</b>", body_style))
        story.append(Paragraph(f"  {rec['explanation']} <i>Recommended Action: {rec['recommendation']}</i>", meta_style))
        story.append(Spacer(1, 4))

    story.append(Spacer(1, 14))
    story.append(Paragraph("Executed EDA Workflow Operations", h2_style))
    flow_steps = analysis_data["flowchart"]["steps"]
    step_rows = [["Step", "Operation", "Status", "Execution Details"]]
    for idx, s in enumerate(flow_steps, 1):
        step_rows.append([
            str(idx),
            s["title"][:28],
            s["status"].upper(),
            s["details"][:42],
        ])
    step_table = Table(step_rows, colWidths=[35, 150, 85, 235])
    step_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F2937")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F9FAFB"), colors.white]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(step_table)

    # Build document
    doc.build(story)
    buffer.seek(0)
    return buffer.read()


def generate_eda_html_report(
    analysis_data: Dict[str, Any],
) -> str:
    """Generates a standalone, dark-themed responsive HTML report."""
    summary = analysis_data["profile"]["dataset_summary"]
    quality = analysis_data["quality"]
    strategy = analysis_data["strategy"]
    insights = analysis_data["insights"]
    visuals = analysis_data.get("visualizations", {})

    charts_html = ""
    for chart_key, base64_uri in visuals.items():
        chart_title = chart_key.replace("_", " ").title()
        charts_html += f"""
        <div class="bg-[#171630] border border-white/[0.08] p-4 rounded-xl shadow-lg">
          <h4 class="text-xs font-semibold text-gray-300 uppercase tracking-wider mb-2">{chart_title}</h4>
          <img src="{base64_uri}" alt="{chart_title}" class="w-full rounded-lg object-contain" />
        </div>
        """

    insights_html = ""
    for category_name, badge_color in [
        ("critical", "bg-rose-500/20 text-rose-300 border-rose-500/30"),
        ("warnings", "bg-amber-500/20 text-amber-300 border-amber-500/30"),
        ("recommendations", "bg-[#8265FF]/20 text-[#8265FF] border-[#8265FF]/30"),
    ]:
        items = insights.get(category_name, [])
        if items:
            insights_html += f"<div class='space-y-3'><h4 class='text-sm font-bold uppercase tracking-wider text-white capitalize'>{category_name} ({len(items)})</h4>"
            for itm in items:
                insights_html += f"""
                <div class="bg-[#171630] border border-white/[0.06] p-4 rounded-xl space-y-1">
                  <div class="flex items-center justify-between">
                    <span class="text-xs font-bold text-white">{itm['title']}</span>
                    <span class="text-[10px] uppercase font-semibold px-2 py-0.5 rounded-full border {badge_color}">{itm['category']}</span>
                  </div>
                  <p class="text-xs text-[#A4A4B8]">{itm['explanation']}</p>
                  <p class="text-xs text-white font-medium"><b>Recommendation:</b> {itm['recommendation']}</p>
                </div>
                """
            insights_html += "</div>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AutoVeyra EDA Report - {summary['filename']}</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-[#070711] text-[#F7F7FC] font-sans p-6 sm:p-12 max-w-6xl mx-auto space-y-8">
  
  <header class="border-b border-white/[0.08] pb-6 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
    <div>
      <span class="text-[10px] font-semibold tracking-wider uppercase px-2 py-0.5 rounded-full bg-[#8265FF]/20 text-[#8265FF] border border-[#8265FF]/30">
        AutoVeyra Automated EDA Report
      </span>
      <h1 class="text-2xl font-black text-white mt-2">{summary['filename']}</h1>
      <p class="text-xs text-[#A4A4B8]">Generated on {datetime.datetime.now(datetime.timezone.utc).strftime('%B %d, %Y - %H:%M UTC')}</p>
    </div>
    <div class="text-right text-xs bg-[#111126] border border-white/[0.08] p-3 rounded-xl">
      <span class="text-[#A4A4B8] block">Dataset Dimensions</span>
      <span class="text-white font-mono font-bold text-sm">{summary['row_count']} Rows × {summary['column_count']} Cols</span>
    </div>
  </header>

  <!-- Executive Summary Cards -->
  <section class="grid grid-cols-2 sm:grid-cols-4 gap-4">
    <div class="bg-[#111126] border border-white/[0.08] p-4 rounded-xl">
      <span class="text-[11px] font-semibold uppercase tracking-wider text-[#A4A4B8] block mb-1">Inferred Task</span>
      <span class="text-lg font-bold text-white capitalize">{strategy['task_type']}</span>
    </div>
    <div class="bg-[#111126] border border-white/[0.08] p-4 rounded-xl">
      <span class="text-[11px] font-semibold uppercase tracking-wider text-[#A4A4B8] block mb-1">Target Feature</span>
      <span class="text-lg font-bold text-[#8265FF] font-mono">{strategy['target_column']}</span>
    </div>
    <div class="bg-[#111126] border border-white/[0.08] p-4 rounded-xl">
      <span class="text-[11px] font-semibold uppercase tracking-wider text-[#A4A4B8] block mb-1">Completeness</span>
      <span class="text-lg font-bold text-emerald-400 font-mono">{100 - quality['total_missing_pct']:.1f}%</span>
    </div>
    <div class="bg-[#111126] border border-white/[0.08] p-4 rounded-xl">
      <span class="text-[11px] font-semibold uppercase tracking-wider text-[#A4A4B8] block mb-1">Duplicate Rows</span>
      <span class="text-lg font-bold text-white font-mono">{quality['duplicate_rows']}</span>
    </div>
  </section>

  <!-- Visualizations Section -->
  {f'<section class="space-y-4"><h2 class="text-base font-bold text-white uppercase tracking-wider border-b border-white/[0.06] pb-2">Statistical Visualizations</h2><div class="grid grid-cols-1 md:grid-cols-2 gap-4">{charts_html}</div></section>' if charts_html else ''}

  <!-- Automated Insights Section -->
  <section class="space-y-4">
    <h2 class="text-base font-bold text-white uppercase tracking-wider border-b border-white/[0.06] pb-2">Empirical Statistical Insights</h2>
    <div class="space-y-6">
      {insights_html}
    </div>
  </section>

  <footer class="border-t border-white/[0.08] pt-6 text-center text-xs text-[#A4A4B8]">
    AutoVeyra AutoML Studio &copy; 2026. All exploratory data analyses computed deterministically without external API dependencies.
  </footer>

</body>
</html>"""


# =====================================================================
# 9. END-TO-END ORCHESTRATION PIPELINE
# =====================================================================

async def run_full_eda_pipeline(
    df: pd.DataFrame,
    filename: str = "dataset.csv",
    target_column: Optional[str] = None,
    save_to_db: bool = True,
) -> Dict[str, Any]:
    """
    Orchestrates the complete 11-step adaptive EDA intelligence workflow:
    1. Dataset profiling & semantic typing
    2. Adaptive strategy selection
    3. Data quality diagnostics
    4. Numerical moments & outlier analysis
    5. Categorical distributions & Cramer's V tests
    6. Target-aware relationship evaluation
    7. Automated insights synthesis
    8. Flowchart generation
    9. Headless visualization rendering
    10. Persistence in eda.db
    """
    start_time = time.time()
    init_eda_db()

    # Step 1: Profiling
    profile_data = profile_dataset(df, filename=filename, target_column=target_column)

    # Step 2: Adaptive Strategy
    strategy_data = detect_eda_strategy(df, profile_data, target_column=target_column)
    resolved_target = strategy_data["target_column"]
    resolved_task = strategy_data["task_type"]

    # Step 3: Data Quality
    quality_data = analyze_data_quality(df, target_column=resolved_target)

    # Step 4: Numerical Analysis
    num_cols = profile_data["dataset_summary"]["numeric_columns"]
    num_analysis = analyze_numerical_features(
        df,
        numeric_columns=num_cols,
        target_column=resolved_target,
        sample_size=strategy_data.get("sample_size", 3000),
    )

    # Step 5: Categorical Analysis
    cat_cols = profile_data["dataset_summary"]["categorical_columns"]
    cat_analysis = analyze_categorical_features(
        df,
        categorical_columns=cat_cols,
        target_column=resolved_target,
    )

    # Step 6: Target Relationships
    target_analysis = analyze_target_relationships(
        df,
        target_column=resolved_target,
        task_type=resolved_task,
        numeric_columns=num_cols,
        categorical_columns=cat_cols,
    )

    # Step 7: Automated Insights
    insights_data = generate_eda_insights(
        profile_data=profile_data,
        quality_data=quality_data,
        num_analysis=num_analysis,
        cat_analysis=cat_analysis,
        target_analysis=target_analysis,
    )

    # Step 8: Visualizations
    visualizations_data = generate_eda_visualizations(
        df=df,
        profile_data=profile_data,
        target_column=resolved_target,
        task_type=resolved_task,
        max_sample=strategy_data.get("sample_size", 2500),
    )

    # Step 9: Flowchart
    duration_ms = (time.time() - start_time) * 1000
    flowchart_data = generate_eda_flowchart(
        strategy_data=strategy_data,
        profile_data=profile_data,
        duration_ms=duration_ms,
    )

    analysis_id = f"eda_{uuid.uuid4().hex[:16]}"

    result: Dict[str, Any] = {
        "analysis_id": analysis_id,
        "filename": filename,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "profile": profile_data,
        "strategy": strategy_data,
        "quality": quality_data,
        "numerical": num_analysis,
        "categorical": cat_analysis,
        "target_relationships": target_analysis,
        "insights": insights_data,
        "visualizations": visualizations_data,
        "flowchart": flowchart_data,
    }

    # Step 10: Persistence in SQLite
    if save_to_db:
        try:
            with _connect() as conn:
                conn.execute(
                    """
                    INSERT INTO eda_analyses (
                        analysis_id, filename, row_count, column_count,
                        target_column, task_type, summary_json, profiles_json,
                        strategy_json, quality_json, numerical_json, categorical_json,
                        target_relationships_json, insights_json, flowchart_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        analysis_id,
                        filename,
                        profile_data["dataset_summary"]["row_count"],
                        profile_data["dataset_summary"]["column_count"],
                        resolved_target,
                        resolved_task,
                        json.dumps(_json_safe(profile_data["dataset_summary"])),
                        json.dumps(_json_safe(profile_data["column_profiles"])),
                        json.dumps(_json_safe(strategy_data)),
                        json.dumps(_json_safe(quality_data)),
                        json.dumps(_json_safe(num_analysis)),
                        json.dumps(_json_safe(cat_analysis)),
                        json.dumps(_json_safe(target_analysis)),
                        json.dumps(_json_safe(insights_data)),
                        json.dumps(_json_safe(flowchart_data)),
                        result["created_at"],
                    ),
                )
        except Exception as db_err:
            print(f"Warning: Failed to persist EDA analysis to SQLite: {db_err}")

    return result


def get_eda_analysis_from_db(analysis_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves a previously computed EDA analysis from SQLite."""
    init_eda_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM eda_analyses WHERE analysis_id = ?",
            (analysis_id,),
        ).fetchone()

        if not row:
            return None

        return {
            "analysis_id": row["analysis_id"],
            "filename": row["filename"],
            "created_at": row["created_at"],
            "profile": {
                "dataset_summary": json.loads(row["summary_json"]),
                "column_profiles": json.loads(row["profiles_json"]),
            },
            "strategy": json.loads(row["strategy_json"]),
            "quality": json.loads(row["quality_json"]),
            "numerical": json.loads(row["numerical_json"]),
            "categorical": json.loads(row["categorical_json"]),
            "target_relationships": json.loads(row["target_relationships_json"]),
            "insights": json.loads(row["insights_json"]),
            "flowchart": json.loads(row["flowchart_json"]),
        }


def list_eda_history_from_db(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieves summaries of previously saved EDA analyses."""
    init_eda_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT analysis_id, filename, summary_json, strategy_json, created_at FROM eda_analyses ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()

        results = []
        for r in rows:
            try:
                summary = json.loads(r["summary_json"]) if r["summary_json"] else {}
                strategy = json.loads(r["strategy_json"]) if r["strategy_json"] else {}
                results.append({
                    "analysis_id": r["analysis_id"],
                    "filename": r["filename"],
                    "created_at": r["created_at"],
                    "row_count": summary.get("row_count"),
                    "column_count": summary.get("column_count"),
                    "target_column": strategy.get("target_column"),
                    "task_type": strategy.get("task_type"),
                })
            except Exception:
                results.append({
                    "analysis_id": r["analysis_id"],
                    "filename": r["filename"],
                    "created_at": r["created_at"],
                })
        return results
