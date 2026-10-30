from contextlib import contextmanager
# app/services/feature_service.py

import io
import json
import math
import re
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from fastapi import HTTPException, UploadFile
from app.utils.security import safe_read_csv
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import (
    RFE,
    SelectFromModel,
    SelectKBest,
    VarianceThreshold,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    explained_variance_score,
    f1_score,
    log_loss,
    max_error as sklearn_max_error,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
    TargetEncoder,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from app.services.model_registry_service import register_model

SERVER_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = SERVER_DIR / "data"
FEATURE_DB_PATH = DATA_DIR / "feature_history.db"


# =========================================================
# Database Setup & Migrations
# =========================================================

@contextmanager
def _connect():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(FEATURE_DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def init_feature_db() -> None:
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feature_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT UNIQUE NOT NULL,
                task_type TEXT NOT NULL,
                target_column TEXT NOT NULL,
                dataset_name TEXT,
                model_type TEXT NOT NULL,
                ranking_metric TEXT NOT NULL,
                winning_strategy TEXT NOT NULL,
                strategy_results_json TEXT NOT NULL,
                baseline_metrics_json TEXT NOT NULL,
                winning_holdout_metrics_json TEXT NOT NULL,
                delta_metrics_json TEXT NOT NULL,
                selected_features_json TEXT NOT NULL,
                removed_features_json TEXT NOT NULL,
                transformation_config_json TEXT NOT NULL,
                registered_model_id INTEGER,
                data_quality_json TEXT,
                duration_seconds REAL,
                created_at TEXT NOT NULL
            )
        """)
        # Backward-compatible migrations for Phase 3
        cursor = conn.execute("PRAGMA table_info(feature_runs)")
        existing_cols = {row["name"] for row in cursor.fetchall()}
        if "eda_analysis_id" not in existing_cols:
            conn.execute("ALTER TABLE feature_runs ADD COLUMN eda_analysis_id TEXT")
        if "feature_lineage_json" not in existing_cols:
            conn.execute("ALTER TABLE feature_runs ADD COLUMN feature_lineage_json TEXT")
        if "strategy_preset" not in existing_cols:
            conn.execute("ALTER TABLE feature_runs ADD COLUMN strategy_preset TEXT")
        if "comparison_matrix_json" not in existing_cols:
            conn.execute("ALTER TABLE feature_runs ADD COLUMN comparison_matrix_json TEXT")


def _json_safe(value: Any) -> Any:
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
        return round(value, 4) if math.isfinite(value) else None
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)


def _safe_float(value: Any, ndigits: int = 4) -> Optional[float]:
    if value is None:
        return None
    try:
        val = float(value)
        return round(val, ndigits) if math.isfinite(val) else None
    except (TypeError, ValueError, OverflowError):
        return None


# =========================================================
# Feature Profiling & Diagnostics
# =========================================================

def profile_dataset(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    max_sample_rows: int = 10000,
    correlation_threshold: float = 0.85,
    eda_analysis_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Profile raw dataset with statistics, data quality alerts,
    collinearity diagnostics, and type classification without using target to drop columns.
    Enriched with Phase 2 EDA cross-referencing where eda_analysis_id is provided.
    """
    initial_rows, total_cols = df.shape
    is_sampled = False
    profile_df = df

    if initial_rows > max_sample_rows:
        profile_df = df.sample(n=max_sample_rows, random_state=42)
        is_sampled = True

    duplicate_rows_count = int(df.duplicated().sum())
    duplicate_rows_pct = round((duplicate_rows_count / initial_rows) * 100, 2) if initial_rows > 0 else 0.0

    columns_profile = []
    numeric_cols = []
    categorical_cols = []
    datetime_candidates = []
    text_candidates = []
    constant_cols = []
    near_constant_cols = []
    id_like_cols = []
    high_cardinality_cols = []
    quality_alerts = []

    for col in df.columns:
        series = df[col]
        n_missing = int(series.isna().sum())
        missing_pct = round((n_missing / initial_rows) * 100, 2) if initial_rows > 0 else 0.0
        n_unique = int(series.nunique(dropna=True))
        cardinality_ratio = round(n_unique / initial_rows, 4) if initial_rows > 0 else 0.0

        inferred_type = "categorical"
        is_numeric = pd.api.types.is_numeric_dtype(series)

        # Check datetime
        is_datetime = False
        if not is_numeric and series.dropna().shape[0] > 0:
            sample_vals = series.dropna().head(20).astype(str)
            date_pattern = re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4}")
            if any(date_pattern.search(v) for v in sample_vals):
                try:
                    pd.to_datetime(sample_vals, errors="raise")
                    is_datetime = True
                    inferred_type = "datetime"
                    datetime_candidates.append(col)
                except Exception:
                    pass

        # Check text (long string average)
        if not is_numeric and not is_datetime and series.dropna().shape[0] > 0:
            str_sample = series.dropna().head(30).astype(str)
            avg_char_len = float(str_sample.apply(len).mean())
            if avg_char_len > 35.0 and n_unique > 15:
                inferred_type = "text"
                text_candidates.append(col)

        if is_numeric and not is_datetime:
            inferred_type = "numeric"
            numeric_cols.append(col)
        elif not is_datetime and inferred_type != "text":
            categorical_cols.append(col)

        # Constant and near constant
        if n_unique <= 1:
            constant_cols.append(col)
            quality_alerts.append({
                "column": col,
                "type": "constant",
                "severity": "warning",
                "message": f"Column '{col}' is constant (has only {n_unique} unique value). Recommended for removal.",
            })
        else:
            top_freq_pct = (series.value_counts(dropna=False, normalize=True).iloc[0] * 100) if initial_rows > 0 else 0
            if top_freq_pct >= 95.0:
                near_constant_cols.append(col)
                quality_alerts.append({
                    "column": col,
                    "type": "near_constant",
                    "severity": "warning",
                    "message": f"Column '{col}' is near-constant ({top_freq_pct:.1f}% single dominant value).",
                })

        # Identifier-like heuristics
        col_lower = str(col).lower()
        id_terms = ["id", "uuid", "guid", "key", "index", "code"]
        is_id_named = any(term in col_lower.split("_") or term in col_lower.split() or col_lower == term for term in id_terms)
        if (is_id_named or cardinality_ratio >= 0.98) and n_unique > 20 and col != target_column:
            id_like_cols.append(col)
            quality_alerts.append({
                "column": col,
                "type": "identifier_candidate",
                "severity": "info",
                "message": f"Column '{col}' shows high distinctness ({n_unique} unique values). May be an identifier; verify before modeling.",
            })

        # High cardinality categoricals
        if inferred_type == "categorical" and n_unique > 50 and col != target_column:
            high_cardinality_cols.append(col)
            quality_alerts.append({
                "column": col,
                "type": "high_cardinality",
                "severity": "warning",
                "message": f"Categorical '{col}' has {n_unique} distinct values. Consider grouping rare categories or target encoding.",
            })

        # Missing values alert
        if missing_pct > 30.0:
            quality_alerts.append({
                "column": col,
                "type": "high_missing",
                "severity": "warning",
                "message": f"Column '{col}' has {missing_pct}% missing values.",
            })

        col_stats = {
            "name": col,
            "inferred_type": inferred_type,
            "missing_count": n_missing,
            "missing_pct": missing_pct,
            "unique_count": n_unique,
            "cardinality_ratio": cardinality_ratio,
            "is_target": col == target_column,
        }

        if is_numeric and not series.dropna().empty:
            s_clean = series.dropna()
            col_stats["min"] = _safe_float(s_clean.min())
            col_stats["max"] = _safe_float(s_clean.max())
            col_stats["mean"] = _safe_float(s_clean.mean())
            col_stats["std"] = _safe_float(s_clean.std())
            col_stats["skewness"] = _safe_float(s_clean.skew()) if len(s_clean) > 2 else 0.0
            col_stats["kurtosis"] = _safe_float(s_clean.kurt()) if len(s_clean) > 3 else 0.0
            col_stats["is_non_negative"] = bool((s_clean >= 0).all())

        columns_profile.append(col_stats)

    # Correlation Matrix on Numeric Features (excluding constant columns)
    collinear_pairs = []
    correlation_matrix = {}
    eligible_num_cols = [c for c in numeric_cols if c not in constant_cols]
    if len(eligible_num_cols) >= 2:
        try:
            num_df = profile_df[eligible_num_cols].dropna()
            if num_df.shape[0] >= 5:
                corr = num_df.corr(method="pearson").round(4)
                for c in corr.columns:
                    correlation_matrix[c] = {}
                    for row_col in corr.index:
                        val = corr.loc[row_col, c]
                        correlation_matrix[c][row_col] = _safe_float(val)

                # Find pairs exceeding threshold
                cols_list = corr.columns.tolist()
                for i in range(len(cols_list)):
                    for j in range(i + 1, len(cols_list)):
                        col_a, col_b = cols_list[i], cols_list[j]
                        if col_a == target_column or col_b == target_column:
                            continue
                        r_val = corr.iloc[i, j]
                        if r_val is not None and not np.isnan(r_val) and abs(r_val) >= correlation_threshold:
                            collinear_pairs.append({
                                "feature_a": col_a,
                                "feature_b": col_b,
                                "correlation": float(round(r_val, 4)),
                                "abs_correlation": round(abs(float(r_val)), 4),
                                "action": f"Redundant pair (|r| >= {correlation_threshold}). Consider removing one.",
                            })
        except Exception:
            pass

    return _json_safe({
        "dataset_summary": {
            "total_rows": initial_rows,
            "total_columns": total_cols,
            "duplicate_rows": duplicate_rows_count,
            "duplicate_rows_pct": duplicate_rows_pct,
            "numeric_columns_count": len(numeric_cols),
            "categorical_columns_count": len(categorical_cols),
            "datetime_columns_count": len(datetime_candidates),
            "text_columns_count": len(text_candidates),
            "is_sampled": is_sampled,
            "sampled_rows": max_sample_rows if is_sampled else initial_rows,
        },
        "target_column": target_column,
        "columns": columns_profile,
        "quality_alerts": quality_alerts,
        "redundant_pairs": collinear_pairs,
        "correlation_matrix": correlation_matrix,
        "datetime_candidates": datetime_candidates,
        "text_candidates": text_candidates,
        "constant_columns": constant_cols,
        "near_constant_columns": near_constant_cols,
        "id_like_columns": id_like_cols,
        "eda_analysis_id": eda_analysis_id,
    })


# =========================================================
# Custom Transformers for Leakage-Free Scikit-Learn Pipelines
# =========================================================

class TabularFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Advanced, leakage-free feature engineering transformer supporting:
    - Calendar component extraction & Cyclical Sine/Cosine encoding
    - Time-series past-only lag features & past-only rolling statistics (zero leakage)
    - Safe log1p and Square-root transformations on non-negative numerical columns
    - Yeo-Johnson power transformations with parameters fit strictly on training slice
    - Outlier percentile clipping learned strictly during fit
    - Rare category grouping and frequency encoding learned strictly during fit
    - Bounded pairwise numerical interactions (multiplication, safe division with epsilon)
    - Lightweight text features (character count, word count, avg word length, emptiness)
    - Full feature lineage tracking for explainability
    """
    def __init__(
        self,
        datetime_cols: Optional[List[str]] = None,
        drop_original_datetime: bool = True,
        cyclical_datetime: bool = False,
        log_cols: Optional[List[str]] = None,
        yeo_johnson_cols: Optional[List[str]] = None,
        sqrt_cols: Optional[List[str]] = None,
        clip_outliers: bool = False,
        outlier_cols: Optional[List[str]] = None,
        outlier_lower_pct: float = 1.0,
        outlier_upper_pct: float = 99.0,
        group_rare_categories: bool = False,
        rare_cat_cols: Optional[List[str]] = None,
        rare_cat_min_freq: float = 0.02,
        frequency_encode_cols: Optional[List[str]] = None,
        interaction_pairs: Optional[List[Tuple[str, str, str]]] = None,
        polynomial_cols: Optional[List[str]] = None,
        polynomial_degree: int = 2,
        max_interaction_features: int = 25,
        time_series_col: Optional[str] = None,
        lag_features: Optional[Dict[str, List[int]]] = None,
        rolling_features: Optional[Dict[str, List[int]]] = None,
        text_cols: Optional[List[str]] = None,
        drop_raw_text: bool = True,
        tfidf_cols: Optional[List[str]] = None,
        tfidf_max_features: int = 15,
    ):
        self.datetime_cols = datetime_cols
        self.drop_original_datetime = drop_original_datetime
        self.cyclical_datetime = cyclical_datetime
        self.log_cols = log_cols
        self.yeo_johnson_cols = yeo_johnson_cols
        self.sqrt_cols = sqrt_cols
        self.clip_outliers = clip_outliers
        self.outlier_cols = outlier_cols
        self.outlier_lower_pct = outlier_lower_pct
        self.outlier_upper_pct = outlier_upper_pct
        self.group_rare_categories = group_rare_categories
        self.rare_cat_cols = rare_cat_cols
        self.rare_cat_min_freq = rare_cat_min_freq
        self.frequency_encode_cols = frequency_encode_cols
        self.interaction_pairs = interaction_pairs
        self.polynomial_cols = polynomial_cols
        self.polynomial_degree = polynomial_degree
        self.max_interaction_features = max_interaction_features
        self.time_series_col = time_series_col
        self.lag_features = lag_features
        self.rolling_features = rolling_features
        self.text_cols = text_cols
        self.drop_raw_text = drop_raw_text
        self.tfidf_cols = tfidf_cols
        self.tfidf_max_features = tfidf_max_features

        # Learned parameters during fit
        self.clipping_bounds_: Dict[str, Tuple[float, float]] = {}
        self.frequent_categories_: Dict[str, List[Any]] = {}
        self.frequency_maps_: Dict[str, Dict[Any, float]] = {}
        self.yeo_transformers_: Dict[str, PowerTransformer] = {}
        self.tfidf_vectorizers_: Dict[str, TfidfVectorizer] = {}
        self.output_feature_names_: List[str] = []
        self.feature_lineage_: List[Dict[str, Any]] = []

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # 1. Outlier clipping bounds learned strictly on training slice
        if self.clip_outliers and self.outlier_cols:
            self.clipping_bounds_ = {}
            for col in self.outlier_cols:
                if col in X.columns and pd.api.types.is_numeric_dtype(X[col]):
                    valid_series = X[col].dropna()
                    if not valid_series.empty:
                        low = float(np.percentile(valid_series, self.outlier_lower_pct))
                        high = float(np.percentile(valid_series, self.outlier_upper_pct))
                        self.clipping_bounds_[col] = (low, high)

        # 2. Frequent categories learned strictly on training slice
        if self.group_rare_categories and self.rare_cat_cols:
            self.frequent_categories_ = {}
            n_rows = len(X)
            min_count = max(2, int(n_rows * self.rare_cat_min_freq))
            for col in self.rare_cat_cols:
                if col in X.columns:
                    counts = X[col].value_counts(dropna=True)
                    frequent = counts[counts >= min_count].index.tolist()
                    self.frequent_categories_[col] = frequent

        # 3. Frequency encoding distributions learned on training slice
        if self.frequency_encode_cols:
            self.frequency_maps_ = {}
            for col in self.frequency_encode_cols:
                if col in X.columns:
                    freq = X[col].value_counts(normalize=True).to_dict()
                    self.frequency_maps_[col] = freq

        # 4. Yeo-Johnson PowerTransformers fitted strictly on training slice
        if self.yeo_johnson_cols:
            self.yeo_transformers_ = {}
            for col in self.yeo_johnson_cols:
                if col in X.columns and pd.api.types.is_numeric_dtype(X[col]):
                    series = X[col].dropna()
                    if len(series) >= 5:
                        pt = PowerTransformer(method="yeo-johnson", standardize=False)
                        try:
                            pt.fit(series.values.reshape(-1, 1))
                            self.yeo_transformers_[col] = pt
                        except Exception:
                            pass

        # 5. TF-IDF vectorizers fitted strictly on training slice
        if self.tfidf_cols:
            self.tfidf_vectorizers_ = {}
            for col in self.tfidf_cols:
                if col in X.columns:
                    text_vals = X[col].fillna("").astype(str)
                    vec = TfidfVectorizer(max_features=self.tfidf_max_features, stop_words="english", lowercase=True)
                    try:
                        vec.fit(text_vals)
                        self.tfidf_vectorizers_[col] = vec
                    except Exception:
                        pass

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X_out = X.copy()
        lineage = []

        # Record original features
        for col in X.columns:
            lineage.append({
                "feature_name": col,
                "parent_columns": [col],
                "operation": "original_input",
                "parameters": {},
                "is_engineered": False,
            })

        # 1. Datetime extraction & Cyclical sine/cosine
        for dt_col in (self.datetime_cols or []):
            if dt_col in X_out.columns:
                try:
                    dt_series = pd.to_datetime(X_out[dt_col], errors="coerce")
                    y_col = f"{dt_col}_year"
                    m_col = f"{dt_col}_month"
                    d_col = f"{dt_col}_day"
                    dow_col = f"{dt_col}_dayofweek"
                    wk_col = f"{dt_col}_is_weekend"
                    q_col = f"{dt_col}_quarter"

                    X_out[y_col] = dt_series.dt.year.fillna(2000).astype(int)
                    X_out[m_col] = dt_series.dt.month.fillna(1).astype(int)
                    X_out[d_col] = dt_series.dt.day.fillna(1).astype(int)
                    X_out[dow_col] = dt_series.dt.dayofweek.fillna(0).astype(int)
                    X_out[wk_col] = (X_out[dow_col] >= 5).astype(int)
                    X_out[q_col] = dt_series.dt.quarter.fillna(1).astype(int)

                    lineage.extend([
                        {"feature_name": y_col, "parent_columns": [dt_col], "operation": "calendar_year", "parameters": {}, "is_engineered": True},
                        {"feature_name": m_col, "parent_columns": [dt_col], "operation": "calendar_month", "parameters": {}, "is_engineered": True},
                        {"feature_name": d_col, "parent_columns": [dt_col], "operation": "calendar_day", "parameters": {}, "is_engineered": True},
                        {"feature_name": dow_col, "parent_columns": [dt_col], "operation": "calendar_dayofweek", "parameters": {}, "is_engineered": True},
                        {"feature_name": wk_col, "parent_columns": [dt_col], "operation": "calendar_weekend", "parameters": {}, "is_engineered": True},
                        {"feature_name": q_col, "parent_columns": [dt_col], "operation": "calendar_quarter", "parameters": {}, "is_engineered": True},
                    ])

                    if self.cyclical_datetime:
                        m_sin = f"{dt_col}_month_sin"
                        m_cos = f"{dt_col}_month_cos"
                        dow_sin = f"{dt_col}_dayofweek_sin"
                        dow_cos = f"{dt_col}_dayofweek_cos"

                        X_out[m_sin] = np.round(np.sin(2 * np.pi * X_out[m_col] / 12.0), 4)
                        X_out[m_cos] = np.round(np.cos(2 * np.pi * X_out[m_col] / 12.0), 4)
                        X_out[dow_sin] = np.round(np.sin(2 * np.pi * X_out[dow_col] / 7.0), 4)
                        X_out[dow_cos] = np.round(np.cos(2 * np.pi * X_out[dow_col] / 7.0), 4)

                        lineage.extend([
                            {"feature_name": m_sin, "parent_columns": [dt_col], "operation": "cyclical_sine_month", "parameters": {"period": 12}, "is_engineered": True},
                            {"feature_name": m_cos, "parent_columns": [dt_col], "operation": "cyclical_cosine_month", "parameters": {"period": 12}, "is_engineered": True},
                            {"feature_name": dow_sin, "parent_columns": [dt_col], "operation": "cyclical_sine_dayofweek", "parameters": {"period": 7}, "is_engineered": True},
                            {"feature_name": dow_cos, "parent_columns": [dt_col], "operation": "cyclical_cosine_dayofweek", "parameters": {"period": 7}, "is_engineered": True},
                        ])

                    if self.drop_original_datetime:
                        X_out.drop(columns=[dt_col], inplace=True)
                except Exception:
                    pass

        # 2. Time-series past-only lags and rolling stats (Strictly past-only, zero future leakage)
        if self.lag_features:
            for col, lags in self.lag_features.items():
                if col in X_out.columns:
                    for k in lags:
                        if k >= 1:
                            lag_name = f"{col}_lag_{k}"
                            X_out[lag_name] = X_out[col].shift(k).bfill().fillna(0)
                            lineage.append({
                                "feature_name": lag_name,
                                "parent_columns": [col],
                                "operation": "past_only_lag",
                                "parameters": {"lag_order": k},
                                "is_engineered": True,
                            })

        if self.rolling_features:
            for col, windows in self.rolling_features.items():
                if col in X_out.columns:
                    for w in windows:
                        if w >= 2:
                            r_mean = f"{col}_rolling_mean_{w}"
                            r_std = f"{col}_rolling_std_{w}"
                            # .shift(1) ensures the rolling window at time t uses past observations up to t-1
                            past_series = X_out[col].shift(1)
                            X_out[r_mean] = past_series.rolling(w, min_periods=1).mean().bfill().fillna(0)
                            X_out[r_std] = past_series.rolling(w, min_periods=1).std().fillna(0)
                            lineage.extend([
                                {"feature_name": r_mean, "parent_columns": [col], "operation": "rolling_mean", "parameters": {"window": w}, "is_engineered": True},
                                {"feature_name": r_std, "parent_columns": [col], "operation": "rolling_std", "parameters": {"window": w}, "is_engineered": True},
                            ])

        # 3. Safe Log1p Transformation on non-negative values
        for l_col in (self.log_cols or []):
            if l_col in X_out.columns and pd.api.types.is_numeric_dtype(X_out[l_col]):
                X_out[l_col] = np.log1p(np.maximum(0, X_out[l_col].fillna(0).astype(float)))
                lineage.append({
                    "feature_name": l_col,
                    "parent_columns": [l_col],
                    "operation": "log1p_transform",
                    "parameters": {"non_negative_clipped": True},
                    "is_engineered": True,
                })

        # 4. Safe Sqrt Transformation
        for sq_col in (self.sqrt_cols or []):
            if sq_col in X_out.columns and pd.api.types.is_numeric_dtype(X_out[sq_col]):
                new_col = f"{sq_col}_sqrt"
                X_out[new_col] = np.sqrt(np.maximum(0, X_out[sq_col].fillna(0).astype(float)))
                lineage.append({
                    "feature_name": new_col,
                    "parent_columns": [sq_col],
                    "operation": "sqrt_transform",
                    "parameters": {},
                    "is_engineered": True,
                })

        # 5. Yeo-Johnson Power Transformation
        for y_col in (self.yeo_johnson_cols or []):
            if y_col in X_out.columns and y_col in self.yeo_transformers_:
                pt = self.yeo_transformers_[y_col]
                try:
                    med = float(X_out[y_col].median()) if not X_out[y_col].dropna().empty else 0.0
                    clean_vals = X_out[y_col].fillna(med).values.reshape(-1, 1)
                    X_out[y_col] = pt.transform(clean_vals).flatten()
                    lineage.append({
                        "feature_name": y_col,
                        "parent_columns": [y_col],
                        "operation": "yeo_johnson_power",
                        "parameters": {"standardize": False},
                        "is_engineered": True,
                    })
                except Exception:
                    pass

        # 6. Outlier Clipping
        if self.clip_outliers:
            for col, (low, high) in self.clipping_bounds_.items():
                if col in X_out.columns and pd.api.types.is_numeric_dtype(X_out[col]):
                    X_out[col] = X_out[col].clip(lower=low, upper=high)
                    lineage.append({
                        "feature_name": col,
                        "parent_columns": [col],
                        "operation": "outlier_percentile_clipping",
                        "parameters": {"lower_bound": low, "upper_bound": high},
                        "is_engineered": True,
                    })

        # 7. Rare category grouping
        if self.group_rare_categories:
            for col, frequent in self.frequent_categories_.items():
                if col in X_out.columns:
                    X_out[col] = X_out[col].apply(
                        lambda val: val if (pd.isna(val) or val in frequent) else "__other__"
                    )
                    lineage.append({
                        "feature_name": col,
                        "parent_columns": [col],
                        "operation": "rare_category_grouping",
                        "parameters": {"retained_frequent_count": len(frequent)},
                        "is_engineered": True,
                    })

        # 8. Frequency Encoding
        if self.frequency_encode_cols:
            for col in self.frequency_encode_cols:
                if col in X_out.columns and col in self.frequency_maps_:
                    f_map = self.frequency_maps_[col]
                    freq_name = f"{col}_freq"
                    X_out[freq_name] = X_out[col].map(f_map).fillna(0.0).astype(float)
                    lineage.append({
                        "feature_name": freq_name,
                        "parent_columns": [col],
                        "operation": "frequency_encoding",
                        "parameters": {},
                        "is_engineered": True,
                    })

        # 9. Bounded Pairwise Numerical Interactions
        if self.interaction_pairs:
            bounded_pairs = self.interaction_pairs[: self.max_interaction_features]
            for col_a, col_b, op in bounded_pairs:
                if col_a in X_out.columns and col_b in X_out.columns:
                    s_a = pd.to_numeric(X_out[col_a], errors="coerce").fillna(0).astype(float)
                    s_b = pd.to_numeric(X_out[col_b], errors="coerce").fillna(0).astype(float)

                    if op in {"multiply", "product", "*"}:
                        inter_name = f"{col_a}_x_{col_b}"
                        X_out[inter_name] = s_a * s_b
                        lineage.append({"feature_name": inter_name, "parent_columns": [col_a, col_b], "operation": "interaction_multiply", "parameters": {}, "is_engineered": True})
                    elif op in {"divide", "ratio", "/"}:
                        inter_name = f"{col_a}_div_{col_b}"
                        safe_b = np.where(np.abs(s_b) < 1e-6, np.sign(s_b + 1e-6) * 1e-6, s_b)
                        ratio = np.clip(s_a / safe_b, -1e6, 1e6)
                        X_out[inter_name] = pd.Series(ratio, index=X_out.index).fillna(0)
                        lineage.append({"feature_name": inter_name, "parent_columns": [col_a, col_b], "operation": "interaction_safe_ratio", "parameters": {"epsilon": 1e-6}, "is_engineered": True})
                    elif op in {"add", "sum", "+"}:
                        inter_name = f"{col_a}_add_{col_b}"
                        X_out[inter_name] = s_a + s_b
                        lineage.append({"feature_name": inter_name, "parent_columns": [col_a, col_b], "operation": "interaction_sum", "parameters": {}, "is_engineered": True})
                    elif op in {"subtract", "diff", "-"}:
                        inter_name = f"{col_a}_sub_{col_b}"
                        X_out[inter_name] = s_a - s_b
                        lineage.append({"feature_name": inter_name, "parent_columns": [col_a, col_b], "operation": "interaction_difference", "parameters": {}, "is_engineered": True})

        # 10. Polynomial squared terms
        if self.polynomial_cols:
            for p_col in self.polynomial_cols:
                if p_col in X_out.columns and pd.api.types.is_numeric_dtype(X_out[p_col]):
                    sq_name = f"{p_col}_squared"
                    num_val = pd.to_numeric(X_out[p_col], errors="coerce").fillna(0).astype(float)
                    X_out[sq_name] = num_val ** 2
                    lineage.append({"feature_name": sq_name, "parent_columns": [p_col], "operation": "polynomial_squared", "parameters": {"degree": 2}, "is_engineered": True})

        # 11. Lightweight Text Features
        if self.text_cols:
            for t_col in self.text_cols:
                if t_col in X_out.columns:
                    str_s = X_out[t_col].fillna("").astype(str)
                    char_c = f"{t_col}_char_count"
                    word_c = f"{t_col}_word_count"
                    avg_w = f"{t_col}_avg_word_len"
                    is_emp = f"{t_col}_is_empty"

                    X_out[char_c] = str_s.apply(len).astype(int)
                    X_out[word_c] = str_s.apply(lambda s: len(s.split())).astype(int)
                    X_out[avg_w] = str_s.apply(lambda s: round(len(s) / max(1, len(s.split())), 2)).astype(float)
                    X_out[is_emp] = (str_s.str.strip() == "").astype(int)

                    lineage.extend([
                        {"feature_name": char_c, "parent_columns": [t_col], "operation": "text_char_count", "parameters": {}, "is_engineered": True},
                        {"feature_name": word_c, "parent_columns": [t_col], "operation": "text_word_count", "parameters": {}, "is_engineered": True},
                        {"feature_name": avg_w, "parent_columns": [t_col], "operation": "text_avg_word_length", "parameters": {}, "is_engineered": True},
                        {"feature_name": is_emp, "parent_columns": [t_col], "operation": "text_is_empty", "parameters": {}, "is_engineered": True},
                    ])

                    if self.drop_raw_text:
                        X_out.drop(columns=[t_col], inplace=True)

        # 12. Bounded TF-IDF features
        if self.tfidf_cols:
            for col in self.tfidf_cols:
                if col in self.tfidf_vectorizers_ and col in X.columns:
                    vec = self.tfidf_vectorizers_[col]
                    try:
                        str_data = X[col].fillna("").astype(str)
                        sparse_mat = vec.transform(str_data).toarray()
                        feature_names = vec.get_feature_names_out()
                        for idx, feat in enumerate(feature_names):
                            clean_tok = re.sub(r"[^a-zA-Z0-9_]", "", str(feat))
                            t_feat_name = f"{col}_tfidf_{clean_tok}"
                            X_out[t_feat_name] = sparse_mat[:, idx]
                            lineage.append({
                                "feature_name": t_feat_name,
                                "parent_columns": [col],
                                "operation": "tfidf_vectorizer",
                                "parameters": {"token": str(feat)},
                                "is_engineered": True,
                            })
                    except Exception:
                        pass

        self.output_feature_names_ = list(X_out.columns)
        self.feature_lineage_ = lineage
        return X_out

    def get_feature_names_out(self, input_features=None):
        return np.array(self.output_feature_names_)

    def get_lineage(self) -> List[Dict[str, Any]]:
        return getattr(self, "feature_lineage_", [])


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """
    Filters redundant numeric features exceeding a correlation threshold (Pearson or Spearman).
    Correlation matrix is fit strictly on training slice.
    """
    def __init__(self, threshold: float = 0.9, method: str = "pearson"):
        self.threshold = threshold
        self.method = method
        self.dropped_columns_: List[str] = []
        self.retained_columns_: List[str] = []

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        numeric_df = X.select_dtypes(include=[np.number])
        self.dropped_columns_ = []

        if numeric_df.shape[1] >= 2 and numeric_df.shape[0] >= 5:
            corr_matrix = numeric_df.corr(method=self.method).abs()
            cols = corr_matrix.columns.tolist()
            to_drop = set()

            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    col_i, col_j = cols[i], cols[j]
                    if col_i in to_drop or col_j in to_drop:
                        continue
                    val = corr_matrix.iloc[i, j]
                    if val is not None and not np.isnan(val) and val >= self.threshold:
                        to_drop.add(col_j)

            self.dropped_columns_ = sorted(list(to_drop))

        self.retained_columns_ = [c for c in X.columns if c not in self.dropped_columns_]
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        cols = [c for c in self.retained_columns_ if c in X.columns]
        return X[cols]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.retained_columns_)

    def get_support(self, indices=False):
        if not hasattr(self, "retained_columns_"):
            return np.ones(1, dtype=bool)
        # All columns in retained_columns_ get True
        mask = [True] * len(self.retained_columns_)
        return np.array(mask, dtype=bool)


class EnhancedFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Intelligent unified feature selection transformer supporting:
    - VarianceThreshold (filters zero/near-zero variance)
    - CorrelationFilter (Pearson & Spearman collinearity)
    - Mutual Information (non-linear dependency selection)
    - Tree-based & ExtraTrees importance (SelectFromModel)
    - L1 Regularization (Lasso / Logistic L1 sparsity)
    - Bounded Recursive Feature Elimination (RFE)
    - SelectKBest (ANOVA F-test)
    - Passthrough ('none')
    """
    def __init__(
        self,
        method: str = "none",
        variance_threshold: float = 0.01,
        correlation_threshold: float = 0.85,
        k_best_count: int = 10,
        task_type: str = "classification",
        random_state: int = 42,
    ):
        self.method = method
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.k_best_count = k_best_count
        self.task_type = task_type
        self.random_state = random_state
        self.selector_ = None
        self.support_mask_: Optional[np.ndarray] = None
        self.retained_columns_: List[str] = []

    def fit(self, X, y=None):
        if hasattr(X, "toarray"):
            X = X.toarray()
        if isinstance(X, pd.DataFrame):
            col_names = list(X.columns)
            X_mat = X.to_numpy()
        else:
            X_mat = np.asarray(X)
            col_names = [f"f_{i}" for i in range(X_mat.shape[1])]

        n_features = X_mat.shape[1]
        k_val = max(1, min(self.k_best_count or 10, n_features))

        if self.method == "variance":
            self.selector_ = VarianceThreshold(threshold=self.variance_threshold)
            self.selector_.fit(X_mat, y)
            self.support_mask_ = self.selector_.get_support()

        elif self.method == "correlation":
            corr_filt = CorrelationFilter(threshold=self.correlation_threshold)
            df_x = pd.DataFrame(X_mat, columns=col_names)
            corr_filt.fit(df_x, y)
            self.selector_ = corr_filt
            self.support_mask_ = np.array([c in corr_filt.retained_columns_ for c in col_names], dtype=bool)

        elif self.method == "mutual_info":
            score_func = mutual_info_classif if self.task_type == "classification" else mutual_info_regression
            self.selector_ = SelectKBest(score_func=score_func, k=k_val)
            self.selector_.fit(X_mat, y)
            self.support_mask_ = self.selector_.get_support()

        elif self.method == "model_based":
            base_est = (
                RandomForestClassifier(n_estimators=50, max_depth=8, n_jobs=1, random_state=self.random_state)
                if self.task_type == "classification"
                else RandomForestRegressor(n_estimators=50, max_depth=8, n_jobs=1, random_state=self.random_state)
            )
            self.selector_ = SelectFromModel(estimator=base_est, max_features=k_val)
            self.selector_.fit(X_mat, y)
            self.support_mask_ = self.selector_.get_support()

        elif self.method == "l1_regularized":
            if self.task_type == "classification":
                base_est = LogisticRegression(penalty="l1", solver="liblinear", C=0.2, random_state=self.random_state)
            else:
                base_est = Lasso(alpha=0.05, random_state=self.random_state)
            self.selector_ = SelectFromModel(estimator=base_est, max_features=k_val)
            self.selector_.fit(X_mat, y)
            self.support_mask_ = self.selector_.get_support()

        elif self.method == "rfe":
            # Fast estimator for bounded RFE
            fast_est = (
                DecisionTreeClassifier(max_depth=6, random_state=self.random_state)
                if self.task_type == "classification"
                else DecisionTreeRegressor(max_depth=6, random_state=self.random_state)
            )
            self.selector_ = RFE(estimator=fast_est, n_features_to_select=k_val, step=0.25)
            self.selector_.fit(X_mat, y)
            self.support_mask_ = self.selector_.support_

        elif self.method == "k_best":
            score_fn = f_classif if self.task_type == "classification" else f_regression
            self.selector_ = SelectKBest(score_func=score_fn, k=k_val)
            self.selector_.fit(X_mat, y)
            self.support_mask_ = self.selector_.get_support()

        else:
            # passthrough ('none')
            self.support_mask_ = np.ones(n_features, dtype=bool)

        self.retained_columns_ = [col_names[i] for i, m in enumerate(self.support_mask_) if m]
        return self

    def transform(self, X):
        if hasattr(X, "toarray"):
            X = X.toarray()
        if isinstance(X, pd.DataFrame):
            X_mat = X.to_numpy()
        else:
            X_mat = np.asarray(X)

        if self.support_mask_ is not None and len(self.support_mask_) == X_mat.shape[1]:
            return X_mat[:, self.support_mask_]
        return X_mat

    def get_support(self, indices=False):
        if self.support_mask_ is None:
            return np.ones(1, dtype=bool)
        if indices:
            return np.where(self.support_mask_)[0]
        return self.support_mask_

    def get_feature_names_out(self, input_features=None):
        if input_features is not None and self.support_mask_ is not None:
            return np.array([input_features[i] for i, m in enumerate(self.support_mask_) if m])
        return np.array(self.retained_columns_)


# =========================================================
# Pipeline Builders & Evaluation Services
# =========================================================

def build_feature_preprocessor(
    numeric_columns: List[str],
    categorical_columns: List[str],
    scaler: str = "standard",
    missing: str = "median",
    target_encode_cols: Optional[List[str]] = None,
    task_type: str = "classification",
) -> ColumnTransformer:
    """
    Constructs a ColumnTransformer for numerical imputation & scaling,
    categorical one-hot encoding, and zero-leakage out-of-fold Target Encoding.
    """
    transformers = []
    target_enc_set = set(target_encode_cols or [])
    ohe_cats = [c for c in categorical_columns if c not in target_enc_set]

    if numeric_columns:
        num_imputer = "mean" if missing == "mean" else "median"
        num_steps = [("imputer", SimpleImputer(strategy=num_imputer))]
        if scaler == "standard":
            num_steps.append(("scaler", StandardScaler()))
        elif scaler == "minmax":
            num_steps.append(("scaler", MinMaxScaler()))
        elif scaler == "robust":
            num_steps.append(("scaler", RobustScaler()))
        elif scaler == "quantile":
            num_steps.append(("scaler", QuantileTransformer(output_distribution="normal", random_state=42)))
        transformers.append(("num", Pipeline(steps=num_steps), numeric_columns))

    if ohe_cats:
        cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("cat_ohe", cat_pipeline, ohe_cats))

    if target_encode_cols:
        target_type = "continuous" if task_type == "regression" else "multiclass"
        te_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("target_encoder", TargetEncoder(smooth="auto", cv=3, random_state=42)),
            ]
        )
        transformers.append(("cat_te", te_pipeline, list(target_enc_set)))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def _instantiate_model(model_type: str, random_state: int = 42):
    if model_type == "RandomForestClassifier":
        return RandomForestClassifier(n_estimators=100, class_weight="balanced", n_jobs=1, random_state=random_state)
    elif model_type == "DecisionTreeClassifier":
        return DecisionTreeClassifier(class_weight="balanced", random_state=random_state)
    elif model_type == "ExtraTreesClassifier":
        return ExtraTreesClassifier(n_estimators=100, class_weight="balanced", n_jobs=1, random_state=random_state)
    elif model_type == "LogisticRegression":
        return LogisticRegression(class_weight="balanced", max_iter=1000, solver="lbfgs", random_state=random_state)
    elif model_type == "RandomForest":
        return RandomForestRegressor(n_estimators=100, n_jobs=1, random_state=random_state)
    elif model_type == "DecisionTree":
        return DecisionTreeRegressor(random_state=random_state)
    elif model_type == "ExtraTreesRegressor":
        return ExtraTreesRegressor(n_estimators=100, n_jobs=1, random_state=random_state)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported model type: {model_type}")


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str,
    estimator: Any = None,
    X_val: Any = None,
    unique_classes: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    if task_type == "classification":
        acc = _safe_float(accuracy_score(y_true, y_pred))
        bal_acc = _safe_float(balanced_accuracy_score(y_true, y_pred))
        f1_mac = _safe_float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        f1_wgt = _safe_float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

        roc_auc = None
        log_loss_val = None
        if hasattr(estimator, "predict_proba") and X_val is not None and unique_classes:
            try:
                probs = estimator.predict_proba(X_val)
                present_classes = set(pd.Series(y_true).unique())
                if len(present_classes) == len(unique_classes):
                    if len(unique_classes) == 2:
                        roc_auc = _safe_float(roc_auc_score(y_true, probs[:, 1]))
                    else:
                        roc_auc = _safe_float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro", labels=unique_classes))
                log_loss_val = _safe_float(log_loss(y_true, probs, labels=unique_classes))
            except Exception:
                pass

        return {
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "f1_macro": f1_mac,
            "f1_weighted": f1_wgt,
            "roc_auc": roc_auc,
            "log_loss": log_loss_val,
            "testing_samples": int(len(y_true)),
        }
    else:
        actual = np.asarray(y_true, dtype=float)
        preds = np.asarray(y_pred, dtype=float)
        return {
            "rmse": _safe_float(np.sqrt(mean_squared_error(actual, preds))),
            "mae": _safe_float(mean_absolute_error(actual, preds)),
            "r2": _safe_float(r2_score(actual, preds)),
            "explained_variance": _safe_float(explained_variance_score(actual, preds)),
            "max_error": _safe_float(sklearn_max_error(actual, preds)),
            "testing_samples": int(len(actual)),
        }


# =========================================================
# Phase 2 EDA Integration & Strategy Preset Selector
# =========================================================

def get_eda_feature_recommendations(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    eda_analysis_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Connects with Phase 2 EDA intelligence to evaluate the 11 dataset profiles
    and automatically configure Conservative, Balanced, and Advanced strategy presets.
    """
    from app.services.eda_service import get_eda_analysis_from_db, detect_eda_strategy

    eda_data = None
    if eda_analysis_id:
        eda_data = get_eda_analysis_from_db(eda_analysis_id)

    total_rows, total_cols = df.shape
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != target_column]
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and c != target_column]

    # Detect active profiles
    active_profiles = []
    if eda_data and "strategy" in eda_data:
        active_profiles = eda_data["strategy"].get("active_profiles", [])
    else:
        if len(num_cols) / max(1, total_cols) >= 0.65:
            active_profiles.append("numerical_heavy")
        if len(cat_cols) / max(1, total_cols) >= 0.60:
            active_profiles.append("categorical_heavy")
        if len(num_cols) >= 2 and len(cat_cols) >= 2:
            active_profiles.append("mixed_tabular")
        if total_rows < 100:
            active_profiles.append("small_dataset")
        if total_rows > 4000:
            active_profiles.append("large_dataset")
        if total_cols > 30:
            active_profiles.append("high_dimensional")

    # Detect skewed numerical features
    skewed_numeric = []
    for col in num_cols:
        s = df[col].dropna()
        if len(s) > 5 and abs(float(s.skew())) > 1.2:
            skewed_numeric.append(col)

    # Detect high-cardinality categoricals (> 15 unique values)
    high_card_cats = []
    for col in cat_cols:
        if df[col].nunique() > 15:
            high_card_cats.append(col)

    # Detect text-like columns
    text_cols = []
    for col in cat_cols:
        s = df[col].dropna().head(20).astype(str)
        if s.apply(len).mean() > 30.0:
            text_cols.append(col)

    # Detect datetime columns
    date_cols = []
    date_pattern = re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4}")
    for col in cat_cols:
        s = df[col].dropna().head(20).astype(str)
        if any(date_pattern.search(v) for v in s):
            date_cols.append(col)

    # Recommended pairwise interaction candidates (top 3-5 correlated numeric pairs)
    interaction_candidates = []
    if len(num_cols) >= 2:
        try:
            corr = df[num_cols].corr().abs()
            cols = corr.columns.tolist()
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    c1, c2 = cols[i], cols[j]
                    val = corr.loc[c1, c2]
                    if 0.35 <= val <= 0.85:
                        interaction_candidates.append([c1, c2, "multiply"])
                        interaction_candidates.append([c1, c2, "divide"])
        except Exception:
            pass

    # ---------------------------------------------------------
    # Construct Strategy Presets
    # ---------------------------------------------------------
    presets = {
        "conservative": {
            "name": "Conservative / Auto",
            "description": "Low-risk, parsimonious transformations. Standard scaling, calendar extraction, constant/collinear filtering, and rare category grouping.",
            "scaler": "standard",
            "imputation": "median",
            "selection_method": "variance",
            "variance_threshold": 0.01,
            "correlation_threshold": 0.90,
            "clip_outliers": True,
            "group_rare_categories": True,
            "datetime_cols": date_cols,
            "cyclical_datetime": False,
            "log_cols": skewed_numeric[:2],
            "yeo_johnson_cols": [],
            "interaction_pairs": [],
            "polynomial_cols": [],
            "target_encode_cols": [],
            "text_cols": [],
            "risk_profile": "Minimal",
            "estimated_speed": "Very Fast (< 2s)",
        },
        "balanced": {
            "name": "Balanced (Recommended)",
            "description": "Optimal trade-off between predictive lift and model complexity. Yeo-Johnson power transforms for skewed columns, cyclical datetime encoding, bounded numerical interactions, and mutual information selection.",
            "scaler": "robust" if len(skewed_numeric) > 0 else "standard",
            "imputation": "median",
            "selection_method": "mutual_info",
            "k_best_count": min(15, max(5, total_cols)),
            "variance_threshold": 0.01,
            "correlation_threshold": 0.85,
            "clip_outliers": True,
            "group_rare_categories": True,
            "datetime_cols": date_cols,
            "cyclical_datetime": True,
            "log_cols": [c for c in skewed_numeric if (df[c].dropna() >= 0).all()][:3],
            "yeo_johnson_cols": [c for c in skewed_numeric if not (df[c].dropna() >= 0).all()][:3],
            "interaction_pairs": interaction_candidates[:6],
            "polynomial_cols": num_cols[:2],
            "target_encode_cols": high_card_cats[:2],
            "text_cols": text_cols[:2],
            "risk_profile": "Low",
            "estimated_speed": "Fast (2 - 5s)",
        },
        "advanced": {
            "name": "Advanced / Research",
            "description": "Comprehensive exploratory transformation: Degree-2 polynomial expansions, arithmetic safe-ratio interactions, cyclical temporal representations, lightweight text metrics, and tree-based or RFE feature selection.",
            "scaler": "robust",
            "imputation": "median",
            "selection_method": "model_based",
            "k_best_count": min(20, max(8, total_cols)),
            "variance_threshold": 0.005,
            "correlation_threshold": 0.85,
            "clip_outliers": True,
            "group_rare_categories": True,
            "datetime_cols": date_cols,
            "cyclical_datetime": True,
            "log_cols": [c for c in skewed_numeric if (df[c].dropna() >= 0).all()],
            "yeo_johnson_cols": skewed_numeric[:5],
            "interaction_pairs": interaction_candidates[:12],
            "polynomial_cols": num_cols[:4],
            "target_encode_cols": high_card_cats,
            "text_cols": text_cols,
            "risk_profile": "Moderate",
            "estimated_speed": "Moderate (5 - 15s)",
        },
    }

    explanations = [
        f"Detected {len(num_cols)} numerical features, {len(cat_cols)} categoricals, and {len(date_cols)} datetime candidates across {total_rows} rows.",
    ]
    if "numerical_heavy" in active_profiles:
        explanations.append("Numerical-heavy dataset: Prioritizing Yeo-Johnson transformations and bounded ratio interactions.")
    if "categorical_heavy" in active_profiles:
        explanations.append("Categorical-heavy dataset: Recommending rare category grouping and out-of-fold target encoding.")
    if skewed_numeric:
        explanations.append(f"Detected {len(skewed_numeric)} skewed numerical features ({', '.join(skewed_numeric[:3])}); power transformations recommended.")
    if high_card_cats:
        explanations.append(f"Identified high-cardinality features ({', '.join(high_card_cats[:2])}); target encoding or frequency encoding suggested to avoid one-hot dimensionality explosion.")

    return _json_safe({
        "dataset_summary": {
            "total_rows": total_rows,
            "total_columns": total_cols,
            "numeric_count": len(num_cols),
            "categorical_count": len(cat_cols),
            "datetime_count": len(date_cols),
            "text_count": len(text_cols),
        },
        "active_profiles": active_profiles,
        "eda_analysis_id": eda_analysis_id,
        "recommendation_explanations": explanations,
        "presets": presets,
        "detected_skewed_features": skewed_numeric,
        "detected_high_cardinality_features": high_card_cats,
        "detected_datetime_features": date_cols,
        "detected_text_features": text_cols,
    })


# =========================================================
# Feature Strategy Evaluation Engine (Enhanced for Phase 3)
# =========================================================

async def evaluate_feature_strategies(
    file: UploadFile,
    task_type: str,
    target_column: str,
    model_type: str,
    ranking_metric: Optional[str] = None,
    n_splits: int = 3,
    split_ratio: float = 0.8,
    scaler: str = "standard",
    missing: str = "median",
    # Feature Engineering Config
    datetime_cols: Optional[List[str]] = None,
    cyclical_datetime: bool = False,
    log_cols: Optional[List[str]] = None,
    yeo_johnson_cols: Optional[List[str]] = None,
    sqrt_cols: Optional[List[str]] = None,
    clip_outliers: bool = False,
    group_rare_categories: bool = False,
    frequency_encode_cols: Optional[List[str]] = None,
    interaction_pairs: Optional[List[Tuple[str, str, str]]] = None,
    polynomial_cols: Optional[List[str]] = None,
    target_encode_cols: Optional[List[str]] = None,
    text_cols: Optional[List[str]] = None,
    # Feature Selection Config
    selection_method: str = "none",
    variance_threshold: float = 0.01,
    correlation_threshold: float = 0.85,
    k_best_count: int = 10,
    random_state: int = 42,
    eda_analysis_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluates baseline vs feature-engineered vs feature-selected vs combined pipelines using
    strict training-only cross-validation. Evaluates winning pipeline on untouched holdout partition.
    """
    start_time = time.time()
    init_feature_db()
    run_id = uuid.uuid4().hex
    warnings: List[str] = []

    normalized_task = (task_type or "").strip().lower()
    if normalized_task not in {"classification", "regression"}:
        raise HTTPException(status_code=400, detail="task_type must be 'classification' or 'regression'.")

    # Read dataset
    dataset_name = file.filename or "dataset.csv"
    df = await safe_read_csv(file)

    if target_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found.")

    df_clean = df.dropna(subset=[target_column]).copy()
    if df_clean.empty:
        raise HTTPException(status_code=400, detail=f"Target '{target_column}' contains only missing values.")

    unique_classes: List[Any] = []
    if normalized_task == "classification":
        class_counts = df_clean[target_column].value_counts()
        if len(class_counts) < 2:
            raise HTTPException(status_code=400, detail="Classification requires at least 2 target classes.")
        unique_classes = sorted(list(class_counts.index.astype(str)))
        ranking_metric = (ranking_metric or "f1_macro").strip().lower()
    else:
        if not pd.api.types.is_numeric_dtype(df_clean[target_column]):
            raise HTTPException(status_code=400, detail="Regression requires numeric target.")
        ranking_metric = (ranking_metric or "rmse").strip().lower()

    higher_is_better = ranking_metric in {"accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "roc_auc", "r2"}

    feature_cols = [c for c in df_clean.columns if c != target_column]
    if not feature_cols:
        raise HTTPException(status_code=400, detail="No feature columns found.")

    X = df_clean[feature_cols]
    y = df_clean[target_column]

    # Zero-leakage holdout split
    stratify_target = y if (normalized_task == "classification" and y.value_counts().min() >= 2) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=split_ratio, random_state=random_state, stratify=stratify_target
    )

    actual_splits = max(2, min(int(n_splits), 5))
    if normalized_task == "classification":
        min_class = int(y_train.value_counts().min())
        if min_class < actual_splits:
            actual_splits = max(2, min_class)
            warnings.append(f"CV folds adjusted to {actual_splits} due to class sample size.")

    # Detect initial column types
    num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(X_train[c])]
    cat_cols = [c for c in feature_cols if c not in num_cols]

    # Validate feature selection options
    valid_selection_methods = {"none", "variance", "correlation", "model_based", "k_best", "mutual_info", "rfe", "l1_regularized"}
    if selection_method not in valid_selection_methods:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid selection_method '{selection_method}'. Supported: {sorted(valid_selection_methods)}",
        )

    # ---------------------------------------------------------
    # Helper to construct candidate pipelines
    # ---------------------------------------------------------
    def create_candidate_pipeline(enable_engineering: bool, enable_selection: bool) -> Tuple[Pipeline, List[str]]:
        steps = []
        expected_features = list(feature_cols)

        # 1. Feature Engineering Step (if enabled)
        if enable_engineering:
            fe_transformer = TabularFeatureEngineer(
                datetime_cols=datetime_cols,
                drop_original_datetime=True,
                cyclical_datetime=cyclical_datetime,
                log_cols=log_cols,
                yeo_johnson_cols=yeo_johnson_cols,
                sqrt_cols=sqrt_cols,
                clip_outliers=clip_outliers,
                outlier_cols=num_cols,
                group_rare_categories=group_rare_categories,
                rare_cat_cols=cat_cols,
                frequency_encode_cols=frequency_encode_cols,
                interaction_pairs=interaction_pairs,
                polynomial_cols=polynomial_cols,
                text_cols=text_cols,
            )
            steps.append(("feature_engineer", fe_transformer))

            # Transform sample to obtain updated column structure
            sample_trans = fe_transformer.fit_transform(X_train.head(10))
            current_num = [c for c in sample_trans.columns if pd.api.types.is_numeric_dtype(sample_trans[c])]
            current_cat = [c for c in sample_trans.columns if c not in current_num]
        else:
            current_num = num_cols
            current_cat = cat_cols

        # 2. Preprocessing Step
        preprocessor = build_feature_preprocessor(
            numeric_columns=current_num,
            categorical_columns=current_cat,
            scaler=scaler,
            missing=missing,
            target_encode_cols=target_encode_cols if enable_engineering else None,
            task_type=normalized_task,
        )
        steps.append(("preprocessor", preprocessor))

        # 3. Feature Selection Step (if enabled)
        if enable_selection and selection_method != "none":
            selector = EnhancedFeatureSelector(
                method=selection_method,
                variance_threshold=variance_threshold,
                correlation_threshold=correlation_threshold,
                k_best_count=k_best_count,
                task_type=normalized_task,
                random_state=random_state,
            )
            steps.append(("selector", selector))

        # 4. Estimator
        estimator = _instantiate_model(model_type, random_state=random_state)
        steps.append(("estimator", estimator))

        return Pipeline(steps=steps), expected_features

    # Define Candidate Strategies
    strategies = {
        "baseline": (False, False),
        "feature_engineered": (True, False),
        "feature_selected": (False, True),
        "combined": (True, True),
    }

    strategy_results: List[Dict[str, Any]] = []

    # ---------------------------------------------------------
    # Evaluate Strategies on Training Partition with CV
    # ---------------------------------------------------------
    cv_splitter = (
        StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=random_state)
        if normalized_task == "classification"
        else KFold(n_splits=actual_splits, shuffle=True, random_state=random_state)
    )

    sklearn_scorer = (
        "f1_macro" if ranking_metric == "f1_macro"
        else "accuracy" if ranking_metric == "accuracy"
        else "r2" if ranking_metric == "r2"
        else "neg_root_mean_squared_error" if ranking_metric == "rmse"
        else "neg_mean_absolute_error" if ranking_metric == "mae"
        else "f1_macro" if normalized_task == "classification" else "neg_root_mean_squared_error"
    )

    for strat_name, (enable_fe, enable_fs) in strategies.items():
        pipe, _ = create_candidate_pipeline(enable_fe, enable_fs)
        try:
            cv_res = cross_validate(
                pipe,
                X_train,
                y_train,
                cv=cv_splitter,
                scoring=sklearn_scorer,
                return_train_score=False,
                error_score="raise",
            )
            raw_scores = cv_res["test_score"]
            scores = -raw_scores if sklearn_scorer.startswith("neg_") else raw_scores
            cv_mean = _safe_float(np.mean(scores))
            cv_std = _safe_float(np.std(scores))

            strategy_results.append({
                "strategy": strat_name,
                "cv_mean": cv_mean,
                "cv_std": cv_std,
                "status": "success",
                "error": None,
            })
        except Exception as exc:
            strategy_results.append({
                "strategy": strat_name,
                "cv_mean": None,
                "cv_std": None,
                "status": "failed",
                "error": str(exc),
            })
            warnings.append(f"Strategy '{strat_name}' failed CV: {exc}")

    # Rank strategies based solely on training CV
    successful_strats = [s for s in strategy_results if s["status"] == "success" and s["cv_mean"] is not None]
    if not successful_strats:
        raise HTTPException(status_code=500, detail="All feature strategies failed during cross-validation.")

    successful_strats.sort(key=lambda s: s["cv_mean"], reverse=higher_is_better)
    for rank, s in enumerate(successful_strats, start=1):
        s["rank"] = rank

    winning_strat_name = successful_strats[0]["strategy"]
    winning_cv_score = successful_strats[0]["cv_mean"]

    # ---------------------------------------------------------
    # Holdout Test Evaluation: Baseline vs Winner
    # ---------------------------------------------------------
    baseline_pipe, _ = create_candidate_pipeline(False, False)
    baseline_pipe.fit(X_train, y_train)
    y_test_pred_base = baseline_pipe.predict(X_test)
    baseline_holdout = _compute_metrics(
        y_true=y_test.to_numpy(),
        y_pred=y_test_pred_base,
        task_type=normalized_task,
        estimator=baseline_pipe.named_steps["estimator"],
        X_val=baseline_pipe.named_steps["preprocessor"].transform(X_test),
        unique_classes=unique_classes,
    )

    winning_pipe, _ = create_candidate_pipeline(
        strategies[winning_strat_name][0],
        strategies[winning_strat_name][1],
    )
    winning_pipe.fit(X_train, y_train)
    y_test_pred_win = winning_pipe.predict(X_test)
    winning_holdout = _compute_metrics(
        y_true=y_test.to_numpy(),
        y_pred=y_test_pred_win,
        task_type=normalized_task,
        estimator=winning_pipe.named_steps["estimator"],
        unique_classes=unique_classes,
    )

    # Compute delta improvements
    delta_metrics = {}
    keys = ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "roc_auc"] if normalized_task == "classification" else ["rmse", "mae", "r2"]
    for k in keys:
        b_val = baseline_holdout.get(k)
        w_val = winning_holdout.get(k)
        if b_val is not None and w_val is not None:
            abs_delta = round(w_val - b_val, 4)
            pct = round(((w_val - b_val) / abs(b_val)) * 100, 2) if b_val != 0 else None
            improved = (w_val > b_val) if k in {"accuracy", "f1_macro", "balanced_accuracy", "f1_weighted", "roc_auc", "r2"} else (w_val < b_val)
            delta_metrics[k] = {
                "baseline": b_val,
                "winning": w_val,
                "delta": abs_delta,
                "percentage_change": pct,
                "improved": improved,
            }

    # Inspect selected vs removed features and extract lineage
    selected_features = list(feature_cols)
    removed_features = []
    lineage_records = []

    if "feature_engineer" in winning_pipe.named_steps:
        fe_step = winning_pipe.named_steps["feature_engineer"]
        if hasattr(fe_step, "get_lineage"):
            lineage_records = fe_step.get_lineage()

    if not lineage_records:
        lineage_records = [
            {
                "feature_name": col,
                "parent_columns": [col],
                "operation": "original_input",
                "parameters": {},
                "is_engineered": False,
            }
            for col in feature_cols
        ]

    if "selector" in winning_pipe.named_steps:
        selector_step = winning_pipe.named_steps["selector"]
        if hasattr(selector_step, "get_support"):
            support = selector_step.get_support()
            prep_step = winning_pipe.named_steps["preprocessor"]
            if hasattr(prep_step, "get_feature_names_out"):
                try:
                    prep_names = list(prep_step.get_feature_names_out())
                    selected_features = [prep_names[i] for i, val in enumerate(support) if val]
                    removed_features = [prep_names[i] for i, val in enumerate(support) if not val]
                except Exception:
                    pass

    # ---------------------------------------------------------
    # Register Winning Pipeline in Model Registry
    # ---------------------------------------------------------
    full_winning_pipeline, _ = create_candidate_pipeline(
        strategies[winning_strat_name][0],
        strategies[winning_strat_name][1],
    )
    full_winning_pipeline.fit(X, y)

    reg_record = register_model(
        pipeline=full_winning_pipeline,
        model_name=f"FeatureEng_{winning_strat_name}_{run_id[:8]}",
        task_type=normalized_task,
        model_type=model_type,
        dataset_name=dataset_name,
        target_column=target_column,
        feature_columns=feature_cols,
        metrics=winning_holdout,
    )
    registered_model_id = reg_record.get("id")

    # ---------------------------------------------------------
    # Persist Feature Run Record
    # ---------------------------------------------------------
    total_duration = round(time.time() - start_time, 2)
    created_at = datetime.now(timezone.utc).isoformat()
    config_dict = {
        "datetime_cols": datetime_cols or [],
        "cyclical_datetime": cyclical_datetime,
        "log_cols": log_cols or [],
        "yeo_johnson_cols": yeo_johnson_cols or [],
        "sqrt_cols": sqrt_cols or [],
        "clip_outliers": clip_outliers,
        "group_rare_categories": group_rare_categories,
        "frequency_encode_cols": frequency_encode_cols or [],
        "interaction_pairs": interaction_pairs or [],
        "polynomial_cols": polynomial_cols or [],
        "target_encode_cols": target_encode_cols or [],
        "text_cols": text_cols or [],
        "selection_method": selection_method,
        "variance_threshold": variance_threshold,
        "correlation_threshold": correlation_threshold,
        "k_best_count": k_best_count,
        "scaler": scaler,
        "missing": missing,
    }
    dq_info = {
        "total_rows": len(df_clean),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "original_feature_count": len(feature_cols),
        "selected_feature_count": len(selected_features),
    }

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO feature_runs (
                run_id, task_type, target_column, dataset_name, model_type,
                ranking_metric, winning_strategy, strategy_results_json,
                baseline_metrics_json, winning_holdout_metrics_json,
                delta_metrics_json, selected_features_json, removed_features_json,
                transformation_config_json, registered_model_id, data_quality_json,
                duration_seconds, created_at, eda_analysis_id, feature_lineage_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                normalized_task,
                target_column,
                dataset_name,
                model_type,
                ranking_metric,
                winning_strat_name,
                json.dumps(_json_safe(strategy_results)),
                json.dumps(_json_safe(baseline_holdout)),
                json.dumps(_json_safe(winning_holdout)),
                json.dumps(_json_safe(delta_metrics)),
                json.dumps(_json_safe(selected_features)),
                json.dumps(_json_safe(removed_features)),
                json.dumps(_json_safe(config_dict)),
                registered_model_id,
                json.dumps(_json_safe(dq_info)),
                float(total_duration),
                created_at,
                eda_analysis_id,
                json.dumps(_json_safe(lineage_records)),
            ),
        )

    return {
        "run_id": run_id,
        "task_type": normalized_task,
        "target_column": target_column,
        "dataset_name": dataset_name,
        "model_type": model_type,
        "ranking_metric": ranking_metric,
        "winning_strategy": winning_strat_name,
        "strategy_results": strategy_results,
        "baseline_holdout_metrics": baseline_holdout,
        "winning_holdout_metrics": winning_holdout,
        "delta_metrics": delta_metrics,
        "selected_features": selected_features,
        "removed_features": removed_features,
        "feature_lineage": lineage_records,
        "transformation_config": config_dict,
        "registered_model_id": registered_model_id,
        "data_quality": dq_info,
        "duration_seconds": total_duration,
        "warnings": warnings,
        "created_at": created_at,
        "eda_analysis_id": eda_analysis_id,
    }


# =========================================================
# Multi-Strategy Comparison Engine (Conservative vs Balanced vs Advanced)
# =========================================================

async def compare_feature_strategies(
    file: UploadFile,
    task_type: str,
    target_column: str,
    model_type: str,
    ranking_metric: Optional[str] = None,
    n_splits: int = 3,
    split_ratio: float = 0.8,
    eda_analysis_id: Optional[str] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Evaluates Original Baseline vs Conservative vs Balanced vs Advanced presets
    under strictly identical CV folds. Evaluates winning preset on untouched holdout.
    """
    start_time = time.time()
    init_feature_db()
    run_id = uuid.uuid4().hex

    df = await safe_read_csv(file)
    dataset_name = file.filename or "dataset.csv"

    if target_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found.")

    df_clean = df.dropna(subset=[target_column]).copy()
    feature_cols = [c for c in df_clean.columns if c != target_column]

    normalized_task = (task_type or "").strip().lower()
    ranking_metric = ranking_metric or ("f1_macro" if normalized_task == "classification" else "rmse")
    higher_is_better = ranking_metric in {"accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "roc_auc", "r2"}

    # Get dataset-adaptive preset recommendations
    recs = get_eda_feature_recommendations(df_clean, target_column=target_column, eda_analysis_id=eda_analysis_id)
    presets = recs["presets"]

    X = df_clean[feature_cols]
    y = df_clean[target_column]

    stratify_target = y if (normalized_task == "classification" and y.value_counts().min() >= 2) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=split_ratio, random_state=random_state, stratify=stratify_target
    )

    actual_splits = max(2, min(int(n_splits), 5))
    cv_splitter = (
        StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=random_state)
        if normalized_task == "classification"
        else KFold(n_splits=actual_splits, shuffle=True, random_state=random_state)
    )

    sklearn_scorer = (
        "f1_macro" if ranking_metric == "f1_macro"
        else "accuracy" if ranking_metric == "accuracy"
        else "r2" if ranking_metric == "r2"
        else "neg_root_mean_squared_error" if ranking_metric == "rmse"
        else "neg_mean_absolute_error" if ranking_metric == "mae"
        else "f1_macro" if normalized_task == "classification" else "neg_root_mean_squared_error"
    )

    def _build_preset_pipeline(preset_cfg: Dict[str, Any], is_baseline: bool = False) -> Pipeline:
        steps = []
        if not is_baseline:
            fe = TabularFeatureEngineer(
                datetime_cols=preset_cfg.get("datetime_cols", []),
                cyclical_datetime=preset_cfg.get("cyclical_datetime", False),
                log_cols=preset_cfg.get("log_cols", []),
                yeo_johnson_cols=preset_cfg.get("yeo_johnson_cols", []),
                clip_outliers=preset_cfg.get("clip_outliers", False),
                outlier_cols=[c for c in feature_cols if pd.api.types.is_numeric_dtype(X_train[c])],
                group_rare_categories=preset_cfg.get("group_rare_categories", False),
                rare_cat_cols=[c for c in feature_cols if not pd.api.types.is_numeric_dtype(X_train[c])],
                interaction_pairs=preset_cfg.get("interaction_pairs", []),
                polynomial_cols=preset_cfg.get("polynomial_cols", []),
                text_cols=preset_cfg.get("text_cols", []),
            )
            steps.append(("feature_engineer", fe))
            sample_t = fe.fit_transform(X_train.head(10))
            c_num = [c for c in sample_t.columns if pd.api.types.is_numeric_dtype(sample_t[c])]
            c_cat = [c for c in sample_t.columns if c not in c_num]
        else:
            c_num = [c for c in feature_cols if pd.api.types.is_numeric_dtype(X_train[c])]
            c_cat = [c for c in feature_cols if c not in c_num]

        prep = build_feature_preprocessor(
            numeric_columns=c_num,
            categorical_columns=c_cat,
            scaler=preset_cfg.get("scaler", "standard") if not is_baseline else "standard",
            missing=preset_cfg.get("imputation", "median") if not is_baseline else "median",
            target_encode_cols=preset_cfg.get("target_encode_cols") if not is_baseline else None,
            task_type=normalized_task,
        )
        steps.append(("preprocessor", prep))

        if not is_baseline and preset_cfg.get("selection_method", "none") != "none":
            sel = EnhancedFeatureSelector(
                method=preset_cfg.get("selection_method", "none"),
                variance_threshold=preset_cfg.get("variance_threshold", 0.01),
                correlation_threshold=preset_cfg.get("correlation_threshold", 0.85),
                k_best_count=preset_cfg.get("k_best_count", 10),
                task_type=normalized_task,
                random_state=random_state,
            )
            steps.append(("selector", sel))

        estimator = _instantiate_model(model_type, random_state=random_state)
        steps.append(("estimator", estimator))
        return Pipeline(steps=steps)

    comparison_results = []
    pipeline_candidates = {
        "baseline": ({"name": "Original Features (Baseline)"}, True),
        "conservative": (presets["conservative"], False),
        "balanced": (presets["balanced"], False),
        "advanced": (presets["advanced"], False),
    }

    fitted_pipelines = {}

    for strat_key, (cfg, is_base) in pipeline_candidates.items():
        t0 = time.time()
        pipe = _build_preset_pipeline(cfg, is_baseline=is_base)
        try:
            cv_res = cross_validate(
                pipe,
                X_train,
                y_train,
                cv=cv_splitter,
                scoring=sklearn_scorer,
                return_train_score=False,
                error_score="raise",
            )
            raw = cv_res["test_score"]
            scores = -raw if sklearn_scorer.startswith("neg_") else raw
            mean_score = _safe_float(np.mean(scores))
            std_score = _safe_float(np.std(scores))

            # Fit on full X_train to measure feature counts
            pipe.fit(X_train, y_train)
            fitted_pipelines[strat_key] = pipe

            feat_count = len(feature_cols)
            if "selector" in pipe.named_steps:
                sel = pipe.named_steps["selector"]
                if hasattr(sel, "get_support"):
                    feat_count = int(np.sum(sel.get_support()))
            elif "feature_engineer" in pipe.named_steps:
                feat_count = len(pipe.named_steps["feature_engineer"].output_feature_names_)

            comparison_results.append({
                "strategy": strat_key,
                "strategy_name": cfg["name"],
                "cv_mean": mean_score,
                "cv_std": std_score,
                "active_features_count": feat_count,
                "duration_seconds": round(time.time() - t0, 3),
                "status": "success",
            })
        except Exception as exc:
            comparison_results.append({
                "strategy": strat_key,
                "strategy_name": cfg["name"],
                "cv_mean": None,
                "cv_std": None,
                "active_features_count": len(feature_cols),
                "duration_seconds": round(time.time() - t0, 3),
                "status": "failed",
                "error": str(exc),
            })

    # Rank strategies
    valid_res = [c for c in comparison_results if c["status"] == "success" and c["cv_mean"] is not None]
    if not valid_res:
        raise HTTPException(status_code=500, detail="All feature strategies failed in cross-validation.")

    valid_res.sort(key=lambda s: s["cv_mean"], reverse=higher_is_better)
    for rank, s in enumerate(valid_res, start=1):
        s["rank"] = rank

    winner_key = valid_res[0]["strategy"]
    winning_pipe = fitted_pipelines[winner_key]
    baseline_pipe = fitted_pipelines["baseline"]

    # Evaluate on pristine holdout
    y_base_pred = baseline_pipe.predict(X_test)
    y_win_pred = winning_pipe.predict(X_test)

    baseline_metrics = _compute_metrics(
        y_true=y_test.to_numpy(),
        y_pred=y_base_pred,
        task_type=normalized_task,
        estimator=baseline_pipe.named_steps["estimator"],
    )
    winning_metrics = _compute_metrics(
        y_true=y_test.to_numpy(),
        y_pred=y_win_pred,
        task_type=normalized_task,
        estimator=winning_pipe.named_steps["estimator"],
    )

    # Compute deltas
    delta_metrics = {}
    keys = ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted"] if normalized_task == "classification" else ["rmse", "mae", "r2"]
    for k in keys:
        b = baseline_metrics.get(k)
        w = winning_metrics.get(k)
        if b is not None and w is not None:
            delta = round(w - b, 4)
            pct = round(((w - b) / abs(b)) * 100, 2) if b != 0 else None
            improved = (w > b) if k in {"accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "r2"} else (w < b)
            delta_metrics[k] = {"baseline": b, "winning": w, "delta": delta, "percentage_change": pct, "improved": improved}

    # Statistical verdict
    primary_key = "f1_macro" if normalized_task == "classification" else "rmse"
    pct_gain = delta_metrics.get(primary_key, {}).get("percentage_change", 0.0) or 0.0

    if pct_gain >= 2.0:
        verdict = f"Substantial Generalization Gain: '{winner_key.capitalize()}' strategy improved validation {primary_key} by +{pct_gain}%."
    elif pct_gain >= 0.2:
        verdict = f"Moderate Generalization Gain: '{winner_key.capitalize()}' strategy provided reliable +{pct_gain}% improvement."
    elif pct_gain >= -0.5:
        verdict = f"Parsimonious Performance: '{winner_key.capitalize()}' preserved model accuracy within margin while optimizing feature representation."
    else:
        verdict = f"Caution / Overfitting Warning: Engineered features showed higher CV score but deteriorated holdout performance by {pct_gain}%. Recommending Conservative strategy."

    # Fit winning pipeline on full dataset and register in Model Registry
    full_winning_pipe = _build_preset_pipeline(pipeline_candidates[winner_key][0], is_baseline=(winner_key == "baseline"))
    full_winning_pipe.fit(X, y)

    reg_record = register_model(
        pipeline=full_winning_pipe,
        model_name=f"FeaturePreset_{winner_key.capitalize()}_{run_id[:8]}",
        task_type=normalized_task,
        model_type=model_type,
        dataset_name=dataset_name,
        target_column=target_column,
        feature_columns=feature_cols,
        metrics=winning_metrics,
    )

    lineage = []
    if "feature_engineer" in winning_pipe.named_steps:
        fe_t = winning_pipe.named_steps["feature_engineer"]
        if hasattr(fe_t, "get_lineage"):
            lineage = fe_t.get_lineage()

    if not lineage:
        lineage = [
            {
                "feature_name": col,
                "parent_columns": [col],
                "operation": "original_input",
                "parameters": {},
                "is_engineered": False,
            }
            for col in feature_cols
        ]

    total_duration = round(time.time() - start_time, 2)
    created_at = datetime.now(timezone.utc).isoformat()

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO feature_runs (
                run_id, task_type, target_column, dataset_name, model_type,
                ranking_metric, winning_strategy, strategy_results_json,
                baseline_metrics_json, winning_holdout_metrics_json,
                delta_metrics_json, selected_features_json, removed_features_json,
                transformation_config_json, registered_model_id, data_quality_json,
                duration_seconds, created_at, eda_analysis_id, feature_lineage_json,
                strategy_preset, comparison_matrix_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                normalized_task,
                target_column,
                dataset_name,
                model_type,
                ranking_metric,
                winner_key,
                json.dumps(_json_safe(comparison_results)),
                json.dumps(_json_safe(baseline_metrics)),
                json.dumps(_json_safe(winning_metrics)),
                json.dumps(_json_safe(delta_metrics)),
                json.dumps(_json_safe(feature_cols)),
                json.dumps(_json_safe([])),
                json.dumps(_json_safe(pipeline_candidates[winner_key][0])),
                reg_record.get("id"),
                json.dumps({"total_rows": len(df_clean), "train_samples": len(X_train), "test_samples": len(X_test)}),
                float(total_duration),
                created_at,
                eda_analysis_id,
                json.dumps(_json_safe(lineage)),
                winner_key,
                json.dumps(_json_safe(comparison_results)),
            ),
        )

    return _json_safe({
        "run_id": run_id,
        "task_type": normalized_task,
        "target_column": target_column,
        "dataset_name": dataset_name,
        "model_type": model_type,
        "ranking_metric": ranking_metric,
        "winning_strategy": winner_key,
        "comparison_matrix": comparison_results,
        "baseline_holdout_metrics": baseline_metrics,
        "winning_holdout_metrics": winning_metrics,
        "delta_metrics": delta_metrics,
        "statistical_verdict": verdict,
        "registered_model_id": reg_record.get("id"),
        "feature_lineage": lineage,
        "eda_analysis_id": eda_analysis_id,
        "active_profiles": recs.get("active_profiles", []),
        "duration_seconds": total_duration,
        "created_at": created_at,
    })


# =========================================================
# Query & Management
# =========================================================

def list_feature_runs(limit: int = 50, offset: int = 0) -> Dict[str, Any]:
    init_feature_db()
    with _connect() as conn:
        total = conn.execute("SELECT COUNT(*) FROM feature_runs").fetchone()[0]
        cursor = conn.execute(
            """
            SELECT id, run_id, task_type, target_column, dataset_name, model_type,
                   ranking_metric, winning_strategy, registered_model_id, duration_seconds,
                   created_at, eda_analysis_id, strategy_preset
            FROM feature_runs
            ORDER BY id DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        )
        items = [dict(row) for row in cursor.fetchall()]

    return {"items": items, "total": total, "limit": limit, "offset": offset}


def get_feature_run(run_id: str) -> Dict[str, Any]:
    init_feature_db()
    with _connect() as conn:
        cursor = conn.execute("SELECT * FROM feature_runs WHERE run_id = ?", (run_id,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"Feature run '{run_id}' not found.")

        item = dict(row)
        item["strategy_results"] = json.loads(item.pop("strategy_results_json") or "[]")
        item["baseline_metrics"] = json.loads(item.pop("baseline_metrics_json") or "{}")
        item["winning_holdout_metrics"] = json.loads(item.pop("winning_holdout_metrics_json") or "{}")
        item["delta_metrics"] = json.loads(item.pop("delta_metrics_json") or "{}")
        item["selected_features"] = json.loads(item.pop("selected_features_json") or "[]")
        item["removed_features"] = json.loads(item.pop("removed_features_json") or "[]")
        item["transformation_config"] = json.loads(item.pop("transformation_config_json") or "{}")
        item["data_quality"] = json.loads(item.pop("data_quality_json") or "{}")
        item["feature_lineage"] = json.loads(item.pop("feature_lineage_json") or "[]")
        item["comparison_matrix"] = json.loads(item.pop("comparison_matrix_json") or "[]")
        return item


def delete_feature_run(run_id: str) -> Dict[str, Any]:
    init_feature_db()
    with _connect() as conn:
        cursor = conn.execute("SELECT id FROM feature_runs WHERE run_id = ?", (run_id,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail=f"Feature run '{run_id}' not found.")
        conn.execute("DELETE FROM feature_runs WHERE run_id = ?", (run_id,))
    return {"message": f"Feature run '{run_id}' deleted successfully."}
