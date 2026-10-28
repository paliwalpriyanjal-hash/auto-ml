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
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_selection import (
    SelectFromModel,
    SelectKBest,
    VarianceThreshold,
    f_classif,
    f_regression,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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
    RobustScaler,
    StandardScaler,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from app.services.model_registry_service import register_model

SERVER_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = SERVER_DIR / "data"
FEATURE_DB_PATH = DATA_DIR / "feature_history.db"


# =========================================================
# Database Setup
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
) -> Dict[str, Any]:
    """
    Profile raw dataset with statistics, data quality alerts,
    collinearity diagnostics, and type classification without using target to drop columns.
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
            # Match common ISO, date, or timestamp patterns
            date_pattern = re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4}")
            if any(date_pattern.search(v) for v in sample_vals):
                try:
                    pd.to_datetime(sample_vals, errors="raise")
                    is_datetime = True
                    inferred_type = "datetime"
                    datetime_candidates.append(col)
                except Exception:
                    pass

        if is_numeric and not is_datetime:
            inferred_type = "numeric"
            numeric_cols.append(col)
        elif not is_datetime:
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
            col_stats["min"] = _safe_float(series.min())
            col_stats["max"] = _safe_float(series.max())
            col_stats["mean"] = _safe_float(series.mean())
            col_stats["std"] = _safe_float(series.std())
            col_stats["is_non_negative"] = bool((series.dropna() >= 0).all())

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
            "is_sampled": is_sampled,
            "sampled_rows": max_sample_rows if is_sampled else initial_rows,
        },
        "target_column": target_column,
        "columns": columns_profile,
        "quality_alerts": quality_alerts,
        "redundant_pairs": collinear_pairs,
        "correlation_matrix": correlation_matrix,
        "datetime_candidates": datetime_candidates,
        "constant_columns": constant_cols,
        "near_constant_columns": near_constant_cols,
        "id_like_columns": id_like_cols,
    })


# =========================================================
# Custom Transformers for Leakage-Free Scikit-Learn Pipelines
# =========================================================

class TabularFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Leakage-free feature engineering transformer handling:
    - Calendar component extraction for datetime columns
    - Safe log1p transformation on eligible non-negative columns
    - Robust outlier clipping with percentiles learned strictly during fit
    - Rare category grouping learned strictly during fit
    """
    def __init__(
        self,
        datetime_cols: Optional[List[str]] = None,
        drop_original_datetime: bool = True,
        log_cols: Optional[List[str]] = None,
        clip_outliers: bool = False,
        outlier_cols: Optional[List[str]] = None,
        outlier_lower_pct: float = 1.0,
        outlier_upper_pct: float = 99.0,
        group_rare_categories: bool = False,
        rare_cat_cols: Optional[List[str]] = None,
        rare_cat_min_freq: float = 0.02,
    ):
        self.datetime_cols = datetime_cols or []
        self.drop_original_datetime = drop_original_datetime
        self.log_cols = log_cols or []
        self.clip_outliers = clip_outliers
        self.outlier_cols = outlier_cols or []
        self.outlier_lower_pct = outlier_lower_pct
        self.outlier_upper_pct = outlier_upper_pct
        self.group_rare_categories = group_rare_categories
        self.rare_cat_cols = rare_cat_cols or []
        self.rare_cat_min_freq = rare_cat_min_freq

        # Learned parameters
        self.clipping_bounds_: Dict[str, Tuple[float, float]] = {}
        self.frequent_categories_: Dict[str, List[Any]] = {}
        self.output_feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # 1. Learn outlier bounds on training slice
        if self.clip_outliers and self.outlier_cols:
            self.clipping_bounds_ = {}
            for col in self.outlier_cols:
                if col in X.columns and pd.api.types.is_numeric_dtype(X[col]):
                    valid_series = X[col].dropna()
                    if not valid_series.empty:
                        low = float(np.percentile(valid_series, self.outlier_lower_pct))
                        high = float(np.percentile(valid_series, self.outlier_upper_pct))
                        self.clipping_bounds_[col] = (low, high)

        # 2. Learn frequent categories on training slice
        if self.group_rare_categories and self.rare_cat_cols:
            self.frequent_categories_ = {}
            n_rows = len(X)
            min_count = max(2, int(n_rows * self.rare_cat_min_freq))
            for col in self.rare_cat_cols:
                if col in X.columns:
                    counts = X[col].value_counts(dropna=True)
                    frequent = counts[counts >= min_count].index.tolist()
                    self.frequent_categories_[col] = frequent

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X_out = X.copy()

        # 1. Datetime extraction
        for dt_col in self.datetime_cols:
            if dt_col in X_out.columns:
                try:
                    dt_series = pd.to_datetime(X_out[dt_col], errors="coerce")
                    X_out[f"{dt_col}_year"] = dt_series.dt.year.fillna(2000).astype(int)
                    X_out[f"{dt_col}_month"] = dt_series.dt.month.fillna(1).astype(int)
                    X_out[f"{dt_col}_day"] = dt_series.dt.day.fillna(1).astype(int)
                    X_out[f"{dt_col}_dayofweek"] = dt_series.dt.dayofweek.fillna(0).astype(int)
                    if self.drop_original_datetime:
                        X_out.drop(columns=[dt_col], inplace=True)
                except Exception:
                    pass

        # 2. Safe Log1p Transformation on non-negative values
        for l_col in self.log_cols:
            if l_col in X_out.columns and pd.api.types.is_numeric_dtype(X_out[l_col]):
                # Clip negative values to zero before log1p to ensure mathematical safety
                X_out[l_col] = np.log1p(np.maximum(0, X_out[l_col].fillna(0).astype(float)))

        # 3. Outlier Clipping
        if self.clip_outliers:
            for col, (low, high) in self.clipping_bounds_.items():
                if col in X_out.columns and pd.api.types.is_numeric_dtype(X_out[col]):
                    X_out[col] = X_out[col].clip(lower=low, upper=high)

        # 4. Rare category grouping
        if self.group_rare_categories:
            for col, frequent in self.frequent_categories_.items():
                if col in X_out.columns:
                    X_out[col] = X_out[col].apply(
                        lambda val: val if (pd.isna(val) or val in frequent) else "__other__"
                    )

        self.output_feature_names_ = list(X_out.columns)
        return X_out

    def get_feature_names_out(self, input_features=None):
        return np.array(self.output_feature_names_)


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """
    Filters redundant numeric features exceeding a Pearson correlation threshold.
    Correlation matrix is fit strictly on training slice.
    """
    def __init__(self, threshold: float = 0.9):
        self.threshold = threshold
        self.dropped_columns_: List[str] = []
        self.retained_columns_: List[str] = []

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        numeric_df = X.select_dtypes(include=[np.number])
        self.dropped_columns_ = []

        if numeric_df.shape[1] >= 2 and numeric_df.shape[0] >= 5:
            corr_matrix = numeric_df.corr().abs()
            cols = corr_matrix.columns.tolist()
            to_drop = set()

            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    col_i, col_j = cols[i], cols[j]
                    if col_i in to_drop or col_j in to_drop:
                        continue
                    if corr_matrix.iloc[i, j] >= self.threshold:
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


# =========================================================
# Pipeline Builders & Evaluation Services
# =========================================================

def build_feature_preprocessor(
    numeric_columns: List[str],
    categorical_columns: List[str],
    scaler: str = "standard",
    missing: str = "median",
) -> ColumnTransformer:
    transformers = []

    if numeric_columns:
        num_imputer = "mean" if missing == "mean" else "median"
        num_steps = [("imputer", SimpleImputer(strategy=num_imputer))]
        if scaler == "standard":
            num_steps.append(("scaler", StandardScaler()))
        elif scaler == "minmax":
            num_steps.append(("scaler", MinMaxScaler()))
        elif scaler == "robust":
            num_steps.append(("scaler", RobustScaler()))
        transformers.append(("num", Pipeline(steps=num_steps), numeric_columns))

    if categorical_columns:
        cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("cat", cat_pipeline, categorical_columns))

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
# Feature Strategy Evaluation Engine
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
    log_cols: Optional[List[str]] = None,
    clip_outliers: bool = False,
    group_rare_categories: bool = False,
    # Feature Selection Config
    selection_method: str = "none",  # none, variance, correlation, model_based, k_best
    variance_threshold: float = 0.01,
    correlation_threshold: float = 0.85,
    k_best_count: int = 10,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Evaluates baseline vs feature-engineered vs feature-selected pipelines using
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
    valid_selection_methods = {"none", "variance", "correlation", "model_based", "k_best"}
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
                log_cols=log_cols,
                clip_outliers=clip_outliers,
                outlier_cols=num_cols,
                group_rare_categories=group_rare_categories,
                rare_cat_cols=cat_cols,
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
        )
        steps.append(("preprocessor", preprocessor))

        # 3. Feature Selection Step (if enabled)
        if enable_selection and selection_method != "none":
            if selection_method == "variance":
                selector = VarianceThreshold(threshold=variance_threshold)
                steps.append(("selector", selector))
            elif selection_method == "correlation":
                selector = CorrelationFilter(threshold=correlation_threshold)
                steps.append(("selector", selector))
            elif selection_method == "model_based":
                base_est = (
                    RandomForestClassifier(n_estimators=50, max_depth=8, n_jobs=1, random_state=random_state)
                    if normalized_task == "classification"
                    else RandomForestRegressor(n_estimators=50, max_depth=8, n_jobs=1, random_state=random_state)
                )
                selector = SelectFromModel(estimator=base_est, max_features=k_best_count or 10)
                steps.append(("selector", selector))
            elif selection_method == "k_best":
                score_fn = f_classif if normalized_task == "classification" else f_regression
                k_val = max(1, min(k_best_count, len(feature_cols)))
                selector = SelectKBest(score_func=score_fn, k=k_val)
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

    # Inspect selected vs removed features
    selected_features = list(feature_cols)
    removed_features = []
    if "selector" in winning_pipe.named_steps:
        selector_step = winning_pipe.named_steps["selector"]
        if hasattr(selector_step, "get_support"):
            support = selector_step.get_support()
            # Feature names before selector
            prep_step = winning_pipe.named_steps["preprocessor"]
            if hasattr(prep_step, "get_feature_names_out"):
                try:
                    prep_names = prep_step.get_feature_names_out()
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
        "log_cols": log_cols or [],
        "clip_outliers": clip_outliers,
        "group_rare_categories": group_rare_categories,
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
                duration_seconds, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        "transformation_config": config_dict,
        "registered_model_id": registered_model_id,
        "data_quality": dq_info,
        "duration_seconds": total_duration,
        "warnings": warnings,
        "created_at": created_at,
    }


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
                   ranking_metric, winning_strategy, registered_model_id, duration_seconds, created_at
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
        return item


def delete_feature_run(run_id: str) -> Dict[str, Any]:
    init_feature_db()
    with _connect() as conn:
        cursor = conn.execute("SELECT id FROM feature_runs WHERE run_id = ?", (run_id,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail=f"Feature run '{run_id}' not found.")
        conn.execute("DELETE FROM feature_runs WHERE run_id = ?", (run_id,))
    return {"message": f"Feature run '{run_id}' deleted successfully."}
