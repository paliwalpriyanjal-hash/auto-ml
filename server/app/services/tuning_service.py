from contextlib import contextmanager
# app/services/tuning_service.py

import io
import json
import math
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import optuna
import pandas as pd
from fastapi import HTTPException, UploadFile
from app.utils.security import safe_read_csv
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
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
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from app.services.model_registry_service import register_model

# Silence optuna logs to keep terminal output clean and predictable
optuna.logging.set_verbosity(optuna.logging.WARNING)

SERVER_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = SERVER_DIR / "data"
TUNING_DB_PATH = DATA_DIR / "tuning_history.db"

# In-memory tracking of active run cancellations
_CANCELLED_RUNS: Set[str] = set()


# =========================================================
# Database operations
# =========================================================

@contextmanager
def _connect():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(TUNING_DB_PATH), timeout=10)
    connection.row_factory = sqlite3.Row
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def init_tuning_db() -> None:
    with _connect() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS tuning_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT UNIQUE NOT NULL,
                task_type TEXT NOT NULL,
                model_type TEXT NOT NULL,
                target_column TEXT NOT NULL,
                dataset_name TEXT,
                metric TEXT NOT NULL,
                direction TEXT NOT NULL,
                n_trials INTEGER NOT NULL,
                n_splits INTEGER NOT NULL,
                status TEXT NOT NULL,
                best_trial_number INTEGER,
                best_score REAL,
                best_params_json TEXT,
                baseline_metrics_json TEXT,
                tuned_metrics_json TEXT,
                delta_metrics_json TEXT,
                registered_model_id INTEGER,
                feature_columns_json TEXT,
                data_quality_json TEXT,
                warnings_json TEXT,
                duration_seconds REAL,
                error_message TEXT,
                created_at TEXT NOT NULL
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS tuning_trials (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                trial_number INTEGER NOT NULL,
                params_json TEXT NOT NULL,
                cv_score REAL,
                duration_seconds REAL,
                state TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_tuning_trials_run_id
            ON tuning_trials(run_id)
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
# Search space and Model definitions
# =========================================================

SUPPORTED_CLASSIFICATION_MODELS = {
    "LogisticRegression",
    "DecisionTreeClassifier",
    "RandomForestClassifier",
    "ExtraTreesClassifier",
}

SUPPORTED_REGRESSION_MODELS = {
    "DecisionTree",
    "RandomForest",
}

NON_TUNABLE_MODELS = {
    "LinearRegression": (
        "LinearRegression (Ordinary Least Squares) does not possess tunable hyperparameters. "
        "Please select DecisionTree or RandomForest for regression tuning."
    )
}

METRIC_DIRECTIONS = {
    "f1_macro": "maximize",
    "accuracy": "maximize",
    "balanced_accuracy": "maximize",
    "f1_weighted": "maximize",
    "roc_auc": "maximize",
    "r2": "maximize",
    "rmse": "minimize",
    "mae": "minimize",
    "log_loss": "minimize",
}


def create_baseline_model(model_type: str, random_state: int = 42):
    if model_type == "LogisticRegression":
        return LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=random_state,
        )
    elif model_type == "DecisionTreeClassifier":
        return DecisionTreeClassifier(
            class_weight="balanced",
            random_state=random_state,
        )
    elif model_type == "RandomForestClassifier":
        return RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            n_jobs=1,
            random_state=random_state,
        )
    elif model_type == "ExtraTreesClassifier":
        return ExtraTreesClassifier(
            n_estimators=100,
            class_weight="balanced",
            n_jobs=1,
            random_state=random_state,
        )
    elif model_type == "DecisionTree":
        return DecisionTreeRegressor(
            random_state=random_state,
        )
    elif model_type == "RandomForest":
        return RandomForestRegressor(
            n_estimators=100,
            n_jobs=1,
            random_state=random_state,
        )
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model type for baseline: {model_type}",
        )


def suggest_model_params(
    trial: optuna.Trial,
    model_type: str,
    random_state: int = 42,
) -> Tuple[Any, Dict[str, Any]]:
    """Sample hyperparameters using Optuna trial and return (estimator, params_dict)."""
    if model_type == "LogisticRegression":
        c_val = trial.suggest_float("C", 1e-4, 100.0, log=True)
        max_iter = trial.suggest_categorical("max_iter", [500, 1000, 2000])
        params = {"C": c_val, "max_iter": max_iter}
        model = LogisticRegression(
            C=c_val,
            max_iter=max_iter,
            solver="lbfgs",
            class_weight="balanced",
            random_state=random_state,
        )
        return model, params

    elif model_type == "DecisionTreeClassifier":
        criterion = trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"])
        max_depth = trial.suggest_int("max_depth", 2, 20)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
        params = {
            "criterion": criterion,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
        }
        model = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced",
            random_state=random_state,
        )
        return model, params

    elif model_type == "RandomForestClassifier":
        n_estimators = trial.suggest_int("n_estimators", 10, 100, step=10)
        criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
        max_depth = trial.suggest_int("max_depth", 3, 15)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)
        params = {
            "n_estimators": n_estimators,
            "criterion": criterion,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
        }
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced",
            n_jobs=1,
            random_state=random_state,
        )
        return model, params

    elif model_type == "ExtraTreesClassifier":
        n_estimators = trial.suggest_int("n_estimators", 10, 100, step=10)
        criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
        max_depth = trial.suggest_int("max_depth", 3, 15)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)
        params = {
            "n_estimators": n_estimators,
            "criterion": criterion,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
        }
        model = ExtraTreesClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced",
            n_jobs=1,
            random_state=random_state,
        )
        return model, params

    elif model_type == "DecisionTree":
        criterion = trial.suggest_categorical(
            "criterion", ["squared_error", "friedman_mse", "absolute_error"]
        )
        max_depth = trial.suggest_int("max_depth", 2, 20)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
        params = {
            "criterion": criterion,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
        }
        model = DecisionTreeRegressor(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
        return model, params

    elif model_type == "RandomForest":
        n_estimators = trial.suggest_int("n_estimators", 10, 100, step=10)
        criterion = trial.suggest_categorical("criterion", ["squared_error", "absolute_error"])
        max_depth = trial.suggest_int("max_depth", 3, 15)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)
        params = {
            "n_estimators": n_estimators,
            "criterion": criterion,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
        }
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            n_jobs=1,
            random_state=random_state,
        )
        return model, params

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model type: {model_type}",
        )


def build_model_from_params(
    model_type: str,
    params: Dict[str, Any],
    random_state: int = 42,
):
    """Instantiate a model with explicit parameter dictionary."""
    if model_type == "LogisticRegression":
        return LogisticRegression(
            C=params.get("C", 1.0),
            max_iter=params.get("max_iter", 1000),
            solver="lbfgs",
            class_weight="balanced",
            random_state=random_state,
        )
    elif model_type == "DecisionTreeClassifier":
        return DecisionTreeClassifier(
            criterion=params.get("criterion", "gini"),
            max_depth=params.get("max_depth"),
            min_samples_split=params.get("min_samples_split", 2),
            min_samples_leaf=params.get("min_samples_leaf", 1),
            class_weight="balanced",
            random_state=random_state,
        )
    elif model_type == "RandomForestClassifier":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100),
            criterion=params.get("criterion", "gini"),
            max_depth=params.get("max_depth"),
            min_samples_split=params.get("min_samples_split", 2),
            min_samples_leaf=params.get("min_samples_leaf", 1),
            class_weight="balanced",
            n_jobs=1,
            random_state=random_state,
        )
    elif model_type == "ExtraTreesClassifier":
        return ExtraTreesClassifier(
            n_estimators=params.get("n_estimators", 100),
            criterion=params.get("criterion", "gini"),
            max_depth=params.get("max_depth"),
            min_samples_split=params.get("min_samples_split", 2),
            min_samples_leaf=params.get("min_samples_leaf", 1),
            class_weight="balanced",
            n_jobs=1,
            random_state=random_state,
        )
    elif model_type == "DecisionTree":
        return DecisionTreeRegressor(
            criterion=params.get("criterion", "squared_error"),
            max_depth=params.get("max_depth"),
            min_samples_split=params.get("min_samples_split", 2),
            min_samples_leaf=params.get("min_samples_leaf", 1),
            random_state=random_state,
        )
    elif model_type == "RandomForest":
        return RandomForestRegressor(
            n_estimators=params.get("n_estimators", 100),
            criterion=params.get("criterion", "squared_error"),
            max_depth=params.get("max_depth"),
            min_samples_split=params.get("min_samples_split", 2),
            min_samples_leaf=params.get("min_samples_leaf", 1),
            n_jobs=1,
            random_state=random_state,
        )
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model type: {model_type}",
        )


# =========================================================
# Preprocessor and Pipeline Builder
# =========================================================

def build_preprocessor(
    numeric_columns: List[str],
    categorical_columns: List[str],
    scaler: str = "standard",
    missing: str = "median",
) -> ColumnTransformer:
    transformers = []

    if numeric_columns:
        numeric_imputer = "mean" if missing == "mean" else "median"
        numeric_steps = [("imputer", SimpleImputer(strategy=numeric_imputer))]

        if scaler == "standard":
            numeric_steps.append(("scaler", StandardScaler()))
        elif scaler == "minmax":
            numeric_steps.append(("scaler", MinMaxScaler()))
        elif scaler == "robust":
            numeric_steps.append(("scaler", RobustScaler()))

        transformers.append(
            ("num", Pipeline(steps=numeric_steps), numeric_columns)
        )

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(
            ("cat", categorical_pipeline, categorical_columns)
        )

    if not transformers:
        raise HTTPException(
            status_code=400,
            detail="No valid feature columns found to build preprocessor.",
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def _detect_feature_columns(
    df: pd.DataFrame,
    target_column: str,
    max_categorical_cardinality: int = 50,
) -> Tuple[List[str], List[str]]:
    feature_columns = [col for col in df.columns if col != target_column]
    if not feature_columns:
        raise HTTPException(
            status_code=400,
            detail="No feature columns found in dataset.",
        )

    numeric_columns: List[str] = []
    categorical_columns: List[str] = []

    for column in feature_columns:
        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            if series.dropna().empty:
                raise HTTPException(
                    status_code=400,
                    detail=f"Feature '{column}' contains only missing values.",
                )
            numeric_columns.append(column)
        else:
            non_missing = series.dropna().astype(str)
            if non_missing.empty:
                raise HTTPException(
                    status_code=400,
                    detail=f"Feature '{column}' contains only missing values.",
                )
            cardinality = non_missing.nunique()
            if cardinality > max_categorical_cardinality:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Feature '{column}' has {cardinality} unique values; "
                        f"the supported maximum is {max_categorical_cardinality}."
                    ),
                )
            categorical_columns.append(column)

    return numeric_columns, categorical_columns


# =========================================================
# Holdout and Cross-Validation Metric Evaluation
# =========================================================

def _compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    estimator: Any,
    X_val: Any,
    unique_classes: List[Any],
) -> Dict[str, Any]:
    acc = _safe_float(accuracy_score(y_true, y_pred))
    bal_acc = _safe_float(balanced_accuracy_score(y_true, y_pred))
    prec_macro = _safe_float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec_macro = _safe_float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_mac = _safe_float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_wgt = _safe_float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    roc_auc = None
    log_loss_val = None

    if hasattr(estimator, "predict_proba"):
        try:
            probs = estimator.predict_proba(X_val)
            present_classes = set(pd.Series(y_true).unique())
            if len(present_classes) == len(unique_classes):
                if len(unique_classes) == 2:
                    roc_auc = _safe_float(roc_auc_score(y_true, probs[:, 1]))
                else:
                    roc_auc = _safe_float(
                        roc_auc_score(
                            y_true,
                            probs,
                            multi_class="ovr",
                            average="macro",
                            labels=unique_classes,
                        )
                    )
            log_loss_val = _safe_float(log_loss(y_true, probs, labels=unique_classes))
        except Exception:
            pass

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_mac,
        "f1_weighted": f1_wgt,
        "roc_auc": roc_auc,
        "log_loss": log_loss_val,
        "testing_samples": int(len(y_true)),
    }


def _compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
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


def _calculate_delta_metrics(
    baseline: Dict[str, Any],
    tuned: Dict[str, Any],
    task_type: str,
) -> Dict[str, Any]:
    deltas = {}
    metric_keys = (
        ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "roc_auc", "log_loss"]
        if task_type == "classification"
        else ["rmse", "mae", "r2", "explained_variance", "max_error"]
    )

    for key in metric_keys:
        base_val = baseline.get(key)
        tuned_val = tuned.get(key)

        if base_val is not None and tuned_val is not None and isinstance(base_val, (int, float)) and isinstance(tuned_val, (int, float)):
            abs_delta = round(tuned_val - base_val, 4)
            pct_change = (
                round(((tuned_val - base_val) / abs(base_val)) * 100, 2)
                if base_val != 0
                else None
            )

            # Determine whether this is an improvement
            direction = METRIC_DIRECTIONS.get(key, "maximize")
            if direction == "maximize":
                improved = tuned_val > base_val
            else:
                improved = tuned_val < base_val

            deltas[key] = {
                "baseline": base_val,
                "tuned": tuned_val,
                "absolute_delta": abs_delta,
                "percentage_change": pct_change,
                "improved": improved,
                "direction": direction,
            }
        else:
            deltas[key] = {
                "baseline": base_val,
                "tuned": tuned_val,
                "absolute_delta": None,
                "percentage_change": None,
                "improved": None,
                "direction": METRIC_DIRECTIONS.get(key, "maximize"),
            }

    return deltas


# =========================================================
# Hyperparameter Tuning Service Core
# =========================================================

async def run_hyperparameter_tuning(
    file: UploadFile,
    task_type: str,
    target_column: str,
    model_type: str,
    metric: Optional[str] = None,
    n_trials: int = 20,
    n_splits: int = 3,
    scaler: str = "standard",
    missing: str = "median",
    split_ratio: float = 0.8,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Execute leakage-free Optuna hyperparameter tuning, compare against baseline, and persist."""
    start_time = time.time()
    init_tuning_db()

    run_id = uuid.uuid4().hex
    warnings: List[str] = []

    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------
    normalized_task = (task_type or "").strip().lower()
    if normalized_task not in {"classification", "regression"}:
        raise HTTPException(
            status_code=400,
            detail="task_type must be 'classification' or 'regression'.",
        )

    if model_type in NON_TUNABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=NON_TUNABLE_MODELS[model_type],
        )

    if normalized_task == "classification":
        if model_type not in SUPPORTED_CLASSIFICATION_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model_type}' is not supported for classification tuning. Supported: {sorted(SUPPORTED_CLASSIFICATION_MODELS)}",
            )
        metric = (metric or "f1_macro").strip().lower()
        if metric not in {"f1_macro", "accuracy", "balanced_accuracy", "f1_weighted", "roc_auc", "log_loss"}:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid classification metric: {metric}",
            )
    else:
        if model_type not in SUPPORTED_REGRESSION_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model_type}' is not supported for regression tuning. Supported: {sorted(SUPPORTED_REGRESSION_MODELS)}",
            )
        metric = (metric or "rmse").strip().lower()
        if metric not in {"rmse", "mae", "r2"}:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid regression metric: {metric}",
            )

    direction = METRIC_DIRECTIONS.get(metric, "maximize")

    # Clamping budgets to safe limits
    n_trials = max(1, min(int(n_trials), 50))
    n_splits = max(2, min(int(n_splits), 5))

    # -----------------------------------------------------
    # Read and validate CSV dataset
    # -----------------------------------------------------
    dataset_name = file.filename or "dataset.csv"
    df = await safe_read_csv(file)

    initial_shape = df.shape
    if target_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{target_column}' not found. Available columns: {list(df.columns)}",
        )

    df_clean = df.dropna(subset=[target_column]).copy()
    dropped_target_nans = len(df) - len(df_clean)
    if df_clean.empty:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{target_column}' contains only missing values.",
        )

    # Validate target for task
    unique_classes: List[Any] = []
    if normalized_task == "classification":
        class_counts = df_clean[target_column].value_counts()
        if len(class_counts) < 2:
            raise HTTPException(
                status_code=400,
                detail="Classification requires at least 2 distinct target classes.",
            )
        if len(class_counts) > 20:
            raise HTTPException(
                status_code=400,
                detail=f"Target has {len(class_counts)} classes; maximum supported is 20.",
            )
        unique_classes = sorted(list(class_counts.index.astype(str)))
    else:
        if not pd.api.types.is_numeric_dtype(df_clean[target_column]):
            raise HTTPException(
                status_code=400,
                detail="Regression requires a numeric target column.",
            )

    numeric_columns, categorical_columns = _detect_feature_columns(
        df_clean, target_column=target_column
    )
    feature_columns = [col for col in df_clean.columns if col != target_column]

    X = df_clean[feature_columns]
    y = df_clean[target_column]

    # -----------------------------------------------------
    # Strict Leakage-Free Data Partitioning
    # -----------------------------------------------------
    # Holdout test set is created first and strictly preserved.
    # Optuna optimization and CV evaluate solely on the training split.
    stratify_target = y if (normalized_task == "classification" and y.value_counts().min() >= 2) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=split_ratio,
        random_state=random_state,
        stratify=stratify_target,
    )

    # Determine CV feasibility on training slice
    n_train_samples = len(X_train)
    min_class_count = (
        int(y_train.value_counts().min())
        if normalized_task == "classification"
        else None
    )

    actual_splits = n_splits
    if normalized_task == "classification" and min_class_count is not None:
        if min_class_count < actual_splits:
            actual_splits = max(2, min_class_count)
            warnings.append(
                f"CV folds adjusted to {actual_splits} due to smallest class having {min_class_count} samples in training partition."
            )
    elif n_train_samples < actual_splits * 2:
        actual_splits = max(2, n_train_samples // 2)

    # -----------------------------------------------------
    # Optuna Optimization on Training Split
    # -----------------------------------------------------
    trial_records: List[Dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        # Check if cancellation requested
        if run_id in _CANCELLED_RUNS:
            trial.study.stop()
            raise optuna.TrialPruned("Run cancelled by user.")

        trial_t0 = time.time()
        candidate_model, candidate_params = suggest_model_params(
            trial, model_type, random_state=random_state
        )

        if normalized_task == "classification":
            splitter = StratifiedKFold(
                n_splits=actual_splits,
                shuffle=True,
                random_state=random_state,
            )
            split_gen = splitter.split(X_train, y_train)
        else:
            splitter = KFold(
                n_splits=actual_splits,
                shuffle=True,
                random_state=random_state,
            )
            split_gen = splitter.split(X_train, y_train)

        fold_scores = []
        for train_idx, val_idx in split_gen:
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            # Fit preprocessor strictly inside fold
            fold_prep = build_preprocessor(
                numeric_columns=numeric_columns,
                categorical_columns=categorical_columns,
                scaler=scaler,
                missing=missing,
            )
            fold_prep.fit(X_tr)
            X_tr_trans = fold_prep.transform(X_tr)
            X_val_trans = fold_prep.transform(X_val)

            # Fit candidate model
            try:
                candidate_model.fit(X_tr_trans, y_tr)
                y_val_pred = candidate_model.predict(X_val_trans)
            except Exception as exc:
                raise optuna.TrialPruned(f"Model fitting failed: {exc}")

            # Compute fold metric
            if normalized_task == "classification":
                metrics_res = _compute_classification_metrics(
                    y_true=y_val.to_numpy(),
                    y_pred=y_val_pred,
                    estimator=candidate_model,
                    X_val=X_val_trans,
                    unique_classes=unique_classes,
                )
            else:
                metrics_res = _compute_regression_metrics(
                    y_true=y_val.to_numpy(),
                    y_pred=y_val_pred,
                )

            val_score = metrics_res.get(metric)
            if val_score is None:
                # If metric calculation produced None (e.g. roc_auc failure), fallback
                raise optuna.TrialPruned(f"Metric {metric} could not be computed.")
            fold_scores.append(val_score)

        mean_cv_score = float(np.mean(fold_scores))
        trial_duration = round(time.time() - trial_t0, 4)

        trial_records.append({
            "trial_number": trial.number,
            "params": candidate_params,
            "cv_score": round(mean_cv_score, 4),
            "duration_seconds": trial_duration,
            "state": "COMPLETE",
        })

        return mean_cv_score

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(
        study_name=f"tuning_{run_id}",
        direction=direction,
        sampler=sampler,
    )

    try:
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=180,  # 3 minute timeout safety bound
            catch=(Exception,),
        )
    finally:
        _CANCELLED_RUNS.discard(run_id)

    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed_trials:
        raise HTTPException(
            status_code=500,
            detail="All tuning trials failed or were pruned. Check dataset quality and parameters.",
        )

    best_trial = study.best_trial
    best_params = best_trial.params
    best_score = round(best_trial.value, 4)

    # -----------------------------------------------------
    # Baseline vs Tuned Holdout Evaluation
    # -----------------------------------------------------
    # Evaluate baseline (default parameters) on untouched holdout set
    baseline_estimator = create_baseline_model(model_type, random_state=random_state)
    baseline_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(
                    numeric_columns=numeric_columns,
                    categorical_columns=categorical_columns,
                    scaler=scaler,
                    missing=missing,
                ),
            ),
            ("estimator", baseline_estimator),
        ]
    )
    baseline_pipeline.fit(X_train, y_train)
    y_test_pred_base = baseline_pipeline.predict(X_test)

    if normalized_task == "classification":
        # Extract preprocessor transformed X_test for probability metrics
        prep_step = baseline_pipeline.named_steps["preprocessor"]
        est_step = baseline_pipeline.named_steps["estimator"]
        X_test_prep = prep_step.transform(X_test)
        baseline_holdout_metrics = _compute_classification_metrics(
            y_true=y_test.to_numpy(),
            y_pred=y_test_pred_base,
            estimator=est_step,
            X_val=X_test_prep,
            unique_classes=unique_classes,
        )
    else:
        baseline_holdout_metrics = _compute_regression_metrics(
            y_true=y_test.to_numpy(),
            y_pred=y_test_pred_base,
        )

    # Evaluate tuned pipeline with best parameters on untouched holdout set
    tuned_estimator = build_model_from_params(
        model_type=model_type,
        params=best_params,
        random_state=random_state,
    )
    tuned_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(
                    numeric_columns=numeric_columns,
                    categorical_columns=categorical_columns,
                    scaler=scaler,
                    missing=missing,
                ),
            ),
            ("estimator", tuned_estimator),
        ]
    )
    tuned_pipeline.fit(X_train, y_train)
    y_test_pred_tuned = tuned_pipeline.predict(X_test)

    if normalized_task == "classification":
        prep_step_tuned = tuned_pipeline.named_steps["preprocessor"]
        est_step_tuned = tuned_pipeline.named_steps["estimator"]
        X_test_prep_tuned = prep_step_tuned.transform(X_test)
        tuned_holdout_metrics = _compute_classification_metrics(
            y_true=y_test.to_numpy(),
            y_pred=y_test_pred_tuned,
            estimator=est_step_tuned,
            X_val=X_test_prep_tuned,
            unique_classes=unique_classes,
        )
    else:
        tuned_holdout_metrics = _compute_regression_metrics(
            y_true=y_test.to_numpy(),
            y_pred=y_test_pred_tuned,
        )

    # Calculate delta improvement
    delta_metrics = _calculate_delta_metrics(
        baseline=baseline_holdout_metrics,
        tuned=tuned_holdout_metrics,
        task_type=normalized_task,
    )

    # -----------------------------------------------------
    # Model Registration for Persistence & Explainability
    # -----------------------------------------------------
    # Refit winning pipeline on full dataset for downstream inference/registry
    full_winning_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(
                    numeric_columns=numeric_columns,
                    categorical_columns=categorical_columns,
                    scaler=scaler,
                    missing=missing,
                ),
            ),
            (
                "estimator",
                build_model_from_params(
                    model_type=model_type,
                    params=best_params,
                    random_state=random_state,
                ),
            ),
        ]
    )
    full_winning_pipeline.fit(X, y)

    registered_model_record = register_model(
        pipeline=full_winning_pipeline,
        model_name=f"Tuned_{model_type}_{run_id[:8]}",
        task_type=normalized_task,
        model_type=model_type,
        dataset_name=dataset_name,
        target_column=target_column,
        feature_columns=feature_columns,
        metrics=tuned_holdout_metrics,
    )
    registered_model_id = registered_model_record.get("id")

    # -----------------------------------------------------
    # Save Tuning Run & Trial Records to SQLite
    # -----------------------------------------------------
    total_duration = round(time.time() - start_time, 2)
    created_at = datetime.now(timezone.utc).isoformat()

    data_quality_info = {
        "initial_shape": f"{initial_shape[0]} rows × {initial_shape[1]} cols",
        "final_shape": f"{df_clean.shape[0]} rows × {df_clean.shape[1]} cols",
        "training_samples": int(len(X_train)),
        "testing_samples": int(len(X_test)),
        "dropped_target_nans": int(dropped_target_nans),
    }

    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO tuning_runs (
                run_id, task_type, model_type, target_column, dataset_name,
                metric, direction, n_trials, n_splits, status,
                best_trial_number, best_score, best_params_json,
                baseline_metrics_json, tuned_metrics_json, delta_metrics_json,
                registered_model_id, feature_columns_json, data_quality_json,
                warnings_json, duration_seconds, error_message, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                normalized_task,
                model_type,
                target_column,
                dataset_name,
                metric,
                direction,
                int(n_trials),
                int(actual_splits),
                "completed",
                int(best_trial.number),
                float(best_score),
                json.dumps(_json_safe(best_params)),
                json.dumps(_json_safe(baseline_holdout_metrics)),
                json.dumps(_json_safe(tuned_holdout_metrics)),
                json.dumps(_json_safe(delta_metrics)),
                registered_model_id,
                json.dumps(_json_safe(feature_columns)),
                json.dumps(_json_safe(data_quality_info)),
                json.dumps(_json_safe(warnings)),
                float(total_duration),
                None,
                created_at,
            ),
        )

        for rec in trial_records:
            connection.execute(
                """
                INSERT INTO tuning_trials (
                    run_id, trial_number, params_json, cv_score,
                    duration_seconds, state, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    rec["trial_number"],
                    json.dumps(_json_safe(rec["params"])),
                    rec["cv_score"],
                    rec["duration_seconds"],
                    rec["state"],
                    created_at,
                ),
            )

    return {
        "run_id": run_id,
        "task_type": normalized_task,
        "model_type": model_type,
        "target_column": target_column,
        "dataset_name": dataset_name,
        "metric": metric,
        "direction": direction,
        "n_trials": n_trials,
        "n_splits": actual_splits,
        "status": "completed",
        "best_trial_number": int(best_trial.number),
        "best_score": float(best_score),
        "best_params": best_params,
        "baseline_metrics": baseline_holdout_metrics,
        "tuned_metrics": tuned_holdout_metrics,
        "delta_metrics": delta_metrics,
        "registered_model_id": registered_model_id,
        "feature_columns": feature_columns,
        "trials": trial_records,
        "data_quality": data_quality_info,
        "warnings": warnings,
        "duration_seconds": total_duration,
        "created_at": created_at,
    }


# =========================================================
# Query & Management Endpoints
# =========================================================

def list_tuning_runs(limit: int = 50, offset: int = 0) -> Dict[str, Any]:
    init_tuning_db()
    with _connect() as connection:
        total = connection.execute("SELECT COUNT(*) FROM tuning_runs").fetchone()[0]
        cursor = connection.execute(
            """
            SELECT id, run_id, task_type, model_type, target_column, dataset_name,
                   metric, direction, n_trials, n_splits, status, best_trial_number,
                   best_score, registered_model_id, duration_seconds, created_at
            FROM tuning_runs
            ORDER BY id DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        )
        rows = [dict(row) for row in cursor.fetchall()]

    return {
        "items": rows,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


def get_tuning_run(run_id: str) -> Dict[str, Any]:
    init_tuning_db()
    with _connect() as connection:
        cursor = connection.execute(
            "SELECT * FROM tuning_runs WHERE run_id = ?",
            (run_id,),
        )
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"Tuning run '{run_id}' not found.")

        item = dict(row)
        item["best_params"] = json.loads(item.pop("best_params_json") or "{}")
        item["baseline_metrics"] = json.loads(item.pop("baseline_metrics_json") or "{}")
        item["tuned_metrics"] = json.loads(item.pop("tuned_metrics_json") or "{}")
        item["delta_metrics"] = json.loads(item.pop("delta_metrics_json") or "{}")
        item["feature_columns"] = json.loads(item.pop("feature_columns_json") or "[]")
        item["data_quality"] = json.loads(item.pop("data_quality_json") or "{}")
        item["warnings"] = json.loads(item.pop("warnings_json") or "[]")

        # Fetch trials
        trials_cursor = connection.execute(
            """
            SELECT trial_number, params_json, cv_score, duration_seconds, state
            FROM tuning_trials
            WHERE run_id = ?
            ORDER BY trial_number ASC
            """,
            (run_id,),
        )
        trials = []
        for t_row in trials_cursor.fetchall():
            t_dict = dict(t_row)
            t_dict["params"] = json.loads(t_dict.pop("params_json") or "{}")
            trials.append(t_dict)

        item["trials"] = trials
        return item


def delete_tuning_run(run_id: str, delete_registered_model: bool = False) -> Dict[str, Any]:
    init_tuning_db()
    with _connect() as connection:
        cursor = connection.execute(
            "SELECT id, registered_model_id FROM tuning_runs WHERE run_id = ?",
            (run_id,),
        )
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"Tuning run '{run_id}' not found.")

        registered_model_id = row["registered_model_id"]

        connection.execute("DELETE FROM tuning_trials WHERE run_id = ?", (run_id,))
        connection.execute("DELETE FROM tuning_runs WHERE run_id = ?", (run_id,))

    if delete_registered_model and registered_model_id:
        try:
            from app.services.model_registry_service import delete_model
            delete_model(registered_model_id)
        except Exception:
            pass

    return {"message": f"Tuning run '{run_id}' deleted successfully."}


def cancel_tuning_run(run_id: str) -> Dict[str, Any]:
    _CANCELLED_RUNS.add(run_id)
    return {"message": f"Cancellation requested for tuning run '{run_id}'."}
