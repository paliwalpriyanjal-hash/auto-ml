"""
Automated model comparison service for tabular datasets.

Compares all supported models for a given task type using cross-validation
on the training split, then evaluates the best model on a held-out test set.
Preprocessing is performed inside sklearn Pipelines to prevent data leakage.
"""
import io
import math
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from fastapi import UploadFile, HTTPException

from sklearn.model_selection import (
    train_test_split,
    cross_validate,
    StratifiedKFold,
    KFold,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    ExtraTreesClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    confusion_matrix,
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    explained_variance_score,
    max_error as sklearn_max_error,
)


# ---------------------------------------------------------------------------
# JSON-safety helpers
# ---------------------------------------------------------------------------

def _safe_float(val: Any) -> Optional[float]:
    """Convert value to float, ensuring JSON serializability (no NaN/Inf)."""
    if val is None:
        return None
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return round(f, 4)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Candidate model registries — matching existing project names exactly
# ---------------------------------------------------------------------------

# Classification models match classification_service._get_classifier()
CLASSIFICATION_MODELS: Dict[str, Any] = {
    "LogisticRegression": lambda rs: LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=rs
    ),
    "DecisionTreeClassifier": lambda rs: DecisionTreeClassifier(
        class_weight="balanced", random_state=rs
    ),
    "RandomForestClassifier": lambda rs: RandomForestClassifier(
        n_estimators=100, class_weight="balanced", n_jobs=-1, random_state=rs
    ),
    "ExtraTreesClassifier": lambda rs: ExtraTreesClassifier(
        n_estimators=100, class_weight="balanced", n_jobs=-1, random_state=rs
    ),
}

# Regression models match train_service model_map keys
REGRESSION_MODELS: Dict[str, Any] = {
    "LinearRegression": lambda rs: LinearRegression(),
    "DecisionTree": lambda rs: DecisionTreeRegressor(random_state=rs),
    "RandomForest": lambda rs: RandomForestRegressor(
        n_estimators=100, n_jobs=-1, random_state=rs
    ),
}

# Ranking metric configuration
CLASSIFICATION_RANKING_METRICS = {
    "f1_macro": {"sklearn_scorer": "f1_macro", "higher_is_better": True},
    "balanced_accuracy": {"sklearn_scorer": "balanced_accuracy", "higher_is_better": True},
    "accuracy": {"sklearn_scorer": "accuracy", "higher_is_better": True},
    "f1_weighted": {"sklearn_scorer": "f1_weighted", "higher_is_better": True},
}

REGRESSION_RANKING_METRICS = {
    "rmse": {"sklearn_scorer": "neg_root_mean_squared_error", "higher_is_better": False},
    "mae": {"sklearn_scorer": "neg_mean_absolute_error", "higher_is_better": False},
    "r2": {"sklearn_scorer": "r2", "higher_is_better": True},
}


# ---------------------------------------------------------------------------
# Validation helpers — reuse the same validation logic as classification_service
# ---------------------------------------------------------------------------

def _validate_target_classification(
    df: pd.DataFrame,
    target_column: str,
    max_classes: int = 20,
    min_classes: int = 2,
):
    """Validate target for classification. Returns (df_clean, dropped_nans, value_counts)."""
    if target_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{target_column}' was not found in dataset columns: {list(df.columns)}",
        )

    initial_rows = len(df)
    df_clean = df.dropna(subset=[target_column]).copy()
    dropped_target_nans = initial_rows - len(df_clean)

    if df_clean.empty:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{target_column}' contains only missing/NaN values.",
        )

    target_series = df_clean[target_column]

    if pd.api.types.is_float_dtype(target_series):
        nunique = target_series.nunique()
        if nunique > max_classes or (nunique > 10 and nunique / len(df_clean) > 0.1):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Target column '{target_column}' has continuous floating-point values "
                    f"({nunique} distinct values). This looks like a continuous regression target. "
                    "Please use regression mode or discretize the target."
                ),
            )

    value_counts = target_series.value_counts(dropna=True)
    num_classes = len(value_counts)

    if num_classes < min_classes:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Target column '{target_column}' contains only {num_classes} unique class: "
                f"{list(value_counts.index)}. Classification requires at least 2 distinct classes."
            ),
        )

    if num_classes > max_classes:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Target column '{target_column}' contains {num_classes} classes, exceeding the "
                f"initial supported limit of {max_classes}."
            ),
        )

    return df_clean, dropped_target_nans, value_counts


def _validate_target_regression(df: pd.DataFrame, target_column: str):
    """Validate target for regression. Returns (df_clean, dropped_nans)."""
    if target_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{target_column}' was not found in dataset columns: {list(df.columns)}",
        )

    initial_rows = len(df)
    df_clean = df.dropna(subset=[target_column]).copy()
    dropped_target_nans = initial_rows - len(df_clean)

    if df_clean.empty:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{target_column}' contains only missing/NaN values.",
        )

    target_series = df_clean[target_column]
    if not pd.api.types.is_numeric_dtype(target_series):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Target column '{target_column}' is not numeric. "
                "Regression requires a continuous numeric target."
            ),
        )

    return df_clean, dropped_target_nans


def _detect_feature_columns(
    df: pd.DataFrame,
    target_column: str,
    max_categorical_cardinality: int = 50,
    max_text_words_threshold: float = 3.0,
):
    """
    Validates feature columns and identifies numeric vs categorical features.
    Mirrors classification_service._detect_and_validate_columns().
    """
    feature_cols = [c for c in df.columns if c != target_column]
    if not feature_cols:
        raise HTTPException(
            status_code=400,
            detail=f"No feature columns found. Dataset only contains target column '{target_column}'.",
        )

    numeric_cols: List[str] = []
    categorical_cols: List[str] = []

    for col in feature_cols:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            if series.dropna().empty:
                raise HTTPException(
                    status_code=400,
                    detail=f"Feature column '{col}' contains only empty/NaN values.",
                )
            numeric_cols.append(col)
        else:
            non_na = series.dropna().astype(str)
            if non_na.empty:
                raise HTTPException(
                    status_code=400,
                    detail=f"Feature column '{col}' contains only empty/NaN values.",
                )
            avg_word_count = non_na.apply(lambda s: len(s.split())).mean()
            if avg_word_count > max_text_words_threshold:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Column '{col}' appears to contain free-form text "
                        f"(average {avg_word_count:.1f} words per row). "
                        "High-dimensional free text is not supported."
                    ),
                )
            nunique = non_na.nunique()
            if nunique > max_categorical_cardinality:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Column '{col}' has excessive cardinality ({nunique} distinct values "
                        f"> limit {max_categorical_cardinality}). "
                        "Categorical features with excessive distinct levels are not supported."
                    ),
                )
            categorical_cols.append(col)

    return numeric_cols, categorical_cols


# ---------------------------------------------------------------------------
# Preprocessor builder — matches classification_service._build_preprocessor()
# ---------------------------------------------------------------------------

def _build_preprocessor(
    numeric_cols: List[str],
    categorical_cols: List[str],
    scaler: str,
    missing: str,
) -> ColumnTransformer:
    """
    Constructs a ColumnTransformer for use inside a Pipeline.
    Numeric: SimpleImputer + optional Scaler.
    Categorical: SimpleImputer(most_frequent) + OneHotEncoder(sparse_output=True, handle_unknown='ignore').
    """
    transformers = []

    if numeric_cols:
        num_impute_strategy = "mean" if missing == "mean" else "median"
        num_steps = [("imputer", SimpleImputer(strategy=num_impute_strategy))]
        if scaler == "standard":
            num_steps.append(("scaler", StandardScaler()))
        elif scaler == "minmax":
            num_steps.append(("scaler", MinMaxScaler()))
        elif scaler == "robust":
            num_steps.append(("scaler", RobustScaler()))
        num_pipe = Pipeline(num_steps)
        transformers.append(("num", num_pipe, numeric_cols))

    if categorical_cols:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ])
        transformers.append(("cat", cat_pipe, categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


# ---------------------------------------------------------------------------
# Determine feasible CV fold count
# ---------------------------------------------------------------------------

def _determine_cv_folds(
    n_samples_train: int,
    min_class_count: Optional[int] = None,
    is_classification: bool = True,
) -> int:
    """
    Determine the number of CV folds based on data size and class distribution.
    Returns a value between 2 and 10.
    """
    max_folds = 10

    if is_classification and min_class_count is not None:
        # Cannot have more folds than the smallest class count in training
        max_folds = min(max_folds, min_class_count)

    # General data size constraint
    max_folds = min(max_folds, n_samples_train // 2)

    # Clamp to [2, 10]
    return max(2, min(max_folds, 10))


# ---------------------------------------------------------------------------
# Core comparison engine
# ---------------------------------------------------------------------------

async def compare_models_service(
    file: UploadFile,
    task_type: str,
    target_column: Optional[str],
    scaler: str,
    split_ratio: float,
    missing: str,
    encoding: str,
    ranking_metric: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compare all candidate models for a given task type.

    Evaluation sequence:
    1. Validate the dataset and selected target.
    2. Split into train + held-out test.
    3. Run cross-validation on training split only (preprocessing inside pipeline).
    4. Rank models by CV score.
    5. Refit the winner on the full training split.
    6. Evaluate winner on the untouched holdout test set.
    """
    warnings: List[str] = []
    RANDOM_STATE = 42

    # -----------------------------------------------------------------------
    # 0. Parameter validation
    # -----------------------------------------------------------------------
    normalized_task = (task_type or "").strip().lower()
    if normalized_task not in ("classification", "regression"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported task_type '{task_type}'. Supported: 'regression', 'classification'.",
        )

    if normalized_task == "classification":
        if not target_column or not target_column.strip():
            raise HTTPException(
                status_code=400,
                detail="The 'target_column' parameter is required for classification tasks.",
            )
        target_column = target_column.strip()

    if normalized_task == "regression":
        if not target_column or not target_column.strip():
            raise HTTPException(
                status_code=400,
                detail="The 'target_column' parameter is required for the comparison endpoint.",
            )
        target_column = target_column.strip()

    if split_ratio <= 0.0 or split_ratio >= 1.0:
        raise HTTPException(
            status_code=400,
            detail=f"splitRatio must be strictly between 0 and 1, got {split_ratio}.",
        )

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    # -----------------------------------------------------------------------
    # 1. Read CSV
    # -----------------------------------------------------------------------
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except pd.errors.EmptyDataError:
        raise HTTPException(
            status_code=400, detail="The CSV file appears to be empty or corrupt."
        )
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to parse CSV file: {str(e)}"
        )

    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded file contains no rows.")

    initial_shape = df.shape

    # -----------------------------------------------------------------------
    # 2. Validate target and features
    # -----------------------------------------------------------------------
    if normalized_task == "classification":
        df_clean, dropped_target_nans, class_counts = _validate_target_classification(
            df, target_column
        )
        if dropped_target_nans > 0:
            warnings.append(
                f"Dropped {dropped_target_nans} rows where target '{target_column}' was missing/NaN."
            )
    else:
        df_clean, dropped_target_nans = _validate_target_regression(df, target_column)
        class_counts = None
        if dropped_target_nans > 0:
            warnings.append(
                f"Dropped {dropped_target_nans} rows where target '{target_column}' was missing/NaN."
            )

    numeric_cols, categorical_cols = _detect_feature_columns(df_clean, target_column)

    # Handle explicit row-drop missing strategy
    dropped_feature_nans = 0
    if missing == "drop":
        before_drop = len(df_clean)
        df_clean = df_clean.dropna().copy()
        dropped_feature_nans = before_drop - len(df_clean)
        if df_clean.empty:
            raise HTTPException(
                status_code=400,
                detail="All rows were removed after dropping rows with missing feature values.",
            )
        if normalized_task == "classification":
            class_counts = df_clean[target_column].value_counts()
            if len(class_counts) < 2:
                raise HTTPException(
                    status_code=400,
                    detail=f"After dropping missing rows, target '{target_column}' has fewer than 2 distinct classes.",
                )

    # Separate X and y
    X = df_clean[numeric_cols + categorical_cols]
    if normalized_task == "classification":
        unique_classes = sorted(list(class_counts.index.map(str)))
        y = df_clean[target_column].astype(str)
    else:
        y = df_clean[target_column]

    # -----------------------------------------------------------------------
    # 3. Determine ranking metric
    # -----------------------------------------------------------------------
    if normalized_task == "classification":
        valid_metrics = CLASSIFICATION_RANKING_METRICS
        default_metric = "f1_macro"
        candidate_models = CLASSIFICATION_MODELS
    else:
        valid_metrics = REGRESSION_RANKING_METRICS
        default_metric = "rmse"
        candidate_models = REGRESSION_MODELS

    ranking_metric = (ranking_metric or default_metric).strip().lower()
    if ranking_metric not in valid_metrics:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported ranking_metric '{ranking_metric}'. Supported: {list(valid_metrics.keys())}.",
        )

    metric_config = valid_metrics[ranking_metric]
    sklearn_scorer = metric_config["sklearn_scorer"]
    higher_is_better = metric_config["higher_is_better"]

    # -----------------------------------------------------------------------
    # 4. Train-test split
    # -----------------------------------------------------------------------
    test_size = 1.0 - split_ratio

    if normalized_task == "classification":
        min_class_count = int(class_counts.min())
        if min_class_count < 2:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Cannot perform stratified evaluation: class '{class_counts.idxmin()}' "
                    f"has only {min_class_count} sample(s). Each class must have at least 2 samples."
                ),
            )
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
            )
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Stratified train-test split failed: {str(e)}",
            )
        # Recount min class in training split for CV fold calculation
        train_class_counts = y_train.value_counts()
        min_train_class = int(train_class_counts.min())
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_STATE
        )
        min_train_class = None

    # -----------------------------------------------------------------------
    # 5. Determine CV folds
    # -----------------------------------------------------------------------
    cv_folds = _determine_cv_folds(
        n_samples_train=len(X_train),
        min_class_count=min_train_class,
        is_classification=(normalized_task == "classification"),
    )

    if normalized_task == "classification":
        cv_splitter = StratifiedKFold(
            n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE
        )
    else:
        cv_splitter = KFold(
            n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE
        )

    if len(X_train) < 2 * cv_folds:
        warnings.append(
            f"Training set has only {len(X_train)} samples for {cv_folds}-fold CV. "
            "Results may have high variance."
        )

    # -----------------------------------------------------------------------
    # 6. Cross-validate each candidate model
    # -----------------------------------------------------------------------
    model_results: List[Dict[str, Any]] = []

    for model_name, model_factory in candidate_models.items():
        preprocessor = _build_preprocessor(
            numeric_cols, categorical_cols, scaler, missing
        )
        estimator = model_factory(RANDOM_STATE)
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("estimator", estimator),
        ])

        try:
            cv_out = cross_validate(
                pipeline,
                X_train,
                y_train,
                cv=cv_splitter,
                scoring=sklearn_scorer,
                return_train_score=False,
                error_score="raise",
            )
            raw_scores = cv_out["test_score"]

            # Convert sklearn's negative scoring convention for RMSE/MAE
            if sklearn_scorer.startswith("neg_"):
                display_scores = -raw_scores  # make positive
            else:
                display_scores = raw_scores

            cv_mean = _safe_float(np.mean(display_scores))
            cv_std = _safe_float(np.std(display_scores))

            model_results.append({
                "model": model_name,
                "cv_mean": cv_mean,
                "cv_std": cv_std,
                "status": "success",
                "error": None,
            })
        except Exception as e:
            model_results.append({
                "model": model_name,
                "cv_mean": None,
                "cv_std": None,
                "status": "failed",
                "error": str(e),
            })
            warnings.append(f"Model '{model_name}' failed during cross-validation: {str(e)}")

    # -----------------------------------------------------------------------
    # 7. Rank models
    # -----------------------------------------------------------------------
    successful = [r for r in model_results if r["status"] == "success"]

    if not successful:
        raise HTTPException(
            status_code=500,
            detail="All candidate models failed during cross-validation. Cannot determine a winner.",
        )

    # Sort: higher_is_better → descending; lower_is_better → ascending
    successful.sort(
        key=lambda r: r["cv_mean"] if r["cv_mean"] is not None else (float("-inf") if higher_is_better else float("inf")),
        reverse=higher_is_better,
    )

    # Assign ranks (1-based)
    for rank_idx, result in enumerate(successful):
        result["rank"] = rank_idx + 1

    # Failed models get no rank
    for result in model_results:
        if result["status"] == "failed":
            result["rank"] = None

    best_model_name = successful[0]["model"]
    best_cv_mean = successful[0]["cv_mean"]
    best_cv_std = successful[0]["cv_std"]

    # -----------------------------------------------------------------------
    # 8. Refit winner on full training set and evaluate on holdout
    # -----------------------------------------------------------------------
    preprocessor = _build_preprocessor(numeric_cols, categorical_cols, scaler, missing)
    best_estimator = candidate_models[best_model_name](RANDOM_STATE)
    best_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("estimator", best_estimator),
    ])

    try:
        best_pipeline.fit(X_train, y_train)
        y_pred = best_pipeline.predict(X_test)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Best model '{best_model_name}' failed during holdout evaluation: {str(e)}",
        )

    # -----------------------------------------------------------------------
    # 9. Compute holdout metrics
    # -----------------------------------------------------------------------
    if normalized_task == "classification":
        holdout_metrics = _compute_classification_holdout_metrics(
            best_pipeline, X_test, y_test, y_pred, unique_classes, warnings
        )
    else:
        holdout_metrics = _compute_regression_holdout_metrics(y_test, y_pred)

    # -----------------------------------------------------------------------
    # 10. Build response
    # -----------------------------------------------------------------------
    response: Dict[str, Any] = {
        "task_type": normalized_task,
        "target_column": target_column,
        "ranking_metric": ranking_metric,
        "cv_folds": cv_folds,
        "split_ratio": split_ratio,
        "model_results": model_results,
        "best_model": {
            "model": best_model_name,
            "cv_mean": best_cv_mean,
            "cv_std": best_cv_std,
            "holdout_metrics": holdout_metrics,
        },
        "warnings": warnings,
        "data_quality": {
            "initial_shape": f"{initial_shape[0]} rows × {initial_shape[1]} cols",
            "final_shape": f"{df_clean.shape[0]} rows × {df_clean.shape[1]} cols",
            "training_samples": int(len(X_train)),
            "testing_samples": int(len(X_test)),
            "dropped_target_nans": dropped_target_nans,
            "dropped_feature_nans": dropped_feature_nans,
            "feature_counts": {
                "numeric": len(numeric_cols),
                "categorical": len(categorical_cols),
            },
        },
    }

    if normalized_task == "classification":
        response["classes"] = unique_classes
        response["data_quality"]["class_distribution"] = {
            str(k): int(v) for k, v in class_counts.items()
        }

    return response


# ---------------------------------------------------------------------------
# Holdout metric helpers
# ---------------------------------------------------------------------------

def _compute_classification_holdout_metrics(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    unique_classes: List[str],
    warnings: List[str],
) -> Dict[str, Any]:
    """Compute classification holdout metrics matching existing project conventions."""
    acc = _safe_float(accuracy_score(y_test, y_pred))
    bal_acc = _safe_float(balanced_accuracy_score(y_test, y_pred))
    prec_macro = _safe_float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    rec_macro = _safe_float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    f1_macro = _safe_float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    f1_weighted = _safe_float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    # ROC-AUC
    roc_auc = None
    y_prob = None
    has_predict_proba = hasattr(pipeline.named_steps["estimator"], "predict_proba")
    if has_predict_proba:
        try:
            y_prob = pipeline.predict_proba(X_test)
        except Exception:
            y_prob = None

    if y_prob is not None:
        try:
            present_classes_test = set(y_test.unique())
            if len(present_classes_test) == len(unique_classes):
                if len(unique_classes) == 2:
                    roc_auc = _safe_float(roc_auc_score(y_test, y_prob[:, 1]))
                else:
                    roc_auc = _safe_float(roc_auc_score(
                        y_test, y_prob, multi_class="ovr",
                        average="macro", labels=unique_classes
                    ))
            else:
                warnings.append(
                    "ROC-AUC could not be computed: not all training classes were present in the test holdout."
                )
        except Exception as e:
            warnings.append(f"ROC-AUC computation skipped: {str(e)}")

    # Log Loss
    logloss_val = None
    if y_prob is not None:
        try:
            logloss_val = _safe_float(log_loss(y_test, y_prob, labels=unique_classes))
        except Exception as e:
            warnings.append(f"Log loss computation skipped: {str(e)}")

    # Confusion Matrix
    cm_matrix = confusion_matrix(y_test, y_pred, labels=unique_classes)
    confusion_matrix_data = {
        "labels": unique_classes,
        "matrix": [[int(cell) for cell in row] for row in cm_matrix],
    }

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "roc_auc": roc_auc,
        "log_loss": logloss_val,
        "confusion_matrix": confusion_matrix_data,
        "training_samples": int(len(X_test) * 0),  # placeholder, overridden below
        "testing_samples": int(len(X_test)),
    }


def _compute_regression_holdout_metrics(
    y_test: pd.Series,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    """Compute regression holdout metrics matching existing project conventions."""
    y_test_arr = np.asarray(y_test)
    return {
        "rmse": _safe_float(np.sqrt(mean_squared_error(y_test_arr, y_pred))),
        "r2": _safe_float(r2_score(y_test_arr, y_pred)),
        "mae": _safe_float(mean_absolute_error(y_test_arr, y_pred)),
        "explained_variance": _safe_float(explained_variance_score(y_test_arr, y_pred)),
        "max_error": _safe_float(sklearn_max_error(y_test_arr, y_pred)),
        "testing_samples": int(len(y_test)),
    }
