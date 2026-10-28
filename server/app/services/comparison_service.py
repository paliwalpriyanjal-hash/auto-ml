
"""
Automated model comparison service for tabular datasets.

Features:
- Classification and regression model comparison
- Cross-validation on the training split
- Holdout evaluation
- Preprocessing inside sklearn pipelines
- JSON-safe metrics and error handling
- Optional fitted winning pipeline for Model Registry
"""

import io
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from fastapi import HTTPException, UploadFile
from app.utils.security import safe_read_csv

from sklearn.model_selection import (
    train_test_split,
    cross_validate,
    StratifiedKFold,
    KFold,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    OneHotEncoder,
)
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


# =========================================================
# JSON safety
# =========================================================

def _safe_float(value: Any) -> Optional[float]:
    """Convert a value into a finite, JSON-safe float."""
    if value is None:
        return None

    try:
        result = float(value)

        if not math.isfinite(result):
            return None

        return round(result, 4)

    except (TypeError, ValueError, OverflowError):
        return None


# =========================================================
# Candidate models
# =========================================================

CLASSIFICATION_MODELS: Dict[str, Any] = {
    "LogisticRegression": lambda rs: LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=rs,
    ),
    "DecisionTreeClassifier": lambda rs: DecisionTreeClassifier(
        class_weight="balanced",
        random_state=rs,
    ),
    "RandomForestClassifier": lambda rs: RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        n_jobs=1,
        random_state=rs,
    ),
    "ExtraTreesClassifier": lambda rs: ExtraTreesClassifier(
        n_estimators=100,
        class_weight="balanced",
        n_jobs=1,
        random_state=rs,
    ),
}

REGRESSION_MODELS: Dict[str, Any] = {
    "LinearRegression": lambda rs: LinearRegression(),
    "DecisionTree": lambda rs: DecisionTreeRegressor(
        random_state=rs,
    ),
    "RandomForest": lambda rs: RandomForestRegressor(
        n_estimators=100,
        n_jobs=1,
        random_state=rs,
    ),
}

CLASSIFICATION_RANKING_METRICS = {
    "f1_macro": {
        "sklearn_scorer": "f1_macro",
        "higher_is_better": True,
    },
    "balanced_accuracy": {
        "sklearn_scorer": "balanced_accuracy",
        "higher_is_better": True,
    },
    "accuracy": {
        "sklearn_scorer": "accuracy",
        "higher_is_better": True,
    },
    "f1_weighted": {
        "sklearn_scorer": "f1_weighted",
        "higher_is_better": True,
    },
}

REGRESSION_RANKING_METRICS = {
    "rmse": {
        "sklearn_scorer": "neg_root_mean_squared_error",
        "higher_is_better": False,
    },
    "mae": {
        "sklearn_scorer": "neg_mean_absolute_error",
        "higher_is_better": False,
    },
    "r2": {
        "sklearn_scorer": "r2",
        "higher_is_better": True,
    },
}


# =========================================================
# Target validation
# =========================================================

def _validate_target_classification(
    df: pd.DataFrame,
    target_column: str,
    max_classes: int = 20,
):
    if target_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Target column '{target_column}' was not found. "
                f"Available columns: {list(df.columns)}"
            ),
        )

    original_rows = len(df)
    cleaned = df.dropna(subset=[target_column]).copy()
    dropped_target_nans = original_rows - len(cleaned)

    if cleaned.empty:
        raise HTTPException(
            status_code=400,
            detail=f"Target '{target_column}' contains only missing values.",
        )

    target = cleaned[target_column]

    # Catch likely continuous floating-point regression targets.
    if pd.api.types.is_float_dtype(target):
        unique_count = target.nunique()

        if (
            unique_count > max_classes
            or (
                unique_count > 10
                and unique_count / len(cleaned) > 0.1
            )
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Target '{target_column}' appears continuous. "
                    "Use regression mode or discretize the target."
                ),
            )

    class_counts = target.value_counts(dropna=True)
    number_of_classes = len(class_counts)

    if number_of_classes < 2:
        raise HTTPException(
            status_code=400,
            detail="Classification requires at least two target classes.",
        )

    if number_of_classes > max_classes:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Target has {number_of_classes} classes; "
                f"the supported maximum is {max_classes}."
            ),
        )

    return cleaned, dropped_target_nans, class_counts


def _validate_target_regression(
    df: pd.DataFrame,
    target_column: str,
):
    if target_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Target column '{target_column}' was not found. "
                f"Available columns: {list(df.columns)}"
            ),
        )

    original_rows = len(df)
    cleaned = df.dropna(subset=[target_column]).copy()
    dropped_target_nans = original_rows - len(cleaned)

    if cleaned.empty:
        raise HTTPException(
            status_code=400,
            detail=f"Target '{target_column}' contains only missing values.",
        )

    if not pd.api.types.is_numeric_dtype(cleaned[target_column]):
        raise HTTPException(
            status_code=400,
            detail="Regression requires a numeric target column.",
        )

    return cleaned, dropped_target_nans


# =========================================================
# Feature detection
# =========================================================

def _detect_feature_columns(
    df: pd.DataFrame,
    target_column: str,
    max_categorical_cardinality: int = 50,
    max_text_words_threshold: float = 3.0,
):
    feature_columns = [
        column for column in df.columns
        if column != target_column
    ]

    if not feature_columns:
        raise HTTPException(
            status_code=400,
            detail="No feature columns were found besides the target.",
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

            average_words = non_missing.apply(
                lambda value: len(value.split())
            ).mean()

            if average_words > max_text_words_threshold:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Feature '{column}' appears to contain free-form text. "
                        "High-dimensional free text is not supported here."
                    ),
                )

            cardinality = non_missing.nunique()

            if cardinality > max_categorical_cardinality:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Feature '{column}' has {cardinality} unique values; "
                        f"the supported maximum is {max_categorical_cardinality}. "
                        "Check whether this is an ID or high-cardinality column."
                    ),
                )

            categorical_columns.append(column)

    return numeric_columns, categorical_columns


# =========================================================
# Preprocessing
# =========================================================

def _build_preprocessor(
    numeric_columns: List[str],
    categorical_columns: List[str],
    scaler: str,
    missing: str,
) -> ColumnTransformer:
    transformers = []

    if numeric_columns:
        numeric_imputer = (
            "mean" if missing == "mean" else "median"
        )

        numeric_steps = [
            (
                "imputer",
                SimpleImputer(strategy=numeric_imputer),
            )
        ]

        if scaler == "standard":
            numeric_steps.append(("scaler", StandardScaler()))
        elif scaler == "minmax":
            numeric_steps.append(("scaler", MinMaxScaler()))
        elif scaler == "robust":
            numeric_steps.append(("scaler", RobustScaler()))

        numeric_pipeline = Pipeline(steps=numeric_steps)

        transformers.append(
            ("num", numeric_pipeline, numeric_columns)
        )

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent"),
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=True,
                    ),
                ),
            ]
        )

        transformers.append(
            ("cat", categorical_pipeline, categorical_columns)
        )

    if not transformers:
        raise HTTPException(
            status_code=400,
            detail="No valid preprocessing columns were found.",
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


# =========================================================
# Cross-validation fold selection
# =========================================================

def _determine_cv_folds(
    n_samples_train: int,
    min_class_count: Optional[int] = None,
    is_classification: bool = True,
) -> int:
    """Return a feasible CV fold count, or zero if CV is not feasible."""
    if n_samples_train < 2:
        return 0

    max_folds = min(10, n_samples_train)

    if is_classification and min_class_count is not None:
        max_folds = min(max_folds, min_class_count)

    return max_folds if max_folds >= 2 else 0


# =========================================================
# Main comparison service
# =========================================================

async def compare_models_service(
    file: UploadFile,
    task_type: str,
    target_column: Optional[str],
    scaler: str,
    split_ratio: float,
    missing: str,
    encoding: str,
    ranking_metric: Optional[str] = None,
    return_pipeline: bool = False,
) -> Union[
    Dict[str, Any],
    Tuple[Dict[str, Any], Pipeline],
]:
    """
    Compare supported models.

    When return_pipeline=False:
        Return the JSON-compatible comparison result.

    When return_pipeline=True:
        Return (result, fitted_winning_pipeline) for model registration.
    """

    warnings: List[str] = []
    random_state = 42

    # -----------------------------------------------------
    # Validate request parameters
    # -----------------------------------------------------

    normalized_task = (task_type or "").strip().lower()

    if normalized_task not in {"classification", "regression"}:
        raise HTTPException(
            status_code=400,
            detail="task_type must be 'classification' or 'regression'.",
        )

    if not target_column or not target_column.strip():
        raise HTTPException(
            status_code=400,
            detail="target_column is required.",
        )

    target_column = target_column.strip()

    if not 0.0 < split_ratio < 1.0:
        raise HTTPException(
            status_code=400,
            detail="split_ratio must be between 0 and 1.",
        )

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported.",
        )

    scaler = (scaler or "standard").strip().lower()
    missing = (missing or "median").strip().lower()
    encoding = (encoding or "onehot").strip().lower()

    if scaler not in {"standard", "minmax", "robust", "none"}:
        raise HTTPException(
            status_code=400,
            detail="scaler must be standard, minmax, robust, or none.",
        )

    if missing not in {"drop", "mean", "median", "most_frequent"}:
        raise HTTPException(
            status_code=400,
            detail="Unsupported missing-value strategy.",
        )

    if encoding not in {"onehot", "one-hot", "one_hot"}:
        raise HTTPException(
            status_code=400,
            detail="Only one-hot encoding is currently supported.",
        )

    # -----------------------------------------------------
    # Read CSV Safely
    # -----------------------------------------------------

    df = await safe_read_csv(file)

    if any(not column for column in df.columns):
        raise HTTPException(
            status_code=400,
            detail="CSV contains an empty column name.",
        )

    if len(set(df.columns)) != len(df.columns):
        raise HTTPException(
            status_code=400,
            detail="CSV contains duplicate column names after trimming.",
        )

    if target_column not in df.columns:
        matches = [
            column for column in df.columns
            if column.lower() == target_column.lower()
        ]

        if len(matches) == 1:
            target_column = matches[0]
        else:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Target '{target_column}' was not found. "
                    f"Available columns: {list(df.columns)}"
                ),
            )

    initial_shape = df.shape

    # -----------------------------------------------------
    # Validate target and identify features
    # -----------------------------------------------------

    if normalized_task == "classification":
        (
            df_clean,
            dropped_target_nans,
            class_counts,
        ) = _validate_target_classification(df, target_column)
    else:
        (
            df_clean,
            dropped_target_nans,
        ) = _validate_target_regression(df, target_column)

        class_counts = None

    if dropped_target_nans:
        warnings.append(
            f"Dropped {dropped_target_nans} rows with missing target values."
        )

    numeric_columns, categorical_columns = _detect_feature_columns(
        df_clean,
        target_column,
    )

    # Important: this order is also stored in the response and registry.
    feature_columns = numeric_columns + categorical_columns

    # Treat blank categorical strings as missing values.
    for column in categorical_columns:
        df_clean[column] = df_clean[column].replace(
            r"^\s*$",
            np.nan,
            regex=True,
        )

    # -----------------------------------------------------
    # Optional row removal
    # -----------------------------------------------------

    dropped_feature_nans = 0

    if missing == "drop":
        rows_before_drop = len(df_clean)

        df_clean = df_clean.dropna(
            subset=feature_columns
        ).copy()

        dropped_feature_nans = rows_before_drop - len(df_clean)

        if df_clean.empty:
            raise HTTPException(
                status_code=400,
                detail="All rows were removed due to missing feature values.",
            )

        if normalized_task == "classification":
            class_counts = df_clean[target_column].value_counts()

            if len(class_counts) < 2:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Fewer than two classes remain after dropping "
                        "rows with missing feature values."
                    ),
                )

        if dropped_feature_nans:
            warnings.append(
                f"Dropped {dropped_feature_nans} rows with missing features."
            )

    # -----------------------------------------------------
    # Separate X and y
    # -----------------------------------------------------

    X = df_clean.loc[:, feature_columns].copy()

    if normalized_task == "classification":
        y = df_clean[target_column].astype(str).copy()
        unique_classes = sorted(y.unique().tolist())
        class_counts = y.value_counts()
    else:
        y = df_clean[target_column].copy()
        unique_classes = []

    if X.empty or X.shape[1] == 0:
        raise HTTPException(
            status_code=400,
            detail="No usable feature columns remain.",
        )

    # -----------------------------------------------------
    # Select candidate models and ranking metric
    # -----------------------------------------------------

    if normalized_task == "classification":
        candidate_models = CLASSIFICATION_MODELS
        valid_metrics = CLASSIFICATION_RANKING_METRICS
        default_metric = "f1_macro"
    else:
        candidate_models = REGRESSION_MODELS
        valid_metrics = REGRESSION_RANKING_METRICS
        default_metric = "rmse"

    ranking_metric = (
        (ranking_metric or default_metric).strip().lower()
    )

    if ranking_metric not in valid_metrics:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported ranking_metric '{ranking_metric}'. "
                f"Supported metrics: {list(valid_metrics.keys())}"
            ),
        )

    metric_config = valid_metrics[ranking_metric]
    sklearn_scorer = metric_config["sklearn_scorer"]
    higher_is_better = metric_config["higher_is_better"]

    # -----------------------------------------------------
    # Train-test split
    # -----------------------------------------------------

    test_size = 1.0 - split_ratio

    if normalized_task == "classification":
        class_counts = y.value_counts()
        for c, cnt in class_counts.items():
            if cnt < 2:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Class '{c}' has only 1 sample, which prevents stratified "
                        "train/test splitting. Each class must have at least 2 samples."
                    ),
                )

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=y,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Stratified train-test split failed: {str(exc)}",
            )

        train_class_counts = y_train.value_counts()
        min_train_class = int(train_class_counts.min())

    else:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Train-test split failed: {str(exc)}",
            )

        min_train_class = None

    # -----------------------------------------------------
    # Cross-validation setup
    # -----------------------------------------------------

    cv_folds = _determine_cv_folds(
        n_samples_train=len(X_train),
        min_class_count=min_train_class,
        is_classification=(normalized_task == "classification"),
    )

    if cv_folds < 2:
        raise HTTPException(
            status_code=400,
            detail=(
                "Not enough training samples for cross-validation. "
                "Upload more data or change the train-test split."
            ),
        )

    if normalized_task == "classification":
        cv_splitter = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=random_state,
        )
    else:
        cv_splitter = KFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=random_state,
        )

    # -----------------------------------------------------
    # Cross-validate all candidate models
    # -----------------------------------------------------

    model_results: List[Dict[str, Any]] = []

    for model_name, model_factory in candidate_models.items():
        preprocessor = _build_preprocessor(
            numeric_columns,
            categorical_columns,
            scaler,
            missing,
        )

        estimator = model_factory(random_state)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("estimator", estimator),
            ]
        )

        try:
            cv_result = cross_validate(
                pipeline,
                X_train,
                y_train,
                cv=cv_splitter,
                scoring=sklearn_scorer,
                return_train_score=False,
                error_score="raise",
            )

            raw_scores = cv_result["test_score"]

            # sklearn returns negative scores for RMSE and MAE.
            if sklearn_scorer.startswith("neg_"):
                scores = -raw_scores
            else:
                scores = raw_scores

            cv_mean = _safe_float(np.mean(scores))
            cv_std = _safe_float(np.std(scores))

            model_results.append(
                {
                    "model": model_name,
                    "cv_mean": cv_mean,
                    "cv_std": cv_std,
                    "status": "success",
                    "error": None,
                }
            )

        except Exception as exc:
            model_results.append(
                {
                    "model": model_name,
                    "cv_mean": None,
                    "cv_std": None,
                    "status": "failed",
                    "error": str(exc),
                }
            )

            warnings.append(
                f"Model '{model_name}' failed during cross-validation: {exc}"
            )

    # -----------------------------------------------------
    # Rank successful models
    # -----------------------------------------------------

    successful_models = [
        item for item in model_results
        if item["status"] == "success"
        and item["cv_mean"] is not None
    ]

    if not successful_models:
        raise HTTPException(
            status_code=500,
            detail=(
                "All candidate models failed during cross-validation. "
                "No winning model could be selected."
            ),
        )

    successful_models.sort(
        key=lambda item: item["cv_mean"],
        reverse=higher_is_better,
    )

    for rank, item in enumerate(successful_models, start=1):
        item["rank"] = rank

    for item in model_results:
        if item["status"] == "failed":
            item["rank"] = None

    best_model_name = successful_models[0]["model"]
    best_cv_mean = successful_models[0]["cv_mean"]
    best_cv_std = successful_models[0]["cv_std"]

    # -----------------------------------------------------
    # Fit winning model on training split
    # -----------------------------------------------------

    best_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                _build_preprocessor(
                    numeric_columns,
                    categorical_columns,
                    scaler,
                    missing,
                ),
            ),
            (
                "estimator",
                candidate_models[best_model_name](random_state),
            ),
        ]
    )

    try:
        best_pipeline.fit(X_train, y_train)
        y_pred = best_pipeline.predict(X_test)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Best model '{best_model_name}' failed during "
                f"holdout evaluation: {exc}"
            ),
        )

    # -----------------------------------------------------
    # Calculate holdout metrics
    # -----------------------------------------------------

    if normalized_task == "classification":
        holdout_metrics = _compute_classification_holdout_metrics(
            pipeline=best_pipeline,
            X_test=X_test,
            y_test=y_test,
            y_pred=y_pred,
            unique_classes=unique_classes,
            warnings=warnings,
        )
    else:
        holdout_metrics = _compute_regression_holdout_metrics(
            y_test=y_test,
            y_pred=y_pred,
        )

    holdout_metrics["training_samples"] = int(len(X_train))

    # -----------------------------------------------------
    # Build JSON-compatible result
    # -----------------------------------------------------

    response: Dict[str, Any] = {
        "task_type": normalized_task,
        "target_column": target_column,

        # Required by the Model Registry.
        # Must match the columns expected by the fitted pipeline.
        "feature_columns": feature_columns,

        "ranking_metric": ranking_metric,
        "cv_folds": int(cv_folds),
        "split_ratio": float(split_ratio),
        "model_results": model_results,
        "best_model": {
            "model": best_model_name,
            "cv_mean": best_cv_mean,
            "cv_std": best_cv_std,
            "holdout_metrics": holdout_metrics,
        },
        "warnings": warnings,
        "data_quality": {
            "initial_shape": (
                f"{initial_shape[0]} rows × {initial_shape[1]} cols"
            ),
            "final_shape": (
                f"{df_clean.shape[0]} rows × {df_clean.shape[1]} cols"
            ),
            "training_samples": int(len(X_train)),
            "testing_samples": int(len(X_test)),
            "dropped_target_nans": int(dropped_target_nans),
            "dropped_feature_nans": int(dropped_feature_nans),
            "feature_counts": {
                "numeric": len(numeric_columns),
                "categorical": len(categorical_columns),
            },
        },
    }

    if normalized_task == "classification":
        response["classes"] = unique_classes
        response["data_quality"]["class_distribution"] = {
            str(label): int(count)
            for label, count in class_counts.items()
        }

    # -----------------------------------------------------
    # Optional refit for Model Registry
    # -----------------------------------------------------

    if return_pipeline:
        # Holdout metrics remain based on the held-out test set.
        # The final refit is for future inference.
        try:
            best_pipeline.fit(X, y)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Comparison completed, but the winning model could "
                    f"not be fitted for registration: {exc}"
                ),
            )

        return response, best_pipeline

    return response


# =========================================================
# Classification holdout metrics
# =========================================================

def _compute_classification_holdout_metrics(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    unique_classes: List[str],
    warnings: List[str],
) -> Dict[str, Any]:
    accuracy = _safe_float(
        accuracy_score(y_test, y_pred)
    )

    balanced_accuracy = _safe_float(
        balanced_accuracy_score(y_test, y_pred)
    )

    precision_macro = _safe_float(
        precision_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )

    recall_macro = _safe_float(
        recall_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )

    f1_macro = _safe_float(
        f1_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )

    f1_weighted = _safe_float(
        f1_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0,
        )
    )

    # Probability-based metrics.
    roc_auc = None
    probability_predictions = None

    if hasattr(pipeline, "predict_proba"):
        try:
            probability_predictions = pipeline.predict_proba(X_test)
        except Exception as exc:
            warnings.append(
                f"Probability prediction unavailable: {exc}"
            )

    if probability_predictions is not None:
        try:
            present_classes = set(y_test.unique())

            if len(present_classes) == len(unique_classes):
                if len(unique_classes) == 2:
                    roc_auc = _safe_float(
                        roc_auc_score(
                            y_test,
                            probability_predictions[:, 1],
                        )
                    )
                else:
                    roc_auc = _safe_float(
                        roc_auc_score(
                            y_test,
                            probability_predictions,
                            multi_class="ovr",
                            average="macro",
                            labels=unique_classes,
                        )
                    )
            else:
                warnings.append(
                    "ROC-AUC was skipped because not all target classes "
                    "were present in the test set."
                )

        except Exception as exc:
            warnings.append(
                f"ROC-AUC computation skipped: {exc}"
            )

    log_loss_value = None

    if probability_predictions is not None:
        try:
            log_loss_value = _safe_float(
                log_loss(
                    y_test,
                    probability_predictions,
                    labels=unique_classes,
                )
            )
        except Exception as exc:
            warnings.append(
                f"Log-loss computation skipped: {exc}"
            )

    matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=unique_classes,
    )

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "roc_auc": roc_auc,
        "log_loss": log_loss_value,
        "confusion_matrix": {
            "labels": unique_classes,
            "matrix": [
                [int(value) for value in row]
                for row in matrix
            ],
        },
        "testing_samples": int(len(X_test)),
    }


# =========================================================
# Regression holdout metrics
# =========================================================

def _compute_regression_holdout_metrics(
    y_test: pd.Series,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    actual_values = np.asarray(y_test)

    return {
        "rmse": _safe_float(
            np.sqrt(
                mean_squared_error(actual_values, y_pred)
            )
        ),
        "r2": _safe_float(
            r2_score(actual_values, y_pred)
        ),
        "mae": _safe_float(
            mean_absolute_error(actual_values, y_pred)
        ),
        "explained_variance": _safe_float(
            explained_variance_score(actual_values, y_pred)
        ),
        "max_error": _safe_float(
            sklearn_max_error(actual_values, y_pred)
        ),
        "testing_samples": int(len(y_test)),
    }
