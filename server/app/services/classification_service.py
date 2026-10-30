
import io
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from fastapi import HTTPException, UploadFile
from app.utils.security import safe_read_csv
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    OneHotEncoder,
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    GradientBoostingClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    confusion_matrix,
)
from app.services.ml_intelligence_service import generate_training_diagnostics


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------

def _safe_float(value: Any) -> Optional[float]:
    """Return a JSON-safe float, or None for invalid values."""
    try:
        result = float(value)
        if not math.isfinite(result):
            return None
        return round(result, 4)
    except (TypeError, ValueError, OverflowError):
        return None


def _json_safe(value: Any) -> Any:
    """Convert pandas/numpy values into JSON-compatible values."""
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, (str, int, bool)):
        return value

    return str(value)


def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and reject duplicate/empty headers."""
    df = df.copy()
    cleaned = [str(column).strip() for column in df.columns]

    if any(not name for name in cleaned):
        raise HTTPException(
            status_code=400,
            detail=(
                "CSV contains an empty column name. "
                "Rename the column and retry."
            ),
        )

    if len(set(cleaned)) != len(cleaned):
        raise HTTPException(
            status_code=400,
            detail=(
                "CSV contains duplicate column names. "
                "Make every column name unique."
            ),
        )

    df.columns = cleaned
    return df


def _normalize_strategy(
    value: Optional[str],
    allowed: set,
    default: str,
) -> str:
    strategy = (value or default).strip().lower()

    if strategy not in allowed:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported option '{strategy}'. "
                f"Allowed options: {sorted(allowed)}"
            ),
        )

    return strategy


# ---------------------------------------------------------
# Target validation
# ---------------------------------------------------------

def _validate_target(
    df: pd.DataFrame,
    target_column: str,
    max_classes: int = 20,
) -> Tuple[pd.DataFrame, int, pd.Series]:
    if target_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Target column '{target_column}' was not found in dataset columns. "
                f"Available columns: {list(df.columns)}"
            ),
        )

    original_rows = len(df)

    df = df.dropna(subset=[target_column]).copy()
    dropped = original_rows - len(df)

    if df.empty:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{target_column}' has no usable values.",
        )

    # Use strings consistently for class labels.
    df[target_column] = df[target_column].astype(str).str.strip()

    # Remove blank target labels.
    df = df[df[target_column] != ""].copy()

    if df.empty:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Target column '{target_column}' "
                "contains only empty labels."
            ),
        )

    counts = df[target_column].value_counts()

    if len(counts) < 2:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Target column '{target_column}' must have at least 2 distinct classes. "
                f"Found only {len(counts)}."
            ),
        )

    if len(counts) > max_classes:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Target column '{target_column}' has {len(counts)} distinct classes, "
                f"exceeding the initial supported limit of {max_classes}. "
                "For high-cardinality targets, review whether this is a continuous "
                "target or consider grouping rare classes."
            ),
        )

    return df, dropped, counts


# ---------------------------------------------------------
# Feature detection
# ---------------------------------------------------------

def _detect_and_validate_columns(
    df: pd.DataFrame,
    target_column: str,
    max_categorical_cardinality: int = 100,
) -> Tuple[List[str], List[str]]:
    feature_columns = [
        column for column in df.columns
        if column != target_column
    ]

    if not feature_columns:
        raise HTTPException(
            status_code=400,
            detail=(
                "No input features remain after "
                "excluding the target column."
            ),
        )

    numeric_columns: List[str] = []
    categorical_columns: List[str] = []

    for column in feature_columns:
        series = df[column]

        if series.isna().all():
            raise HTTPException(
                status_code=400,
                detail=f"Feature '{column}' contains only missing values.",
            )

        # Preserve genuinely numeric columns.
        if pd.api.types.is_numeric_dtype(series):
            numeric_columns.append(column)
            continue

        # Object/string columns are treated as categorical.
        non_missing = series.dropna().astype(str).str.strip()
        non_missing = non_missing[non_missing != ""]

        if non_missing.empty:
            raise HTTPException(
                status_code=400,
                detail=f"Feature '{column}' contains no usable values.",
            )

        # Check for free-form text (average words > 3.0)
        avg_word_count = non_missing.apply(lambda s: len(str(s).split())).mean()
        if avg_word_count > 3.0:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Column '{column}' appears to contain free-form text (average {avg_word_count:.1f} words per row). "
                    "High-dimensional free text is not supported in the tabular classification engine."
                ),
            )

        # Avoid creating very large one-hot feature matrices.
        cardinality = non_missing.nunique()

        if cardinality > max_categorical_cardinality:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Feature '{column}' has {cardinality} distinct values. "
                    f"The supported limit is {max_categorical_cardinality}. "
                    "Check for ID columns or high-cardinality text."
                ),
            )

        categorical_columns.append(column)

    if not numeric_columns and not categorical_columns:
        raise HTTPException(
            status_code=400,
            detail=(
                "No usable numeric or categorical features were detected."
            ),
        )

    return numeric_columns, categorical_columns


# ---------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------

def _build_preprocessor(
    numeric_columns: List[str],
    categorical_columns: List[str],
    scaler: str,
    missing: str,
) -> ColumnTransformer:
    """
    Construct a preprocessing transformer.

    Numeric:
        Mean/median imputation and optional scaling.

    Categorical:
        Most-frequent imputation and one-hot encoding.

    The transformer is fitted within the sklearn Pipeline.
    """

    transformers = []

    if numeric_columns:
        numeric_imputer = (
            "mean" if missing == "mean" else "median"
        )

        numeric_steps = [
            ("imputer", SimpleImputer(strategy=numeric_imputer)),
        ]

        if scaler == "standard":
            numeric_steps.append(
                ("scaler", StandardScaler())
            )
        elif scaler == "minmax":
            numeric_steps.append(
                ("scaler", MinMaxScaler())
            )
        elif scaler == "robust":
            numeric_steps.append(
                ("scaler", RobustScaler())
            )
        # "none" intentionally skips scaling.

        transformers.append(
            (
                "numeric",
                Pipeline(numeric_steps),
                numeric_columns,
            )
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
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            )
        )

    if not transformers:
        raise HTTPException(
            status_code=400,
            detail=(
                "Preprocessing produced no transformers. "
                "Check the selected target and input feature columns."
            ),
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )


# ---------------------------------------------------------
# Model selection
# ---------------------------------------------------------

def _get_classifier(model_name: str):
    name = (model_name or "").strip().lower().replace(" ", "")

    if name in {
        "logisticregression",
        "logistic_regression",
        "logistic",
        "lr",
    }:
        return (
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=42,
            ),
            "LogisticRegression",
        )

    if name in {
        "decisiontree",
        "decisiontreeclassifier",
        "decision_tree",
        "dt",
    }:
        return (
            DecisionTreeClassifier(
                class_weight="balanced",
                random_state=42,
            ),
            "DecisionTreeClassifier",
        )

    if name in {
        "randomforest",
        "randomforestclassifier",
        "random_forest",
        "rf",
    }:
        return (
            RandomForestClassifier(
                n_estimators=100,
                class_weight="balanced",
                n_jobs=1,
                random_state=42,
            ),
            "RandomForestClassifier",
        )

    if name in {
        "extratrees",
        "extratreesclassifier",
        "extra_trees",
        "et",
    }:
        return (
            ExtraTreesClassifier(
                n_estimators=100,
                class_weight="balanced",
                n_jobs=1,
                random_state=42,
            ),
            "ExtraTreesClassifier",
        )

    if name in {
        "histgradientboosting",
        "histgradientboostingclassifier",
        "hist_gradient_boosting",
        "hgb",
    }:
        return (
            HistGradientBoostingClassifier(
                random_state=42,
            ),
            "HistGradientBoostingClassifier",
        )

    if name in {
        "gradientboosting",
        "gradientboostingclassifier",
        "gradient_boosting",
        "gb",
        "gbm",
    }:
        return (
            GradientBoostingClassifier(
                random_state=42,
            ),
            "GradientBoostingClassifier",
        )

    if name in {
        "kneighbors",
        "kneighborsclassifier",
        "knn",
        "k_neighbors",
    }:
        return (
            KNeighborsClassifier(
                n_neighbors=5,
            ),
            "KNeighborsClassifier",
        )

    if name in {
        "svc",
        "svm",
        "supportvectormachine",
        "supportvectorclassifier",
    }:
        return (
            SVC(
                probability=True,
                class_weight="balanced",
                random_state=42,
            ),
            "SVC",
        )

    raise HTTPException(
        status_code=400,
        detail=(
            f"Unsupported classification model '{model_name}'. "
            "Choose LogisticRegression, DecisionTreeClassifier, "
            "RandomForestClassifier, ExtraTreesClassifier, "
            "HistGradientBoostingClassifier, GradientBoostingClassifier, "
            "KNeighborsClassifier, or SVC."
        ),
    )


# ---------------------------------------------------------
# Main classification service
# ---------------------------------------------------------

async def train_classification_service(
    file: UploadFile,
    model: str,
    scaler: str,
    split_ratio: float,
    missing: str,
    encoding: str,
    target_column: Optional[str],
    return_pipeline: bool = False,
) -> Union[Dict[str, Any], Tuple[Dict[str, Any], Pipeline]]:
    """
    Train and evaluate a classification model.

    By default, returns the usual JSON-compatible result dictionary.

    When return_pipeline=True, returns:
        (result_dictionary, fitted_pipeline)

    The pipeline is refitted on all usable data only after the
    held-out evaluation metrics have been calculated.
    """

    warnings: List[str] = []

    if not target_column or not target_column.strip():
        raise HTTPException(
            status_code=400,
            detail=(
                "Select a target_column before training "
                "a classification model."
            ),
        )

    target_column = target_column.strip()

    if not 0.0 < split_ratio < 1.0:
        raise HTTPException(
            status_code=400,
            detail="The training splitRatio must be between 0 and 1.",
        )

    scaler = _normalize_strategy(
        scaler,
        {"standard", "minmax", "robust", "none"},
        "standard",
    )

    missing = _normalize_strategy(
        missing,
        {"drop", "mean", "median", "most_frequent"},
        "median",
    )

    encoding = _normalize_strategy(
        encoding,
        {"onehot", "one-hot", "one_hot"},
        "onehot",
    )

    # This service currently uses one-hot encoding.
    encoding = "onehot"

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid CSV file.",
        )

    # -----------------------------------------------------
    # Read CSV Safely
    # -----------------------------------------------------

    df = await safe_read_csv(file)
    df = _clean_column_names(df)

    initial_rows, initial_columns = df.shape

    # -----------------------------------------------------
    # Validate target
    # -----------------------------------------------------

    if target_column not in df.columns:
        matches = [
            column
            for column in df.columns
            if column.strip().lower() == target_column.lower()
        ]

        if len(matches) == 1:
            target_column = matches[0]
        else:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Target column '{target_column}' was not found in dataset columns. "
                    f"Available columns: {list(df.columns)}"
                ),
            )

    df, dropped_target_rows, _ = _validate_target(
        df,
        target_column,
    )

    if dropped_target_rows:
        warnings.append(
            f"Removed {dropped_target_rows} rows with missing target values."
        )

    # -----------------------------------------------------
    # Detect feature types
    # -----------------------------------------------------

    numeric_columns, categorical_columns = (
        _detect_and_validate_columns(
            df,
            target_column,
        )
    )

    # Preserve the exact input feature ordering for the registry.
    feature_columns = [
        column for column in df.columns
        if column != target_column
    ]

    # Normalize blank strings to missing values.
    for column in feature_columns:
        if (
            pd.api.types.is_object_dtype(df[column])
            or pd.api.types.is_string_dtype(df[column])
        ):
            df[column] = df[column].replace(
                r"^\s*$",
                np.nan,
                regex=True,
            )

    # -----------------------------------------------------
    # Missing-value handling
    # -----------------------------------------------------

    dropped_feature_rows = 0

    if missing == "drop":
        before = len(df)

        df = df.dropna(
            subset=feature_columns
        ).copy()

        dropped_feature_rows = before - len(df)

        if df.empty:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No rows remain after dropping missing feature values."
                ),
            )

        if dropped_feature_rows:
            warnings.append(
                f"Removed {dropped_feature_rows} rows "
                "with missing feature values."
            )

    # Recalculate target distribution after row removal.
    class_counts = df[target_column].value_counts()

    if len(class_counts) < 2:
        raise HTTPException(
            status_code=400,
            detail=(
                "Fewer than two target classes remain after preprocessing. "
                "Choose a different missing-value strategy "
                "or inspect the data."
            ),
        )

    # -----------------------------------------------------
    # Separate X and y
    # -----------------------------------------------------

    X = df.loc[:, feature_columns].copy()
    y = df[target_column].astype(str).copy()

    if X.shape[1] == 0:
        raise HTTPException(
            status_code=400,
            detail=(
                "No feature columns remain after removing the target."
            ),
        )

    if X.shape[0] != y.shape[0]:
        raise HTTPException(
            status_code=400,
            detail="Feature and target row counts do not match.",
        )

    # -----------------------------------------------------
    # Validate stratified split
    # -----------------------------------------------------

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

    test_size = 1.0 - split_ratio
    number_of_classes = y.nunique()

    test_rows = math.ceil(len(y) * test_size)
    train_rows = len(y) - test_rows

    if (
        test_rows < number_of_classes
        or train_rows < number_of_classes
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                f"The selected split creates {train_rows} training rows "
                f"and {test_rows} test rows for "
                f"{number_of_classes} classes. "
                "Increase the dataset size or adjust the training split."
            ),
        )

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
            stratify=y,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                "Could not create a stratified train/test split: "
                f"{str(exc)}"
            ),
        )

    # -----------------------------------------------------
    # Build and train evaluation pipeline
    # -----------------------------------------------------

    preprocessor = _build_preprocessor(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        scaler=scaler,
        missing=missing,
    )

    classifier, resolved_model = _get_classifier(model)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    # Fit only on training data for honest holdout evaluation.
    try:
        pipeline.fit(X_train, y_train)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Model training failed. Check feature values, "
                "missing-value handling and model settings. "
                f"Details: {str(exc)}"
            ),
        )

    # -----------------------------------------------------
    # Cross-validation on training folds only (Leak-Free)
    # -----------------------------------------------------
    cv_metrics: Dict[str, Any] = {}
    train_class_counts = y_train.value_counts()
    min_train_class_cnt = int(train_class_counts.min()) if not train_class_counts.empty else 0
    max_feasible_folds = min(5, min_train_class_cnt)

    if max_feasible_folds >= 2:
        try:
            skf = StratifiedKFold(n_splits=max_feasible_folds, shuffle=True, random_state=42)
            fold_accs = []
            fold_bal_accs = []
            fold_f1s = []

            for fold_train_idx, fold_val_idx in skf.split(X_train, y_train):
                cv_pipe = clone(pipeline)
                X_tr = X_train.iloc[fold_train_idx]
                y_tr = y_train.iloc[fold_train_idx]
                X_va = X_train.iloc[fold_val_idx]
                y_va = y_train.iloc[fold_val_idx]

                cv_pipe.fit(X_tr, y_tr)
                val_preds = cv_pipe.predict(X_va)

                fold_accs.append(accuracy_score(y_va, val_preds))
                fold_bal_accs.append(balanced_accuracy_score(y_va, val_preds))
                fold_f1s.append(f1_score(y_va, val_preds, average="macro", zero_division=0))

            cv_metrics = {
                "cv_folds": max_feasible_folds,
                "accuracy_mean": _safe_float(np.mean(fold_accs)),
                "accuracy_std": _safe_float(np.std(fold_accs)),
                "balanced_accuracy_mean": _safe_float(np.mean(fold_bal_accs)),
                "balanced_accuracy_std": _safe_float(np.std(fold_bal_accs)),
                "f1_macro_mean": _safe_float(np.mean(fold_f1s)),
                "f1_macro_std": _safe_float(np.std(fold_f1s)),
            }
        except Exception as cv_exc:
            warnings.append(f"Training cross-validation skipped: {str(cv_exc)}")

    # -----------------------------------------------------
    # Training metrics (to assess overfitting gap)
    # -----------------------------------------------------
    training_metrics: Dict[str, Any] = {}
    try:
        train_preds = pipeline.predict(X_train)
        training_metrics = {
            "accuracy": _safe_float(accuracy_score(y_train, train_preds)),
            "balanced_accuracy": _safe_float(balanced_accuracy_score(y_train, train_preds)),
            "f1_macro": _safe_float(f1_score(y_train, train_preds, average="macro", zero_division=0)),
            "f1_weighted": _safe_float(f1_score(y_train, train_preds, average="weighted", zero_division=0)),
        }
    except Exception:
        training_metrics = {}

    try:
        predictions = pipeline.predict(X_test)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        )

    # -----------------------------------------------------
    # Classification labels and probabilities
    # -----------------------------------------------------

    fitted_classifier = pipeline.named_steps["classifier"]

    labels = [
        str(label)
        for label in fitted_classifier.classes_
    ]

    y_prob = None

    if hasattr(fitted_classifier, "predict_proba"):
        try:
            y_prob = pipeline.predict_proba(X_test)

        except Exception as exc:
            warnings.append(
                f"Probability metrics unavailable: {str(exc)}"
            )

    # -----------------------------------------------------
    # Calculate classification metrics
    # -----------------------------------------------------

    metrics: Dict[str, Any] = {
        "accuracy": _safe_float(
            accuracy_score(y_test, predictions)
        ),
        "balanced_accuracy": _safe_float(
            balanced_accuracy_score(y_test, predictions)
        ),
        "precision_macro": _safe_float(
            precision_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "precision_weighted": _safe_float(
            precision_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),
        "recall_macro": _safe_float(
            recall_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "recall_weighted": _safe_float(
            recall_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),
        "f1_macro": _safe_float(
            f1_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "f1_weighted": _safe_float(
            f1_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),
        "roc_auc": None,
        "log_loss": None,
        "training_samples": int(len(X_train)),
        "testing_samples": int(len(X_test)),
        "feature_importances": {},
    }

    # -----------------------------------------------------
    # ROC-AUC and log loss
    # -----------------------------------------------------

    if y_prob is not None:
        try:
            if len(labels) == 2:
                positive_label = labels[1]

                metrics["roc_auc"] = _safe_float(
                    roc_auc_score(
                        (y_test == positive_label).astype(int),
                        y_prob[:, 1],
                    )
                )

            elif set(y_test.unique()) == set(labels):
                metrics["roc_auc"] = _safe_float(
                    roc_auc_score(
                        y_test,
                        y_prob,
                        labels=labels,
                        multi_class="ovr",
                        average="macro",
                    )
                )

            else:
                warnings.append(
                    "ROC-AUC omitted because the test set "
                    "does not contain every class."
                )

        except Exception as exc:
            warnings.append(
                f"ROC-AUC could not be computed: {str(exc)}"
            )

        try:
            metrics["log_loss"] = _safe_float(
                log_loss(
                    y_test,
                    y_prob,
                    labels=labels,
                )
            )

        except Exception as exc:
            warnings.append(
                f"Log loss could not be computed: {str(exc)}"
            )

    # -----------------------------------------------------
    # Confusion matrix
    # -----------------------------------------------------

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=labels,
    )

    confusion_data = {
        "labels": labels,
        "matrix": [
            [int(value) for value in row]
            for row in cm
        ],
    }

    # -----------------------------------------------------
    # Feature importance / coefficient magnitude
    # -----------------------------------------------------

    try:
        transformed_names = (
            pipeline.named_steps[
                "preprocessor"
            ].get_feature_names_out()
        )

        if hasattr(fitted_classifier, "feature_importances_"):
            values = fitted_classifier.feature_importances_

        elif hasattr(fitted_classifier, "coef_"):
            coefficients = np.asarray(
                fitted_classifier.coef_
            )

            values = np.mean(
                np.abs(coefficients),
                axis=0,
            )

        else:
            values = []

        metrics["feature_importances"] = {
            str(name): round(float(value), 6)
            for name, value in zip(transformed_names, values)
            if math.isfinite(float(value))
        }

    except Exception:
        metrics["feature_importances"] = {}

    if len(df) < 50:
        warnings.append(
            "This is a small dataset. Test metrics may vary considerably "
            "with a different train/test split."
        )

    # -----------------------------------------------------
    # Prepare sample rows for the frontend
    # -----------------------------------------------------

    sample_rows = [
        {
            str(key): _json_safe(value)
            for key, value in row.items()
        }
        for row in df.head(5).to_dict(orient="records")
    ]

    # -----------------------------------------------------
    # Class imbalance evaluation & Diagnostics
    # -----------------------------------------------------
    class_series = df[target_column].value_counts()
    majority_count = int(class_series.iloc[0]) if len(class_series) > 0 else 1
    minority_count = int(class_series.iloc[-1]) if len(class_series) > 0 else 1
    imbalance_ratio = round(majority_count / max(1, minority_count), 2)
    if imbalance_ratio >= 10.0:
        imbalance_severity = "severe"
    elif imbalance_ratio >= 3.0:
        imbalance_severity = "moderate"
    else:
        imbalance_severity = "balanced"

    imbalance_info = {
        "majority_count": majority_count,
        "minority_count": minority_count,
        "imbalance_ratio": imbalance_ratio,
        "imbalance_severity": imbalance_severity,
        "applied_strategy": "balanced_class_weight",
        "description": f"Class ratio is {imbalance_ratio}:1 ({imbalance_severity}). Cost-sensitive balanced weights applied to penalize minority misclassifications.",
    }

    holdout_acc = metrics.get("accuracy") or 0.0
    train_acc = training_metrics.get("accuracy") or 0.0
    overfitting_gap = round(float(train_acc - holdout_acc), 4)

    diagnostics = generate_training_diagnostics(
        task_type="classification",
        train_score=train_acc,
        val_score=cv_metrics.get("accuracy_mean") if cv_metrics else holdout_acc,
        primary_metric="accuracy" if imbalance_severity == "balanced" else "f1_macro",
        imbalance_info=imbalance_info,
        n_samples=len(df),
        n_features=len(feature_columns),
    )

    # -----------------------------------------------------
    # Construct JSON-compatible response
    # -----------------------------------------------------

    result: Dict[str, Any] = {
        "task_type": "classification",
        "target_column": target_column,
        "classes": labels,
        "model_type": resolved_model,
        "scaler_type": scaler,
        "missing_strategy": missing,
        "encoding_strategy": encoding,
        "split_ratio": split_ratio,
        "metrics": metrics,
        "training_metrics": training_metrics,
        "cv_metrics": cv_metrics,
        "overfitting_gap": overfitting_gap,
        "imbalance_info": imbalance_info,
        "diagnostics": diagnostics,
        "confusion_matrix": confusion_data,
        "warnings": warnings,
        "columns": feature_columns,
        "rows": sample_rows,
        "raw_rows": sample_rows,
        "data_quality": {
            "initial_shape": (
                f"{initial_rows} rows x {initial_columns} columns"
            ),
            "final_shape": (
                f"{len(df)} rows x {len(df.columns)} columns"
            ),
            "dropped_target_nans": int(dropped_target_rows),
            "dropped_feature_nans": int(dropped_feature_rows),
            "class_distribution": {
                str(label): int(count)
                for label, count
                in df[target_column].value_counts().items()
            },
            "feature_counts": {
                "numeric": len(numeric_columns),
                "categorical": len(categorical_columns),
            },
        },
    }

    # -----------------------------------------------------
    # Optional final fit for Model Registry
    # -----------------------------------------------------

    if return_pipeline:
        # Evaluation metrics above are calculated on the held-out
        # test split. Now train a final pipeline on all usable rows.
        try:
            pipeline.fit(X, y)

        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Evaluation completed, but the final model "
                    "could not be fitted for registration. "
                    f"Details: {str(exc)}"
                ),
            )

        # Return the fitted pipeline separately from the JSON result.
        # Never put this sklearn object inside the API response.
        return result, pipeline

    # Default behavior remains compatible with existing callers.
    return result
