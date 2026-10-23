import io
import math
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from fastapi import UploadFile, HTTPException

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    confusion_matrix
)


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


def _detect_and_validate_columns(
    df: pd.DataFrame,
    target_column: str,
    max_categorical_cardinality: int = 50,
    max_text_words_threshold: float = 3.0
):
    """
    Validates feature columns and identifies numeric vs categorical features.
    Rejects unsupported free text and excessive feature cardinality with clear diagnostics.
    """
    feature_cols = [c for c in df.columns if c != target_column]
    if not feature_cols:
        raise HTTPException(
            status_code=400,
            detail=f"No feature columns found. Dataset only contains target column '{target_column}'."
        )

    numeric_cols: List[str] = []
    categorical_cols: List[str] = []

    for col in feature_cols:
        series = df[col]
        # Check if numeric
        if pd.api.types.is_numeric_dtype(series):
            # Check if all values are NaN
            if series.dropna().empty:
                raise HTTPException(
                    status_code=400,
                    detail=f"Feature column '{col}' contains only empty/NaN values."
                )
            numeric_cols.append(col)
        else:
            # Treat as categorical / string
            non_na = series.dropna().astype(str)
            if non_na.empty:
                raise HTTPException(
                    status_code=400,
                    detail=f"Feature column '{col}' contains only empty/NaN values."
                )

            # Check for unsupported free-text: average word count > threshold
            avg_word_count = non_na.apply(lambda s: len(s.split())).mean()
            if avg_word_count > max_text_words_threshold:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Column '{col}' appears to contain free-form text (average {avg_word_count:.1f} words per row). "
                        "High-dimensional free text is not supported in the tabular classification engine."
                    )
                )

            # Check cardinality
            nunique = non_na.nunique()
            if nunique > max_categorical_cardinality:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Column '{col}' has excessive cardinality ({nunique} distinct values > limit {max_categorical_cardinality}). "
                        "Categorical features with excessive distinct levels (e.g. IDs, hashes, timestamps) are not supported."
                    )
                )

            categorical_cols.append(col)

    return numeric_cols, categorical_cols


def _validate_target(
    df: pd.DataFrame,
    target_column: str,
    max_classes: int = 20,
    min_classes: int = 2
):
    """
    Validates target column for single-label tabular classification:
    - Target existence
    - Drops rows where target is NaN
    - Validates class counts (2-20)
    - Rejects likely continuous float targets
    """
    if target_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{target_column}' was not found in dataset columns: {list(df.columns)}"
        )

    # Count initial rows and missing target rows
    initial_rows = len(df)
    df_clean = df.dropna(subset=[target_column]).copy()
    dropped_target_nans = initial_rows - len(df_clean)

    if df_clean.empty:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{target_column}' contains only missing/NaN values."
        )

    target_series = df_clean[target_column]

    # Detect continuous float target
    if pd.api.types.is_float_dtype(target_series):
        nunique = target_series.nunique()
        # If float has many distinct values, it is almost certainly a regression target
        if nunique > max_classes or (nunique > 10 and nunique / len(df_clean) > 0.1):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Target column '{target_column}' has continuous floating-point values ({nunique} distinct values). "
                    "This looks like a continuous regression target. Please use regression mode or discretize the target."
                )
            )

    # Class distribution analysis
    value_counts = target_series.value_counts(dropna=True)
    num_classes = len(value_counts)

    if num_classes < min_classes:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Target column '{target_column}' contains only {num_classes} unique class: {list(value_counts.index)}. "
                "Classification requires at least 2 distinct classes."
            )
        )

    if num_classes > max_classes:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Target column '{target_column}' contains {num_classes} classes, exceeding the initial supported limit of {max_classes}. "
                "Please check if this column contains unique identifiers or continuous measurements."
            )
        )

    return df_clean, dropped_target_nans, value_counts


def _build_preprocessor(
    numeric_cols: List[str],
    categorical_cols: List[str],
    scaler: str,
    missing: str
) -> ColumnTransformer:
    """
    Constructs a ColumnTransformer with training-fold-only transformers:
    - Numeric: SimpleImputer (median or mean) + Scaler (Standard, MinMax, Robust, None)
    - Categorical: SimpleImputer (most_frequent) + OneHotEncoder(sparse_output=True, handle_unknown="ignore")
    """
    transformers = []

    if numeric_cols:
        # Determine imputation strategy
        num_impute_strategy = "mean" if missing == "mean" else "median"
        num_steps = [
            ("imputer", SimpleImputer(strategy=num_impute_strategy))
        ]

        if scaler == "standard":
            num_steps.append(("scaler", StandardScaler()))
        elif scaler == "minmax":
            num_steps.append(("scaler", MinMaxScaler()))
        elif scaler == "robust":
            num_steps.append(("scaler", RobustScaler()))
        # scaler == "none" or other: passthrough without scaling

        num_pipe = Pipeline(num_steps)
        transformers.append(("num", num_pipe, numeric_cols))

    if categorical_cols:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
        ])
        transformers.append(("cat", cat_pipe, categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def _get_classifier(model_name: str, random_state: int = 42):
    """
    Resolves supported classification models.
    Supports canonical names and safe legacy aliases.
    """
    clean_name = model_name.strip().lower()
    
    # Logistic Regression
    if clean_name in ["logisticregression", "logistic_regression", "logistic", "lr"]:
        return LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state
        ), "LogisticRegression"

    # Decision Tree
    if clean_name in ["decisiontree", "decisiontreeclassifier", "decision_tree", "dt"]:
        return DecisionTreeClassifier(
            class_weight="balanced",
            random_state=random_state
        ), "DecisionTreeClassifier"

    # Random Forest
    if clean_name in ["randomforest", "randomforestclassifier", "random_forest", "rf"]:
        return RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state
        ), "RandomForestClassifier"

    # Extra Trees
    if clean_name in ["extratrees", "extratreesclassifier", "extra_trees", "et"]:
        return ExtraTreesClassifier(
            n_estimators=100,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state
        ), "ExtraTreesClassifier"

    supported = [
        "LogisticRegression",
        "DecisionTreeClassifier",
        "RandomForestClassifier",
        "ExtraTreesClassifier"
    ]
    raise HTTPException(
        status_code=400,
        detail=f"Unsupported classification model '{model_name}'. Available models: {supported}"
    )


async def train_classification_service(
    file: UploadFile,
    model: str,
    scaler: str,
    split_ratio: float,
    missing: str,
    encoding: str,
    target_column: Optional[str]
) -> Dict[str, Any]:
    """
    Executes a leakage-safe supervised classification training and evaluation pipeline:
    1. Reads and decodes uploaded CSV
    2. Validates target column and drops missing target rows
    3. Validates features (rejects free text and high cardinality columns)
    4. Performs stratified holdout train-test split (with feasibility check)
    5. Fits ColumnTransformer and Classifier strictly on X_train
    6. Predicts on X_test holdout (with unseen category tolerance)
    7. Computes robust multi-metric classification report and structured confusion matrix
    8. Formats JSON-compliant response
    """
    warnings: List[str] = []

    # 1. Parameter Validation
    if not target_column or not target_column.strip():
        raise HTTPException(
            status_code=400,
            detail="The 'target_column' parameter is required for classification tasks."
        )
    target_column = target_column.strip()

    if split_ratio <= 0.0 or split_ratio >= 1.0:
        raise HTTPException(
            status_code=400,
            detail=f"splitRatio must be strictly between 0 and 1, got {split_ratio}."
        )

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="The CSV file appears to be empty or corrupt.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV file: {str(e)}")

    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded file contains no rows.")

    initial_shape = df.shape

    # 2. Target Column Validation & NaN Target Row Removal
    df_clean, dropped_target_nans, class_counts = _validate_target(df, target_column)
    if dropped_target_nans > 0:
        warnings.append(
            f"Dropped {dropped_target_nans} rows where target '{target_column}' was missing/NaN."
        )

    # 3. Feature Validation & Classification
    numeric_cols, categorical_cols = _detect_and_validate_columns(df_clean, target_column)

    # Handle explicit row drop missing strategy if requested
    dropped_feature_nans = 0
    if missing == "drop":
        before_drop = len(df_clean)
        df_clean = df_clean.dropna().copy()
        dropped_feature_nans = before_drop - len(df_clean)
        if df_clean.empty:
            raise HTTPException(
                status_code=400,
                detail="All rows were removed after dropping rows with missing feature values."
            )
        # Re-check class distribution after row drops
        class_counts = df_clean[target_column].value_counts()
        if len(class_counts) < 2:
            raise HTTPException(
                status_code=400,
                detail=f"After dropping missing rows, target '{target_column}' has fewer than 2 distinct classes."
            )

    # Separate X and y
    X = df_clean[numeric_cols + categorical_cols]
    y_raw = df_clean[target_column]

    # Convert y labels to string or int, and get sorted class labels
    # Preserving natural representation for labels
    unique_classes = sorted(list(class_counts.index.map(str)))
    y = y_raw.astype(str)

    # 4. Stratified Split Feasibility & Execution
    min_class_count = int(class_counts.min())
    test_size = 1.0 - split_ratio

    if min_class_count < 2:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Cannot perform stratified evaluation: class '{class_counts.idxmin()}' has only {min_class_count} sample(s). "
                "Each class must have at least 2 samples to allow train and test representation."
            )
        )

    # Check if any class has too few samples to guarantee holdout presence
    min_expected_test = min_class_count * test_size
    if min_expected_test < 1.0:
        warnings.append(
            f"Class '{class_counts.idxmin()}' has only {min_class_count} total samples. "
            "Evaluation metrics on this small class may have high variance."
        )

    # Check dataset size
    if len(df_clean) < 20:
        warnings.append(
            f"Dataset is very small ({len(df_clean)} rows). "
            "Holdout test metrics may be noisy and should be interpreted with caution."
        )

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
            stratify=y
        )
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Stratified train-test split failed: {str(e)}"
        )

    # 5. Build Preprocessor & Estimator Pipeline
    preprocessor = _build_preprocessor(numeric_cols, categorical_cols, scaler, missing)
    clf, resolved_model_name = _get_classifier(model, random_state=42)

    full_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])

    # Fit exclusively on training partition
    try:
        full_pipeline.fit(X_train, y_train)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model training failed during pipeline fitting: {str(e)}"
        )

    # 6. Predict on Holdout Partition
    try:
        y_pred = full_pipeline.predict(X_test)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Holdout prediction failed: {str(e)}"
        )

    # Predict probabilities if estimator supports it
    y_prob = None
    has_predict_proba = hasattr(full_pipeline.named_steps["classifier"], "predict_proba")
    if has_predict_proba:
        try:
            y_prob = full_pipeline.predict_proba(X_test)
        except Exception:
            y_prob = None

    # 7. Compute Metrics
    # Ensure all labels in test set are accounted for
    test_classes = unique_classes

    acc = _safe_float(accuracy_score(y_test, y_pred))
    bal_acc = _safe_float(balanced_accuracy_score(y_test, y_pred))
    prec_macro = _safe_float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    prec_weighted = _safe_float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    rec_macro = _safe_float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    rec_weighted = _safe_float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1_macro = _safe_float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    f1_weighted = _safe_float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    # ROC-AUC calculation
    roc_auc = None
    roc_auc_warning = None
    if y_prob is not None:
        try:
            # Check if all classes are present in y_test
            present_classes_test = set(y_test.unique())
            if len(present_classes_test) == len(test_classes):
                if len(test_classes) == 2:
                    # Binary classification
                    pos_idx = 1
                    roc_auc = _safe_float(roc_auc_score(y_test, y_prob[:, pos_idx]))
                else:
                    # Multiclass classification (One-vs-Rest)
                    roc_auc = _safe_float(roc_auc_score(
                        y_test,
                        y_prob,
                        multi_class="ovr",
                        average="macro",
                        labels=test_classes
                    ))
            else:
                roc_auc_warning = "ROC-AUC could not be computed: not all training classes were present in the test holdout."
        except Exception as e:
            roc_auc_warning = f"ROC-AUC computation skipped: {str(e)}"
    else:
        roc_auc_warning = "Estimator does not provide probability estimates (predict_proba)."

    if roc_auc_warning:
        warnings.append(roc_auc_warning)

    # Log Loss calculation
    logloss_val = None
    logloss_warning = None
    if y_prob is not None:
        try:
            logloss_val = _safe_float(log_loss(y_test, y_prob, labels=test_classes))
        except Exception as e:
            logloss_warning = f"Log loss computation skipped: {str(e)}"
            warnings.append(logloss_warning)

    # Structured Confusion Matrix
    cm_matrix = confusion_matrix(y_test, y_pred, labels=test_classes)
    confusion_matrix_data = {
        "labels": test_classes,
        "matrix": [[int(cell) for cell in row] for row in cm_matrix]
    }

    # Extract feature importances if available
    feature_importances: Dict[str, float] = {}
    classifier_step = full_pipeline.named_steps["classifier"]
    try:
        feature_names_out = full_pipeline.named_steps["preprocessor"].get_feature_names_out()
        if hasattr(classifier_step, "feature_importances_"):
            raw_imp = classifier_step.feature_importances_
            feature_importances = {
                str(k): round(float(v), 4)
                for k, v in zip(feature_names_out, raw_imp)
            }
        elif hasattr(classifier_step, "coef_"):
            # For linear/logistic models
            coef = classifier_step.coef_
            if coef.ndim == 2:
                # Average absolute coefficient across classes
                mean_abs_coef = np.mean(np.abs(coef), axis=0)
            else:
                mean_abs_coef = np.abs(coef)
            feature_importances = {
                str(k): round(float(v), 4)
                for k, v in zip(feature_names_out, mean_abs_coef)
            }
    except Exception:
        # Graceful fallback if feature_names_out is unavailable
        feature_importances = {}

    # Sample preview rows
    sample_preview_rows = df_clean.head(5).to_dict(orient="records")
    # Clean rows for JSON serializability
    safe_sample_rows = [
        {k: (None if (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) else v) for k, v in row.items()}
        for row in sample_preview_rows
    ]

    # Metrics dictionary
    metrics = {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision_macro": prec_macro,
        "precision_weighted": prec_weighted,
        "recall_macro": rec_macro,
        "recall_weighted": rec_weighted,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "roc_auc": roc_auc,
        "log_loss": logloss_val,
        "training_samples": int(len(X_train)),
        "testing_samples": int(len(X_test)),
        "feature_importances": feature_importances
    }

    return {
        "task_type": "classification",
        "target_column": target_column,
        "classes": test_classes,
        "model_type": resolved_model_name,
        "scaler_type": scaler,
        "missing_strategy": missing,
        "encoding_strategy": encoding,
        "split_ratio": split_ratio,
        "metrics": metrics,
        "confusion_matrix": confusion_matrix_data,
        "warnings": warnings,
        "columns": [c for c in df_clean.columns if c != target_column],
        "rows": safe_sample_rows,
        "raw_rows": safe_sample_rows,
        "data_quality": {
            "initial_shape": f"{initial_shape[0]} rows × {initial_shape[1]} cols",
            "final_shape": f"{df_clean.shape[0]} rows × {df_clean.shape[1]} cols",
            "dropped_target_nans": dropped_target_nans,
            "dropped_feature_nans": dropped_feature_nans,
            "class_distribution": {str(k): int(v) for k, v in class_counts.items()},
            "feature_counts": {
                "numeric": len(numeric_cols),
                "categorical": len(categorical_cols)
            }
        }
    }
