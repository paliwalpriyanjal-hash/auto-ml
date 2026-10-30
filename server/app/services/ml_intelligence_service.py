# server/app/services/ml_intelligence_service.py
"""
AutoVeyra Machine Learning Intelligence Engine.

Provides:
- Intelligent dataset pre-training analysis (task inference, class imbalance, leakage, outliers, dtypes).
- Leak-free preprocessing pipeline generation using ColumnTransformer and sklearn Pipelines.
- Dataset-adaptive model suites and selection (HistGradientBoosting, RandomForest, Ridge, ElasticNet, etc.).
- Robust evaluation strategy: Training-only Stratified/K-Fold cross-validation + untouched holdout split.
- Generalization gap monitoring (overfitting/underfitting detection).
- Dataset-informed failure diagnostics and actionable recommendations.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from fastapi import HTTPException
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    OneHotEncoder,
    OrdinalEncoder,
    FunctionTransformer,
)
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    KFold,
    cross_validate,
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
    median_absolute_error,
    explained_variance_score,
    max_error as sklearn_max_error,
    mean_absolute_percentage_error,
)

# Classification Algorithms
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

# Regression Algorithms
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    GradientBoostingRegressor,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR


def safe_float(val: Any, digits: int = 4) -> Optional[float]:
    """Convert value to a finite, JSON-compliant float."""
    if val is None:
        return None
    try:
        num = float(val)
        if not math.isfinite(num):
            return None
        return round(num, digits)
    except (TypeError, ValueError, OverflowError):
        return None


# =====================================================================
# 1. INTELLIGENT DATASET ANALYSIS BEFORE TRAINING
# =====================================================================

def analyze_dataset_intelligence(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    task_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Performs comprehensive statistical profiling of the dataset prior to training.
    Detects dtypes, target candidate, task type, imbalance, leakage, outliers, and collinearity.
    """
    total_rows, total_cols = df.shape
    if total_rows == 0 or total_cols == 0:
        raise HTTPException(status_code=400, detail="Uploaded dataset is empty.")

    # 1. Clean and normalize column names
    col_names = [str(c).strip() for c in df.columns]
    df = df.copy()
    df.columns = col_names

    # Check duplicate rows and duplicate columns
    dup_rows = int(df.duplicated().sum())
    dup_cols_count = len(col_names) - len(set(col_names))

    # 2. Target Column Resolution
    detected_target = target_column
    if not detected_target or detected_target not in df.columns:
        # Heuristic search for standard target names
        candidates = ["target", "label", "y", "churn", "outcome", "status", "price", "class", "survived"]
        found = False
        for cand in candidates:
            for col in col_names:
                if col.lower() == cand:
                    detected_target = col
                    found = True
                    break
            if found:
                break
        if not detected_target:
            # Default to last column
            detected_target = col_names[-1]

    target_series = df[detected_target] if detected_target in df.columns else None

    # 3. Determine Task Type
    detected_task = task_type
    if not detected_task:
        if target_series is not None:
            non_null_target = target_series.dropna()
            n_unique = non_null_target.nunique()
            is_numeric = pd.api.types.is_numeric_dtype(non_null_target)
            
            if not is_numeric or n_unique == 2:
                detected_task = "classification"
            elif n_unique <= 15 and (n_unique / max(1, len(non_null_target))) < 0.05:
                # Small distinct integers are often multiclass targets
                detected_task = "classification"
            else:
                detected_task = "regression"
        else:
            detected_task = "classification"

    # 4. Detailed Column Profiling
    column_profiles: List[Dict[str, Any]] = []
    num_cols: List[str] = []
    cat_cols: List[str] = []
    datetime_cols: List[str] = []
    constant_cols: List[str] = []
    id_cols: List[str] = []
    quality_alerts: List[Dict[str, str]] = []

    target_numeric_vals = None
    if target_series is not None and pd.api.types.is_numeric_dtype(target_series):
        target_numeric_vals = pd.to_numeric(target_series, errors="coerce")

    for col in col_names:
        series = df[col]
        missing_count = int(series.isna().sum())
        missing_pct = safe_float((missing_count / total_rows) * 100.0, 2)
        non_null = series.dropna()
        unique_count = int(non_null.nunique())
        cardinality_ratio = safe_float(unique_count / max(1, len(non_null)), 4) if len(non_null) > 0 else 0.0

        is_constant = unique_count <= 1
        if is_constant:
            constant_cols.append(col)
            quality_alerts.append({
                "column": col,
                "type": "constant_feature",
                "message": f"Column '{col}' has {unique_count} distinct value(s) and provides zero variance."
            })

        # Check for possible ID column
        is_id = False
        lower_name = col.lower()
        if col != detected_target and unique_count > 10 and cardinality_ratio >= 0.98:
            if any(term in lower_name for term in ["id", "uuid", "guid", "key", "index", "code", "hash"]):
                is_id = True
                id_cols.append(col)
                quality_alerts.append({
                    "column": col,
                    "type": "potential_id",
                    "message": f"Column '{col}' has high uniqueness ({cardinality_ratio * 100:.1f}%) and matches identifier naming patterns."
                })

        # Inferred type
        col_type = "numeric"
        skew = None
        outliers_count = 0
        mean_val = None
        std_val = None
        min_val = None
        max_val = None
        leakage_suspect = False

        if pd.api.types.is_numeric_dtype(series):
            col_type = "numeric"
            num_cols.append(col)
            if len(non_null) > 0:
                mean_val = safe_float(non_null.mean())
                std_val = safe_float(non_null.std())
                min_val = safe_float(non_null.min())
                max_val = safe_float(non_null.max())
                skew = safe_float(non_null.skew())

                # Outlier detection via IQR
                q25 = non_null.quantile(0.25)
                q75 = non_null.quantile(0.75)
                iqr = q75 - q25
                if iqr > 0:
                    outliers_count = int(((non_null < (q25 - 1.5 * iqr)) | (non_null > (q75 + 1.5 * iqr))).sum())

                # Target leakage check
                if col != detected_target and target_numeric_vals is not None and len(non_null) > 5:
                    try:
                        valid_mask = series.notna() & target_numeric_vals.notna()
                        if valid_mask.sum() > 5:
                            corr = abs(float(series[valid_mask].corr(target_numeric_vals[valid_mask])))
                            if corr > 0.98:
                                leakage_suspect = True
                                quality_alerts.append({
                                    "column": col,
                                    "type": "target_leakage",
                                    "message": f"Column '{col}' has extreme correlation ({corr:.4f}) with the target. Check for target leakage."
                                })
                    except Exception:
                        pass
        else:
            # Check if date/time
            is_date = False
            if "date" in lower_name or "time" in lower_name or "timestamp" in lower_name:
                try:
                    sample_dt = pd.to_datetime(non_null.head(20), errors="coerce")
                    if sample_dt.notna().sum() > 10:
                        is_date = True
                except Exception:
                    pass

            if is_date:
                col_type = "datetime"
                datetime_cols.append(col)
            else:
                col_type = "categorical"
                cat_cols.append(col)
                if unique_count > 50 and not is_id:
                    quality_alerts.append({
                        "column": col,
                        "type": "high_cardinality",
                        "message": f"Categorical column '{col}' has {unique_count} distinct categories. Rare category pooling recommended."
                    })

        column_profiles.append({
            "name": col,
            "inferred_type": col_type,
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "unique_count": unique_count,
            "cardinality_ratio": cardinality_ratio,
            "is_constant": is_constant,
            "is_id": is_id,
            "leakage_suspect": leakage_suspect,
            "mean": mean_val,
            "std": std_val,
            "min": min_val,
            "max": max_val,
            "skewness": skew,
            "outliers_count": outliers_count,
        })

    # 5. Target Analysis & Class Imbalance
    target_profile: Dict[str, Any] = {}
    if target_series is not None:
        usable_target = target_series.dropna()
        n_unique_target = usable_target.nunique()
        if detected_task == "classification":
            counts = usable_target.astype(str).value_counts().to_dict()
            min_c = min(counts.values()) if counts else 1
            max_c = max(counts.values()) if counts else 1
            ratio = safe_float(max_c / max(1, min_c), 2) or 1.0

            if ratio >= 10.0:
                severity = "severe"
            elif ratio >= 4.0:
                severity = "moderate"
            elif ratio >= 2.0:
                severity = "mild"
            else:
                severity = "none"

            target_profile = {
                "target_column": detected_target,
                "task_type": "classification",
                "classes_count": n_unique_target,
                "class_distribution": counts,
                "imbalance_ratio": ratio,
                "imbalance_severity": severity,
                "is_imbalanced": severity in ["moderate", "severe"],
            }
            if severity in ["moderate", "severe"]:
                quality_alerts.append({
                    "column": detected_target,
                    "type": "class_imbalance",
                    "message": f"Target has {severity} class imbalance (ratio {ratio}:1). Macro F1 or Balanced Accuracy recommended over standard accuracy."
                })
        else:
            target_profile = {
                "target_column": detected_target,
                "task_type": "regression",
                "mean": safe_float(pd.to_numeric(usable_target, errors="coerce").mean()),
                "std": safe_float(pd.to_numeric(usable_target, errors="coerce").std()),
                "min": safe_float(pd.to_numeric(usable_target, errors="coerce").min()),
                "max": safe_float(pd.to_numeric(usable_target, errors="coerce").max()),
            }

    # 6. Collinearity check among numeric columns
    collinear_pairs: List[Dict[str, Any]] = []
    if len(num_cols) >= 2:
        try:
            num_sub = df[num_cols].dropna()
            if len(num_sub) > 10:
                corr_matrix = num_sub.corr().abs()
                for i in range(len(num_cols)):
                    for j in range(i + 1, len(num_cols)):
                        val = corr_matrix.iloc[i, j]
                        if val > 0.88:
                            c1, c2 = num_cols[i], num_cols[j]
                            collinear_pairs.append({
                                "feature_1": c1,
                                "feature_2": c2,
                                "correlation": safe_float(val, 4),
                            })
                            quality_alerts.append({
                                "column": f"{c1} & {c2}",
                                "type": "collinearity",
                                "message": f"Features '{c1}' and '{c2}' are strongly collinear (r = {val:.2f}). Regularization (Ridge/ElasticNet) or tree models recommended."
                            })
        except Exception:
            pass

    # 7. Preprocessing & Model Recommendations
    recommendations = {
        "recommended_task": detected_task,
        "recommended_target": detected_target,
        "recommended_scaler": "robust" if any(p["outliers_count"] > 10 for p in column_profiles) else "standard",
        "recommended_missing": "median",
        "recommended_encoding": "onehot",
        "recommended_metric": (
            "f1_macro" if (detected_task == "classification" and target_profile.get("is_imbalanced"))
            else ("accuracy" if detected_task == "classification" else "rmse")
        ),
        "recommended_models": (
            ["HistGradientBoostingClassifier", "RandomForestClassifier", "LogisticRegression", "ExtraTreesClassifier"]
            if detected_task == "classification"
            else ["HistGradientBoostingRegressor", "RandomForestRegressor", "Ridge", "LinearRegression"]
        ),
        "excluded_columns": constant_cols + id_cols,
    }

    return {
        "dataset_summary": {
            "total_rows": total_rows,
            "total_columns": total_cols,
            "duplicate_rows": dup_rows,
            "duplicate_columns": dup_cols_count,
            "numeric_count": len(num_cols),
            "categorical_count": len(cat_cols),
            "datetime_count": len(datetime_cols),
            "constant_count": len(constant_cols),
            "id_count": len(id_cols),
        },
        "target_profile": target_profile,
        "target_analysis": {
            "inferred_task": (
                "binary_classification"
                if target_profile.get("classes_count") == 2
                else ("multiclass_classification" if (target_profile.get("classes_count") or 0) > 2 else target_profile.get("task_type"))
            ),
            "inferred_target": target_profile.get("target_column"),
            "imbalance_severity": target_profile.get("imbalance_severity", "balanced"),
            "imbalance_ratio": target_profile.get("imbalance_ratio", 1.0),
            "is_imbalanced": target_profile.get("is_imbalanced", False),
            "classes_count": target_profile.get("classes_count", 0),
        },
        "dimensions": {
            "rows": total_rows,
            "columns": total_cols,
        },
        "column_profiles": column_profiles,
        "quality_alerts": quality_alerts,
        "collinear_pairs": collinear_pairs,
        "collinear_features": [
            {"feature_a": p["feature_1"], "feature_b": p["feature_2"], "correlation": p["correlation"]}
            for p in collinear_pairs
        ],
        "recommendations": recommendations,
    }


# =====================================================================
# 2. ADVANCED DATASET-AWARE PREPROCESSING ENGINE
# =====================================================================

def build_dataset_aware_preprocessor(
    numeric_columns: List[str],
    categorical_columns: List[str],
    scaler: str = "standard",
    missing: str = "median",
    skewed_columns: Optional[List[str]] = None,
) -> ColumnTransformer:
    """
    Constructs an sklearn ColumnTransformer fitted strictly inside CV folds.
    Handles numeric imputation (mean/median), scaling (Standard/Robust/MinMax/None),
    and categorical imputation + OneHotEncoding with unknown-category safety.
    """
    transformers = []

    if numeric_columns:
        num_impute_strategy = "mean" if missing == "mean" else "median"
        num_steps = [("imputer", SimpleImputer(strategy=num_impute_strategy))]

        norm_scaler = scaler.strip().lower()
        if norm_scaler == "standard":
            num_steps.append(("scaler", StandardScaler()))
        elif norm_scaler == "robust":
            num_steps.append(("scaler", RobustScaler()))
        elif norm_scaler == "minmax":
            num_steps.append(("scaler", MinMaxScaler()))
        # "none" leaves raw imputed numbers

        transformers.append((
            "numeric",
            Pipeline(steps=num_steps),
            numeric_columns
        ))

    if categorical_columns:
        cat_steps = [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]
        transformers.append((
            "categorical",
            Pipeline(steps=cat_steps),
            categorical_columns
        ))

    if not transformers:
        raise HTTPException(
            status_code=400,
            detail="No valid numeric or categorical features available for preprocessing."
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


# =====================================================================
# 3. EXPANDED MODEL SUITES (CLASSIFICATION & REGRESSION)
# =====================================================================

def get_adaptive_classification_models(
    n_samples: int,
    n_features: int,
    is_imbalanced: bool = False,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Returns candidate classification models adapted to dataset size and imbalance.
    """
    cw = "balanced" if is_imbalanced else None

    models: Dict[str, Any] = {
        "LogisticRegression": LogisticRegression(
            class_weight=cw,
            max_iter=1500,
            random_state=random_state,
        ),
        "DecisionTreeClassifier": DecisionTreeClassifier(
            class_weight=cw,
            max_depth=12 if n_samples > 200 else 6,
            random_state=random_state,
        ),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=100,
            class_weight=cw,
            n_jobs=1,
            random_state=random_state,
        ),
        "ExtraTreesClassifier": ExtraTreesClassifier(
            n_estimators=100,
            class_weight=cw,
            n_jobs=1,
            random_state=random_state,
        ),
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier(
            max_iter=100,
            class_weight=cw,
            random_state=random_state,
        ),
        "GradientBoostingClassifier": GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            random_state=random_state,
        ),
    }

    # Only include distance-based algorithms if dataset is small/medium to avoid freezing CPU
    if n_samples <= 5000:
        models["KNeighborsClassifier"] = KNeighborsClassifier(n_neighbors=5, n_jobs=1)
        models["SVC"] = SVC(class_weight=cw, probability=True, random_state=random_state)

    return models


def get_adaptive_regression_models(
    n_samples: int,
    n_features: int,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Returns candidate regression models adapted to dataset scale and complexity.
    """
    models: Dict[str, Any] = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=random_state),
        "Lasso": Lasso(alpha=0.1, random_state=random_state),
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=random_state),
        "DecisionTree": DecisionTreeRegressor(
            max_depth=12 if n_samples > 200 else 6,
            random_state=random_state,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=100,
            n_jobs=1,
            random_state=random_state,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=100,
            n_jobs=1,
            random_state=random_state,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=100,
            random_state=random_state,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            random_state=random_state,
        ),
    }

    if n_samples <= 5000:
        models["KNeighborsRegressor"] = KNeighborsRegressor(n_neighbors=5, n_jobs=1)
        models["SVR"] = SVR(kernel="rbf")

    return models


# =====================================================================
# 4. ROBUST CROSS-VALIDATION & HOLDOUT EVALUATION
# =====================================================================

def evaluate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    classes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Calculates comprehensive classification metrics."""
    metrics = {
        "accuracy": safe_float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": safe_float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": safe_float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": safe_float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_macro": safe_float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_weighted": safe_float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro": safe_float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": safe_float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "roc_auc": None,
        "log_loss": None,
    }

    if y_prob is not None and len(np.unique(y_true)) >= 2:
        try:
            if len(np.unique(y_true)) == 2:
                # Binary ROC-AUC
                p_pos = y_prob[:, 1] if y_prob.shape[1] > 1 else y_prob[:, 0]
                metrics["roc_auc"] = safe_float(roc_auc_score(y_true, p_pos))
            else:
                # Multiclass ROC-AUC (OvR)
                metrics["roc_auc"] = safe_float(roc_auc_score(y_true, y_prob, multi_class="ovr"))
        except Exception:
            metrics["roc_auc"] = None

        try:
            metrics["log_loss"] = safe_float(log_loss(y_true, y_prob))
        except Exception:
            metrics["log_loss"] = None

    return metrics


def evaluate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    """Calculates comprehensive regression metrics."""
    n = len(y_true)
    if n == 0:
        return {}

    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse) if mse >= 0 else 0.0

    metrics = {
        "r2": safe_float(r2_score(y_true, y_pred)),
        "rmse": safe_float(rmse),
        "mae": safe_float(mean_absolute_error(y_true, y_pred)),
        "median_absolute_error": safe_float(median_absolute_error(y_true, y_pred)),
        "explained_variance": safe_float(explained_variance_score(y_true, y_pred)),
        "max_error": safe_float(sklearn_max_error(y_true, y_pred)),
        "mape": None,
    }

    # Only compute MAPE if no true values are zero
    if not np.any(np.isclose(y_true, 0.0)):
        try:
            metrics["mape"] = safe_float(mean_absolute_percentage_error(y_true, y_pred))
        except Exception:
            pass

    return metrics


# =====================================================================
# 5. GENERALIZATION & FAILURE DIAGNOSTICS
# =====================================================================

def generate_training_diagnostics(
    task_type: str = "classification",
    training_score: Optional[float] = None,
    cv_score: Optional[float] = None,
    holdout_score: Optional[float] = None,
    dataset_summary: Optional[Dict[str, Any]] = None,
    target_profile: Optional[Dict[str, Any]] = None,
    primary_metric: str = "score",
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Evaluates overfitting gap, class imbalance impact, and model generalization.
    Provides plain-language diagnostic findings and actionable recommendations.
    Accepts flexible keyword arguments (training_score/train_score, cv_score/val_score).
    """
    if training_score is None:
        training_score = kwargs.get("train_score")
    if cv_score is None:
        cv_score = kwargs.get("val_score") or kwargs.get("cv_score")
    if holdout_score is None:
        holdout_score = kwargs.get("holdout_score")

    dataset_summary = dataset_summary or {}
    target_profile = target_profile or {}

    imbalance_info = kwargs.get("imbalance_info") or {}
    if not target_profile and imbalance_info:
        target_profile = {
            "is_imbalanced": imbalance_info.get("imbalance_severity") in {"severe", "moderate"},
            "imbalance_severity": imbalance_info.get("imbalance_severity", "moderate"),
            "imbalance_ratio": imbalance_info.get("imbalance_ratio", 1.0),
        }

    n_samples = kwargs.get("n_samples") or dataset_summary.get("total_rows", 0)

    findings: List[str] = []
    recommendations: List[str] = []
    generalization_status = "healthy"
    overfitting_gap = None

    if training_score is not None and cv_score is not None:
        overfitting_gap = safe_float(training_score - cv_score, 4)

        # Detect severe overfitting (training score is significantly higher than CV)
        if overfitting_gap > 0.18:
            generalization_status = "overfitting"
            findings.append(
                f"Significant overfitting detected: Training {primary_metric} ({training_score:.3f}) exceeds Validation ({cv_score:.3f}) by {overfitting_gap * 100:.1f}%."
            )
            recommendations.append(
                "To reduce overfitting: increase model regularization, constrain tree depth, prune uninformative features, or gather additional training records."
            )
        elif overfitting_gap < -0.05:
            # Underfitting or CV instability
            generalization_status = "unstable_cv"
            findings.append("Cross-validation scores show high fold variance or severe sample sensitivity.")

    # Underfitting check
    if cv_score is not None and ((task_type == "classification" and cv_score < 0.55) or (task_type == "regression" and cv_score < 0.10)):
        findings.append(f"Model exhibits potential underfitting: Validation {primary_metric} ({cv_score:.3f}) indicates weak predictive signal.")
        recommendations.append("Consider engineering polynomial or interaction features, trying gradient boosting, or evaluating alternative input variables.")

    # Imbalance diagnostic
    if task_type == "classification" and target_profile.get("is_imbalanced"):
        findings.append(f"Target has {target_profile.get('imbalance_severity', 'moderate')} class imbalance (ratio {target_profile.get('imbalance_ratio', 1.0)}:1). Standard accuracy may be misleading.")
        recommendations.append("Focus on Macro F1 and Balanced Accuracy when making model deployment decisions.")

    # Small dataset diagnostic
    if n_samples > 0 and n_samples < 100:
        findings.append(f"Small dataset size ({n_samples} samples). Holdout evaluations may exhibit higher statistical noise.")
        recommendations.append("Use cross-validation metrics as the primary source of truth rather than single holdout splits.")

    return {
        "status": generalization_status,
        "overfitting_gap": overfitting_gap,
        "findings": findings,
        "recommendations": recommendations,
        "summary": (
            "Model generalizes well with balanced training and validation metrics."
            if generalization_status == "healthy"
            else "Generalization caveats detected; review diagnostics before production deployment."
        ),
    }
