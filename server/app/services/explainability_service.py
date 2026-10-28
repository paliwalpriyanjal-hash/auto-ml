# app/services/explainability_service.py

import json
import math
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from fastapi import HTTPException
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from app.services.model_registry_service import load_model, _json_safe

# Optional SHAP import with clear capability flag
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    shap = None
    SHAP_AVAILABLE = False


# Supported candidate models mapping
TREE_ESTIMATOR_NAMES = {
    "randomforestclassifier",
    "decisiontreeclassifier",
    "extratreesclassifier",
    "randomforestregressor",
    "decisiontreeregressor",
    "randomforest",
    "decisiontree",
}

LINEAR_ESTIMATOR_NAMES = {
    "logisticregression",
    "linearregression",
}

CLASSIFICATION_SCORERS = {
    "f1_macro": "f1_macro",
    "accuracy": "accuracy",
    "roc_auc": "roc_auc",
    "balanced_accuracy": "balanced_accuracy",
}

REGRESSION_SCORERS = {
    "r2": "r2",
    "rmse": "neg_root_mean_squared_error",
    "neg_root_mean_squared_error": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "neg_mean_absolute_error": "neg_mean_absolute_error",
}


def _extract_preprocessor_and_estimator(pipeline: Any) -> Tuple[Optional[Any], Any]:
    """
    Extract the preprocessor step and final estimator from an sklearn Pipeline.
    Never modifies or refits the pipeline.
    """
    if isinstance(pipeline, Pipeline):
        if "preprocessor" in pipeline.named_steps:
            return pipeline.named_steps["preprocessor"], pipeline.named_steps["estimator"]
        elif len(pipeline.steps) > 1:
            preproc = Pipeline(pipeline.steps[:-1])
            est = pipeline.steps[-1][1]
            return preproc, est
        elif len(pipeline.steps) == 1:
            return None, pipeline.steps[0][1]
    return None, pipeline


def _map_transformed_to_original(transformed_name: str, expected_columns: List[str]) -> str:
    """
    Map an internal transformer column name (e.g. 'cat__city_NY' or 'num__age')
    back to the original dataset column name ('city' or 'age').
    """
    cleaned = transformed_name
    for prefix in ["num__", "cat__", "remainder__"]:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
            break

    if cleaned in expected_columns:
        return cleaned

    for col in expected_columns:
        if cleaned == col or cleaned.startswith(f"{col}_"):
            return col

    return cleaned


def get_explainability_capabilities(model_id: int) -> dict:
    """
    Return supported explainability methods, task type, estimator type,
    feature metadata, and known limitations for a registered model.
    """
    pipeline, metadata = load_model(model_id)
    feature_columns = json.loads(metadata.get("feature_columns_json") or "[]")
    target_column = metadata.get("target_column")
    task_type = metadata.get("task_type", "classification")
    model_type = metadata.get("model_type", "")
    normalized_model = model_type.lower().replace(" ", "").replace("_", "")

    _, estimator = _extract_preprocessor_and_estimator(pipeline)
    estimator_class = estimator.__class__.__name__

    # Check SHAP compatibility
    is_tree = normalized_model in TREE_ESTIMATOR_NAMES or any(
        t in estimator_class.lower() for t in ["tree", "forest"]
    )
    is_linear = normalized_model in LINEAR_ESTIMATOR_NAMES or any(
        t in estimator_class.lower() for t in ["linear", "logistic"]
    )

    shap_supported = False
    explainer_type = None
    shap_limitations = None

    if SHAP_AVAILABLE:
        if is_tree:
            shap_supported = True
            explainer_type = "TreeExplainer"
            shap_limitations = "Exact Shapley values via TreeSHAP. High dimensional one-hot features are aggregated back to original columns."
        elif is_linear:
            shap_supported = True
            explainer_type = "LinearExplainer"
            shap_limitations = "Linear model Shapley values using background evaluation distribution."
        else:
            shap_supported = False
            shap_limitations = f"Estimator '{estimator_class}' is not directly supported by TreeExplainer or LinearExplainer. Use Permutation Importance instead."
    else:
        shap_supported = False
        shap_limitations = "The SHAP library is not installed in the environment."

    scoring_options = (
        list(CLASSIFICATION_SCORERS.keys())
        if task_type == "classification"
        else list(REGRESSION_SCORERS.keys())
    )

    return {
        "model_id": model_id,
        "model_name": metadata.get("model_name"),
        "task_type": task_type,
        "model_type": model_type,
        "target_column": target_column,
        "feature_columns": feature_columns,
        "expected_columns_count": len(feature_columns),
        "supported_methods": [
            {
                "method": "permutation",
                "name": "Permutation Feature Importance",
                "scope": "global",
                "supported": True,
                "description": "Model-agnostic evaluation measuring metric drop when each feature is shuffled.",
                "requires_target": True,
                "strengths": "Completely model-agnostic; evaluates features directly in original column space before one-hot expansion.",
                "limitations": "Requires ground-truth target; correlated features may share or obscure individual importance."
            },
            {
                "method": "shap",
                "name": "SHAP (SHapley Additive exPlanations)",
                "scope": "local_and_global",
                "supported": shap_supported,
                "explainer_type": explainer_type,
                "requires_target": False,
                "description": "Game-theoretic feature attributions explaining individual predictions and mean magnitude.",
                "strengths": "Additive local explanations satisfying efficiency and symmetry axioms; supports waterfall visualization.",
                "limitations": shap_limitations
            }
        ],
        "default_scoring": "f1_macro" if task_type == "classification" else "r2",
        "available_scorers": scoring_options,
    }


def compute_permutation_importance(
    *,
    model_id: int,
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    n_repeats: int = 5,
    scoring: Optional[str] = None,
    random_state: int = 42,
    dataset_source: str = "evaluation_data",
) -> dict:
    """
    Compute permutation feature importance for a registered model.
    Evaluates on original input feature columns using the fitted pipeline.
    Never mutates or refits the registered model.
    """
    t0 = time.time()

    # 1. Load pipeline and metadata
    pipeline, metadata = load_model(model_id)
    expected_columns = json.loads(metadata.get("feature_columns_json") or "[]")
    resolved_target = target_column or metadata.get("target_column")
    task_type = metadata.get("task_type", "classification")

    if not expected_columns:
        raise HTTPException(
            status_code=400,
            detail="Registered model metadata has no record of expected feature columns.",
        )

    # 2. Validate input DataFrame
    if df is None or df.empty:
        raise HTTPException(
            status_code=400,
            detail="Evaluation dataset is empty. Provide at least one valid row.",
        )

    if len(df) < 5:
        raise HTTPException(
            status_code=400,
            detail=f"Evaluation dataset has only {len(df)} rows. Provide at least 5 rows for reliable permutation evaluation.",
        )

    if len(df) > 5000:
        raise HTTPException(
            status_code=400,
            detail=f"Evaluation dataset has {len(df)} rows, exceeding the safe limit of 5,000 rows. Please downsample for interactive explainability.",
        )

    # 3. Check target presence for supervised permutation scoring
    if not resolved_target or resolved_target not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Permutation importance requires ground-truth target values to measure performance degradation. "
                f"Target column '{resolved_target}' was not found in the evaluation dataset. "
                f"Available columns: {list(df.columns)}"
            ),
        )

    # 4. Validate feature columns strictly against expected schema
    missing_columns = [col for col in expected_columns if col not in df.columns]
    extra_columns = [
        col for col in df.columns
        if col not in expected_columns and col != resolved_target
    ]

    if missing_columns:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Evaluation dataset is missing required features for this model.",
                "missing_columns": missing_columns,
                "expected_columns": expected_columns,
            },
        )

    if extra_columns:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Evaluation dataset contains extra unexpected features.",
                "extra_columns": extra_columns,
                "expected_columns": expected_columns,
            },
        )

    # 5. Extract X and y
    clean_df = df.dropna(subset=[resolved_target]).copy()
    if clean_df.empty:
        raise HTTPException(
            status_code=400,
            detail=f"All rows have missing values in target column '{resolved_target}'.",
        )

    X = clean_df.loc[:, expected_columns].copy()
    y = clean_df[resolved_target].copy()

    # Normalize target for classification to match fitted estimator class type
    if task_type == "classification":
        _, estimator = _extract_preprocessor_and_estimator(pipeline)
        if hasattr(estimator, "classes_") and len(estimator.classes_) > 0:
            sample_class = estimator.classes_[0]
            try:
                if isinstance(sample_class, (int, np.integer)):
                    y = y.astype(int)
                elif isinstance(sample_class, (float, np.floating)):
                    y = y.astype(float)
                else:
                    y = y.astype(str).str.strip()
            except Exception:
                y = y.astype(str).str.strip()
        else:
            y = y.astype(str).str.strip()

        if y.nunique() < 2:
            raise HTTPException(
                status_code=400,
                detail="Evaluation target must contain at least 2 distinct classes to compute classification permutation importance.",
            )

    # 6. Validate repeats and scoring
    bounded_repeats = max(1, min(int(n_repeats), 10))

    if task_type == "classification":
        chosen_scorer = scoring or "f1_macro"
        sklearn_scorer = CLASSIFICATION_SCORERS.get(chosen_scorer.lower(), "f1_macro")
    else:
        chosen_scorer = scoring or "r2"
        sklearn_scorer = REGRESSION_SCORERS.get(chosen_scorer.lower(), "r2")

    # 7. Execute permutation importance
    try:
        perm_res = permutation_importance(
            pipeline,
            X,
            y,
            n_repeats=bounded_repeats,
            random_state=random_state,
            scoring=sklearn_scorer,
            n_jobs=1,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Permutation importance calculation failed: {str(exc)}",
        )

    # 8. Build structured results
    importances_mean = perm_res.importances_mean
    importances_std = perm_res.importances_std
    raw_importances = perm_res.importances

    feature_results = []
    has_negative_importance = False

    for idx, feature_name in enumerate(expected_columns):
        mean_val = float(importances_mean[idx])
        std_val = float(importances_std[idx])
        repeats_vals = [float(v) for v in raw_importances[idx]]

        if mean_val < -1e-6:
            has_negative_importance = True

        feature_results.append({
            "feature": feature_name,
            "importance_mean": _json_safe(round(mean_val, 6)),
            "importance_std": _json_safe(round(std_val, 6)),
            "raw_repeats": [_json_safe(round(v, 6)) for v in repeats_vals],
        })

    # Sort descending by importance_mean
    feature_results.sort(key=lambda x: x["importance_mean"], reverse=True)

    # Assign ranks
    for rank, item in enumerate(feature_results, start=1):
        item["rank"] = rank

    duration = round(time.time() - t0, 3)

    # 9. Formulate warnings and educational context
    warnings = []
    if dataset_source == "training_data":
        warnings.append(
            "Permutation importance computed on training data reflects model fit rather than unbiased generalization, and may be overly optimistic."
        )
    else:
        warnings.append(
            "Evaluated on external holdout dataset. Permutation importance reflects impact on out-of-sample generalization."
        )

    if has_negative_importance:
        warnings.append(
            "Some features exhibit negative importance. Shuffling these features slightly improved the validation score, indicating they may represent noise or uninformative variance."
        )

    warnings.append(
        "Notice: Permutation importance describes model attribution, not real-world causality. Correlated features can share or obscure individual importance."
    )

    return {
        "model_id": model_id,
        "model_name": metadata.get("model_name"),
        "task_type": task_type,
        "model_type": metadata.get("model_type"),
        "method": "permutation",
        "dataset_source": dataset_source,
        "rows_evaluated": len(clean_df),
        "target_column": resolved_target,
        "scoring_metric": chosen_scorer,
        "n_repeats": bounded_repeats,
        "duration_seconds": duration,
        "features": feature_results,
        "top_features": [f["feature"] for f in feature_results[:5]],
        "warnings": warnings,
    }


def compute_shap_explanations(
    *,
    model_id: int,
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    row_indices: Optional[List[int]] = None,
    max_rows: int = 50,
    selected_class: Optional[Union[str, int]] = None,
    dataset_source: str = "evaluation_data",
) -> dict:
    """
    Compute SHAP local feature attributions and global mean |SHAP| values.
    Supports TreeExplainer for tree ensembles and LinearExplainer for linear models.
    Maps transformed one-hot features back to original feature names.
    Never mutates or refits the registered model.
    """
    if not SHAP_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="SHAP is not available in the backend environment. Please use Permutation Importance.",
        )

    t0 = time.time()

    # 1. Load model and metadata
    pipeline, metadata = load_model(model_id)
    expected_columns = json.loads(metadata.get("feature_columns_json") or "[]")
    task_type = metadata.get("task_type", "classification")
    model_type = metadata.get("model_type", "")
    resolved_target = target_column or metadata.get("target_column")

    # 2. Validate input DataFrame
    if df is None or df.empty:
        raise HTTPException(
            status_code=400,
            detail="Evaluation dataset is empty. Provide at least one valid row.",
        )

    missing_columns = [col for col in expected_columns if col not in df.columns]
    extra_columns = [
        col for col in df.columns
        if col not in expected_columns and col != resolved_target
    ]

    if missing_columns:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Evaluation dataset is missing required features for this model.",
                "missing_columns": missing_columns,
                "expected_columns": expected_columns,
            },
        )

    if extra_columns:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Evaluation dataset contains extra unexpected features.",
                "extra_columns": extra_columns,
                "expected_columns": expected_columns,
            },
        )

    # 3. Extract estimator and preprocessor
    preprocessor, estimator = _extract_preprocessor_and_estimator(pipeline)
    estimator_class = estimator.__class__.__name__
    normalized_model = model_type.lower().replace(" ", "").replace("_", "")

    is_tree = normalized_model in TREE_ESTIMATOR_NAMES or any(
        t in estimator_class.lower() for t in ["tree", "forest"]
    )
    is_linear = normalized_model in LINEAR_ESTIMATOR_NAMES or any(
        t in estimator_class.lower() for t in ["linear", "logistic"]
    )

    if not is_tree and not is_linear:
        raise HTTPException(
            status_code=400,
            detail=(
                f"SHAP explanation is not currently supported for estimator '{estimator_class}'. "
                "Supported estimators include Decision Trees, Random Forests, Extra Trees, Linear Regression, and Logistic Regression. "
                "Please use Permutation Feature Importance for this model."
            ),
        )

    # 4. Limit row count for interactive responsiveness
    bound_rows = min(len(df), max(1, min(int(max_rows), 100)))
    eval_df = df.iloc[:bound_rows].copy()

    X_orig = eval_df.loc[:, expected_columns].copy()

    # Transform through pipeline preprocessor
    if preprocessor is not None:
        try:
            X_trans = preprocessor.transform(X_orig)
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Preprocessing evaluation data failed: {str(exc)}",
            )

        if hasattr(X_trans, "toarray"):
            X_trans = X_trans.toarray()
        else:
            X_trans = np.asarray(X_trans)

        try:
            transformed_names = list(preprocessor.get_feature_names_out())
        except Exception:
            transformed_names = [f"trans_feat_{i}" for i in range(X_trans.shape[1])]
    else:
        X_trans = np.asarray(X_orig.values)
        transformed_names = list(expected_columns)

    # Map transformed names to original columns
    name_mapping = [
        _map_transformed_to_original(t_name, expected_columns)
        for t_name in transformed_names
    ]

    # 5. Initialize explainer
    try:
        if is_tree:
            explainer = shap.TreeExplainer(estimator)
            shap_output = explainer(X_trans)
        elif is_linear:
            # LinearExplainer uses the transformed background
            explainer = shap.LinearExplainer(estimator, X_trans)
            shap_output = explainer(X_trans)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"SHAP explainer initialization failed: {str(exc)}. Please use Permutation Importance.",
        )

    raw_values = shap_output.values
    base_values = getattr(shap_output, "base_values", 0.0)

    # 6. Handle task dimensions (Classification classes vs Regression)
    classes_list = []
    class_idx = 0

    if task_type == "classification":
        if hasattr(estimator, "classes_"):
            classes_list = [str(c) for c in estimator.classes_]
        else:
            classes_list = ["Class 0", "Class 1"]

        # Resolve selected class
        if selected_class is not None:
            if isinstance(selected_class, int) and 0 <= selected_class < len(classes_list):
                class_idx = selected_class
            elif str(selected_class) in classes_list:
                class_idx = classes_list.index(str(selected_class))

        # Check if 3D array (samples, features, classes)
        if raw_values.ndim == 3:
            # Shape: (n_samples, n_features, n_classes)
            if class_idx >= raw_values.shape[2]:
                class_idx = raw_values.shape[2] - 1
            active_shap_matrix = raw_values[:, :, class_idx]
            if isinstance(base_values, np.ndarray) and base_values.ndim >= 2:
                active_base_vals = base_values[:, class_idx]
            elif isinstance(base_values, np.ndarray) and len(base_values) == len(classes_list):
                active_base_vals = np.repeat(base_values[class_idx], len(X_trans))
            else:
                active_base_vals = base_values
        else:
            active_shap_matrix = raw_values
            active_base_vals = base_values
    else:
        # Regression
        active_shap_matrix = raw_values
        active_base_vals = base_values

    # Model predictions on evaluation data
    model_preds = pipeline.predict(X_orig)
    model_probs = None
    if hasattr(pipeline, "predict_proba"):
        try:
            model_probs = pipeline.predict_proba(X_orig)
        except Exception:
            pass

    # 7. Aggregate transformed feature SHAP values to original features
    # Global mean absolute SHAP value across all evaluated rows
    global_orig_mean_abs = {col: 0.0 for col in expected_columns}
    for t_idx, orig_col in enumerate(name_mapping):
        mean_abs_t = np.mean(np.abs(active_shap_matrix[:, t_idx]))
        if orig_col in global_orig_mean_abs:
            global_orig_mean_abs[orig_col] += float(mean_abs_t)
        else:
            global_orig_mean_abs[orig_col] = float(mean_abs_t)

    global_ranking = [
        {
            "feature": col,
            "mean_abs_shap": _json_safe(round(val, 6)),
        }
        for col, val in global_orig_mean_abs.items()
    ]
    global_ranking.sort(key=lambda x: x["mean_abs_shap"], reverse=True)
    for rank, item in enumerate(global_ranking, start=1):
        item["rank"] = rank

    # 8. Local row explanations
    target_row_indices = row_indices if row_indices is not None else list(range(min(5, len(eval_df))))
    local_explanations = []

    for r_idx in target_row_indices:
        if r_idx < 0 or r_idx >= len(eval_df):
            continue

        row_orig = X_orig.iloc[r_idx].to_dict()
        row_base = (
            float(active_base_vals[r_idx])
            if isinstance(active_base_vals, (np.ndarray, list))
            else float(active_base_vals)
        )

        # Aggregate contributions to original features
        orig_contributions = {col: 0.0 for col in expected_columns}
        for t_idx, orig_col in enumerate(name_mapping):
            val = float(active_shap_matrix[r_idx, t_idx])
            if orig_col in orig_contributions:
                orig_contributions[orig_col] += val
            else:
                orig_contributions[orig_col] = val

        # Format feature contribution list
        contrib_list = []
        for feat, val in orig_contributions.items():
            contrib_list.append({
                "feature": feat,
                "original_value": _json_safe(row_orig.get(feat)),
                "shap_value": _json_safe(round(val, 6)),
                "direction": "positive" if val > 0 else "negative" if val < 0 else "neutral",
            })

        # Sort contributions by absolute magnitude
        contrib_list.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        actual_label = None
        if resolved_target and resolved_target in eval_df.columns:
            actual_label = _json_safe(eval_df.iloc[r_idx][resolved_target])

        prob_value = None
        if model_probs is not None and class_idx < model_probs.shape[1]:
            prob_value = _json_safe(round(float(model_probs[r_idx, class_idx]), 4))

        local_explanations.append({
            "row_index": r_idx,
            "prediction": _json_safe(model_preds[r_idx]),
            "prediction_probability": prob_value,
            "actual_target": actual_label,
            "base_value": _json_safe(round(row_base, 6)),
            "sum_shap_values": _json_safe(round(float(sum(orig_contributions.values())), 6)),
            "contributions": contrib_list,
        })

    duration = round(time.time() - t0, 3)

    return {
        "model_id": model_id,
        "model_name": metadata.get("model_name"),
        "task_type": task_type,
        "model_type": model_type,
        "method": "shap",
        "explainer_type": explainer.__class__.__name__,
        "dataset_source": dataset_source,
        "rows_evaluated": len(eval_df),
        "target_column": resolved_target,
        "classes": classes_list,
        "explained_class": classes_list[class_idx] if classes_list else None,
        "duration_seconds": duration,
        "global_importance": global_ranking,
        "local_explanations": local_explanations,
        "warnings": [
            "SHAP attributions quantify model prediction shifts relative to the expected base value.",
            "Attributions reflect model structure, not causal mechanisms.",
            f"Evaluated on {len(eval_df)} rows. Transformed one-hot categorical features were mapped to original feature columns via Shapley efficiency."
        ],
    }
