# server/app/services/train_service.py
"""
Dedicated Regression Model Training Service for AutoVeyra.

Upgraded to prevent data leakage:
- Preprocessing and scalers are wrapped inside sklearn ColumnTransformer and Pipeline.
- Split occurs BEFORE any preprocessing; scalers/encoders fit strictly on training data.
- K-fold cross-validation is performed on training data to assess generalization stability.
- Generalization gap (overfitting vs underfitting) is monitored and diagnosed.
- Returns fitted sklearn Pipeline for persistent registration in Model Registry.
"""

import io
import base64
import math
import traceback
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from fastapi import HTTPException, UploadFile
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    OneHotEncoder,
)
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    median_absolute_error,
    explained_variance_score,
    max_error as sklearn_max_error,
    mean_absolute_percentage_error,
)

# Regression Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    GradientBoostingRegressor,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from app.utils.security import safe_read_csv
from app.utils.visualization import get_visualization_plot, get_prediction_plot
from app.services.ml_intelligence_service import (
    safe_float,
    analyze_dataset_intelligence,
    generate_training_diagnostics,
)


def _resolve_regression_model(model_name: str) -> Tuple[Any, str]:
    """Map user/frontend model names to scikit-learn regression estimators."""
    key = (model_name or "").strip().lower().replace(" ", "").replace("_", "")

    if key in {"linearregression", "linear", "lr"}:
        return LinearRegression(), "LinearRegression"
    elif key in {"ridge"}:
        return Ridge(alpha=1.0, random_state=42), "Ridge"
    elif key in {"lasso"}:
        return Lasso(alpha=0.1, random_state=42), "Lasso"
    elif key in {"elasticnet", "elastic"}:
        return ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42), "ElasticNet"
    elif key in {"decisiontree", "decisiontreeregressor", "dt"}:
        return DecisionTreeRegressor(random_state=42), "DecisionTree"
    elif key in {"randomforest", "randomforestregressor", "rf"}:
        return RandomForestRegressor(n_estimators=100, n_jobs=1, random_state=42), "RandomForest"
    elif key in {"extratrees", "extratreesregressor", "et"}:
        return ExtraTreesRegressor(n_estimators=100, n_jobs=1, random_state=42), "ExtraTrees"
    elif key in {"histgradientboosting", "histgradientboostingregressor", "histgb", "hgb"}:
        return HistGradientBoostingRegressor(max_iter=100, random_state=42), "HistGradientBoosting"
    elif key in {"gradientboosting", "gradientboostingregressor", "gb", "gbr"}:
        return GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42), "GradientBoosting"
    elif key in {"kneighbors", "kneighborsregressor", "knn"}:
        return KNeighborsRegressor(n_neighbors=5, n_jobs=1), "KNeighborsRegressor"
    elif key in {"svr", "supportvectorregression", "svm"}:
        return SVR(kernel="rbf"), "SVR"
    else:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid model type '{model_name}'. Available regression options: "
                "LinearRegression, Ridge, Lasso, ElasticNet, DecisionTree, "
                "RandomForest, ExtraTrees, HistGradientBoosting, GradientBoosting, "
                "KNeighborsRegressor, SVR"
            )
        )


async def train_model_service(
    file: UploadFile,
    model: str,
    scaler: str,
    split_ratio: float,
    missing: str,
    encoding: str,
    target_column: Optional[str] = None,
    return_pipeline: bool = False,
) -> Union[Dict[str, Any], Tuple[Dict[str, Any], Pipeline]]:
    """
    Train and evaluate a regression model with 100% leak-free Pipeline architecture.
    Fits all imputers and scalers exclusively on the training split.
    """
    try:
        # Validate input parameters
        if split_ratio <= 0.0 or split_ratio >= 1.0:
            raise HTTPException(status_code=400, detail="The training splitRatio must be between 0 and 1.")

        if not file.filename or not file.filename.lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only CSV files are supported.")

        df = await safe_read_csv(file)
        if df.empty or df.shape[1] < 2:
            raise HTTPException(status_code=400, detail="The CSV must contain at least 2 columns and 1 row.")

        # Clean column names
        df.columns = [str(c).strip() for c in df.columns]
        original_shape = df.shape

        # Resolve target column: use requested column, or default to last column for backward compatibility
        if target_column and target_column in df.columns:
            resolved_target = target_column
        else:
            resolved_target = df.columns[-1]

        # Ensure target is numeric
        cleaned_df = df.dropna(subset=[resolved_target]).copy()
        if cleaned_df.empty:
            raise HTTPException(status_code=400, detail=f"Target column '{resolved_target}' contains only missing values.")

        # Convert target to numeric
        try:
            cleaned_df[resolved_target] = pd.to_numeric(cleaned_df[resolved_target])
        except Exception:
            raise HTTPException(status_code=400, detail=f"Regression requires a numeric target column. '{resolved_target}' cannot be converted to numeric.")

        feature_cols = [c for c in cleaned_df.columns if c != resolved_target]
        if not feature_cols:
            raise HTTPException(status_code=400, detail="No feature columns remain besides target.")

        # Identify numeric and categorical columns
        numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(cleaned_df[c])]
        categorical_cols = [c for c in feature_cols if c not in numeric_cols]

        # Handle 'drop' missing strategy if explicitly requested
        missing_strategy = (missing or "median").strip().lower()
        if missing_strategy == "drop":
            cleaned_df = cleaned_df.dropna(subset=feature_cols).copy()
            if cleaned_df.empty:
                raise HTTPException(status_code=400, detail="All rows were dropped during missing value handling.")

        # Separate X and y
        X = cleaned_df[feature_cols].copy()
        y = cleaned_df[resolved_target].copy()

        # Split features and target FIRST (0% data leakage into test set!)
        test_size = 1.0 - split_ratio
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Build leak-free ColumnTransformer
        transformers = []
        if numeric_cols:
            impute_strat = "mean" if missing_strategy == "mean" else "median"
            num_steps = [("imputer", SimpleImputer(strategy=impute_strat))]

            norm_scaler = (scaler or "standard").strip().lower()
            if norm_scaler == "standard":
                num_steps.append(("scaler", StandardScaler()))
            elif norm_scaler == "minmax":
                num_steps.append(("scaler", MinMaxScaler()))
            elif norm_scaler == "robust":
                num_steps.append(("scaler", RobustScaler()))

            transformers.append(("numeric", Pipeline(steps=num_steps), numeric_cols))

        if categorical_cols:
            cat_steps = [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
            transformers.append(("categorical", Pipeline(steps=cat_steps), categorical_cols))

        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

        # Resolve estimator
        estimator, resolved_model_name = _resolve_regression_model(model)

        # Construct full Pipeline
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ])

        # Cross-validation on training fold (Training-Only CV)
        cv_folds = min(3, max(2, len(X_train) // 5)) if len(X_train) >= 6 else 2
        cv_results = {}
        try:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            cv_scores = cross_validate(
                pipeline,
                X_train,
                y_train,
                cv=cv,
                scoring=["neg_root_mean_squared_error", "r2", "neg_mean_absolute_error"],
                n_jobs=1,
                return_train_score=True,
            )
            cv_results = {
                "cv_rmse_mean": safe_float(abs(float(np.mean(cv_scores["test_neg_root_mean_squared_error"])))),
                "cv_rmse_std": safe_float(float(np.std(cv_scores["test_neg_root_mean_squared_error"]))),
                "cv_r2_mean": safe_float(float(np.mean(cv_scores["test_r2"]))),
                "cv_r2_std": safe_float(float(np.std(cv_scores["test_r2"]))),
                "cv_mae_mean": safe_float(abs(float(np.mean(cv_scores["test_neg_mean_absolute_error"])))),
                "cv_mae_std": safe_float(float(np.std(cv_scores["test_neg_mean_absolute_error"]))),
                "train_r2_mean": safe_float(float(np.mean(cv_scores["train_r2"]))),
                "r2_mean": safe_float(float(np.mean(cv_scores["test_r2"]))),
                "rmse_mean": safe_float(abs(float(np.mean(cv_scores["test_neg_root_mean_squared_error"])))),
                "mae_mean": safe_float(abs(float(np.mean(cv_scores["test_neg_mean_absolute_error"])))),
                "folds": cv_folds,
            }
        except Exception:
            pass

        # Fit pipeline on training data
        pipeline.fit(X_train, y_train)

        # Predict on holdout test set
        y_pred = pipeline.predict(X_test)
        y_train_pred = pipeline.predict(X_train)

        # Calculate Holdout Metrics
        mse = mean_squared_error(y_test, y_pred) if len(y_test) > 0 else 0.0
        rmse = math.sqrt(mse) if mse >= 0 else 0.0
        r2 = r2_score(y_test, y_pred) if len(y_test) > 0 else 0.0
        mae = mean_absolute_error(y_test, y_pred) if len(y_test) > 0 else 0.0
        evs = explained_variance_score(y_test, y_pred) if len(y_test) > 0 else 0.0
        max_err = sklearn_max_error(y_test, y_pred) if len(y_test) > 0 else 0.0
        mape_val = None
        if not np.any(np.isclose(y_test.values, 0.0)):
            try:
                mape_val = safe_float(mean_absolute_percentage_error(y_test, y_pred))
            except Exception:
                pass

        train_r2 = safe_float(r2_score(y_train, y_train_pred))

        # Backward compatible metrics dict
        metrics = {
            "rmse": safe_float(rmse),
            "r2": safe_float(r2),
            "mae": safe_float(mae),
            "explained_variance": safe_float(evs),
            "max_error": safe_float(max_err),
            "mape": mape_val,
            "training_samples": int(len(X_train)),
            "testing_samples": int(len(X_test)),
        }

        # Feature importances extraction
        feature_importances = {}
        fitted_model = pipeline.named_steps["model"]
        try:
            # Transform column names from preprocessor
            transformed_names = []
            if hasattr(pipeline.named_steps["preprocessor"], "get_feature_names_out"):
                raw_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
                transformed_names = [n.replace("numeric__", "").replace("categorical__", "") for n in raw_names]
            else:
                transformed_names = feature_cols

            if hasattr(fitted_model, "feature_importances_"):
                feature_importances = {
                    str(k): safe_float(v) for k, v in zip(transformed_names, fitted_model.feature_importances_)
                }
            elif hasattr(fitted_model, "coef_"):
                feature_importances = {
                    str(k): safe_float(v) for k, v in zip(transformed_names, fitted_model.coef_)
                }
        except Exception:
            pass

        if feature_importances:
            metrics["feature_importances"] = feature_importances

        # Generate Visualizations (PNG Base64)
        vis = None
        try:
            vis = get_visualization_plot(fitted_model, list(feature_importances.keys())[:15], resolved_model_name)
        except Exception:
            pass

        pred_plot = None
        try:
            pred_plot = get_prediction_plot(y_test, y_pred)
        except Exception:
            pass

        # Diagnostic evaluation
        cv_r2_val = cv_results.get("cv_r2_mean")
        diagnostics = generate_training_diagnostics(
            task_type="regression",
            training_score=train_r2,
            cv_score=cv_r2_val if cv_r2_val is not None else safe_float(r2),
            holdout_score=safe_float(r2),
            dataset_summary={"total_rows": len(df)},
            target_profile={"is_imbalanced": False},
            primary_metric="R²",
        )

        # Preprocessing tracking stats
        preprocessing_stats = {
            "initial_shape": original_shape,
            "final_shape": cleaned_df.shape,
            "numeric_columns_count": len(numeric_cols),
            "categorical_columns_count": len(categorical_cols),
            "scaler_applied": scaler,
            "imputation_strategy": missing_strategy,
        }

        dashboard_metrics = {
            "model_performance": {
                "metrics": metrics,
                "cv_metrics": cv_results,
                "plots": {
                    "feature_importance": bool(feature_importances),
                    "prediction_plot": bool(pred_plot),
                },
            },
            "data_quality": {
                "initial_shape": f"{original_shape[0]} rows × {original_shape[1]} cols",
                "final_shape": f"{cleaned_df.shape[0]} rows × {cleaned_df.shape[1]} cols",
                "missing_values": int(df.isna().sum().sum()),
                "feature_types": {
                    "numeric": len(numeric_cols),
                    "categorical": len(categorical_cols),
                },
            },
            "feature_analysis": {
                "importances": feature_importances,
                "top_features": dict(
                    sorted(
                        feature_importances.items(),
                        key=lambda item: abs(item[1] or 0),
                        reverse=True
                    )[:5]
                ) if feature_importances else {},
            },
            "generalization": diagnostics,
        }

        result = {
            "metrics": metrics,
            "cv_metrics": cv_results,
            "training_metrics": {"r2": train_r2},
            "preprocessing_stats": preprocessing_stats,
            "visualization": vis,
            "prediction_plot": pred_plot,
            "columns": feature_cols,
            "target_column": resolved_target,
            "rows": X_test.head(5).to_dict(orient="records"),
            "raw_rows": df.head(5).to_dict(orient="records"),
            "model_type": resolved_model_name,
            "scaler_type": scaler,
            "diagnostics": diagnostics,
            "dashboard_metrics": dashboard_metrics,
        }

        if return_pipeline:
            # Refit pipeline on full dataset before saving to registry
            full_pipeline = Pipeline(steps=[
                ("preprocessor", ColumnTransformer(transformers=transformers, remainder="drop")),
                ("model", _resolve_regression_model(model)[0]),
            ])
            full_pipeline.fit(X, y)
            return result, full_pipeline

        return result

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Regression model training failed: {str(e)}"
        )
