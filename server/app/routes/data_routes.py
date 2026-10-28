
from typing import Optional
import json

from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse

from app.services.train_service import train_model_service
from app.services.classification_service import train_classification_service
from app.services.comparison_service import compare_models_service
from app.services.adanceCleaning import AdvancedDataCleaner
from app.services.model_registry_service import register_model
from app.services.experiment_service import create_experiment


router = APIRouter()


# ---------------------------------------------------------
# Experiment history helper
# ---------------------------------------------------------

def save_experiment_safely(
    *,
    experiment_name: str,
    task_type: str,
    dataset_name: Optional[str],
    target_column: Optional[str],
    model_type: Optional[str],
    scaler_type: Optional[str],
    missing_strategy: Optional[str],
    encoding_strategy: Optional[str],
    split_ratio: Optional[float],
    metrics: Optional[dict],
    result: dict,
):
    """Save experiment history without breaking a successful request."""
    try:
        return create_experiment(
            experiment_name=experiment_name,
            task_type=task_type,
            dataset_name=dataset_name,
            target_column=target_column,
            model_type=model_type,
            scaler_type=scaler_type,
            missing_strategy=missing_strategy,
            encoding_strategy=encoding_strategy,
            split_ratio=split_ratio,
            metrics=metrics or {},
            result=result,
        )
    except Exception as exc:
        print(f"Experiment history save failed: {exc}")
        return None


# ---------------------------------------------------------
# Train model
# ---------------------------------------------------------

@router.post("/train")
async def train(
    file: UploadFile = File(...),
    model: str = Form(...),
    scaler: str = Form(...),
    splitRatio: float = Form(...),
    missing: str = Form(...),
    encoding: str = Form(...),
    task_type: str = Form("regression"),
    target_column: Optional[str] = Form(None),
):
    try:
        normalized_task = (
            task_type.strip().lower() if task_type else "regression"
        )

        if normalized_task == "regression":
            result = await train_model_service(
                file,
                model,
                scaler,
                splitRatio,
                missing,
                encoding,
            )

            # Keep existing regression behavior unchanged.
            save_experiment_safely(
                experiment_name=f"{file.filename} - {model}",
                task_type="regression",
                dataset_name=file.filename,
                target_column=target_column,
                model_type=model,
                scaler_type=scaler,
                missing_strategy=missing,
                encoding_strategy=encoding,
                split_ratio=splitRatio,
                metrics=result.get("metrics", {}),
                result=result,
            )

            return JSONResponse(content=result)

        elif normalized_task == "classification":
            result, fitted_pipeline = await train_classification_service(
                file=file,
                model=model,
                scaler=scaler,
                split_ratio=splitRatio,
                missing=missing,
                encoding=encoding,
                target_column=target_column,
                return_pipeline=True,
            )

            registered_model = register_model(
                pipeline=fitted_pipeline,
                model_name=(
                    f"{file.filename.rsplit('.', 1)[0]} "
                    f"- {result['model_type']}"
                ),
                task_type="classification",
                model_type=result["model_type"],
                dataset_name=file.filename,
                target_column=result["target_column"],
                feature_columns=result["columns"],
                metrics=result["metrics"],
            )

            result["registered_model"] = registered_model

            save_experiment_safely(
                experiment_name=(
                    f"{file.filename} - {result['model_type']}"
                ),
                task_type="classification",
                dataset_name=file.filename,
                target_column=result["target_column"],
                model_type=result["model_type"],
                scaler_type=result["scaler_type"],
                missing_strategy=result["missing_strategy"],
                encoding_strategy=result["encoding_strategy"],
                split_ratio=result["split_ratio"],
                metrics=result["metrics"],
                result=result,
            )

            return JSONResponse(content=result)

        else:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported task_type '{task_type}'. "
                    "Supported task types are 'regression' "
                    "and 'classification'."
                ),
            )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------
# Advanced data cleaning
# ---------------------------------------------------------

@router.post("/api/clean")
@router.post("/clean")
async def clean_data(
    file: UploadFile = File(...),
    preprocessing: str = Form("{}"),
):
    try:
        preprocessing_config = json.loads(preprocessing)

        cleaner = AdvancedDataCleaner(
            file,
            preprocessing_config,
        )

        result = await cleaner.clean_data()
        return JSONResponse(content=result)

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail="Invalid preprocessing JSON",
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------
# Model comparison + best model registration
# ---------------------------------------------------------

@router.post("/compare")
async def compare(
    file: UploadFile = File(...),
    task_type: str = Form(...),
    target_column: str = Form(...),
    scaler: str = Form("standard"),
    splitRatio: float = Form(0.8),
    missing: str = Form("median"),
    encoding: str = Form("onehot"),
    ranking_metric: Optional[str] = Form(None),
):
    try:
        normalized_task = task_type.strip().lower()

        # Request the fitted winning pipeline as well as its metrics.
        result, fitted_pipeline = await compare_models_service(
            file=file,
            task_type=normalized_task,
            target_column=target_column,
            scaler=scaler,
            split_ratio=splitRatio,
            missing=missing,
            encoding=encoding,
            ranking_metric=ranking_metric,
            return_pipeline=True,
        )

        best_model = result.get("best_model") or {}
        best_model_name = best_model.get("model")

        if not best_model_name:
            raise HTTPException(
                status_code=500,
                detail="Model comparison did not return a winning model.",
            )

        holdout_metrics = best_model.get("holdout_metrics") or {}

        # The comparison service must provide the exact feature order
        # expected by the fitted pipeline.
        feature_columns = result.get("feature_columns")

        if not feature_columns:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Comparison result is missing feature_columns. "
                    "Add feature_columns to the response in "
                    "comparison_service.py."
                ),
            )

        # Register the winning fitted pipeline.
        registered_model = register_model(
            pipeline=fitted_pipeline,
            model_name=(
                f"{file.filename.rsplit('.', 1)[0]} "
                f"- {best_model_name}"
            ),
            task_type=normalized_task,
            model_type=best_model_name,
            dataset_name=file.filename,
            target_column=result.get("target_column", target_column),
            feature_columns=feature_columns,
            metrics=holdout_metrics,
        )

        result["registered_model"] = registered_model

        # Record the comparison experiment.
        save_experiment_safely(
            experiment_name=f"{file.filename} - Model Comparison",
            task_type=normalized_task,
            dataset_name=file.filename,
            target_column=result.get("target_column", target_column),
            model_type=best_model_name,
            scaler_type=scaler,
            missing_strategy=missing,
            encoding_strategy=encoding,
            split_ratio=splitRatio,
            metrics=holdout_metrics,
            result=result,
        )

        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------
# Health check
# ---------------------------------------------------------

@router.get("/test")
def test():
    return {
        "message": "Backend is working!"
    }
