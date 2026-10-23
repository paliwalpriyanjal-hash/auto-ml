# app/routes/data_routes.py
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
from app.services.train_service import train_model_service
from app.services.classification_service import train_classification_service
from app.services.comparison_service import compare_models_service
from app.services.adanceCleaning import AdvancedDataCleaner
import json

router = APIRouter()

@router.post("/train")
async def train(
    file: UploadFile = File(...),
    model: str = Form(...),
    scaler: str = Form(...),
    splitRatio: float = Form(...),
    missing: str = Form(...),
    encoding: str = Form(...),
    task_type: str = Form("regression"),
    target_column: Optional[str] = Form(None)
):
    try:
        # Normalize and validate task type
        normalized_task = task_type.strip().lower() if task_type else "regression"

        if normalized_task == "regression":
            result = await train_model_service(file, model, scaler, splitRatio, missing, encoding)
            return JSONResponse(content=result)
        elif normalized_task == "classification":
            result = await train_classification_service(
                file=file,
                model=model,
                scaler=scaler,
                split_ratio=splitRatio,
                missing=missing,
                encoding=encoding,
                target_column=target_column
            )
            return JSONResponse(content=result)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported task_type '{task_type}'. Supported task types are 'regression' and 'classification'."
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/clean")
@router.post("/clean")
async def clean_data(
    file: UploadFile = File(...),
    preprocessing: str = Form("{}")
):
    try:
        # Parse preprocessing JSON
        preprocessing_config = json.loads(preprocessing)
        # print(preprocessing_config)
        
        # Initialize cleaner and process data
        cleaner = AdvancedDataCleaner(file, preprocessing_config)
        result = await cleaner.clean_data()
        
        return JSONResponse(content=result)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid preprocessing JSON")
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/compare")
async def compare(
    file: UploadFile = File(...),
    task_type: str = Form(...),
    target_column: str = Form(...),
    scaler: str = Form("standard"),
    splitRatio: float = Form(0.8),
    missing: str = Form("median"),
    encoding: str = Form("onehot"),
    ranking_metric: Optional[str] = Form(None)
):
    try:
        result = await compare_models_service(
            file=file,
            task_type=task_type,
            target_column=target_column,
            scaler=scaler,
            split_ratio=splitRatio,
            missing=missing,
            encoding=encoding,
            ranking_metric=ranking_metric,
        )
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/test")
def test():
    return {"message": "Backend is working!"}