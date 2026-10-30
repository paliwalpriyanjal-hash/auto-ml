"""
AutoVeyra Phase 2: Exploratory Data Analysis API Router
Module: eda_routes.py
"""

from typing import Any, Dict, Optional
import io
import json
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Response
from fastapi.responses import JSONResponse, StreamingResponse

from app.utils.security import safe_read_csv
from app.services.eda_service import (
    profile_dataset,
    detect_eda_strategy,
    run_full_eda_pipeline,
    get_eda_analysis_from_db,
    generate_eda_pdf_report,
    generate_eda_html_report,
    _connect,
)

router = APIRouter(prefix="/eda", tags=["Exploratory Data Analysis (EDA)"])


@router.post("/profile")
async def profile_endpoint(
    file: UploadFile = File(...),
    target_column: Optional[str] = Form(None),
):
    """
    Rapid dataset profiling without intensive visualization or correlation matrix rendering.
    Returns dataset dimensions, column-level summaries, and recommended EDA strategy.
    """
    try:
        df = await safe_read_csv(file)
        profile_data = profile_dataset(
            df=df,
            filename=file.filename or "dataset.csv",
            target_column=target_column,
        )
        strategy_data = detect_eda_strategy(
            df=df,
            profile_data=profile_data,
            target_column=target_column,
        )
        return JSONResponse(
            content={
                "filename": file.filename or "dataset.csv",
                "profile": profile_data,
                "strategy": strategy_data,
            }
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/analyze")
async def analyze_endpoint(
    file: UploadFile = File(...),
    target_column: Optional[str] = Form(None),
):
    """
    Executes the full adaptive EDA intelligence pipeline.
    Computes statistical moments, data quality audit, numerical and categorical associations,
    target-aware relationships, evidence-based insights, and rendered visualizations.
    """
    try:
        df = await safe_read_csv(file)
        result = await run_full_eda_pipeline(
            df=df,
            filename=file.filename or "dataset.csv",
            target_column=target_column,
            save_to_db=True,
        )
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/history")
def list_eda_history():
    """Lists previous EDA analyses stored in the platform."""
    try:
        with _connect() as conn:
            rows = conn.execute(
                """
                SELECT analysis_id, filename, row_count, column_count,
                       target_column, task_type, created_at
                FROM eda_analyses
                ORDER BY created_at DESC
                LIMIT 50
                """
            ).fetchall()
            return [dict(r) for r in rows]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/{analysis_id}")
def get_analysis_endpoint(analysis_id: str):
    """Retrieves full statistical analysis data by analysis ID."""
    analysis = get_eda_analysis_from_db(analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail=f"EDA analysis '{analysis_id}' not found.")
    return JSONResponse(content=analysis)


@router.get("/{analysis_id}/insights")
def get_insights_endpoint(analysis_id: str):
    """Retrieves synthesized evidence-based statistical insights for an analysis."""
    analysis = get_eda_analysis_from_db(analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail=f"EDA analysis '{analysis_id}' not found.")
    return JSONResponse(content=analysis.get("insights", {}))


@router.get("/{analysis_id}/flowchart")
def get_flowchart_endpoint(analysis_id: str):
    """Retrieves the dynamic EDA execution flowchart data."""
    analysis = get_eda_analysis_from_db(analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail=f"EDA analysis '{analysis_id}' not found.")
    return JSONResponse(content=analysis.get("flowchart", {}))


@router.get("/{analysis_id}/report/pdf")
def download_pdf_report(analysis_id: str):
    """Downloads an immutable, professional multi-page PDF evaluation report."""
    analysis = get_eda_analysis_from_db(analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail=f"EDA analysis '{analysis_id}' not found.")

    try:
        pdf_bytes = generate_eda_pdf_report(analysis)
        filename = f"AutoVeyra_EDA_{analysis['filename'].rsplit('.', 1)[0]}_{analysis_id[:8]}.pdf"
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {exc}")


@router.get("/{analysis_id}/report/html")
def download_html_report(analysis_id: str):
    """Downloads a standalone, responsive dark-themed HTML report."""
    analysis = get_eda_analysis_from_db(analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail=f"EDA analysis '{analysis_id}' not found.")

    try:
        html_content = generate_eda_html_report(analysis)
        filename = f"AutoVeyra_EDA_{analysis['filename'].rsplit('.', 1)[0]}_{analysis_id[:8]}.html"
        return Response(
            content=html_content,
            media_type="text/html",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"HTML generation failed: {exc}")


@router.get("/{analysis_id}/report/json")
def download_json_report(analysis_id: str):
    """Downloads structured machine-readable JSON analysis output."""
    analysis = get_eda_analysis_from_db(analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail=f"EDA analysis '{analysis_id}' not found.")

    filename = f"AutoVeyra_EDA_{analysis['filename'].rsplit('.', 1)[0]}_{analysis_id[:8]}.json"
    return Response(
        content=json.dumps(analysis, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
