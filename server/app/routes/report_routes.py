# app/routes/report_routes.py

import json
import re
from typing import Any, Dict, Optional
from fastapi import APIRouter, Body, HTTPException, Query, Response

from app.services.report_service import (
    delete_report,
    export_html_report,
    export_pdf_report,
    generate_model_report,
    get_report,
    list_reports,
    update_model_card,
)

router = APIRouter(prefix="/reports", tags=["Model Evaluation Reports & Model Cards"])


@router.post("/generate")
def create_report_endpoint(
    payload: Dict[str, Any] = Body(
        ...,
        example={
            "model_id": 1,
            "report_title": "Production Validation Report",
            "model_card": {
                "intended_use": "Decision support for risk assessment",
                "out_of_scope_use": "Autonomous loan rejection",
            },
        },
    ),
):
    """
    Generate an immutable, evidence-backed evaluation snapshot and standardized Model Card
    for a registered model.
    """
    model_id = payload.get("model_id")
    if model_id is None:
        raise HTTPException(status_code=400, detail="Request payload must contain 'model_id'.")

    try:
        model_id = int(model_id)
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="'model_id' must be an integer.")

    report_title = payload.get("report_title")
    user_model_card = payload.get("model_card")

    return generate_model_report(
        model_id=model_id,
        report_title=report_title,
        user_model_card=user_model_card,
    )


@router.get("")
def list_reports_endpoint(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    model_id: Optional[int] = Query(default=None),
):
    """List historical evaluation reports with pagination and optional model_id filter."""
    return list_reports(limit=limit, offset=offset, model_id=model_id)


@router.get("/{report_id}")
def get_report_endpoint(report_id: str):
    """Retrieve structured snapshot of a specific evaluation report."""
    return get_report(report_id)


@router.patch("/{report_id}/model-card")
def update_model_card_endpoint(
    report_id: str,
    payload: Dict[str, Any] = Body(
        ...,
        example={
            "intended_use": "Updated intended operational use.",
            "known_limitations": "Updated limitations note.",
        },
    ),
):
    """
    Update user-editable narrative fields of a Model Card without modifying measured performance metrics.
    """
    return update_model_card(report_id=report_id, updates=payload)


@router.get("/{report_id}/download")
def download_report_endpoint(
    report_id: str,
    format: str = Query(default="json"),
):
    """
    Download report in PDF, HTML, or structured JSON format.
    """
    fmt = format.strip().lower()
    if fmt not in {"pdf", "html", "json"}:
        raise HTTPException(status_code=400, detail="Supported download formats are 'pdf', 'html', and 'json'.")

    report = get_report(report_id)

    clean_id = re.sub(r"[^a-zA-Z0-9_\-]", "", report_id)[:16] or "report"

    if fmt == "pdf":
        try:
            pdf_bytes = export_pdf_report(report_id)
            return Response(
                content=pdf_bytes,
                media_type="application/pdf",
                headers={"Content-Disposition": f'attachment; filename="autoveyra_report_{clean_id}.pdf"'},
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to render PDF: {exc}")

    elif fmt == "html":
        try:
            html_content = export_html_report(report_id)
            return Response(
                content=html_content,
                media_type="text/html; charset=utf-8",
                headers={"Content-Disposition": f'attachment; filename="autoveyra_report_{clean_id}.html"'},
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to render HTML: {exc}")

    else:
        # JSON
        return Response(
            content=json.dumps(report, indent=2),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="autoveyra_report_{clean_id}.json"'},
        )


@router.delete("/{report_id}")
def delete_report_endpoint(report_id: str):
    """Safely delete a report record without affecting registered models or artifacts."""
    return delete_report(report_id)
