# app/routes/report_routes.py
"""
AutoVeyra Phase 5: Unified AI/ML Project Report Studio & Model Governance API Router.
Supports both Project-Wide Multi-Module Reports (with original dataset export & ZIP bundling)
and Individual Model Evaluation Snapshots (Model Cards).
"""

import json
import re
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Body, HTTPException, Query, Response

# Single Model Evaluation Reporting (Preserving Existing Contracts)
from app.services.report_service import (
    delete_report,
    export_html_report,
    export_pdf_report,
    generate_model_report,
    get_report,
    list_reports,
    update_model_card,
)

# Unified Project Report Studio (Phase 5)
from app.services.project_report_service import (
    create_project_report,
    delete_project_report,
    generate_project_html,
    generate_project_pdf,
    generate_project_zip_bundle,
    get_project_report,
    list_project_reports,
    load_dataset_df,
    render_project_charts,
    validate_project_runs_compatibility,
)

router = APIRouter(prefix="/reports", tags=["Project Report Studio & Model Evaluation"])


# ======================================================================
# 1. Project Report Studio Endpoints (Phase 5)
# ======================================================================

@router.post("/project/validate")
def validate_project_runs_endpoint(
    payload: Dict[str, Any] = Body(
        ...,
        example={
            "dataset_id": "ds_123456789abc",
            "selected_runs": {
                "eda": "run_eda_01",
                "cleaning": "run_clean_01",
                "training": "run_train_01",
            },
        },
    ),
):
    """
    Validate experiment compatibility across selected independent runs
    before compiling the final project documentation.
    """
    dataset_id = payload.get("dataset_id")
    if not dataset_id:
        raise HTTPException(status_code=400, detail="'dataset_id' is required for compatibility validation.")

    selected_runs = payload.get("selected_runs") or {}
    return validate_project_runs_compatibility(dataset_id=dataset_id, selected_runs=selected_runs)


@router.post("/project/generate")
def create_project_report_endpoint(
    payload: Dict[str, Any] = Body(
        ...,
        example={
            "project_name": "End-to-End Churn Classification System",
            "project_description": "Comprehensive pipeline evaluation and model card.",
            "author_name": "Senior ML Engineer",
            "institution": "Data Science Lab",
            "report_template": "complete",
            "original_dataset_id": "ds_123456789abc",
            "selected_runs": {
                "eda": "run_eda_01",
                "training": "run_train_01",
            },
            "include_original_dataset": True,
            "include_cleaned_dataset": False,
        },
    ),
):
    """
    Compile and generate a unified project report document spanning all selected
    module runs with guaranteed original dataset preservation.
    """
    return create_project_report(payload)


@router.get("/project")
def list_project_reports_endpoint(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """List compiled project reports with pagination."""
    return list_project_reports(limit=limit, offset=offset)


@router.get("/project/{report_id}")
def get_project_report_endpoint(report_id: str):
    """Retrieve structured 15-section technical document for a project report."""
    return get_project_report(report_id)


@router.delete("/project/{report_id}")
def delete_project_report_endpoint(report_id: str):
    """Delete a compiled project report record."""
    return delete_project_report(report_id)


@router.get("/project/{report_id}/download")
def download_project_report_endpoint(
    report_id: str,
    format: str = Query(default="pdf"),
    include_original: Optional[bool] = Query(default=None),
    include_cleaned: Optional[bool] = Query(default=None),
    include_model: Optional[bool] = Query(default=None),
):
    """
    Download compiled project report in PDF, HTML, JSON, or complete project ZIP bundle.
    When format='zip', packages report documents, verified original uploaded dataset,
    visualizations, workflow flowchart, and cryptographic MANIFEST.json.
    """
    fmt = format.strip().lower()
    if fmt not in {"pdf", "html", "json", "zip"}:
        raise HTTPException(status_code=400, detail="Supported formats: 'pdf', 'html', 'json', 'zip'.")

    report_record = get_project_report(report_id)
    report_data = report_record["report_data"]
    clean_id = re.sub(r"[^a-zA-Z0-9_\-]", "", report_id)[:16] or "project_report"
    safe_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", report_record["project_name"])[:30] or "report"

    # Assemble bundle options if query overrides provided
    bundle_options = dict(report_record.get("bundle_options") or {})
    if include_original is not None:
        bundle_options["include_original_dataset"] = include_original
    if include_cleaned is not None:
        bundle_options["include_cleaned_dataset"] = include_cleaned
    if include_model is not None:
        bundle_options["include_model_artifact"] = include_model

    if fmt == "zip":
        try:
            zip_bytes = generate_project_zip_bundle(report_id, bundle_options=bundle_options)
            return Response(
                content=zip_bytes,
                media_type="application/zip",
                headers={"Content-Disposition": f'attachment; filename="autoveyra_project_{safe_name}_{clean_id}.zip"'},
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to generate project ZIP bundle: {exc}")

    # Generate charts for PDF/HTML
    df = load_dataset_df(report_record["original_dataset_id"])
    charts = render_project_charts(df, report_data)

    if fmt == "pdf":
        try:
            pdf_bytes = generate_project_pdf(report_data, charts)
            return Response(
                content=pdf_bytes,
                media_type="application/pdf",
                headers={"Content-Disposition": f'attachment; filename="autoveyra_report_{safe_name}_{clean_id}.pdf"'},
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to render project PDF: {exc}")

    elif fmt == "html":
        try:
            html_content = generate_project_html(report_data, charts)
            return Response(
                content=html_content,
                media_type="text/html; charset=utf-8",
                headers={"Content-Disposition": f'attachment; filename="autoveyra_report_{safe_name}_{clean_id}.html"'},
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to render project HTML: {exc}")

    else:
        # JSON
        return Response(
            content=json.dumps(report_data, indent=2),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="autoveyra_report_{safe_name}_{clean_id}.json"'},
        )


# ======================================================================
# 2. Single Model Evaluation Reporting (Preserving Existing Contracts)
# ======================================================================

@router.post("/generate")
def create_model_report_endpoint(
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
    for a single registered model.
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
def list_model_reports_endpoint(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    model_id: Optional[int] = Query(default=None),
):
    """List historical single-model evaluation reports with pagination."""
    return list_reports(limit=limit, offset=offset, model_id=model_id)


@router.get("/{report_id}")
def get_model_report_endpoint(report_id: str):
    """Retrieve structured snapshot of a single-model evaluation report."""
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
def download_model_report_endpoint(
    report_id: str,
    format: str = Query(default="json"),
):
    """
    Download single-model report in PDF, HTML, or structured JSON format.
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
def delete_model_report_endpoint(report_id: str):
    """Safely delete a single-model report record."""
    return delete_report(report_id)
