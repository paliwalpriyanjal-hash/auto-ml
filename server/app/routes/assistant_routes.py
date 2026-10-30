# server/app/routes/assistant_routes.py
"""
AutoVeyra Phase 6: Guided Workspace & Intelligent User Assistance API Router
Module: assistant_routes.py
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Body, HTTPException, Query
from fastapi.responses import JSONResponse

from app.services.assistant_service import (
    ML_GLOSSARY,
    ALGORITHM_PROFILES,
    recommend_next_action,
    explain_metric_result,
    explain_generalization_gap,
    validate_preflight,
    create_or_update_workspace_project,
    get_workspace_project,
    list_workspace_projects,
    delete_workspace_project,
)

router = APIRouter(prefix="/assistant", tags=["Intelligent Assistant & Guided Workspace"])


# ======================================================================
# 1. ML Glossary Endpoints
# ======================================================================

@router.get("/glossary")
def get_glossary(
    category: Optional[str] = Query(None, description="Filter by term category"),
    q: Optional[str] = Query(None, description="Search term or definition text"),
):
    """
    Returns beginner-friendly explanations for technical machine learning terms.
    Each term contains a simple definition, why it matters, an example, and where it appears in AutoVeyra.
    """
    results = list(ML_GLOSSARY.values())

    if category:
        cat_lower = category.lower().strip()
        results = [t for t in results if t["category"].lower() == cat_lower]

    if q:
        search = q.lower().strip()
        results = [
            t for t in results
            if search in t["term"].lower()
            or search in t["simple_definition"].lower()
            or search in t["example"].lower()
        ]

    return JSONResponse(content={"total": len(results), "terms": results})


@router.get("/glossary/{term}")
def get_glossary_term(term: str):
    """Retrieve detailed explanation for a single technical concept."""
    key = term.lower().strip()
    entry = ML_GLOSSARY.get(key)
    if not entry:
        # Try substring match
        matched = [v for k, v in ML_GLOSSARY.items() if key in k or k in key]
        if matched:
            return JSONResponse(content=matched[0])
        raise HTTPException(status_code=404, detail=f"Concept '{term}' not found in ML Glossary.")
    return JSONResponse(content=entry)


# ======================================================================
# 2. Algorithm Profiles
# ======================================================================

@router.get("/algorithms")
def get_algorithm_profiles():
    """Returns beginner-friendly summaries, analogies, and pros/cons of supported ML algorithms."""
    return JSONResponse(content={"algorithms": ALGORITHM_PROFILES})


# ======================================================================
# 3. Next-Action Recommendation
# ======================================================================

@router.post("/recommend-next")
def recommend_next_step(payload: Dict[str, Any] = Body(...)):
    """
    Rule-based recommendation engine evaluating actual project state
    (dataset, runs, lineage, target, task, completed stages).
    """
    recommendation = recommend_next_action(payload)
    return JSONResponse(content=recommendation)


# ======================================================================
# 4. Context Explanations & Metric Interpretation
# ======================================================================

@router.post("/explain-context")
def explain_context(payload: Dict[str, Any] = Body(...)):
    """
    Provides plain-English translations for metrics, generalization gaps,
    or algorithm decisions.
    """
    explanation_type = payload.get("type", "metric")

    if explanation_type == "metric":
        metric_name = payload.get("metric_name", "Accuracy")
        value = float(payload.get("value", 0.0))
        task_type = payload.get("task_type", "classification")
        res = explain_metric_result(metric_name, value, task_type)
        return JSONResponse(content=res)

    elif explanation_type == "generalization_gap":
        cv = float(payload.get("cv_score", 0.0))
        holdout = float(payload.get("holdout_score", 0.0))
        metric_name = payload.get("metric_name", "Score")
        res = explain_generalization_gap(cv, holdout, metric_name)
        return JSONResponse(content=res)

    elif explanation_type == "algorithm":
        algo_name = payload.get("algorithm_name", "").lower().replace(" ", "").replace("_", "")
        profile = ALGORITHM_PROFILES.get(algo_name)
        if not profile:
            return JSONResponse(content={"name": payload.get("algorithm_name"), "description": "Standard ML model."})
        return JSONResponse(content=profile)

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported explanation type '{explanation_type}'.")


# ======================================================================
# 5. Pre-Flight Validation
# ======================================================================

@router.post("/validate-preflight")
def preflight_validation(payload: Dict[str, Any] = Body(...)):
    """
    Validates configuration before major operations to prevent common beginner errors.
    """
    action = payload.get("action", "train")
    data = payload.get("payload", {})
    validation = validate_preflight(action=action, payload=data)
    return JSONResponse(content=validation)


# ======================================================================
# 6. Workspace Projects Management (Session Continuity)
# ======================================================================

@router.get("/projects")
def list_projects_endpoint(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """List guided projects for resuming previous work."""
    return JSONResponse(content=list_workspace_projects(limit=limit, offset=offset))


@router.post("/projects")
def save_project_endpoint(payload: Dict[str, Any] = Body(...)):
    """Create or update a guided workspace project context."""
    name = payload.get("name", "Untitled Project")
    project = create_or_update_workspace_project(
        name=name,
        dataset_id=payload.get("dataset_id"),
        dataset_name=payload.get("dataset_name"),
        target_column=payload.get("target_column"),
        task_type=payload.get("task_type", "classification"),
        experience_mode=payload.get("experience_mode", "guided"),
        current_step=int(payload.get("current_step", 1)),
        completed_steps=payload.get("completed_steps", []),
        active_runs=payload.get("active_runs", {}),
        metadata=payload.get("metadata", {}),
        project_id=payload.get("project_id"),
    )
    return JSONResponse(content=project)


@router.get("/projects/{project_id}")
def get_project_endpoint(project_id: str):
    """Retrieve full project context for resuming."""
    project = get_workspace_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Workspace project not found.")
    return JSONResponse(content=project)


@router.delete("/projects/{project_id}")
def delete_project_endpoint(project_id: str):
    """Delete a workspace project."""
    success = delete_workspace_project(project_id)
    if not success:
        raise HTTPException(status_code=404, detail="Workspace project not found.")
    return JSONResponse(content={"deleted": True, "project_id": project_id})
