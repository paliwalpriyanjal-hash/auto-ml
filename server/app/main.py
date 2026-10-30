# server/app/main.py

import logging
import time
import uuid
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ConfigDict, create_model
from fastapi.dependencies import utils as fastapi_utils

# Resolve Pydantic v2 protected namespace warnings ("model_") on dynamic body models
# without modifying public API field names or breaking request schemas.
_orig_create_body_model = fastapi_utils.create_body_model
def _safe_create_body_model(*, fields, model_name):
    field_params = {f.name: (f.field_info.annotation, f.field_info) for f in fields}
    return create_model(
        model_name,
        __config__=ConfigDict(protected_namespaces=()),
        **field_params
    )
fastapi_utils.create_body_model = _safe_create_body_model

from app.config import settings
from app.routes.data_routes import router as data_router
from app.routes.experiment_routes import router as experiment_router
from app.routes.model_routes import router as model_router
from app.routes.tuning_routes import router as tuning_router
from app.routes.feature_routes import router as feature_router
from app.routes.report_routes import router as report_router
from app.routes.auth_routes import router as auth_router
from app.routes.monitoring_routes import router as monitoring_router
from app.routes.eda_routes import router as eda_router
from app.routes.assistant_routes import router as assistant_router

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Correlation ID and Request Metrics Middleware
@app.middleware("http")
async def correlation_and_metrics_middleware(request: Request, call_next):
    correlation_id = (
        request.headers.get("X-Correlation-ID")
        or request.headers.get("X-Request-ID")
        or uuid.uuid4().hex
    )
    request.state.correlation_id = correlation_id
    start_time = time.time()

    try:
        response = await call_next(request)
        duration_ms = (time.time() - start_time) * 1000
        response.headers["X-Correlation-ID"] = correlation_id
        return response
    except Exception as exc:
        duration_ms = (time.time() - start_time) * 1000
        logging.error(
            f"[CID={correlation_id}] Unhandled error on {request.method} {request.url.path} "
            f"after {duration_ms:.1f}ms: {exc}"
        )
        raise exc


# Exception Handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    cid = getattr(request.state, "correlation_id", "unknown")
    headers = getattr(exc, "headers", None) or {}
    headers["X-Correlation-ID"] = cid
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers=headers,
    )


# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Standard Operations & Health Endpoints
@app.get("/health", tags=["Operations & Health"])
def health_probe():
    """Shallow liveness probe confirming API service availability."""
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
    }


@app.get("/ready", tags=["Operations & Health"])
def readiness_probe():
    """
    Deep readiness probe verifying database connectivity and persistent disk storage access.
    """
    checks = {}
    from app.services.model_registry_service import _connect as _model_conn
    from app.services.report_service import _connect as _rep_conn
    from app.services.auth_service import _connect as _auth_conn, init_auth_db
    from app.services.monitoring_service import _connect as _mon_conn, init_monitoring_db

    # Check database read/write readiness
    try:
        init_auth_db()
        init_monitoring_db()
        with _model_conn() as conn:
            conn.execute("SELECT 1")
        with _rep_conn() as conn:
            conn.execute("SELECT 1")
        with _auth_conn() as conn:
            conn.execute("SELECT 1")
        with _mon_conn() as conn:
            conn.execute("SELECT 1")
        checks["sqlite_databases"] = "healthy"
    except Exception as err:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "error": f"Database check failed: {err}"},
        )

    # Check storage filesystem readiness
    try:
        models_dir = Path(__file__).resolve().parent.parent / "data" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        probe_file = models_dir / f".ready_probe_{uuid.uuid4().hex}.tmp"
        probe_file.write_text("ready", encoding="utf-8")
        probe_file.unlink(missing_ok=True)
        checks["model_storage"] = "writable"
    except Exception as err:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "error": f"Storage check failed: {err}"},
        )

    return {"status": "ready", "checks": checks}


# Register Application Routers
app.include_router(data_router)
app.include_router(experiment_router)
app.include_router(model_router)
app.include_router(tuning_router)
app.include_router(feature_router)
app.include_router(report_router)
app.include_router(auth_router)
app.include_router(monitoring_router)
app.include_router(eda_router)
app.include_router(assistant_router)
