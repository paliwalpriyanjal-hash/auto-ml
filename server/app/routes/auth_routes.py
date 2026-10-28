# server/app/routes/auth_routes.py
"""
Authentication and session management API routes.
"""

from typing import Any, Dict, Optional
from fastapi import APIRouter, Body, Depends, Header, HTTPException

from app.config import settings
from app.services.auth_service import (
    authenticate_user,
    create_access_token,
    get_current_user,
    register_user,
    revoke_token,
)

router = APIRouter(prefix="/auth", tags=["Authentication & Access Control"])


@router.get("/status")
def auth_status():
    """Returns whether authentication is enforced in the current environment."""
    return {
        "auth_enabled": settings.AUTH_ENABLED,
        "token_algorithm": settings.JWT_ALGORITHM,
        "app_version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
    }


@router.post("/register")
def register_endpoint(
    payload: Dict[str, Any] = Body(
        ...,
        example={"username": "data_scientist", "password": "SecurePassword123!"},
    ),
):
    """Register a new user account."""
    username = payload.get("username", "")
    password = payload.get("password", "")
    role = payload.get("role", "user")

    if not username or not password:
        raise HTTPException(status_code=400, detail="Username and password are required.")

    return register_user(username=username, password=password, role=role)


@router.post("/login")
def login_endpoint(
    payload: Dict[str, Any] = Body(
        ...,
        example={"username": "admin", "password": "AdminPassword123!"},
    ),
):
    """Authenticate credentials and issue a signed JWT Bearer access token."""
    username = payload.get("username", "")
    password = payload.get("password", "")

    if not username or not password:
        raise HTTPException(status_code=400, detail="Username and password are required.")

    user = authenticate_user(username=username, password=password)
    access_token = create_access_token(
        data={"sub": user["username"], "role": user["role"], "id": user["id"]}
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in_hours": settings.JWT_EXPIRATION_HOURS,
        "user": user,
    }


@router.get("/me")
def get_authenticated_user_profile(
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Retrieve profile and role information for the active session."""
    return current_user


@router.post("/logout")
def logout_endpoint(
    authorization: Optional[str] = Header(None),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Revoke active JWT Bearer token and terminate session."""
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ", 1)[1].strip()
        revoke_token(token)
    return {"message": "Session logged out and token successfully revoked."}
