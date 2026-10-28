# server/app/services/auth_service.py
"""
Authentication and Authorization service for AutoVeyra platform.
Provides secure password hashing (bcrypt), JWT bearer tokens (python-jose),
and SQLite-backed session revocation and user management.
"""

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import bcrypt
from fastapi import Depends, Header, HTTPException, Request
from jose import JWTError, jwt

from app.config import settings

SERVER_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = SERVER_DIR / "data"
AUTH_DB_PATH = DATA_DIR / "auth.db"


@contextmanager
def _connect():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(AUTH_DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def init_auth_db() -> None:
    """Initializes authentication tables and bootstraps default admin if database is empty."""
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                hashed_password TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                created_at TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS revoked_tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                token TEXT UNIQUE NOT NULL,
                revoked_at TEXT NOT NULL
            )
            """
        )

        # Bootstrap admin if no users exist
        cursor = conn.execute("SELECT COUNT(*) FROM users")
        if cursor.fetchone()[0] == 0:
            default_admin_user = os.getenv("AUTOVEYRA_ADMIN_USERNAME", "admin")
            default_admin_pass = os.getenv("AUTOVEYRA_ADMIN_PASSWORD", "AdminPassword123!")
            salt = bcrypt.gensalt()
            hashed = bcrypt.hashpw(default_admin_pass.encode("utf-8"), salt).decode("utf-8")
            now_iso = datetime.now(timezone.utc).isoformat()
            conn.execute(
                """
                INSERT INTO users (username, hashed_password, role, created_at, is_active)
                VALUES (?, ?, 'admin', ?, 1)
                """,
                (default_admin_user, hashed, now_iso),
            )


def hash_password(plain_password: str) -> str:
    """Safely hash password using direct bcrypt with salt."""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(plain_password.encode("utf-8"), salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify plain password against stored bcrypt hash."""
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
    except Exception:
        return False


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Generates signed JWT access token with payload and expiration."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(hours=settings.JWT_EXPIRATION_HOURS)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def decode_access_token(token: str) -> Dict[str, Any]:
    """Decodes and validates a JWT token, raising JWTError if invalid."""
    return jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])


def is_token_revoked(token: str) -> bool:
    """Checks whether token has been explicitly revoked upon logout."""
    init_auth_db()
    with _connect() as conn:
        cursor = conn.execute("SELECT 1 FROM revoked_tokens WHERE token = ?", (token,))
        return cursor.fetchone() is not None


def revoke_token(token: str) -> None:
    """Stores token in revocation list."""
    init_auth_db()
    now_iso = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        try:
            conn.execute(
                "INSERT INTO revoked_tokens (token, revoked_at) VALUES (?, ?)",
                (token, now_iso),
            )
        except sqlite3.IntegrityError:
            pass  # Already revoked


def register_user(username: str, password: str, role: str = "user") -> Dict[str, Any]:
    """Registers a new user account with validated credentials."""
    init_auth_db()
    uname = username.strip().lower()
    if len(uname) < 3 or len(uname) > 32:
        raise HTTPException(
            status_code=400,
            detail="Username must be between 3 and 32 characters.",
        )
    if not uname.isalnum() and "_" not in uname:
        raise HTTPException(
            status_code=400,
            detail="Username may only contain alphanumeric characters and underscores.",
        )
    if len(password) < 6:
        raise HTTPException(
            status_code=400,
            detail="Password must be at least 6 characters long.",
        )

    assigned_role = "admin" if role.lower() == "admin" else "user"
    hashed = hash_password(password)
    now_iso = datetime.now(timezone.utc).isoformat()

    with _connect() as conn:
        cursor = conn.execute("SELECT 1 FROM users WHERE username = ?", (uname,))
        if cursor.fetchone():
            raise HTTPException(
                status_code=409,
                detail=f"Username '{uname}' is already registered.",
            )
        cursor = conn.execute(
            """
            INSERT INTO users (username, hashed_password, role, created_at, is_active)
            VALUES (?, ?, ?, ?, 1)
            """,
            (uname, hashed, assigned_role, now_iso),
        )
        user_id = cursor.lastrowid

    return {
        "id": user_id,
        "username": uname,
        "role": assigned_role,
        "created_at": now_iso,
    }


def authenticate_user(username: str, password: str) -> Dict[str, Any]:
    """Authenticates credentials against stored password hash."""
    init_auth_db()
    uname = username.strip().lower()

    with _connect() as conn:
        cursor = conn.execute("SELECT * FROM users WHERE username = ?", (uname,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(
                status_code=401,
                detail="Invalid username or password.",
            )
        user_dict = dict(row)

    if not user_dict.get("is_active"):
        raise HTTPException(
            status_code=403,
            detail="User account has been deactivated.",
        )

    if not verify_password(password, user_dict["hashed_password"]):
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password.",
        )

    return {
        "id": user_dict["id"],
        "username": user_dict["username"],
        "role": user_dict["role"],
        "created_at": user_dict["created_at"],
    }


def get_current_user(
    request: Request,
    authorization: Optional[str] = Header(None),
) -> Dict[str, Any]:
    """
    FastAPI dependency resolving the authenticated caller.
    If AUTOVEYRA_AUTH_ENABLED is false, returns a default guest local context.
    If true, strictly validates JWT Bearer authorization header and session revocation.
    """
    if not settings.AUTH_ENABLED:
        # Backward-compatible local mode
        return {"id": 1, "username": "local_admin", "role": "admin"}

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Provide 'Authorization: Bearer <token>'.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = authorization.split(" ", 1)[1].strip()

    if is_token_revoked(token):
        raise HTTPException(
            status_code=401,
            detail="Session has been revoked or logged out. Please sign in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET,
            algorithms=[settings.JWT_ALGORITHM],
        )
        username: str = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token payload.")
    except JWTError:
        raise HTTPException(
            status_code=401,
            detail="Could not validate credentials or token expired.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    init_auth_db()
    with _connect() as conn:
        cursor = conn.execute("SELECT id, username, role, is_active FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=401, detail="User account no longer exists.")
        user = dict(row)

    if not user.get("is_active"):
        raise HTTPException(status_code=403, detail="Account is deactivated.")

    return user


def require_admin(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """Dependency asserting administrator role authorization."""
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=403,
            detail="Administrator access required for this operation.",
        )
    return current_user
