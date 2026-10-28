# server/tests/test_auth_and_security.py
# Phase 7: Verification of Authentication, Role-based Authorization & Token Revocation

import os
import sys
import unittest
from unittest.mock import patch

# Add server root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient

from app.main import app
from app.config import settings
from app.services.auth_service import (
    hash_password,
    verify_password,
    create_access_token,
    decode_access_token,
    register_user,
    authenticate_user,
    revoke_token,
    is_token_revoked,
    init_auth_db
)


class TestAuthAndSecurity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_auth_db()
        cls.client = TestClient(app)

    def test_01_direct_password_hashing(self):
        """Verify bcrypt password hashing and verification."""
        raw = "SecureSecretPassword123!"
        hashed = hash_password(raw)
        self.assertNotEqual(raw, hashed)
        self.assertTrue(hashed.startswith("$2b$") or hashed.startswith("$2a$"))
        self.assertTrue(verify_password(raw, hashed))
        self.assertFalse(verify_password("WrongPassword!", hashed))

    def test_02_jwt_token_creation_and_validation(self):
        """Verify JWT generation, claims parsing, and expiration handling."""
        token = create_access_token({"sub": "alice_ops", "role": "admin"})
        self.assertIsInstance(token, str)
        self.assertGreater(len(token), 20)

        payload = decode_access_token(token)
        self.assertIsNotNone(payload)
        self.assertEqual(payload.get("sub"), "alice_ops")
        self.assertEqual(payload.get("role"), "admin")
        self.assertIn("exp", payload)

    def test_03_user_registration_and_login_flow(self):
        """Verify user registration, duplicate prevention, and login validation."""
        unique_user = f"user_{abs(hash(self)) % 100000}"
        unique_pass = "ComplexPass456!"

        # Register
        reg_res = self.client.post("/auth/register", json={
            "username": unique_user,
            "password": unique_pass,
            "role": "user"
        })
        self.assertEqual(reg_res.status_code, 200)
        data = reg_res.json()
        self.assertEqual(data["username"], unique_user)
        self.assertEqual(data["role"], "user")

        # Duplicate username should be rejected with 409
        dup_res = self.client.post("/auth/register", json={
            "username": unique_user,
            "password": unique_pass
        })
        self.assertEqual(dup_res.status_code, 409)

        # Login with correct credentials
        login_res = self.client.post("/auth/login", json={
            "username": unique_user,
            "password": unique_pass
        })
        self.assertEqual(login_res.status_code, 200)
        login_data = login_res.json()
        self.assertIn("access_token", login_data)
        token = login_data["access_token"]

        # Login with invalid password
        bad_login = self.client.post("/auth/login", json={
            "username": unique_user,
            "password": "WrongPassword999"
        })
        self.assertEqual(bad_login.status_code, 401)

        # Test token revocation on logout
        self.assertFalse(is_token_revoked(token))
        logout_res = self.client.post("/auth/logout", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(logout_res.status_code, 200)
        self.assertTrue(is_token_revoked(token))

    def test_04_auth_status_endpoint(self):
        """Verify /auth/status endpoint reports enforcement mode."""
        res = self.client.get("/auth/status")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("auth_enabled", data)
        self.assertIn("token_algorithm", data)

    def test_05_strict_auth_enforcement(self):
        """Verify that when AUTH_ENABLED=True, endpoints strictly validate JWT and reject unauthenticated or revoked tokens."""
        unique_user = f"sec_user_{abs(hash(self)) % 100000}"
        unique_pass = "SecureTokenPass789!"
        reg_res = self.client.post("/auth/register", json={
            "username": unique_user,
            "password": unique_pass,
            "role": "admin"
        })
        self.assertEqual(reg_res.status_code, 200)

        login_res = self.client.post("/auth/login", json={
            "username": unique_user,
            "password": unique_pass
        })
        self.assertEqual(login_res.status_code, 200)
        valid_token = login_res.json()["access_token"]

        with patch.object(settings, "AUTH_ENABLED", True):
            # Missing token -> 401
            res_missing = self.client.get("/auth/me")
            self.assertEqual(res_missing.status_code, 401)

            # Invalid token -> 401
            res_invalid = self.client.get("/auth/me", headers={"Authorization": "Bearer invalid.jwt.token"})
            self.assertEqual(res_invalid.status_code, 401)

            # Valid token -> 200
            res_valid = self.client.get("/auth/me", headers={"Authorization": f"Bearer {valid_token}"})
            self.assertEqual(res_valid.status_code, 200)
            self.assertEqual(res_valid.json()["username"], unique_user)

            # Revoke token
            revoke_token(valid_token)

            # Revoked token -> 401
            res_revoked = self.client.get("/auth/me", headers={"Authorization": f"Bearer {valid_token}"})
            self.assertEqual(res_revoked.status_code, 401)


if __name__ == "__main__":
    unittest.main()
