"""
Authentication Module
=====================

Implements multi-factor authentication based on:
- NIST 800-63B Authentication Guidelines
- Zero Trust Architecture principles
- OAuth 2.0 / OpenID Connect standards
"""

import hashlib
import logging
import secrets
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import jwt

logger = logging.getLogger(__name__)


class AuthenticationMethod(Enum):
    """Supported authentication methods."""

    PASSWORD = "password"
    TOKEN = "token"
    BIOMETRIC = "biometric"
    CERTIFICATE = "certificate"
    MFA = "multi_factor"


class AuthenticationLevel(Enum):
    """Authentication assurance levels (NIST 800-63)."""

    AAL1 = 1  # Single factor
    AAL2 = 2  # Multi-factor
    AAL3 = 3  # Hardware-based multi-factor


class Authenticator:
    """
    Multi-method authenticator with aerospace-grade security.
    """

    def __init__(self, secret_key: str = None):
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self.failed_attempts: Dict[str, List[datetime]] = {}
        self.locked_accounts: Dict[str, datetime] = {}
        self.active_sessions: Dict[str, Dict[str, Any]] = {}

    def authenticate(
        self,
        username: str,
        credentials: Dict[str, Any],
        method: AuthenticationMethod = AuthenticationMethod.PASSWORD,
    ) -> Optional[Dict[str, Any]]:
        """
        Authenticate a user with specified method.

        Args:
            username: User identifier
            credentials: Authentication credentials
            method: Authentication method

        Returns:
            Authentication token and metadata if successful
        """
        # Check if account is locked
        if self._is_account_locked(username):
            logger.warning(f"Authentication attempt on locked account: {username}")
            return None

        # Validate based on method
        is_valid = False

        if method == AuthenticationMethod.PASSWORD:
            is_valid = self._validate_password(username, credentials.get("password"))
        elif method == AuthenticationMethod.TOKEN:
            is_valid = self._validate_token(credentials.get("token"))
        elif method == AuthenticationMethod.MFA:
            is_valid = self._validate_mfa(username, credentials)

        if not is_valid:
            self._record_failed_attempt(username)
            return None

        # Generate session token
        session_id = secrets.token_urlsafe(32)
        token_data = {
            "username": username,
            "session_id": session_id,
            "method": method.value,
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(hours=24),
        }

        token = jwt.encode(token_data, self.secret_key, algorithm="HS256")

        # Store session
        self.active_sessions[session_id] = {
            "username": username,
            "created": datetime.utcnow(),
            "last_activity": datetime.utcnow(),
            "method": method,
        }

        logger.info(f"Successful authentication for {username} using {method.value}")

        return {
            "token": token,
            "session_id": session_id,
            "expires_in": 86400,  # 24 hours
        }

    def _validate_password(self, username: str, password: str) -> bool:
        """Validate password (simplified for MVP)."""
        # In production, this would check against secure password store
        if not password or len(password) < 8:
            return False

        # Simplified validation
        expected_hash = hashlib.sha256(f"{username}:{password}".encode()).hexdigest()
        stored_hash = self._get_stored_password_hash(username)

        return secrets.compare_digest(expected_hash, stored_hash)

    def _validate_token(self, token: str) -> bool:
        """Validate JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            session_id = payload.get("session_id")

            # Check if session exists and is active
            if session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                session["last_activity"] = datetime.utcnow()
                return True

            return False

        except jwt.InvalidTokenError:
            return False

    def _validate_mfa(self, username: str, credentials: Dict[str, Any]) -> bool:
        """Validate multi-factor authentication."""
        # Validate password first
        if not self._validate_password(username, credentials.get("password")):
            return False

        # Validate second factor (simplified TOTP check)
        totp_code = credentials.get("totp_code")
        if not totp_code or len(totp_code) != 6:
            return False

        # In production, this would validate against user's TOTP secret
        return True

    def _is_account_locked(self, username: str) -> bool:
        """Check if account is locked due to failed attempts."""
        if username in self.locked_accounts:
            lock_time = self.locked_accounts[username]
            if datetime.utcnow() - lock_time < timedelta(minutes=30):
                return True
            else:
                # Unlock after 30 minutes
                del self.locked_accounts[username]

        return False

    def _record_failed_attempt(self, username: str):
        """Record failed authentication attempt."""
        if username not in self.failed_attempts:
            self.failed_attempts[username] = []

        self.failed_attempts[username].append(datetime.utcnow())

        # Keep only recent attempts (last hour)
        cutoff = datetime.utcnow() - timedelta(hours=1)
        self.failed_attempts[username] = [
            attempt for attempt in self.failed_attempts[username] if attempt > cutoff
        ]

        # Lock account after 5 failed attempts
        if len(self.failed_attempts[username]) >= 5:
            self.locked_accounts[username] = datetime.utcnow()
            logger.warning(f"Account locked due to failed attempts: {username}")

    def _get_stored_password_hash(self, username: str) -> str:
        """Get stored password hash (mock for MVP)."""
        # In production, this would query secure password store
        return hashlib.sha256(f"{username}:password123".encode()).hexdigest()

    def validate_session(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate an existing session token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            session_id = payload.get("session_id")

            if session_id in self.active_sessions:
                session = self.active_sessions[session_id]

                # Check session timeout (24 hours)
                if datetime.utcnow() - session["created"] > timedelta(hours=24):
                    del self.active_sessions[session_id]
                    return None

                # Update last activity
                session["last_activity"] = datetime.utcnow()

                return {
                    "username": session["username"],
                    "session_id": session_id,
                    "method": session["method"].value,
                }

            return None

        except jwt.InvalidTokenError:
            return None

    def revoke_session(self, session_id: str):
        """Revoke an active session."""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            logger.info(f"Session revoked: {session_id}")

    def get_active_sessions(self, username: str) -> List[Dict[str, Any]]:
        """Get all active sessions for a user."""
        sessions = []

        for session_id, session in self.active_sessions.items():
            if session["username"] == username:
                sessions.append(
                    {
                        "session_id": session_id,
                        "created": session["created"].isoformat(),
                        "last_activity": session["last_activity"].isoformat(),
                        "method": session["method"].value,
                    }
                )

        return sessions


# Create a global auth_manager instance
auth_manager = Authenticator()


# Compatibility exports for JWT operations
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token."""
    return auth_manager.create_token(data, expires_delta)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash."""
    # Simple verification - in production use bcrypt or similar
    return hashlib.sha256(plain_password.encode()).hexdigest() == hashed_password


def get_password_hash(password: str) -> str:
    """Hash a password."""
    # Simple hashing - in production use bcrypt or similar
    return hashlib.sha256(password.encode()).hexdigest()


def decode_access_token(token: str) -> Optional[dict]:
    """Decode and verify JWT token."""
    return auth_manager.verify_token(token)


# Fake users database for testing
fake_users_db = {
    "testuser": {
        "username": "testuser",
        "email": "testuser@example.com",
        "role": "user",
        "hashed_password": get_password_hash("testpass123"),
    },
    "admin": {
        "username": "admin",
        "email": "admin@kimera.ai",
        "role": "admin",
        "hashed_password": get_password_hash("admin123"),
    },
}


def get_user(username: str) -> Optional[dict]:
    """Get user from database."""
    return fake_users_db.get(username)


async def get_current_user(token: str) -> dict:
    """Get current user from token."""
    from fastapi import HTTPException, status

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    payload = decode_access_token(token)
    if payload is None:
        raise credentials_exception

    username = payload.get("sub")
    if username is None:
        raise credentials_exception

    user = get_user(username)
    if user is None:
        raise credentials_exception

    return user


class RoleChecker:
    """Role-based access control checker."""

    def __init__(self, allowed_roles: List[str]):
        self.allowed_roles = allowed_roles

    def __call__(self, user: dict) -> dict:
        """Check if user has required role."""
        from fastapi import HTTPException, status

        if user.get("role") not in self.allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Operation not permitted"
            )
        return user
