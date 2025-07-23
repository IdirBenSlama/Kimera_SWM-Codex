"""
Security Integration for KIMERA System
Integrates all security components
Phase 4, Weeks 12-13: Security Hardening
"""

import logging
from typing import Optional, Any

from fastapi import FastAPI

from backend.config import get_settings
from .request_hardening import security_middleware
from .authentication import auth_manager, get_current_user, RoleChecker, get_user
from .sql_injection_prevention import SafeQueryBuilder
from fastapi import HTTPException
from datetime import datetime

logger = logging.getLogger(__name__)


class SecurityManager:
    """
    Central manager for all security components
    """
    
    def __init__(self):
        self.settings = get_settings()
        self._initialized = False
        
        logger.info("SecurityManager created")
    
    def initialize(self, app: FastAPI):
        """
        Initialize all security components and integrate with the FastAPI app
        
        Args:
            app: The FastAPI application instance
        """
        if self._initialized:
            logger.warning("SecurityManager already initialized")
            return
        
        # Add security middleware
        if self.settings.security.rate_limit_enabled:
            app.middleware("http")(security_middleware)
            logger.info("Security middleware added")
        
        # Add authentication routes (e.g., /token)
        self._add_auth_routes(app)
        
        self._initialized = True
        logger.info("SecurityManager fully initialized")
    
    def _add_auth_routes(self, app: FastAPI):
        """
        Add authentication-related routes to the app
        """
        from fastapi.security import OAuth2PasswordRequestForm
        from fastapi import Depends
        
        @app.post("/token")
        async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
            user = get_user(form_data.username)
            if not user or not auth_manager.verify_password(form_data.password, user.hashed_password):
                raise HTTPException(
                    status_code=401,
                    detail="Incorrect username or password",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            access_token = auth_manager.create_access_token(
                data={"sub": user.username, "scopes": form_data.scopes}
            )
            return {"access_token": access_token, "token_type": "bearer"}


# Global security manager
_security_manager: Optional[SecurityManager] = None


def get_security_manager() -> SecurityManager:
    """
    Get global security manager instance
    """
    global _security_manager
    if _security_manager is None:
        _security_manager = SecurityManager()
    return _security_manager


# Export key security dependencies for use in endpoints
__all__ = [
    "get_security_manager",
    "get_current_user",
    "RoleChecker",
    "SafeQueryBuilder"
]