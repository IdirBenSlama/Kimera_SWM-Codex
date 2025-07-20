"""
Authentication and Authorization for KIMERA System
Implements JWT-based authentication and role-based access control
Phase 4, Weeks 12-13: Security Hardening
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

from fastapi import Depends, HTTPException, Security
from fastapi.security import OAuth2PasswordBearer, SecurityScopes
from jose import JWTError, jwt
from pydantic import BaseModel, ValidationError
from passlib.context import CryptContext

from backend.config import get_settings

logger = logging.getLogger(__name__)


class TokenData(BaseModel):
    """Data model for JWT token payload"""
    username: str
    scopes: List[str] = []


class User(BaseModel):
    """User model"""
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None
    roles: List[str] = []


class UserInDB(User):
    """User model with hashed password"""
    hashed_password: str


class AuthManager:
    """
    Manages authentication and authorization
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.oauth2_scheme = OAuth2PasswordBearer(
            tokenUrl="/token",
            scopes={"read": "Read access", "write": "Write access"}
        )
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a plain password against a hashed password"""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Hash a plain password"""
        return self.pwd_context.hash(password)
    
    def create_access_token(
        self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create a JWT access token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + self.settings.security.jwt_expiration
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(
            to_encode, 
            self.settings.security.secret_key.get_secret_value(), 
            algorithm=self.settings.security.jwt_algorithm
        )
        return encoded_jwt
    
    def decode_access_token(self, token: str) -> Optional[TokenData]:
        """Decode a JWT access token"""
        try:
            payload = jwt.decode(
                token, 
                self.settings.security.secret_key.get_secret_value(), 
                algorithms=[self.settings.security.jwt_algorithm]
            )
            username: str = payload.get("sub")
            scopes = payload.get("scopes", [])
            if username is None:
                return None
            return TokenData(username=username, scopes=scopes)
        except (JWTError, ValidationError):
            return None


# Dummy user database for demonstration
# In a real application, this would be a database query
DUMMY_USERS_DB = {
    "testuser": {
        "username": "testuser",
        "full_name": "Test User",
        "email": "test@example.com",
        "hashed_password": AuthManager().get_password_hash("testpassword"),
        "disabled": False,
        "roles": ["user"]
    },
    "adminuser": {
        "username": "adminuser",
        "full_name": "Admin User",
        "email": "admin@example.com",
        "hashed_password": AuthManager().get_password_hash("adminpassword"),
        "disabled": False,
        "roles": ["user", "admin"]
    }
}


def get_user(username: str) -> Optional[UserInDB]:
    """Get user from the database"""
    if username in DUMMY_USERS_DB:
        user_dict = DUMMY_USERS_DB[username]
        return UserInDB(**user_dict)
    return None


# FastAPI dependencies for authentication and authorization

auth_manager = AuthManager()


async def get_current_user(
    security_scopes: SecurityScopes, token: str = Depends(auth_manager.oauth2_scheme)
) -> User:
    """Get the current user from the access token"""
    if security_scopes.scopes:
        authenticate_value = f'Bearer scope="{security_scopes.scope_str}"'
    else:
        authenticate_value = "Bearer"
    
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": authenticate_value},
    )
    
    token_data = auth_manager.decode_access_token(token)
    if token_data is None:
        raise credentials_exception
    
    user = get_user(token_data.username)
    if user is None or user.disabled:
        raise credentials_exception
    
    # Check scopes
    for scope in security_scopes.scopes:
        if scope not in token_data.scopes:
            raise HTTPException(
                status_code=403,
                detail="Not enough permissions",
                headers={"WWW-Authenticate": authenticate_value},
            )
    
    return user


class RoleChecker:
    """Dependency to check for required roles"""
    
    def __init__(self, required_roles: List[str]):
        self.required_roles = required_roles
    
    def __call__(self, current_user: User = Depends(get_current_user)):
        for role in self.required_roles:
            if role not in current_user.roles:
                raise HTTPException(
                    status_code=403,
                    detail=f"User does not have the required '{role}' role"
                )


# Example usage in an endpoint
# from fastapi import APIRouter
# router = APIRouter()
# 
# @router.get("/users/me")
# async def read_users_me(current_user: User = Depends(get_current_user)):
#     return current_user
# 
# @router.get("/admin/dashboard", dependencies=[Depends(RoleChecker(["admin"]))])
# async def read_admin_dashboard():
#     return {"message": "Welcome, admin!"}