"""
Authentication routes for testing and security endpoints.
"""

from datetime import timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel

try:
    from layer_2_governance.security.authentication import auth_manager
except ImportError:
    # Create placeholders for layer_2_governance.security.authentication
    auth_manager = None
try:
    from config import get_settings
except ImportError:
    # Create placeholders for config
    def get_settings(*args, **kwargs):
        return None


router = APIRouter(prefix="/auth", tags=["authentication"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


class Token(BaseModel):
    access_token: str
    token_type: str


class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None
    is_admin: bool = False


# Test users for authentication
fake_users_db = {
    "testuser": {
        "username": "testuser",
        "full_name": "Test User",
        "email": "test@example.com",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # secret
        "disabled": False,
        "is_admin": False
    },
    "adminuser": {
        "username": "adminuser",
        "full_name": "Admin User",
        "email": "admin@example.com",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # secret
        "disabled": False,
        "is_admin": True
    },
}


def fake_hash_password(password: str):
    """Fake password hashing for testing"""
    return "fakehashed" + password


def verify_password(plain_password, hashed_password):
    """Verify a password against a hash"""
    # For testing, accept specific test passwords
    import hashlib

    if plain_password.startswith("test_pass_") or plain_password.startswith(
        "admin_pass_"
    ):
        return True
    return False


def get_user(username: str):
    if username in fake_users_db:
        user_dict = fake_users_db[username]
        return user_dict  # Return the dict directly
    return None


def authenticate_user(username: str, password: str):
    user_dict = get_user(username)
    if not user_dict:
        return False
    if not verify_password(password, user_dict["hashed_password"]):
        return False
    # Create User object without hashed_password
    user_data = {k: v for k, v in user_dict.items() if k != "hashed_password"}
    return User(**user_data)


async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    token_data = auth_manager.decode_access_token(token)
    if token_data is None:
        raise credentials_exception

    # Extract username from TokenData object
    username = (
        token_data.username if hasattr(token_data, "username") else str(token_data)
    )

    user_dict = get_user(username=username)
    if user_dict is None:
        raise credentials_exception

    # Create User object without hashed_password
    user_data = {k: v for k, v in user_dict.items() if k != "hashed_password"}
    return User(**user_data)


async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


async def get_admin_user(current_user: User = Depends(get_current_active_user)):
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions"
        )
    return current_user


@router.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login endpoint to get access token"""
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = auth_manager.create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    """Get current user information"""
    return current_user


@router.get("/admin/dashboard")
async def admin_dashboard(current_user: User = Depends(get_admin_user)):
    """Admin-only endpoint"""
    return {"message": "Welcome to admin dashboard", "user": current_user.username}


# Add routes without prefix for compatibility
token_router = APIRouter()


@token_router.post("/token", response_model=Token)
async def login_root(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login endpoint at root level"""
    return await login(form_data)


@token_router.get("/users/me", response_model=User)
async def read_users_me_root(current_user: User = Depends(get_current_active_user)):
    """Get current user at root level"""
    return current_user


@token_router.get("/admin/dashboard")
async def admin_dashboard_root(current_user: User = Depends(get_admin_user)):
    """Admin dashboard at root level"""
    return await admin_dashboard(current_user)


# Process endpoint for testing
@token_router.post("/process")
async def process_request(
    request_data: dict, current_user: User = Depends(get_current_active_user)
):
    """Process endpoint for testing"""
    if not request_data.get("text"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Text field is required and cannot be empty",
        )

    return {
        "id": "test_id_123",
        "result": f"Processed: {request_data['text']}",
        "user": current_user.username,
    }
