#!/usr/bin/env python
"""
Kimera SWM System Startup Script

This script initializes and starts the Kimera SWM system with optimized configuration.
It handles database connections, environment setup, and system initialization.
"""

import os
import sys
import logging
import uvicorn
from typing import Optional, Dict, Any
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("kimera_startup")

def setup_environment() -> Dict[str, Any]:
    """
    Set up environment variables and configuration.
    
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    # Load .env file if it exists
    env_path = Path(".env")
    if env_path.exists():
        logger.info("Loading environment from .env file")
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                key, value = line.split("=", 1)
                os.environ[key] = value
    
    # Set default DATABASE_URL if not provided
    if "DATABASE_URL" not in os.environ:
        logger.warning("DATABASE_URL not found in environment, using default configuration")
        os.environ["DATABASE_URL"] = "postgresql+psycopg2://kimera:kimera_secure_pass_2025@localhost:5432/kimera_swm"
    
    # Print the database URL (with password masked)
    db_url = os.environ["DATABASE_URL"]
    masked_url = mask_password(db_url)
    print(f"Database URL: {masked_url}")
    
    return {
        "database_url": os.environ["DATABASE_URL"],
        "environment": os.environ.get("ENVIRONMENT", "development"),
        "log_level": os.environ.get("LOG_LEVEL", "INFO"),
        "debug": os.environ.get("DEBUG", "False").lower() in ("true", "1", "t"),
        "api_host": os.environ.get("API_HOST", "0.0.0.0"),
        "api_port": int(os.environ.get("API_PORT", "8000")),
        "api_workers": int(os.environ.get("API_WORKERS", "1")),
    }

def mask_password(url: str) -> str:
    """
    Mask password in database URL for secure logging.
    
    Args:
        url (str): Database URL with password
        
    Returns:
        str: Database URL with masked password
    """
    if "://" not in url:
        return url
    
    parts = url.split("://", 1)
    if "@" not in parts[1]:
        return url
    
    auth, rest = parts[1].split("@", 1)
    if ":" not in auth:
        return url
    
    user, _ = auth.split(":", 1)
    return f"{parts[0]}://{user}:****@{rest}"

def verify_database_connection(database_url: str) -> bool:
    """
    Verify database connection before starting the system.
    
    Args:
        database_url (str): Database connection URL
        
    Returns:
        bool: True if connection successful, False otherwise
    """
    try:
        from sqlalchemy import create_engine, text
        
        # Create engine with minimal connection pool
        engine = create_engine(
            database_url,
            connect_args={
                "connect_timeout": 5,
                "application_name": "Kimera SWM Startup"
            },
            pool_size=1,
            max_overflow=0,
            pool_timeout=5
        )
        
        # Test connection
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version()")).scalar()
            logger.info(f"Database connection successful: {result}")
            
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False

def check_system_requirements() -> bool:
    """
    Check if system meets minimum requirements.
    
    Returns:
        bool: True if requirements met, False otherwise
    """
    import platform
    import psutil
    
    # Check Python version
    python_version = tuple(map(int, platform.python_version_tuple()))
    if python_version < (3, 9, 0):
        logger.warning(f"Python version {platform.python_version()} is below recommended 3.9+")
    
    # Check available memory
    available_memory_gb = psutil.virtual_memory().total / (1024 ** 3)
    if available_memory_gb < 8:
        logger.warning(f"Available memory ({available_memory_gb:.1f} GB) is below recommended 8GB")
    
    # Check available disk space
    disk_space_gb = psutil.disk_usage(os.getcwd()).free / (1024 ** 3)
    if disk_space_gb < 5:
        logger.warning(f"Available disk space ({disk_space_gb:.1f} GB) is below recommended 5GB")
    
    # Check for GPU if CUDA is available
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"CUDA device available: {device_name}")
        else:
            logger.warning("CUDA device not available, using CPU only")
    except ImportError:
        logger.warning("PyTorch not installed, GPU acceleration will not be available")
    
    return True

def main():
    """Main entry point for Kimera SWM system startup."""
    print("Starting Kimera SWM System...")
    
    # Set up environment
    config = setup_environment()
    
    # Check system requirements
    if not check_system_requirements():
        logger.error("System requirements check failed")
        sys.exit(1)
    
    # Verify database connection
    if not verify_database_connection(config["database_url"]):
        logger.error("Database connection verification failed")
        sys.exit(1)
    
    # Start the API server
    uvicorn.run(
        "backend.api.main:create_app",
        host=config["api_host"],
        port=config["api_port"],
        workers=config["api_workers"],
        log_level=config["log_level"].lower(),
        reload=False
    )

if __name__ == "__main__":
    main() 