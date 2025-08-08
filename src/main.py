#!/usr/bin/env python3
"""
KIMERA SWM Simplified Entry Point
=================================

Simplified entry point with proper syntax and basic functionality.
This is a working foundation that can be enhanced.

Author: Kimera SWM Autonomous Architect
Date: 2025-01-31
Version: 1.0.0 (CLEAN START)
Classification: WORKING FOUNDATION
"""

import asyncio
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

# Configure application-level logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment setup
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logger.warning("python-dotenv not available")

# Fix Python path for imports
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# FastAPI imports
try:
    import uvicorn
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
except ImportError as e:
    logger.error(f"FastAPI not available: {e}")
    raise

# Core system imports with error handling
try:
    from src.core.system.kimera_system_clean import (KimeraSystem, get_kimera_system,
                                                      kimera_singleton)
except ImportError as e:
    logger.warning(f"Core system import failed: {e}")
    kimera_singleton = None

try:
    from src.utils.kimera_logger import get_system_logger
except ImportError as e:
    logger.warning(f"Kimera logger import failed: {e}")
    get_system_logger = None
except SyntaxError as e:
    logger.warning(f"Kimera logger syntax error: {e}")
    get_system_logger = None

# Simple initialization modes
class InitializationMode:
    MINIMAL = "minimal"
    SAFE = "safe"
    FULL = "full"

# Get initialization mode from environment
INIT_MODE = os.getenv("KIMERA_INIT_MODE", InitializationMode.MINIMAL)

# Global state
kimera_system_instance: Optional[Any] = None


@asynccontextmanager
async def app_lifespan(app: FastAPI):
    """Simple lifespan manager with proper error handling."""
    global kimera_system_instance

    logger.info("🚀 KIMERA SWM Starting...")
    logger.info(f"📋 Initialization Mode: {INIT_MODE}")

    startup_start = time.time()

    try:
        # Initialize based on mode
        if INIT_MODE == InitializationMode.MINIMAL:
            await initialize_minimal(app)
        elif INIT_MODE == InitializationMode.SAFE:
            await initialize_safe(app)
        elif INIT_MODE == InitializationMode.FULL:
            await initialize_full(app)
        else:
            logger.warning(f"Unknown mode {INIT_MODE}, defaulting to minimal")
            await initialize_minimal(app)

        startup_time = time.time() - startup_start
        logger.info(f"✅ KIMERA SWM Started successfully in {startup_time:.2f}s")

        yield  # FastAPI runs here

    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    finally:
        # Cleanup
        logger.info("🔻 KIMERA SWM Shutting down...")
        try:
            if kimera_system_instance and hasattr(kimera_system_instance, 'shutdown'):
                await kimera_system_instance.shutdown()
        except Exception as e:
            logger.error(f"❌ Shutdown error: {e}")
        logger.info("✅ KIMERA SWM Shutdown complete")


async def initialize_minimal(app: FastAPI):
    """Minimal initialization - just enough to run."""
    global kimera_system_instance

    logger.info("🔧 Minimal initialization...")

    # Basic API state
    app.state.initialization_mode = InitializationMode.MINIMAL
    app.state.startup_time = time.time()

    # Try to initialize core system if available
    if kimera_singleton:
        try:
            kimera_system_instance = kimera_singleton
            logger.info("✅ Core system available")
        except Exception as e:
            logger.warning(f"⚠️ Core system initialization failed: {e}")

    logger.info("✅ Minimal initialization complete")


async def initialize_safe(app: FastAPI):
    """Safe initialization with error tolerance."""
    global kimera_system_instance

    logger.info("🛡️ Safe initialization...")

    # Start with minimal
    await initialize_minimal(app)

    # Add safe enhancements
    try:
        if kimera_system_instance and hasattr(kimera_system_instance, 'initialize'):
            kimera_system_instance.initialize()
            logger.info("✅ Core system initialized")
    except Exception as e:
        logger.warning(f"⚠️ Core system initialization failed (continuing): {e}")

    app.state.initialization_mode = InitializationMode.SAFE
    logger.info("✅ Safe initialization complete")


async def initialize_full(app: FastAPI):
    """Full initialization with all features."""
    global kimera_system_instance

    logger.info("🚀 Full initialization...")

    # Start with safe
    await initialize_safe(app)

    # Add full features
    try:
        # Database initialization
        try:
            from src.config.database_config import initialize_database
            db_success = initialize_database()
            if db_success:
                logger.info("✅ Database initialized")
            else:
                logger.warning("⚠️ Database initialization failed")
        except ImportError:
            logger.warning("⚠️ Database module not available")

        # Monitoring initialization
        try:
            from src.monitoring.kimera_prometheus_metrics import initialize_background_collection
            initialize_background_collection()
            logger.info("✅ Monitoring initialized")
        except ImportError:
            logger.warning("⚠️ Monitoring module not available")

    except Exception as e:
        logger.warning(f"⚠️ Full initialization partial failure: {e}")

    app.state.initialization_mode = InitializationMode.FULL
    logger.info("✅ Full initialization complete")


# Create FastAPI application
app = FastAPI(
    title="KIMERA SWM System",
    description="KIMERA SWM - Simplified Working Foundation",
    version="1.0.0",
    lifespan=app_lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "KIMERA SWM System - Simplified Foundation",
        "status": "running",
        "mode": INIT_MODE,
        "timestamp": time.time()
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "mode": getattr(app.state, 'initialization_mode', 'unknown'),
        "uptime": time.time() - getattr(app.state, 'startup_time', time.time()),
        "kimera_system": kimera_system_instance is not None
    }


@app.get("/status")
async def system_status():
    """System status endpoint."""
    status = {
        "mode": getattr(app.state, 'initialization_mode', 'unknown'),
        "uptime": time.time() - getattr(app.state, 'startup_time', time.time()),
        "components": {
            "kimera_system": kimera_system_instance is not None,
            "database": False,  # TODO: Add database status check
            "monitoring": False,  # TODO: Add monitoring status check
        }
    }

    if kimera_system_instance and hasattr(kimera_system_instance, 'get_status'):
        try:
            status["kimera_status"] = kimera_system_instance.get_status()
        except Exception as e:
            status["kimera_error"] = str(e)

    return status


def main():
    """Main entry point."""
    host = os.getenv("KIMERA_HOST", "127.0.0.1")
    port = int(os.getenv("KIMERA_PORT", "8000"))

    logger.info(f"🌐 Starting KIMERA SWM on {host}:{port}")
    logger.info(f"📋 Mode: {INIT_MODE}")

    uvicorn.run(
        "src.main:app",
        host=host,
        port=port,
        log_level="info",
        reload=False  # Disable reload for production
    )


if __name__ == "__main__":
    main()
