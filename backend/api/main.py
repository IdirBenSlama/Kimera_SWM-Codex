"""
Main API module for Kimera SWM

This module provides the FastAPI application for the Kimera SWM system.
It initializes the database connection and creates necessary endpoints.
"""

import logging
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time
import os
from typing import Dict, Any, List, Optional

# Configure logging
logger = logging.getLogger(__name__)
logger.info("--- STARTING API MAIN ---")
logger.info("Importing core components...")

# Import core components
from ..core.kimera_system import KimeraSystem, kimera_singleton
from ..utils.memory_manager import MemoryManager
from ..monitoring.prometheus_metrics import KimeraPrometheusMetrics

logger.info("Core components imported.")
logger.info("Importing routers...")

# Import routers
from .routers import (
    # health_router,  # Missing
    # status_router,  # Missing
    # cognitive_field_router,  # Missing
    geoid_scar_router,
    vault_router,
    contradiction_router,
    metrics_router,
    omnidimensional_router
)

logger.info("Routers imported.")

# Initialize system components
memory_manager = MemoryManager()
metrics = KimeraPrometheusMetrics()
# Use kimera_singleton instead of creating a new instance
# kimera_system = KimeraSystem()  # REMOVED

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan event handler.
    
    This function is called when the FastAPI application starts and stops.
    It initializes the database connection and creates necessary tables.
    """
    # Startup: initialize the database and system components
    logger.info("Initializing system components...")
    
    # Initialize vault module (which initializes the database)
    from ..vault import initialize_vault
    if not initialize_vault():
        logger.error("Failed to initialize vault module")
        raise RuntimeError("Failed to initialize vault module")
    
    # Initialize kimera_singleton (the same instance used by all endpoints)
    try:
        kimera_singleton.initialize()
        logger.info("Kimera singleton initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Kimera singleton: {e}")
        raise RuntimeError(f"Failed to initialize Kimera singleton: {e}")
    
    logger.info("System components initialized successfully.")
    
    yield  # FastAPI runs the application here
    
    # Shutdown: clean up resources
    logger.info("--- KIMERA LIFESPAN SHUTDOWN ---")
    try:
        await kimera_singleton.shutdown()
        logger.info("Kimera singleton shutdown complete.")
    except Exception as e:
        logger.error(f"Error during Kimera singleton shutdown: {e}")

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        FastAPI: Configured FastAPI application
    """
    app = FastAPI(
        title="Kimera SWM API",
        description="Kinetic Intelligence for Multidimensional Emergent Reasoning and Analysis",
        version="0.1.0",
        lifespan=lifespan
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, restrict this to specific origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add exception handler for graceful error handling
    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "error": str(exc)}
        )
    
    # Include routers
    # app.include_router(health_router.router)
    # app.include_router(status_router.router)
    # app.include_router(cognitive_field_router.router)
    app.include_router(geoid_scar_router.router)
    app.include_router(vault_router.router)
    app.include_router(contradiction_router.router)
    app.include_router(metrics_router.router)
    app.include_router(omnidimensional_router.router)
    
    return app