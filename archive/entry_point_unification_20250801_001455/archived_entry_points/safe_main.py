"""
KIMERA Safe Main Application
===========================

A safer version of the main KIMERA application with:
- Progressive component loading
- Enhanced error handling
- Graceful degradation
- Memory-safe initialization
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import asyncio
import logging
import time
import os

# Setup logger
logger = logging.getLogger(__name__)

# Global system state
kimera_system = {
    'status': 'initializing',
    'components': {},
    'errors': []
}

@asynccontextmanager
async def safe_lifespan(app: FastAPI):
    """
    Safe lifespan manager with progressive loading and error handling
    """
    logger.info("🚀 KIMERA Safe Mode starting up...")
    
    # Initialize app state
    app.state.kimera_system = kimera_system
    app.state.loading_progress = 0
    
    try:
        # Step 1: Basic initialization
        logger.info("Step 1: Basic system initialization...")
        kimera_system['status'] = 'loading_basic'
        app.state.loading_progress = 10
        await asyncio.sleep(0.1)  # Allow other tasks
        
        # Step 2: GPU Foundation (with error handling)
        logger.info("Step 2: GPU Foundation initialization...")
        try:
            from src.utils.gpu_foundation import GPUFoundation
            gpu_foundation = GPUFoundation()
            kimera_system['components']['gpu_foundation'] = gpu_foundation
            app.state.gpu_foundation = gpu_foundation
            logger.info("✅ GPU Foundation initialized")
            app.state.loading_progress = 30
        except Exception as e:
            logger.warning(f"GPU Foundation failed: {e}")
            kimera_system['errors'].append(f"GPU Foundation: {e}")
            app.state.gpu_foundation = None
        
        await asyncio.sleep(0.1)
        
        # Step 3: Embedding Model (optional)
        logger.info("Step 3: Embedding model initialization...")
        try:
            from src.core.embedding_utils import initialize_embedding_model
            embedding_model = initialize_embedding_model()
            kimera_system['components']['embedding_model'] = embedding_model
            app.state.embedding_model = embedding_model
            logger.info("✅ Embedding model initialized")
            app.state.loading_progress = 60
        except Exception as e:
            logger.warning(f"Embedding model failed: {e}")
            kimera_system['errors'].append(f"Embedding model: {e}")
            app.state.embedding_model = None
        
        await asyncio.sleep(0.1)
        
        # Step 4: Basic engines (safe subset)
        logger.info("Step 4: Basic KIMERA engines...")
        try:
            # Only initialize critical, safe components
            from src.vault import get_vault_manager
            vault_manager = get_vault_manager()
            kimera_system['components']['vault_manager'] = vault_manager
            logger.info("✅ Vault Manager initialized")
            app.state.loading_progress = 80
        except Exception as e:
            logger.warning(f"Vault Manager failed: {e}")
            kimera_system['errors'].append(f"Vault Manager: {e}")
        
        await asyncio.sleep(0.1)
        
        # Step 5: Finalization
        logger.info("Step 5: Finalizing initialization...")
        kimera_system['status'] = 'operational'
        app.state.loading_progress = 100
        
        logger.info("✅ KIMERA Safe Mode initialization complete")
        logger.info(f"Components loaded: {len(kimera_system['components'])}")
        if kimera_system['errors']:
            logger.warning(f"Errors encountered: {len(kimera_system['errors'])}")
        
    except Exception as e:
        logger.critical(f"Critical error during safe initialization: {e}")
        kimera_system['status'] = 'error'
        kimera_system['errors'].append(f"Critical: {e}")
    
    yield
    
    # Cleanup
    logger.info("🛑 KIMERA Safe Mode shutting down")
    kimera_system['status'] = 'shutdown'
    app.state.gpu_foundation = None
    app.state.embedding_model = None

# Create FastAPI app
app = FastAPI(
    title="KIMERA SWM - Safe Mode",
    description="KIMERA with enhanced safety and progressive loading",
    version="0.1.0-safe",
    lifespan=safe_lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "details": str(exc),
            "kimera_mode": "safe"
        }
    )

# Basic endpoints
@app.get("/")
async def root():
    return {
        "message": "KIMERA SWM Safe Mode",
        "status": kimera_system['status'],
        "version": "0.1.0-safe"
    }

@app.get("/system/health")
async def system_health():
    return {
        "status": "healthy" if kimera_system['status'] == 'operational' else "degraded",
        "mode": "safe",
        "kimera_status": kimera_system['status'],
        "components_loaded": len(kimera_system['components']),
        "errors": len(kimera_system['errors'])
    }

@app.get("/system/status")
async def system_status():
    return {
        "kimera_status": kimera_system['status'],
        "mode": "safe",
        "loading_progress": getattr(app.state, 'loading_progress', 0),
        "components": list(kimera_system['components'].keys()),
        "errors": kimera_system['errors'][-5:],  # Last 5 errors
        "gpu_available": app.state.gpu_foundation is not None,
        "embedding_available": app.state.embedding_model is not None
    }

# Commented out to avoid conflict with monitoring_routes.py
# @app.get("/metrics")
# async def get_metrics():
#     return {"message": "Metrics endpoint - under construction"}

@app.get("/system/components")
async def get_components():
    """Get information about loaded components"""
    return {
        "components": {
            name: {
                "loaded": True,
                "type": str(type(component).__name__)
            } for name, component in kimera_system['components'].items()
        },
        "total_components": len(kimera_system['components']),
        "status": kimera_system['status']
    }

# Safe geoid creation (basic version)
@app.post("/geoids/safe")
async def create_safe_geoid(content: str):
    """Create a basic geoid without heavy processing"""
    import uuid
    from datetime import datetime
    
    geoid_id = str(uuid.uuid4())
    
    return {
        "geoid_id": geoid_id,
        "content": content,
        "created_at": datetime.now().isoformat(),
        "mode": "safe",
        "status": "created"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="info") 