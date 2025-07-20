"""
Graceful Shutdown and Health Checks for KIMERA System
Phase 4, Weeks 14-15: Deployment Preparation
"""

import asyncio
import logging
from typing import Dict, Any

from fastapi import FastAPI, APIRouter
from fastapi.responses import JSONResponse

from backend.core.performance_integration import get_performance_manager
from backend.core.database_optimization import get_db_optimization
from sqlalchemy import text

logger = logging.getLogger(__name__)


class HealthCheckManager:
    """
    Manages health checks for the application and its dependencies
    """
    
    def __init__(self):
        self.router = APIRouter()
        self.router.add_api_route("/health", self.health_check, methods=["GET"])
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check of the application and its dependencies
        """
        # Check database connection
        db_status = await self.check_database()
        
        # Check other dependencies (e.g., Redis, external APIs)
        
        if db_status["status"] == "ok":
            return {"status": "ok", "dependencies": [db_status]}
        else:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "error",
                    "dependencies": [db_status]
                }
            )
    
    async def check_database(self) -> Dict[str, Any]:
        """
        Check the health of the database connection
        """
        try:
            db_optimizer = await get_db_optimization()
            async with db_optimizer.connection_pool.engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return {"name": "database", "status": "ok"}
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {"name": "database", "status": "error", "error": str(e)}


class GracefulShutdownManager:
    """
    Manages the graceful shutdown of the application
    """
    
    def __init__(self, app: FastAPI):
        self.app = app
        self.app.add_event_handler("startup", self.startup)
        self.app.add_event_handler("shutdown", self.shutdown)
    
    async def startup(self):
        """Application startup event handler"""
        # Initialize performance manager (which includes other initializations)
        perf_manager = await get_performance_manager()
        await perf_manager.initialize_system()
    
    async def shutdown(self):
        """Application shutdown event handler"""
        logger.info("Starting graceful shutdown...")
        
        # Get performance manager
        perf_manager = await get_performance_manager()
        
        # Shutdown performance components (which includes db connections, etc.)
        await perf_manager.shutdown()
        
        logger.info("Graceful shutdown complete.")


# Integration with the main FastAPI app

def setup_deployment_features(app: FastAPI):
    """
    Set up deployment features for the FastAPI application
    """
    # Add health check endpoint
    health_checker = HealthCheckManager()
    app.include_router(health_checker.router)
    
    # Add graceful shutdown handler
    GracefulShutdownManager(app)
    
    logger.info("Deployment features (health checks, graceful shutdown) configured.")
