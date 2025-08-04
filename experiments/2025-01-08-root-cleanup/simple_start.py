#!/usr/bin/env python3
"""
Simple Kimera SWM Startup Script
================================
Simplified startup script for Kimera SWM system.
"""

import os
import sys
import uvicorn
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Start the Kimera SWM system."""
    logger.info("🚀 Starting Kimera SWM System...")
    
    # Set default database URL if not already set
    if not os.getenv("DATABASE_URL"):
        os.environ["DATABASE_URL"] = "postgresql+psycopg2://kimera:kimera_secure_pass_2025@localhost:5432/kimera_swm"
        logger.info(f"📊 Database URL: {os.environ['DATABASE_URL']}")
    
    # Get port from environment or use default
    port = int(os.getenv("PORT", 8000))
    logger.info(f"🌐 Starting on port: {port}")
    
    try:
        # Start the FastAPI application
        uvicorn.run(
            "backend.api.main:create_app",
            host="0.0.0.0",
            port=port,
            reload=False,
            log_level="info"
        )
    except Exception as e:
        logger.info(f"❌ Error starting Kimera: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 