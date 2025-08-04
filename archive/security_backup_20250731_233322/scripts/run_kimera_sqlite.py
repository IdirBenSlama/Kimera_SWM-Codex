#!/usr/bin/env python3
"""
Run Kimera with SQLite database for development/testing
"""
import os
import sys
import uvicorn
import logging
logger = logging.getLogger(__name__)

# Override DATABASE_URL to use SQLite
os.environ['DATABASE_URL'] = 'sqlite:///./kimera_swm.db'

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

if __name__ == "__main__":
    logger.info("ðŸš€ Launching Kimera SWM with SQLite database...")
    logger.info(f"Database: {os.environ['DATABASE_URL']}")
    
    uvicorn.run(
        "backend.api.main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True
    )