#!/usr/bin/env python3
"""
Kimera SWM Project - Main Entry Point
-------------------------------------
This script serves as the single, authoritative entry point for running
the Kimera SWM application.

To run the application:
    $ python kimera.py

This will start the Uvicorn server with the main FastAPI application.
"""

import os
import sys

# ----------------------------------------------------------------------
# System Path Correction
# ----------------------------------------------------------------------
# This is a critical step to ensure that all modules can be imported
# correctly, regardless of how the script is executed. It adds the
# project's root directory to the system path.
# This MUST be at the top of the file, before any other imports.
# ----------------------------------------------------------------------
try:
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        print(f"✅ Project root added to system path: {project_root}")
except Exception as e:
    print(f"❌ Failed to add project root to system path: {e}", file=sys.stderr)

import uvicorn
import logging

if __name__ == "__main__":
    # Configure basic logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # --- Server Configuration ---
    # Uvicorn is configured to use the application factory `create_app`
    # This is a more robust pattern for complex applications.
    uvicorn.run(
        "backend.api.main:create_app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        factory=True,
        reload=False
    ) 