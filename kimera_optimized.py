#!/usr/bin/env python3
"""
Optimized Kimera Startup Script
==============================
Starts Kimera with performance optimizations enabled.
Windows-compatible version.
"""

import os
import sys
import uvicorn
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import optimized app
from backend.api.main_optimized import app

if __name__ == "__main__":
    # Performance-optimized Uvicorn configuration
    # Note: uvloop is not available on Windows, using default loop
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        workers=1,  # Single worker for now, can increase based on CPU cores
        access_log=False,  # Disable access logs for performance
        log_level="warning",  # Reduce log verbosity
        limit_concurrency=1000,  # High concurrency limit
        limit_max_requests=10000,  # Request limit before worker restart
        timeout_keep_alive=5,  # Keep-alive timeout
    )