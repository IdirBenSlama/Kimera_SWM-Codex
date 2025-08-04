#!/usr/bin/env python3
"""
KIMERA SWM - Main Entry Point
Run this script to start the Kimera SWM system
"""

import sys
import os
import subprocess
import logging
logger = logging.getLogger(__name__)

# Add the current directory to Python path to enable imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, current_dir)
sys.path.insert(0, src_dir)

if __name__ == "__main__":
    # Set PYTHONPATH environment variable for subprocess
    env = os.environ.copy()
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = f"{current_dir}{os.pathsep}{src_dir}{os.pathsep}{env['PYTHONPATH']}"
    else:
        env['PYTHONPATH'] = f"{current_dir}{os.pathsep}{src_dir}"

    # Run the main module using python -m to handle imports correctly
    logger.info("🚀 Starting KIMERA SWM System...")
    logger.info("🔍 Server will start on an available port (8000-8003 or 8080)")
    logger.info("📚 API Documentation will be available at: http://127.0.0.1:{port}/docs")
    logger.info("=" * 80)

    # Use subprocess to run with correct module path and environment
    subprocess.run([sys.executable, "-m", "src.main"], cwd=current_dir, env=env)
