#!/usr/bin/env python3
"""
KIMERA SWM - Unified Root Entry Point
====================================

Updated root entry point that launches the unified main.py v3.0.0
Run this script to start the Kimera SWM system with complete unification.

Updated by: Kimera SWM Autonomous Architect
Date: 2025-08-01T00:14:55.181879
Version: 3.0.0 (Complete Unification)
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

    # Run the unified main module
    logger.info("🚀 Starting KIMERA SWM System...")
    logger.info("🎯 Using Complete Unified Entry Point v3.0.0")
    logger.info("🏗️ Integrated with Unified Cognitive Architecture")
    logger.info("🔍 Server will start on an available port (8000-8003 or 8080)")
    logger.info("📚 API Documentation will be available at: http://127.0.0.1:{port}/docs")
    logger.info("🎮 Set KIMERA_MODE environment variable:")
    logger.info("   • progressive (default) - Lazy loading with background enhancement")
    logger.info("   • full - Complete initialization with all features")
    logger.info("   • safe - Maximum fallbacks and error tolerance")
    logger.info("   • fast - Minimal features for rapid startup")
    logger.info("   • optimized - Performance-focused initialization")
    logger.info("   • hybrid - Adaptive based on system capabilities")
    logger.info("=" * 80)

    # Use subprocess to run with correct module path and environment
    subprocess.run([sys.executable, "-m", "src.main"], cwd=current_dir, env=env)
