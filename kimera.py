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
    print("🚀 Starting KIMERA SWM System...")
    print("🎯 Using Complete Unified Entry Point v3.0.0")
    print("🏗️ Integrated with Unified Cognitive Architecture")
    print("🔍 Server will start on an available port (8000-8003 or 8080)")
    print("📚 API Documentation will be available at: http://127.0.0.1:{port}/docs")
    print("🎮 Set KIMERA_MODE environment variable:")
    print("   • progressive (default) - Lazy loading with background enhancement")
    print("   • full - Complete initialization with all features")
    print("   • safe - Maximum fallbacks and error tolerance")
    print("   • fast - Minimal features for rapid startup")
    print("   • optimized - Performance-focused initialization")
    print("   • hybrid - Adaptive based on system capabilities")
    print("=" * 80)

    # Use subprocess to run with correct module path and environment
    subprocess.run([sys.executable, "-m", "src.main"], cwd=current_dir, env=env)
