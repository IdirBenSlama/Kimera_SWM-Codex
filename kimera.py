#!/usr/bin/env python3
"""
KIMERA SWM - Main Entry Point
Run this script to start the Kimera SWM system
"""

import sys
import os
import subprocess

# Add the current directory to Python path to enable imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

if __name__ == "__main__":
    # Run the main module using python -m to handle imports correctly
    print("🚀 Starting KIMERA SWM System...")
    print("🔍 Monitoring available at: http://127.0.0.1:8000/health")
    print("📚 API Documentation at: http://127.0.0.1:8000/docs")
    print("=" * 80)
    
    # Use subprocess to run with correct module path
    subprocess.run([sys.executable, "-m", "src.main"], cwd=current_dir) 