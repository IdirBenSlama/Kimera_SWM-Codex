#!/usr/bin/env python3
"""
Launch script for Kimera SWM Monitoring Dashboard

This script starts the monitoring dashboard and optionally the API server.
"""

import os
import sys
import webbrowser
import subprocess
import time
import argparse
from pathlib import Path

# Initialize structured logger
from backend.utils.kimera_logger import get_system_logger
logger = get_system_logger(__name__)


def check_api_running(host="localhost", port=8001):
    """Check if the API server is running"""
    try:
        import requests
        response = requests.get(f"http://{host}:{port}/system/status", timeout=5)
        return response.status_code == 200
    except:
        return False

def start_api_server():
    """Start the API server"""
    logger.info("Starting Kimera SWM API server...")
    try:
        # Try to start the API server
        process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "backend.api.main:app", 
            "--host", "0.0.0.0", 
            "--port", "8001",
            "--reload"
        ], cwd=Path(__file__).parent)
        
        # Wait a bit for the server to start
        time.sleep(3)
        
        if check_api_running():
            logger.info("✅ API server started successfully on http://localhost:8001")
            return process
        else:
            logger.error("❌ Failed to start API server")
            return None
    except Exception as e:
        logger.error(f"❌ Error starting API server: {e}")
        return None

def open_dashboard():
    """Open the monitoring dashboard in the default browser"""
    dashboard_path = Path(__file__).parent / "monitoring_dashboard.html"
    
    if not dashboard_path.exists():
        logger.error(f"❌ Dashboard file not found: {dashboard_path}")
        return False
    
    try:
        # Convert to file URL
        file_url = dashboard_path.as_uri()
        webbrowser.open(file_url)
        logger.info(f"✅ Monitoring dashboard opened: {file_url}")
        return True
    except Exception as e:
        logger.error(f"❌ Error opening dashboard: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Launch Kimera SWM Monitoring Dashboard")
    parser.add_argument("--no-api", action="store_true", 
                       help="Don't start the API server (assume it's already running)")
    parser.add_argument("--no-browser", action="store_true",
                       help="Don't open the browser automatically")
    parser.add_argument("--check-only", action="store_true",
                       help="Only check if API is running")
    
    args = parser.parse_args()
    
    logger.info("🧠 Kimera SWM Monitoring Dashboard Launcher")
    logger.info("=" * 50)
    
    # Check if API is already running
    if check_api_running():
        logger.info("✅ API server is already running on http://localhost:8001")
        api_process = None
    elif args.check_only:
        logger.error("❌ API server is not running")
        return 1
    elif args.no_api:
        logger.warning("⚠️  API server not running, but --no-api flag set")
        logger.info("   Make sure to start the API server manually:")
        logger.info("   python -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8001 --reload")
        api_process = None
    else:
        api_process = start_api_server()
        if not api_process:
            logger.error("❌ Failed to start API server. Exiting.")
            return 1
    
    if args.check_only:
        return 0
    
    # Open the dashboard
    if not args.no_browser:
        if open_dashboard():
            logger.info("\n🎉 Monitoring dashboard is ready!")
            logger.info("\nFeatures available:")
            logger.info("  ��� Real-time system metrics")
            logger.info("  📈 Entropy and thermodynamic trends")
            logger.info("  🧠 Semantic analysis")
            logger.info("  🚨 Alert monitoring")
            logger.info("  🧪 Benchmark testing")
            logger.info("\nAPI endpoints available at: http://localhost:8001/docs")
        else:
            logger.error("❌ Failed to open dashboard")
            return 1
    else:
        dashboard_path = Path(__file__).parent / "monitoring_dashboard.html"
        logger.info(f"📊 Dashboard available at: {dashboard_path.as_uri()
    
    # Keep the script running if we started the API server
    if api_process:
        try:
            logger.info("\n⌨️  Press Ctrl+C to stop the API server and exit")
            api_process.wait()
        except KeyboardInterrupt:
            logger.info("\n🛑 Stopping API server...")
            api_process.terminate()
            try:
                api_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                api_process.kill()
            logger.info("✅ API server stopped")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
