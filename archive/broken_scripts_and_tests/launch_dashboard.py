#!/usr/bin/env python3
"""
KIMERA SWM Real-Time Dashboard Launcher
Opens the real-time dashboard in your default browser
"""

import webbrowser
import os
import sys
from pathlib import Path

# Initialize structured logger
from backend.utils.kimera_logger import get_system_logger
logger = get_system_logger(__name__)


def launch_dashboard():
    """Launch the KIMERA SWM real-time dashboard"""
    
    # Get the absolute path to the dashboard
    dashboard_path = Path(__file__).parent / "real_time_dashboard.html"
    
    if not dashboard_path.exists():
        logger.error("❌ Dashboard file not found!")
        logger.info(f"Expected location: {dashboard_path}")
        return False
    
    # Convert to file URL
    dashboard_url = f"file:///{dashboard_path.as_posix()}"
    
    logger.info("🚀 Launching KIMERA SWM Real-Time Dashboard...")
    logger.info(f"📊 Dashboard URL: {dashboard_url}")
    logger.info("🔗 API Expected at: http://localhost:8001")
    logger.info()
    logger.info("✨ Features:")
    logger.info("  • Real-time system metrics")
    logger.info("  • Live vault status monitoring")
    logger.info("  • Stability metrics tracking")
    logger.info("  • Active geoids management")
    logger.info("  • Contradiction processing")
    logger.info("  • Activity logging")
    logger.info("  • Auto-refresh every 5 seconds")
    logger.info()
    
    try:
        # Open in default browser
        webbrowser.open(dashboard_url)
        logger.info("✅ Dashboard opened in your default browser!")
        logger.info()
        logger.debug("🔧 Troubleshooting:")
        logger.info("  • Ensure KIMERA API is running on port 8001")
        logger.info("  • Check browser console (F12)
        logger.info("  • Verify CORS is enabled on the API")
        logger.info()
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to open dashboard: {e}")
        logger.info(f"📋 Manual URL: {dashboard_url}")
        return False

if __name__ == "__main__":
    success = launch_dashboard()
    
    if not success:
        sys.exit(1)
    
    logger.info("🎯 Dashboard is now running!")
    logger.info("Press Ctrl+C to exit this script (dashboard will continue running)
    
    try:
        # Keep script running to show it's active
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("\n👋 Dashboard launcher stopped (dashboard still running in browser)
