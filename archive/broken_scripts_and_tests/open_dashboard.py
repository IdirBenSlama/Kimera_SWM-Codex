#!/usr/bin/env python3
"""
Open KIMERA SWM Dashboard in Browser
"""
import webbrowser
import os
import time

# Initialize structured logger
from backend.utils.kimera_logger import get_system_logger
logger = get_system_logger(__name__)


def open_dashboard():
    # Get the absolute path to the test dashboard
    dashboard_path = os.path.abspath("test_dashboard.html")
    dashboard_url = f"file:///{dashboard_path.replace(os.sep, '/')}"
    
    logger.info("🚀 Opening KIMERA SWM Dashboard...")
    logger.info(f"📊 Dashboard URL: {dashboard_url}")
    logger.info(f"🔗 API Server: http://localhost:8001")
    logger.info("\n✅ The dashboard should open in your default browser")
    logger.info("📈 It will show real-time metrics from your KIMERA SWM system")
    
    # Open in browser
    webbrowser.open(dashboard_url)
    
    logger.debug("\n🔧 Dashboard Features:")
    logger.info("   - Real-time system metrics")
    logger.info("   - Connection status indicator")
    logger.info("   - Create test geoids")
    logger.info("   - Run cognitive cycles")
    logger.info("   - Search existing geoids")
    logger.info("   - Auto-refresh every 10 seconds")

if __name__ == "__main__":
    open_dashboard()
