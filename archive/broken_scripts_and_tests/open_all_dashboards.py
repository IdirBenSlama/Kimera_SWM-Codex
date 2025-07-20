#!/usr/bin/env python3
"""
Open all KIMERA SWM dashboards
"""

import webbrowser
import os
from pathlib import Path

# Initialize structured logger
from backend.utils.kimera_logger import get_system_logger
logger = get_system_logger(__name__)


def open_dashboards():
    """Open all available KIMERA dashboards"""
    
    base_path = Path(__file__).parent
    
    dashboards = [
        {
            "name": "Original React Dashboard",
            "file": "dashboard/index.html",
            "description": "Main original dashboard with React components"
        },
        {
            "name": "Simple Styled Dashboard", 
            "file": "dashboard/simple_dashboard.html",
            "description": "Beautiful styled dashboard with auto-refresh"
        },
        {
            "name": "Real-Time Enhanced Dashboard",
            "file": "real_time_dashboard.html", 
            "description": "Most comprehensive with detailed metrics"
        },
        {
            "name": "Dashboard Guide",
            "file": "dashboard_guide.html",
            "description": "Complete explanation of what everything means"
        },
        {
            "name": "Quick Reference",
            "file": "dashboard_cheatsheet.html",
            "description": "Quick reference card to keep open"
        }
    ]
    
    logger.info("🚀 Opening KIMERA SWM Dashboards...")
    logger.info("=" * 50)
    
    for i, dashboard in enumerate(dashboards, 1):
        file_path = base_path / dashboard["file"]
        
        if file_path.exists():
            file_url = f"file:///{file_path.as_posix()}"
            logger.info(f"{i}. {dashboard['name']}")
            logger.info(f"   📄 {dashboard['description']}")
            logger.info(f"   🔗 {file_url}")
            
            try:
                webbrowser.open(file_url)
                logger.info(f"   ✅ Opened successfully")
            except Exception as e:
                logger.error(f"   ❌ Failed to open: {e}")
        else:
            logger.error(f"{i}. {dashboard['name']} - ❌ File not found: {file_path}")
        
        logger.info()
    
    logger.info("🎯 All dashboards opened!")
    logger.info("\n📋 Dashboard Comparison:")
    logger.info("• React Dashboard (index.html)
    logger.info("• Simple Dashboard - Clean, styled, auto-refresh")
    logger.info("• Real-Time Dashboard - Most features, comprehensive")
    logger.info("• Guide & Cheatsheet - Help understanding the data")
    
    logger.info(f"\n🔗 API should be running on: http://localhost:8001")
    logger.debug("🔧 If dashboards show connection errors, restart the API server")

if __name__ == "__main__":
    open_dashboards()
