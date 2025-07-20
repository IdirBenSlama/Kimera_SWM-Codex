#!/usr/bin/env python3
"""
System Verification Script
Checks all major endpoints and components
"""

import requests
import json
import logging
from datetime import datetime

# Configure logging properly instead of using print statements
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8001"

def check_endpoint(method, path, data=None, expected_status=200):
    """Check a single endpoint"""
    url = f"{BASE_URL}{path}"
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        else:
            return f"❌ Unsupported method: {method}"
        
        if response.status_code == expected_status:
            return f"✅ {method} {path} - Status: {response.status_code}"
        else:
            return f"❌ {method} {path} - Status: {response.status_code} (Expected: {expected_status})"
    except Exception as e:
        return f"❌ {method} {path} - Error: {str(e)}"

def main():
    logger.info("=" * 60)
    logger.info("KIMERA SYSTEM VERIFICATION")
    logger.info(f"Time: {datetime.now().isoformat()}")
    logger.info("=" * 60)
    
    # Core System Endpoints
    logger.info("\n🔍 CORE SYSTEM")
    logger.info(check_endpoint("GET", "/"))
    logger.info(check_endpoint("GET", "/kimera/status"))
    
    # GPU Foundation
    logger.info("\n🖥️ GPU FOUNDATION")
    logger.info(check_endpoint("GET", "/kimera/system/gpu_foundation"))
    
    # Statistical Engine
    logger.info("\n📊 STATISTICAL ENGINE")
    logger.info(check_endpoint("GET", "/kimera/statistics/capabilities"))
    
    # Cognitive Control
    logger.info("\n🧠 COGNITIVE CONTROL")
    logger.info(check_endpoint("GET", "/kimera/cognitive-control/health"))
    logger.info(check_endpoint("GET", "/kimera/cognitive-control/system/status"))
    logger.info(check_endpoint("GET", "/kimera/cognitive-control/context/status"))
    
    # Thermodynamic Engine
    logger.info("\n🔥 THERMODYNAMIC ENGINE")
    logger.info(check_endpoint("GET", "/kimera/thermodynamic_engine"))
    
    # Geoid Operations
    logger.info("\n🌐 GEOID OPERATIONS")
    test_geoid = {
        "semantic_features": {"test": 1.0, "verification": 0.8},
        "symbolic_content": {"description": "Test semantic object for verification"},
        "metadata": {"type": "test", "timestamp": datetime.now().isoformat()}
    }
    logger.info(check_endpoint("POST", "/kimera/geoids", data=test_geoid))
    
    # System Health
    logger.info("\n🏥 SYSTEM HEALTH")
    logger.info(check_endpoint("GET", "/kimera/system/health"))
    logger.info(check_endpoint("GET", "/kimera/system/health/detailed"))
    
    # Contradiction Engine
    logger.info("\n⚡ CONTRADICTION ENGINE")
    logger.info(check_endpoint("GET", "/kimera/contradiction_engine"))
    
    logger.info("\n" + "=" * 60)
    logger.info("VERIFICATION COMPLETE")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()