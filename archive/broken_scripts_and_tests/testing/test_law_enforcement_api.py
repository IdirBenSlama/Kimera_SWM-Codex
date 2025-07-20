"""
Simple API Test for Law Enforcement System
==========================================

Tests the API endpoints to ensure they're working correctly.
"""

import requests
import json
import time

# Initialize structured logger
from backend.utils.kimera_logger import get_system_logger
logger = get_system_logger(__name__)


# API base URL
BASE_URL = "http://localhost:8001"
LAW_ENFORCEMENT_URL = f"{BASE_URL}/law_enforcement"


def test_api_endpoints():
    """Test all law enforcement API endpoints"""
    
    logger.info("🧪 Testing Law Enforcement API Endpoints")
    logger.info("=" * 50)
    
    # Test 1: Get system status
    logger.info("1. Testing system status...")
    try:
        response = requests.get(f"{LAW_ENFORCEMENT_URL}/system_status")
        if response.status_code == 200:
            data = response.json()
            logger.info(f"   ✓ System status: {data['system_status']['system_stability']['status']}")
        else:
            logger.info(f"   ✗ Status code: {response.status_code}")
    except Exception as e:
        logger.error(f"   ✗ Error: {e}")
    
    # Test 2: Get all laws
    logger.info("2. Testing law retrieval...")
    try:
        response = requests.get(f"{LAW_ENFORCEMENT_URL}/laws")
        if response.status_code == 200:
            data = response.json()
            logger.info(f"   ✓ Retrieved {data['total_count']} laws")
        else:
            logger.info(f"   ✗ Status code: {response.status_code}")
    except Exception as e:
        logger.error(f"   ✗ Error: {e}")
    
    # Test 3: Context assessment
    logger.info("3. Testing context assessment...")
    try:
        test_data = {
            "input_text": "I need help with my homework on quantum physics",
            "user_context": {"is_student": True}
        }
        response = requests.post(f"{LAW_ENFORCEMENT_URL}/assess_context", json=test_data)
        if response.status_code == 200:
            data = response.json()
            logger.info(f"   ✓ Context: {data['context_type']}, Relevance: {data['relevance_level']}")
        else:
            logger.info(f"   ✗ Status code: {response.status_code}")
    except Exception as e:
        logger.error(f"   ✗ Error: {e}")
    
    # Test 4: Compliance assessment
    logger.info("4. Testing compliance assessment...")
    try:
        test_data = {
            "input_text": "Can you explain different political perspectives on climate change?",
            "action": "provide_balanced_analysis",
            "user_context": {"educational_purpose": True}
        }
        response = requests.post(f"{LAW_ENFORCEMENT_URL}/assess_compliance", json=test_data)
        if response.status_code == 200:
            data = response.json()
            logger.info(f"   ✓ Compliant: {data['compliant']}, Action: {data['enforcement_action']}")
        else:
            logger.info(f"   ✗ Status code: {response.status_code}")
    except Exception as e:
        logger.error(f"   ✗ Error: {e}")
    
    # Test 5: Get specific law
    logger.info("5. Testing specific law retrieval...")
    try:
        response = requests.get(f"{LAW_ENFORCEMENT_URL}/laws/N1")
        if response.status_code == 200:
            data = response.json()
            logger.info(f"   ✓ Law N1: {data['law']['name']}")
        else:
            logger.info(f"   ✗ Status code: {response.status_code}")
    except Exception as e:
        logger.error(f"   ✗ Error: {e}")
    
    logger.info("\n✅ API endpoint testing completed!")


if __name__ == "__main__":
    logger.info("🚀 Starting API tests...")
    logger.info("Make sure KIMERA is running on port 8001")
    logger.info()
    
    # Wait a moment for any startup
    time.sleep(1)
    
    test_api_endpoints() 