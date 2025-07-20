#!/usr/bin/env python3
"""
Test the API directly to see what's causing the 400 errors.
"""

import sys
import os
from pathlib import Path

# Initialize structured logger
from backend.utils.kimera_logger import get_system_logger
logger = get_system_logger(__name__)


# Add the backend to the path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi.testclient import TestClient
from backend.api.main import app
import json

def test_api_directly():
    """Test the API directly with the same payload as the crash test."""
    logger.debug("🔍 Testing API directly...")
    
    client = TestClient(app)
    
    # Create the same payload as the crash test
    payload = {
        "semantic_features": {
            "feature_0": 0.5,
            "feature_1": -0.3,
            "feature_2": 0.8
        },
        "symbolic_content": {
            "level_2": {
                "level_1": {
                    "level_0": "abcdefghijklmnopqrstuvwxyzabcdef"
                },
                "data_1": "zyxwvutsrqponmlkjihgfedcbazyxwvu"
            },
            "data_2": "mnbvcxzasdfghjklpoiuytrewqmnbvcx"
        },
        "metadata": {
            "test_id": "test123",
            "timestamp": "2024-01-01T00:00:00",
            "depth": 2,
            "feature_count": 3
        }
    }
    
    logger.info("📤 Sending payload:")
    logger.info(json.dumps(payload, indent=2)
    
    try:
        response = client.post("/geoids", json=payload)
        logger.info(f"\n📥 Response status: {response.status_code}")
        logger.info(f"📥 Response content: {response.text}")
        
        if response.status_code == 200:
            logger.info("✅ API call successful!")
            return True
        else:
            logger.error("❌ API call failed!")
            try:
                error_detail = response.json()
                logger.error(f"Error detail: {error_detail}")
            except:
                logger.error("Could not parse error response as JSON")
            return False
            
    except Exception as e:
        logger.critical(f"💥 Exception during API call: {e}")
        return False

def test_simple_payload():
    """Test with a simpler payload."""
    logger.debug("\n🔍 Testing with simple payload...")
    
    client = TestClient(app)
    
    simple_payload = {
        "echoform_text": "This is a simple test message."
    }
    
    logger.info("📤 Sending simple payload:")
    logger.info(json.dumps(simple_payload, indent=2)
    
    try:
        response = client.post("/geoids", json=simple_payload)
        logger.info(f"\n📥 Response status: {response.status_code}")
        logger.info(f"📥 Response content: {response.text}")
        
        if response.status_code == 200:
            logger.info("✅ Simple API call successful!")
            return True
        else:
            logger.error("❌ Simple API call failed!")
            return False
            
    except Exception as e:
        logger.critical(f"💥 Exception during simple API call: {e}")
        return False

if __name__ == "__main__":
    logger.info("🧪 Testing Kimera API Endpoints...")
    
    # Test the complex payload first
    success1 = test_api_directly()
    
    # Test simple payload
    success2 = test_simple_payload()
    
    if success1 and success2:
        logger.info("\n🎉 All API tests passed!")
    else:
        logger.error("\n❌ Some API tests failed!")
        logger.error("This explains why the tyrannic crash test is failing.")