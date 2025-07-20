#!/usr/bin/env python3
"""Test script to verify all fixes are working"""

import os
import sys
import time
import requests
from dotenv import load_dotenv

# Initialize structured logger
from backend.utils.kimera_logger import get_system_logger
logger = get_system_logger(__name__)


# Load environment variables
load_dotenv()

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_neo4j_connection():
    """Test Neo4j connection"""
    logger.info("\n1. Testing Neo4j Connection...")
    try:
        from backend.graph.session import driver_liveness_check
        if driver_liveness_check():
            logger.info("✅ Neo4j connection successful")
            return True
        else:
            logger.error("❌ Neo4j connection failed")
            return False
    except Exception as e:
        logger.error(f"❌ Neo4j test error: {e}")
        return False

def test_api_endpoints():
    """Test API endpoints"""
    logger.info("\n2. Testing API Endpoints...")
    base_url = "http://localhost:8001"
    
    endpoints = [
        ("/docs", "GET", None),
        ("/system/health", "GET", None),
        ("/monitoring/status", "GET", None),
        ("/system/status", "GET", None),
    ]
    
    results = []
    for endpoint, method, data in endpoints:
        try:
            if method == "GET":
                response = requests.get(f"{base_url}{endpoint}", timeout=5)
            else:
                response = requests.post(f"{base_url}{endpoint}", json=data, timeout=5)
            
            if response.status_code < 400:
                logger.info(f"✅ {method} {endpoint} - Status: {response.status_code}")
                results.append(True)
            else:
                logger.error(f"❌ {method} {endpoint} - Status: {response.status_code}")
                if response.text:
                    logger.error(f"   Error: {response.text[:100]}")
                results.append(False)
        except Exception as e:
            logger.error(f"❌ {method} {endpoint} - Error: {e}")
            results.append(False)
    
    return all(results)

def test_create_geoid():
    """Test creating a geoid"""
    logger.info("\n3. Testing Geoid Creation...")
    try:
        response = requests.post(
            "http://localhost:8001/geoids",
            json={
                "semantic_features": {"test": 1.0, "fix": 0.8},
                "metadata": {"source": "test_script"}
            },
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✅ Created geoid: {data['geoid_id']}")
            logger.info(f"   Entropy: {data['entropy']}")
            return True, data['geoid_id']
        else:
            logger.error(f"❌ Failed to create geoid: {response.status_code}")
            logger.error(f"   Error: {response.text}")
            return False, None
    except Exception as e:
        logger.error(f"❌ Geoid creation error: {e}")
        return False, None

def test_process_contradictions(geoid_id):
    """Test processing contradictions"""
    logger.info("\n4. Testing Contradiction Processing...")
    if not geoid_id:
        logger.warning("⚠️  Skipping - no geoid available")
        return False
    
    try:
        response = requests.post(
            "http://localhost:8001/process/contradictions",
            json={
                "trigger_geoid_id": geoid_id,
                "search_limit": 5
            },
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✅ Processed contradictions")
            logger.info(f"   Contradictions detected: {data.get('contradictions_detected', 0)
            logger.info(f"   SCARs created: {data.get('scars_created', 0)
            return True
        else:
            logger.error(f"❌ Failed to process contradictions: {response.status_code}")
            logger.error(f"   Error: {response.text}")
            return False
    except Exception as e:
        logger.error(f"❌ Contradiction processing error: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("🚀 Kimera SWM Fix Verification")
    logger.info("=" * 50)
    
    # Check if API is running
    try:
        response = requests.get("http://localhost:8001/docs", timeout=2)
        logger.info("✅ API server is running")
    except:
        logger.error("❌ API server is not running. Please start it with: python run_kimera.py")
        return
    
    # Run tests
    neo4j_ok = test_neo4j_connection()
    api_ok = test_api_endpoints()
    geoid_ok, geoid_id = test_create_geoid()
    contradiction_ok = test_process_contradictions(geoid_id)
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 Test Summary:")
    logger.info(f"   Neo4j Connection: {'✅' if neo4j_ok else '❌'}")
    logger.info(f"   API Endpoints: {'✅' if api_ok else '❌'}")
    logger.info(f"   Geoid Creation: {'✅' if geoid_ok else '❌'}")
    logger.info(f"   Contradiction Processing: {'✅' if contradiction_ok else '❌'}")
    
    all_ok = neo4j_ok and api_ok and geoid_ok and contradiction_ok
    logger.warning(f"\n{'🎉 All tests passed!' if all_ok else '⚠️  Some tests failed'}")

if __name__ == "__main__":
    main()