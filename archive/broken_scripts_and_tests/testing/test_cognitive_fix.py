#!/usr/bin/env python3
"""Quick test to fix cognitive processing issue."""

import requests
import json

# Initialize structured logger
from backend.utils.kimera_logger import get_system_logger
logger = get_system_logger(__name__)


def test_cognitive_fix():
    """Test cognitive processing with correct format."""
    logger.info("Testing cognitive processing fix...")
    
    # Fix the geoid creation with correct format
    payload = {
        'echoform_text': 'Analyze the quantum mechanical implications of cognitive processing in artificial intelligence systems',
        'metadata': {'test': 'cognitive_processing_fix'}
    }
    
    try:
        response = requests.post('http://localhost:8001/geoids', json=payload, timeout=30)
        logger.info(f"Status: {response.status_code}")
        
        if response.status_code in [200, 201]:
            result = response.json()
            logger.info("SUCCESS: Geoid created")
            logger.info(f"Response keys: {list(result.keys()
            logger.info(f"Response size: {len(str(result)
            logger.info("✅ Cognitive processing is working!")
            return True
        else:
            logger.error(f"ERROR: {response.text}")
            return False
    
    except Exception as e:
        logger.error(f"ERROR: {e}")
        return False

if __name__ == "__main__":
    test_cognitive_fix() 