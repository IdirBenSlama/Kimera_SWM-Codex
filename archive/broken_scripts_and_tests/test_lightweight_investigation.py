#!/usr/bin/env python3
"""
Investigation script to understand lightweight mode behavior.
"""

import sys
import os
from pathlib import Path

# Initialize structured logger
from backend.utils.kimera_logger import get_system_logger
logger = get_system_logger(__name__)


# Add the backend to the path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

def test_lightweight_behavior():
    """Test to understand how lightweight mode actually works."""
    logger.debug("🔍 Investigating Lightweight Mode Behavior...")
    
    # Test 1: Normal mode first
    logger.info("\n1️⃣ Testing Normal Mode:")
    os.environ['LIGHTWEIGHT_EMBEDDING'] = '0'
    
    from backend.core.embedding_utils import encode_text, _embedding_model
    from backend.core.constants import EMBEDDING_DIM
    
    # Check what LIGHTWEIGHT_MODE is set to
    import backend.core.embedding_utils as utils
    logger.info(f"   LIGHTWEIGHT_MODE = {utils.LIGHTWEIGHT_MODE}")
    logger.info(f"   Expected dimension = {EMBEDDING_DIM}")
    
    # Generate embedding
    embedding1 = encode_text("Test in normal mode")
    logger.info(f"   Generated embedding: {len(embedding1)
    logger.info(f"   First few values: {embedding1[:5]}")
    logger.info(f"   Model cached: {_embedding_model is not None}")
    
    # Test 2: Switch to lightweight mode
    logger.info("\n2️⃣ Testing Lightweight Mode Switch:")
    os.environ['LIGHTWEIGHT_EMBEDDING'] = '1'
    
    # Force reload
    import importlib
    importlib.reload(utils)
    
    logger.info(f"   LIGHTWEIGHT_MODE after reload = {utils.LIGHTWEIGHT_MODE}")
    
    # Generate embedding in lightweight mode
    embedding2 = utils.encode_text("Test in lightweight mode")
    logger.info(f"   Generated embedding: {len(embedding2)
    logger.info(f"   First few values: {embedding2[:5]}")
    
    # Test 3: Check if embeddings are different
    logger.info("\n3️⃣ Comparing Embeddings:")
    logger.info(f"   Normal mode embedding norm: {sum(x*x for x in embedding1)
    logger.info(f"   Lightweight embedding norm: {sum(x*x for x in embedding2)
    logger.info(f"   Are they different? {embedding1 != embedding2}")
    
    # Test 4: Test same text in lightweight mode
    logger.info("\n4️⃣ Testing Deterministic Behavior:")
    embedding3 = utils.encode_text("Test in lightweight mode")
    embedding4 = utils.encode_text("Test in lightweight mode")
    logger.info(f"   Same text, embedding 1: {embedding3[:3]}")
    logger.info(f"   Same text, embedding 2: {embedding4[:3]}")
    logger.info(f"   Are they identical? {embedding3 == embedding4}")
    
    # Test 5: Check the actual function being used
    logger.info("\n5️⃣ Function Analysis:")
    if utils.LIGHTWEIGHT_MODE:
        logger.info("   ✅ Lightweight mode is active")
        logger.info("   📝 Using hash-based deterministic random vectors")
        logger.info("   🎯 This is working as intended for testing/development")
    else:
        logger.info("   ✅ Normal mode is active")
        logger.info("   🧠 Using actual BGE-M3 model")
    
    return True

if __name__ == "__main__":
    test_lightweight_behavior()