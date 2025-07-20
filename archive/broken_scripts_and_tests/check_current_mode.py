#!/usr/bin/env python3
"""
Clear check of what mode we're actually using.
"""

import sys
import os
from pathlib import Path

# Initialize structured logger
from backend.utils.kimera_logger import get_system_logger
logger = get_system_logger(__name__)


# Add the backend to the path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

def check_current_mode():
    """Check exactly what mode we're using."""
    logger.debug("🔍 CHECKING CURRENT EMBEDDING MODE...")
    
    # Check environment
    lightweight_env = os.getenv('LIGHTWEIGHT_EMBEDDING', 'Not Set')
    logger.info(f"Environment LIGHTWEIGHT_EMBEDDING: {lightweight_env}")
    
    # Import and check the actual values
    from backend.core.embedding_utils import LIGHTWEIGHT_MODE, MODEL_NAME, encode_text
    from backend.core.constants import EMBEDDING_DIM
    
    logger.info(f"LIGHTWEIGHT_MODE variable: {LIGHTWEIGHT_MODE}")
    logger.info(f"MODEL_NAME: {MODEL_NAME}")
    logger.info(f"EMBEDDING_DIM: {EMBEDDING_DIM}")
    
    # Test actual embedding generation
    logger.info("\n🧠 TESTING ACTUAL EMBEDDING GENERATION:")
    test_text = "This is a test to see what model we're actually using."
    
    embedding = encode_text(test_text)
    
    logger.info(f"Generated embedding length: {len(embedding)
    logger.info(f"First 5 values: {embedding[:5]}")
    
    # Calculate norm
    norm = sum(x*x for x in embedding) ** 0.5
    logger.info(f"Embedding norm: {norm:.6f}")
    
    # Determine what we're actually using
    logger.info("\n📊 ANALYSIS:")
    if LIGHTWEIGHT_MODE:
        logger.error("❌ USING LIGHTWEIGHT MODE (hash-based random vectors)
        logger.info("   This is NOT the BGE model!")
        logger.info("   This is for testing/development only")
    else:
        if norm > 0.99 and norm < 1.01:  # Normalized embeddings
            logger.info("✅ USING FULL BGE-M3 MODEL")
            logger.info("   Real semantic embeddings with normalization")
            logger.info("   This is the actual migrated model!")
        else:
            logger.warning("⚠️  USING DUMMY/FALLBACK MODE")
            logger.info("   Something went wrong with model loading")
    
    return LIGHTWEIGHT_MODE

if __name__ == "__main__":
    is_lightweight = check_current_mode()
    
    if is_lightweight:
        logger.info("\n🚨 ISSUE: We're in lightweight mode!")
        logger.info("To use the full BGE model, ensure LIGHTWEIGHT_EMBEDDING is not set to '1'")
    else:
        logger.info("\n✅ GOOD: We're using the full BGE-M3 model!")