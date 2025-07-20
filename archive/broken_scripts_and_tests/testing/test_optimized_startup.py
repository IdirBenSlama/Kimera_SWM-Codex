#!/usr/bin/env python3
"""
Test script to verify KIMERA optimized startup works correctly
"""

import sys
import time
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all critical imports work"""
    logger.info("🔍 Testing critical imports...")
    
    try:
        # Test backend imports
        from backend.api.main import app
        logger.info("✅ Main app import successful")
        
        from backend.core.kimera_output_intelligence import KimeraOutputIntelligenceSystem
        logger.info("✅ KimeraOutputIntelligenceSystem import successful")
        
        from backend.core.embedding_utils import get_embedding_model
        logger.info("✅ Embedding utils import successful")
        
        return True
    except Exception as e:
        logger.error(f"❌ Import failed: {e}")
        return False

def test_kimera_output_intelligence():
    """Test that KimeraOutputIntelligenceSystem initializes quickly"""
    logger.info("🧠 Testing KimeraOutputIntelligenceSystem initialization...")
    
    try:
        from backend.core.kimera_output_intelligence import KimeraOutputIntelligenceSystem
        start_time = time.time()
        system = KimeraOutputIntelligenceSystem()
        init_time = time.time() - start_time
        
        logger.info(f"✅ KimeraOutputIntelligenceSystem initialized in {init_time:.3f}s")
        
        # Verify the comprehension engine is bypassed
        if system.comprehension_engine is None:
            logger.info("✅ Universal Output Comprehension Engine correctly bypassed")
        else:
            logger.warning("⚠️ Universal Output Comprehension Engine not bypassed")
        
        return init_time < 1.0  # Should be very fast now
        
    except Exception as e:
        logger.error(f"❌ KimeraOutputIntelligenceSystem test failed: {e}")
        return False

def test_embedding_model():
    """Test embedding model initialization"""
    logger.info("🤖 Testing embedding model initialization...")
    
    try:
        start_time = time.time()
        from backend.core.embedding_utils import get_embedding_model
        model = get_embedding_model()
        init_time = time.time() - start_time
        
        logger.info(f"✅ Embedding model initialized in {init_time:.3f}s")
        
        # Test encoding
        if hasattr(model, 'encode'):
            test_text = "Hello, KIMERA!"
            encoding = model.encode(test_text)
            logger.info(f"✅ Encoding test successful: {len(encoding)} dimensions")
        else:
            logger.warning("⚠️ Model doesn't have encode method")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Embedding model test failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("🚀 KIMERA OPTIMIZED STARTUP TESTS")
    logger.info("=" * 50)
    
    tests = [
        ("Import Tests", test_imports),
        ("KimeraOutputIntelligence Test", test_kimera_output_intelligence),
        ("Embedding Model Test", test_embedding_model),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running: {test_name}")
        try:
            if test_func():
                logger.info(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                logger.error(f"❌ {test_name}: FAILED")
        except Exception as e:
            logger.error(f"💥 {test_name}: ERROR - {e}")
    
    logger.info("\n" + "=" * 50)
    logger.info(f"📊 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED - Optimized startup is working!")
        return 0
    else:
        logger.error("❌ Some tests failed - Check logs for details")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 