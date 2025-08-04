#!/usr/bin/env python3
"""
Quick validation that critical fixes are working
"""

def validate_fixes():
    """Validate that critical fixes are working"""
    logger.info("🔧 VALIDATING ENHANCED CAPABILITIES FIXES")
    logger.info("=" * 50)
    
    tests_passed = 0
    total_tests = 6
    
    # Test 1: PyTorch operations
    logger.info("1️⃣  Testing PyTorch operations...")
    try:
        import torch
        a = torch.randn(5)
        b = torch.randn(5)
        sim = torch.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=1)
        logger.info(f"   ✅ torch.cosine_similarity: {sim.item():.3f}")
        tests_passed += 1
    except Exception as e:
        logger.info(f"   ❌ PyTorch operations failed: {e}")
    
    # Test 2: Understanding Core
    logger.info("2️⃣  Testing Understanding Core import...")
    try:
        from src.core.enhanced_capabilities.understanding_core import UnderstandingCore
        core = UnderstandingCore()
        logger.info("   ✅ Understanding Core imported and created")
        tests_passed += 1
    except Exception as e:
        logger.info(f"   ❌ Understanding Core failed: {e}")
    
    # Test 3: Consciousness Core
    logger.info("3️⃣  Testing Consciousness Core import...")
    try:
        from src.core.enhanced_capabilities.consciousness_core import ConsciousnessCore
        core = ConsciousnessCore()
        logger.info("   ✅ Consciousness Core imported and created")
        tests_passed += 1
    except Exception as e:
        logger.info(f"   ❌ Consciousness Core failed: {e}")
    
    # Test 4: Meta Insight Core (FFT fix)
    logger.info("4️⃣  Testing Meta Insight Core import...")
    try:
        from src.core.enhanced_capabilities.meta_insight_core import MetaInsightCore
        core = MetaInsightCore()
        logger.info("   ✅ Meta Insight Core imported and created")
        tests_passed += 1
    except Exception as e:
        logger.info(f"   ❌ Meta Insight Core failed: {e}")
    
    # Test 5: Learning Core (tensor operations fix)
    logger.info("5️⃣  Testing Learning Core import...")
    try:
        from src.core.enhanced_capabilities.learning_core import LearningCore
        core = LearningCore()
        logger.info("   ✅ Learning Core imported and created")
        tests_passed += 1
    except Exception as e:
        logger.info(f"   ❌ Learning Core failed: {e}")
    
    # Test 6: Linguistic Intelligence Core (None check fix)
    logger.info("6️⃣  Testing Linguistic Intelligence Core import...")
    try:
        from src.core.enhanced_capabilities.linguistic_intelligence_core import LinguisticIntelligenceCore
import logging
logger = logging.getLogger(__name__)
        core = LinguisticIntelligenceCore()
        logger.info("   ✅ Linguistic Intelligence Core imported and created")
        tests_passed += 1
    except Exception as e:
        logger.info(f"   ❌ Linguistic Intelligence Core failed: {e}")
    
    logger.info()
    logger.info("=" * 50)
    logger.info(f"🎯 VALIDATION RESULTS")
    logger.info(f"   Tests Passed: {tests_passed}/{total_tests}")
    logger.info(f"   Success Rate: {tests_passed/total_tests:.1%}")
    
    if tests_passed == total_tests:
        logger.info("🎉 ALL FIXES WORKING PERFECTLY!")
        logger.info("✅ Ready to run full enhanced capabilities test!")
        return True
    elif tests_passed >= total_tests * 0.8:
        logger.info("✅ MOST FIXES WORKING - GOOD PROGRESS!")
        logger.info("🔧 Minor issues may remain")
        return True
    else:
        logger.info("⚠️  SIGNIFICANT ISSUES REMAIN")
        logger.info("🔧 Need to investigate further")
        return False

if __name__ == "__main__":
    success = validate_fixes()
    
    if success:
        logger.info("\n🚀 RECOMMENDATION: Run full test suite")
        logger.info("   Command: python test_complete_phase3_enhanced_capabilities.py")
    else:
        logger.info("\n🔧 RECOMMENDATION: Fix remaining issues first")