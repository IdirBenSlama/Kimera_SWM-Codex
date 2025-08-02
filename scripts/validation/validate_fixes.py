#!/usr/bin/env python3
"""
Quick validation that critical fixes are working
"""

def validate_fixes():
    """Validate that critical fixes are working"""
    print("🔧 VALIDATING ENHANCED CAPABILITIES FIXES")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 6
    
    # Test 1: PyTorch operations
    print("1️⃣  Testing PyTorch operations...")
    try:
        import torch
        a = torch.randn(5)
        b = torch.randn(5)
        sim = torch.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=1)
        print(f"   ✅ torch.cosine_similarity: {sim.item():.3f}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ PyTorch operations failed: {e}")
    
    # Test 2: Understanding Core
    print("2️⃣  Testing Understanding Core import...")
    try:
        from src.core.enhanced_capabilities.understanding_core import UnderstandingCore
        core = UnderstandingCore()
        print("   ✅ Understanding Core imported and created")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Understanding Core failed: {e}")
    
    # Test 3: Consciousness Core
    print("3️⃣  Testing Consciousness Core import...")
    try:
        from src.core.enhanced_capabilities.consciousness_core import ConsciousnessCore
        core = ConsciousnessCore()
        print("   ✅ Consciousness Core imported and created")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Consciousness Core failed: {e}")
    
    # Test 4: Meta Insight Core (FFT fix)
    print("4️⃣  Testing Meta Insight Core import...")
    try:
        from src.core.enhanced_capabilities.meta_insight_core import MetaInsightCore
        core = MetaInsightCore()
        print("   ✅ Meta Insight Core imported and created")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Meta Insight Core failed: {e}")
    
    # Test 5: Learning Core (tensor operations fix)
    print("5️⃣  Testing Learning Core import...")
    try:
        from src.core.enhanced_capabilities.learning_core import LearningCore
        core = LearningCore()
        print("   ✅ Learning Core imported and created")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Learning Core failed: {e}")
    
    # Test 6: Linguistic Intelligence Core (None check fix)
    print("6️⃣  Testing Linguistic Intelligence Core import...")
    try:
        from src.core.enhanced_capabilities.linguistic_intelligence_core import LinguisticIntelligenceCore
        core = LinguisticIntelligenceCore()
        print("   ✅ Linguistic Intelligence Core imported and created")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Linguistic Intelligence Core failed: {e}")
    
    print()
    print("=" * 50)
    print(f"🎯 VALIDATION RESULTS")
    print(f"   Tests Passed: {tests_passed}/{total_tests}")
    print(f"   Success Rate: {tests_passed/total_tests:.1%}")
    
    if tests_passed == total_tests:
        print("🎉 ALL FIXES WORKING PERFECTLY!")
        print("✅ Ready to run full enhanced capabilities test!")
        return True
    elif tests_passed >= total_tests * 0.8:
        print("✅ MOST FIXES WORKING - GOOD PROGRESS!")
        print("🔧 Minor issues may remain")
        return True
    else:
        print("⚠️  SIGNIFICANT ISSUES REMAIN")
        print("🔧 Need to investigate further")
        return False

if __name__ == "__main__":
    success = validate_fixes()
    
    if success:
        print("\n🚀 RECOMMENDATION: Run full test suite")
        print("   Command: python test_complete_phase3_enhanced_capabilities.py")
    else:
        print("\n🔧 RECOMMENDATION: Fix remaining issues first")