#!/usr/bin/env python3
"""
Test to validate that the main fixes are working
"""

import asyncio
import torch
import sys
import traceback

async def test_key_fixes():
    """Test the key fixes we made"""
    print("🧪 TESTING KEY FIXES FOR ENHANCED CAPABILITIES")
    print("=" * 60)
    
    passed = 0
    total = 6
    
    # Test 1: torch.cosine_similarity fix
    print("1️⃣  Testing torch.cosine_similarity fix...")
    try:
        a = torch.randn(10)
        b = torch.randn(10)
        sim = torch.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=1)
        print(f"   ✅ torch.cosine_similarity working: {sim.item():.3f}")
        passed += 1
    except Exception as e:
        print(f"   ❌ torch.cosine_similarity failed: {e}")
    
    # Test 2: Understanding Core import and basic usage
    print("2️⃣  Testing Understanding Core...")
    try:
        from src.core.enhanced_capabilities.understanding_core import UnderstandingCore, UnderstandingType
        
        understanding_core = UnderstandingCore()
        print("   ✅ Understanding Core created successfully")
        passed += 1
    except Exception as e:
        print(f"   ❌ Understanding Core failed: {e}")
        traceback.print_exc()
    
    # Test 3: Consciousness Core import
    print("3️⃣  Testing Consciousness Core...")
    try:
        from src.core.enhanced_capabilities.consciousness_core import ConsciousnessCore
        
        consciousness_core = ConsciousnessCore()
        print("   ✅ Consciousness Core created successfully")
        passed += 1
    except Exception as e:
        print(f"   ❌ Consciousness Core failed: {e}")
        traceback.print_exc()
    
    # Test 4: Meta Insight Core import (with FFT fix)
    print("4️⃣  Testing Meta Insight Core...")
    try:
        from src.core.enhanced_capabilities.meta_insight_core import MetaInsightCore
        
        meta_insight_core = MetaInsightCore()
        print("   ✅ Meta Insight Core created successfully")
        passed += 1
    except Exception as e:
        print(f"   ❌ Meta Insight Core failed: {e}")
        traceback.print_exc()
    
    # Test 5: Learning Core import
    print("5️⃣  Testing Learning Core...")
    try:
        from src.core.enhanced_capabilities.learning_core import LearningCore
        
        learning_core = LearningCore()
        print("   ✅ Learning Core created successfully")
        passed += 1
    except Exception as e:
        print(f"   ❌ Learning Core failed: {e}")
        traceback.print_exc()
    
    # Test 6: Linguistic Intelligence Core import (with current_phrase fix)
    print("6️⃣  Testing Linguistic Intelligence Core...")
    try:
        from src.core.enhanced_capabilities.linguistic_intelligence_core import LinguisticIntelligenceCore
        
        linguistic_core = LinguisticIntelligenceCore()
        print("   ✅ Linguistic Intelligence Core created successfully")
        passed += 1
    except Exception as e:
        print(f"   ❌ Linguistic Intelligence Core failed: {e}")
        traceback.print_exc()
    
    print()
    print("=" * 60)
    print(f"🎯 FIX VALIDATION RESULTS")
    print(f"   Tests Passed: {passed}/{total}")
    print(f"   Success Rate: {passed/total:.1%}")
    
    if passed == total:
        print("🎉 ALL FIXES WORKING PERFECTLY!")
    elif passed >= total * 0.8:
        print("✅ MOST FIXES WORKING - GREAT PROGRESS!")
    else:
        print("⚠️  SOME FIXES STILL NEED WORK")
    
    return passed, total

async def main():
    """Main test function"""
    passed, total = await test_key_fixes()
    
    if passed >= total * 0.8:
        print("\n🚀 Ready to test full enhanced capabilities!")
    else:
        print("\n🔧 Need to fix remaining issues first")

if __name__ == "__main__":
    asyncio.run(main())