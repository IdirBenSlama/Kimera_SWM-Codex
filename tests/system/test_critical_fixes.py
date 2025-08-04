#!/usr/bin/env python3
"""
Quick test to validate our critical fixes work
"""

import asyncio

import torch


async def test_critical_fixes():
    """Test our critical fixes"""
    print("🧪 TESTING CRITICAL FIXES")
    print("=" * 40)

    passed = 0
    total = 4

    # Test 1: Understanding Core tensor dimension fix
    print("1️⃣  Testing Understanding Core tensor fix...")
    try:
        from src.core.enhanced_capabilities.understanding_core import (
            UnderstandingCore,
            UnderstandingType,
        )

        understanding_core = UnderstandingCore()
        test_content = "Test multimodal understanding"

        result = await understanding_core.understand(
            test_content,
            understanding_type=UnderstandingType.SEMANTIC,
            context={"test_mode": True},
        )

        if result.success:
            print("   ✅ Understanding Core working")
            passed += 1
        else:
            print("   ❌ Understanding Core failed")
    except Exception as e:
        print(f"   ❌ Understanding Core error: {e}")

    # Test 2: Consciousness Core boolean tensor fix
    print("2️⃣  Testing Consciousness Core boolean fix...")
    try:
        from src.core.enhanced_capabilities.consciousness_core import ConsciousnessCore

        consciousness_core = ConsciousnessCore()
        cognitive_state = torch.randn(128)

        # This should NOT raise the boolean tensor error anymore
        signature = await consciousness_core.detect_consciousness(
            cognitive_state, None  # energy_field is None
        )

        if signature.success:
            print("   ✅ Consciousness Core working")
            passed += 1
        else:
            print("   ❌ Consciousness Core failed")
    except Exception as e:
        print(f"   ❌ Consciousness Core error: {e}")

    # Test 3: Language detection fix
    print("3️⃣  Testing Language detection fix...")
    try:
        from src.core.enhanced_capabilities.linguistic_intelligence_core import (
            LinguisticIntelligenceCore,
        )

        linguistic_core = LinguisticIntelligenceCore()

        # Test language detection
        en_detected = linguistic_core._detect_language("hello world")
        es_detected = linguistic_core._detect_language("hola mundo")
        fr_detected = linguistic_core._detect_language("bonjour monde")

        if en_detected == "en" and es_detected == "es" and fr_detected == "fr":
            print("   ✅ Language detection working")
            passed += 1
        else:
            print(
                f"   ❌ Language detection failed: en={en_detected}, es={es_detected}, fr={fr_detected}"
            )
    except Exception as e:
        print(f"   ❌ Language detection error: {e}")

    # Test 4: Learning Core basic functionality
    print("4️⃣  Testing Learning Core...")
    try:
        from src.core.enhanced_capabilities.learning_core import LearningCore

        learning_core = LearningCore()
        test_data = torch.randn(50)

        result = await learning_core.learn_unsupervised(test_data)

        if result.success:
            print("   ✅ Learning Core working")
            passed += 1
        else:
            print("   ❌ Learning Core failed")
    except Exception as e:
        print(f"   ❌ Learning Core error: {e}")

    print()
    print(f"🎯 CRITICAL FIXES TEST: {passed}/{total} passed ({passed/total:.1%})")

    if passed >= 3:
        print("✅ MOST CRITICAL FIXES WORKING!")
        return True
    else:
        print("⚠️  SOME CRITICAL ISSUES REMAIN")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_critical_fixes())

    if success:
        print("\n🚀 Ready to run full enhanced capabilities test!")
    else:
        print("\n🔧 Need to fix remaining critical issues")
