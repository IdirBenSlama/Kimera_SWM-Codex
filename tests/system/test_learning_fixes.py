#!/usr/bin/env python3
"""
Test learning core fixes for unsupervised learning and knowledge integration
"""

import asyncio

import torch


async def test_learning_fixes():
    """Test our learning core fixes"""
    print("🧪 TESTING LEARNING CORE FIXES")
    print("=" * 50)

    try:
        from src.core.enhanced_capabilities.learning_core import (
            LearningCore,
            LearningMode,
        )

        learning_core = LearningCore(
            default_learning_mode=LearningMode.THERMODYNAMIC_ORG,
            learning_threshold=0.5,
            device="cpu",
        )
        print("✅ Learning Core created")

        # Test 1: Unsupervised Learning with patterns
        print("\n1️⃣  Testing Unsupervised Learning...")
        learning_data = (
            torch.sin(torch.linspace(0, 6 * 3.14159, 100)) + torch.randn(100) * 0.2
        )

        learning_result = await learning_core.learn_unsupervised(
            learning_data,
            learning_mode=LearningMode.THERMODYNAMIC_ORG,
            context={"learning_test": True, "pattern_complexity": 0.7},
        )

        print(f"   Success: {learning_result.success}")
        print(f"   Learning efficiency: {learning_result.learning_efficiency:.4f}")
        print(f"   Discovered patterns: {len(learning_result.discovered_patterns)}")

        test1_pass = (
            learning_result.success
            and learning_result.learning_efficiency > 0.0
            and len(learning_result.discovered_patterns) >= 0
        )

        if test1_pass:
            print("   ✅ Unsupervised learning test PASSES!")
        else:
            print("   ❌ Unsupervised learning test FAILS")

        # Test 2: Knowledge Integration
        print("\n2️⃣  Testing Knowledge Integration...")
        integration_data_1 = torch.sin(torch.linspace(0, 2 * 3.14159, 60)) * 0.7
        integration_data_2 = (
            torch.sin(torch.linspace(0, 2 * 3.14159, 60)) * 0.9
        )  # Similar but stronger

        # First learning session
        first_result = await learning_core.learn_unsupervised(integration_data_1)
        print(
            f"   First session - Success: {first_result.success}, Integration: {first_result.knowledge_integration:.4f}"
        )

        # Second learning session (should integrate with first)
        second_result = await learning_core.learn_unsupervised(integration_data_2)
        print(
            f"   Second session - Success: {second_result.success}, Integration: {second_result.knowledge_integration:.4f}"
        )

        test2_pass = (
            first_result.success
            and second_result.success
            and second_result.knowledge_integration > 0.0
        )

        if test2_pass:
            print("   ✅ Knowledge integration test PASSES!")
        else:
            print("   ❌ Knowledge integration test FAILS")

        # Summary
        print("\n" + "=" * 50)
        passed_tests = sum([test1_pass, test2_pass])
        print(f"🎯 LEARNING CORE FIXES: {passed_tests}/2 tests passed")

        if passed_tests == 2:
            print("🎉 ALL LEARNING CORE FIXES WORKING!")
            return True
        else:
            print("⚠️  Some learning core issues remain")
            return False

    except Exception as e:
        print(f"❌ Error testing learning core: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_learning_fixes())

    if success:
        print("\n🚀 Learning Core ready for full enhanced capabilities test!")
    else:
        print("\n🔧 Need additional learning core fixes")
