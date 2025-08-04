#!/usr/bin/env python3
"""
Debug learning core issues
"""

import asyncio
import traceback

import torch


async def debug_learning_issue():
    """Debug the specific learning core issue"""
    print("🔍 DEBUGGING LEARNING CORE ISSUES")
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

        # Create learning data with patterns
        learning_data = (
            torch.sin(torch.linspace(0, 6 * 3.14159, 100)) + torch.randn(100) * 0.2
        )
        print(f"   Learning data shape: {learning_data.shape}")

        learning_result = await learning_core.learn_unsupervised(
            learning_data,
            learning_mode=LearningMode.THERMODYNAMIC_ORG,
            context={"learning_test": True, "pattern_complexity": 0.7},
        )

        print("❌ This should have failed")
        print(f"   Result efficiency: {learning_result.learning_efficiency}")

    except Exception as e:
        print(f"✅ Found the error: {e}")
        print("\nFull traceback:")
        traceback.print_exc()

        # Check for specific errors
        if "exp()" in str(e):
            print("\n🎯 CONFIRMED: exp() function issue")
        elif "tensor" in str(e).lower():
            print("\n🎯 CONFIRMED: Tensor operation issue")
        else:
            print(f"\n⚠️  Error type: {type(e).__name__}")


if __name__ == "__main__":
    asyncio.run(debug_learning_issue())
