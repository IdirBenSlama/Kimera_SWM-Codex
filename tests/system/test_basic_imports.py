#!/usr/bin/env python3
"""
Basic Import Test
================

Tests if the cognitive components can be imported without hanging.
"""

import sys
import time


def test_basic_imports():
    """Test basic imports of cognitive components"""
    print("🧪 BASIC IMPORT TEST")
    print("=" * 30)

    try:
        print("1️⃣  Testing Understanding Core import...")
        start = time.time()
        from src.core.enhanced_capabilities.understanding_core import UnderstandingCore

        print(f"   ✅ Understanding Core imported ({time.time() - start:.3f}s)")

        print("2️⃣  Testing Consciousness Core import...")
        start = time.time()
        from src.core.enhanced_capabilities.consciousness_core import ConsciousnessCore

        print(f"   ✅ Consciousness Core imported ({time.time() - start:.3f}s)")

        print("3️⃣  Testing Learning Core import...")
        start = time.time()
        from src.core.enhanced_capabilities.learning_core import LearningCore

        print(f"   ✅ Learning Core imported ({time.time() - start:.3f}s)")

        print("4️⃣  Testing instantiation...")
        start = time.time()
        understanding = UnderstandingCore()
        print(f"   ✅ Understanding Core instantiated ({time.time() - start:.3f}s)")

        start = time.time()
        consciousness = ConsciousnessCore()
        print(f"   ✅ Consciousness Core instantiated ({time.time() - start:.3f}s)")

        start = time.time()
        learning = LearningCore()
        print(f"   ✅ Learning Core instantiated ({time.time() - start:.3f}s)")

        print("\n🎉 ALL IMPORTS AND INSTANTIATION SUCCESSFUL!")
        return True

    except Exception as e:
        print(f"❌ Import/instantiation error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🚀 Testing Basic Imports")
    success = test_basic_imports()

    if success:
        print("\n✅ Ready for async testing")
    else:
        print("\n❌ Import issues need resolution")
