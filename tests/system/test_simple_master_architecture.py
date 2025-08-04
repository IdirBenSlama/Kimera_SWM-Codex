#!/usr/bin/env python3
"""
Simple test for Master Cognitive Architecture imports
"""

import sys

print("🧪 TESTING MASTER ARCHITECTURE IMPORTS")
print("=" * 50)

try:
    print("1️⃣  Testing basic imports...")

    # Test import step by step
    print("   - Importing MasterCognitiveArchitecture...")
    from src.core.master_cognitive_architecture import MasterCognitiveArchitecture

    print("   ✅ MasterCognitiveArchitecture imported")

    print("   - Importing enums...")
    from src.core.master_cognitive_architecture import (
        ArchitectureState,
        CognitiveWorkflow,
        ProcessingMode,
    )

    print("   ✅ Enums imported")

    print("   - Importing data classes...")
    from src.core.master_cognitive_architecture import (
        CognitiveRequest,
        CognitiveResponse,
    )

    print("   ✅ Data classes imported")

    print("\n2️⃣  Testing architecture creation...")
    architecture = MasterCognitiveArchitecture(
        device="cpu", enable_gpu=False, processing_mode=ProcessingMode.ADAPTIVE
    )
    print(f"   ✅ Architecture created: {architecture.system_id}")
    print(f"   State: {architecture.state.value}")
    print(f"   Device: {architecture.device}")

    print("\n🎉 ALL BASIC TESTS PASSED!")
    print("✅ Master Cognitive Architecture is importable and creatable")

except ImportError as e:
    print(f"❌ Import error: {e}")
    import traceback

    traceback.print_exc()
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback

    traceback.print_exc()

print("\n🏁 Simple test complete")
