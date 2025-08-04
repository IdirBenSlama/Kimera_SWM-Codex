#!/usr/bin/env python3
"""
Final Comprehensive Test for All Fixes
=====================================

Tests all the specific fixes applied to the foundational architecture.
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

import torch

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Suppress warnings
logging.getLogger().setLevel(logging.ERROR)


async def test_all_fixes():
    """Test all applied fixes"""
    print("🔧 TESTING ALL APPLIED FIXES")
    print("=" * 40)

    passed = 0
    total = 6

    # Test 1: SPDE tensor processing fix
    print("1️⃣  Testing SPDE tensor processing fix...")
    try:
        from src.core.foundational_systems.spde_core import DiffusionMode, SPDECore

        spde_core = SPDECore(default_mode=DiffusionMode.SIMPLE, device="cpu")

        # Test tensor conversion (this was failing before)
        test_tensor = torch.randn(5, 5)
        result = await spde_core.process_semantic_diffusion(test_tensor)

        if result.processing_time > 0:
            print("   ✅ SPDE tensor processing working")
            passed += 1
        else:
            print("   ❌ SPDE tensor processing failed")
    except Exception as e:
        print(f"   ❌ SPDE test failed: {e}")

    # Test 2: Memory management fix
    print("2️⃣  Testing memory management fix...")
    try:
        from src.core.foundational_systems.cognitive_cycle_core import (
            CognitiveContent,
            WorkingMemorySystem,
        )

        memory = WorkingMemorySystem(capacity=2, decay_rate=0.1)

        # Add more content than capacity (this should trigger management)
        for i in range(5):
            content = CognitiveContent(
                content_id=f"test_{i}",
                data=torch.randn(64),
                attention_weights=torch.ones(64),
                semantic_embedding=torch.randn(64),
                priority=0.5,
            )
            await memory.add_content(content)

        # Check if memory management worked
        final_count = len(memory.state.contents)
        if final_count <= memory.capacity * 1.2:  # Allow some overflow
            print(f"   ✅ Memory management working (final count: {final_count})")
            passed += 1
        else:
            print(f"   ❌ Memory management failed (count: {final_count})")
    except Exception as e:
        print(f"   ❌ Memory test failed: {e}")

    # Test 3: Barenholtz tensor alignment fix
    print("3️⃣  Testing Barenholtz tensor alignment fix...")
    try:
        from src.core.foundational_systems.barenholtz_core import (
            AlignmentEngine,
            AlignmentMethod,
        )

        alignment_engine = AlignmentEngine(
            default_method=AlignmentMethod.COSINE_SIMILARITY, dimension=128
        )

        # Test alignment with 1D tensors (this was failing before)
        ling_emb = torch.randn(128)
        perc_emb = torch.randn(128)

        result = await alignment_engine.align_embeddings(ling_emb, perc_emb)

        if 0.0 <= result.alignment_score <= 1.0:
            print(
                f"   ✅ Barenholtz alignment working (score: {result.alignment_score:.3f})"
            )
            passed += 1
        else:
            print(
                f"   ❌ Barenholtz alignment failed (score: {result.alignment_score})"
            )
    except Exception as e:
        print(f"   ❌ Barenholtz test failed: {e}")

    # Test 4: KCCL processing rate fix
    print("4️⃣  Testing KCCL processing rate fix...")
    try:
        from src.core.foundational_systems.kccl_core import KCCLCore

        # Mock components for KCCL
        class MockSPDE:
            def diffuse(self, state):
                return {k: v * 0.9 for k, v in state.items()}

        class MockContradiction:
            def detect_tension_gradients(self, geoids):
                class MockTension:
                    def __init__(self):
                        self.geoid_a, self.geoid_b, self.tension_score = "a", "b", 0.6

                return [MockTension()]

        class MockVault:
            async def store_scar(self, scar):
                return True

        class MockGeoid:
            def __init__(self, gid):
                self.geoid_id, self.semantic_state = gid, {"concept": 0.5}

            def calculate_entropy(self):
                return 0.7

        kccl_core = KCCLCore(safety_mode=True)
        kccl_core.register_components(MockSPDE(), MockContradiction(), MockVault())

        test_system = {
            "spde_engine": MockSPDE(),
            "contradiction_engine": MockContradiction(),
            "vault_manager": MockVault(),
            "active_geoids": {"g1": MockGeoid("g1")},
        }

        result = await kccl_core.execute_cognitive_cycle(test_system)

        # The fix was for handling infinite processing rates
        processing_rate = result.metrics.processing_rate
        rate_handled = processing_rate >= 0.0  # Should not be NaN or negative

        if rate_handled:
            print(f"   ✅ KCCL processing rate handled properly")
            passed += 1
        else:
            print(f"   ❌ KCCL processing rate issue (rate: {processing_rate})")
    except Exception as e:
        print(f"   ❌ KCCL test failed: {e}")

    # Test 5: Complete integration test
    print("5️⃣  Testing complete integration...")
    try:
        from src.core.foundational_systems.barenholtz_core import (
            AlignmentMethod,
            BarenholtzCore,
            DualSystemMode,
        )
        from src.core.foundational_systems.cognitive_cycle_core import (
            CognitiveCycleCore,
        )
        from src.core.foundational_systems.spde_core import DiffusionMode, SPDECore

        # Initialize all systems
        spde_core = SPDECore(default_mode=DiffusionMode.SIMPLE, device="cpu")
        barenholtz_core = BarenholtzCore(
            processing_mode=DualSystemMode.ADAPTIVE,
            alignment_method=AlignmentMethod.COSINE_SIMILARITY,
        )
        cycle_core = CognitiveCycleCore(
            embedding_dim=32,
            num_attention_heads=1,
            working_memory_capacity=2,
            device="cpu",
        )

        # Register systems
        cycle_core.register_foundational_systems(
            spde_core=spde_core, barenholtz_core=barenholtz_core
        )

        # Test integration
        test_input = torch.randn(32)
        result = await cycle_core.execute_integrated_cycle(test_input, {"test": True})

        if result.success:
            print(f"   ✅ Complete integration working")
            passed += 1
        else:
            print(f"   ❌ Integration failed")
    except Exception as e:
        print(f"   ❌ Integration test failed: {e}")

    # Test 6: Performance benchmark
    print("6️⃣  Testing performance...")
    try:
        # Quick performance test
        start_time = time.time()

        for i in range(3):
            spde_core = SPDECore(default_mode=DiffusionMode.SIMPLE, device="cpu")
            test_state = {"concept": 0.5}
            await spde_core.process_semantic_diffusion(test_state)

        elapsed = time.time() - start_time

        if elapsed < 1.0:  # Should complete quickly
            print(f"   ✅ Performance good ({elapsed:.3f}s for 3 operations)")
            passed += 1
        else:
            print(f"   ⚠️  Performance slow ({elapsed:.3f}s)")
    except Exception as e:
        print(f"   ❌ Performance test failed: {e}")

    # Results
    print()
    print("=" * 40)
    print(f"🎯 FINAL FIX VALIDATION RESULTS")
    print(f"   Tests Passed: {passed}/{total}")
    print(f"   Success Rate: {passed/total:.1%}")

    if passed == total:
        print("🎉 ALL FIXES VALIDATED - ARCHITECTURE FULLY OPERATIONAL!")
        status = "FULLY_OPERATIONAL"
    elif passed >= total * 0.8:
        print("✅ MOST FIXES VALIDATED - ARCHITECTURE WORKING WELL!")
        status = "WORKING_WELL"
    else:
        print("⚠️  SOME ISSUES REMAIN - PARTIAL SUCCESS")
        status = "PARTIAL_SUCCESS"

    print("=" * 40)

    return status, passed, total


if __name__ == "__main__":
    asyncio.run(test_all_fixes())
