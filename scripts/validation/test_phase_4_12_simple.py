#!/usr/bin/env python3
"""
Simple Phase 4.12 Integration Test
=================================

Basic validation to test Phase 4.12 Response Generation integration.
"""

import sys
import asyncio
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from core.kimera_system import get_kimera_system


async def test_phase_4_12():
    """Simple test for Phase 4.12 integration"""
    print("🔬 Testing Phase 4.12: Response Generation and Security")
    print("=" * 60)

    try:
        # Test 1: KimeraSystem initialization
        print("\n1️⃣ Initializing KimeraSystem...")
        kimera = get_kimera_system()
        print("✅ KimeraSystem initialized")

        # Test 2: Check response generation component
        print("\n2️⃣ Checking response generation component...")
        response_gen = kimera.get_response_generation()

        if response_gen:
            print("✅ Response Generation component loaded")

            # Test 3: Check orchestrator status
            print("\n3️⃣ Checking orchestrator status...")
            try:
                status = response_gen.get_orchestrator_status()
                print(f"   Status: {status.get('status', 'unknown')}")
                print(f"   Version: {status.get('version', 'unknown')}")
                print(f"   Components: {len(status.get('system_health', {}).get('components', {}))}")

                # Test 4: Simple response generation
                print("\n4️⃣ Testing simple response generation...")

                from core.response_generation import ResponseGenerationRequest, ResponseGenerationMode

                request = ResponseGenerationRequest(
                    query="Hello, how are you?",
                    mode=ResponseGenerationMode.MINIMAL
                )

                result = await response_gen.generate_response(request)

                print(f"   Response length: {len(result.response.content)} characters")
                print(f"   Quality score: {result.response.quality_score:.3f}")
                print(f"   Processing time: {result.processing_time_ms:.1f}ms")
                print(f"   Valid: {result.response.is_valid()}")

                if result.response.is_valid():
                    print("✅ Response generation successful")
                else:
                    print("❌ Invalid response generated")
                    return False

            except Exception as e:
                print(f"❌ Response generation test failed: {e}")
                return False
        else:
            print("❌ Response Generation component not found")
            return False

        print("\n" + "=" * 60)
        print("✅ Phase 4.12 basic integration test PASSED")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_phase_4_12())
    sys.exit(0 if success else 1)
