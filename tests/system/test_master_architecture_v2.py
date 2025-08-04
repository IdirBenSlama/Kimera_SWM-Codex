#!/usr/bin/env python3
"""
Test Master Cognitive Architecture V2 with lazy imports
"""

import asyncio


async def test_master_architecture_v2():
    """Test the V2 master architecture with lazy loading"""
    print("🧠 TESTING MASTER COGNITIVE ARCHITECTURE V2")
    print("=" * 55)

    try:
        from src.core.master_cognitive_architecture_v2 import (
            CognitiveRequest,
            CognitiveWorkflow,
            MasterCognitiveArchitecture,
            ProcessingMode,
            create_master_architecture_v2,
        )

        passed_tests = 0
        total_tests = 0

        # Test 1: Basic Creation
        print("1️⃣  Testing Architecture Creation...")
        total_tests += 1
        try:
            architecture = MasterCognitiveArchitecture(
                device="cpu", enable_gpu=False, processing_mode=ProcessingMode.ADAPTIVE
            )
            print(f"   ✅ Architecture created: {architecture.system_id}")
            print(f"   State: {architecture.state.value}")
            passed_tests += 1
        except Exception as e:
            print(f"   ❌ Creation failed: {e}")

        # Test 2: Initialization
        print("\n2️⃣  Testing Initialization...")
        total_tests += 1
        try:
            success = await architecture.initialize_architecture()
            if success:
                print(f"   ✅ Initialization successful")
                print(f"   Components loaded: {len(architecture.component_registry)}")

                # List components
                for component_name in architecture.component_registry.keys():
                    print(f"     - {component_name}")
                passed_tests += 1
            else:
                print(f"   ❌ Initialization failed")
        except Exception as e:
            print(f"   ❌ Initialization error: {e}")

        # Test 3: System Status
        print("\n3️⃣  Testing System Status...")
        total_tests += 1
        try:
            status = architecture.get_system_status()
            print(f"   State: {status['state']}")
            print(f"   Device: {status['device']}")
            print(f"   Components: {status['components']['total']}")
            print(f"   Success Rate: {status['performance']['success_rate']:.1%}")

            if status["state"] == "ready":
                passed_tests += 1
                print("   ✅ System status check passed")
            else:
                print("   ❌ System not ready")
        except Exception as e:
            print(f"   ❌ Status check failed: {e}")

        # Test 4: Basic Processing
        print("\n4️⃣  Testing Basic Processing...")
        total_tests += 1
        try:
            request = CognitiveRequest(
                request_id="test_v2_001",
                workflow_type=CognitiveWorkflow.BASIC_COGNITION,
                input_data="Test cognitive processing with V2 architecture.",
                context={"test": True},
            )

            response = await architecture.process_cognitive_request(request)

            print(f"   Success: {response.success}")
            print(f"   Processing Time: {response.processing_time:.3f}s")
            print(f"   Quality Score: {response.quality_score:.3f}")
            print(f"   Components Used: {', '.join(response.components_used)}")

            if response.success:
                passed_tests += 1
                print("   ✅ Basic processing passed")
            else:
                print("   ❌ Basic processing failed")
        except Exception as e:
            print(f"   ❌ Processing error: {e}")

        # Test 5: Understanding Integration
        print("\n5️⃣  Testing Understanding Integration...")
        total_tests += 1
        try:
            request = CognitiveRequest(
                request_id="test_understanding_v2",
                workflow_type=CognitiveWorkflow.DEEP_UNDERSTANDING,
                input_data="Analyze this complex statement about cognitive architectures.",
                context={"depth": "deep"},
            )

            response = await architecture.process_cognitive_request(request)

            has_understanding = bool(response.understanding)
            print(f"   Understanding Analysis: {has_understanding}")

            if has_understanding:
                print(f"   Quality: {response.understanding.get('quality', 0.0):.3f}")
                print(f"   Type: {response.understanding.get('type', 'N/A')}")
                passed_tests += 1
                print("   ✅ Understanding integration passed")
            else:
                print("   ❌ No understanding analysis")
        except Exception as e:
            print(f"   ❌ Understanding test error: {e}")

        # Test 6: Cleanup
        print("\n6️⃣  Testing Shutdown...")
        total_tests += 1
        try:
            await architecture.shutdown()
            status = architecture.get_system_status()

            if status["state"] == "shutdown":
                passed_tests += 1
                print("   ✅ Shutdown completed successfully")
            else:
                print("   ❌ Shutdown incomplete")
        except Exception as e:
            print(f"   ❌ Shutdown error: {e}")

        # Results
        print("\n" + "=" * 55)
        print("🎯 MASTER ARCHITECTURE V2 TEST RESULTS")
        print("=" * 55)

        success_rate = passed_tests / total_tests
        print(f"Tests Passed: {passed_tests}/{total_tests}")
        print(f"Success Rate: {success_rate:.1%}")

        if success_rate >= 0.8:
            print("🎉 MASTER ARCHITECTURE V2 TESTS PASSED!")
            print("✅ Ready for Phase 4 development!")
        elif success_rate >= 0.5:
            print("⚠️  MASTER ARCHITECTURE V2 PARTIALLY FUNCTIONAL")
            print("🔧 Some improvements needed")
        else:
            print("❌ MASTER ARCHITECTURE V2 NEEDS WORK")
            print("🛠️  Significant issues to resolve")

        return success_rate >= 0.8

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_master_architecture_v2())

    if success:
        print("\n🚀 Phase 4 Master Architecture is operational!")
    else:
        print("\n🔧 Phase 4 needs additional work")
