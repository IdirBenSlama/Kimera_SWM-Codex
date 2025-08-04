#!/usr/bin/env python3
"""
Test suite for Master Cognitive Architecture
============================================

Comprehensive testing of the unified cognitive architecture system
integrating all Phase 1-3 components.
"""

import asyncio
import json
import time
from typing import Any, Dict


async def test_master_cognitive_architecture():
    """Test the complete Master Cognitive Architecture system"""
    print("🧠 TESTING MASTER COGNITIVE ARCHITECTURE")
    print("=" * 60)

    try:
        from src.core.master_cognitive_architecture import (
            CognitiveRequest,
            CognitiveWorkflow,
            MasterCognitiveArchitecture,
            ProcessingMode,
            create_master_architecture,
            quick_cognitive_processing,
        )

        test_results = {}
        total_tests = 0
        passed_tests = 0

        # Test 1: Architecture Creation
        print("1️⃣  Testing Architecture Creation...")
        total_tests += 1
        try:
            architecture = MasterCognitiveArchitecture(
                device="cpu", enable_gpu=False, processing_mode=ProcessingMode.ADAPTIVE
            )
            print(f"   ✅ Architecture created - System ID: {architecture.system_id}")
            print(f"   Device: {architecture.device}")
            print(f"   Processing mode: {architecture.processing_mode.value}")
            passed_tests += 1
            test_results["architecture_creation"] = "PASSED"
        except Exception as e:
            print(f"   ❌ Architecture creation failed: {e}")
            test_results["architecture_creation"] = f"FAILED: {e}"

        # Test 2: Component Initialization
        print("\n2️⃣  Testing Component Initialization...")
        total_tests += 1
        try:
            init_start = time.time()
            success = await architecture.initialize_architecture()
            init_time = time.time() - init_start

            if success:
                print(f"   ✅ Architecture initialized successfully")
                print(f"   Initialization time: {init_time:.2f}s")
                print(
                    f"   Components registered: {len(architecture.component_registry)}"
                )

                # List all components
                print("   Registered components:")
                for component_name in architecture.component_registry.keys():
                    print(f"     - {component_name}")

                passed_tests += 1
                test_results["component_initialization"] = "PASSED"
            else:
                print(f"   ❌ Architecture initialization failed")
                test_results["component_initialization"] = (
                    "FAILED: Initialization returned False"
                )
        except Exception as e:
            print(f"   ❌ Component initialization failed: {e}")
            test_results["component_initialization"] = f"FAILED: {e}"

        # Test 3: System Status
        print("\n3️⃣  Testing System Status...")
        total_tests += 1
        try:
            status = architecture.get_system_status()

            print(f"   System State: {status['state']}")
            print(f"   Uptime: {status['uptime']:.2f}s")
            print(f"   Total Components: {status['components']['total']}")
            print(f"   Processing Mode: {status.get('processing_mode', 'N/A')}")
            print(f"   Success Rate: {status['performance']['success_rate']:.1%}")

            if status["state"] == "ready" and status["components"]["total"] > 0:
                passed_tests += 1
                test_results["system_status"] = "PASSED"
            else:
                test_results["system_status"] = (
                    "FAILED: System not ready or no components"
                )
        except Exception as e:
            print(f"   ❌ System status check failed: {e}")
            test_results["system_status"] = f"FAILED: {e}"

        # Test 4: Basic Cognitive Processing
        print("\n4️⃣  Testing Basic Cognitive Processing...")
        total_tests += 1
        try:
            request = CognitiveRequest(
                request_id="test_basic_001",
                workflow_type=CognitiveWorkflow.BASIC_COGNITION,
                input_data="This is a test of the cognitive architecture system.",
                context={"test_mode": True, "priority": "high"},
            )

            print(f"   Processing request: {request.request_id}")
            print(f"   Workflow: {request.workflow_type.value}")
            print(f"   Input: {request.input_data[:50]}...")

            response = await architecture.process_cognitive_request(request)

            print(f"   Response Success: {response.success}")
            print(f"   Processing Time: {response.processing_time:.3f}s")
            print(f"   Quality Score: {response.quality_score:.3f}")
            print(f"   Confidence: {response.confidence:.3f}")
            print(f"   Components Used: {', '.join(response.components_used)}")

            if response.success and response.quality_score > 0:
                passed_tests += 1
                test_results["basic_cognitive_processing"] = "PASSED"
            else:
                test_results["basic_cognitive_processing"] = (
                    "FAILED: Processing unsuccessful or zero quality"
                )
        except Exception as e:
            print(f"   ❌ Basic cognitive processing failed: {e}")
            test_results["basic_cognitive_processing"] = f"FAILED: {e}"

        # Test 5: Understanding Core Integration
        print("\n5️⃣  Testing Understanding Core Integration...")
        total_tests += 1
        try:
            request = CognitiveRequest(
                request_id="test_understanding_001",
                workflow_type=CognitiveWorkflow.DEEP_UNDERSTANDING,
                input_data="Analyze this complex philosophical statement about consciousness and reality.",
                context={"depth": "deep", "focus": "philosophical"},
            )

            response = await architecture.process_cognitive_request(request)

            has_understanding = bool(response.understanding)
            understanding_quality = (
                response.understanding.get("quality", 0.0)
                if response.understanding
                else 0.0
            )

            print(f"   Understanding Analysis: {has_understanding}")
            if has_understanding:
                print(f"   Understanding Quality: {understanding_quality:.3f}")
                print(
                    f"   Understanding Type: {response.understanding.get('type', 'N/A')}"
                )

            if response.success and has_understanding:
                passed_tests += 1
                test_results["understanding_integration"] = "PASSED"
            else:
                test_results["understanding_integration"] = (
                    "FAILED: No understanding analysis"
                )
        except Exception as e:
            print(f"   ❌ Understanding integration test failed: {e}")
            test_results["understanding_integration"] = f"FAILED: {e}"

        # Test 6: Consciousness Core Integration
        print("\n6️⃣  Testing Consciousness Core Integration...")
        total_tests += 1
        try:
            request = CognitiveRequest(
                request_id="test_consciousness_001",
                workflow_type=CognitiveWorkflow.CONSCIOUSNESS_ANALYSIS,
                input_data="Examine the consciousness patterns in this cognitive state.",
                context={"analysis_depth": "full", "consciousness_focus": True},
            )

            response = await architecture.process_cognitive_request(request)

            has_consciousness = bool(response.consciousness)
            consciousness_prob = (
                response.consciousness.get("probability", 0.0)
                if response.consciousness
                else 0.0
            )

            print(f"   Consciousness Analysis: {has_consciousness}")
            if has_consciousness:
                print(f"   Consciousness Probability: {consciousness_prob:.3f}")
                print(
                    f"   Consciousness State: {response.consciousness.get('state', 'N/A')}"
                )
                print(
                    f"   Signature Strength: {response.consciousness.get('strength', 0.0):.3f}"
                )

            if response.success and has_consciousness:
                passed_tests += 1
                test_results["consciousness_integration"] = "PASSED"
            else:
                test_results["consciousness_integration"] = (
                    "FAILED: No consciousness analysis"
                )
        except Exception as e:
            print(f"   ❌ Consciousness integration test failed: {e}")
            test_results["consciousness_integration"] = f"FAILED: {e}"

        # Test 7: Multiple Concurrent Requests
        print("\n7️⃣  Testing Concurrent Processing...")
        total_tests += 1
        try:
            requests = [
                CognitiveRequest(
                    request_id=f"concurrent_test_{i}",
                    workflow_type=CognitiveWorkflow.BASIC_COGNITION,
                    input_data=f"Concurrent test request number {i}",
                    context={"batch": "concurrent_test"},
                )
                for i in range(3)
            ]

            print(f"   Processing {len(requests)} concurrent requests...")

            # Process requests concurrently
            tasks = [architecture.process_cognitive_request(req) for req in requests]
            responses = await asyncio.gather(*tasks)

            successful_responses = sum(1 for r in responses if r.success)
            avg_processing_time = sum(r.processing_time for r in responses) / len(
                responses
            )

            print(f"   Successful responses: {successful_responses}/{len(requests)}")
            print(f"   Average processing time: {avg_processing_time:.3f}s")

            if successful_responses >= len(requests) * 0.8:  # 80% success rate
                passed_tests += 1
                test_results["concurrent_processing"] = "PASSED"
            else:
                test_results["concurrent_processing"] = "FAILED: Low success rate"
        except Exception as e:
            print(f"   ❌ Concurrent processing test failed: {e}")
            test_results["concurrent_processing"] = f"FAILED: {e}"

        # Test 8: System Metrics and Performance
        print("\n8️⃣  Testing System Metrics...")
        total_tests += 1
        try:
            # Get updated status after processing
            final_status = architecture.get_system_status()

            print(
                f"   Total Operations: {final_status['performance']['total_operations']}"
            )
            print(f"   Success Rate: {final_status['performance']['success_rate']:.1%}")
            print(
                f"   Average Processing Time: {final_status['performance']['average_processing_time']:.3f}s"
            )
            print(
                f"   Component Health: {len(final_status['health']['component_health'])} components"
            )

            if (
                final_status["performance"]["total_operations"] > 0
                and final_status["performance"]["success_rate"] > 0.5
            ):
                passed_tests += 1
                test_results["system_metrics"] = "PASSED"
            else:
                test_results["system_metrics"] = (
                    "FAILED: Insufficient operations or low success rate"
                )
        except Exception as e:
            print(f"   ❌ System metrics test failed: {e}")
            test_results["system_metrics"] = f"FAILED: {e}"

        # Test 9: Quick Processing Function
        print("\n9️⃣  Testing Quick Processing Function...")
        total_tests += 1
        try:
            print("   Testing quick_cognitive_processing utility...")

            quick_response = await quick_cognitive_processing(
                input_data="Quick test of the cognitive system",
                workflow=CognitiveWorkflow.BASIC_COGNITION,
                context={"quick_test": True},
            )

            print(f"   Quick Processing Success: {quick_response.success}")
            print(f"   Quality Score: {quick_response.quality_score:.3f}")

            if quick_response.success:
                passed_tests += 1
                test_results["quick_processing"] = "PASSED"
            else:
                test_results["quick_processing"] = "FAILED: Processing unsuccessful"
        except Exception as e:
            print(f"   ❌ Quick processing test failed: {e}")
            test_results["quick_processing"] = f"FAILED: {e}"

        # Test 10: Architecture Shutdown
        print("\n🔟 Testing Architecture Shutdown...")
        total_tests += 1
        try:
            print("   Initiating graceful shutdown...")
            await architecture.shutdown()

            final_status = architecture.get_system_status()
            print(f"   Final state: {final_status['state']}")

            if final_status["state"] == "shutdown":
                passed_tests += 1
                test_results["architecture_shutdown"] = "PASSED"
            else:
                test_results["architecture_shutdown"] = "FAILED: Not in shutdown state"
        except Exception as e:
            print(f"   ❌ Architecture shutdown failed: {e}")
            test_results["architecture_shutdown"] = f"FAILED: {e}"

        # Final Results
        print("\n" + "=" * 60)
        print("🎯 MASTER COGNITIVE ARCHITECTURE TEST RESULTS")
        print("=" * 60)

        success_rate = passed_tests / total_tests
        print(f"Tests Passed: {passed_tests}/{total_tests}")
        print(f"Success Rate: {success_rate:.1%}")

        if success_rate >= 0.8:
            print("🎉 MASTER COGNITIVE ARCHITECTURE TESTS PASSED!")
            print("✅ System ready for production deployment!")
        elif success_rate >= 0.6:
            print("⚠️  MASTER COGNITIVE ARCHITECTURE PARTIALLY FUNCTIONAL")
            print("🔧 Some components need attention")
        else:
            print("❌ MASTER COGNITIVE ARCHITECTURE TESTS FAILED")
            print("🛠️  Significant issues need resolution")

        print("\nDetailed Results:")
        for test_name, result in test_results.items():
            status_icon = "✅" if result == "PASSED" else "❌"
            print(f"  {status_icon} {test_name}: {result}")

        return success_rate >= 0.8

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all cognitive components are properly installed")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_master_cognitive_architecture())

    if success:
        print("\n🚀 Ready to proceed with Phase 4 enhancements!")
    else:
        print("\n🔧 Phase 4 components need fixes before proceeding")
