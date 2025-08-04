#!/usr/bin/env python3
"""
Phase 4.13: Testing and Protocols Integration Validation
========================================================

DO-178C Level A validation script for comprehensive testing of the
large-scale testing framework and omnidimensional protocol engine.

This script validates:
- Complete 96-test matrix generation
- Test orchestrator functionality
- Omnidimensional protocol communication
- System integration and health monitoring

Author: KIMERA Development Team
Version: 1.0.0 (DO-178C Level A)
"""

import sys
import os
import asyncio
import json
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from core.kimera_system import KimeraSystem
from utils.kimera_logger import get_logger, LogCategory

logger = get_logger(__name__, LogCategory.SYSTEM)


async def validate_phase_4_13_integration():
    """Comprehensive validation of Phase 4.13 integration"""

    print("🔬 Phase 4.13: Testing and Protocols Integration Validation")
    print("=" * 70)

    validation_results = {
        "timestamp": datetime.now().isoformat(),
        "phase": "4.13",
        "description": "Large-Scale Testing and Omnidimensional Protocols",
        "tests": {}
    }

    try:
        # Initialize KimeraSystem
        print("\n📋 Test 1: KimeraSystem Initialization")
        print("-" * 40)

        kimera = KimeraSystem()
        kimera.initialize()
        print("✅ KimeraSystem initialized successfully")
        validation_results["tests"]["kimera_initialization"] = {"status": "PASSED", "details": "System initialized"}

        # Test Testing and Protocols component access
        print("\n📋 Test 2: Testing and Protocols Component Access")
        print("-" * 50)

        testing_protocols = kimera.get_testing_and_protocols()
        if testing_protocols is not None:
            print("✅ Testing and Protocols component accessible")
            validation_results["tests"]["component_access"] = {"status": "PASSED", "details": "Component accessible"}
        else:
            print("❌ Testing and Protocols component not found")
            validation_results["tests"]["component_access"] = {"status": "FAILED", "details": "Component not accessible"}
            return validation_results

        # Test system status
        print("\n📋 Test 3: System Status and Configuration")
        print("-" * 45)

        try:
            status = testing_protocols.get_system_status()
            print(f"   Integration Active: {status.get('integration_active', False)}")
            print(f"   Testing Enabled: {status.get('config', {}).get('testing_enabled', False)}")
            print(f"   Protocols Enabled: {status.get('config', {}).get('protocols_enabled', False)}")
            print(f"   Health Score: {status.get('health_score', 0.0):.2f}")

            validation_results["tests"]["system_status"] = {
                "status": "PASSED",
                "details": f"System status retrieved: {json.dumps(status, default=str)}"
            }
            print("✅ System status retrieved successfully")

        except Exception as e:
            print(f"❌ Failed to get system status: {e}")
            validation_results["tests"]["system_status"] = {"status": "FAILED", "details": str(e)}

        # Test matrix validation
        print("\n📋 Test 4: Test Matrix Validation")
        print("-" * 40)

        try:
            # Access matrix validator through the integrator
            matrix_validator = testing_protocols.matrix_validator

            # Generate test matrix
            configurations = matrix_validator.generate_complete_matrix()

            if len(configurations) == 96:
                print(f"✅ Test matrix generated: {len(configurations)} configurations")
                validation_results["tests"]["matrix_generation"] = {
                    "status": "PASSED",
                    "details": f"Generated {len(configurations)} test configurations"
                }
            else:
                print(f"❌ Invalid matrix size: {len(configurations)} (expected 96)")
                validation_results["tests"]["matrix_generation"] = {
                    "status": "FAILED",
                    "details": f"Generated {len(configurations)} configurations, expected 96"
                }

            # Validate matrix
            validation_report = matrix_validator.validate_complete_matrix()
            print(f"   Valid configurations: {validation_report.valid_configurations}")
            print(f"   Invalid configurations: {validation_report.invalid_configurations}")
            print(f"   Coverage completeness: {validation_report.coverage_analysis}")

        except Exception as e:
            print(f"❌ Matrix validation failed: {e}")
            validation_results["tests"]["matrix_generation"] = {"status": "FAILED", "details": str(e)}

        # Test protocol engine
        print("\n📋 Test 5: Protocol Engine Functionality")
        print("-" * 45)

        try:
            # Access protocol engine through the integrator
            protocol_engine = testing_protocols.protocol_engine

            # Get engine statistics
            stats = protocol_engine.get_engine_statistics()
            print(f"   Local dimension: {stats.get('local_dimension', 'unknown')}")
            print(f"   Messages sent: {stats.get('messages', {}).get('sent', 0)}")
            print(f"   Active connections: {stats.get('connections', {}).get('active_connections', 0)}")
            print(f"   Registry status: {stats.get('registry_status', {})}")

            validation_results["tests"]["protocol_engine"] = {
                "status": "PASSED",
                "details": f"Protocol engine operational: {json.dumps(stats, default=str)}"
            }
            print("✅ Protocol engine operational")

        except Exception as e:
            print(f"❌ Protocol engine test failed: {e}")
            validation_results["tests"]["protocol_engine"] = {"status": "FAILED", "details": str(e)}

        # Test configuration components
        print("\n📋 Test 6: Configuration Components")
        print("-" * 40)

        try:
            # Test complexity manager
            complexity_manager = testing_protocols.complexity_manager
            complexity_levels = complexity_manager.get_all_levels()
            print(f"   Complexity levels: {len(complexity_levels)} ({[level.value for level in complexity_levels]})")

            # Test input generator
            input_generator = testing_protocols.input_generator
            input_stats = input_generator.get_statistics()
            print(f"   Input types supported: {input_stats.get('supported_types', 0)}")

            # Test context manager
            context_manager = testing_protocols.context_manager
            context_report = context_manager.get_comprehensive_report()
            print(f"   Cognitive contexts: {context_report.get('contexts', {}).keys()}")

            validation_results["tests"]["configuration_components"] = {
                "status": "PASSED",
                "details": "All configuration components accessible"
            }
            print("✅ Configuration components operational")

        except Exception as e:
            print(f"❌ Configuration components test failed: {e}")
            validation_results["tests"]["configuration_components"] = {"status": "FAILED", "details": str(e)}

        # Test system initialization
        print("\n📋 Test 7: System Initialization Test")
        print("-" * 40)

        try:
            # Initialize the testing and protocols system
            await testing_protocols.initialize()

            # Check if initialization was successful
            status = testing_protocols.get_system_status()
            if status.get('integration_active', False):
                print("✅ System initialization successful")
                validation_results["tests"]["system_initialization"] = {
                    "status": "PASSED",
                    "details": "System initialized and active"
                }
            else:
                print("❌ System initialization failed - not active")
                validation_results["tests"]["system_initialization"] = {
                    "status": "FAILED",
                    "details": "System not active after initialization"
                }

        except Exception as e:
            print(f"❌ System initialization failed: {e}")
            validation_results["tests"]["system_initialization"] = {"status": "FAILED", "details": str(e)}

        # Test health monitoring
        print("\n📋 Test 8: Health Monitoring")
        print("-" * 35)

        try:
            health_report = testing_protocols.get_latest_health_report()
            if health_report:
                print(f"   Health score: {health_report.overall_health_score:.2f}")
                print(f"   Alerts: {len(health_report.alerts)}")
                print(f"   Recommendations: {len(health_report.recommendations)}")

                validation_results["tests"]["health_monitoring"] = {
                    "status": "PASSED",
                    "details": f"Health report available with score {health_report.overall_health_score:.2f}"
                }
                print("✅ Health monitoring operational")
            else:
                print("⚠️ No health report available yet")
                validation_results["tests"]["health_monitoring"] = {
                    "status": "WARNING",
                    "details": "No health report available"
                }

        except Exception as e:
            print(f"❌ Health monitoring test failed: {e}")
            validation_results["tests"]["health_monitoring"] = {"status": "FAILED", "details": str(e)}

        # Test sample test execution
        print("\n📋 Test 9: Sample Test Execution Setup")
        print("-" * 45)

        try:
            # Setup test orchestrator
            test_orchestrator = testing_protocols.test_orchestrator

            # Get a small subset of test configurations for quick validation
            test_configs = matrix_validator.get_configurations_by_criteria(
                complexity_level=matrix_validator.configurations[0].complexity_level
            )[:4]  # Just take first 4 for quick test

            if test_configs:
                test_orchestrator.setup_test_execution(test_configs)
                print(f"✅ Test execution setup with {len(test_configs)} configurations")
                validation_results["tests"]["test_execution_setup"] = {
                    "status": "PASSED",
                    "details": f"Test execution setup with {len(test_configs)} configurations"
                }
            else:
                print("❌ No test configurations available for setup")
                validation_results["tests"]["test_execution_setup"] = {
                    "status": "FAILED",
                    "details": "No test configurations available"
                }

        except Exception as e:
            print(f"❌ Test execution setup failed: {e}")
            validation_results["tests"]["test_execution_setup"] = {"status": "FAILED", "details": str(e)}

        # Cleanup
        print("\n📋 Test 10: System Cleanup")
        print("-" * 30)

        try:
            await testing_protocols.shutdown()
            print("✅ System shutdown completed")
            validation_results["tests"]["system_cleanup"] = {
                "status": "PASSED",
                "details": "System shutdown successful"
            }

        except Exception as e:
            print(f"❌ System cleanup failed: {e}")
            validation_results["tests"]["system_cleanup"] = {"status": "FAILED", "details": str(e)}

    except Exception as e:
        print(f"\n❌ Critical validation error: {e}")
        validation_results["tests"]["critical_error"] = {"status": "FAILED", "details": str(e)}

    # Summary
    print("\n📊 Validation Summary")
    print("=" * 70)

    total_tests = len(validation_results["tests"])
    passed_tests = sum(1 for test in validation_results["tests"].values() if test["status"] == "PASSED")
    failed_tests = sum(1 for test in validation_results["tests"].values() if test["status"] == "FAILED")
    warning_tests = sum(1 for test in validation_results["tests"].values() if test["status"] == "WARNING")

    print(f"Total Tests: {total_tests}")
    print(f"✅ Passed: {passed_tests}")
    print(f"❌ Failed: {failed_tests}")
    print(f"⚠️ Warnings: {warning_tests}")
    print(f"Success Rate: {(passed_tests / total_tests * 100):.1f}%")

    validation_results["summary"] = {
        "total_tests": total_tests,
        "passed": passed_tests,
        "failed": failed_tests,
        "warnings": warning_tests,
        "success_rate": passed_tests / total_tests if total_tests > 0 else 0.0
    }

    # Determine overall status
    if failed_tests == 0:
        if warning_tests == 0:
            overall_status = "✅ PHASE 4.13 VALIDATION SUCCESSFUL"
        else:
            overall_status = "⚠️ PHASE 4.13 VALIDATION COMPLETED WITH WARNINGS"
    else:
        overall_status = "❌ PHASE 4.13 VALIDATION FAILED"

    print(f"\n{overall_status}")
    validation_results["overall_status"] = overall_status

    # Save results
    results_file = Path(__file__).parent.parent.parent / "docs" / "reports" / "validation" / f"phase_4_13_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)

    with open(results_file, 'w') as f:
        json.dump(validation_results, f, indent=2, default=str)

    print(f"📄 Validation results saved to: {results_file}")

    return validation_results


if __name__ == "__main__":
    try:
        results = asyncio.run(validate_phase_4_13_integration())

        # Exit with appropriate code
        if results["summary"]["failed"] == 0:
            sys.exit(0)  # Success
        else:
            sys.exit(1)  # Failure

    except KeyboardInterrupt:
        print("\n⏹️ Validation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Validation script error: {e}")
        sys.exit(1)
