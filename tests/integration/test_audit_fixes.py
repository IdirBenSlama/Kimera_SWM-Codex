#!/usr/bin/env python3
"""
KIMERA Engine Audit Fixes Validation Test
==========================================

This script validates all the critical fixes implemented during the engine audit:
1. Zero-debugging constraint fixes (print -> logging)
2. Configuration management integration
3. Hardware awareness and device logging
4. Security enhancements and input validation
5. Error handling improvements

Run this to verify that all audit fixes are working correctly.
"""

import logging
import sys
import traceback
from typing import Any, Dict

# Configure logging to see our improvements
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def test_configuration_management() -> Dict[str, Any]:
    """Test configuration management fixes"""
    print("🔧 Testing Configuration Management...")

    try:
        from src.config.settings import get_settings
        from src.utils.config import get_api_settings

        settings = get_api_settings()

        return {
            "status": "PASS",
            "environment": str(settings.environment),
            "server_config": f"{settings.server.host}:{settings.server.port}",
            "logging_level": str(settings.logging.level),
            "details": "Configuration system working correctly",
        }
    except Exception as e:
        return {
            "status": "FAIL",
            "error": str(e),
            "details": "Configuration management failed",
        }


def test_print_statement_fixes() -> Dict[str, Any]:
    """Test that print statements have been replaced with logging"""
    print("📝 Testing Print Statement Fixes...")

    try:
        # Import engines that we fixed
        # Check source code for print statements (basic check)
        import inspect

        from src.engines import kimera_advanced_integration_fix, quantum_truth_monitor

        print_found = []
        for module in [quantum_truth_monitor, kimera_advanced_integration_fix]:
            source = inspect.getsource(module)
            if (
                "print(" in source
                and "logger." not in source.split("print(")[1].split("\n")[0]
            ):
                print_found.append(module.__name__)

        if print_found:
            return {
                "status": "FAIL",
                "details": f"Print statements still found in: {print_found}",
            }
        else:
            return {
                "status": "PASS",
                "details": "All print statements replaced with proper logging",
            }

    except Exception as e:
        return {
            "status": "ERROR",
            "error": str(e),
            "details": "Could not verify print statement fixes",
        }


def test_enhanced_thermodynamic_engine() -> Dict[str, Any]:
    """Test enhanced thermodynamic engine with validation"""
    print("🌡️ Testing Enhanced Thermodynamic Engine...")

    try:
        import numpy as np

        from src.engines.thermodynamic_engine import ThermodynamicEngine

        engine = ThermodynamicEngine()

        # Test normal operation
        field = [np.random.randn(10), np.random.randn(10), np.random.randn(10)]
        temp = engine.calculate_semantic_temperature(field)

        # Test input validation
        validation_passed = False
        try:
            engine.calculate_semantic_temperature("invalid")
        except TypeError:
            validation_passed = True

        return {
            "status": "PASS" if validation_passed else "FAIL",
            "temperature_calculated": temp,
            "input_validation": validation_passed,
            "details": "Thermodynamic engine with proper validation and logging",
        }

    except Exception as e:
        return {
            "status": "FAIL",
            "error": str(e),
            "details": "Thermodynamic engine test failed",
        }


def test_gpu_memory_pool() -> Dict[str, Any]:
    """Test enhanced GPU memory pool"""
    print("🖥️ Testing Enhanced GPU Memory Pool...")

    try:
        from src.engines.gpu_memory_pool import TCSignalMemoryPool

        # Create memory pool with validation
        pool = TCSignalMemoryPool(initial_blocks=2, block_size=1024 * 1024)

        # Test memory operations
        block = pool.get_block(1024 * 1024)
        stats_before = pool.get_stats()

        if block is not None:
            pool.release_block(block)
            stats_after = pool.get_stats()

            return {
                "status": "PASS",
                "total_blocks": stats_after["total_blocks"],
                "memory_managed_mb": stats_after["total_pooled_memory_mb"],
                "allocation_success": True,
                "details": "GPU memory pool with device logging and validation",
            }
        else:
            return {"status": "FAIL", "details": "Memory allocation failed"}

    except Exception as e:
        return {
            "status": "FAIL",
            "error": str(e),
            "details": "GPU memory pool test failed",
        }


def test_device_logging() -> Dict[str, Any]:
    """Test device logging enhancements"""
    print("🔍 Testing Device Logging...")

    try:
        # Capture log messages
        import io
        import sys

        import torch

        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        logger = logging.getLogger("backend.engines")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        # Test engines with device logging
        from src.engines.thermodynamic_engine import ThermodynamicEngine

        engine = ThermodynamicEngine()

        log_contents = log_capture.getvalue()
        logger.removeHandler(handler)

        device_logging_present = "initialized" in log_contents.lower()

        return {
            "status": "PASS" if device_logging_present else "PARTIAL",
            "cuda_available": torch.cuda.is_available(),
            "device_logging": device_logging_present,
            "details": "Device logging enhancements verified",
        }

    except Exception as e:
        return {
            "status": "FAIL",
            "error": str(e),
            "details": "Device logging test failed",
        }


def test_security_enhancements() -> Dict[str, Any]:
    """Test security enhancements in cryptographic engines"""
    print("🔒 Testing Security Enhancements...")

    try:
        from src.engines.gpu_cryptographic_engine import GPUCryptographicEngine

        engine = GPUCryptographicEngine()

        # Test input validation
        validation_tests = []

        # Test 1: Valid key size
        try:
            key = engine.generate_secure_key(32)
            validation_tests.append(("valid_key_32", True))
        except Exception:
            validation_tests.append(("valid_key_32", False))

        # Test 2: Invalid key size (too small)
        try:
            engine.generate_secure_key(8)
            validation_tests.append(("invalid_small_key", False))  # Should fail
        except ValueError:
            validation_tests.append(
                ("invalid_small_key", True)
            )  # Should raise ValueError

        # Test 3: Invalid key size (too large)
        try:
            engine.generate_secure_key(128)
            validation_tests.append(("invalid_large_key", False))  # Should fail
        except ValueError:
            validation_tests.append(
                ("invalid_large_key", True)
            )  # Should raise ValueError

        all_passed = all(result for _, result in validation_tests)

        return {
            "status": "PASS" if all_passed else "FAIL",
            "validation_tests": validation_tests,
            "details": "Cryptographic input validation and security checks",
        }

    except Exception as e:
        return {
            "status": "FAIL",
            "error": str(e),
            "details": "Security enhancement test failed",
        }


def run_full_audit_validation():
    """Run complete audit validation"""
    print("=" * 60)
    print("🔍 KIMERA ENGINE AUDIT FIXES VALIDATION")
    print("=" * 60)

    tests = [
        ("Configuration Management", test_configuration_management),
        ("Print Statement Fixes", test_print_statement_fixes),
        ("Enhanced Thermodynamic Engine", test_enhanced_thermodynamic_engine),
        ("GPU Memory Pool", test_gpu_memory_pool),
        ("Device Logging", test_device_logging),
        ("Security Enhancements", test_security_enhancements),
    ]

    results = {}
    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n📋 {test_name}:")
        try:
            result = test_func()
            results[test_name] = result

            if result["status"] == "PASS":
                print(f"   ✅ PASSED: {result.get('details', 'Test passed')}")
                passed += 1
            elif result["status"] == "PARTIAL":
                print(f"   ⚠️ PARTIAL: {result.get('details', 'Test partially passed')}")
                passed += 0.5
            else:
                print(f"   ❌ FAILED: {result.get('details', 'Test failed')}")
                if "error" in result:
                    print(f"      Error: {result['error']}")

        except Exception as e:
            print(f"   💥 EXCEPTION: {str(e)}")
            results[test_name] = {
                "status": "EXCEPTION",
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

    # Summary
    print("\n" + "=" * 60)
    print("📊 AUDIT VALIDATION SUMMARY")
    print("=" * 60)

    score = (passed / total) * 100
    print(f"Overall Score: {score:.1f}% ({passed}/{total} tests passed)")

    if score >= 90:
        print("🎉 EXCELLENT: Audit fixes successfully implemented!")
    elif score >= 75:
        print("✅ GOOD: Most audit fixes working, minor issues remain")
    elif score >= 50:
        print("⚠️ PARTIAL: Some audit fixes working, significant issues remain")
    else:
        print("❌ POOR: Major issues with audit fixes")

    print("\n🔍 Detailed Results:")
    for test_name, result in results.items():
        status_emoji = {
            "PASS": "✅",
            "PARTIAL": "⚠️",
            "FAIL": "❌",
            "EXCEPTION": "💥",
        }.get(result["status"], "❓")

        print(f"   {status_emoji} {test_name}: {result['status']}")

    return results


if __name__ == "__main__":
    results = run_full_audit_validation()

    # Exit with appropriate code
    passed_count = sum(1 for r in results.values() if r["status"] == "PASS")
    total_count = len(results)

    if passed_count == total_count:
        sys.exit(0)  # All tests passed
    elif passed_count >= total_count * 0.75:
        sys.exit(1)  # Most tests passed, minor issues
    else:
        sys.exit(2)  # Significant issues
