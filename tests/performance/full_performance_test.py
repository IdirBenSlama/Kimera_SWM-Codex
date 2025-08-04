#!/usr/bin/env python3
"""
KIMERA Full Performance Test Suite
Comprehensive testing of all engines, performance metrics, and system stability
"""

import asyncio
import concurrent.futures
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil
import requests
import torch

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))

from src.engines.cognitive_field_dynamics import (
    ThermodynamicEngine as CognitiveFieldEngine,
)
from src.engines.gpu_memory_pool import TCSignalMemoryPool
from src.engines.gyroscopic_universal_translator import GyroscopicUniversalTranslator
from src.engines.quantum_cognitive_engine import QuantumCognitiveEngine
from src.engines.thermodynamic_engine import ThermodynamicEngine

# Import KIMERA components
from src.utils.config import get_api_settings

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class KimeraPerformanceTest:
    """Comprehensive KIMERA performance testing suite"""

    def __init__(self):
        self.results = {
            "test_start": datetime.now().isoformat(),
            "system_info": self._get_system_info(),
            "engine_tests": {},
            "performance_metrics": {},
            "stress_tests": {},
            "api_tests": {},
            "summary": {},
        }
        self.base_url = "http://127.0.0.1:8000"

    def _get_system_info(self):
        """Collect system information"""
        return {
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "gpu_available": torch.cuda.is_available(),
            "gpu_name": (
                torch.cuda.get_device_name() if torch.cuda.is_available() else None
            ),
            "gpu_memory_gb": (
                torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if torch.cuda.is_available()
                else None
            ),
            "python_version": sys.version,
            "torch_version": torch.__version__,
        }

    def test_thermodynamic_engine(self):
        """Test thermodynamic engine performance"""
        logger.info("🔥 Testing Thermodynamic Engine...")

        start_time = time.time()
        try:
            engine = ThermodynamicEngine()

            # Test multiple calculations
            test_cases = [
                (300, 8.314, 1.0),  # Standard conditions
                (373, 8.314, 2.0),  # Boiling water
                (77, 8.314, 0.5),  # Liquid nitrogen
                (1000, 8.314, 5.0),  # High temperature
            ]

            results = []
            for temp, gas_const, moles in test_cases:
                result = engine.calculate_pressure(temp, gas_const, moles)
                results.append(result)

            # Test error handling
            try:
                engine.calculate_pressure("invalid", 8.314, 1.0)
                error_handling = False
            except (TypeError, ValueError):
                error_handling = True

            execution_time = time.time() - start_time

            self.results["engine_tests"]["thermodynamic"] = {
                "status": "PASS",
                "execution_time": execution_time,
                "calculations_performed": len(test_cases),
                "error_handling": error_handling,
                "sample_results": results[:2],  # First 2 results
            }
            logger.info(f"✅ Thermodynamic Engine: {execution_time:.3f}s")

        except Exception as e:
            self.results["engine_tests"]["thermodynamic"] = {
                "status": "FAIL",
                "error": str(e),
                "execution_time": time.time() - start_time,
            }
            logger.error(f"❌ Thermodynamic Engine failed: {e}")

    def test_gpu_memory_pool(self):
        """Test GPU memory pool performance"""
        logger.info("🖥️ Testing GPU Memory Pool...")

        start_time = time.time()
        try:
            # Check if CuPy is available
            try:
                import cupy as cp

                if not cp.cuda.is_available():
                    raise ImportError("CUDA not available")
            except ImportError as e:
                self.results["engine_tests"]["gpu_memory_pool"] = {
                    "status": "SKIP",
                    "reason": f"CuPy/CUDA not available: {e}",
                    "execution_time": time.time() - start_time,
                }
                logger.warning(f"⚠️ GPU Memory Pool skipped: CuPy/CUDA not available")
                return

            pool = TCSignalMemoryPool()

            # Test memory allocation/deallocation cycles
            allocation_times = []
            deallocation_times = []

            for i in range(5):
                # Allocate memory
                alloc_start = time.time()
                tensor = pool.get_block(1024 * 1024 * 10)  # 10MB
                allocation_times.append(time.time() - alloc_start)

                # Deallocate memory
                dealloc_start = time.time()
                pool.release_block(tensor)
                deallocation_times.append(time.time() - dealloc_start)

            # Get memory stats
            stats = pool.get_stats()
            execution_time = time.time() - start_time

            self.results["engine_tests"]["gpu_memory_pool"] = {
                "status": "PASS",
                "execution_time": execution_time,
                "avg_allocation_time": sum(allocation_times) / len(allocation_times),
                "avg_deallocation_time": sum(deallocation_times)
                / len(deallocation_times),
                "memory_stats": stats,
            }
            logger.info(f"✅ GPU Memory Pool: {execution_time:.3f}s")

        except Exception as e:
            self.results["engine_tests"]["gpu_memory_pool"] = {
                "status": "FAIL",
                "error": str(e),
                "execution_time": time.time() - start_time,
            }
            logger.error(f"❌ GPU Memory Pool failed: {e}")

    def test_quantum_cognitive_engine(self):
        """Test quantum cognitive engine performance"""
        logger.info("🧠 Testing Quantum Cognitive Engine...")

        start_time = time.time()
        try:
            engine = QuantumCognitiveEngine()

            # Test quantum state processing
            test_inputs = [
                "consciousness and quantum coherence",
                "neural network optimization",
                "cognitive pattern recognition",
                "quantum entanglement theory",
            ]

            processing_times = []
            for input_text in test_inputs:
                proc_start = time.time()
                result = engine.process_quantum_state(input_text)
                processing_times.append(time.time() - proc_start)

            execution_time = time.time() - start_time

            self.results["engine_tests"]["quantum_cognitive"] = {
                "status": "PASS",
                "execution_time": execution_time,
                "inputs_processed": len(test_inputs),
                "avg_processing_time": sum(processing_times) / len(processing_times),
                "device": str(engine.device),
            }
            logger.info(f"✅ Quantum Cognitive Engine: {execution_time:.3f}s")

        except Exception as e:
            self.results["engine_tests"]["quantum_cognitive"] = {
                "status": "FAIL",
                "error": str(e),
                "execution_time": time.time() - start_time,
            }
            logger.error(f"❌ Quantum Cognitive Engine failed: {e}")

    def test_cognitive_field_dynamics(self):
        """Test cognitive field dynamics performance"""
        logger.info("🌊 Testing Cognitive Field Dynamics...")

        start_time = time.time()
        try:
            engine = CognitiveFieldEngine()

            # Test field computations
            test_values = [100.0, 500.0, 1000.0]

            computation_times = []
            for temp_value in test_values:
                comp_start = time.time()
                result = engine.calculate_pressure(temp_value, 8.314, 1.0)
                computation_times.append(time.time() - comp_start)

            execution_time = time.time() - start_time

            self.results["engine_tests"]["cognitive_field_dynamics"] = {
                "status": "PASS",
                "execution_time": execution_time,
                "values_processed": len(test_values),
                "avg_computation_time": sum(computation_times) / len(computation_times),
                "device": str(getattr(engine, "device", "CPU")),
            }
            logger.info(f"✅ Cognitive Field Dynamics: {execution_time:.3f}s")

        except Exception as e:
            self.results["engine_tests"]["cognitive_field_dynamics"] = {
                "status": "FAIL",
                "error": str(e),
                "execution_time": time.time() - start_time,
            }
            logger.error(f"❌ Cognitive Field Dynamics failed: {e}")

    def test_api_endpoints(self):
        """Test API endpoint performance"""
        logger.info("🌐 Testing API Endpoints...")

        endpoints = ["/health", "/api/v1/status", "/api/v1/engines/status"]

        for endpoint in endpoints:
            start_time = time.time()
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                execution_time = time.time() - start_time

                self.results["api_tests"][endpoint] = {
                    "status": "PASS" if response.status_code == 200 else "FAIL",
                    "status_code": response.status_code,
                    "response_time": execution_time,
                    "response_size": len(response.content),
                }
                logger.info(
                    f"✅ API {endpoint}: {response.status_code} ({execution_time:.3f}s)"
                )

            except Exception as e:
                execution_time = time.time() - start_time
                self.results["api_tests"][endpoint] = {
                    "status": "FAIL",
                    "error": str(e),
                    "response_time": execution_time,
                }
                logger.error(f"❌ API {endpoint} failed: {e}")

    def test_memory_performance(self):
        """Test memory usage and performance"""
        logger.info("💾 Testing Memory Performance...")

        start_time = time.time()
        initial_memory = psutil.virtual_memory().percent
        initial_gpu_memory = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_gpu_memory = torch.cuda.memory_allocated() / (1024**2)  # MB

        # Memory stress test
        large_tensors = []
        try:
            for i in range(10):
                if torch.cuda.is_available():
                    tensor = torch.randn(1000, 1000, device="cuda")
                else:
                    tensor = torch.randn(1000, 1000)
                large_tensors.append(tensor)

            peak_memory = psutil.virtual_memory().percent
            peak_gpu_memory = None

            if torch.cuda.is_available():
                peak_gpu_memory = torch.cuda.memory_allocated() / (1024**2)  # MB

            # Cleanup
            del large_tensors
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            final_memory = psutil.virtual_memory().percent
            final_gpu_memory = None

            if torch.cuda.is_available():
                final_gpu_memory = torch.cuda.memory_allocated() / (1024**2)  # MB

            execution_time = time.time() - start_time

            self.results["performance_metrics"]["memory"] = {
                "status": "PASS",
                "execution_time": execution_time,
                "initial_memory_percent": initial_memory,
                "peak_memory_percent": peak_memory,
                "final_memory_percent": final_memory,
                "memory_increase": peak_memory - initial_memory,
                "initial_gpu_memory_mb": initial_gpu_memory,
                "peak_gpu_memory_mb": peak_gpu_memory,
                "final_gpu_memory_mb": final_gpu_memory,
            }
            logger.info(f"✅ Memory Performance: {execution_time:.3f}s")

        except Exception as e:
            self.results["performance_metrics"]["memory"] = {
                "status": "FAIL",
                "error": str(e),
                "execution_time": time.time() - start_time,
            }
            logger.error(f"❌ Memory Performance failed: {e}")

    def run_concurrent_stress_test(self):
        """Run concurrent operations stress test"""
        logger.info("⚡ Running Concurrent Stress Test...")

        start_time = time.time()

        def cpu_intensive_task():
            """CPU intensive computation"""
            result = sum(i**2 for i in range(10000))
            return result

        def memory_intensive_task():
            """Memory intensive operation"""
            data = [i for i in range(100000)]
            return len(data)

        def gpu_intensive_task():
            """GPU intensive computation if available"""
            if torch.cuda.is_available():
                tensor = torch.randn(1000, 1000, device="cuda")
                result = torch.matmul(tensor, tensor.T)
                return result.sum().item()
            else:
                tensor = torch.randn(500, 500)
                result = torch.matmul(tensor, tensor.T)
                return result.sum().item()

        try:
            # Run tasks concurrently
            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
                futures = []

                # Submit multiple tasks
                for _ in range(3):
                    futures.append(executor.submit(cpu_intensive_task))
                    futures.append(executor.submit(memory_intensive_task))
                    futures.append(executor.submit(gpu_intensive_task))

                # Wait for completion
                results = [
                    future.result()
                    for future in concurrent.futures.as_completed(futures)
                ]

            execution_time = time.time() - start_time

            self.results["stress_tests"]["concurrent"] = {
                "status": "PASS",
                "execution_time": execution_time,
                "tasks_completed": len(results),
                "avg_task_time": execution_time / len(results),
            }
            logger.info(f"✅ Concurrent Stress Test: {execution_time:.3f}s")

        except Exception as e:
            self.results["stress_tests"]["concurrent"] = {
                "status": "FAIL",
                "error": str(e),
                "execution_time": time.time() - start_time,
            }
            logger.error(f"❌ Concurrent Stress Test failed: {e}")

    def generate_summary(self):
        """Generate performance test summary"""
        logger.info("📊 Generating Performance Summary...")

        # Count test results
        total_tests = 0
        passed_tests = 0
        failed_tests = 0

        for category in [
            "engine_tests",
            "api_tests",
            "performance_metrics",
            "stress_tests",
        ]:
            if category in self.results:
                for test_name, test_result in self.results[category].items():
                    total_tests += 1
                    if test_result.get("status") == "PASS":
                        passed_tests += 1
                    else:
                        failed_tests += 1

        # Calculate success rate
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

        # Generate summary
        self.results["summary"] = {
            "test_end": datetime.now().isoformat(),
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": success_rate,
            "overall_status": "PASS" if success_rate >= 80 else "FAIL",
        }

        logger.info(
            f"📊 Performance Summary: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)"
        )

    def save_results(self):
        """Save test results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"performance_test_results_{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        logger.info(f"💾 Results saved to: {filename}")
        return filename

    async def run_full_test_suite(self):
        """Run the complete performance test suite"""
        logger.info("🚀 Starting KIMERA Full Performance Test Suite")
        logger.info("=" * 60)

        # Engine tests
        self.test_thermodynamic_engine()
        self.test_gpu_memory_pool()
        self.test_quantum_cognitive_engine()
        self.test_cognitive_field_dynamics()

        # API tests
        self.test_api_endpoints()

        # Performance tests
        self.test_memory_performance()

        # Stress tests
        self.run_concurrent_stress_test()

        # Generate summary
        self.generate_summary()

        # Save results
        results_file = self.save_results()

        logger.info("=" * 60)
        logger.info("🎯 KIMERA Performance Test Suite Complete")

        return self.results, results_file


def main():
    """Main function to run performance tests"""
    print("🚀 KIMERA Full Performance Test Suite")
    print("=" * 60)

    # Create test instance
    test_suite = KimeraPerformanceTest()

    # Run tests
    try:
        results, results_file = asyncio.run(test_suite.run_full_test_suite())

        # Print summary
        print("\n📊 FINAL RESULTS:")
        print("=" * 40)
        print(f"Total Tests: {results['summary']['total_tests']}")
        print(f"Passed: {results['summary']['passed_tests']}")
        print(f"Failed: {results['summary']['failed_tests']}")
        print(f"Success Rate: {results['summary']['success_rate']:.1f}%")
        print(f"Overall Status: {results['summary']['overall_status']}")
        print(f"Results File: {results_file}")
        print("=" * 40)

        return results["summary"]["overall_status"] == "PASS"

    except Exception as e:
        logger.error(f"❌ Performance test suite failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
