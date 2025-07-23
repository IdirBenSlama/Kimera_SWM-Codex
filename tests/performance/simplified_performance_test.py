#!/usr/bin/env python3
"""
KIMERA Simplified Performance Test Suite
Comprehensive testing focusing on available engines and system performance
"""

import asyncio
import time
import psutil
import json
import logging
import torch
import requests
import concurrent.futures
from datetime import datetime
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class KimeraSimplifiedPerformanceTest:
    """Simplified KIMERA performance testing suite"""
    
    def __init__(self):
        self.results = {
            'test_start': datetime.now().isoformat(),
            'system_info': self._get_system_info(),
            'engine_tests': {},
            'performance_metrics': {},
            'stress_tests': {},
            'api_tests': {},
            'summary': {}
        }
        self.base_url = "http://127.0.0.1:8000"
        
    def _get_system_info(self):
        """Collect system information"""
        return {
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'gpu_available': torch.cuda.is_available(),
            'gpu_name': torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else None,
            'python_version': sys.version,
            'torch_version': torch.__version__
        }
    
    def test_thermodynamic_engine(self):
        """Test thermodynamic engine performance"""
        logger.info("🔥 Testing Thermodynamic Engine...")
        
        start_time = time.time()
        try:
            # Import locally to handle potential issues
            from src.engines.thermodynamic_engine import ThermodynamicEngine
            engine = ThermodynamicEngine()
            
            # Test multiple calculations
            test_cases = [
                (300, 8.314, 1.0),  # Standard conditions
                (373, 8.314, 2.0),  # Boiling water
                (77, 8.314, 0.5),   # Liquid nitrogen
                (1000, 8.314, 5.0), # High temperature
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
            
            self.results['engine_tests']['thermodynamic'] = {
                'status': 'PASS',
                'execution_time': execution_time,
                'calculations_performed': len(test_cases),
                'error_handling': error_handling,
                'sample_results': results[:2]  # First 2 results
            }
            logger.info(f"✅ Thermodynamic Engine: {execution_time:.3f}s")
            
        except Exception as e:
            self.results['engine_tests']['thermodynamic'] = {
                'status': 'FAIL',
                'error': str(e),
                'execution_time': time.time() - start_time
            }
            logger.error(f"❌ Thermodynamic Engine failed: {e}")
    
    def test_quantum_cognitive_engine(self):
        """Test quantum cognitive engine performance"""
        logger.info("🧠 Testing Quantum Cognitive Engine...")
        
        start_time = time.time()
        try:
            from src.engines.quantum_cognitive_engine import QuantumCognitiveEngine
            engine = QuantumCognitiveEngine()
            
            # Test quantum state processing
            test_inputs = [
                "consciousness and quantum coherence",
                "neural network optimization",
                "cognitive pattern recognition",
                "quantum entanglement theory"
            ]
            
            processing_times = []
            for input_text in test_inputs:
                proc_start = time.time()
                result = engine.process_quantum_state(input_text)
                processing_times.append(time.time() - proc_start)
            
            execution_time = time.time() - start_time
            
            self.results['engine_tests']['quantum_cognitive'] = {
                'status': 'PASS',
                'execution_time': execution_time,
                'inputs_processed': len(test_inputs),
                'avg_processing_time': sum(processing_times) / len(processing_times),
                'device': str(engine.device)
            }
            logger.info(f"✅ Quantum Cognitive Engine: {execution_time:.3f}s")
            
        except Exception as e:
            self.results['engine_tests']['quantum_cognitive'] = {
                'status': 'FAIL',
                'error': str(e),
                'execution_time': time.time() - start_time
            }
            logger.error(f"❌ Quantum Cognitive Engine failed: {e}")
    
    def test_gpu_cryptographic_engine(self):
        """Test GPU cryptographic engine performance"""
        logger.info("🔐 Testing GPU Cryptographic Engine...")
        
        start_time = time.time()
        try:
            from src.engines.gpu_cryptographic_engine import GPUCryptographicEngine
            engine = GPUCryptographicEngine()
            
            # Test encryption/decryption cycles
            test_data = [
                b"Hello, KIMERA!",
                b"Quantum cryptography test",
                b"Performance evaluation data",
                b"GPU acceleration benchmark"
            ]
            
            encryption_times = []
            decryption_times = []
            
            for data in test_data:
                # Test encryption
                enc_start = time.time()
                encrypted = engine.encrypt(data)
                encryption_times.append(time.time() - enc_start)
                
                # Test decryption
                dec_start = time.time()
                decrypted = engine.decrypt(encrypted)
                decryption_times.append(time.time() - dec_start)
                
                # Verify integrity
                assert data == decrypted, "Decryption failed!"
            
            execution_time = time.time() - start_time
            
            self.results['engine_tests']['gpu_cryptographic'] = {
                'status': 'PASS',
                'execution_time': execution_time,
                'data_processed': len(test_data),
                'avg_encryption_time': sum(encryption_times) / len(encryption_times),
                'avg_decryption_time': sum(decryption_times) / len(decryption_times),
                'device': str(getattr(engine, 'device', 'GPU'))
            }
            logger.info(f"✅ GPU Cryptographic Engine: {execution_time:.3f}s")
            
        except Exception as e:
            self.results['engine_tests']['gpu_cryptographic'] = {
                'status': 'FAIL',
                'error': str(e),
                'execution_time': time.time() - start_time
            }
            logger.error(f"❌ GPU Cryptographic Engine failed: {e}")
    
    def test_api_endpoints(self):
        """Test API endpoint performance"""
        logger.info("🌐 Testing API Endpoints...")
        
        endpoints = [
            "/health",
            "/",
            "/docs"
        ]
        
        for endpoint in endpoints:
            start_time = time.time()
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                execution_time = time.time() - start_time
                
                self.results['api_tests'][endpoint] = {
                    'status': 'PASS' if response.status_code < 400 else 'FAIL',
                    'status_code': response.status_code,
                    'response_time': execution_time,
                    'response_size': len(response.content)
                }
                logger.info(f"✅ API {endpoint}: {response.status_code} ({execution_time:.3f}s)")
                
            except Exception as e:
                execution_time = time.time() - start_time
                self.results['api_tests'][endpoint] = {
                    'status': 'FAIL',
                    'error': str(e),
                    'response_time': execution_time
                }
                logger.error(f"❌ API {endpoint} failed: {e}")
    
    def test_pytorch_performance(self):
        """Test PyTorch GPU/CPU performance"""
        logger.info("🔥 Testing PyTorch Performance...")
        
        start_time = time.time()
        try:
            # Test tensor operations
            tensor_sizes = [1000, 2000, 3000]
            gpu_times = []
            cpu_times = []
            
            for size in tensor_sizes:
                # CPU test
                cpu_start = time.time()
                a = torch.randn(size, size)
                b = torch.randn(size, size)
                c = torch.matmul(a, b)
                cpu_times.append(time.time() - cpu_start)
                
                # GPU test (if available)
                if torch.cuda.is_available():
                    gpu_start = time.time()
                    a_gpu = torch.randn(size, size, device='cuda')
                    b_gpu = torch.randn(size, size, device='cuda')
                    c_gpu = torch.matmul(a_gpu, b_gpu)
                    torch.cuda.synchronize()  # Wait for GPU operations
                    gpu_times.append(time.time() - gpu_start)
            
            execution_time = time.time() - start_time
            
            self.results['performance_metrics']['pytorch'] = {
                'status': 'PASS',
                'execution_time': execution_time,
                'tensor_sizes_tested': tensor_sizes,
                'avg_cpu_time': sum(cpu_times) / len(cpu_times),
                'avg_gpu_time': sum(gpu_times) / len(gpu_times) if gpu_times else None,
                'gpu_speedup': (sum(cpu_times) / sum(gpu_times)) if gpu_times else None
            }
            logger.info(f"✅ PyTorch Performance: {execution_time:.3f}s")
            
        except Exception as e:
            self.results['performance_metrics']['pytorch'] = {
                'status': 'FAIL',
                'error': str(e),
                'execution_time': time.time() - start_time
            }
            logger.error(f"❌ PyTorch Performance failed: {e}")
    
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
                    tensor = torch.randn(1000, 1000, device='cuda')
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
            
            self.results['performance_metrics']['memory'] = {
                'status': 'PASS',
                'execution_time': execution_time,
                'initial_memory_percent': initial_memory,
                'peak_memory_percent': peak_memory,
                'final_memory_percent': final_memory,
                'memory_increase': peak_memory - initial_memory,
                'initial_gpu_memory_mb': initial_gpu_memory,
                'peak_gpu_memory_mb': peak_gpu_memory,
                'final_gpu_memory_mb': final_gpu_memory
            }
            logger.info(f"✅ Memory Performance: {execution_time:.3f}s")
            
        except Exception as e:
            self.results['performance_metrics']['memory'] = {
                'status': 'FAIL',
                'error': str(e),
                'execution_time': time.time() - start_time
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
                tensor = torch.randn(1000, 1000, device='cuda')
                result = torch.matmul(tensor, tensor.T)
                torch.cuda.synchronize()
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
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            execution_time = time.time() - start_time
            
            self.results['stress_tests']['concurrent'] = {
                'status': 'PASS',
                'execution_time': execution_time,
                'tasks_completed': len(results),
                'avg_task_time': execution_time / len(results)
            }
            logger.info(f"✅ Concurrent Stress Test: {execution_time:.3f}s")
            
        except Exception as e:
            self.results['stress_tests']['concurrent'] = {
                'status': 'FAIL',
                'error': str(e),
                'execution_time': time.time() - start_time
            }
            logger.error(f"❌ Concurrent Stress Test failed: {e}")
    
    def generate_summary(self):
        """Generate performance test summary"""
        logger.info("📊 Generating Performance Summary...")
        
        # Count test results
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        skipped_tests = 0
        
        for category in ['engine_tests', 'api_tests', 'performance_metrics', 'stress_tests']:
            if category in self.results:
                for test_name, test_result in self.results[category].items():
                    total_tests += 1
                    if test_result.get('status') == 'PASS':
                        passed_tests += 1
                    elif test_result.get('status') == 'SKIP':
                        skipped_tests += 1
                    else:
                        failed_tests += 1
        
        # Calculate success rate
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Generate summary
        self.results['summary'] = {
            'test_end': datetime.now().isoformat(),
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'skipped_tests': skipped_tests,
            'success_rate': success_rate,
            'overall_status': 'PASS' if success_rate >= 70 else 'FAIL'
        }
        
        logger.info(f"📊 Performance Summary: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
    
    def save_results(self):
        """Save test results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"simplified_performance_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"💾 Results saved to: {filename}")
        return filename
    
    async def run_full_test_suite(self):
        """Run the complete performance test suite"""
        logger.info("🚀 Starting KIMERA Simplified Performance Test Suite")
        logger.info("=" * 60)
        
        # Engine tests
        self.test_thermodynamic_engine()
        self.test_quantum_cognitive_engine()
        self.test_gpu_cryptographic_engine()
        
        # API tests
        self.test_api_endpoints()
        
        # Performance tests
        self.test_pytorch_performance()
        self.test_memory_performance()
        
        # Stress tests
        self.run_concurrent_stress_test()
        
        # Generate summary
        self.generate_summary()
        
        # Save results
        results_file = self.save_results()
        
        logger.info("=" * 60)
        logger.info("🎯 KIMERA Simplified Performance Test Suite Complete")
        
        return self.results, results_file

def main():
    """Main function to run performance tests"""
    print("🚀 KIMERA Simplified Performance Test Suite")
    print("=" * 60)
    
    # Create test instance
    test_suite = KimeraSimplifiedPerformanceTest()
    
    # Run tests
    try:
        results, results_file = asyncio.run(test_suite.run_full_test_suite())
        
        # Print summary
        print("\n📊 FINAL RESULTS:")
        print("=" * 40)
        print(f"Total Tests: {results['summary']['total_tests']}")
        print(f"Passed: {results['summary']['passed_tests']}")
        print(f"Failed: {results['summary']['failed_tests']}")
        print(f"Skipped: {results['summary']['skipped_tests']}")
        print(f"Success Rate: {results['summary']['success_rate']:.1f}%")
        print(f"Overall Status: {results['summary']['overall_status']}")
        print(f"Results File: {results_file}")
        
        # Print system info
        print("\n🖥️ SYSTEM INFO:")
        print("=" * 40)
        print(f"CPU Cores: {results['system_info']['cpu_count']}")
        print(f"RAM: {results['system_info']['memory_total_gb']:.1f}GB")
        print(f"GPU: {results['system_info']['gpu_name'] or 'None'}")
        if results['system_info']['gpu_memory_gb']:
            print(f"GPU Memory: {results['system_info']['gpu_memory_gb']:.1f}GB")
        print("=" * 40)
        
        return results['summary']['overall_status'] == 'PASS'
        
    except Exception as e:
        logger.error(f"❌ Performance test suite failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 