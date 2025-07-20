#!/usr/bin/env python3
"""
KIMERA Corrected Performance Test Suite
Testing with actual method names and comprehensive system evaluation
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
import numpy as np

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class KimeraCorrectedPerformanceTest:
    """Corrected KIMERA performance testing suite with real method names"""
    
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
        """Test thermodynamic engine performance with correct methods"""
        logger.info("🔥 Testing Thermodynamic Engine...")
        
        start_time = time.time()
        try:
            from backend.engines.thermodynamic_engine import ThermodynamicEngine
            engine = ThermodynamicEngine()
            
            # Test semantic temperature calculation
            test_fields = [
                [np.random.randn(100), np.random.randn(100)],
                [np.random.randn(200), np.random.randn(200)],
                [np.random.randn(300), np.random.randn(300)]
            ]
            
            calculation_times = []
            results = []
            
            for field in test_fields:
                calc_start = time.time()
                temp = engine.calculate_semantic_temperature(field)
                calculation_times.append(time.time() - calc_start)
                results.append(temp)
            
            # Test semantic Carnot engine
            carnot_start = time.time()
            carnot_result = engine.run_semantic_carnot_engine(
                hot_cognitive_field=test_fields[0],
                cold_cognitive_field=test_fields[1]
            )
            carnot_time = time.time() - carnot_start
            
            execution_time = time.time() - start_time
            
            self.results['engine_tests']['thermodynamic'] = {
                'status': 'PASS',
                'execution_time': execution_time,
                'semantic_temperature_calculations': len(test_fields),
                'avg_calculation_time': sum(calculation_times) / len(calculation_times),
                'carnot_engine_time': carnot_time,
                'sample_temperatures': results[:2]
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
        """Test quantum cognitive engine performance with correct methods"""
        logger.info("🧠 Testing Quantum Cognitive Engine...")
        
        start_time = time.time()
        try:
            from backend.engines.quantum_cognitive_engine import QuantumCognitiveEngine
            engine = QuantumCognitiveEngine()
            
            # Test cognitive superposition creation
            test_cognitive_inputs = [
                [np.random.randn(50), np.random.randn(50)],
                [np.random.randn(100), np.random.randn(100)],
                [np.random.randn(150), np.random.randn(150)]
            ]
            
            superposition_times = []
            interference_times = []
            
            for inputs in test_cognitive_inputs:
                # Test superposition creation
                sup_start = time.time()
                superposition = engine.create_cognitive_superposition(
                    cognitive_inputs=inputs,
                    consciousness_weights=[0.5, 0.5]
                )
                superposition_times.append(time.time() - sup_start)
                
                # Test interference processing
                int_start = time.time()
                interference = engine.process_quantum_cognitive_interference(
                    superposition_states=[superposition],
                    interference_type="constructive"
                )
                interference_times.append(time.time() - int_start)
            
            # Get quantum processing metrics
            metrics = engine.get_quantum_processing_metrics()
            
            execution_time = time.time() - start_time
            
            self.results['engine_tests']['quantum_cognitive'] = {
                'status': 'PASS',
                'execution_time': execution_time,
                'superposition_tests': len(test_cognitive_inputs),
                'avg_superposition_time': sum(superposition_times) / len(superposition_times),
                'avg_interference_time': sum(interference_times) / len(interference_times),
                'device': str(engine.device),
                'quantum_metrics': {
                    'num_qubits': metrics.num_qubits,
                    'coherence_time': metrics.coherence_time,
                    'fidelity': metrics.fidelity
                }
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
        """Test GPU cryptographic engine performance with correct methods"""
        logger.info("🔐 Testing GPU Cryptographic Engine...")
        
        start_time = time.time()
        try:
            # Check CuPy availability first
            try:
                import cupy as cp
                if not cp.cuda.is_available():
                    raise ImportError("CUDA not available")
            except ImportError as e:
                self.results['engine_tests']['gpu_cryptographic'] = {
                    'status': 'SKIP',
                    'reason': f"CuPy/CUDA not available: {e}",
                    'execution_time': time.time() - start_time
                }
                logger.warning(f"⚠️ GPU Cryptographic Engine skipped: CuPy/CUDA not available")
                return
            
            from backend.engines.gpu_cryptographic_engine import GPUCryptographicEngine
            engine = GPUCryptographicEngine()
            
            # Test key generation
            key_gen_start = time.time()
            secure_key = engine.generate_secure_key(32)
            key_gen_time = time.time() - key_gen_start
            
            # Test cognitive data encryption/decryption
            test_data = [
                cp.random.randn(1000, dtype=cp.float32),
                cp.random.randn(2000, dtype=cp.float32),
                cp.random.randn(3000, dtype=cp.float32)
            ]
            
            encryption_times = []
            decryption_times = []
            
            for data in test_data:
                # Test encryption
                enc_start = time.time()
                encrypted, nonce = engine.encrypt_cognitive_data(data, secure_key)
                encryption_times.append(time.time() - enc_start)
                
                # Test decryption
                dec_start = time.time()
                decrypted = engine.decrypt_cognitive_data(encrypted, secure_key, nonce)
                decryption_times.append(time.time() - dec_start)
                
                # Verify integrity (basic check)
                assert decrypted.shape == data.shape, "Decryption shape mismatch!"
            
            # Test hashing
            hash_start = time.time()
            cognitive_hash = engine.hash_cognitive_state(test_data[0])
            hash_time = time.time() - hash_start
            
            # Get benchmark
            benchmark = engine.benchmark_crypto_operations()
            
            execution_time = time.time() - start_time
            
            self.results['engine_tests']['gpu_cryptographic'] = {
                'status': 'PASS',
                'execution_time': execution_time,
                'key_generation_time': key_gen_time,
                'data_processed': len(test_data),
                'avg_encryption_time': sum(encryption_times) / len(encryption_times),
                'avg_decryption_time': sum(decryption_times) / len(decryption_times),
                'hash_time': hash_time,
                'benchmark_results': benchmark
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
        
        # Wait a moment for the backend to fully start
        time.sleep(2)
        
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
    
    def test_pytorch_gpu_performance(self):
        """Test PyTorch GPU performance comprehensively"""
        logger.info("🔥 Testing PyTorch GPU Performance...")
        
        start_time = time.time()
        try:
            tensor_sizes = [1000, 2000, 3000, 4000]
            gpu_times = []
            cpu_times = []
            memory_usage = []
            
            for size in tensor_sizes:
                # CPU test
                cpu_start = time.time()
                a_cpu = torch.randn(size, size)
                b_cpu = torch.randn(size, size)
                c_cpu = torch.matmul(a_cpu, b_cpu)
                cpu_times.append(time.time() - cpu_start)
                
                # GPU test (if available)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    initial_memory = torch.cuda.memory_allocated()
                    
                    gpu_start = time.time()
                    a_gpu = torch.randn(size, size, device='cuda')
                    b_gpu = torch.randn(size, size, device='cuda')
                    c_gpu = torch.matmul(a_gpu, b_gpu)
                    torch.cuda.synchronize()
                    gpu_times.append(time.time() - gpu_start)
                    
                    peak_memory = torch.cuda.memory_allocated()
                    memory_usage.append((peak_memory - initial_memory) / 1024**2)  # MB
                    
                    # Cleanup
                    del a_gpu, b_gpu, c_gpu
                    torch.cuda.empty_cache()
            
            execution_time = time.time() - start_time
            
            self.results['performance_metrics']['pytorch_gpu'] = {
                'status': 'PASS',
                'execution_time': execution_time,
                'tensor_sizes_tested': tensor_sizes,
                'avg_cpu_time': sum(cpu_times) / len(cpu_times),
                'avg_gpu_time': sum(gpu_times) / len(gpu_times) if gpu_times else None,
                'gpu_speedup': (sum(cpu_times) / sum(gpu_times)) if gpu_times else None,
                'avg_gpu_memory_mb': sum(memory_usage) / len(memory_usage) if memory_usage else None,
                'peak_gpu_memory_mb': max(memory_usage) if memory_usage else None
            }
            logger.info(f"✅ PyTorch GPU Performance: {execution_time:.3f}s")
            
        except Exception as e:
            self.results['performance_metrics']['pytorch_gpu'] = {
                'status': 'FAIL',
                'error': str(e),
                'execution_time': time.time() - start_time
            }
            logger.error(f"❌ PyTorch GPU Performance failed: {e}")
    
    def test_system_stability(self):
        """Test system stability under load"""
        logger.info("⚖️ Testing System Stability...")
        
        start_time = time.time()
        try:
            # Monitor system resources during intensive operations
            initial_cpu = psutil.cpu_percent()
            initial_memory = psutil.virtual_memory().percent
            
            # Run multiple intensive tasks
            results = []
            resource_snapshots = []
            
            for i in range(10):
                task_start = time.time()
                
                # CPU intensive task
                _ = sum(j**2 for j in range(50000))
                
                # Memory intensive task
                large_array = np.random.randn(10000, 100)
                
                # GPU task if available
                if torch.cuda.is_available():
                    gpu_tensor = torch.randn(1000, 1000, device='cuda')
                    _ = torch.matmul(gpu_tensor, gpu_tensor.T)
                    del gpu_tensor
                    torch.cuda.empty_cache()
                
                del large_array
                
                task_time = time.time() - task_start
                results.append(task_time)
                
                # Snapshot resources
                resource_snapshots.append({
                    'cpu_percent': psutil.cpu_percent(),
                    'memory_percent': psutil.virtual_memory().percent,
                    'task_time': task_time
                })
            
            final_cpu = psutil.cpu_percent()
            final_memory = psutil.virtual_memory().percent
            
            execution_time = time.time() - start_time
            
            self.results['stress_tests']['system_stability'] = {
                'status': 'PASS',
                'execution_time': execution_time,
                'tasks_completed': len(results),
                'avg_task_time': sum(results) / len(results),
                'task_time_variance': np.var(results),
                'initial_cpu_percent': initial_cpu,
                'final_cpu_percent': final_cpu,
                'initial_memory_percent': initial_memory,
                'final_memory_percent': final_memory,
                'max_cpu_percent': max(snap['cpu_percent'] for snap in resource_snapshots),
                'max_memory_percent': max(snap['memory_percent'] for snap in resource_snapshots)
            }
            logger.info(f"✅ System Stability: {execution_time:.3f}s")
            
        except Exception as e:
            self.results['stress_tests']['system_stability'] = {
                'status': 'FAIL',
                'error': str(e),
                'execution_time': time.time() - start_time
            }
            logger.error(f"❌ System Stability failed: {e}")
    
    def generate_summary(self):
        """Generate comprehensive performance test summary"""
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
            'overall_status': 'EXCELLENT' if success_rate >= 90 else 'GOOD' if success_rate >= 75 else 'ACCEPTABLE' if success_rate >= 60 else 'NEEDS_IMPROVEMENT'
        }
        
        logger.info(f"📊 Performance Summary: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
    
    def save_results(self):
        """Save test results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"corrected_performance_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"💾 Results saved to: {filename}")
        return filename
    
    async def run_full_test_suite(self):
        """Run the complete corrected performance test suite"""
        logger.info("🚀 Starting KIMERA Corrected Performance Test Suite")
        logger.info("=" * 70)
        
        # Engine tests with correct method names
        self.test_thermodynamic_engine()
        self.test_quantum_cognitive_engine()
        self.test_gpu_cryptographic_engine()
        
        # API tests
        self.test_api_endpoints()
        
        # Performance tests
        self.test_pytorch_gpu_performance()
        
        # Stress tests
        self.test_system_stability()
        
        # Generate summary
        self.generate_summary()
        
        # Save results
        results_file = self.save_results()
        
        logger.info("=" * 70)
        logger.info("🎯 KIMERA Corrected Performance Test Suite Complete")
        
        return self.results, results_file

def main():
    """Main function to run corrected performance tests"""
    print("🚀 KIMERA Corrected Performance Test Suite")
    print("=" * 70)
    
    # Create test instance
    test_suite = KimeraCorrectedPerformanceTest()
    
    # Run tests
    try:
        results, results_file = asyncio.run(test_suite.run_full_test_suite())
        
        # Print detailed summary
        print("\n📊 FINAL RESULTS:")
        print("=" * 50)
        print(f"Total Tests: {results['summary']['total_tests']}")
        print(f"Passed: {results['summary']['passed_tests']}")
        print(f"Failed: {results['summary']['failed_tests']}")
        print(f"Skipped: {results['summary']['skipped_tests']}")
        print(f"Success Rate: {results['summary']['success_rate']:.1f}%")
        print(f"Overall Status: {results['summary']['overall_status']}")
        print(f"Results File: {results_file}")
        
        # Print system info
        print("\n🖥️ SYSTEM PERFORMANCE:")
        print("=" * 50)
        print(f"CPU Cores: {results['system_info']['cpu_count']}")
        print(f"RAM: {results['system_info']['memory_total_gb']:.1f}GB")
        print(f"GPU: {results['system_info']['gpu_name'] or 'None'}")
        if results['system_info']['gpu_memory_gb']:
            print(f"GPU Memory: {results['system_info']['gpu_memory_gb']:.1f}GB")
        
        # Print engine performance
        if 'engine_tests' in results:
            print("\n🔧 ENGINE PERFORMANCE:")
            print("=" * 50)
            for engine, result in results['engine_tests'].items():
                status_emoji = "✅" if result['status'] == 'PASS' else "⚠️" if result['status'] == 'SKIP' else "❌"
                print(f"{status_emoji} {engine.title()}: {result['status']} ({result['execution_time']:.3f}s)")
        
        print("=" * 50)
        
        return results['summary']['overall_status'] in ['EXCELLENT', 'GOOD', 'ACCEPTABLE']
        
    except Exception as e:
        logger.error(f"❌ Performance test suite failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 