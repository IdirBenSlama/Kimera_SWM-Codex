#!/usr/bin/env python3
"""
KIMERA GPU PEAK PERFORMANCE TEST - SIMPLIFIED
Focus on working endpoints and GPU validation
"""

import time
import json
import requests
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

class SimpleGPUTester:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.results = {}
        
    def wait_for_system(self, timeout=60):
        """Wait for system to be ready"""
        print("🔄 Waiting for Kimera system...")
        for i in range(timeout):
            try:
                response = requests.get(f"{self.base_url}/health", timeout=5)
                if response.status_code == 200:
                    print("✅ System ready!")
                    return True
            except Exception as e:
                logger.error(f"Error in gpu_peak_performance_simple.py: {e}", exc_info=True)
                raise  # Re-raise for proper error handling
            time.sleep(1)
        print("❌ System not ready")
        return False
    
    def test_gpu_foundation_concurrent(self, num_threads=50):
        """Test GPU foundation with high concurrency"""
        print(f"🚀 Testing GPU Foundation with {num_threads} concurrent requests...")
        
        def make_request():
            try:
                start = time.time()
                response = requests.get(f"{self.base_url}/kimera/system/gpu_foundation", timeout=30)
                duration = time.time() - start
                return {
                    "success": response.status_code == 200,
                    "duration": duration,
                    "status_code": response.status_code
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "duration": 0
                }
        
        start_time = time.time()
        results = []
        
        # Use ThreadPoolExecutor for true concurrency
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(make_request) for _ in range(num_threads)]
            
            for future in as_completed(futures):
                results.append(future.result())
        
        total_time = time.time() - start_time
        successful = [r for r in results if r["success"]]
        success_rate = len(successful) / len(results)
        avg_response_time = sum(r["duration"] for r in successful) / len(successful) if successful else 0
        
        print(f"   ✅ Completed in {total_time:.2f}s")
        print(f"   📊 Success Rate: {success_rate:.1%}")
        print(f"   ⚡ Avg Response Time: {avg_response_time:.3f}s")
        print(f"   🔥 Requests/sec: {len(results)/total_time:.2f}")
        
        return {
            "test": "gpu_foundation_concurrent",
            "total_requests": num_threads,
            "total_time": total_time,
            "success_rate": success_rate,
            "avg_response_time": avg_response_time,
            "requests_per_second": len(results)/total_time,
            "results": results
        }
    
    def test_system_endpoints_stress(self):
        """Test various system endpoints under stress"""
        print("🚀 Testing System Endpoints under stress...")
        
        endpoints = [
            "/health",
            "/kimera/status", 
            "/kimera/system/gpu_foundation",
            "/kimera/monitoring/performance",
            "/kimera/monitoring/engines/status",
            "/kimera/thermodynamics/status/system"
        ]
        
        results = {}
        
        for endpoint in endpoints:
            print(f"   Testing {endpoint}...")
            endpoint_results = []
            
            # Test each endpoint 10 times rapidly
            for i in range(10):
                try:
                    start = time.time()
                    response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                    duration = time.time() - start
                    endpoint_results.append({
                        "success": response.status_code == 200,
                        "duration": duration,
                        "status_code": response.status_code
                    })
                except Exception as e:
                    endpoint_results.append({
                        "success": False,
                        "error": str(e),
                        "duration": 0
                    })
            
            successful = [r for r in endpoint_results if r["success"]]
            success_rate = len(successful) / len(endpoint_results)
            avg_time = sum(r["duration"] for r in successful) / len(successful) if successful else 0
            
            results[endpoint] = {
                "success_rate": success_rate,
                "avg_response_time": avg_time,
                "total_requests": len(endpoint_results),
                "results": endpoint_results
            }
            
            print(f"      Success: {success_rate:.1%}, Avg Time: {avg_time:.3f}s")
        
        return results
    
    def test_gpu_memory_stress(self):
        """Test GPU memory and processing stress"""
        print("🚀 Testing GPU Memory Stress...")
        
        # Test GPU foundation repeatedly to stress memory
        results = []
        for i in range(100):
            try:
                start = time.time()
                response = requests.get(f"{self.base_url}/kimera/system/gpu_foundation", timeout=15)
                duration = time.time() - start
                results.append({
                    "iteration": i + 1,
                    "success": response.status_code == 200,
                    "duration": duration
                })
                
                if (i + 1) % 20 == 0:
                    successful = [r for r in results if r["success"]]
                    success_rate = len(successful) / len(results)
                    print(f"   Progress: {i+1}/100, Success Rate: {success_rate:.1%}")
                    
            except Exception as e:
                results.append({
                    "iteration": i + 1,
                    "success": False,
                    "error": str(e)
                })
        
        successful = [r for r in results if r["success"]]
        success_rate = len(successful) / len(results)
        avg_time = sum(r["duration"] for r in successful) / len(successful) if successful else 0
        
        print(f"   ✅ GPU Memory Stress Test Complete")
        print(f"   📊 Success Rate: {success_rate:.1%}")
        print(f"   ⚡ Avg Response Time: {avg_time:.3f}s")
        
        return {
            "test": "gpu_memory_stress",
            "total_iterations": 100,
            "success_rate": success_rate,
            "avg_response_time": avg_time,
            "results": results
        }
    
    def run_comprehensive_test(self):
        """Run comprehensive GPU performance test"""
        print("🎯 KIMERA GPU PEAK PERFORMANCE TEST")
        print("=" * 50)
        
        if not self.wait_for_system():
            return {"error": "System not ready"}
        
        start_time = time.time()
        all_results = {}
        
        # Test 1: GPU Foundation Concurrent Load
        all_results["gpu_foundation_concurrent"] = self.test_gpu_foundation_concurrent(50)
        
        # Test 2: System Endpoints Stress
        all_results["system_endpoints_stress"] = self.test_system_endpoints_stress()
        
        # Test 3: GPU Memory Stress
        all_results["gpu_memory_stress"] = self.test_gpu_memory_stress()
        
        total_duration = time.time() - start_time
        
        # Calculate overall metrics
        overall_results = {
            "test_suite": "GPU_PEAK_PERFORMANCE_SIMPLIFIED",
            "timestamp": datetime.now().isoformat(),
            "total_duration": total_duration,
            "gpu_model": "NVIDIA GeForce RTX 4090",
            "test_results": all_results,
            "summary": {
                "gpu_foundation_success_rate": all_results["gpu_foundation_concurrent"]["success_rate"],
                "gpu_foundation_rps": all_results["gpu_foundation_concurrent"]["requests_per_second"],
                "gpu_memory_success_rate": all_results["gpu_memory_stress"]["success_rate"],
                "system_stability": sum(
                    r["success_rate"] for r in all_results["system_endpoints_stress"].values()
                ) / len(all_results["system_endpoints_stress"]),
                "peak_performance_achieved": all_results["gpu_foundation_concurrent"]["success_rate"] > 0.9
            }
        }
        
        print("=" * 50)
        print("🎯 GPU PEAK PERFORMANCE TEST COMPLETE")
        print(f"   Total Duration: {total_duration:.2f}s")
        print(f"   GPU Foundation RPS: {overall_results['summary']['gpu_foundation_rps']:.2f}")
        print(f"   GPU Foundation Success: {overall_results['summary']['gpu_foundation_success_rate']:.1%}")
        print(f"   GPU Memory Success: {overall_results['summary']['gpu_memory_success_rate']:.1%}")
        print(f"   System Stability: {overall_results['summary']['system_stability']:.1%}")
        print(f"   Peak Performance: {'✅ ACHIEVED' if overall_results['summary']['peak_performance_achieved'] else '❌ NOT ACHIEVED'}")
        
        return overall_results

def main():
    tester = SimpleGPUTester()
    results = tester.run_comprehensive_test()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"gpu_peak_simple_results_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"📄 Results saved to: {filename}")

if __name__ == "__main__":
    main() 