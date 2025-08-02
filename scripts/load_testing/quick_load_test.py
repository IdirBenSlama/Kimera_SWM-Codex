"""
Quick Kimera Load Test
=====================
Fast production load test for immediate feedback.
"""

import asyncio
import aiohttp
import time
import psutil
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

try:
    import pynvml
    NVML_AVAILABLE = True
    pynvml.nvmlInit()
    # Check for multiple GPUs
    gpu_count = pynvml.nvmlDeviceGetCount()
    print(f"🎮 Detected {gpu_count} GPU(s)")
    for i in range(gpu_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode('utf-8')
        print(f"   GPU {i}: {name}")
except ImportError:
    NVML_AVAILABLE = False
    gpu_count = 0

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuickLoadTester:
    """Quick load testing for immediate feedback"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8000", gpu_id: int = 1):
        self.base_url = base_url
        self.gpu_id = gpu_id
        self.results = []
        
        # Initialize GPU monitoring with specified GPU
        self.gpu_handle = None
        if NVML_AVAILABLE and gpu_count > gpu_id:
            try:
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                gpu_name = pynvml.nvmlDeviceGetName(self.gpu_handle)
                if isinstance(gpu_name, bytes):
                    gpu_name = gpu_name.decode('utf-8')
                logger.info(f"🎮 Monitoring GPU {gpu_id}: {gpu_name}")
            except Exception as e:
                logger.warning(f"Could not access GPU {gpu_id}: {e}")
                # Fallback to GPU 0
                if gpu_count > 0:
                    try:
                        self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        gpu_name = pynvml.nvmlDeviceGetName(self.gpu_handle)
                        if isinstance(gpu_name, bytes):
                            gpu_name = gpu_name.decode('utf-8')
                        logger.info(f"🎮 Fallback to GPU 0: {gpu_name}")
                    except:
                        pass
        elif NVML_AVAILABLE and gpu_count > 0:
            # Use GPU 0 if specified GPU not available
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_name = pynvml.nvmlDeviceGetName(self.gpu_handle)
            if isinstance(gpu_name, bytes):
                gpu_name = gpu_name.decode('utf-8')
            logger.info(f"🎮 Using GPU 0: {gpu_name}")
            
    async def test_endpoint(self, session: aiohttp.ClientSession, method: str, 
                           endpoint: str, payload: Dict = None, iterations: int = 10):
        """Test a single endpoint"""
        logger.info(f"Testing {method} {endpoint} ({iterations} requests)...")
        
        times = []
        successes = 0
        
        for i in range(iterations):
            start_time = time.time()
            try:
                if method.upper() == "GET":
                    async with session.get(f"{self.base_url}{endpoint}") as response:
                        await response.read()
                        status = response.status
                else:  # POST
                    async with session.post(f"{self.base_url}{endpoint}", json=payload) as response:
                        await response.read()
                        status = response.status
                        
                response_time = (time.time() - start_time) * 1000
                times.append(response_time)
                
                if 200 <= status < 400:
                    successes += 1
                    
            except Exception as e:
                logger.error(f"Request failed: {e}")
                
        if times:
            avg_time = sum(times) / len(times)
            success_rate = successes / len(times)
            
            result = {
                "endpoint": endpoint,
                "method": method,
                "avg_response_time_ms": avg_time,
                "success_rate": success_rate,
                "iterations": len(times)
            }
            
            self.results.append(result)
            logger.info(f"  ✅ Avg: {avg_time:.1f}ms, Success: {success_rate:.1%}")
            return result
        else:
            logger.error(f"  ❌ All requests failed")
            return None
            
    async def concurrent_test(self, session: aiohttp.ClientSession, concurrent_users: int = 20):
        """Test concurrent load"""
        logger.info(f"🔥 Concurrent test with {concurrent_users} users...")
        
        async def make_request():
            start_time = time.time()
            try:
                async with session.get(f"{self.base_url}/health") as response:
                    await response.read()
                    return (time.time() - start_time) * 1000, response.status
            except:
                return None, 0
                
        # Launch concurrent requests
        tasks = [make_request() for _ in range(concurrent_users)]
        results = await asyncio.gather(*tasks)
        
        # Analyze results
        times = [r[0] for r in results if r[0] is not None]
        successes = sum(1 for r in results if 200 <= r[1] < 400)
        
        if times:
            avg_time = sum(times) / len(times)
            max_time = max(times)
            min_time = min(times)
            success_rate = successes / len(results)
            
            logger.info(f"  📊 Concurrent Results:")
            logger.info(f"     Avg: {avg_time:.1f}ms, Min: {min_time:.1f}ms, Max: {max_time:.1f}ms")
            logger.info(f"     Success Rate: {success_rate:.1%}")
            
            return {
                "concurrent_users": concurrent_users,
                "avg_response_time_ms": avg_time,
                "min_response_time_ms": min_time,
                "max_response_time_ms": max_time,
                "success_rate": success_rate
            }
        else:
            logger.error("  ❌ All concurrent requests failed")
            return None
            
    def get_system_metrics(self):
        """Get current system performance"""
        # CPU and Memory
        cpu = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # GPU metrics
        gpu_util = gpu_mem = gpu_temp = 0.0
        gpu_name = "N/A"
        
        if self.gpu_handle:
            try:
                gpu_util_info = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                gpu_mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                gpu_util = gpu_util_info.gpu
                gpu_mem = (gpu_mem_info.used / gpu_mem_info.total) * 100
                gpu_temp = pynvml.nvmlDeviceGetTemperature(self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
                gpu_name = pynvml.nvmlDeviceGetName(self.gpu_handle)
                if isinstance(gpu_name, bytes):
                    gpu_name = gpu_name.decode('utf-8')
            except:
                pass
                
        return {
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": cpu,
            "memory_percent": memory.percent,
            "memory_used_gb": memory.used / (1024**3),
            "gpu_id": self.gpu_id,
            "gpu_name": gpu_name,
            "gpu_utilization": gpu_util,
            "gpu_memory_percent": gpu_mem,
            "gpu_temperature": gpu_temp
        }
        
    async def run_quick_test(self):
        """Run comprehensive quick test"""
        logger.info("🚀 STARTING QUICK KIMERA LOAD TEST")
        logger.info("=" * 50)
        
        # Test connectivity first
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/health") as response:
                    if response.status != 200:
                        logger.error("❌ Kimera not responding to health check")
                        return
        except Exception as e:
            logger.error(f"❌ Cannot connect to Kimera: {e}")
            return
            
        # Get baseline metrics
        baseline_metrics = self.get_system_metrics()
        logger.info(f"📊 Baseline - CPU: {baseline_metrics['cpu_percent']:.1f}%, "
                   f"Memory: {baseline_metrics['memory_percent']:.1f}%, "
                   f"GPU: {baseline_metrics['gpu_utilization']:.1f}%")
        
        async with aiohttp.ClientSession() as session:
            # Test basic endpoints
            await self.test_endpoint(session, "GET", "/health", iterations=5)
            await self.test_endpoint(session, "GET", "/api/v1/system/status", iterations=5)
            
            # Test cognitive endpoints
            await self.test_endpoint(session, "POST", "/api/v1/linguistic/analyze", 
                                   {"text": "Test analysis", "level": "basic"}, iterations=3)
            await self.test_endpoint(session, "POST", "/api/v1/cognitive/process",
                                   {"input": "What is AI?", "depth": "simple"}, iterations=3)
            
            # Test concurrent load
            await self.concurrent_test(session, concurrent_users=10)
            await self.concurrent_test(session, concurrent_users=25)
            await self.concurrent_test(session, concurrent_users=50)
            
        # Get final metrics
        final_metrics = self.get_system_metrics()
        logger.info(f"📊 Final - CPU: {final_metrics['cpu_percent']:.1f}%, "
                   f"Memory: {final_metrics['memory_percent']:.1f}%, "
                   f"GPU: {final_metrics['gpu_utilization']:.1f}%")
                   
        # Generate summary
        self.generate_summary(baseline_metrics, final_metrics)
        
    def generate_summary(self, baseline: Dict, final: Dict):
        """Generate test summary"""
        logger.info("\n" + "=" * 50)
        logger.info("📋 QUICK LOAD TEST SUMMARY")
        logger.info("=" * 50)
        
        # Calculate averages
        if self.results:
            avg_response_time = sum(r["avg_response_time_ms"] for r in self.results) / len(self.results)
            avg_success_rate = sum(r["success_rate"] for r in self.results) / len(self.results)
            
            logger.info(f"🎯 Performance Metrics:")
            logger.info(f"   Average Response Time: {avg_response_time:.1f}ms")
            logger.info(f"   Average Success Rate: {avg_success_rate:.1%}")
            
        logger.info(f"\n🖥️ System Utilization:")
        logger.info(f"   CPU: {baseline['cpu_percent']:.1f}% → {final['cpu_percent']:.1f}%")
        logger.info(f"   Memory: {baseline['memory_percent']:.1f}% → {final['memory_percent']:.1f}%")
        logger.info(f"   GPU {final['gpu_id']} ({final['gpu_name']}): {baseline['gpu_utilization']:.1f}% → {final['gpu_utilization']:.1f}%")
        
        # Recommendations
        logger.info(f"\n💡 Recommendations:")
        
        if avg_response_time > 50:
            logger.info("   ⚡ Consider GPU acceleration for response times")
            
        if final['gpu_utilization'] < 20:
            logger.info("   🎮 GPU is underutilized - major opportunity for acceleration")
            
        if final['cpu_percent'] > 80:
            logger.info("   💻 CPU bottleneck detected - consider optimization")
            
        if avg_success_rate < 0.95:
            logger.info("   🔧 Stability issues detected - investigate errors")
        else:
            logger.info("   ✅ System stable - ready for production")
            
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = Path("data/load_testing") / f"quick_test_{timestamp}.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "baseline_metrics": baseline,
            "final_metrics": final,
            "endpoint_results": self.results,
            "average_response_time_ms": avg_response_time if self.results else 0,
            "average_success_rate": avg_success_rate if self.results else 0
        }
        
        with open(results_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"\n📄 Results saved to: {results_path}")
        
        # Next steps
        logger.info(f"\n🚀 NEXT STEPS:")
        logger.info(f"   1. GPU Acceleration: Implement GPU processing")
        logger.info(f"   2. Extended Load Test: Run longer stress tests")
        logger.info(f"   3. Performance Optimization: Target specific bottlenecks")


async def main():
    """Main execution"""
    # Allow GPU selection via command line or default to GPU 1
    import sys
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    
    tester = QuickLoadTester(gpu_id=gpu_id)
    await tester.run_quick_test()


if __name__ == "__main__":
    asyncio.run(main())