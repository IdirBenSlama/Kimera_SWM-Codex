#!/usr/bin/env python3
"""
🚀 COMPREHENSIVE KIMERA PERFORMANCE TEST SUITE 🚀
Tests all aspects of the KIMERA system under maximum load conditions
"""

import asyncio
import aiohttp
import time
import psutil
import requests
import threading
import json
import statistics
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import subprocess
import sys
import gc
import tracemalloc

# System monitoring
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
    GPU_COUNT = torch.cuda.device_count() if CUDA_AVAILABLE else 0
except ImportError:
    CUDA_AVAILABLE = False
    GPU_COUNT = 0

@dataclass
class PerformanceResult:
    """Individual performance test result"""
    test_name: str
    duration_seconds: float
    operations_completed: int
    success_rate: float
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_ops_per_sec: float
    memory_usage_mb: float
    cpu_usage_percent: float
    gpu_usage_percent: float
    errors: List[str]

@dataclass
class SystemMetrics:
    """System resource metrics"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_usage_percent: float
    network_sent_mb: float
    network_recv_mb: float
    gpu_usage_percent: float
    gpu_memory_used_mb: float
    gpu_temperature_c: float

class ComprehensivePerformanceTest:
    """Comprehensive performance testing suite for KIMERA"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8001"):
        self.base_url = base_url
        self.results: List[PerformanceResult] = []
        self.system_metrics: List[SystemMetrics] = []
        self.monitoring_active = False
        self.start_time = None
        
        # Test configurations
        self.test_configs = {
            "light": {"concurrent_users": 10, "operations": 100, "duration": 60},
            "medium": {"concurrent_users": 50, "operations": 500, "duration": 300},
            "heavy": {"concurrent_users": 200, "operations": 2000, "duration": 600},
            "extreme": {"concurrent_users": 1000, "operations": 10000, "duration": 1800}
        }
        
        print("🚀 KIMERA COMPREHENSIVE PERFORMANCE TEST SUITE")
        print("=" * 60)
        print(f"Target URL: {self.base_url}")
        print(f"CUDA Available: {CUDA_AVAILABLE}")
        print(f"GPU Count: {GPU_COUNT}")
        print(f"CPU Cores: {psutil.cpu_count()}")
        print(f"Total RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        print("=" * 60)

    def start_system_monitoring(self):
        """Start continuous system monitoring"""
        self.monitoring_active = True
        
        def monitor():
            while self.monitoring_active:
                try:
                    # Basic system metrics
                    cpu_percent = psutil.cpu_percent(interval=0.1)
                    memory = psutil.virtual_memory()
                    disk = psutil.disk_usage('/')
                    net = psutil.net_io_counters()
                    
                    # GPU metrics
                    gpu_usage = 0
                    gpu_memory = 0
                    gpu_temp = 0
                    
                    if CUDA_AVAILABLE:
                        try:
                            gpu_usage = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0
                            gpu_memory = torch.cuda.memory_allocated() / (1024**2)
                            # Temperature would require nvidia-ml-py
                        except:
                            pass
                    
                    metrics = SystemMetrics(
                        timestamp=time.time(),
                        cpu_percent=cpu_percent,
                        memory_percent=memory.percent,
                        memory_used_mb=memory.used / (1024**2),
                        disk_usage_percent=disk.percent,
                        network_sent_mb=net.bytes_sent / (1024**2),
                        network_recv_mb=net.bytes_recv / (1024**2),
                        gpu_usage_percent=gpu_usage,
                        gpu_memory_used_mb=gpu_memory,
                        gpu_temperature_c=gpu_temp
                    )
                    
                    self.system_metrics.append(metrics)
                    time.sleep(1)  # Monitor every second
                    
                except Exception as e:
                    print(f"Monitoring error: {e}")
                    time.sleep(1)
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        print("✅ System monitoring started")

    def stop_system_monitoring(self):
        """Stop system monitoring"""
        self.monitoring_active = False
        print("✅ System monitoring stopped")

    async def test_api_endpoint_performance(self, test_level: str = "medium") -> PerformanceResult:
        """Test API endpoint performance under load"""
        config = self.test_configs[test_level]
        test_name = f"API_Endpoint_Performance_{test_level}"
        
        print(f"\n🌐 Testing API Endpoint Performance - {test_level.upper()}")
        print(f"   Concurrent Users: {config['concurrent_users']}")
        print(f"   Operations: {config['operations']}")
        print(f"   Duration: {config['duration']}s")
        
        endpoints = [
            "/",
            "/health", 
            "/system/health",
            "/system/status",
            "/metrics"
        ]
        
        latencies = []
        errors = []
        successful_ops = 0
        failed_ops = 0
        
        start_time = time.time()
        start_memory = psutil.virtual_memory().used / (1024**2)
        start_cpu = psutil.cpu_percent()
        
        async with aiohttp.ClientSession() as session:
            # Create semaphore to limit concurrent requests
            semaphore = asyncio.Semaphore(config['concurrent_users'])
            
            async def make_request(endpoint: str) -> Tuple[bool, float]:
                async with semaphore:
                    try:
                        request_start = time.time()
                        async with session.get(f"{self.base_url}{endpoint}") as response:
                            await response.text()
                            latency = (time.time() - request_start) * 1000
                            return response.status == 200, latency
                    except Exception as e:
                        return False, 0
            
            # Generate tasks
            tasks = []
            for _ in range(config['operations']):
                endpoint = endpoints[len(tasks) % len(endpoints)]
                tasks.append(make_request(endpoint))
            
            # Execute tasks
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for result in results:
                if isinstance(result, tuple):
                    success, latency = result
                    if success:
                        successful_ops += 1
                        latencies.append(latency)
                    else:
                        failed_ops += 1
                else:
                    failed_ops += 1
                    errors.append(str(result))
        
        end_time = time.time()
        duration = end_time - start_time
        end_memory = psutil.virtual_memory().used / (1024**2)
        end_cpu = psutil.cpu_percent()
        
        # Calculate metrics
        success_rate = successful_ops / (successful_ops + failed_ops) * 100
        throughput = (successful_ops + failed_ops) / duration
        avg_latency = statistics.mean(latencies) if latencies else 0
        min_latency = min(latencies) if latencies else 0
        max_latency = max(latencies) if latencies else 0
        p95_latency = np.percentile(latencies, 95) if latencies else 0
        p99_latency = np.percentile(latencies, 99) if latencies else 0
        
        result = PerformanceResult(
            test_name=test_name,
            duration_seconds=duration,
            operations_completed=successful_ops + failed_ops,
            success_rate=success_rate,
            avg_latency_ms=avg_latency,
            min_latency_ms=min_latency,
            max_latency_ms=max_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            throughput_ops_per_sec=throughput,
            memory_usage_mb=end_memory - start_memory,
            cpu_usage_percent=(start_cpu + end_cpu) / 2,
            gpu_usage_percent=0,  # Would be calculated if GPU monitoring available
            errors=errors[:10]  # First 10 errors
        )
        
        self.results.append(result)
        
        print(f"   ✅ Completed: {successful_ops}/{successful_ops + failed_ops} successful")
        print(f"   📊 Success Rate: {success_rate:.1f}%")
        print(f"   ⚡ Throughput: {throughput:.1f} ops/sec")
        print(f"   🕒 Avg Latency: {avg_latency:.1f}ms")
        print(f"   📈 P95 Latency: {p95_latency:.1f}ms")
        
        return result

    def test_concurrent_load(self, test_level: str = "medium") -> PerformanceResult:
        """Test concurrent load handling"""
        config = self.test_configs[test_level]
        test_name = f"Concurrent_Load_{test_level}"
        
        print(f"\n👥 Testing Concurrent Load - {test_level.upper()}")
        
        latencies = []
        errors = []
        successful_ops = 0
        failed_ops = 0
        
        start_time = time.time()
        
        def worker_thread(thread_id: int, operations: int) -> Tuple[int, int, List[float], List[str]]:
            """Worker thread for concurrent testing"""
            thread_successful = 0
            thread_failed = 0
            thread_latencies = []
            thread_errors = []
            
            for i in range(operations):
                try:
                    request_start = time.time()
                    response = requests.get(f"{self.base_url}/system/health", timeout=30)
                    latency = (time.time() - request_start) * 1000
                    
                    if response.status_code == 200:
                        thread_successful += 1
                        thread_latencies.append(latency)
                    else:
                        thread_failed += 1
                        thread_errors.append(f"HTTP {response.status_code}")
                        
                except Exception as e:
                    thread_failed += 1
                    thread_errors.append(str(e))
            
            return thread_successful, thread_failed, thread_latencies, thread_errors
        
        # Execute concurrent workers
        with ThreadPoolExecutor(max_workers=config['concurrent_users']) as executor:
            ops_per_thread = config['operations'] // config['concurrent_users']
            futures = []
            
            for thread_id in range(config['concurrent_users']):
                future = executor.submit(worker_thread, thread_id, ops_per_thread)
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                try:
                    s, f, l, e = future.result()
                    successful_ops += s
                    failed_ops += f
                    latencies.extend(l)
                    errors.extend(e)
                except Exception as e:
                    failed_ops += ops_per_thread
                    errors.append(str(e))
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Calculate metrics
        success_rate = successful_ops / (successful_ops + failed_ops) * 100 if (successful_ops + failed_ops) > 0 else 0
        throughput = (successful_ops + failed_ops) / duration
        avg_latency = statistics.mean(latencies) if latencies else 0
        min_latency = min(latencies) if latencies else 0
        max_latency = max(latencies) if latencies else 0
        p95_latency = np.percentile(latencies, 95) if latencies else 0
        p99_latency = np.percentile(latencies, 99) if latencies else 0
        
        result = PerformanceResult(
            test_name=test_name,
            duration_seconds=duration,
            operations_completed=successful_ops + failed_ops,
            success_rate=success_rate,
            avg_latency_ms=avg_latency,
            min_latency_ms=min_latency,
            max_latency_ms=max_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            throughput_ops_per_sec=throughput,
            memory_usage_mb=0,
            cpu_usage_percent=0,
            gpu_usage_percent=0,
            errors=errors[:10]
        )
        
        self.results.append(result)
        
        print(f"   ✅ Completed: {successful_ops}/{successful_ops + failed_ops} successful")
        print(f"   📊 Success Rate: {success_rate:.1f}%")
        print(f"   ⚡ Throughput: {throughput:.1f} ops/sec")
        
        return result

    def test_memory_stress(self) -> PerformanceResult:
        """Test memory usage under stress"""
        test_name = "Memory_Stress_Test"
        
        print(f"\n💾 Testing Memory Stress")
        
        start_time = time.time()
        start_memory = psutil.virtual_memory().used / (1024**2)
        
        # Enable memory tracing
        tracemalloc.start()
        
        # Simulate memory-intensive operations
        data_blocks = []
        operations = 0
        errors = []
        
        try:
            # Allocate memory in blocks
            for i in range(100):
                # Allocate 10MB blocks
                block = bytearray(10 * 1024 * 1024)
                data_blocks.append(block)
                operations += 1
                
                # Test API endpoint while under memory pressure
                try:
                    response = requests.get(f"{self.base_url}/system/health", timeout=10)
                    if response.status_code != 200:
                        errors.append(f"API failed under memory pressure: {response.status_code}")
                except Exception as e:
                    errors.append(f"API error under memory pressure: {e}")
                
                # Brief pause
                time.sleep(0.1)
                
        except MemoryError as e:
            errors.append(f"Memory allocation failed: {e}")
        except Exception as e:
            errors.append(f"Unexpected error: {e}")
        finally:
            # Clean up
            data_blocks.clear()
            gc.collect()
        
        # Get memory tracing results
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        end_time = time.time()
        end_memory = psutil.virtual_memory().used / (1024**2)
        duration = end_time - start_time
        
        result = PerformanceResult(
            test_name=test_name,
            duration_seconds=duration,
            operations_completed=operations,
            success_rate=(operations - len(errors)) / operations * 100 if operations > 0 else 0,
            avg_latency_ms=0,
            min_latency_ms=0,
            max_latency_ms=0,
            p95_latency_ms=0,
            p99_latency_ms=0,
            throughput_ops_per_sec=operations / duration,
            memory_usage_mb=peak / (1024**2),
            cpu_usage_percent=0,
            gpu_usage_percent=0,
            errors=errors
        )
        
        self.results.append(result)
        
        print(f"   ✅ Operations: {operations}")
        print(f"   💾 Peak Memory: {peak / (1024**2):.1f} MB")
        print(f"   📊 Success Rate: {result.success_rate:.1f}%")
        
        return result

    def test_long_duration_stability(self, duration_minutes: int = 10) -> PerformanceResult:
        """Test system stability over extended period"""
        test_name = f"Long_Duration_Stability_{duration_minutes}min"
        
        print(f"\n⏱️  Testing Long Duration Stability - {duration_minutes} minutes")
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        latencies = []
        errors = []
        successful_ops = 0
        failed_ops = 0
        
        while time.time() < end_time:
            try:
                request_start = time.time()
                response = requests.get(f"{self.base_url}/system/health", timeout=10)
                latency = (time.time() - request_start) * 1000
                
                if response.status_code == 200:
                    successful_ops += 1
                    latencies.append(latency)
                else:
                    failed_ops += 1
                    errors.append(f"HTTP {response.status_code}")
                    
            except Exception as e:
                failed_ops += 1
                errors.append(str(e))
            
            # Wait between requests
            time.sleep(1)
        
        actual_duration = time.time() - start_time
        
        # Calculate metrics
        success_rate = successful_ops / (successful_ops + failed_ops) * 100 if (successful_ops + failed_ops) > 0 else 0
        throughput = (successful_ops + failed_ops) / actual_duration
        avg_latency = statistics.mean(latencies) if latencies else 0
        
        result = PerformanceResult(
            test_name=test_name,
            duration_seconds=actual_duration,
            operations_completed=successful_ops + failed_ops,
            success_rate=success_rate,
            avg_latency_ms=avg_latency,
            min_latency_ms=min(latencies) if latencies else 0,
            max_latency_ms=max(latencies) if latencies else 0,
            p95_latency_ms=np.percentile(latencies, 95) if latencies else 0,
            p99_latency_ms=np.percentile(latencies, 99) if latencies else 0,
            throughput_ops_per_sec=throughput,
            memory_usage_mb=0,
            cpu_usage_percent=0,
            gpu_usage_percent=0,
            errors=errors[:10]
        )
        
        self.results.append(result)
        
        print(f"   ✅ Duration: {actual_duration/60:.1f} minutes")
        print(f"   📊 Success Rate: {success_rate:.1f}%")
        print(f"   ⚡ Avg Throughput: {throughput:.1f} ops/sec")
        
        return result

    async def run_comprehensive_test_suite(self):
        """Run the complete comprehensive test suite"""
        print("\n🚀 STARTING COMPREHENSIVE PERFORMANCE TEST SUITE")
        print("=" * 60)
        
        self.start_time = datetime.now()
        self.start_system_monitoring()
        
        try:
            # Phase 1: Basic API Performance Tests
            print("\n📋 PHASE 1: API Performance Tests")
            await self.test_api_endpoint_performance("light")
            await self.test_api_endpoint_performance("medium")
            await self.test_api_endpoint_performance("heavy")
            
            # Phase 2: Concurrent Load Tests
            print("\n📋 PHASE 2: Concurrent Load Tests")
            self.test_concurrent_load("light")
            self.test_concurrent_load("medium")
            self.test_concurrent_load("heavy")
            
            # Phase 3: Resource Stress Tests
            print("\n📋 PHASE 3: Resource Stress Tests")
            self.test_memory_stress()
            
            # Phase 4: Stability Tests
            print("\n📋 PHASE 4: Stability Tests")
            self.test_long_duration_stability(5)  # 5 minute stability test
            
        finally:
            self.stop_system_monitoring()
        
        # Generate comprehensive report
        self.generate_comprehensive_report()

    def generate_comprehensive_report(self):
        """Generate comprehensive performance report"""
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds()
        
        print("\n" + "=" * 80)
        print("🎯 COMPREHENSIVE PERFORMANCE TEST RESULTS")
        print("=" * 80)
        
        print(f"\n📊 TEST SUMMARY")
        print(f"   Start Time: {self.start_time}")
        print(f"   End Time: {end_time}")
        print(f"   Total Duration: {total_duration/60:.1f} minutes")
        print(f"   Tests Completed: {len(self.results)}")
        
        # Overall statistics
        all_success_rates = [r.success_rate for r in self.results]
        all_throughputs = [r.throughput_ops_per_sec for r in self.results]
        all_latencies = [r.avg_latency_ms for r in self.results if r.avg_latency_ms > 0]
        
        print(f"\n📈 OVERALL PERFORMANCE METRICS")
        print(f"   Average Success Rate: {statistics.mean(all_success_rates):.1f}%")
        print(f"   Peak Throughput: {max(all_throughputs):.1f} ops/sec")
        print(f"   Average Latency: {statistics.mean(all_latencies):.1f}ms" if all_latencies else "   Average Latency: N/A")
        
        # Individual test results
        print(f"\n📋 INDIVIDUAL TEST RESULTS")
        for result in self.results:
            print(f"\n   🔍 {result.test_name}")
            print(f"      Duration: {result.duration_seconds:.1f}s")
            print(f"      Operations: {result.operations_completed}")
            print(f"      Success Rate: {result.success_rate:.1f}%")
            print(f"      Throughput: {result.throughput_ops_per_sec:.1f} ops/sec")
            if result.avg_latency_ms > 0:
                print(f"      Avg Latency: {result.avg_latency_ms:.1f}ms")
                print(f"      P95 Latency: {result.p95_latency_ms:.1f}ms")
            if result.errors:
                print(f"      Errors: {len(result.errors)} (showing first few)")
                for error in result.errors[:3]:
                    print(f"        - {error}")
        
        # System resource analysis
        if self.system_metrics:
            cpu_usage = [m.cpu_percent for m in self.system_metrics]
            memory_usage = [m.memory_percent for m in self.system_metrics]
            
            print(f"\n💻 SYSTEM RESOURCE ANALYSIS")
            print(f"   Peak CPU Usage: {max(cpu_usage):.1f}%")
            print(f"   Average CPU Usage: {statistics.mean(cpu_usage):.1f}%")
            print(f"   Peak Memory Usage: {max(memory_usage):.1f}%")
            print(f"   Average Memory Usage: {statistics.mean(memory_usage):.1f}%")
        
        # Performance assessment
        print(f"\n🎯 PERFORMANCE ASSESSMENT")
        
        overall_success_rate = statistics.mean(all_success_rates)
        peak_throughput = max(all_throughputs)
        
        if overall_success_rate >= 99.0:
            print("   ✅ EXCELLENT: >99% success rate - Production ready")
        elif overall_success_rate >= 95.0:
            print("   ✅ GOOD: >95% success rate - Suitable for production")
        elif overall_success_rate >= 90.0:
            print("   ⚠️  ACCEPTABLE: >90% success rate - Monitor closely")
        else:
            print("   ❌ POOR: <90% success rate - Requires optimization")
        
        if peak_throughput >= 100:
            print("   ✅ HIGH THROUGHPUT: >100 ops/sec - Excellent performance")
        elif peak_throughput >= 50:
            print("   ✅ GOOD THROUGHPUT: >50 ops/sec - Good performance")
        elif peak_throughput >= 20:
            print("   ⚠️  MODERATE THROUGHPUT: >20 ops/sec - Acceptable")
        else:
            print("   ❌ LOW THROUGHPUT: <20 ops/sec - Needs optimization")
        
        # Save results to file
        report_data = {
            "test_summary": {
                "start_time": self.start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "total_duration_seconds": total_duration,
                "tests_completed": len(self.results)
            },
            "overall_metrics": {
                "average_success_rate": statistics.mean(all_success_rates),
                "peak_throughput": max(all_throughputs),
                "average_latency": statistics.mean(all_latencies) if all_latencies else None
            },
            "test_results": [asdict(result) for result in self.results],
            "system_metrics": [asdict(metric) for metric in self.system_metrics[-100:]]  # Last 100 metrics
        }
        
        report_filename = f"kimera_performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n💾 Performance report saved to: {report_filename}")
        print("=" * 80)

async def main():
    """Main test execution"""
    # Check if server is running
    try:
        response = requests.get("http://127.0.0.1:8001/health", timeout=5)
        if response.status_code != 200:
            print("❌ KIMERA server is not responding properly")
            print("   Please ensure the server is running with: python minimal_server.py")
            return
    except Exception as e:
        print("❌ KIMERA server is not accessible")
        print(f"   Error: {e}")
        print("   Please ensure the server is running with: python minimal_server.py")
        return
    
    # Run comprehensive tests
    tester = ComprehensivePerformanceTest()
    await tester.run_comprehensive_test_suite()

if __name__ == "__main__":
    asyncio.run(main()) 