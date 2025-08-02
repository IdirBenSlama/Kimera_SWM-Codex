#!/usr/bin/env python3
"""
KIMERA SWM Performance Testing Suite
====================================

Comprehensive performance testing for Kimera's cognitive engines,
API endpoints, and system resource utilization.

Author: Kimera SWM Autonomous Architect
Date: 2025-02-03
"""

import asyncio
import time
import statistics
import json
import psutil
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Any, Tuple
import numpy as np

# Try to import GPU monitoring
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except:
    GPU_AVAILABLE = False


class KimeraPerformanceTester:
    """Comprehensive performance testing for Kimera SWM"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "system_info": self._get_system_info(),
            "api_tests": {},
            "cognitive_tests": {},
            "resource_usage": {},
            "parallel_tests": {},
            "summary": {}
        }
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Gather system information"""
        info = {
            "cpu_count": psutil.cpu_count(),
            "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "platform": {
                "system": psutil.Process().name(),
                "python_version": str(psutil.Process().pid)
            }
        }
        
        if GPU_AVAILABLE:
            try:
                device_count = pynvml.nvmlDeviceGetCount()
                gpu_info = []
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    gpu_info.append({
                        "name": pynvml.nvmlDeviceGetName(handle).decode(),
                        "memory_total_mb": pynvml.nvmlDeviceGetMemoryInfo(handle).total / (1024**2)
                    })
                info["gpu"] = gpu_info
            except:
                info["gpu"] = "Error reading GPU info"
        
        return info
    
    def _measure_request(self, method: str, endpoint: str, data: Dict = None, 
                        iterations: int = 10) -> Dict[str, float]:
        """Measure API request performance"""
        times = []
        errors = 0
        
        for _ in range(iterations):
            try:
                start = time.perf_counter()
                
                if method == "GET":
                    response = requests.get(f"{self.base_url}{endpoint}", timeout=30)
                elif method == "POST":
                    response = requests.post(f"{self.base_url}{endpoint}", 
                                           json=data, timeout=30)
                
                elapsed = time.perf_counter() - start
                
                if response.status_code == 200:
                    times.append(elapsed)
                else:
                    errors += 1
                    
            except Exception as e:
                errors += 1
                print(f"Error testing {endpoint}: {e}")
        
        if times:
            return {
                "mean_ms": statistics.mean(times) * 1000,
                "median_ms": statistics.median(times) * 1000,
                "min_ms": min(times) * 1000,
                "max_ms": max(times) * 1000,
                "stdev_ms": statistics.stdev(times) * 1000 if len(times) > 1 else 0,
                "success_rate": len(times) / iterations,
                "errors": errors
            }
        else:
            return {"error": "All requests failed", "errors": errors}
    
    def test_api_endpoints(self):
        """Test various API endpoints"""
        print("\n🔍 Testing API Endpoints...")
        
        endpoints = [
            ("GET", "/health", None),
            ("GET", "/api/v1/system/status", None),
            ("GET", "/api/v1/system/components", None),
            ("GET", "/api/v1/metrics", None),
            ("POST", "/api/v1/linguistic/analyze", {
                "text": "What is the meaning of consciousness?",
                "mode": "comprehensive"
            }),
            ("POST", "/api/v1/cognitive/process", {
                "input": "Analyze the relationship between quantum mechanics and consciousness",
                "engines": ["quantum_cognitive", "understanding", "contradiction"]
            }),
            ("POST", "/api/v1/contradiction/detect", {
                "statements": [
                    "All swans are white",
                    "Some swans are black"
                ]
            })
        ]
        
        for method, endpoint, data in endpoints:
            print(f"  Testing {method} {endpoint}...")
            self.results["api_tests"][endpoint] = self._measure_request(
                method, endpoint, data
            )
    
    def test_cognitive_engines(self):
        """Test cognitive engine performance"""
        print("\n🧠 Testing Cognitive Engines...")
        
        # Test understanding engine with varying complexity
        complexities = [
            ("simple", "What is 2+2?"),
            ("medium", "Explain the concept of emergence in complex systems"),
            ("complex", "Analyze the philosophical implications of Gödel's incompleteness theorems on artificial consciousness and self-referential systems in the context of modern AI architectures")
        ]
        
        for level, query in complexities:
            print(f"  Testing understanding engine - {level} complexity...")
            self.results["cognitive_tests"][f"understanding_{level}"] = \
                self._measure_request("POST", "/api/v1/cognitive/understand", {
                    "query": query,
                    "depth": "full"
                }, iterations=5)
        
        # Test quantum cognitive engine
        print("  Testing quantum cognitive engine...")
        self.results["cognitive_tests"]["quantum_exploration"] = \
            self._measure_request("POST", "/api/v1/cognitive/quantum/explore", {
                "concept": "consciousness",
                "dimensions": 5,
                "iterations": 100
            }, iterations=3)
    
    def test_parallel_processing(self):
        """Test parallel processing capabilities"""
        print("\n⚡ Testing Parallel Processing...")
        
        def parallel_request(i):
            start = time.perf_counter()
            try:
                response = requests.post(f"{self.base_url}/api/v1/linguistic/analyze", 
                                       json={"text": f"Test query {i}", "mode": "fast"},
                                       timeout=30)
                elapsed = time.perf_counter() - start
                return (True, elapsed) if response.status_code == 200 else (False, elapsed)
            except:
                return (False, time.perf_counter() - start)
        
        # Test with different levels of concurrency
        for workers in [1, 5, 10, 20]:
            print(f"  Testing with {workers} concurrent requests...")
            
            start_time = time.perf_counter()
            successes = 0
            times = []
            
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(parallel_request, i) for i in range(workers * 5)]
                
                for future in as_completed(futures):
                    success, elapsed = future.result()
                    if success:
                        successes += 1
                        times.append(elapsed)
            
            total_time = time.perf_counter() - start_time
            
            self.results["parallel_tests"][f"concurrency_{workers}"] = {
                "total_requests": workers * 5,
                "successful_requests": successes,
                "total_time_s": total_time,
                "requests_per_second": (workers * 5) / total_time,
                "mean_response_time_ms": statistics.mean(times) * 1000 if times else 0,
                "success_rate": successes / (workers * 5)
            }
    
    def monitor_resources(self, duration: int = 10):
        """Monitor system resources during operation"""
        print(f"\n📊 Monitoring Resources for {duration} seconds...")
        
        cpu_usage = []
        memory_usage = []
        gpu_usage = []
        gpu_memory = []
        
        # Generate load while monitoring
        def generate_load():
            for i in range(duration):
                requests.post(f"{self.base_url}/api/v1/cognitive/process", 
                            json={"input": f"Complex query {i}", 
                                  "engines": ["all"]},
                            timeout=30)
                time.sleep(0.5)
        
        # Start load generation in background
        import threading
        load_thread = threading.Thread(target=generate_load)
        load_thread.start()
        
        # Monitor resources
        for _ in range(duration * 2):  # Sample twice per second
            cpu_usage.append(psutil.cpu_percent(interval=0.1))
            memory_usage.append(psutil.virtual_memory().percent)
            
            if GPU_AVAILABLE:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    gpu_usage.append(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
                    gpu_memory.append(pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024**2))
                except:
                    pass
            
            time.sleep(0.4)
        
        load_thread.join()
        
        self.results["resource_usage"] = {
            "cpu": {
                "mean_percent": statistics.mean(cpu_usage),
                "max_percent": max(cpu_usage),
                "min_percent": min(cpu_usage),
                "stdev_percent": statistics.stdev(cpu_usage) if len(cpu_usage) > 1 else 0
            },
            "memory": {
                "mean_percent": statistics.mean(memory_usage),
                "max_percent": max(memory_usage),
                "min_percent": min(memory_usage)
            }
        }
        
        if gpu_usage:
            self.results["resource_usage"]["gpu"] = {
                "mean_percent": statistics.mean(gpu_usage),
                "max_percent": max(gpu_usage),
                "mean_memory_mb": statistics.mean(gpu_memory),
                "max_memory_mb": max(gpu_memory)
            }
    
    def generate_summary(self):
        """Generate performance summary"""
        print("\n📈 Generating Summary...")
        
        # API performance summary
        api_times = []
        for endpoint, metrics in self.results["api_tests"].items():
            if "mean_ms" in metrics:
                api_times.append(metrics["mean_ms"])
        
        if api_times:
            self.results["summary"]["api_performance"] = {
                "mean_response_time_ms": statistics.mean(api_times),
                "fastest_endpoint_ms": min(api_times),
                "slowest_endpoint_ms": max(api_times)
            }
        
        # Cognitive performance summary
        cog_times = []
        for test, metrics in self.results["cognitive_tests"].items():
            if "mean_ms" in metrics:
                cog_times.append(metrics["mean_ms"])
        
        if cog_times:
            self.results["summary"]["cognitive_performance"] = {
                "mean_processing_time_ms": statistics.mean(cog_times),
                "complexity_scaling": {
                    "simple": self.results["cognitive_tests"].get("understanding_simple", {}).get("mean_ms", 0),
                    "medium": self.results["cognitive_tests"].get("understanding_medium", {}).get("mean_ms", 0),
                    "complex": self.results["cognitive_tests"].get("understanding_complex", {}).get("mean_ms", 0)
                }
            }
        
        # Parallel processing summary
        if self.results["parallel_tests"]:
            max_rps = max(test["requests_per_second"] 
                         for test in self.results["parallel_tests"].values())
            self.results["summary"]["parallel_performance"] = {
                "max_requests_per_second": max_rps,
                "optimal_concurrency": max(self.results["parallel_tests"].items(),
                                          key=lambda x: x[1]["requests_per_second"])[0]
            }
        
        # Resource usage summary
        if self.results["resource_usage"]:
            self.results["summary"]["resource_efficiency"] = {
                "cpu_utilization": self.results["resource_usage"]["cpu"]["mean_percent"],
                "memory_utilization": self.results["resource_usage"]["memory"]["mean_percent"],
                "gpu_utilization": self.results["resource_usage"].get("gpu", {}).get("mean_percent", "N/A")
            }
    
    def run_all_tests(self):
        """Run all performance tests"""
        print("🚀 Starting Kimera SWM Performance Testing Suite")
        print("=" * 60)
        
        try:
            # Check if service is available
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code != 200:
                print("❌ Kimera service not responding properly")
                return
        except:
            print("❌ Cannot connect to Kimera service at", self.base_url)
            return
        
        # Run tests
        self.test_api_endpoints()
        self.test_cognitive_engines()
        self.test_parallel_processing()
        self.monitor_resources()
        self.generate_summary()
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"docs/reports/performance/kimera_performance_{timestamp}.json"
        
        import os
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✅ Performance testing complete!")
        print(f"📄 Results saved to: {filename}")
        
        # Print summary
        print("\n📊 PERFORMANCE SUMMARY")
        print("=" * 60)
        if "api_performance" in self.results["summary"]:
            print(f"Average API Response Time: {self.results['summary']['api_performance']['mean_response_time_ms']:.1f} ms")
        if "cognitive_performance" in self.results["summary"]:
            print(f"Average Cognitive Processing: {self.results['summary']['cognitive_performance']['mean_processing_time_ms']:.1f} ms")
        if "parallel_performance" in self.results["summary"]:
            print(f"Max Throughput: {self.results['summary']['parallel_performance']['max_requests_per_second']:.1f} req/s")
        if "resource_efficiency" in self.results["summary"]:
            print(f"CPU Usage: {self.results['summary']['resource_efficiency']['cpu_utilization']:.1f}%")
            print(f"Memory Usage: {self.results['summary']['resource_efficiency']['memory_utilization']:.1f}%")
            if self.results['summary']['resource_efficiency']['gpu_utilization'] != "N/A":
                print(f"GPU Usage: {self.results['summary']['resource_efficiency']['gpu_utilization']:.1f}%")


if __name__ == "__main__":
    tester = KimeraPerformanceTester()
    tester.run_all_tests()