#!/usr/bin/env python3
"""
Simple Performance Test for Kimera SWM
Tests system performance and responsiveness
"""

import os
import sys
import time
import json
import requests
import psutil
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add current directory to Python path
sys.path.insert(0, os.path.abspath('.'))

API_BASE_URL = "http://localhost:8000"  # Based on the server startup logs

class SimplePerformanceTest:
    def __init__(self):
        self.results = {}
        self.start_time = time.time()
        
    def print_header(self, title):
        print(f"\n{'='*60}")
        print(f"🚀 {title}")
        print(f"{'='*60}")
        
    def print_result(self, test_name, value, unit="", status="✅"):
        print(f"{status} {test_name}: {value} {unit}")
        
    def test_system_health(self):
        """Test system health and basic API responsiveness"""
        self.print_header("SYSTEM HEALTH CHECK")
        
        # Test basic endpoints
        endpoints = [
            "/docs",
            "/system/health", 
            "/monitoring/status"
        ]
        
        api_results = {}
        
        for endpoint in endpoints:
            try:
                start = time.time()
                response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=10)
                response_time = (time.time() - start) * 1000
                
                if response.status_code == 200:
                    status = "✅"
                    api_results[endpoint] = response_time
                else:
                    status = "❌"
                    api_results[endpoint] = f"Error {response.status_code}"
                    
                self.print_result(f"API {endpoint}", f"{response_time:.1f}ms", "", status)
                
            except Exception as e:
                self.print_result(f"API {endpoint}", f"FAILED: {str(e)}", "", "❌")
                api_results[endpoint] = f"FAILED: {str(e)}"
        
        self.results['api_health'] = api_results
        
    def test_system_resources(self):
        """Test system resource utilization"""
        self.print_header("SYSTEM RESOURCES")
        
        # CPU Usage
        cpu_percent = psutil.cpu_percent(interval=1)
        self.print_result("CPU Usage", f"{cpu_percent:.1f}%")
        
        # Memory Usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_available_gb = memory.available / (1024**3)
        self.print_result("Memory Usage", f"{memory_percent:.1f}%")
        self.print_result("Memory Available", f"{memory_available_gb:.1f} GB")
        
        # Disk Usage
        disk = psutil.disk_usage('.')
        disk_percent = (disk.used / disk.total) * 100
        disk_free_gb = disk.free / (1024**3)
        self.print_result("Disk Usage", f"{disk_percent:.1f}%")
        self.print_result("Disk Free", f"{disk_free_gb:.1f} GB")
        
        self.results['system_resources'] = {
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'memory_available_gb': memory_available_gb,
            'disk_percent': disk_percent,
            'disk_free_gb': disk_free_gb
        }
        
    def test_concurrent_requests(self, num_requests=50):
        """Test concurrent API request handling"""
        self.print_header("CONCURRENT REQUEST TEST")
        
        def make_request():
            try:
                start = time.time()
                response = requests.get(f"{API_BASE_URL}/docs", timeout=5)
                response_time = time.time() - start
                return response.status_code == 200, response_time
            except:
                return False, 0
        
        print(f"Testing {num_requests} concurrent requests...")
        
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(num_requests)]
            results = [future.result() for future in as_completed(futures)]
        
        total_time = time.time() - start_time
        successful_requests = sum(1 for success, _ in results if success)
        response_times = [rt for success, rt in results if success]
        
        if response_times:
            avg_response_time = sum(response_times) / len(response_times) * 1000
            max_response_time = max(response_times) * 1000
            min_response_time = min(response_times) * 1000
        else:
            avg_response_time = max_response_time = min_response_time = 0
        
        success_rate = (successful_requests / num_requests) * 100
        throughput = num_requests / total_time
        
        self.print_result("Total Requests", num_requests)
        self.print_result("Successful Requests", successful_requests)
        self.print_result("Success Rate", f"{success_rate:.1f}%")
        self.print_result("Total Time", f"{total_time:.2f}s")
        self.print_result("Throughput", f"{throughput:.1f} req/s")
        self.print_result("Avg Response Time", f"{avg_response_time:.1f}ms")
        self.print_result("Min Response Time", f"{min_response_time:.1f}ms")
        self.print_result("Max Response Time", f"{max_response_time:.1f}ms")
        
        self.results['concurrent_test'] = {
            'total_requests': num_requests,
            'successful_requests': successful_requests,
            'success_rate': success_rate,
            'total_time': total_time,
            'throughput': throughput,
            'avg_response_time_ms': avg_response_time,
            'min_response_time_ms': min_response_time,
            'max_response_time_ms': max_response_time
        }
        
    def test_gpu_availability(self):
        """Test GPU availability and CUDA status"""
        self.print_header("GPU STATUS")
        
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                current_device = torch.cuda.current_device()
                gpu_name = torch.cuda.get_device_name(current_device)
                gpu_memory = torch.cuda.get_device_properties(current_device).total_memory / (1024**3)
                
                self.print_result("CUDA Available", "Yes")
                self.print_result("GPU Count", gpu_count)
                self.print_result("Current GPU", gpu_name)
                self.print_result("GPU Memory", f"{gpu_memory:.1f} GB")
                
                self.results['gpu_status'] = {
                    'cuda_available': True,
                    'gpu_count': gpu_count,
                    'gpu_name': gpu_name,
                    'gpu_memory_gb': gpu_memory
                }
            else:
                self.print_result("CUDA Available", "No", "", "⚠️")
                self.results['gpu_status'] = {'cuda_available': False}
                
        except ImportError:
            self.print_result("PyTorch", "Not Available", "", "⚠️")
            self.results['gpu_status'] = {'pytorch_available': False}
            
    def run_performance_test(self):
        """Run comprehensive performance test"""
        print(f"🚀 KIMERA SWM PERFORMANCE TEST")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Run all tests
        self.test_system_health()
        self.test_system_resources()
        self.test_gpu_availability()
        self.test_concurrent_requests()
        
        # Summary
        total_time = time.time() - self.start_time
        self.print_header("PERFORMANCE SUMMARY")
        self.print_result("Total Test Time", f"{total_time:.2f}s")
        
        # Save results
        self.results['test_metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'total_test_time': total_time,
            'test_version': '1.0'
        }
        
        # Write results to file
        results_file = f"performance_test_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        self.print_result("Results saved to", results_file)
        
        print(f"\n✅ Performance test completed successfully!")
        return self.results

if __name__ == "__main__":
    test = SimplePerformanceTest()
    test.run_performance_test() 