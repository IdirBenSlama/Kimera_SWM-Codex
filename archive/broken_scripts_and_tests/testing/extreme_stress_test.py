#!/usr/bin/env python3
"""
🔥 EXTREME STRESS TEST FOR KIMERA 🔥
Push the system to its absolute limits
"""

import asyncio
import aiohttp
import time
import psutil
import threading
import statistics
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
import json

@dataclass
class ExtremeTestResult:
    test_name: str
    concurrent_users: int
    total_operations: int
    duration_seconds: float
    successful_operations: int
    failed_operations: int
    success_rate: float
    peak_throughput: float
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    max_latency_ms: float
    peak_cpu_percent: float
    peak_memory_percent: float
    errors_count: int
    breaking_point_reached: bool

class ExtremeStressTester:
    def __init__(self, base_url: str = "http://127.0.0.1:8001"):
        self.base_url = base_url
        self.monitoring_active = False
        self.peak_cpu = 0
        self.peak_memory = 0
        
        print("🔥 KIMERA EXTREME STRESS TESTER")
        print("=" * 50)
        print("WARNING: This test will push the system to its limits")
        print("Monitor system resources carefully")
        print("=" * 50)

    def start_resource_monitoring(self):
        """Monitor system resources during extreme testing"""
        self.monitoring_active = True
        self.peak_cpu = 0
        self.peak_memory = 0
        
        def monitor():
            while self.monitoring_active:
                try:
                    cpu = psutil.cpu_percent(interval=0.1)
                    memory = psutil.virtual_memory().percent
                    self.peak_cpu = max(self.peak_cpu, cpu)
                    self.peak_memory = max(self.peak_memory, memory)
                    time.sleep(0.1)
                except:
                    pass
        
        threading.Thread(target=monitor, daemon=True).start()

    def stop_resource_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring_active = False

    async def extreme_concurrent_test(self, concurrent_users: int, operations_per_user: int) -> ExtremeTestResult:
        """Run extreme concurrent load test"""
        test_name = f"Extreme_Concurrent_{concurrent_users}_users"
        total_operations = concurrent_users * operations_per_user
        
        print(f"\n🚀 EXTREME TEST: {concurrent_users} concurrent users")
        print(f"   Operations per user: {operations_per_user}")
        print(f"   Total operations: {total_operations}")
        
        self.start_resource_monitoring()
        
        latencies = []
        successful = 0
        failed = 0
        errors = []
        
        start_time = time.time()
        
        # Create semaphore for controlling concurrency
        semaphore = asyncio.Semaphore(concurrent_users)
        
        async def user_session(user_id: int):
            """Simulate a user session with multiple operations"""
            user_successful = 0
            user_failed = 0
            user_latencies = []
            
            async with aiohttp.ClientSession() as session:
                for op in range(operations_per_user):
                    async with semaphore:
                        try:
                            op_start = time.time()
                            async with session.get(f"{self.base_url}/system/health") as response:
                                await response.text()
                                latency = (time.time() - op_start) * 1000
                                user_latencies.append(latency)
                                
                                if response.status == 200:
                                    user_successful += 1
                                else:
                                    user_failed += 1
                                    
                        except Exception as e:
                            user_failed += 1
                            errors.append(str(e))
                        
                        # Brief pause to prevent overwhelming
                        await asyncio.sleep(0.001)
            
            return user_successful, user_failed, user_latencies
        
        # Create and execute user sessions
        tasks = [user_session(i) for i in range(concurrent_users)]
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for result in results:
                if isinstance(result, tuple):
                    s, f, l = result
                    successful += s
                    failed += f
                    latencies.extend(l)
                else:
                    failed += operations_per_user
                    errors.append(str(result))
                    
        except Exception as e:
            print(f"❌ Extreme test failed: {e}")
            failed += total_operations - successful
            errors.append(str(e))
        
        end_time = time.time()
        duration = end_time - start_time
        
        self.stop_resource_monitoring()
        
        # Calculate metrics
        success_rate = (successful / total_operations) * 100 if total_operations > 0 else 0
        peak_throughput = successful / duration if duration > 0 else 0
        avg_latency = statistics.mean(latencies) if latencies else 0
        p95_latency = np.percentile(latencies, 95) if latencies else 0
        p99_latency = np.percentile(latencies, 99) if latencies else 0
        max_latency = max(latencies) if latencies else 0
        breaking_point = success_rate < 95.0  # Consider <95% success rate as breaking point
        
        result = ExtremeTestResult(
            test_name=test_name,
            concurrent_users=concurrent_users,
            total_operations=total_operations,
            duration_seconds=duration,
            successful_operations=successful,
            failed_operations=failed,
            success_rate=success_rate,
            peak_throughput=peak_throughput,
            avg_latency_ms=avg_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            max_latency_ms=max_latency,
            peak_cpu_percent=self.peak_cpu,
            peak_memory_percent=self.peak_memory,
            errors_count=len(errors),
            breaking_point_reached=breaking_point
        )
        
        # Print results
        print(f"   ✅ Duration: {duration:.1f}s")
        print(f"   📊 Success Rate: {success_rate:.1f}%")
        print(f"   ⚡ Peak Throughput: {peak_throughput:.1f} ops/sec")
        print(f"   🕒 Avg Latency: {avg_latency:.1f}ms")
        print(f"   📈 P95 Latency: {p95_latency:.1f}ms")
        print(f"   💻 Peak CPU: {self.peak_cpu:.1f}%")
        print(f"   💾 Peak Memory: {self.peak_memory:.1f}%")
        print(f"   ❌ Errors: {len(errors)}")
        
        if breaking_point:
            print(f"   🔥 BREAKING POINT REACHED at {concurrent_users} users")
        else:
            print(f"   ✅ System stable at {concurrent_users} users")
        
        return result

    async def find_breaking_point(self):
        """Find the system's breaking point by gradually increasing load"""
        print("\n🎯 FINDING SYSTEM BREAKING POINT")
        print("=" * 50)
        
        # Test configurations: (concurrent_users, operations_per_user)
        test_configs = [
            (50, 20),    # 1,000 total ops
            (100, 20),   # 2,000 total ops
            (200, 20),   # 4,000 total ops
            (500, 20),   # 10,000 total ops
            (1000, 10),  # 10,000 total ops
            (1500, 10),  # 15,000 total ops
            (2000, 10),  # 20,000 total ops
            (3000, 5),   # 15,000 total ops
            (5000, 5),   # 25,000 total ops
        ]
        
        results = []
        breaking_point_found = False
        
        for concurrent_users, ops_per_user in test_configs:
            if breaking_point_found:
                print(f"⏹️  Stopping at breaking point")
                break
                
            result = await self.extreme_concurrent_test(concurrent_users, ops_per_user)
            results.append(result)
            
            # Check if we've reached breaking point
            if result.breaking_point_reached:
                breaking_point_found = True
                print(f"🔥 BREAKING POINT FOUND: {concurrent_users} concurrent users")
            
            # Brief pause between tests
            await asyncio.sleep(5)
        
        return results

    def generate_extreme_test_report(self, results):
        """Generate comprehensive extreme test report"""
        print("\n" + "=" * 80)
        print("🔥 EXTREME STRESS TEST RESULTS")
        print("=" * 80)
        
        # Find maximum stable configuration
        max_stable_users = 0
        max_stable_throughput = 0
        breaking_point_users = None
        
        for result in results:
            if not result.breaking_point_reached:
                max_stable_users = max(max_stable_users, result.concurrent_users)
                max_stable_throughput = max(max_stable_throughput, result.peak_throughput)
            else:
                if breaking_point_users is None:
                    breaking_point_users = result.concurrent_users
        
        print(f"\n📊 EXTREME TEST SUMMARY")
        print(f"   Tests Completed: {len(results)}")
        print(f"   Maximum Stable Users: {max_stable_users}")
        print(f"   Maximum Stable Throughput: {max_stable_throughput:.1f} ops/sec")
        if breaking_point_users:
            print(f"   Breaking Point: {breaking_point_users} concurrent users")
        else:
            print(f"   Breaking Point: Not reached in testing")
        
        print(f"\n📋 DETAILED RESULTS")
        for result in results:
            status = "🔥 BREAKING POINT" if result.breaking_point_reached else "✅ STABLE"
            print(f"\n   {status} - {result.concurrent_users} users:")
            print(f"      Success Rate: {result.success_rate:.1f}%")
            print(f"      Throughput: {result.peak_throughput:.1f} ops/sec")
            print(f"      Avg Latency: {result.avg_latency_ms:.1f}ms")
            print(f"      P95 Latency: {result.p95_latency_ms:.1f}ms")
            print(f"      Peak CPU: {result.peak_cpu_percent:.1f}%")
            print(f"      Peak Memory: {result.peak_memory_percent:.1f}%")
            print(f"      Errors: {result.errors_count}")
        
        # System capacity assessment
        print(f"\n🎯 SYSTEM CAPACITY ASSESSMENT")
        if max_stable_users >= 2000:
            print("   🏆 ENTERPRISE GRADE: >2000 concurrent users supported")
        elif max_stable_users >= 1000:
            print("   ✅ PRODUCTION GRADE: >1000 concurrent users supported")
        elif max_stable_users >= 500:
            print("   ✅ BUSINESS GRADE: >500 concurrent users supported")
        elif max_stable_users >= 200:
            print("   ⚠️  STANDARD GRADE: >200 concurrent users supported")
        else:
            print("   ❌ LIMITED CAPACITY: <200 concurrent users supported")
        
        if max_stable_throughput >= 5000:
            print("   🚀 ULTRA HIGH THROUGHPUT: >5000 ops/sec")
        elif max_stable_throughput >= 2000:
            print("   🚀 HIGH THROUGHPUT: >2000 ops/sec")
        elif max_stable_throughput >= 1000:
            print("   ✅ GOOD THROUGHPUT: >1000 ops/sec")
        else:
            print("   ⚠️  MODERATE THROUGHPUT: <1000 ops/sec")
        
        # Save results
        report_data = {
            "extreme_test_summary": {
                "test_date": datetime.now().isoformat(),
                "tests_completed": len(results),
                "max_stable_users": max_stable_users,
                "max_stable_throughput": max_stable_throughput,
                "breaking_point_users": breaking_point_users
            },
            "detailed_results": [
                {
                    "test_name": r.test_name,
                    "concurrent_users": r.concurrent_users,
                    "total_operations": r.total_operations,
                    "success_rate": r.success_rate,
                    "peak_throughput": r.peak_throughput,
                    "avg_latency_ms": r.avg_latency_ms,
                    "p95_latency_ms": r.p95_latency_ms,
                    "p99_latency_ms": r.p99_latency_ms,
                    "peak_cpu_percent": r.peak_cpu_percent,
                    "peak_memory_percent": r.peak_memory_percent,
                    "breaking_point_reached": r.breaking_point_reached
                }
                for r in results
            ]
        }
        
        filename = f"kimera_extreme_stress_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n💾 Extreme test report saved to: {filename}")
        print("=" * 80)

async def main():
    """Run extreme stress tests"""
    import requests
    
    # Check server availability
    try:
        response = requests.get("http://127.0.0.1:8001/health", timeout=5)
        if response.status_code != 200:
            print("❌ KIMERA server not responding")
            return
    except Exception as e:
        print(f"❌ Cannot connect to KIMERA server: {e}")
        return
    
    tester = ExtremeStressTester()
    
    print("\n⚠️  WARNING: EXTREME STRESS TEST")
    print("This test will push your system to its limits.")
    print("Monitor system resources and be prepared to stop if needed.")
    
    # Wait for user confirmation
    input("\nPress ENTER to continue with extreme testing...")
    
    # Run extreme tests
    results = await tester.find_breaking_point()
    
    # Generate report
    tester.generate_extreme_test_report(results)

if __name__ == "__main__":
    asyncio.run(main()) 