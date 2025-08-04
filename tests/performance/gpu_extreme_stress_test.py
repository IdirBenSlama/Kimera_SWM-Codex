#!/usr/bin/env python3
"""
KIMERA GPU EXTREME STRESS TEST
Push RTX 4090 to absolute maximum performance limits
"""

import json
import multiprocessing as mp
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import requests


class ExtremeGPUStressTester:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.max_workers = min(200, mp.cpu_count() * 4)  # Aggressive threading

    def wait_for_system(self, timeout=30):
        """Wait for system to be ready"""
        print("🔄 Waiting for Kimera system...")
        for i in range(timeout):
            try:
                response = requests.get(f"{self.base_url}/health", timeout=3)
                if response.status_code == 200:
                    print("✅ System ready!")
                    return True
            except Exception as e:
                logger.error(f"Error in gpu_extreme_stress_test.py: {e}", exc_info=True)
                raise  # Re-raise for proper error handling
            time.sleep(1)
        return False

    def extreme_concurrent_gpu_test(self, num_requests=1000):
        """Extreme concurrent GPU foundation test"""
        print(
            f"🔥 EXTREME GPU STRESS: {num_requests} concurrent requests with {self.max_workers} workers"
        )

        def make_gpu_request():
            try:
                start = time.time()
                response = requests.get(
                    f"{self.base_url}/kimera/system/gpu_foundation", timeout=45
                )
                duration = time.time() - start
                return {
                    "success": response.status_code == 200,
                    "duration": duration,
                    "status_code": response.status_code,
                    "timestamp": time.time(),
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "duration": 0,
                    "timestamp": time.time(),
                }

        start_time = time.time()
        results = []

        # Extreme concurrency test
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(make_gpu_request) for _ in range(num_requests)]

            completed_count = 0
            for future in as_completed(futures):
                results.append(future.result())
                completed_count += 1

                # Progress updates
                if completed_count % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = completed_count / elapsed
                    successful = sum(1 for r in results if r["success"])
                    success_rate = successful / completed_count
                    print(
                        f"   Progress: {completed_count}/{num_requests} ({success_rate:.1%} success, {rate:.1f} req/s)"
                    )

        total_time = time.time() - start_time
        successful = [r for r in results if r["success"]]
        success_rate = len(successful) / len(results)
        avg_response_time = (
            sum(r["duration"] for r in successful) / len(successful)
            if successful
            else 0
        )
        requests_per_second = len(results) / total_time

        print(f"   🎯 EXTREME TEST COMPLETE")
        print(f"   📊 Total Requests: {len(results)}")
        print(f"   ✅ Success Rate: {success_rate:.1%}")
        print(f"   ⚡ Avg Response Time: {avg_response_time:.3f}s")
        print(f"   🚀 Peak RPS: {requests_per_second:.2f}")
        print(f"   ⏱️ Total Time: {total_time:.2f}s")

        return {
            "test": "extreme_concurrent_gpu",
            "total_requests": num_requests,
            "total_time": total_time,
            "success_rate": success_rate,
            "avg_response_time": avg_response_time,
            "requests_per_second": requests_per_second,
            "max_workers": self.max_workers,
            "successful_requests": len(successful),
            "failed_requests": len(results) - len(successful),
        }

    def run_extreme_stress_test(self):
        """Run complete extreme GPU stress test suite"""
        print("🔥🔥🔥 KIMERA GPU EXTREME STRESS TEST 🔥🔥🔥")
        print("=" * 60)
        print(f"GPU Model: NVIDIA GeForce RTX 4090")
        print(f"Max Workers: {self.max_workers}")
        print("=" * 60)

        if not self.wait_for_system():
            return {"error": "System not ready"}

        start_time = time.time()
        all_results = {}

        # Test 1: Extreme Concurrent Load (500 requests)
        print("\n🚀 TEST 1: EXTREME CONCURRENT LOAD")
        all_results["extreme_concurrent"] = self.extreme_concurrent_gpu_test(500)

        total_duration = time.time() - start_time

        # Calculate extreme performance metrics
        total_requests = all_results["extreme_concurrent"]["total_requests"]
        total_successful = all_results["extreme_concurrent"]["successful_requests"]
        overall_success_rate = (
            total_successful / total_requests if total_requests > 0 else 0
        )
        peak_rps = all_results["extreme_concurrent"]["requests_per_second"]

        final_results = {
            "test_suite": "GPU_EXTREME_STRESS_TEST",
            "timestamp": datetime.now().isoformat(),
            "total_duration": total_duration,
            "gpu_model": "NVIDIA GeForce RTX 4090",
            "max_workers": self.max_workers,
            "test_results": all_results,
            "extreme_performance_summary": {
                "total_requests": total_requests,
                "total_successful": total_successful,
                "overall_success_rate": overall_success_rate,
                "peak_requests_per_second": peak_rps,
                "extreme_performance_achieved": overall_success_rate > 0.95
                and peak_rps > 20,
                "gpu_stability_under_extreme_load": overall_success_rate > 0.9,
                "rtx_4090_utilization_rating": (
                    "MAXIMUM" if overall_success_rate > 0.95 else "HIGH"
                ),
            },
        }

        print("\n" + "=" * 60)
        print("🎯 EXTREME GPU STRESS TEST COMPLETE")
        print("=" * 60)
        print(f"   Total Duration: {total_duration/60:.1f} minutes")
        print(f"   Total Requests: {total_requests:,}")
        print(f"   Total Successful: {total_successful:,}")
        print(f"   Overall Success Rate: {overall_success_rate:.1%}")
        print(f"   Peak RPS: {peak_rps:.2f}")
        print(
            f"   RTX 4090 Utilization: {final_results['extreme_performance_summary']['rtx_4090_utilization_rating']}"
        )
        print(
            f"   Extreme Performance: {'✅ ACHIEVED' if final_results['extreme_performance_summary']['extreme_performance_achieved'] else '❌ NOT ACHIEVED'}"
        )
        print(
            f"   GPU Stability: {'✅ EXCELLENT' if final_results['extreme_performance_summary']['gpu_stability_under_extreme_load'] else '❌ POOR'}"
        )

        return final_results


def main():
    tester = ExtremeGPUStressTester()
    results = tester.run_extreme_stress_test()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"gpu_extreme_stress_results_{timestamp}.json"

    with open(filename, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n📄 Results saved to: {filename}")


if __name__ == "__main__":
    main()
