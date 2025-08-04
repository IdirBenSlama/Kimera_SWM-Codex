#!/usr/bin/env python3
"""
Kimera SWM Real-World Performance Test Suite
===========================================
This script performs comprehensive performance testing on a running Kimera server
to push it to its peak and gather actual real-world metrics.

No mocks, no simulations - only actual API calls and real measurements.
"""

import asyncio
import concurrent.futures
import json
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import aiohttp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil

# Configuration
KIMERA_BASE_URL = "http://localhost:8001"
RESULTS_DIR = Path("performance_results")
RESULTS_DIR.mkdir(exist_ok=True)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""

    endpoint: str
    method: str
    response_time_ms: float
    status_code: int
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    error: str = None
    response_size_bytes: int = 0


class KimeraPerformanceTester:
    def __init__(self, base_url: str = KIMERA_BASE_URL):
        self.base_url = base_url
        self.metrics: List[PerformanceMetrics] = []
        self.system_metrics = []

    async def measure_request(
        self,
        session: aiohttp.ClientSession,
        method: str,
        endpoint: str,
        data: Dict = None,
    ) -> PerformanceMetrics:
        """Make a request and measure performance"""
        url = f"{self.base_url}{endpoint}"

        # Capture system state before request
        cpu_before = psutil.cpu_percent(interval=0.1)
        mem_before = psutil.virtual_memory().percent

        start_time = time.perf_counter()

        try:
            async with session.request(method, url, json=data) as response:
                response_data = await response.read()
                end_time = time.perf_counter()

                # Calculate metrics
                response_time_ms = (end_time - start_time) * 1000

                return PerformanceMetrics(
                    endpoint=endpoint,
                    method=method,
                    response_time_ms=response_time_ms,
                    status_code=response.status,
                    timestamp=datetime.now(),
                    cpu_percent=psutil.cpu_percent(interval=0.1),
                    memory_percent=psutil.virtual_memory().percent,
                    response_size_bytes=len(response_data),
                )

        except Exception as e:
            end_time = time.perf_counter()
            return PerformanceMetrics(
                endpoint=endpoint,
                method=method,
                response_time_ms=(end_time - start_time) * 1000,
                status_code=0,
                timestamp=datetime.now(),
                cpu_percent=psutil.cpu_percent(interval=0.1),
                memory_percent=psutil.virtual_memory().percent,
                error=str(e),
            )

    async def test_endpoint_load(
        self,
        endpoint: str,
        method: str = "GET",
        data: Dict = None,
        concurrent_requests: int = 10,
        total_requests: int = 100,
    ) -> Dict[str, Any]:
        """Test an endpoint under load"""
        print(
            f"\n🔥 Testing {method} {endpoint} with {concurrent_requests} concurrent requests..."
        )

        async with aiohttp.ClientSession() as session:
            tasks = []

            for i in range(total_requests):
                if i > 0 and i % concurrent_requests == 0:
                    # Wait for batch to complete
                    batch_results = await asyncio.gather(*tasks)
                    self.metrics.extend(batch_results)
                    tasks = []

                task = self.measure_request(session, method, endpoint, data)
                tasks.append(task)

            # Complete remaining tasks
            if tasks:
                batch_results = await asyncio.gather(*tasks)
                self.metrics.extend(batch_results)

        # Calculate statistics for this endpoint
        endpoint_metrics = [
            m for m in self.metrics if m.endpoint == endpoint and m.method == method
        ]
        response_times = [
            m.response_time_ms for m in endpoint_metrics if m.status_code != 0
        ]

        if response_times:
            stats = {
                "endpoint": endpoint,
                "method": method,
                "total_requests": total_requests,
                "successful_requests": len(response_times),
                "failed_requests": total_requests - len(response_times),
                "avg_response_time_ms": statistics.mean(response_times),
                "min_response_time_ms": min(response_times),
                "max_response_time_ms": max(response_times),
                "median_response_time_ms": statistics.median(response_times),
                "p95_response_time_ms": np.percentile(response_times, 95),
                "p99_response_time_ms": np.percentile(response_times, 99),
                "requests_per_second": len(response_times)
                / (sum(response_times) / 1000),
            }
        else:
            stats = {
                "endpoint": endpoint,
                "method": method,
                "total_requests": total_requests,
                "successful_requests": 0,
                "failed_requests": total_requests,
                "error": "All requests failed",
            }

        return stats

    async def capture_system_metrics(
        self, duration_seconds: int = 60, interval: float = 1.0
    ):
        """Capture system metrics during test"""
        print(f"\n📊 Capturing system metrics for {duration_seconds} seconds...")

        start_time = time.time()
        while time.time() - start_time < duration_seconds:
            # Get current system metrics
            metrics = {
                "timestamp": datetime.now(),
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "memory_percent": psutil.virtual_memory().percent,
                "memory_available_mb": psutil.virtual_memory().available
                / (1024 * 1024),
                "disk_io_read_mb": psutil.disk_io_counters().read_bytes / (1024 * 1024),
                "disk_io_write_mb": psutil.disk_io_counters().write_bytes
                / (1024 * 1024),
                "network_sent_mb": psutil.net_io_counters().bytes_sent / (1024 * 1024),
                "network_recv_mb": psutil.net_io_counters().bytes_recv / (1024 * 1024),
            }

            # Try to get Kimera-specific metrics
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{self.base_url}/system-metrics/"
                    ) as response:
                        if response.status == 200:
                            kimera_metrics = await response.json()
                            metrics["kimera_metrics"] = kimera_metrics
            except aiohttp.ClientError as e:
                # This can fail if the server is under heavy load, which is fine.
                pass

            self.system_metrics.append(metrics)
            await asyncio.sleep(interval)

    async def run_comprehensive_test(self):
        """Run comprehensive performance test suite"""
        print("🚀 Starting Kimera SWM Comprehensive Performance Test")
        print("=" * 60)

        # Define test scenarios
        test_scenarios = [
            # Basic endpoints
            {"endpoint": "/", "method": "GET", "concurrent": 50, "total": 500},
            {"endpoint": "/health", "method": "GET", "concurrent": 100, "total": 1000},
            {
                "endpoint": "/system-metrics/",
                "method": "GET",
                "concurrent": 50,
                "total": 500,
            },
            # Core functionality tests
            {
                "endpoint": "/kimera/system/status",
                "method": "GET",
                "concurrent": 20,
                "total": 200,
            },
            {
                "endpoint": "/kimera/system/cycle",
                "method": "POST",
                "concurrent": 10,
                "total": 100,
            },
            # Heavy load tests - Geoid creation
            {
                "endpoint": "/kimera/geoids",
                "method": "POST",
                "data": {
                    "content": "Performance test geoid content with substantial text to simulate real usage. "
                    * 10,
                    "metadata": {"test": True, "timestamp": str(datetime.now())},
                },
                "concurrent": 20,
                "total": 200,
            },
            # SCAR creation stress test
            {
                "endpoint": "/kimera/scars",
                "method": "POST",
                "data": {
                    "geoid_id": "test_geoid_123",
                    "scar_type": "performance_test",
                    "content": "SCAR content for performance testing",
                },
                "concurrent": 30,
                "total": 300,
            },
            # Embedding generation load test
            {
                "endpoint": "/kimera/embed",
                "method": "POST",
                "data": {"text": "This is a test text for embedding generation. " * 20},
                "concurrent": 25,
                "total": 250,
            },
            # Thermodynamic analysis
            {
                "endpoint": "/kimera/thermodynamics/analyze",
                "method": "POST",
                "data": {
                    "geoid_ids": [f"geoid_{i}" for i in range(10)],
                    "analysis_type": "temperature",
                },
                "concurrent": 15,
                "total": 150,
            },
        ]

        # Start system metrics capture in background
        metrics_task = asyncio.create_task(
            self.capture_system_metrics(duration_seconds=300)
        )

        # Run all test scenarios
        all_results = []
        for scenario in test_scenarios:
            result = await self.test_endpoint_load(
                endpoint=scenario["endpoint"],
                method=scenario["method"],
                data=scenario.get("data"),
                concurrent_requests=scenario["concurrent"],
                total_requests=scenario["total"],
            )
            all_results.append(result)

            # Brief pause between tests
            await asyncio.sleep(2)

        # Stop metrics capture
        metrics_task.cancel()
        try:
            await metrics_task
        except asyncio.CancelledError:
            pass

        return all_results

    def generate_report(self, test_results: List[Dict[str, Any]]):
        """Generate comprehensive performance report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = RESULTS_DIR / f"kimera_performance_report_{timestamp}.json"

        # Prepare report data
        report = {
            "test_timestamp": timestamp,
            "kimera_base_url": self.base_url,
            "test_results": test_results,
            "system_metrics_summary": self._summarize_system_metrics(),
            "detailed_metrics": [
                asdict(m) for m in self.metrics[-1000:]
            ],  # Last 1000 requests
            "recommendations": self._generate_recommendations(test_results),
        }

        # Save JSON report
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)

        # Generate visualizations
        self._generate_visualizations(timestamp)

        # Print summary
        self._print_summary(test_results)

        return report_file

    def _summarize_system_metrics(self) -> Dict[str, Any]:
        """Summarize system metrics collected during test"""
        if not self.system_metrics:
            return {}

        cpu_values = [m["cpu_percent"] for m in self.system_metrics]
        memory_values = [m["memory_percent"] for m in self.system_metrics]

        return {
            "cpu": {
                "avg_percent": statistics.mean(cpu_values),
                "max_percent": max(cpu_values),
                "min_percent": min(cpu_values),
            },
            "memory": {
                "avg_percent": statistics.mean(memory_values),
                "max_percent": max(memory_values),
                "min_percent": min(memory_values),
            },
            "samples_collected": len(self.system_metrics),
        }

    def _generate_recommendations(
        self, test_results: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate performance recommendations based on results"""
        recommendations = []

        for result in test_results:
            if "avg_response_time_ms" in result:
                if result["avg_response_time_ms"] > 1000:
                    recommendations.append(
                        f"⚠️ {result['endpoint']} has high average response time "
                        f"({result['avg_response_time_ms']:.0f}ms). Consider optimization."
                    )

                if (
                    result.get("failed_requests", 0)
                    > result.get("total_requests", 1) * 0.05
                ):
                    recommendations.append(
                        f"⚠️ {result['endpoint']} has high failure rate "
                        f"({result['failed_requests']}/{result['total_requests']}). "
                        "Check server capacity and error handling."
                    )

        # System-level recommendations
        system_summary = self._summarize_system_metrics()
        if system_summary.get("cpu", {}).get("max_percent", 0) > 90:
            recommendations.append(
                "⚠️ CPU usage peaked above 90%. Consider scaling or optimization."
            )

        if system_summary.get("memory", {}).get("max_percent", 0) > 85:
            recommendations.append(
                "⚠️ Memory usage peaked above 85%. Monitor for memory leaks."
            )

        return recommendations

    def _generate_visualizations(self, timestamp: str):
        """Generate performance visualization charts"""
        if not self.metrics:
            return

        # Create DataFrame for easier plotting
        df = pd.DataFrame([asdict(m) for m in self.metrics])
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Kimera Performance Test Results - {timestamp}", fontsize=16)

        # 1. Response time distribution by endpoint
        ax1 = axes[0, 0]
        endpoint_groups = df.groupby("endpoint")["response_time_ms"]
        endpoint_stats = endpoint_groups.agg(["mean", "std"]).sort_values(
            "mean", ascending=False
        )
        endpoint_stats["mean"].plot(kind="barh", ax=ax1, xerr=endpoint_stats["std"])
        ax1.set_xlabel("Response Time (ms)")
        ax1.set_title("Average Response Time by Endpoint")

        # 2. Response time over time
        ax2 = axes[0, 1]
        df_sorted = df.sort_values("timestamp")
        df_sorted.set_index("timestamp")["response_time_ms"].rolling(
            window=10
        ).mean().plot(ax=ax2)
        ax2.set_ylabel("Response Time (ms)")
        ax2.set_title("Response Time Trend (10-sample rolling average)")

        # 3. System metrics over time
        ax3 = axes[1, 0]
        if self.system_metrics:
            sys_df = pd.DataFrame(self.system_metrics)
            sys_df["timestamp"] = pd.to_datetime(sys_df["timestamp"])
            sys_df.set_index("timestamp")[["cpu_percent", "memory_percent"]].plot(
                ax=ax3
            )
            ax3.set_ylabel("Usage %")
            ax3.set_title("System Resource Usage")
            ax3.legend(["CPU %", "Memory %"])

        # 4. Success rate by endpoint
        ax4 = axes[1, 1]
        success_rates = []
        endpoints = []
        for endpoint in df["endpoint"].unique():
            endpoint_data = df[df["endpoint"] == endpoint]
            success_rate = (endpoint_data["status_code"] != 0).mean() * 100
            success_rates.append(success_rate)
            endpoints.append(endpoint)

        ax4.bar(range(len(endpoints)), success_rates)
        ax4.set_xticks(range(len(endpoints)))
        ax4.set_xticklabels(endpoints, rotation=45, ha="right")
        ax4.set_ylabel("Success Rate %")
        ax4.set_title("Request Success Rate by Endpoint")
        ax4.axhline(y=95, color="r", linestyle="--", label="95% target")

        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"kimera_performance_charts_{timestamp}.png", dpi=300)
        plt.close()

    def _print_summary(self, test_results: List[Dict[str, Any]]):
        """Print test summary to console"""
        print("\n" + "=" * 60)
        print("📊 KIMERA PERFORMANCE TEST SUMMARY")
        print("=" * 60)

        for result in test_results:
            if "avg_response_time_ms" in result:
                print(f"\n📍 {result['method']} {result['endpoint']}")
                print(f"   Total Requests: {result['total_requests']}")
                print(
                    f"   Successful: {result['successful_requests']} | Failed: {result['failed_requests']}"
                )
                print(f"   Avg Response Time: {result['avg_response_time_ms']:.2f}ms")
                print(f"   P95 Response Time: {result['p95_response_time_ms']:.2f}ms")
                print(f"   P99 Response Time: {result['p99_response_time_ms']:.2f}ms")
                print(f"   Requests/Second: {result['requests_per_second']:.2f}")

        # System metrics summary
        system_summary = self._summarize_system_metrics()
        if system_summary:
            print(f"\n💻 SYSTEM METRICS")
            print(
                f"   CPU Usage - Avg: {system_summary['cpu']['avg_percent']:.1f}% | "
                f"Max: {system_summary['cpu']['max_percent']:.1f}%"
            )
            print(
                f"   Memory Usage - Avg: {system_summary['memory']['avg_percent']:.1f}% | "
                f"Max: {system_summary['memory']['max_percent']:.1f}%"
            )

        # Recommendations
        recommendations = self._generate_recommendations(test_results)
        if recommendations:
            print(f"\n💡 RECOMMENDATIONS")
            for rec in recommendations:
                print(f"   {rec}")

        print("\n" + "=" * 60)


async def main():
    """Main test execution"""
    tester = KimeraPerformanceTester()

    print("🔍 Checking Kimera server availability...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{KIMERA_BASE_URL}/health") as response:
                if response.status == 200:
                    print("✅ Kimera server is running and healthy")
                else:
                    print(f"⚠️ Kimera server returned status {response.status}")
    except Exception as e:
        print(f"❌ Cannot connect to Kimera server: {e}")
        return

    # Run comprehensive test
    test_results = await tester.run_comprehensive_test()

    # Generate report
    report_file = tester.generate_report(test_results)
    print(f"\n✅ Performance report saved to: {report_file}")

    # Additional peak load test
    print("\n🔥 Running PEAK LOAD TEST...")
    peak_results = await tester.test_endpoint_load(
        endpoint="/kimera/geoids",
        method="POST",
        data={
            "content": "Peak load test content " * 50,
            "metadata": {"peak_test": True},
        },
        concurrent_requests=100,
        total_requests=1000,
    )

    print(f"\n🎯 Peak Load Test Results:")
    print(f"   Endpoint: {peak_results['endpoint']}")
    print(f"   Concurrent Requests: 100")
    print(f"   Total Requests: {peak_results['total_requests']}")
    print(
        f"   Success Rate: {(peak_results['successful_requests']/peak_results['total_requests']*100):.1f}%"
    )
    print(
        f"   Avg Response Time: {peak_results.get('avg_response_time_ms', 'N/A'):.2f}ms"
    )
    print(
        f"   Max Response Time: {peak_results.get('max_response_time_ms', 'N/A'):.2f}ms"
    )


if __name__ == "__main__":
    asyncio.run(main())
