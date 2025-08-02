"""
Kimera SWM Production Load Testing Suite
=======================================
Comprehensive load testing framework for production readiness validation.
"""

import asyncio
import aiohttp
import time
import psutil
import json
import logging
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

try:
    import pynvml
    NVML_AVAILABLE = True
    pynvml.nvmlInit()
except ImportError:
    NVML_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LoadTestConfig:
    """Load test configuration"""
    name: str
    duration_seconds: int
    concurrent_users: int
    requests_per_second: int
    ramp_up_seconds: int
    endpoints: List[Dict[str, Any]]
    expected_response_time_ms: float
    expected_success_rate: float


@dataclass
class TestResult:
    """Individual test result"""
    timestamp: float
    endpoint: str
    response_time_ms: float
    status_code: int
    success: bool
    payload_size: int
    error: Optional[str] = None


@dataclass
class SystemMetrics:
    """System resource metrics"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    gpu_utilization: float
    gpu_memory_percent: float
    gpu_temperature: float
    disk_io_read: float
    disk_io_write: float
    network_sent: float
    network_recv: float


class ProductionLoadTester:
    """Production-grade load testing framework"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url
        self.results: List[TestResult] = []
        self.metrics: List[SystemMetrics] = []
        self.start_time = time.time()
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Initialize GPU monitoring if available
        self.gpu_handle = None
        if NVML_AVAILABLE:
            try:
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except:
                pass
                
        # Test configurations
        self.test_configs = self._create_test_configs()
        
    def _create_test_configs(self) -> List[LoadTestConfig]:
        """Create comprehensive test configurations"""
        
        # Common endpoints to test
        endpoints = [
            {"method": "GET", "path": "/health", "weight": 0.2},
            {"method": "GET", "path": "/api/v1/system/status", "weight": 0.15},
            {"method": "POST", "path": "/api/v1/linguistic/analyze", 
             "payload": {"text": "Analyze this complex sentence structure.", "level": "enhanced"}, "weight": 0.25},
            {"method": "POST", "path": "/api/v1/cognitive/process", 
             "payload": {"input": "What is the meaning of consciousness?", "depth": "deep"}, "weight": 0.2},
            {"method": "POST", "path": "/api/v1/cognitive/quantum/explore",
             "payload": {"query": "Quantum superposition in cognitive states", "qubits": 10}, "weight": 0.1},
            {"method": "POST", "path": "/api/v1/contradiction/detect",
             "payload": {"statements": ["AI is beneficial", "AI poses risks"], "context": "technology"}, "weight": 0.1}
        ]
        
        return [
            # 1. Baseline Test - Normal Load
            LoadTestConfig(
                name="baseline_normal_load",
                duration_seconds=300,  # 5 minutes
                concurrent_users=10,
                requests_per_second=50,
                ramp_up_seconds=30,
                endpoints=endpoints,
                expected_response_time_ms=15.0,
                expected_success_rate=0.99
            ),
            
            # 2. Stress Test - High Load
            LoadTestConfig(
                name="stress_high_load", 
                duration_seconds=600,  # 10 minutes
                concurrent_users=50,
                requests_per_second=200,
                ramp_up_seconds=60,
                endpoints=endpoints,
                expected_response_time_ms=50.0,
                expected_success_rate=0.95
            ),
            
            # 3. Spike Test - Sudden Load Increase
            LoadTestConfig(
                name="spike_test",
                duration_seconds=300,  # 5 minutes
                concurrent_users=100,
                requests_per_second=500,
                ramp_up_seconds=10,  # Quick ramp-up
                endpoints=endpoints,
                expected_response_time_ms=100.0,
                expected_success_rate=0.90
            ),
            
            # 4. Endurance Test - Sustained Load
            LoadTestConfig(
                name="endurance_sustained",
                duration_seconds=1800,  # 30 minutes
                concurrent_users=25,
                requests_per_second=100,
                ramp_up_seconds=120,
                endpoints=endpoints,
                expected_response_time_ms=25.0,
                expected_success_rate=0.98
            ),
            
            # 5. Breaking Point Test - Find Limits
            LoadTestConfig(
                name="breaking_point",
                duration_seconds=900,  # 15 minutes
                concurrent_users=200,
                requests_per_second=1000,
                ramp_up_seconds=180,
                endpoints=endpoints,
                expected_response_time_ms=200.0,
                expected_success_rate=0.80
            ),
            
            # 6. Cognitive Heavy Test - Complex Workloads
            LoadTestConfig(
                name="cognitive_heavy",
                duration_seconds=600,  # 10 minutes
                concurrent_users=20,
                requests_per_second=30,
                ramp_up_seconds=60,
                endpoints=[ep for ep in endpoints if ep.get("payload")],  # Only complex endpoints
                expected_response_time_ms=200.0,
                expected_success_rate=0.95
            )
        ]
        
    async def _collect_system_metrics(self):
        """Collect system performance metrics"""
        while True:
            try:
                # CPU and Memory
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                # Disk I/O
                disk_io = psutil.disk_io_counters()
                disk_read = disk_io.read_bytes if disk_io else 0
                disk_write = disk_io.write_bytes if disk_io else 0
                
                # Network I/O
                net_io = psutil.net_io_counters()
                net_sent = net_io.bytes_sent if net_io else 0
                net_recv = net_io.bytes_recv if net_io else 0
                
                # GPU metrics
                gpu_util = gpu_mem = gpu_temp = 0.0
                if self.gpu_handle:
                    try:
                        gpu_util_info = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                        gpu_mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                        gpu_util = gpu_util_info.gpu
                        gpu_mem = (gpu_mem_info.used / gpu_mem_info.total) * 100
                        gpu_temp = pynvml.nvmlDeviceGetTemperature(self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
                    except:
                        pass
                
                metric = SystemMetrics(
                    timestamp=time.time(),
                    cpu_percent=cpu_percent,
                    memory_percent=memory.percent,
                    memory_used_gb=memory.used / (1024**3),
                    gpu_utilization=gpu_util,
                    gpu_memory_percent=gpu_mem,
                    gpu_temperature=gpu_temp,
                    disk_io_read=disk_read,
                    disk_io_write=disk_write,
                    network_sent=net_sent,
                    network_recv=net_recv
                )
                
                self.metrics.append(metric)
                await asyncio.sleep(2)  # Collect every 2 seconds
                
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                await asyncio.sleep(5)
                
    async def _make_request(self, method: str, endpoint: str, payload: Optional[Dict] = None) -> TestResult:
        """Make individual HTTP request and measure performance"""
        start_time = time.time()
        
        try:
            url = f"{self.base_url}{endpoint}"
            
            if method.upper() == "GET":
                async with self.session.get(url) as response:
                    content = await response.read()
                    status_code = response.status
            else:  # POST
                async with self.session.post(url, json=payload) as response:
                    content = await response.read()
                    status_code = response.status
                    
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            success = 200 <= status_code < 400
            
            return TestResult(
                timestamp=start_time,
                endpoint=endpoint,
                response_time_ms=response_time,
                status_code=status_code,
                success=success,
                payload_size=len(content)
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return TestResult(
                timestamp=start_time,
                endpoint=endpoint,
                response_time_ms=response_time,
                status_code=0,
                success=False,
                payload_size=0,
                error=str(e)
            )
            
    async def _user_simulation(self, user_id: int, config: LoadTestConfig):
        """Simulate individual user behavior"""
        logger.info(f"User {user_id} starting simulation")
        
        # Calculate request interval
        requests_per_user = config.requests_per_second / config.concurrent_users
        request_interval = 1.0 / requests_per_user if requests_per_user > 0 else 1.0
        
        # Random delays to simulate realistic behavior
        np.random.seed(user_id)
        
        end_time = time.time() + config.duration_seconds
        
        while time.time() < end_time:
            # Select endpoint based on weights
            endpoint_choice = np.random.choice(
                config.endpoints, 
                p=[ep.get("weight", 1.0) for ep in config.endpoints]
            )
            
            # Make request
            result = await self._make_request(
                endpoint_choice["method"],
                endpoint_choice["path"],
                endpoint_choice.get("payload")
            )
            
            self.results.append(result)
            
            # Add realistic user think time (0.1-2.0 seconds)
            think_time = np.random.uniform(0.1, 2.0)
            await asyncio.sleep(max(request_interval - think_time, 0.1))
            
        logger.info(f"User {user_id} completed simulation")
        
    async def run_load_test(self, config: LoadTestConfig) -> Dict[str, Any]:
        """Execute a complete load test"""
        logger.info(f"🚀 Starting load test: {config.name}")
        logger.info(f"   Duration: {config.duration_seconds}s")
        logger.info(f"   Users: {config.concurrent_users}")
        logger.info(f"   Target RPS: {config.requests_per_second}")
        
        # Clear previous results
        self.results.clear()
        self.metrics.clear()
        
        # Create HTTP session with optimizations
        timeout = aiohttp.ClientTimeout(total=30)
        connector = aiohttp.TCPConnector(limit=1000, limit_per_host=100)
        self.session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        
        try:
            # Start metrics collection
            metrics_task = asyncio.create_task(self._collect_system_metrics())
            
            # Ramp up users gradually
            user_tasks = []
            users_per_batch = max(1, config.concurrent_users // 10)
            batches = config.concurrent_users // users_per_batch
            
            for batch in range(batches):
                batch_users = []
                start_user = batch * users_per_batch
                end_user = min(start_user + users_per_batch, config.concurrent_users)
                
                for user_id in range(start_user, end_user):
                    task = asyncio.create_task(self._user_simulation(user_id, config))
                    batch_users.append(task)
                    
                user_tasks.extend(batch_users)
                
                # Wait between batches for ramp-up
                if batch < batches - 1:
                    ramp_delay = config.ramp_up_seconds / batches
                    await asyncio.sleep(ramp_delay)
                    
            # Wait for all users to complete
            await asyncio.gather(*user_tasks)
            
            # Stop metrics collection
            metrics_task.cancel()
            
        finally:
            await self.session.close()
            
        # Analyze results
        analysis = self._analyze_results(config)
        
        logger.info(f"✅ Load test completed: {config.name}")
        logger.info(f"   Requests: {analysis['total_requests']}")
        logger.info(f"   Success Rate: {analysis['success_rate']:.2%}")
        logger.info(f"   Avg Response Time: {analysis['avg_response_time']:.1f}ms")
        logger.info(f"   95th Percentile: {analysis['p95_response_time']:.1f}ms")
        
        return analysis
        
    def _analyze_results(self, config: LoadTestConfig) -> Dict[str, Any]:
        """Analyze test results and generate comprehensive report"""
        
        if not self.results:
            return {"error": "No results to analyze"}
            
        # Basic statistics
        total_requests = len(self.results)
        successful_requests = sum(1 for r in self.results if r.success)
        success_rate = successful_requests / total_requests if total_requests > 0 else 0
        
        # Response time statistics
        response_times = [r.response_time_ms for r in self.results if r.success]
        if response_times:
            avg_response_time = statistics.mean(response_times)
            median_response_time = statistics.median(response_times)
            p95_response_time = np.percentile(response_times, 95)
            p99_response_time = np.percentile(response_times, 99)
            max_response_time = max(response_times)
            min_response_time = min(response_times)
        else:
            avg_response_time = median_response_time = p95_response_time = p99_response_time = 0
            max_response_time = min_response_time = 0
            
        # Throughput calculation
        test_duration = max(r.timestamp for r in self.results) - min(r.timestamp for r in self.results)
        actual_rps = total_requests / test_duration if test_duration > 0 else 0
        
        # Error analysis
        errors = defaultdict(int)
        for result in self.results:
            if not result.success:
                error_key = result.error or f"HTTP_{result.status_code}"
                errors[error_key] += 1
                
        # System resource analysis
        if self.metrics:
            avg_cpu = statistics.mean(m.cpu_percent for m in self.metrics)
            max_cpu = max(m.cpu_percent for m in self.metrics)
            avg_memory = statistics.mean(m.memory_percent for m in self.metrics)
            max_memory = max(m.memory_percent for m in self.metrics)
            avg_gpu = statistics.mean(m.gpu_utilization for m in self.metrics)
            max_gpu = max(m.gpu_utilization for m in self.metrics)
        else:
            avg_cpu = max_cpu = avg_memory = max_memory = avg_gpu = max_gpu = 0
            
        # Performance assessment
        response_time_ok = avg_response_time <= config.expected_response_time_ms
        success_rate_ok = success_rate >= config.expected_success_rate
        overall_pass = response_time_ok and success_rate_ok
        
        # Endpoint performance breakdown
        endpoint_stats = defaultdict(lambda: {"count": 0, "success": 0, "total_time": 0})
        for result in self.results:
            ep_stat = endpoint_stats[result.endpoint]
            ep_stat["count"] += 1
            if result.success:
                ep_stat["success"] += 1
                ep_stat["total_time"] += result.response_time_ms
                
        endpoint_performance = {}
        for endpoint, stats in endpoint_stats.items():
            if stats["count"] > 0:
                endpoint_performance[endpoint] = {
                    "requests": stats["count"],
                    "success_rate": stats["success"] / stats["count"],
                    "avg_response_time": stats["total_time"] / stats["success"] if stats["success"] > 0 else 0
                }
        
        return {
            "test_name": config.name,
            "timestamp": datetime.now().isoformat(),
            "configuration": asdict(config),
            "overall_pass": overall_pass,
            
            # Request statistics
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": total_requests - successful_requests,
            "success_rate": success_rate,
            "actual_rps": actual_rps,
            "test_duration_seconds": test_duration,
            
            # Response time statistics
            "avg_response_time": avg_response_time,
            "median_response_time": median_response_time,
            "min_response_time": min_response_time,
            "max_response_time": max_response_time,
            "p95_response_time": p95_response_time,
            "p99_response_time": p99_response_time,
            
            # Resource utilization
            "avg_cpu_percent": avg_cpu,
            "max_cpu_percent": max_cpu,
            "avg_memory_percent": avg_memory,
            "max_memory_percent": max_memory,
            "avg_gpu_utilization": avg_gpu,
            "max_gpu_utilization": max_gpu,
            
            # Error analysis
            "errors": dict(errors),
            "endpoint_performance": endpoint_performance,
            
            # Assessment
            "response_time_assessment": "PASS" if response_time_ok else "FAIL",
            "success_rate_assessment": "PASS" if success_rate_ok else "FAIL",
            
            # Bottleneck identification
            "bottlenecks": self._identify_bottlenecks(avg_cpu, max_cpu, avg_memory, max_memory, avg_gpu, max_gpu)
        }
        
    def _identify_bottlenecks(self, avg_cpu: float, max_cpu: float, avg_memory: float, 
                             max_memory: float, avg_gpu: float, max_gpu: float) -> List[str]:
        """Identify system bottlenecks from metrics"""
        bottlenecks = []
        
        if max_cpu > 90:
            bottlenecks.append(f"CPU_CRITICAL (max: {max_cpu:.1f}%)")
        elif avg_cpu > 70:
            bottlenecks.append(f"CPU_HIGH (avg: {avg_cpu:.1f}%)")
            
        if max_memory > 90:
            bottlenecks.append(f"MEMORY_CRITICAL (max: {max_memory:.1f}%)")
        elif avg_memory > 80:
            bottlenecks.append(f"MEMORY_HIGH (avg: {avg_memory:.1f}%)")
            
        if avg_gpu < 10 and max_gpu < 20:
            bottlenecks.append("GPU_UNDERUTILIZED (opportunity for acceleration)")
            
        return bottlenecks
        
    def save_results(self, analysis: Dict[str, Any], output_dir: str = "data/load_testing"):
        """Save test results to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_name = analysis["test_name"]
        
        # Save JSON results
        json_path = output_path / f"{test_name}_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(analysis, f, indent=2)
            
        # Save CSV of individual results
        csv_path = output_path / f"{test_name}_{timestamp}_raw.csv"
        with open(csv_path, 'w') as f:
            f.write("timestamp,endpoint,response_time_ms,status_code,success,payload_size,error\n")
            for result in self.results:
                f.write(f"{result.timestamp},{result.endpoint},{result.response_time_ms},"
                       f"{result.status_code},{result.success},{result.payload_size},"
                       f'"{result.error or ""}"\n')
                       
        # Save metrics CSV
        metrics_csv = output_path / f"{test_name}_{timestamp}_metrics.csv"
        with open(metrics_csv, 'w') as f:
            f.write("timestamp,cpu_percent,memory_percent,memory_used_gb,gpu_utilization,"
                   "gpu_memory_percent,gpu_temperature,disk_io_read,disk_io_write,"
                   "network_sent,network_recv\n")
            for metric in self.metrics:
                f.write(f"{metric.timestamp},{metric.cpu_percent},{metric.memory_percent},"
                       f"{metric.memory_used_gb},{metric.gpu_utilization},{metric.gpu_memory_percent},"
                       f"{metric.gpu_temperature},{metric.disk_io_read},{metric.disk_io_write},"
                       f"{metric.network_sent},{metric.network_recv}\n")
                       
        logger.info(f"📊 Results saved to {output_path}")
        return json_path
        
    def generate_visualization(self, analysis: Dict[str, Any], output_dir: str = "data/load_testing"):
        """Generate performance visualization charts"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_name = analysis["test_name"]
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Load Test Results: {test_name}', fontsize=16, fontweight='bold')
        
        # Response time over time
        if self.results:
            timestamps = [(r.timestamp - self.results[0].timestamp) for r in self.results if r.success]
            response_times = [r.response_time_ms for r in self.results if r.success]
            
            ax1.scatter(timestamps, response_times, alpha=0.6, s=1)
            ax1.set_xlabel('Time (seconds)')
            ax1.set_ylabel('Response Time (ms)')
            ax1.set_title('Response Time Distribution')
            ax1.grid(True, alpha=0.3)
            
        # System resource utilization
        if self.metrics:
            metric_timestamps = [(m.timestamp - self.metrics[0].timestamp) for m in self.metrics]
            cpu_usage = [m.cpu_percent for m in self.metrics]
            memory_usage = [m.memory_percent for m in self.metrics]
            gpu_usage = [m.gpu_utilization for m in self.metrics]
            
            ax2.plot(metric_timestamps, cpu_usage, label='CPU %', linewidth=2)
            ax2.plot(metric_timestamps, memory_usage, label='Memory %', linewidth=2)
            ax2.plot(metric_timestamps, gpu_usage, label='GPU %', linewidth=2)
            ax2.set_xlabel('Time (seconds)')
            ax2.set_ylabel('Utilization (%)')
            ax2.set_title('System Resource Utilization')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
        # Endpoint performance comparison
        if analysis.get("endpoint_performance"):
            endpoints = list(analysis["endpoint_performance"].keys())
            avg_times = [analysis["endpoint_performance"][ep]["avg_response_time"] for ep in endpoints]
            success_rates = [analysis["endpoint_performance"][ep]["success_rate"] * 100 for ep in endpoints]
            
            x_pos = np.arange(len(endpoints))
            ax3.bar(x_pos, avg_times, alpha=0.7)
            ax3.set_xlabel('Endpoints')
            ax3.set_ylabel('Avg Response Time (ms)')
            ax3.set_title('Endpoint Performance')
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels([ep.split('/')[-1] for ep in endpoints], rotation=45)
            ax3.grid(True, alpha=0.3)
            
        # Success rate and throughput summary
        summary_data = [
            analysis.get("success_rate", 0) * 100,
            min(analysis.get("actual_rps", 0) / 10, 100),  # Scale RPS to percentage
            100 - analysis.get("avg_cpu_percent", 0),  # CPU availability
            100 - analysis.get("avg_memory_percent", 0)  # Memory availability
        ]
        summary_labels = ['Success Rate %', 'Throughput (x10 RPS)', 'CPU Available %', 'Memory Available %']
        
        colors = ['green' if x > 80 else 'yellow' if x > 60 else 'red' for x in summary_data]
        ax4.bar(summary_labels, summary_data, color=colors, alpha=0.7)
        ax4.set_ylabel('Percentage')
        ax4.set_title('Performance Summary')
        ax4.set_ylim(0, 100)
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = output_path / f"{test_name}_{timestamp}_visualization.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📈 Visualization saved to {plot_path}")
        return plot_path
        
    async def run_full_test_suite(self) -> Dict[str, Any]:
        """Run the complete load testing suite"""
        logger.info("🏋️ Starting FULL PRODUCTION LOAD TEST SUITE")
        logger.info("=" * 80)
        
        suite_results = {
            "suite_start_time": datetime.now().isoformat(),
            "total_tests": len(self.test_configs),
            "test_results": [],
            "summary": {}
        }
        
        all_passed = True
        total_requests = 0
        
        for i, config in enumerate(self.test_configs, 1):
            logger.info(f"\n📋 Test {i}/{len(self.test_configs)}: {config.name}")
            logger.info("-" * 50)
            
            try:
                # Run the test
                result = await self.run_load_test(config)
                
                # Save results
                json_path = self.save_results(result)
                viz_path = self.generate_visualization(result)
                
                result["output_files"] = {
                    "json": str(json_path),
                    "visualization": str(viz_path)
                }
                
                suite_results["test_results"].append(result)
                total_requests += result["total_requests"]
                
                if not result["overall_pass"]:
                    all_passed = False
                    
                # Wait between tests to allow system recovery
                if i < len(self.test_configs):
                    logger.info("⏳ Waiting 30 seconds for system recovery...")
                    await asyncio.sleep(30)
                    
            except Exception as e:
                logger.error(f"❌ Test {config.name} failed: {e}")
                all_passed = False
                
        # Generate suite summary
        suite_results["suite_end_time"] = datetime.now().isoformat()
        suite_results["all_tests_passed"] = all_passed
        suite_results["total_requests_sent"] = total_requests
        
        # Calculate aggregated metrics
        if suite_results["test_results"]:
            avg_success_rate = statistics.mean(r["success_rate"] for r in suite_results["test_results"])
            avg_response_time = statistics.mean(r["avg_response_time"] for r in suite_results["test_results"])
            max_rps = max(r["actual_rps"] for r in suite_results["test_results"])
            
            suite_results["summary"] = {
                "overall_success_rate": avg_success_rate,
                "average_response_time_ms": avg_response_time,
                "maximum_rps_achieved": max_rps,
                "performance_grade": self._calculate_performance_grade(avg_success_rate, avg_response_time, max_rps)
            }
            
        # Save suite results
        suite_path = Path("data/load_testing") / f"load_test_suite_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(suite_path, 'w') as f:
            json.dump(suite_results, f, indent=2)
            
        logger.info("\n" + "=" * 80)
        logger.info("🏆 LOAD TEST SUITE COMPLETE!")
        logger.info(f"📊 Total Tests: {suite_results['total_tests']}")
        logger.info(f"📈 Total Requests: {total_requests:,}")
        logger.info(f"✅ All Passed: {all_passed}")
        logger.info(f"🎯 Performance Grade: {suite_results['summary'].get('performance_grade', 'N/A')}")
        logger.info(f"📄 Suite Results: {suite_path}")
        logger.info("=" * 80)
        
        return suite_results
        
    def _calculate_performance_grade(self, success_rate: float, avg_response_time: float, max_rps: float) -> str:
        """Calculate overall performance grade"""
        score = 0
        
        # Success rate scoring (40% weight)
        if success_rate >= 0.99:
            score += 40
        elif success_rate >= 0.95:
            score += 35
        elif success_rate >= 0.90:
            score += 25
        else:
            score += 10
            
        # Response time scoring (30% weight)
        if avg_response_time <= 20:
            score += 30
        elif avg_response_time <= 50:
            score += 25
        elif avg_response_time <= 100:
            score += 15
        else:
            score += 5
            
        # Throughput scoring (30% weight)
        if max_rps >= 500:
            score += 30
        elif max_rps >= 200:
            score += 25
        elif max_rps >= 100:
            score += 15
        else:
            score += 5
            
        # Convert to letter grade
        if score >= 90:
            return "A+ (Excellent)"
        elif score >= 80:
            return "A (Very Good)"
        elif score >= 70:
            return "B (Good)"
        elif score >= 60:
            return "C (Acceptable)"
        else:
            return "D (Needs Improvement)"


async def main():
    """Main execution function"""
    tester = ProductionLoadTester()
    
    # Check if Kimera is running
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://127.0.0.1:8000/health") as response:
                if response.status != 200:
                    logger.error("❌ Kimera not responding to health check")
                    return
    except Exception as e:
        logger.error(f"❌ Cannot connect to Kimera: {e}")
        logger.error("💡 Make sure Kimera is running: python scripts/start_kimera_optimized_v2.py")
        return
        
    # Run the full test suite
    results = await tester.run_full_test_suite()
    
    # Print final recommendations
    logger.info("\n🎯 NEXT OPTIMIZATION RECOMMENDATIONS:")
    
    max_rps = results["summary"].get("maximum_rps_achieved", 0)
    avg_response_time = results["summary"].get("average_response_time_ms", 0)
    
    if max_rps < 200:
        logger.info("🔧 PRIORITY: Increase throughput capacity")
        logger.info("   → Consider: Connection pooling, async optimizations")
        
    if avg_response_time > 50:
        logger.info("⚡ PRIORITY: Reduce response times")
        logger.info("   → Consider: GPU acceleration, caching")
        
    # Check for bottlenecks across all tests
    all_bottlenecks = []
    for test_result in results["test_results"]:
        all_bottlenecks.extend(test_result.get("bottlenecks", []))
        
    if "GPU_UNDERUTILIZED" in str(all_bottlenecks):
        logger.info("🎮 OPPORTUNITY: Enable GPU acceleration for major performance boost")
        
    if "CPU_HIGH" in str(all_bottlenecks) or "CPU_CRITICAL" in str(all_bottlenecks):
        logger.info("💻 CONSIDERATION: Horizontal scaling to distribute CPU load")
        
    logger.info("\n🚀 Ready for Phase 2: GPU Acceleration!")


if __name__ == "__main__":
    asyncio.run(main())