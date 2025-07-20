"""
Controlled Performance Crash Test with Full Monitoring
=====================================================

This script performs a controlled crash test that gradually increases load
while monitoring system performance metrics.
"""

import asyncio
import httpx
import psutil
import time
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass, asdict

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.utils.kimera_logger import get_system_logger

logger = get_system_logger(__name__)

@dataclass
class TestMetrics:
    """Metrics collected during the test"""
    timestamp: float
    concurrent_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time: float
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    active_connections: int
    error_types: Dict[str, int]

class ControlledCrashTest:
    """Controlled crash test with comprehensive monitoring"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.metrics_history: List[TestMetrics] = []
        self.start_time = time.time()
        self.test_id = f"crash_test_{int(self.start_time)}"
        
        # Test configuration - more conservative
        self.request_batches = [5, 10, 15, 20]  # Gradual escalation
        self.requests_per_batch = 50  # Reduced total requests
        self.delay_between_batches = 5  # seconds
        
        logger.info(f"🚀 Initializing controlled crash test: {self.test_id}")
    
    async def create_test_geoid(self, session: httpx.AsyncClient, geoid_id: int) -> Dict[str, Any]:
        """Create a single test geoid"""
        try:
            start_time = time.time()
            
            response = await session.post(
                f"{self.base_url}/api/cognitive-field/geoids",
                json={
                    "name": f"crash_test_geoid_{geoid_id}",
                    "coordinates": [float(geoid_id % 180 - 90), float(geoid_id % 360 - 180)],
                    "metadata": {
                        "test_type": "controlled_crash_test",
                        "test_id": self.test_id
                    }
                },
                timeout=30.0
            )
            
            response_time = time.time() - start_time
            
            if response.status_code in [200, 201]:
                return {
                    "success": True,
                    "response_time": response_time,
                    "status_code": response.status_code,
                    "geoid_id": geoid_id
                }
            else:
                return {
                    "success": False,
                    "response_time": response_time,
                    "status_code": response.status_code,
                    "error": f"HTTP {response.status_code}",
                    "geoid_id": geoid_id
                }
                
        except Exception as e:
            return {
                "success": False,
                "response_time": 0,
                "error": str(e),
                "geoid_id": geoid_id
            }
    
    def collect_system_metrics(self, concurrent_requests: int, results: List[Dict]) -> TestMetrics:
        """Collect comprehensive system metrics"""
        
        # Calculate request metrics
        successful = sum(1 for r in results if r.get("success", False))
        failed = len(results) - successful
        avg_response_time = sum(r.get("response_time", 0) for r in results) / len(results) if results else 0
        
        # Collect error types
        error_types = {}
        for result in results:
            if not result.get("success", False):
                error = result.get("error", "Unknown")
                error_types[error] = error_types.get(error, 0) + 1
        
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # Network connections
        try:
            connections = len([c for c in psutil.net_connections() if c.laddr and c.laddr.port == 8000])
        except:
            connections = 0
        
        return TestMetrics(
            timestamp=time.time(),
            concurrent_requests=concurrent_requests,
            successful_requests=successful,
            failed_requests=failed,
            avg_response_time=avg_response_time,
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_mb=memory.used // (1024 * 1024),
            active_connections=connections,
            error_types=error_types
        )
    
    async def run_batch_test(self, concurrent_requests: int, total_requests: int) -> List[Dict]:
        """Run a batch of requests with specified concurrency"""
        
        logger.info(f"🔥 Starting batch: {concurrent_requests} concurrent requests, {total_requests} total")
        
        results = []
        semaphore = asyncio.Semaphore(concurrent_requests)
        
        async def limited_request(session, geoid_id):
            async with semaphore:
                return await self.create_test_geoid(session, geoid_id)
        
        async with httpx.AsyncClient() as session:
            tasks = [
                limited_request(session, i) 
                for i in range(total_requests)
            ]
            
            # Execute with progress tracking
            batch_start = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions
            valid_results = []
            for result in results:
                if isinstance(result, dict):
                    valid_results.append(result)
                else:
                    valid_results.append({
                        "success": False,
                        "response_time": 0,
                        "error": str(result),
                        "geoid_id": -1
                    })
        
        batch_duration = time.time() - batch_start
        success_rate = sum(1 for r in valid_results if r.get("success", False)) / len(valid_results) * 100 if valid_results else 0
        
        logger.info(f"✅ Batch completed in {batch_duration:.2f}s - Success rate: {success_rate:.1f}%")
        
        return valid_results
    
    async def check_system_health(self) -> Dict[str, Any]:
        """Check system health endpoints"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Check basic health
                health_response = await client.get(f"{self.base_url}/health")
                
                return {
                    "health_status": health_response.status_code,
                    "system_responsive": True
                }
        except Exception as e:
            return {
                "health_status": None,
                "system_responsive": False,
                "error": str(e)
            }
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        
        if not self.metrics_history:
            return {"error": "No metrics collected"}
        
        total_requests = sum(m.successful_requests + m.failed_requests for m in self.metrics_history)
        total_successful = sum(m.successful_requests for m in self.metrics_history)
        total_failed = sum(m.failed_requests for m in self.metrics_history)
        
        # Performance metrics
        avg_response_times = [m.avg_response_time for m in self.metrics_history if m.avg_response_time > 0]
        peak_cpu = max(m.cpu_percent for m in self.metrics_history)
        peak_memory = max(m.memory_percent for m in self.metrics_history)
        peak_connections = max(m.active_connections for m in self.metrics_history)
        
        # Error analysis
        all_errors = {}
        for metrics in self.metrics_history:
            for error, count in metrics.error_types.items():
                all_errors[error] = all_errors.get(error, 0) + count
        
        return {
            "test_id": self.test_id,
            "duration_seconds": time.time() - self.start_time,
            "summary": {
                "total_requests": total_requests,
                "successful_requests": total_successful,
                "failed_requests": total_failed,
                "success_rate_percent": (total_successful / total_requests * 100) if total_requests > 0 else 0,
                "average_response_time": sum(avg_response_times) / len(avg_response_times) if avg_response_times else 0,
                "peak_cpu_percent": peak_cpu,
                "peak_memory_percent": peak_memory,
                "peak_connections": peak_connections
            },
            "error_analysis": all_errors,
            "metrics_timeline": [asdict(m) for m in self.metrics_history]
        }
    
    async def run_controlled_crash_test(self):
        """Execute the full controlled crash test"""
        
        logger.info("🎯 Starting Controlled Performance Crash Test")
        logger.info("=" * 60)
        
        try:
            # Initial system health check
            logger.info("🔍 Performing initial system health check...")
            initial_health = await self.check_system_health()
            
            if not initial_health.get("system_responsive", False):
                logger.error("❌ System is not responsive. Aborting test.")
                return
            
            logger.info("✅ System is responsive. Proceeding with crash test.")
            
            # Execute batches with increasing load
            for batch_num, concurrent_requests in enumerate(self.request_batches, 1):
                logger.info(f"\n🚀 BATCH {batch_num}/{len(self.request_batches)}")
                logger.info(f"📊 Concurrent requests: {concurrent_requests}")
                logger.info(f"📊 Total requests in batch: {self.requests_per_batch}")
                
                # Run the batch
                batch_results = await self.run_batch_test(concurrent_requests, self.requests_per_batch)
                
                # Collect metrics
                metrics = self.collect_system_metrics(concurrent_requests, batch_results)
                self.metrics_history.append(metrics)
                
                # Log batch summary
                success_rate = (metrics.successful_requests/self.requests_per_batch*100) if self.requests_per_batch > 0 else 0
                logger.info(f"📈 Batch {batch_num} Results:")
                logger.info(f"   Success: {metrics.successful_requests}/{self.requests_per_batch} ({success_rate:.1f}%)")
                logger.info(f"   Avg Response Time: {metrics.avg_response_time:.3f}s")
                logger.info(f"   CPU Usage: {metrics.cpu_percent:.1f}%")
                logger.info(f"   Memory Usage: {metrics.memory_percent:.1f}% ({metrics.memory_mb} MB)")
                logger.info(f"   Active Connections: {metrics.active_connections}")
                
                if metrics.error_types:
                    logger.warning(f"   Errors: {metrics.error_types}")
                
                # Check if system is still responsive
                health_check = await self.check_system_health()
                if not health_check.get("system_responsive", False):
                    logger.error(f"❌ System became unresponsive after batch {batch_num}. Stopping test.")
                    break
                
                # Wait between batches (except for the last one)
                if batch_num < len(self.request_batches):
                    logger.info(f"⏳ Waiting {self.delay_between_batches}s before next batch...")
                    await asyncio.sleep(self.delay_between_batches)
            
            # Generate and save report
            report = self.generate_report()
            report["initial_health"] = initial_health
            
            # Save detailed report
            report_filename = f"test_results/controlled_crash_test_{self.test_id}.json"
            os.makedirs("test_results", exist_ok=True)
            
            with open(report_filename, 'w') as f:
                json.dump(report, f, indent=2)
            
            # Print summary
            logger.info("\n" + "=" * 60)
            logger.info("📊 CONTROLLED CRASH TEST COMPLETED")
            logger.info("=" * 60)
            logger.info(f"📁 Detailed report saved: {report_filename}")
            logger.info(f"⏱️  Total duration: {report['duration_seconds']:.2f} seconds")
            logger.info(f"📈 Total requests: {report['summary']['total_requests']}")
            logger.info(f"✅ Success rate: {report['summary']['success_rate_percent']:.2f}%")
            logger.info(f"⚡ Average response time: {report['summary']['average_response_time']:.3f}s")
            logger.info(f"🖥️  Peak CPU usage: {report['summary']['peak_cpu_percent']:.1f}%")
            logger.info(f"💾 Peak memory usage: {report['summary']['peak_memory_percent']:.1f}%")
            logger.info(f"🔗 Peak connections: {report['summary']['peak_connections']}")
            
            if report["error_analysis"]:
                logger.warning("⚠️  Error summary:")
                for error, count in report["error_analysis"].items():
                    logger.warning(f"   {error}: {count} occurrences")
            
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"❌ Crash test failed with exception: {e}")
            logger.exception("Full traceback:")

async def main():
    """Main execution function"""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create and run the crash test
    crash_test = ControlledCrashTest()
    await crash_test.run_controlled_crash_test()

if __name__ == "__main__":
    asyncio.run(main()) 