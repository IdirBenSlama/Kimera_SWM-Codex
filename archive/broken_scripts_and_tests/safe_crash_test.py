"""
Safe Performance Crash Test with Enhanced Monitoring
===================================================

This script performs a very conservative crash test with extensive monitoring
to identify the exact point where the system becomes unstable.
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
class SafeTestMetrics:
    """Metrics collected during the safe test"""
    timestamp: float
    test_phase: str
    concurrent_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time: float
    max_response_time: float
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    system_responsive: bool
    error_details: Dict[str, Any]

class SafeCrashTest:
    """Ultra-conservative crash test with extensive monitoring"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.metrics_history: List[SafeTestMetrics] = []
        self.start_time = time.time()
        self.test_id = f"safe_crash_test_{int(self.start_time)}"
        
        # Very conservative test configuration
        self.phases = [
            {"name": "warmup", "concurrent": 1, "total": 5, "timeout": 10},
            {"name": "light_load", "concurrent": 2, "total": 10, "timeout": 15},
            {"name": "moderate_load", "concurrent": 3, "total": 15, "timeout": 20},
            {"name": "stress_test", "concurrent": 5, "total": 20, "timeout": 30},
        ]
        
        self.delay_between_phases = 10  # seconds
        self.health_check_interval = 3  # seconds
        
        logger.info(f"🛡️ Initializing SAFE crash test: {self.test_id}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                start_time = time.time()
                response = await client.get(f"{self.base_url}/health")
                response_time = time.time() - start_time
                
                return {
                    "responsive": True,
                    "status_code": response.status_code,
                    "response_time": response_time,
                    "timestamp": time.time()
                }
        except Exception as e:
            return {
                "responsive": False,
                "error": str(e),
                "timestamp": time.time()
            }
    
    async def safe_request(self, session: httpx.AsyncClient, request_id: int, timeout: float) -> Dict[str, Any]:
        """Make a single safe request with comprehensive error handling"""
        try:
            start_time = time.time()
            
            response = await session.post(
                f"{self.base_url}/api/cognitive-field/geoids",
                json={
                    "name": f"safe_test_geoid_{request_id}_{self.test_id}",
                    "coordinates": [float(request_id % 90), float(request_id % 180)],
                    "metadata": {
                        "test_type": "safe_crash_test",
                        "test_id": self.test_id,
                        "request_id": request_id
                    }
                },
                timeout=timeout
            )
            
            response_time = time.time() - start_time
            
            return {
                "success": True,
                "response_time": response_time,
                "status_code": response.status_code,
                "request_id": request_id
            }
            
        except httpx.TimeoutException:
            return {
                "success": False,
                "response_time": timeout,
                "error": "timeout",
                "request_id": request_id
            }
        except httpx.ConnectError:
            return {
                "success": False,
                "response_time": 0,
                "error": "connection_refused",
                "request_id": request_id
            }
        except Exception as e:
            return {
                "success": False,
                "response_time": 0,
                "error": str(e),
                "request_id": request_id
            }
    
    def collect_system_metrics(self, phase_name: str, concurrent: int, results: List[Dict]) -> SafeTestMetrics:
        """Collect comprehensive system metrics safely"""
        
        # Request metrics
        successful = sum(1 for r in results if r.get("success", False))
        failed = len(results) - successful
        
        response_times = [r.get("response_time", 0) for r in results if r.get("response_time", 0) > 0]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        max_response_time = max(response_times) if response_times else 0
        
        # Error analysis
        error_details = {}
        for result in results:
            if not result.get("success", False):
                error = result.get("error", "unknown")
                error_details[error] = error_details.get(error, 0) + 1
        
        # System metrics with error handling
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_mb = memory.used // (1024 * 1024)
        except Exception as e:
            logger.warning(f"Failed to collect system metrics: {e}")
            cpu_percent = 0
            memory_percent = 0
            memory_mb = 0
        
        # System responsiveness check
        system_responsive = (successful / len(results) > 0.5) if results else False
        
        return SafeTestMetrics(
            timestamp=time.time(),
            test_phase=phase_name,
            concurrent_requests=concurrent,
            successful_requests=successful,
            failed_requests=failed,
            avg_response_time=avg_response_time,
            max_response_time=max_response_time,
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_mb=memory_mb,
            system_responsive=system_responsive,
            error_details=error_details
        )
    
    async def run_safe_phase(self, phase: Dict[str, Any]) -> bool:
        """Run a single test phase safely"""
        
        phase_name = phase["name"]
        concurrent = phase["concurrent"]
        total_requests = phase["total"]
        timeout = phase["timeout"]
        
        logger.info(f"🔍 Starting phase: {phase_name}")
        logger.info(f"   Concurrent: {concurrent}, Total: {total_requests}, Timeout: {timeout}s")
        
        # Pre-phase health check
        health = await self.health_check()
        if not health.get("responsive", False):
            logger.error(f"❌ System not responsive before phase {phase_name}")
            return False
        
        # Run requests with controlled concurrency
        semaphore = asyncio.Semaphore(concurrent)
        
        async def limited_request(request_id):
            async with semaphore:
                async with httpx.AsyncClient() as session:
                    return await self.safe_request(session, request_id, timeout)
        
        try:
            # Execute phase
            phase_start = time.time()
            tasks = [limited_request(i) for i in range(total_requests)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            valid_results = []
            for result in results:
                if isinstance(result, dict):
                    valid_results.append(result)
                else:
                    valid_results.append({
                        "success": False,
                        "response_time": 0,
                        "error": f"exception: {str(result)}",
                        "request_id": -1
                    })
            
            phase_duration = time.time() - phase_start
            
            # Collect metrics
            metrics = self.collect_system_metrics(phase_name, concurrent, valid_results)
            self.metrics_history.append(metrics)
            
            # Log phase results
            success_rate = (metrics.successful_requests / total_requests * 100) if total_requests > 0 else 0
            logger.info(f"✅ Phase {phase_name} completed in {phase_duration:.2f}s")
            logger.info(f"   Success: {metrics.successful_requests}/{total_requests} ({success_rate:.1f}%)")
            logger.info(f"   Avg Response: {metrics.avg_response_time:.3f}s")
            logger.info(f"   Max Response: {metrics.max_response_time:.3f}s")
            logger.info(f"   CPU: {metrics.cpu_percent:.1f}%")
            logger.info(f"   Memory: {metrics.memory_percent:.1f}% ({metrics.memory_mb} MB)")
            
            if metrics.error_details:
                logger.warning(f"   Errors: {metrics.error_details}")
            
            # Post-phase health check
            await asyncio.sleep(2)  # Brief pause
            post_health = await self.health_check()
            if not post_health.get("responsive", False):
                logger.error(f"❌ System became unresponsive after phase {phase_name}")
                return False
            
            # Success criteria
            if success_rate < 80:
                logger.warning(f"⚠️ Low success rate ({success_rate:.1f}%) in phase {phase_name}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Phase {phase_name} failed with exception: {e}")
            return False
    
    def generate_safe_report(self) -> Dict[str, Any]:
        """Generate comprehensive safe test report"""
        
        if not self.metrics_history:
            return {"error": "No metrics collected"}
        
        # Aggregate metrics
        total_requests = sum(m.successful_requests + m.failed_requests for m in self.metrics_history)
        total_successful = sum(m.successful_requests for m in self.metrics_history)
        total_failed = sum(m.failed_requests for m in self.metrics_history)
        
        # Performance analysis
        avg_response_times = [m.avg_response_time for m in self.metrics_history if m.avg_response_time > 0]
        max_response_times = [m.max_response_time for m in self.metrics_history if m.max_response_time > 0]
        
        # System resource analysis
        peak_cpu = max(m.cpu_percent for m in self.metrics_history)
        peak_memory = max(m.memory_percent for m in self.metrics_history)
        
        # Phase analysis
        phases_completed = len(set(m.test_phase for m in self.metrics_history))
        last_successful_phase = self.metrics_history[-1].test_phase if self.metrics_history else "none"
        
        return {
            "test_id": self.test_id,
            "test_type": "safe_crash_test",
            "duration_seconds": time.time() - self.start_time,
            "phases_completed": phases_completed,
            "last_successful_phase": last_successful_phase,
            "summary": {
                "total_requests": total_requests,
                "successful_requests": total_successful,
                "failed_requests": total_failed,
                "success_rate_percent": (total_successful / total_requests * 100) if total_requests > 0 else 0,
                "average_response_time": sum(avg_response_times) / len(avg_response_times) if avg_response_times else 0,
                "max_response_time": max(max_response_times) if max_response_times else 0,
                "peak_cpu_percent": peak_cpu,
                "peak_memory_percent": peak_memory
            },
            "phase_details": [asdict(m) for m in self.metrics_history],
            "recommendations": self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        if not self.metrics_history:
            return ["No data available for recommendations"]
        
        # Analyze success rates
        avg_success_rate = sum(
            m.successful_requests / (m.successful_requests + m.failed_requests) 
            for m in self.metrics_history 
            if (m.successful_requests + m.failed_requests) > 0
        ) / len(self.metrics_history) if self.metrics_history else 0
        
        if avg_success_rate < 0.9:
            recommendations.append("System shows instability under load - consider performance optimization")
        
        # Analyze response times
        avg_response_times = [m.avg_response_time for m in self.metrics_history if m.avg_response_time > 0]
        if avg_response_times and max(avg_response_times) > 5.0:
            recommendations.append("High response times detected - investigate database/API performance")
        
        # Analyze resource usage
        peak_cpu = max(m.cpu_percent for m in self.metrics_history)
        peak_memory = max(m.memory_percent for m in self.metrics_history)
        
        if peak_cpu > 80:
            recommendations.append("High CPU usage detected - consider CPU optimization or scaling")
        if peak_memory > 80:
            recommendations.append("High memory usage detected - investigate memory leaks or increase resources")
        
        if not recommendations:
            recommendations.append("System performed well under conservative load testing")
        
        return recommendations
    
    async def run_safe_crash_test(self):
        """Execute the complete safe crash test"""
        
        logger.info("🛡️ Starting SAFE Performance Crash Test")
        logger.info("=" * 60)
        
        try:
            # Initial system check
            logger.info("🔍 Performing comprehensive initial system check...")
            initial_health = await self.health_check()
            
            if not initial_health.get("responsive", False):
                logger.error("❌ System is not responsive initially. Aborting test.")
                return
            
            logger.info(f"✅ System responsive (health check: {initial_health.get('response_time', 0):.3f}s)")
            
            # Execute phases
            completed_phases = 0
            for phase_num, phase in enumerate(self.phases, 1):
                logger.info(f"\n🚀 PHASE {phase_num}/{len(self.phases)}: {phase['name'].upper()}")
                logger.info("-" * 40)
                
                success = await self.run_safe_phase(phase)
                
                if success:
                    completed_phases += 1
                    logger.info(f"✅ Phase {phase['name']} completed successfully")
                    
                    # Wait between phases (except last)
                    if phase_num < len(self.phases):
                        logger.info(f"⏳ Waiting {self.delay_between_phases}s before next phase...")
                        await asyncio.sleep(self.delay_between_phases)
                else:
                    logger.error(f"❌ Phase {phase['name']} failed - stopping test")
                    break
            
            # Generate and save report
            report = self.generate_safe_report()
            report["initial_health"] = initial_health
            report["completed_phases"] = completed_phases
            report["total_phases"] = len(self.phases)
            
            # Save report
            report_filename = f"test_results/safe_crash_test_{self.test_id}.json"
            os.makedirs("test_results", exist_ok=True)
            
            with open(report_filename, 'w') as f:
                json.dump(report, f, indent=2)
            
            # Print comprehensive summary
            logger.info("\n" + "=" * 60)
            logger.info("🛡️ SAFE CRASH TEST COMPLETED")
            logger.info("=" * 60)
            logger.info(f"📁 Report saved: {report_filename}")
            logger.info(f"⏱️  Duration: {report['duration_seconds']:.2f} seconds")
            logger.info(f"📊 Phases completed: {completed_phases}/{len(self.phases)}")
            logger.info(f"📈 Total requests: {report['summary']['total_requests']}")
            logger.info(f"✅ Success rate: {report['summary']['success_rate_percent']:.2f}%")
            logger.info(f"⚡ Avg response time: {report['summary']['average_response_time']:.3f}s")
            logger.info(f"🔥 Max response time: {report['summary']['max_response_time']:.3f}s")
            logger.info(f"🖥️  Peak CPU: {report['summary']['peak_cpu_percent']:.1f}%")
            logger.info(f"💾 Peak memory: {report['summary']['peak_memory_percent']:.1f}%")
            
            logger.info("\n🎯 RECOMMENDATIONS:")
            for rec in report["recommendations"]:
                logger.info(f"   • {rec}")
            
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"❌ Safe crash test failed with exception: {e}")
            logger.exception("Full traceback:")

async def main():
    """Main execution function"""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create and run the safe crash test
    safe_test = SafeCrashTest()
    await safe_test.run_safe_crash_test()

if __name__ == "__main__":
    asyncio.run(main()) 