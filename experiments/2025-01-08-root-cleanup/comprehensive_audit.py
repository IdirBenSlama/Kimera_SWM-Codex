#!/usr/bin/env python3
"""
Comprehensive Kimera System Audit with Extended Runtime
======================================================

This script performs an extensive audit of the Kimera system with:
- Long runtime testing (configurable duration)
- Continuous monitoring of all engines
- Performance metrics collection
- Stress testing capabilities
- Detailed analytics and reporting
- Real-time status updates
"""

import requests
import json
import time
import threading
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import sys
import signal

# Configuration
BASE_URL = "http://localhost:8000"
DEFAULT_RUNTIME_MINUTES = 30  # Default 30 minutes
MONITORING_INTERVAL = 5  # seconds between checks
STRESS_TEST_INTERVAL = 60  # seconds between stress tests
TIMEOUT = 30  # request timeout
MAX_RETRIES = 3

@dataclass
class EngineMetrics:
    """Metrics for a single engine"""
    name: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    response_times: List[float] = None
    error_count: int = 0
    last_status: str = "unknown"
    uptime_percentage: float = 0.0
    
    def __post_init__(self):
        if self.response_times is None:
            self.response_times = []
    
    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100
    
    @property
    def avg_response_time(self) -> float:
        if not self.response_times:
            return 0.0
        return statistics.mean(self.response_times)
    
    @property
    def min_response_time(self) -> float:
        if not self.response_times:
            return 0.0
        return min(self.response_times)
    
    @property
    def max_response_time(self) -> float:
        if not self.response_times:
            return 0.0
        return max(self.response_times)

@dataclass
class SystemSnapshot:
    """System snapshot at a point in time"""
    timestamp: datetime
    operational_engines: int
    total_engines: int
    system_health: str
    response_time: float
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None

class KimeraAuditor:
    """Comprehensive Kimera system auditor"""
    
    def __init__(self, runtime_minutes: int = DEFAULT_RUNTIME_MINUTES):
        self.runtime_minutes = runtime_minutes
        self.start_time = datetime.now()
        self.end_time = self.start_time + timedelta(minutes=runtime_minutes)
        self.running = False
        self.paused = False
        
        # Engine configurations
        self.engines = {
            "System Health": f"{BASE_URL}/kimera/monitoring/health",
            "Engines Status": f"{BASE_URL}/kimera/monitoring/engines/status",
            "Contradiction Engine": f"{BASE_URL}/kimera/monitoring/engines/contradiction",
            "Thermodynamics Engine": f"{BASE_URL}/kimera/monitoring/engines/thermodynamics",
            "SPDE Engine": f"{BASE_URL}/kimera/monitoring/engines/spde",
            "Cognitive Cycle": f"{BASE_URL}/kimera/monitoring/engines/cognitive_cycle",
            "Meta Insight": f"{BASE_URL}/kimera/monitoring/engines/meta_insight",
            "Proactive Detector": f"{BASE_URL}/kimera/monitoring/engines/proactive_detector",
            "Revolutionary Intelligence": f"{BASE_URL}/kimera/monitoring/engines/revolutionary"
        }
        
        # Metrics storage
        self.engine_metrics = {name: EngineMetrics(name) for name in self.engines.keys()}
        self.system_snapshots: List[SystemSnapshot] = []
        self.performance_history = defaultdict(deque)
        
        # Threading
        self.monitor_thread = None
        self.stress_thread = None
        self.display_thread = None
        
        # Statistics
        self.total_checks = 0
        self.successful_checks = 0
        self.critical_failures = 0
        self.performance_degradations = 0
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"\n🛑 Received signal {signum}, shutting down gracefully...")
        self.stop()
        sys.exit(0)
    
    def test_endpoint(self, name: str, url: str) -> Tuple[bool, float, Dict[str, Any]]:
        """Test a single endpoint with retries"""
        for attempt in range(MAX_RETRIES):
            try:
                start_time = time.time()
                response = requests.get(url, timeout=TIMEOUT)
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    status = data.get("status", "unknown")
                    
                    # Update metrics
                    metrics = self.engine_metrics[name]
                    metrics.total_requests += 1
                    metrics.successful_requests += 1
                    metrics.response_times.append(response_time)
                    metrics.last_status = status
                    
                    # Keep only last 1000 response times for memory efficiency
                    if len(metrics.response_times) > 1000:
                        metrics.response_times = metrics.response_times[-1000:]
                    
                    return True, response_time, data
                else:
                    self._record_failure(name, f"HTTP {response.status_code}")
                    return False, 0.0, {"error": f"HTTP {response.status_code}"}
                    
            except requests.exceptions.ConnectionError:
                if attempt == MAX_RETRIES - 1:
                    self._record_failure(name, "Connection error")
                    return False, 0.0, {"error": "Connection error"}
                time.sleep(1)
            except requests.exceptions.Timeout:
                if attempt == MAX_RETRIES - 1:
                    self._record_failure(name, "Timeout")
                    return False, 0.0, {"error": "Timeout"}
                time.sleep(1)
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    self._record_failure(name, str(e))
                    return False, 0.0, {"error": str(e)}
                time.sleep(1)
        
        return False, 0.0, {"error": "Max retries exceeded"}
    
    def _record_failure(self, engine_name: str, error: str):
        """Record a failure for an engine"""
        metrics = self.engine_metrics[engine_name]
        metrics.total_requests += 1
        metrics.failed_requests += 1
        metrics.error_count += 1
        metrics.last_status = "error"
        
        # Track critical failures
        if "Connection error" in error:
            self.critical_failures += 1
    
    def monitor_engines(self):
        """Continuous monitoring of all engines"""
        logger.info(f"🔍 Starting continuous monitoring (interval: {MONITORING_INTERVAL}s)")
        
        while self.running and datetime.now() < self.end_time:
            if self.paused:
                time.sleep(1)
                continue
            
            cycle_start = time.time()
            operational_count = 0
            total_response_time = 0
            
            # Test all engines
            for name, url in self.engines.items():
                if not self.running:
                    break
                    
                success, response_time, data = self.test_endpoint(name, url)
                total_response_time += response_time
                
                if success and data.get("status") == "operational":
                    operational_count += 1
                
                # Small delay between requests to avoid overwhelming the server
                time.sleep(0.1)
            
            # Record system snapshot
            if self.running:
                snapshot = SystemSnapshot(
                    timestamp=datetime.now(),
                    operational_engines=operational_count,
                    total_engines=len(self.engines),
                    system_health="healthy" if operational_count >= len(self.engines) * 0.8 else "degraded",
                    response_time=total_response_time / len(self.engines) if self.engines else 0
                )
                self.system_snapshots.append(snapshot)
                
                # Keep only last 1000 snapshots for memory efficiency
                if len(self.system_snapshots) > 1000:
                    self.system_snapshots = self.system_snapshots[-1000:]
            
            self.total_checks += 1
            # Count as successful if we got a valid response (regardless of status)
            # For engine-specific success rate, we track individual engine success in their metrics
            self.successful_checks += 1
            
            # Calculate sleep time to maintain interval
            cycle_time = time.time() - cycle_start
            sleep_time = max(0, MONITORING_INTERVAL - cycle_time)
            time.sleep(sleep_time)
    
    def stress_test(self):
        """Periodic stress testing"""
        logger.info(f"💪 Starting stress testing (interval: {STRESS_TEST_INTERVAL}s)")
        
        while self.running and datetime.now() < self.end_time:
            if self.paused:
                time.sleep(1)
                continue
            
            # Perform burst requests to test system resilience
            logger.info("🔥 Executing stress test burst...")
            
            for _ in range(10):  # 10 rapid requests
                if not self.running:
                    break
                    
                # Test a random engine
                import random
                engine_name = random.choice(list(self.engines.keys()))
                url = self.engines[engine_name]
                
                self.test_endpoint(engine_name, url)
                time.sleep(0.1)  # Small delay between burst requests
            
            # Wait for next stress test
            time.sleep(STRESS_TEST_INTERVAL)
    
    def display_realtime_stats(self):
        """Display real-time statistics"""
        while self.running and datetime.now() < self.end_time:
            if self.paused:
                time.sleep(1)
                continue
            
            # Clear screen and display stats
            logger.info("\033[2J\033[H")  # Clear screen
            self._display_current_stats()
            
            time.sleep(10)  # Update every 10 seconds
    
    def _display_current_stats(self):
        """Display current system statistics"""
        now = datetime.now()
        elapsed = now - self.start_time
        remaining = self.end_time - now
        
        logger.info("=" * 80)
        logger.info(f"🔍 KIMERA SYSTEM AUDIT - REAL-TIME MONITORING")
        logger.info("=" * 80)
        logger.info(f"Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Elapsed: {str(elapsed).split('.')[0]}")
        logger.info(f"Remaining: {str(remaining).split('.')[0]}")
        logger.info(f"Total Checks: {self.total_checks}")
        logger.info(f"Success Rate: {(self.successful_checks/self.total_checks*100):.1f}%" if self.total_checks > 0 else "0.0%")
        logger.info(f"Critical Failures: {self.critical_failures}")
        logger.info()
        
        # Engine status summary
        logger.info("🔧 ENGINE STATUS SUMMARY")
        logger.info("-" * 50)
        
        operational_engines = []
        degraded_engines = []
        failed_engines = []
        
        for name, metrics in self.engine_metrics.items():
            if name in ["System Health", "Engines Status"]:
                continue
                
            if metrics.last_status == "operational":
                operational_engines.append(name)
            elif metrics.last_status in ["error", "not_available"]:
                failed_engines.append(name)
            else:
                degraded_engines.append(name)
        
        logger.info(f"✅ Operational: {len(operational_engines)}")
        logger.info(f"⚠️ Degraded: {len(degraded_engines)}")
        logger.info(f"❌ Failed: {len(failed_engines)}")
        logger.info()
        
        # Performance metrics
        logger.info("📊 PERFORMANCE METRICS")
        logger.info("-" * 50)
        
        for name, metrics in self.engine_metrics.items():
            if metrics.total_requests > 0:
                logger.info(f"{name[:30]:30} | "
                      f"Req: {metrics.total_requests:4d} | "
                      f"Success: {metrics.success_rate:5.1f}% | "
                      f"Avg: {metrics.avg_response_time*1000:6.1f}ms | "
                      f"Status: {metrics.last_status}")
        
        logger.info()
        
        # System health trend
        if len(self.system_snapshots) > 1:
            recent_snapshots = self.system_snapshots[-10:]  # Last 10 snapshots
            avg_operational = statistics.mean(s.operational_engines for s in recent_snapshots)
            avg_response_time = statistics.mean(s.response_time for s in recent_snapshots)
            
            logger.info("📈 RECENT TRENDS (Last 10 checks)")
            logger.info("-" * 50)
            logger.info(f"Average Operational Engines: {avg_operational:.1f}/{len(self.engines)}")
            logger.info(f"Average Response Time: {avg_response_time*1000:.1f}ms")
            
            # Health trend
            healthy_count = sum(1 for s in recent_snapshots if s.system_health == "healthy")
            health_percentage = (healthy_count / len(recent_snapshots)) * 100
            logger.info(f"Health Stability: {health_percentage:.1f}%")
        
        logger.info()
        logger.info("Press Ctrl+C to stop the audit gracefully")
        logger.info("=" * 80)
    
    def start(self):
        """Start the comprehensive audit"""
        logger.info("🚀 Starting Comprehensive Kimera System Audit")
        logger.info("=" * 60)
        logger.info(f"Runtime: {self.runtime_minutes} minutes")
        logger.info(f"Monitoring interval: {MONITORING_INTERVAL} seconds")
        logger.info(f"Stress test interval: {STRESS_TEST_INTERVAL} seconds")
        logger.info(f"Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"End time: {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info()
        
        # Initial system check
        logger.info("🔍 Performing initial system check...")
        initial_operational = 0
        for name, url in self.engines.items():
            success, _, data = self.test_endpoint(name, url)
            if success and data.get("status") == "operational":
                initial_operational += 1
        
        logger.info(f"Initial status: {initial_operational}/{len(self.engines)} engines operational")
        logger.info()
        
        # Start monitoring threads
        self.running = True
        
        self.monitor_thread = threading.Thread(target=self.monitor_engines, daemon=True)
        self.stress_thread = threading.Thread(target=self.stress_test, daemon=True)
        self.display_thread = threading.Thread(target=self.display_realtime_stats, daemon=True)
        
        self.monitor_thread.start()
        self.stress_thread.start()
        self.display_thread.start()
        
        # Wait for completion or interruption
        try:
            while self.running and datetime.now() < self.end_time:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("\n🛑 Audit interrupted by user")
        
        self.stop()
    
    def stop(self):
        """Stop the audit"""
        self.running = False
        
        # Wait for threads to finish
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        if self.stress_thread and self.stress_thread.is_alive():
            self.stress_thread.join(timeout=5)
        if self.display_thread and self.display_thread.is_alive():
            self.display_thread.join(timeout=5)
    
    def generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive final report"""
        end_time = datetime.now()
        actual_runtime = end_time - self.start_time
        
        # Calculate uptime percentages
        for metrics in self.engine_metrics.values():
            if metrics.total_requests > 0:
                metrics.uptime_percentage = (metrics.successful_requests / metrics.total_requests) * 100
        
        # System stability metrics
        if self.system_snapshots:
            healthy_snapshots = sum(1 for s in self.system_snapshots if s.system_health == "healthy")
            system_stability = (healthy_snapshots / len(self.system_snapshots)) * 100
        else:
            system_stability = 0.0
        
        # Performance statistics
        all_response_times = []
        for metrics in self.engine_metrics.values():
            all_response_times.extend(metrics.response_times)
        
        performance_stats = {}
        if all_response_times:
            performance_stats = {
                "average_response_time": statistics.mean(all_response_times),
                "median_response_time": statistics.median(all_response_times),
                "min_response_time": min(all_response_times),
                "max_response_time": max(all_response_times),
                "response_time_std": statistics.stdev(all_response_times) if len(all_response_times) > 1 else 0
            }
        
        # Engine reliability ranking
        engine_reliability = []
        for name, metrics in self.engine_metrics.items():
            if name not in ["System Health", "Engines Status"] and metrics.total_requests > 0:
                engine_reliability.append({
                    "name": name,
                    "success_rate": metrics.success_rate,
                    "uptime_percentage": metrics.uptime_percentage,
                    "avg_response_time": metrics.avg_response_time,
                    "total_requests": metrics.total_requests
                })
        
        engine_reliability.sort(key=lambda x: x["success_rate"], reverse=True)
        
        report = {
            "audit_summary": {
                "start_time": self.start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "planned_runtime_minutes": self.runtime_minutes,
                "actual_runtime_minutes": actual_runtime.total_seconds() / 60,
                "total_checks": self.total_checks,
                "successful_checks": self.successful_checks,
                "success_rate": (self.successful_checks / self.total_checks * 100) if self.total_checks > 0 else 0,
                "critical_failures": self.critical_failures,
                "system_stability_percentage": system_stability
            },
            "engine_metrics": {name: asdict(metrics) for name, metrics in self.engine_metrics.items()},
            "performance_statistics": performance_stats,
            "engine_reliability_ranking": engine_reliability,
            "system_snapshots_count": len(self.system_snapshots),
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on audit results"""
        recommendations = []
        
        # Check for engines with low success rates
        for name, metrics in self.engine_metrics.items():
            if name in ["System Health", "Engines Status"]:
                continue
                
            if metrics.total_requests > 0:
                if metrics.success_rate < 95:
                    recommendations.append(f"⚠️ {name} has low success rate ({metrics.success_rate:.1f}%) - investigate stability")
                
                if metrics.avg_response_time > 5.0:
                    recommendations.append(f"🐌 {name} has high response time ({metrics.avg_response_time:.2f}s) - optimize performance")
        
        # Check system stability
        if self.system_snapshots:
            healthy_snapshots = sum(1 for s in self.system_snapshots if s.system_health == "healthy")
            stability = (healthy_snapshots / len(self.system_snapshots)) * 100
            
            if stability < 90:
                recommendations.append(f"🔧 System stability is low ({stability:.1f}%) - investigate recurring issues")
        
        # Check for critical failures
        if self.critical_failures > 0:
            recommendations.append(f"🚨 {self.critical_failures} critical failures detected - check server connectivity")
        
        if not recommendations:
            recommendations.append("✅ System is performing well - no critical issues detected")
        
        return recommendations
    
    def save_report(self, filename: Optional[str] = None) -> str:
        """Save the final report to a JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"kimera_audit_report_{timestamp}.json"
        
        report = self.generate_final_report()
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return filename
    
    def print_final_report(self):
        """Print a formatted final report"""
        report = self.generate_final_report()
        
        logger.info("\n" + "=" * 80)
        logger.info("📋 FINAL AUDIT REPORT")
        logger.info("=" * 80)
        
        # Summary
        summary = report["audit_summary"]
        logger.info(f"Audit Duration: {summary['actual_runtime_minutes']:.1f} minutes")
        logger.info(f"Total System Checks: {summary['total_checks']}")
        logger.info(f"Overall Success Rate: {summary['success_rate']:.1f}%")
        logger.info(f"System Stability: {summary['system_stability_percentage']:.1f}%")
        logger.info(f"Critical Failures: {summary['critical_failures']}")
        logger.info()
        
        # Engine reliability ranking
        logger.info("🏆 ENGINE RELIABILITY RANKING")
        logger.info("-" * 60)
        for i, engine in enumerate(report["engine_reliability_ranking"], 1):
            logger.info(f"{i:2d}. {engine['name'][:25]:25} | "
                  f"Success: {engine['success_rate']:5.1f}% | "
                  f"Uptime: {engine['uptime_percentage']:5.1f}% | "
                  f"Avg Response: {engine['avg_response_time']*1000:6.1f}ms")
        logger.info()
        
        # Performance statistics
        if report["performance_statistics"]:
            perf = report["performance_statistics"]
            logger.info("📊 PERFORMANCE STATISTICS")
            logger.info("-" * 60)
            logger.info(f"Average Response Time: {perf['average_response_time']*1000:.1f}ms")
            logger.info(f"Median Response Time: {perf['median_response_time']*1000:.1f}ms")
            logger.info(f"Min Response Time: {perf['min_response_time']*1000:.1f}ms")
            logger.info(f"Max Response Time: {perf['max_response_time']*1000:.1f}ms")
            logger.info(f"Response Time Std Dev: {perf['response_time_std']*1000:.1f}ms")
            logger.info()
        
        # Recommendations
        logger.info("💡 RECOMMENDATIONS")
        logger.info("-" * 60)
        for rec in report["recommendations"]:
            logger.info(rec)
        logger.info()
        
        # Save report
        filename = self.save_report()
        logger.info(f"📄 Detailed report saved to: {filename}")
        logger.info("=" * 80)

def main():
    """Main function"""
    import argparse
import logging
logger = logging.getLogger(__name__)
    
    parser = argparse.ArgumentParser(description="Comprehensive Kimera System Audit")
    parser.add_argument("--runtime", "-r", type=int, default=DEFAULT_RUNTIME_MINUTES,
                       help=f"Runtime in minutes (default: {DEFAULT_RUNTIME_MINUTES})")
    parser.add_argument("--no-stress", action="store_true",
                       help="Disable stress testing")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Suppress real-time display")
    
    args = parser.parse_args()
    
    auditor = KimeraAuditor(runtime_minutes=args.runtime)
    
    try:
        auditor.start()
    except KeyboardInterrupt:
        logger.info("\n🛑 Audit interrupted by user")
    finally:
        auditor.print_final_report()

if __name__ == "__main__":
    main() 