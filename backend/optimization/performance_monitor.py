"""
Real-time Performance Monitoring System
======================================
Advanced performance monitoring with anomaly detection and auto-tuning.

Features:
1. Real-time latency tracking
2. Request rate monitoring
3. Error rate analysis
4. Resource utilization tracking
5. Automatic performance tuning
"""

import time
import asyncio
import numpy as np
from typing import Dict, Any, List, Optional, Deque
from collections import deque, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

@dataclass
class RequestMetrics:
    """Metrics for a single request"""
    endpoint: str
    method: str
    start_time: float
    end_time: float
    status_code: int
    response_size: int
    
    @property
    def latency_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000
    
    @property
    def throughput_mbps(self) -> float:
        duration_s = self.end_time - self.start_time
        if duration_s > 0:
            return (self.response_size / (1024 * 1024)) / duration_s
        return 0.0


@dataclass
class EndpointStats:
    """Statistics for a specific endpoint"""
    endpoint: str
    request_count: int = 0
    error_count: int = 0
    total_latency_ms: float = 0.0
    latencies: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    
    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.request_count if self.request_count > 0 else 0
    
    @property
    def error_rate(self) -> float:
        return self.error_count / self.request_count if self.request_count > 0 else 0
    
    @property
    def p50_latency_ms(self) -> float:
        if self.latencies:
            return np.percentile(list(self.latencies), 50)
        return 0
    
    @property
    def p95_latency_ms(self) -> float:
        if self.latencies:
            return np.percentile(list(self.latencies), 95)
        return 0
    
    @property
    def p99_latency_ms(self) -> float:
        if self.latencies:
            return np.percentile(list(self.latencies), 99)
        return 0


class PerformanceMonitor:
    """
    Advanced performance monitoring with auto-tuning capabilities.
    
    Tracks performance metrics in real-time and can automatically
    adjust system parameters to maintain optimal performance.
    """
    
    def __init__(self, 
                 target_latency_ms: float = 100,
                 target_error_rate: float = 0.01,
                 monitoring_window_seconds: int = 60):
        
        self.target_latency_ms = target_latency_ms
        self.target_error_rate = target_error_rate
        self.monitoring_window = timedelta(seconds=monitoring_window_seconds)
        
        # Metrics storage
        self.endpoint_stats: Dict[str, EndpointStats] = defaultdict(
            lambda: EndpointStats(endpoint="unknown")
        )
        self.recent_requests: Deque[RequestMetrics] = deque(maxlen=10000)
        
        # Performance tracking
        self.performance_history: Deque[Dict[str, Any]] = deque(maxlen=3600)
        self.anomalies: List[Dict[str, Any]] = []
        
        # Auto-tuning parameters
        self.auto_tune_enabled = True
        self.current_settings = {
            "max_workers": 4,
            "cache_ttl_seconds": 1,
            "batch_size": 10,
            "connection_pool_size": 100
        }
        
        # Monitoring task
        self._monitor_task = None
        self._running = False
    
    def record_request(self, metrics: RequestMetrics):
        """Record metrics for a completed request"""
        # Update endpoint stats
        stats = self.endpoint_stats[metrics.endpoint]
        stats.endpoint = metrics.endpoint
        stats.request_count += 1
        stats.total_latency_ms += metrics.latency_ms
        stats.latencies.append(metrics.latency_ms)
        
        if metrics.status_code >= 400:
            stats.error_count += 1
        
        # Store recent request
        self.recent_requests.append(metrics)
        
        # Check for anomalies
        self._check_anomalies(metrics)
    
    def _check_anomalies(self, metrics: RequestMetrics):
        """Check for performance anomalies"""
        # Latency anomaly
        if metrics.latency_ms > self.target_latency_ms * 10:
            self.anomalies.append({
                "type": "high_latency",
                "endpoint": metrics.endpoint,
                "latency_ms": metrics.latency_ms,
                "timestamp": datetime.fromtimestamp(metrics.end_time),
                "severity": "critical" if metrics.latency_ms > self.target_latency_ms * 50 else "warning"
            })
        
        # Error rate anomaly
        stats = self.endpoint_stats[metrics.endpoint]
        if stats.request_count > 100 and stats.error_rate > self.target_error_rate * 5:
            self.anomalies.append({
                "type": "high_error_rate",
                "endpoint": metrics.endpoint,
                "error_rate": stats.error_rate,
                "timestamp": datetime.now(),
                "severity": "critical"
            })
    
    async def start_monitoring(self):
        """Start the monitoring loop"""
        if not self._running:
            self._running = True
            self._monitor_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Performance monitoring started")
    
    async def stop_monitoring(self):
        """Stop the monitoring loop"""
        self._running = False
        if self._monitor_task:
            await self._monitor_task
        logger.info("Performance monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self._running:
            try:
                # Collect current performance snapshot
                snapshot = self._collect_performance_snapshot()
                self.performance_history.append(snapshot)
                
                # Auto-tune if enabled
                if self.auto_tune_enabled:
                    await self._auto_tune_performance(snapshot)
                
                # Log performance summary
                self._log_performance_summary(snapshot)
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(10)
    
    def _collect_performance_snapshot(self) -> Dict[str, Any]:
        """Collect current performance metrics"""
        now = datetime.now()
        cutoff_time = time.time() - self.monitoring_window.total_seconds()
        
        # Filter recent requests
        recent = [r for r in self.recent_requests if r.end_time > cutoff_time]
        
        if not recent:
            return {
                "timestamp": now,
                "request_rate": 0,
                "avg_latency_ms": 0,
                "error_rate": 0,
                "endpoints": {}
            }
        
        # Calculate aggregate metrics
        total_requests = len(recent)
        total_errors = sum(1 for r in recent if r.status_code >= 400)
        avg_latency = sum(r.latency_ms for r in recent) / total_requests
        
        # Calculate per-endpoint metrics
        endpoint_metrics = {}
        for endpoint, stats in self.endpoint_stats.items():
            if stats.request_count > 0:
                endpoint_metrics[endpoint] = {
                    "requests": stats.request_count,
                    "avg_latency_ms": stats.avg_latency_ms,
                    "p95_latency_ms": stats.p95_latency_ms,
                    "p99_latency_ms": stats.p99_latency_ms,
                    "error_rate": stats.error_rate
                }
        
        return {
            "timestamp": now,
            "request_rate": total_requests / self.monitoring_window.total_seconds(),
            "avg_latency_ms": avg_latency,
            "error_rate": total_errors / total_requests,
            "endpoints": endpoint_metrics,
            "settings": self.current_settings.copy()
        }
    
    async def _auto_tune_performance(self, snapshot: Dict[str, Any]):
        """Automatically tune performance parameters"""
        avg_latency = snapshot["avg_latency_ms"]
        error_rate = snapshot["error_rate"]
        request_rate = snapshot["request_rate"]
        
        # Latency-based tuning
        if avg_latency > self.target_latency_ms * 2:
            # Increase resources
            if self.current_settings["max_workers"] < 16:
                self.current_settings["max_workers"] += 2
                logger.info(f"Increased max_workers to {self.current_settings['max_workers']}")
            
            if self.current_settings["cache_ttl_seconds"] < 5:
                self.current_settings["cache_ttl_seconds"] += 1
                logger.info(f"Increased cache TTL to {self.current_settings['cache_ttl_seconds']}s")
        
        elif avg_latency < self.target_latency_ms * 0.5 and request_rate < 10:
            # Reduce resources if underutilized
            if self.current_settings["max_workers"] > 2:
                self.current_settings["max_workers"] -= 1
                logger.info(f"Reduced max_workers to {self.current_settings['max_workers']}")
        
        # Error rate tuning
        if error_rate > self.target_error_rate * 2:
            # Increase connection pool
            if self.current_settings["connection_pool_size"] < 500:
                self.current_settings["connection_pool_size"] += 50
                logger.info(f"Increased connection pool to {self.current_settings['connection_pool_size']}")
        
        # Apply settings (this would integrate with actual system components)
        await self._apply_settings()
    
    async def _apply_settings(self):
        """Apply auto-tuned settings to the system"""
        # This would integrate with actual system components
        # For now, just log the intent
        logger.info(f"Applied performance settings: {self.current_settings}")
    
    def _log_performance_summary(self, snapshot: Dict[str, Any]):
        """Log performance summary"""
        if snapshot["request_rate"] > 0:
            logger.info(
                f"Performance: {snapshot['request_rate']:.1f} req/s, "
                f"{snapshot['avg_latency_ms']:.1f}ms avg latency, "
                f"{snapshot['error_rate']*100:.2f}% errors"
            )
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        if not self.performance_history:
            return {"status": "no_data"}
        
        # Get latest snapshot
        latest = self.performance_history[-1]
        
        # Calculate trends
        if len(self.performance_history) > 10:
            recent_latencies = [s["avg_latency_ms"] for s in list(self.performance_history)[-10:]]
            latency_trend = "increasing" if recent_latencies[-1] > recent_latencies[0] else "decreasing"
        else:
            latency_trend = "stable"
        
        # Find slowest endpoints
        slowest_endpoints = sorted(
            latest["endpoints"].items(),
            key=lambda x: x[1]["avg_latency_ms"],
            reverse=True
        )[:5]
        
        return {
            "current_performance": {
                "request_rate": latest["request_rate"],
                "avg_latency_ms": latest["avg_latency_ms"],
                "error_rate": latest["error_rate"]
            },
            "targets": {
                "latency_ms": self.target_latency_ms,
                "error_rate": self.target_error_rate
            },
            "trends": {
                "latency": latency_trend
            },
            "slowest_endpoints": slowest_endpoints,
            "recent_anomalies": self.anomalies[-10:],
            "auto_tune_settings": self.current_settings,
            "recommendations": self._generate_recommendations(latest)
        }
    
    def _generate_recommendations(self, snapshot: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        if snapshot["avg_latency_ms"] > self.target_latency_ms:
            recommendations.append(
                f"Average latency ({snapshot['avg_latency_ms']:.1f}ms) exceeds target "
                f"({self.target_latency_ms}ms). Consider scaling resources."
            )
        
        if snapshot["error_rate"] > self.target_error_rate:
            recommendations.append(
                f"Error rate ({snapshot['error_rate']*100:.2f}%) exceeds target "
                f"({self.target_error_rate*100:.2f}%). Check error logs."
            )
        
        # Check for specific endpoint issues
        for endpoint, metrics in snapshot["endpoints"].items():
            if metrics["p99_latency_ms"] > self.target_latency_ms * 10:
                recommendations.append(
                    f"Endpoint {endpoint} has high P99 latency "
                    f"({metrics['p99_latency_ms']:.1f}ms). Needs optimization."
                )
        
        return recommendations


# Global performance monitor instance
_performance_monitor = PerformanceMonitor()

def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance"""
    return _performance_monitor