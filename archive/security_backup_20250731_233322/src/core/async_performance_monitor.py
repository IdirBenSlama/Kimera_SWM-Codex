"""
Async Performance Monitor for KIMERA System
Monitors and tracks async operation performance
Phase 2, Week 5: Async/Await Patterns Implementation
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics
from enum import Enum
import psutil
import weakref
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics to track"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class PerformanceMetric:
    """Individual performance metric"""
    name: str
    type: MetricType
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary"""
        return {
            "name": self.name,
            "type": self.type.value,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "labels": self.labels
        }


@dataclass
class AsyncOperationMetrics:
    """Metrics for async operations"""
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    success: bool = True
    error: Optional[str] = None
    memory_before: Optional[float] = None
    memory_after: Optional[float] = None
    
    @property
    def duration(self) -> Optional[float]:
        """Calculate operation duration"""
        if self.end_time:
            return self.end_time - self.start_time
        return None
    
    @property
    def memory_delta(self) -> Optional[float]:
        """Calculate memory usage change"""
        if self.memory_before and self.memory_after:
            return self.memory_after - self.memory_before
        return None


class AsyncPerformanceMonitor:
    """
    Monitor and track performance of async operations
    """
    
    def __init__(
        self,
        history_size: int = 10000,
        aggregation_interval: float = 60.0,
        enable_memory_tracking: bool = True
    ):
        self.history_size = history_size
        self.aggregation_interval = aggregation_interval
        self.enable_memory_tracking = enable_memory_tracking
        
        # Metrics storage
        self.operation_metrics: deque = deque(maxlen=history_size)
        self.custom_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=history_size))
        
        # Aggregated stats
        self.aggregated_stats: Dict[str, Dict[str, Any]] = {}
        
        # Active operations tracking
        self.active_operations: Dict[int, AsyncOperationMetrics] = {}
        
        # Background aggregation task
        self._aggregation_task: Optional[asyncio.Task] = None
        self._shutdown = False
        self._lock = asyncio.Lock()
        
        # Process info
        self._process = psutil.Process()
        
        logger.info(
            f"AsyncPerformanceMonitor initialized: history_size={history_size}, "
            f"aggregation_interval={aggregation_interval}s"
        )
    
    async def start(self) -> None:
        """Start the performance monitor"""
        if not self._aggregation_task:
            self._aggregation_task = asyncio.create_task(self._aggregation_loop())
            logger.info("Performance monitor started")
    
    async def stop(self) -> None:
        """Stop the performance monitor"""
        self._shutdown = True
        if self._aggregation_task:
            self._aggregation_task.cancel()
            try:
                await self._aggregation_task
            except asyncio.CancelledError:
                pass
        logger.info("Performance monitor stopped")
    
    async def _aggregation_loop(self) -> None:
        """Background task to aggregate metrics"""
        while not self._shutdown:
            try:
                await asyncio.sleep(self.aggregation_interval)
                await self._aggregate_metrics()
            except Exception as e:
                logger.error(f"Error in aggregation loop: {e}")
    
    async def _aggregate_metrics(self) -> None:
        """Aggregate metrics for the current interval"""
        async with self._lock:
            now = datetime.now()
            cutoff_time = now - timedelta(seconds=self.aggregation_interval)
            
            # Aggregate operation metrics
            recent_operations = [
                op for op in self.operation_metrics
                if datetime.fromtimestamp(op.start_time) > cutoff_time
            ]
            
            if recent_operations:
                # Group by operation name
                by_operation = defaultdict(list)
                for op in recent_operations:
                    by_operation[op.operation_name].append(op)
                
                # Calculate stats for each operation
                for op_name, ops in by_operation.items():
                    durations = [op.duration for op in ops if op.duration is not None]
                    success_count = sum(1 for op in ops if op.success)
                    
                    stats = {
                        "count": len(ops),
                        "success_count": success_count,
                        "error_count": len(ops) - success_count,
                        "success_rate": success_count / len(ops) if ops else 0,
                        "avg_duration": statistics.mean(durations) if durations else 0,
                        "min_duration": min(durations) if durations else 0,
                        "max_duration": max(durations) if durations else 0,
                        "p50_duration": statistics.median(durations) if durations else 0,
                        "p95_duration": self._percentile(durations, 0.95) if durations else 0,
                        "p99_duration": self._percentile(durations, 0.99) if durations else 0,
                        "timestamp": now.isoformat()
                    }
                    
                    if self.enable_memory_tracking:
                        memory_deltas = [
                            op.memory_delta for op in ops 
                            if op.memory_delta is not None
                        ]
                        if memory_deltas:
                            stats["avg_memory_delta"] = statistics.mean(memory_deltas)
                            stats["max_memory_delta"] = max(memory_deltas)
                    
                    self.aggregated_stats[op_name] = stats
            
            logger.debug(f"Aggregated metrics for {len(self.aggregated_stats)} operations")
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of data"""
        if not data:
            return 0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    @asynccontextmanager
    async def track_operation(self, operation_name: str):
        """
        Context manager to track an async operation
        
        Usage:
            async with monitor.track_operation("database_query"):
                result = await db.query(...)
        """
        operation_id = id(asyncio.current_task())
        
        # Create metrics object
        metrics = AsyncOperationMetrics(
            operation_name=operation_name,
            start_time=time.time()
        )
        
        # Track memory if enabled
        if self.enable_memory_tracking:
            metrics.memory_before = self._process.memory_info().rss / 1024 / 1024  # MB
        
        async with self._lock:
            self.active_operations[operation_id] = metrics
        
        try:
            yield metrics
            metrics.success = True
            
        except Exception as e:
            metrics.success = False
            metrics.error = str(e)
            raise
            
        finally:
            metrics.end_time = time.time()
            
            # Track memory after
            if self.enable_memory_tracking:
                metrics.memory_after = self._process.memory_info().rss / 1024 / 1024  # MB
            
            async with self._lock:
                # Remove from active operations
                self.active_operations.pop(operation_id, None)
                
                # Add to history
                self.operation_metrics.append(metrics)
            
            # Log slow operations
            if metrics.duration and metrics.duration > 1.0:
                logger.warning(
                    f"Slow operation detected: {operation_name} took {metrics.duration:.2f}s"
                )
    
    async def record_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Record a custom metric
        
        Args:
            name: Metric name
            value: Metric value
            metric_type: Type of metric
            labels: Optional labels for the metric
        """
        metric = PerformanceMetric(
            name=name,
            type=metric_type,
            value=value,
            timestamp=datetime.now(),
            labels=labels or {}
        )
        
        async with self._lock:
            self.custom_metrics[name].append(metric)
    
    async def get_operation_stats(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics for operations
        
        Args:
            operation_name: Optional specific operation to get stats for
            
        Returns:
            Dictionary of operation statistics
        """
        async with self._lock:
            if operation_name:
                return self.aggregated_stats.get(operation_name, {})
            return dict(self.aggregated_stats)
    
    async def get_active_operations(self) -> List[Dict[str, Any]]:
        """Get list of currently active operations"""
        async with self._lock:
            return [
                {
                    "operation_name": op.operation_name,
                    "duration_so_far": time.time() - op.start_time,
                    "start_time": datetime.fromtimestamp(op.start_time).isoformat()
                }
                for op in self.active_operations.values()
            ]
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        cpu_percent = self._process.cpu_percent(interval=0.1)
        memory_info = self._process.memory_info()
        
        # Get event loop stats
        loop = asyncio.get_running_loop()
        
        return {
            "cpu_percent": cpu_percent,
            "memory_rss_mb": memory_info.rss / 1024 / 1024,
            "memory_vms_mb": memory_info.vms / 1024 / 1024,
            "active_operations": len(self.active_operations),
            "total_operations_tracked": len(self.operation_metrics),
            "event_loop_running": loop.is_running(),
            "timestamp": datetime.now().isoformat()
        }
    
    async def get_slow_operations(
        self,
        threshold: float = 1.0,
        limit: int = 10
    ) -> List[AsyncOperationMetrics]:
        """
        Get slowest operations
        
        Args:
            threshold: Minimum duration to consider slow (seconds)
            limit: Maximum number of operations to return
            
        Returns:
            List of slow operations
        """
        async with self._lock:
            slow_ops = [
                op for op in self.operation_metrics
                if op.duration and op.duration > threshold
            ]
            
            # Sort by duration descending
            slow_ops.sort(key=lambda x: x.duration or 0, reverse=True)
            
            return slow_ops[:limit]
    
    async def get_error_operations(self, limit: int = 10) -> List[AsyncOperationMetrics]:
        """Get operations that resulted in errors"""
        async with self._lock:
            error_ops = [op for op in self.operation_metrics if not op.success]
            return error_ops[-limit:]  # Return most recent errors
    
    def create_performance_report(self) -> Dict[str, Any]:
        """Create a comprehensive performance report"""
        return asyncio.run(self._create_performance_report())
    
    async def _create_performance_report(self) -> Dict[str, Any]:
        """Async implementation of performance report creation"""
        operation_stats = await self.get_operation_stats()
        system_metrics = await self.get_system_metrics()
        slow_operations = await self.get_slow_operations()
        error_operations = await self.get_error_operations()
        
        return {
            "report_timestamp": datetime.now().isoformat(),
            "system_metrics": system_metrics,
            "operation_statistics": operation_stats,
            "slow_operations": [
                {
                    "name": op.operation_name,
                    "duration": op.duration,
                    "timestamp": datetime.fromtimestamp(op.start_time).isoformat()
                }
                for op in slow_operations
            ],
            "recent_errors": [
                {
                    "name": op.operation_name,
                    "error": op.error,
                    "timestamp": datetime.fromtimestamp(op.start_time).isoformat()
                }
                for op in error_operations
            ],
            "summary": {
                "total_operations": len(self.operation_metrics),
                "unique_operations": len(operation_stats),
                "error_rate": sum(
                    stats.get("error_count", 0) 
                    for stats in operation_stats.values()
                ) / max(len(self.operation_metrics), 1)
            }
        }


# Global monitor instance
_performance_monitor: Optional[AsyncPerformanceMonitor] = None


def get_performance_monitor() -> AsyncPerformanceMonitor:
    """Get global performance monitor instance"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = AsyncPerformanceMonitor()
    return _performance_monitor


async def initialize_performance_monitor(
    history_size: int = 10000,
    aggregation_interval: float = 60.0,
    enable_memory_tracking: bool = True
) -> AsyncPerformanceMonitor:
    """Initialize and start the global performance monitor"""
    global _performance_monitor
    _performance_monitor = AsyncPerformanceMonitor(
        history_size=history_size,
        aggregation_interval=aggregation_interval,
        enable_memory_tracking=enable_memory_tracking
    )
    await _performance_monitor.start()
    return _performance_monitor