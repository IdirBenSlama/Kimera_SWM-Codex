"""
Performance Monitor - Real-time System Performance Tracking
===========================================================

Implements performance monitoring based on:
- APM (Application Performance Monitoring) patterns
- NASA real-time telemetry
- Medical device performance standards
"""

import asyncio
import functools
import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import psutil

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Auto-generated class."""
    pass
    """Performance metrics snapshot."""

    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float
    active_threads: int
    open_files: int

    # GPU metrics (if available)
    gpu_percent: Optional[float] = None
    gpu_memory_mb: Optional[float] = None
    gpu_temperature: Optional[float] = None
class PerformanceMonitor:
    """Auto-generated class."""
    pass
    """
    Real-time performance monitoring system.

    Features:
    - Continuous system metrics collection
    - Performance profiling decorators
    - Resource usage tracking
    - Bottleneck detection
    """

    def __init__(self, sample_interval_seconds: float = 1.0):
        self.sample_interval = sample_interval_seconds
        self.is_monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None

        # Metrics storage
        self.current_metrics: Optional[PerformanceMetrics] = None
        self.metrics_history: List[PerformanceMetrics] = []
        self.max_history = 3600  # 1 hour at 1s intervals

        # Performance profiles
        self.profiles: Dict[str, List[float]] = {}
        self._profile_lock = threading.Lock()

        # Baseline metrics for comparison
        self.baseline_metrics: Optional[PerformanceMetrics] = None

        # Thresholds for alerts
        self.thresholds = {
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "disk_io_mb_per_sec": 100.0,
        }

        # Callbacks for threshold violations
        self.alert_callbacks: List[Callable] = []

        logger.info(
            f"PerformanceMonitor initialized (interval={sample_interval_seconds}s)"
        )

    async def start_monitoring(self):
        """Start continuous performance monitoring."""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop())

        # Capture baseline
        await asyncio.sleep(2)  # Let system settle
        self.baseline_metrics = await self._capture_metrics()

        logger.info("Performance monitoring started")

    async def stop_monitoring(self):
        """Stop performance monitoring."""
        self.is_monitoring = False

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        logger.info("Performance monitoring stopped")

    async def _monitoring_loop(self):
        """Main monitoring loop."""
        last_disk_io = psutil.disk_io_counters()
        last_network_io = psutil.net_io_counters()

        while self.is_monitoring:
            try:
                # Capture metrics
                metrics = await self._capture_metrics()

                # Calculate rates
                current_disk_io = psutil.disk_io_counters()
                current_network_io = psutil.net_io_counters()

                if last_disk_io:
                    metrics.disk_io_read_mb = (
                        (current_disk_io.read_bytes - last_disk_io.read_bytes)
                        / (1024 * 1024)
                        / self.sample_interval
                    )
                    metrics.disk_io_write_mb = (
                        (current_disk_io.write_bytes - last_disk_io.write_bytes)
                        / (1024 * 1024)
                        / self.sample_interval
                    )

                if last_network_io:
                    metrics.network_sent_mb = (
                        (current_network_io.bytes_sent - last_network_io.bytes_sent)
                        / (1024 * 1024)
                        / self.sample_interval
                    )
                    metrics.network_recv_mb = (
                        (current_network_io.bytes_recv - last_network_io.bytes_recv)
                        / (1024 * 1024)
                        / self.sample_interval
                    )

                last_disk_io = current_disk_io
                last_network_io = current_network_io

                # Store metrics
                self.current_metrics = metrics
                self.metrics_history.append(metrics)

                # Maintain history size
                if len(self.metrics_history) > self.max_history:
                    self.metrics_history = self.metrics_history[-self.max_history :]

                # Check thresholds
                await self._check_thresholds(metrics)

                # Sleep until next sample
                await asyncio.sleep(self.sample_interval)

            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(self.sample_interval)

    async def _capture_metrics(self) -> PerformanceMetrics:
        """Capture current performance metrics."""
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()

        # Process info
        process = psutil.Process()

        metrics = PerformanceMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_mb=memory.used / (1024 * 1024),
            disk_io_read_mb=0,  # Will be calculated in loop
            disk_io_write_mb=0,
            network_sent_mb=0,
            network_recv_mb=0,
            active_threads=threading.active_count(),
            open_files=len(process.open_files()),
        )

        # GPU metrics if available
        try:
            import torch

            if torch.cuda.is_available():
                metrics.gpu_percent = torch.cuda.utilization()
                metrics.gpu_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                # Temperature would require nvidia-ml-py
        except Exception as e:
            logger.error(f"Error in performance_monitor.py: {e}", exc_info=True)
            raise  # Re-raise for proper error handling

        return metrics

    async def _check_thresholds(self, metrics: PerformanceMetrics):
        """Check if any thresholds are violated."""
        violations = []

        if metrics.cpu_percent > self.thresholds["cpu_percent"]:
            violations.append(f"CPU usage: {metrics.cpu_percent:.1f}%")

        if metrics.memory_percent > self.thresholds["memory_percent"]:
            violations.append(f"Memory usage: {metrics.memory_percent:.1f}%")

        total_disk_io = metrics.disk_io_read_mb + metrics.disk_io_write_mb
        if total_disk_io > self.thresholds["disk_io_mb_per_sec"]:
            violations.append(f"Disk I/O: {total_disk_io:.1f} MB/s")

        if violations:
            for callback in self.alert_callbacks:
                try:
                    await callback(violations, metrics)
                except Exception as e:
                    logger.error(f"Alert callback error: {e}")

    @contextmanager
    def profile(self, name: str):
        """
        Context manager for profiling code blocks.

        Usage:
            with monitor.profile("database_query"):
                # Code to profile
        """
        start_time = time.time()

        try:
            yield
        finally:
            duration = time.time() - start_time

            with self._profile_lock:
                if name not in self.profiles:
                    self.profiles[name] = []

                self.profiles[name].append(duration)

                # Keep last 1000 measurements
                if len(self.profiles[name]) > 1000:
                    self.profiles[name] = self.profiles[name][-1000:]

    def profile_async(self, name: str):
        """
        Decorator for profiling async functions.

        Usage:
            @monitor.profile_async("api_endpoint")
            async def my_function():
                # Function code
        """

        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                with self.profile(name):
                    return await func(*args, **kwargs)

            return wrapper

        return decorator

    def profile_sync(self, name: str):
        """
        Decorator for profiling sync functions.

        Usage:
            @monitor.profile_sync("calculation")
            def my_function():
                # Function code
        """

        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with self.profile(name):
                    return func(*args, **kwargs)

            return wrapper

        return decorator

    def get_profile_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a profiled operation."""
        with self._profile_lock:
            if name not in self.profiles or not self.profiles[name]:
                return {}

            durations = self.profiles[name]

            return {
                "count": len(durations),
                "total": sum(durations),
                "mean": sum(durations) / len(durations),
                "min": min(durations),
                "max": max(durations),
                "last": durations[-1],
            }

    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get the most recent performance metrics."""
        return self.current_metrics

    def get_metrics_summary(self, window_seconds: int = 60) -> Dict[str, Any]:
        """Get summary of recent metrics."""
        if not self.metrics_history:
            return {}

        cutoff = time.time() - window_seconds
        recent = [m for m in self.metrics_history if m.timestamp >= cutoff]

        if not recent:
            return {}

        return {
            "window_seconds": window_seconds,
            "samples": len(recent),
            "cpu": {
                "mean": sum(m.cpu_percent for m in recent) / len(recent),
                "max": max(m.cpu_percent for m in recent),
                "min": min(m.cpu_percent for m in recent),
            },
            "memory": {
                "mean": sum(m.memory_percent for m in recent) / len(recent),
                "max": max(m.memory_percent for m in recent),
                "min": min(m.memory_percent for m in recent),
            },
            "disk_io": {
                "read_mb_per_sec": sum(m.disk_io_read_mb for m in recent) / len(recent),
                "write_mb_per_sec": sum(m.disk_io_write_mb for m in recent)
                / len(recent),
            },
            "network": {
                "sent_mb_per_sec": sum(m.network_sent_mb for m in recent) / len(recent),
                "recv_mb_per_sec": sum(m.network_recv_mb for m in recent) / len(recent),
            },
        }

    def detect_bottlenecks(self) -> List[str]:
        """Detect potential performance bottlenecks."""
        bottlenecks = []

        if not self.current_metrics or not self.baseline_metrics:
            return bottlenecks

        # CPU bottleneck
        if self.current_metrics.cpu_percent > 80:
            bottlenecks.append(
                f"High CPU usage: {self.current_metrics.cpu_percent:.1f}%"
            )

        # Memory bottleneck
        if self.current_metrics.memory_percent > 85:
            bottlenecks.append(
                f"High memory usage: {self.current_metrics.memory_percent:.1f}%"
            )

        # I/O bottleneck
        total_io = (
            self.current_metrics.disk_io_read_mb + self.current_metrics.disk_io_write_mb
        )
        if total_io > 50:  # 50 MB/s sustained
            bottlenecks.append(f"High disk I/O: {total_io:.1f} MB/s")

        # Thread explosion
        if self.current_metrics.active_threads > 100:
            bottlenecks.append(
                f"High thread count: {self.current_metrics.active_threads}"
            )

        # File descriptor leak
        baseline_files = self.baseline_metrics.open_files
        if self.current_metrics.open_files > baseline_files * 2:
            bottlenecks.append(
                f"Potential file descriptor leak: "
                f"{self.current_metrics.open_files} open files"
            )

        return bottlenecks


# Global performance monitor
_performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor
