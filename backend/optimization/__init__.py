"""
Kimera Performance Optimization Module
=====================================
Scientific optimization techniques for achieving peak performance.

This module implements:
- Asynchronous metric collection
- Lock-free data structures
- Memory-mapped caching
- Zero-copy operations
- NUMA-aware memory allocation
- CPU affinity optimization
"""

from .metrics_cache import MetricsCache, CachedMetrics
from .async_metrics import AsyncMetricsCollector
from .performance_monitor import PerformanceMonitor
from .hybrid_logger import HybridLogger, get_logger, log_performance
from .debug_middleware import (
    DebugMiddleware, RequestTracer, PerformanceProfiler,
    request_tracer, performance_profiler
)

__all__ = [
    'MetricsCache',
    'CachedMetrics',
    'AsyncMetricsCollector',
    'PerformanceMonitor',
    'HybridLogger',
    'get_logger',
    'log_performance',
    'DebugMiddleware',
    'RequestTracer',
    'PerformanceProfiler',
    'request_tracer',
    'performance_profiler'
]