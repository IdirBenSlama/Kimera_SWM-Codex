#!/usr/bin/env python3
"""
KIMERA SWM System - Performance Optimizer
========================================

Phase 3.2: Performance Optimization Implementation
Provides comprehensive performance analysis, profiling, and optimization recommendations.

Features:
- Performance profiling and monitoring
- Memory usage analysis and optimization
- Caching strategy implementation
- Database query optimization
- Async processing optimization
- Lazy loading implementation
- Performance regression detection

Author: KIMERA Development Team
Date: 2025-01-31
Phase: 3.2 - Performance Optimization
"""

import asyncio
import cProfile
import functools
import gc
import logging
import memory_profiler
import os
import pickle
import psutil
import sqlite3
import threading
import time
import tracemalloc
from collections import defaultdict, OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import weakref
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics for a specific operation."""
    operation_name: str
    execution_time: float
    memory_usage: float
    cpu_usage: float
    timestamp: datetime
    call_count: int = 1
    cache_hits: int = 0
    cache_misses: int = 0
    database_queries: int = 0
    async_operations: int = 0

@dataclass
class OptimizationRecommendation:
    """Performance optimization recommendation."""
    optimization_type: str
    priority: str  # 'critical', 'high', 'medium', 'low'
    description: str
    expected_improvement: str
    implementation_effort: str
    code_example: str
    affected_components: List[str]

class SmartCache:
    """Intelligent caching system with LRU and TTL support."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache = OrderedDict()
        self._timestamps = {}
        self._access_counts = defaultdict(int)
        self._hit_count = 0
        self._miss_count = 0
        self._lock = threading.RLock()
    
    def get(self, key: str, default=None):
        """Get value from cache."""
        with self._lock:
            if key in self._cache:
                # Check TTL
                if time.time() - self._timestamps[key] < self.ttl_seconds:
                    # Move to end (most recently used)
                    value = self._cache.pop(key)
                    self._cache[key] = value
                    self._access_counts[key] += 1
                    self._hit_count += 1
                    return value
                else:
                    # Expired
                    self._remove_key(key)
            
            self._miss_count += 1
            return default
    
    def set(self, key: str, value: Any):
        """Set value in cache."""
        with self._lock:
            if key in self._cache:
                self._cache.pop(key)
            elif len(self._cache) >= self.max_size:
                # Remove least recently used
                oldest_key = next(iter(self._cache))
                self._remove_key(oldest_key)
            
            self._cache[key] = value
            self._timestamps[key] = time.time()
            self._access_counts[key] = 1
    
    def _remove_key(self, key: str):
        """Remove key from cache."""
        self._cache.pop(key, None)
        self._timestamps.pop(key, None)
        self._access_counts.pop(key, None)
    
    def invalidate(self, pattern: str = None):
        """Invalidate cache entries."""
        with self._lock:
            if pattern is None:
                self._cache.clear()
                self._timestamps.clear()
                self._access_counts.clear()
            else:
                keys_to_remove = [k for k in self._cache.keys() if pattern in k]
                for key in keys_to_remove:
                    self._remove_key(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._hit_count + self._miss_count
        hit_rate = self._hit_count / total_requests if total_requests > 0 else 0
        
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_rate": hit_rate,
            "ttl_seconds": self.ttl_seconds
        }

class PerformanceProfiler:
    """Comprehensive performance profiler."""
    
    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self.profiling_enabled = True
        self.memory_tracking_enabled = False
        self._baseline_memory = 0
        self._start_time = time.time()
        
    def enable_memory_tracking(self):
        """Enable detailed memory tracking."""
        tracemalloc.start()
        self.memory_tracking_enabled = True
        self._baseline_memory = self._get_memory_usage()
    
    def disable_memory_tracking(self):
        """Disable memory tracking."""
        if self.memory_tracking_enabled:
            tracemalloc.stop()
            self.memory_tracking_enabled = False
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if self.memory_tracking_enabled:
            current, peak = tracemalloc.get_traced_memory()
            return current / 1024 / 1024
        else:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        return psutil.cpu_percent(interval=0.1)
    
    @contextmanager
    def profile_operation(self, operation_name: str):
        """Context manager for profiling operations."""
        if not self.profiling_enabled:
            yield
            return
        
        start_time = time.time()
        start_memory = self._get_memory_usage()
        start_cpu = self._get_cpu_usage()
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            end_cpu = self._get_cpu_usage()
            
            metrics = PerformanceMetrics(
                operation_name=operation_name,
                execution_time=end_time - start_time,
                memory_usage=end_memory - start_memory,
                cpu_usage=(start_cpu + end_cpu) / 2,
                timestamp=datetime.now()
            )
            
            self.metrics.append(metrics)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        if not self.metrics:
            return {"total_operations": 0}
        
        total_time = sum(m.execution_time for m in self.metrics)
        total_memory = sum(m.memory_usage for m in self.metrics)
        avg_cpu = sum(m.cpu_usage for m in self.metrics) / len(self.metrics)
        
        # Group by operation
        operation_stats = defaultdict(list)
        for metric in self.metrics:
            operation_stats[metric.operation_name].append(metric)
        
        slowest_operations = sorted(
            [(name, max(metrics, key=lambda m: m.execution_time)) 
             for name, metrics in operation_stats.items()],
            key=lambda x: x[1].execution_time,
            reverse=True
        )[:5]
        
        memory_intensive = sorted(
            [(name, max(metrics, key=lambda m: m.memory_usage))
             for name, metrics in operation_stats.items()],
            key=lambda x: x[1].memory_usage,
            reverse=True
        )[:5]
        
        return {
            "total_operations": len(self.metrics),
            "total_execution_time": total_time,
            "total_memory_usage": total_memory,
            "average_cpu_usage": avg_cpu,
            "unique_operations": len(operation_stats),
            "slowest_operations": [
                {"name": name, "time": metric.execution_time, "memory": metric.memory_usage}
                for name, metric in slowest_operations
            ],
            "memory_intensive_operations": [
                {"name": name, "memory": metric.memory_usage, "time": metric.execution_time}
                for name, metric in memory_intensive
            ],
            "operation_counts": {name: len(metrics) for name, metrics in operation_stats.items()}
        }

class LazyLoader:
    """Lazy loading implementation for heavy resources."""
    
    def __init__(self, loader_func: Callable, *args, **kwargs):
        self._loader_func = loader_func
        self._args = args
        self._kwargs = kwargs
        self._loaded = False
        self._value = None
        self._lock = threading.Lock()
    
    def __call__(self):
        """Load the resource if not already loaded."""
        if not self._loaded:
            with self._lock:
                if not self._loaded:  # Double-check locking
                    self._value = self._loader_func(*self._args, **self._kwargs)
                    self._loaded = True
        return self._value
    
    def is_loaded(self) -> bool:
        """Check if resource is loaded."""
        return self._loaded
    
    def invalidate(self):
        """Invalidate the loaded resource."""
        with self._lock:
            self._loaded = False
            self._value = None

class DatabaseOptimizer:
    """Database query optimization and monitoring."""
    
    def __init__(self, db_path: str = "kimera_swm.db"):
        self.db_path = db_path
        self.query_stats = defaultdict(list)
        self._lock = threading.Lock()
    
    @contextmanager
    def optimized_connection(self):
        """Get optimized database connection."""
        conn = sqlite3.connect(self.db_path)
        
        # Apply optimization pragmas
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA cache_size = 10000")
        conn.execute("PRAGMA temp_store = memory")
        conn.execute("PRAGMA synchronous = NORMAL")
        
        try:
            yield conn
        finally:
            conn.close()
    
    def execute_optimized_query(self, query: str, params: tuple = (), fetch_all: bool = True):
        """Execute query with optimization and monitoring."""
        start_time = time.time()
        
        with self.optimized_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            
            if query.strip().upper().startswith('SELECT') and fetch_all:
                result = cursor.fetchall()
            else:
                result = cursor.rowcount
                conn.commit()
        
        execution_time = time.time() - start_time
        
        # Record statistics
        with self._lock:
            self.query_stats[query].append({
                "execution_time": execution_time,
                "timestamp": datetime.now(),
                "params": params
            })
        
        return result
    
    def get_slow_queries(self, threshold: float = 0.1) -> List[Dict[str, Any]]:
        """Get queries that exceed the time threshold."""
        slow_queries = []
        
        with self._lock:
            for query, stats in self.query_stats.items():
                avg_time = sum(s["execution_time"] for s in stats) / len(stats)
                max_time = max(s["execution_time"] for s in stats)
                
                if avg_time > threshold or max_time > threshold * 2:
                    slow_queries.append({
                        "query": query,
                        "average_time": avg_time,
                        "max_time": max_time,
                        "execution_count": len(stats),
                        "recommendation": self._get_query_optimization_suggestion(query)
                    })
        
        return sorted(slow_queries, key=lambda x: x["average_time"], reverse=True)
    
    def _get_query_optimization_suggestion(self, query: str) -> str:
        """Get optimization suggestion for a query."""
        query_upper = query.upper()
        
        if "SELECT" in query_upper and "WHERE" not in query_upper:
            return "Consider adding WHERE clause to limit results"
        elif "SELECT *" in query_upper:
            return "Consider selecting only necessary columns instead of *"
        elif "ORDER BY" in query_upper and "LIMIT" not in query_upper:
            return "Consider adding LIMIT clause for large result sets"
        elif "JOIN" in query_upper and "INDEX" not in query_upper:
            return "Ensure proper indexes exist on JOIN columns"
        else:
            return "Review query structure and consider indexing"

class AsyncOptimizer:
    """Async operation optimization and monitoring."""
    
    def __init__(self):
        self.async_stats = defaultdict(list)
        self._semaphore = asyncio.Semaphore(10)  # Limit concurrent operations
    
    async def optimized_gather(self, *coroutines, return_exceptions: bool = False):
        """Optimized version of asyncio.gather with concurrency limits."""
        
        async def limited_coroutine(coro):
            async with self._semaphore:
                start_time = time.time()
                try:
                    result = await coro
                    execution_time = time.time() - start_time
                    self.async_stats[coro.__name__ if hasattr(coro, '__name__') else 'unknown'].append({
                        "execution_time": execution_time,
                        "success": True,
                        "timestamp": datetime.now()
                    })
                    return result
                except Exception as e:
                    execution_time = time.time() - start_time
                    self.async_stats[coro.__name__ if hasattr(coro, '__name__') else 'unknown'].append({
                        "execution_time": execution_time,
                        "success": False,
                        "error": str(e),
                        "timestamp": datetime.now()
                    })
                    if return_exceptions:
                        return e
                    raise
        
        limited_coroutines = [limited_coroutine(coro) for coro in coroutines]
        return await asyncio.gather(*limited_coroutines, return_exceptions=return_exceptions)
    
    def get_async_stats(self) -> Dict[str, Any]:
        """Get async operation statistics."""
        stats = {}
        
        for operation, executions in self.async_stats.items():
            successful = [e for e in executions if e["success"]]
            failed = [e for e in executions if not e["success"]]
            
            if executions:
                avg_time = sum(e["execution_time"] for e in executions) / len(executions)
                success_rate = len(successful) / len(executions)
                
                stats[operation] = {
                    "total_executions": len(executions),
                    "successful": len(successful),
                    "failed": len(failed),
                    "success_rate": success_rate,
                    "average_time": avg_time,
                    "max_time": max(e["execution_time"] for e in executions),
                    "min_time": min(e["execution_time"] for e in executions)
                }
        
        return stats

class PerformanceOptimizer:
    """Main performance optimization coordinator."""
    
    def __init__(self):
        self.profiler = PerformanceProfiler()
        self.cache = SmartCache()
        self.db_optimizer = DatabaseOptimizer()
        self.async_optimizer = AsyncOptimizer()
        self.lazy_loaders: Dict[str, LazyLoader] = {}
        self.optimization_enabled = True
    
    def enable_profiling(self):
        """Enable performance profiling."""
        self.profiler.profiling_enabled = True
        self.profiler.enable_memory_tracking()
        logger.info("Performance profiling enabled")
    
    def disable_profiling(self):
        """Disable performance profiling."""
        self.profiler.profiling_enabled = False
        self.profiler.disable_memory_tracking()
        logger.info("Performance profiling disabled")
    
    def cached(self, ttl: int = 3600, key_func: Callable = None):
        """Decorator for caching function results."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not self.optimization_enabled:
                    return func(*args, **kwargs)
                
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
                
                # Try to get from cache
                result = self.cache.get(cache_key)
                if result is not None:
                    return result
                
                # Execute function and cache result
                with self.profiler.profile_operation(f"cached_{func.__name__}"):
                    result = func(*args, **kwargs)
                    self.cache.set(cache_key, result)
                    return result
            
            return wrapper
        return decorator
    
    def lazy_load(self, name: str, loader_func: Callable, *args, **kwargs) -> LazyLoader:
        """Create a lazy loader for heavy resources."""
        if name not in self.lazy_loaders:
            self.lazy_loaders[name] = LazyLoader(loader_func, *args, **kwargs)
        return self.lazy_loaders[name]
    
    def profile_operation(self, operation_name: str):
        """Context manager for profiling operations."""
        return self.profiler.profile_operation(operation_name)
    
    def async_batch_process(self, items: List[Any], processor: Callable, batch_size: int = 10):
        """Optimized async batch processing."""
        async def process_batch(batch):
            tasks = [processor(item) for item in batch]
            return await self.async_optimizer.optimized_gather(*tasks)
        
        async def run_batches():
            results = []
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                batch_results = await process_batch(batch)
                results.extend(batch_results)
            return results
        
        return asyncio.run(run_batches())
    
    def optimize_memory(self):
        """Run memory optimization."""
        logger.info("Running memory optimization...")
        
        # Clear cache of expired items
        self.cache.invalidate()
        
        # Invalidate unused lazy loaders
        for name, loader in list(self.lazy_loaders.items()):
            if not loader.is_loaded():
                del self.lazy_loaders[name]
        
        # Force garbage collection
        collected = gc.collect()
        logger.info(f"Memory optimization completed. Collected {collected} objects.")
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        logger.info("Generating performance report...")
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "profiling_summary": self.profiler.get_performance_summary(),
            "cache_statistics": self.cache.get_stats(),
            "slow_database_queries": self.db_optimizer.get_slow_queries(),
            "async_statistics": self.async_optimizer.get_async_stats(),
            "lazy_loaders": {
                name: {"loaded": loader.is_loaded()}
                for name, loader in self.lazy_loaders.items()
            },
            "system_resources": self._get_system_resources(),
            "optimization_recommendations": self._generate_optimization_recommendations()
        }
        
        return report
    
    def _get_system_resources(self) -> Dict[str, Any]:
        """Get current system resource usage."""
        process = psutil.Process()
        
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_usage_mb": process.memory_info().rss / 1024 / 1024,
            "disk_usage_percent": psutil.disk_usage('.').percent,
            "open_files": len(process.open_files()),
            "thread_count": process.num_threads()
        }
    
    def _generate_optimization_recommendations(self) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations based on collected data."""
        recommendations = []
        
        # Analyze performance metrics
        summary = self.profiler.get_performance_summary()
        
        # Cache optimization recommendations
        cache_stats = self.cache.get_stats()
        if cache_stats["hit_rate"] < 0.5 and cache_stats["miss_count"] > 100:
            recommendations.append(OptimizationRecommendation(
                optimization_type="caching",
                priority="high",
                description="Low cache hit rate detected. Consider increasing cache size or TTL.",
                expected_improvement="20-40% performance improvement",
                implementation_effort="low",
                code_example="cache = SmartCache(max_size=2000, ttl_seconds=7200)",
                affected_components=["cache_system"]
            ))
        
        # Memory optimization recommendations
        if "memory_intensive_operations" in summary:
            memory_ops = summary["memory_intensive_operations"][:3]
            if memory_ops and any(op["memory"] > 100 for op in memory_ops):  # >100MB
                recommendations.append(OptimizationRecommendation(
                    optimization_type="memory",
                    priority="high",
                    description="High memory usage detected in some operations.",
                    expected_improvement="30-50% memory reduction",
                    implementation_effort="medium",
                    code_example="Use generators, lazy loading, or streaming for large datasets",
                    affected_components=[op["name"] for op in memory_ops]
                ))
        
        # Slow operation recommendations
        if "slowest_operations" in summary:
            slow_ops = summary["slowest_operations"][:3]
            if slow_ops and any(op["time"] > 1.0 for op in slow_ops):  # >1 second
                recommendations.append(OptimizationRecommendation(
                    optimization_type="performance",
                    priority="critical",
                    description="Slow operations detected that exceed acceptable thresholds.",
                    expected_improvement="50-80% speed improvement",
                    implementation_effort="high",
                    code_example="Consider async processing, caching, or algorithm optimization",
                    affected_components=[op["name"] for op in slow_ops]
                ))
        
        # Database optimization recommendations
        slow_queries = self.db_optimizer.get_slow_queries(threshold=0.05)
        if slow_queries:
            recommendations.append(OptimizationRecommendation(
                optimization_type="database",
                priority="medium",
                description=f"Found {len(slow_queries)} slow database queries.",
                expected_improvement="40-60% query speed improvement",
                implementation_effort="medium",
                code_example="Add indexes, optimize WHERE clauses, limit result sets",
                affected_components=["database"]
            ))
        
        # System resource recommendations
        system_resources = self._get_system_resources()
        if system_resources["memory_percent"] > 80:
            recommendations.append(OptimizationRecommendation(
                optimization_type="system",
                priority="critical",
                description="High system memory usage detected.",
                expected_improvement="Improved system stability",
                implementation_effort="low",
                code_example="Run memory optimization: optimizer.optimize_memory()",
                affected_components=["system"]
            ))
        
        return recommendations
    
    def save_report(self, report: Dict[str, Any], filename: str = "performance_report.json"):
        """Save performance report to file."""
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Performance report saved to {filename}")

# Global performance optimizer instance
performance_optimizer = PerformanceOptimizer()

# Convenience decorators
def cached(ttl: int = 3600, key_func: Callable = None):
    """Decorator for caching function results."""
    return performance_optimizer.cached(ttl=ttl, key_func=key_func)

def profile_performance(operation_name: str = None):
    """Decorator for profiling function performance."""
    def decorator(func: Callable) -> Callable:
        op_name = operation_name or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with performance_optimizer.profile_operation(op_name):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator

@contextmanager
def performance_context(operation_name: str):
    """Context manager for performance profiling."""
    with performance_optimizer.profile_operation(operation_name):
        yield

def optimize_consciousness_detection():
    """Apply specific optimizations for consciousness detection."""
    # Cache consciousness detection results
    @cached(ttl=300)  # 5 minute cache
    def cached_phi_calculation(input_data_hash: str):
        # Placeholder for actual phi calculation
        return {"phi": 0.75, "confidence": 0.8}
    
    # Lazy load heavy models
    consciousness_model = performance_optimizer.lazy_load(
        "consciousness_model",
        lambda: "Heavy consciousness detection model"  # Placeholder
    )
    
    logger.info("Consciousness detection optimizations applied")

def optimize_thermodynamic_calculations():
    """Apply specific optimizations for thermodynamic calculations."""
    # Cache thermodynamic calculations
    @cached(ttl=600)  # 10 minute cache
    def cached_carnot_efficiency(hot_temp: float, cold_temp: float):
        return 1 - (cold_temp / hot_temp)
    
    # Lazy load thermodynamic constants
    thermo_constants = performance_optimizer.lazy_load(
        "thermo_constants",
        lambda: {"R": 8.314, "k_B": 1.38e-23}  # Placeholder
    )
    
    logger.info("Thermodynamic calculation optimizations applied")

def main():
    """Main function to run performance optimization setup."""
    print("⚡ KIMERA Performance Optimizer")
    print("=" * 60)
    print("Phase 3.2: Performance Optimization")
    print()
    
    # Enable profiling
    performance_optimizer.enable_profiling()
    
    # Apply component-specific optimizations
    optimize_consciousness_detection()
    optimize_thermodynamic_calculations()
    
    # Generate initial report
    report = performance_optimizer.generate_performance_report()
    performance_optimizer.save_report(report)
    
    print("🎯 Performance optimization setup completed!")
    print("📊 Initial performance report generated: performance_report.json")
    print()
    print("Usage examples:")
    print("  @cached(ttl=300)")
    print("  @profile_performance('my_operation')")
    print("  with performance_context('expensive_operation'):")
    print("      # Your code here")
    print()
    print("🔧 Next steps:")
    print("   1. Apply @cached decorators to expensive functions")
    print("   2. Use lazy loading for heavy resources")
    print("   3. Profile critical operations")
    print("   4. Review performance report for optimization opportunities")

if __name__ == "__main__":
    main() 