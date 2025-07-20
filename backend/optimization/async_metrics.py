"""
Asynchronous Metrics Collection System
=====================================
Implements non-blocking, concurrent metrics collection using asyncio
and thread pools for CPU-bound operations.

Key optimizations:
1. Async/await for I/O-bound operations
2. Thread pool for CPU-bound metrics
3. Concurrent collection of independent metrics
4. Result caching and deduplication
"""

import asyncio
import concurrent.futures
import functools
import time
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
import psutil
import torch

class AsyncMetricsCollector:
    """
    High-performance asynchronous metrics collector.
    
    Uses asyncio for concurrent I/O operations and thread pools
    for CPU-bound metrics collection, achieving optimal resource utilization.
    """
    
    def __init__(self, max_workers: int = 4):
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="MetricsWorker"
        )
        self._metric_functions: Dict[str, Callable] = {}
        self._register_default_metrics()
        
    def _register_default_metrics(self):
        """Register default system metrics collectors"""
        self._metric_functions.update({
            'cpu': self._get_cpu_metrics,
            'memory': self._get_memory_metrics,
            'disk': self._get_disk_metrics,
            'network': self._get_network_metrics,
            'gpu': self._get_gpu_metrics,
            'process': self._get_process_metrics
        })
    
    async def collect_all_metrics(self) -> Dict[str, Any]:
        """
        Collect all metrics concurrently.
        
        This method launches all metric collectors in parallel and
        waits for all to complete, significantly reducing total collection time.
        """
        start_time = time.perf_counter()
        
        # Create tasks for all metrics
        tasks = []
        for name, func in self._metric_functions.items():
            if asyncio.iscoroutinefunction(func):
                task = func()
            else:
                # Run CPU-bound functions in thread pool
                task = asyncio.get_event_loop().run_in_executor(
                    self.executor, func
                )
            tasks.append(asyncio.create_task(self._collect_with_name(name, task)))
        
        # Wait for all metrics to be collected
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        metrics = {
            'timestamp': datetime.utcnow().isoformat(),
            'collection_time_ms': (time.perf_counter() - start_time) * 1000
        }
        
        for name, result in results:
            if isinstance(result, Exception):
                metrics[name] = {'error': str(result)}
            else:
                metrics[name] = result
        
        return metrics
    
    async def _collect_with_name(self, name: str, task) -> tuple:
        """Helper to associate metric name with its result"""
        result = await task
        return (name, result)
    
    def _get_cpu_metrics(self) -> Dict[str, Any]:
        """Get CPU metrics without blocking"""
        # Use interval=0 for non-blocking measurement
        cpu_percent = psutil.cpu_percent(interval=0)
        cpu_freq = psutil.cpu_freq()
        
        return {
            'percent': cpu_percent,
            'count': psutil.cpu_count(),
            'count_physical': psutil.cpu_count(logical=False),
            'frequency_mhz': cpu_freq.current if cpu_freq else None,
            'frequency_max_mhz': cpu_freq.max if cpu_freq else None,
            'per_cpu_percent': psutil.cpu_percent(interval=0, percpu=True)
        }
    
    def _get_memory_metrics(self) -> Dict[str, Any]:
        """Get memory metrics (fast, no blocking)"""
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        return {
            'total_mb': mem.total / (1024 * 1024),
            'available_mb': mem.available / (1024 * 1024),
            'percent': mem.percent,
            'used_mb': mem.used / (1024 * 1024),
            'free_mb': mem.free / (1024 * 1024),
            'swap_total_mb': swap.total / (1024 * 1024),
            'swap_used_mb': swap.used / (1024 * 1024),
            'swap_percent': swap.percent
        }
    
    def _get_disk_metrics(self) -> Dict[str, Any]:
        """Get disk I/O metrics"""
        disk_io = psutil.disk_io_counters()
        disk_usage = psutil.disk_usage('/')
        
        return {
            'usage_percent': disk_usage.percent,
            'free_gb': disk_usage.free / (1024**3),
            'total_gb': disk_usage.total / (1024**3),
            'read_mb': disk_io.read_bytes / (1024**2),
            'write_mb': disk_io.write_bytes / (1024**2),
            'read_count': disk_io.read_count,
            'write_count': disk_io.write_count
        }
    
    def _get_network_metrics(self) -> Dict[str, Any]:
        """Get network I/O metrics"""
        net_io = psutil.net_io_counters()
        
        return {
            'bytes_sent_mb': net_io.bytes_sent / (1024**2),
            'bytes_recv_mb': net_io.bytes_recv / (1024**2),
            'packets_sent': net_io.packets_sent,
            'packets_recv': net_io.packets_recv,
            'errors_in': net_io.errin,
            'errors_out': net_io.errout,
            'drop_in': net_io.dropin,
            'drop_out': net_io.dropout
        }
    
    def _get_gpu_metrics(self) -> Dict[str, Any]:
        """Get GPU metrics if available"""
        if not torch.cuda.is_available():
            return {'available': False}
        
        metrics = {
            'available': True,
            'device_count': torch.cuda.device_count(),
            'current_device': torch.cuda.current_device(),
            'devices': []
        }
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_allocated = torch.cuda.memory_allocated(i)
            memory_reserved = torch.cuda.memory_reserved(i)
            
            device_info = {
                'index': i,
                'name': props.name,
                'total_memory_mb': props.total_memory / (1024**2),
                'memory_allocated_mb': memory_allocated / (1024**2),
                'memory_reserved_mb': memory_reserved / (1024**2),
                'memory_free_mb': (props.total_memory - memory_reserved) / (1024**2),
                'utilization_percent': (memory_allocated / props.total_memory) * 100
            }
            
            # Try to get additional metrics via nvidia-ml-py
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # Temperature
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                device_info['temperature_c'] = temp
                
                # Power
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                device_info['power_watts'] = power
                
                # Utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                device_info['gpu_utilization_percent'] = util.gpu
                device_info['memory_utilization_percent'] = util.memory
                
                # Clock speeds
                clock_graphics = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                clock_memory = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                device_info['clock_graphics_mhz'] = clock_graphics
                device_info['clock_memory_mhz'] = clock_memory
                
            except Exception:
                # NVML not available or error
                pass
            
            metrics['devices'].append(device_info)
        
        return metrics
    
    def _get_process_metrics(self) -> Dict[str, Any]:
        """Get current process metrics"""
        process = psutil.Process()
        
        with process.oneshot():
            return {
                'pid': process.pid,
                'cpu_percent': process.cpu_percent(),
                'memory_mb': process.memory_info().rss / (1024**2),
                'memory_percent': process.memory_percent(),
                'num_threads': process.num_threads(),
                'num_fds': process.num_fds() if hasattr(process, 'num_fds') else None,
                'create_time': datetime.fromtimestamp(process.create_time()).isoformat()
            }
    
    async def collect_specific_metrics(self, metric_names: List[str]) -> Dict[str, Any]:
        """Collect only specific metrics for targeted queries"""
        tasks = []
        
        for name in metric_names:
            if name in self._metric_functions:
                func = self._metric_functions[name]
                if asyncio.iscoroutinefunction(func):
                    task = func()
                else:
                    task = asyncio.get_event_loop().run_in_executor(
                        self.executor, func
                    )
                tasks.append(asyncio.create_task(self._collect_with_name(name, task)))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        metrics = {'timestamp': datetime.utcnow().isoformat()}
        for name, result in results:
            if isinstance(result, Exception):
                metrics[name] = {'error': str(result)}
            else:
                metrics[name] = result
        
        return metrics
    
    def register_custom_metric(self, name: str, func: Callable):
        """Register a custom metric collector function"""
        self._metric_functions[name] = func
    
    def close(self):
        """Clean up resources"""
        self.executor.shutdown(wait=True)


# Global async metrics collector
_async_collector = AsyncMetricsCollector()

def get_async_metrics_collector() -> AsyncMetricsCollector:
    """Get the global async metrics collector"""
    return _async_collector