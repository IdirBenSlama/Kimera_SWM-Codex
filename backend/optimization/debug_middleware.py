# -*- coding: utf-8 -*-
"""
Debug Middleware for Kimera
==========================
Provides request tracing and performance monitoring without impacting production performance.
"""

import time
import uuid
import json
from typing import Dict, Any, Optional, Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from collections import deque, defaultdict
from datetime import datetime
import asyncio

from .hybrid_logger import get_logger, log_performance

logger = get_logger("debug_middleware")


class RequestTracer:
    """Traces requests with minimal overhead"""
    
    def __init__(self, capacity: int = 1000):
        self.traces = deque(maxlen=capacity)
        self.active_requests = {}
        self.stats = defaultdict(lambda: {
            'count': 0,
            'total_time_ms': 0,
            'min_time_ms': float('inf'),
            'max_time_ms': 0,
            'avg_time_ms': 0,
            'errors': 0
        })
    
    def start_trace(self, request_id: str, request: Request) -> Dict[str, Any]:
        """Start tracing a request"""
        trace = {
            'id': request_id,
            'method': request.method,
            'path': str(request.url.path),
            'query': str(request.url.query) if request.url.query else None,
            'headers': dict(request.headers) if logger.debug_mode else None,
            'start_time': time.perf_counter(),
            'start_timestamp': datetime.utcnow().isoformat(),
            'client': f"{request.client.host}:{request.client.port}" if request.client else None
        }
        self.active_requests[request_id] = trace
        return trace
    
    def end_trace(self, request_id: str, status_code: int, 
                  response_headers: Optional[Dict] = None,
                  error: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """End tracing a request"""
        if request_id not in self.active_requests:
            return None
        
        trace = self.active_requests.pop(request_id)
        end_time = time.perf_counter()
        duration_ms = (end_time - trace['start_time']) * 1000
        
        trace.update({
            'end_time': end_time,
            'end_timestamp': datetime.utcnow().isoformat(),
            'duration_ms': duration_ms,
            'status_code': status_code,
            'response_headers': response_headers if logger.debug_mode else None,
            'error': error
        })
        
        # Update statistics
        path = trace['path']
        stats = self.stats[path]
        stats['count'] += 1
        stats['total_time_ms'] += duration_ms
        stats['min_time_ms'] = min(stats['min_time_ms'], duration_ms)
        stats['max_time_ms'] = max(stats['max_time_ms'], duration_ms)
        stats['avg_time_ms'] = stats['total_time_ms'] / stats['count']
        if error or status_code >= 400:
            stats['errors'] += 1
        
        # Store trace
        self.traces.append(trace)
        
        return trace
    
    def get_traces(self, limit: Optional[int] = None, 
                   path_filter: Optional[str] = None) -> list:
        """Get recent traces with optional filtering"""
        traces = list(self.traces)
        
        if path_filter:
            traces = [t for t in traces if path_filter in t['path']]
        
        if limit:
            traces = traces[-limit:]
        
        return traces
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get request statistics"""
        return {
            'endpoints': dict(self.stats),
            'total_requests': sum(s['count'] for s in self.stats.values()),
            'total_errors': sum(s['errors'] for s in self.stats.values()),
            'active_requests': len(self.active_requests),
            'trace_buffer_size': len(self.traces)
        }


class DebugMiddleware(BaseHTTPMiddleware):
    """Debug middleware with minimal performance impact"""
    
    def __init__(self, app: ASGIApp, tracer: RequestTracer, 
                 enable_profiling: bool = False):
        super().__init__(app)
        self.tracer = tracer
        self.enable_profiling = enable_profiling
        self.slow_request_threshold_ms = 100
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with optional debugging"""
        request_id = str(uuid.uuid4())
        
        # Skip tracing for health checks in performance mode
        if not logger.debug_mode and request.url.path in ['/health', '/']:
            return await call_next(request)
        
        # Start trace
        trace = self.tracer.start_trace(request_id, request)
        
        # Add request ID to headers for correlation
        request.state.request_id = request_id
        
        try:
            # Process request
            response = await call_next(request)
            
            # End trace
            final_trace = self.tracer.end_trace(
                request_id, 
                response.status_code,
                dict(response.headers) if logger.debug_mode else None
            )
            
            # Log slow requests
            if final_trace and final_trace['duration_ms'] > self.slow_request_threshold_ms:
                log_performance(
                    f"Slow request: {request.method} {request.url.path}",
                    final_trace['duration_ms'],
                    request_id=request_id,
                    status_code=response.status_code
                )
            
            # Add debug headers if enabled
            if logger.debug_mode:
                response.headers['X-Request-ID'] = request_id
                response.headers['X-Process-Time-MS'] = f"{final_trace['duration_ms']:.2f}"
            
            return response
            
        except Exception as e:
            # Log error and end trace
            error_msg = f"{type(e).__name__}: {str(e)}"
            self.tracer.end_trace(request_id, 500, error=error_msg)
            logger.log_with_context(
                'ERROR',
                f"Request failed: {error_msg}",
                request_id=request_id,
                path=request.url.path,
                method=request.method
            )
            raise


class PerformanceProfiler:
    """Optional performance profiler for detailed analysis"""
    
    def __init__(self):
        self.profiles = deque(maxlen=100)
        self.enabled = False
        
    async def profile_async(self, func: Callable, *args, **kwargs):
        """Profile an async function"""
        if not self.enabled:
            return await func(*args, **kwargs)
        
        start_time = time.perf_counter()
        start_memory = self._get_memory_usage()
        
        try:
            result = await func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            success = False
            error = str(e)
            raise
        finally:
            end_time = time.perf_counter()
            end_memory = self._get_memory_usage()
            
            profile = {
                'function': func.__name__,
                'module': func.__module__,
                'timestamp': datetime.utcnow().isoformat(),
                'duration_ms': (end_time - start_time) * 1000,
                'memory_delta_mb': (end_memory - start_memory) / 1024 / 1024,
                'success': success,
                'error': error
            }
            
            self.profiles.append(profile)
            
        return result
    
    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss
        except ImportError:
            return 0
    
    def get_profiles(self, limit: Optional[int] = None) -> list:
        """Get recent profiles"""
        profiles = list(self.profiles)
        if limit:
            profiles = profiles[-limit:]
        return profiles
    
    def enable(self):
        """Enable profiling"""
        self.enabled = True
        logger.log_with_context('INFO', 'Performance profiling enabled')
    
    def disable(self):
        """Disable profiling"""
        self.enabled = False
        logger.log_with_context('INFO', 'Performance profiling disabled')


# Global instances
request_tracer = RequestTracer(capacity=5000)
performance_profiler = PerformanceProfiler()


# Debug API endpoints
async def get_debug_info() -> Dict[str, Any]:
    """Get comprehensive debug information"""
    return {
        'logging': logger.get_performance_metrics(),
        'recent_logs': logger.get_recent_logs(limit=50),
        'request_traces': request_tracer.get_traces(limit=50),
        'request_statistics': request_tracer.get_statistics(),
        'performance_profiles': performance_profiler.get_profiles(limit=20),
        'debug_mode': logger.debug_mode,
        'profiling_enabled': performance_profiler.enabled
    }


async def set_debug_mode(enabled: bool) -> Dict[str, str]:
    """Enable or disable debug mode at runtime"""
    if enabled:
        logger.enable_debug_mode()
        return {"status": "Debug mode enabled"}
    else:
        logger.disable_debug_mode()
        return {"status": "Debug mode disabled"}


async def set_profiling(enabled: bool) -> Dict[str, str]:
    """Enable or disable performance profiling"""
    if enabled:
        performance_profiler.enable()
        return {"status": "Profiling enabled"}
    else:
        performance_profiler.disable()
        return {"status": "Profiling disabled"}