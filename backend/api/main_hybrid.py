# -*- coding: utf-8 -*-
"""
Hybrid FastAPI application for Kimera SWM
========================================
Combines high performance with comprehensive debugging capabilities.

Features:
- All optimizations from main_optimized.py
- Hybrid logging system with ring buffer
- Request tracing and performance profiling
- Runtime-adjustable debug mode
- Debug dashboard and API endpoints
"""

import logging
import asyncio
import time
import os
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request, Response, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
try:
    from fastapi.responses import ORJSONResponse
except ImportError:
    ORJSONResponse = JSONResponse
from contextlib import asynccontextmanager

# Import hybrid components
from ..optimization.hybrid_logger import hybrid_logger, get_logger, log_performance
from ..optimization.debug_middleware import (
    DebugMiddleware, request_tracer, performance_profiler,
    get_debug_info, set_debug_mode, set_profiling
)

# Import optimized components
from ..optimization import MetricsCache, AsyncMetricsCollector
from ..core.kimera_system import kimera_singleton

# Import new components
from backend.core.kimera_system import get_kimera_system
from backend.utils.config import get_api_settings, Config
from backend.monitoring.kimera_prometheus_metrics import get_kimera_metrics, initialize_background_collection
from backend.api.middleware.error_handling import http_error_handler
from backend.api.middleware.request_logging import RequestLoggingMiddleware
from backend.utils.threading_utils import start_background_task

# Configure logger
logger = get_logger("main_hybrid")

# Global instances
metrics_cache = MetricsCache(cache_duration_seconds=1)
async_collector = AsyncMetricsCollector()

# Response cache
response_cache: Dict[str, tuple[Any, float]] = {}
CACHE_TTL = 1.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Hybrid lifespan manager with debug capabilities"""
    start_time = time.perf_counter()
    
    logger.log_with_context('INFO', 'Starting Kimera Hybrid API', 
                           debug_mode=logger.debug_mode,
                           performance_mode=logger.performance_mode)
    
    # Start metrics collection
    metrics_cache.start()
    logger.log_with_context('INFO', 'Metrics cache started')
    
    # Initialize Kimera system
    init_task = asyncio.create_task(initialize_kimera_async())
    
    # Wait for initialization
    await init_task
    
    init_time = (time.perf_counter() - start_time) * 1000
    log_performance('Application initialization', init_time)
    
    yield
    
    # Cleanup
    logger.log_with_context('INFO', 'Shutting down Kimera Hybrid API')
    metrics_cache.stop()
    async_collector.close()
    kimera_singleton.shutdown()
    hybrid_logger.shutdown()
    logger.log_with_context('INFO', 'Shutdown complete')


async def initialize_kimera_async():
    """Initialize Kimera system with performance tracking"""
    start_time = time.perf_counter()
    loop = asyncio.get_event_loop()
    
    if performance_profiler.enabled:
        await performance_profiler.profile_async(
            loop.run_in_executor, None, kimera_singleton.initialize
        )
    else:
        await loop.run_in_executor(None, kimera_singleton.initialize)
    
    duration = (time.perf_counter() - start_time) * 1000
    log_performance('Kimera System initialization', duration)


def setup_middleware(app: FastAPI):
    """Configure middleware with debug capabilities"""
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        max_age=3600
    )
    
    # Compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Debug middleware
    app.add_middleware(
        DebugMiddleware,
        tracer=request_tracer,
        enable_profiling=performance_profiler.enabled
    )


# Create hybrid FastAPI app
app = FastAPI(
    title="Kimera SWM API (Hybrid)",
    description="High-performance API with comprehensive debugging capabilities",
    version="0.3.0",
    lifespan=lifespan,
    default_response_class=ORJSONResponse,
    debug=os.getenv('KIMERA_DEBUG', 'false').lower() == 'true'
)

# Setup middleware immediately after app creation
setup_middleware(app)

# Import and include routers
logger.log_with_context('INFO', 'Loading routers')

# Optimized routers
from .routers import optimized_metrics_router, optimized_geoid_router
app.include_router(optimized_metrics_router.router, tags=["System-Metrics-Optimized"])
app.include_router(optimized_geoid_router.router, prefix="/kimera", tags=["Geoids-Optimized"])

# Standard routers
from .routers import (
    geoid_scar_router,
    system_router,
    contradiction_router,
    vault_router,
    insight_router,
    statistics_router,
    output_analysis_router,
    core_actions_router,
    thermodynamic_router,
)

app.include_router(geoid_scar_router.router, prefix="/kimera", tags=["Geoid SCARs"])
app.include_router(system_router.router, prefix="/kimera", tags=["System"])
app.include_router(contradiction_router.router, prefix="/kimera", tags=["Contradiction Engine"])
app.include_router(vault_router.router, prefix="/kimera", tags=["Vault Manager"])
app.include_router(insight_router.router, prefix="/kimera", tags=["Insight Engine"])
app.include_router(statistics_router.router, prefix="/kimera", tags=["Statistics"])
app.include_router(output_analysis_router.router, prefix="/kimera", tags=["Output Analysis"])
app.include_router(core_actions_router.router, prefix="/kimera", tags=["Core Actions"])
app.include_router(thermodynamic_router.router, prefix="/kimera", tags=["Thermodynamic Engine"])


# Core endpoints with caching
@app.get("/", response_class=ORJSONResponse)
async def read_root(request: Request, response: Response,
                   bypass_cache: bool = Query(False, description="Bypass cache for debugging")) -> Dict[str, Any]:
    """
    Root endpoint with optional cache bypass.
    """
    # Check cache unless bypassed
    cache_key = "root"
    if not bypass_cache and cache_key in response_cache:
        cached_data, cache_time = response_cache[cache_key]
        if time.time() - cache_time < CACHE_TTL:
            response.headers["X-Cache"] = "HIT"
            response.headers["X-Cache-Age"] = f"{time.time() - cache_time:.3f}"
            return cached_data
    
    # Generate response
    data = {
        "message": "Welcome to Kimera SWM API (Hybrid)",
        "kimera_status": kimera_singleton.get_status(),
        "timestamp": time.time(),
        "mode": {
            "debug": logger.debug_mode,
            "performance": logger.performance_mode,
            "profiling": performance_profiler.enabled
        },
        "features": {
            "cache_enabled": True,
            "compression_enabled": True,
            "async_enabled": True,
            "debug_api": True,
            "request_tracing": True
        }
    }
    
    # Cache response
    if not bypass_cache:
        response_cache[cache_key] = (data, time.time())
        response.headers["X-Cache"] = "MISS"
    else:
        response.headers["X-Cache"] = "BYPASSED"
    
    return data


@app.get("/health", response_class=ORJSONResponse)
async def health_check() -> Dict[str, Any]:
    """Ultra-fast health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time_ns() // 1_000_000,
        "service": "kimera-swm-api-hybrid"
    }


# Commented out to avoid conflict with monitoring_routes.py
# @app.get("/metrics", response_class=ORJSONResponse)
# async def metrics(request: Request):
#     metrics = get_kimera_metrics()
#     return metrics


# Debug API endpoints
@app.get("/debug/info", response_class=ORJSONResponse, tags=["Debug"])
async def debug_info(
    include_logs: bool = Query(True, description="Include recent logs"),
    include_traces: bool = Query(True, description="Include request traces"),
    include_profiles: bool = Query(True, description="Include performance profiles")
) -> Dict[str, Any]:
    """
    Get comprehensive debug information.
    
    This endpoint provides:
    - Recent logs from ring buffer
    - Request traces and statistics
    - Performance profiles
    - Current debug settings
    """
    info = await get_debug_info()
    
    # Filter based on query parameters
    if not include_logs:
        info.pop('recent_logs', None)
    if not include_traces:
        info.pop('request_traces', None)
    if not include_profiles:
        info.pop('performance_profiles', None)
    
    return info


@app.post("/debug/mode", response_class=ORJSONResponse, tags=["Debug"])
async def toggle_debug_mode(enabled: bool = Query(..., description="Enable or disable debug mode")) -> Dict[str, str]:
    """Toggle debug mode at runtime"""
    return await set_debug_mode(enabled)


@app.post("/debug/profiling", response_class=ORJSONResponse, tags=["Debug"])
async def toggle_profiling(enabled: bool = Query(..., description="Enable or disable profiling")) -> Dict[str, str]:
    """Toggle performance profiling at runtime"""
    return await set_profiling(enabled)


@app.get("/debug/logs", response_class=ORJSONResponse, tags=["Debug"])
async def get_logs(
    limit: int = Query(100, description="Number of logs to retrieve"),
    level: Optional[str] = Query(None, description="Filter by log level (DEBUG, INFO, WARNING, ERROR)")
) -> Dict[str, Any]:
    """Get recent logs from ring buffer"""
    logs = hybrid_logger.get_recent_logs(limit=limit, level=level)
    return {
        "count": len(logs),
        "logs": logs,
        "buffer_capacity": hybrid_logger.ring_buffer.buffer.maxlen
    }


@app.get("/debug/traces", response_class=ORJSONResponse, tags=["Debug"])
async def get_traces(
    limit: int = Query(50, description="Number of traces to retrieve"),
    path_filter: Optional[str] = Query(None, description="Filter traces by path")
) -> Dict[str, Any]:
    """Get recent request traces"""
    traces = request_tracer.get_traces(limit=limit, path_filter=path_filter)
    stats = request_tracer.get_statistics()
    
    return {
        "count": len(traces),
        "traces": traces,
        "statistics": stats
    }


@app.post("/debug/log-level", response_class=ORJSONResponse, tags=["Debug"])
async def set_log_level(
    level: str = Query(..., description="Log level (DEBUG, INFO, WARNING, ERROR)"),
    module: Optional[str] = Query(None, description="Specific module to set level for")
) -> Dict[str, str]:
    """Set log level dynamically"""
    try:
        hybrid_logger.set_level(level, module)
        return {
            "status": "success",
            "level": level,
            "module": module or "root"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/performance/stats", response_class=ORJSONResponse, tags=["Debug"])
async def performance_stats() -> Dict[str, Any]:
    """Get detailed performance statistics"""
    return {
        "cache_stats": metrics_cache.get_cache_stats(),
        "response_cache_size": len(response_cache),
        "logging_metrics": hybrid_logger.get_performance_metrics(),
        "request_statistics": request_tracer.get_statistics(),
        "kimera_status": kimera_singleton.get_system_state()
    }


# Middleware for performance headers
@app.middleware("http")
async def add_performance_headers(request: Request, call_next):
    """Add performance monitoring headers"""
    start_time = time.perf_counter_ns()
    
    response = await call_next(request)
    
    # Add timing header
    process_time_us = (time.perf_counter_ns() - start_time) / 1000
    response.headers["X-Process-Time-Microseconds"] = str(int(process_time_us))
    
    # Add mode headers
    response.headers["X-Debug-Mode"] = str(logger.debug_mode).lower()
    response.headers["X-Performance-Mode"] = str(logger.performance_mode).lower()
    
    # Cache control for static endpoints
    if request.url.path in ["/", "/health", "/metrics"]:
        response.headers["Cache-Control"] = "public, max-age=1"
    
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Enhanced exception handler with debug info"""
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    logger.log_with_context(
        'ERROR',
        f"Unhandled exception: {exc}",
        request_id=request_id,
        path=str(request.url),
        method=request.method,
        exception_type=type(exc).__name__
    )
    
    response_content = {
        "error": "Internal server error",
        "timestamp": time.time(),
        "path": str(request.url)
    }
    
    # Add debug info if enabled
    if logger.debug_mode:
        response_content.update({
            "request_id": request_id,
            "exception": str(exc),
            "type": type(exc).__name__
        })
    
    return ORJSONResponse(
        status_code=500,
        content=response_content
    )


logger.log_with_context('INFO', 'Hybrid API initialization complete')