"""
Kimera SWM - Metrics Integration System
======================================

Comprehensive integration system that connects state-of-the-art monitoring
to all Kimera components, providing deep insights into system behavior.
"""

import asyncio
import functools
import inspect
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Type

# FastAPI integration
from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

# OpenTelemetry integration - make optional
try:
    from opentelemetry import trace
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.propagate import inject

    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False

# Prometheus integration - make optional
try:
    from prometheus_client import CollectorRegistry, start_http_server
    from prometheus_fastapi_instrumentator import Instrumentator

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Custom monitoring
from .kimera_monitoring_core import KimeraMetric, get_monitoring_core

# Kimera components (would import actual modules)
# from ..core.geoid_manager import GeoidManager
# from ..core.scar_repository import ScarRepository
# from ..engines.contradiction_engine import ContradictionEngine
# from ..engines.selective_feedback_engine import SelectiveFeedbackEngine

logger = logging.getLogger(__name__)


@dataclass
class MetricIntegrationConfig:
    """Configuration for metrics integration"""

    # Monitoring settings
    enable_request_metrics: bool = True
    enable_response_metrics: bool = True
    enable_error_metrics: bool = True
    enable_performance_metrics: bool = True

    # Kimera-specific settings
    track_geoid_operations: bool = True
    track_scar_operations: bool = True
    track_contradiction_events: bool = True
    track_selective_feedback: bool = True
    track_revolutionary_insights: bool = True

    # Performance settings
    track_latency: bool = True
    track_throughput: bool = True
    track_memory_usage: bool = True
    track_gpu_usage: bool = True

    # Sampling settings
    sample_rate: float = 1.0  # Sample 100% by default
    slow_request_threshold: float = 1.0  # 1 second

    # Storage settings
    metrics_retention_days: int = 30
    max_metrics_in_memory: int = 10000


class KimeraMetricsMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for comprehensive request/response monitoring"""

    def __init__(self, app, config: MetricIntegrationConfig):
        super().__init__(app)
        self.config = config
        self.monitoring_core = get_monitoring_core()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and collect metrics"""

        start_time = time.time()

        # Extract request info
        method = request.method
        path = request.url.path

        # Start trace if enabled and available
        if OPENTELEMETRY_AVAILABLE:
            tracer = trace.get_tracer(__name__)
            span = tracer.start_as_current_span(f"{method} {path}")
        else:
            span = None

        try:
            # Add request attributes to span
            if span:
                span.set_attribute("http.method", method)
                span.set_attribute("http.url", str(request.url))
                span.set_attribute(
                    "http.user_agent", request.headers.get("user-agent", "")
                )

            # Process request
            response = await call_next(request)

            # Calculate metrics
            duration = time.time() - start_time

            # Record metrics
            if self.config.enable_request_metrics:
                self.monitoring_core.request_count.labels(
                    method=method, endpoint=path, status=response.status_code
                ).inc()

            if self.config.enable_performance_metrics:
                self.monitoring_core.request_duration.labels(
                    method=method, endpoint=path
                ).observe(duration)

            # Add response attributes to span
            if span:
                span.set_attribute("http.status_code", response.status_code)
                span.set_attribute("http.response_time", duration)

                # Check for slow requests
                if duration > self.config.slow_request_threshold:
                    span.add_event(
                        "slow_request",
                        {
                            "duration": duration,
                            "threshold": self.config.slow_request_threshold,
                        },
                    )

            return response

        except Exception as e:
            # Record error metrics
            if self.config.enable_error_metrics:
                self.monitoring_core.error_count.labels(
                    error_type=type(e).__name__, component="middleware"
                ).inc()

            # Add error to span
            if span and OPENTELEMETRY_AVAILABLE:
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))

            raise
        finally:
            # Close span if it was created
            if span and OPENTELEMETRY_AVAILABLE:
                span.end()


class MetricsDecorator:
    """Decorator for automatic metrics collection on functions/methods"""

    def __init__(
        self,
        metric_name: Optional[str] = None,
        track_latency: bool = True,
        track_calls: bool = True,
        track_errors: bool = True,
        component: Optional[str] = None,
    ):

        self.metric_name = metric_name
        self.track_latency = track_latency
        self.track_calls = track_calls
        self.track_errors = track_errors
        self.component = component
        self.monitoring_core = get_monitoring_core()

    def __call__(self, func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await self._execute_with_metrics(func, args, kwargs, is_async=True)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            return asyncio.run(
                self._execute_with_metrics(func, args, kwargs, is_async=False)
            )

        # Return appropriate wrapper based on function type
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    async def _execute_with_metrics(self, func, args, kwargs, is_async: bool):
        """Execute function with metrics collection"""

        start_time = time.time()
        metric_name = self.metric_name or f"{func.__module__}.{func.__name__}"
        component = self.component or func.__module__.split(".")[-1]

        # Start tracing
        if OPENTELEMETRY_AVAILABLE:
            tracer = trace.get_tracer(__name__)
            span = tracer.start_as_current_span(metric_name)
        else:
            span = None

        try:
            # Add function info to span
            if span:
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)
                span.set_attribute("component", component)

            # Execute function
            if is_async:
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            # Calculate latency
            duration = time.time() - start_time

            # Record metrics
            if self.track_calls:
                if hasattr(self.monitoring_core, "kimera_prometheus_metrics"):
                    # Use Kimera-specific counter if available
                    pass

            return result

        except Exception as e:
            # Record error metrics
            if self.track_errors:
                self.monitoring_core.error_count.labels(
                    error_type=type(e).__name__, component=component
                ).inc()

            # Add error to span
            if span and OPENTELEMETRY_AVAILABLE:
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))

            raise
        finally:
            # Close span if it was created
            if span and OPENTELEMETRY_AVAILABLE:
                span.end()


class KimeraComponentIntegration:
    """Integration layer for monitoring Kimera-specific components"""

    def __init__(self, config: MetricIntegrationConfig):
        self.config = config
        self.monitoring_core = get_monitoring_core()
        self.component_metrics = {}

        # Initialize component-specific metrics
        self._init_component_metrics()

    def _init_component_metrics(self):
        """Initialize metrics for each Kimera component"""

        # Geoid Manager metrics
        if self.config.track_geoid_operations:
            self.component_metrics["geoid"] = {
                "creation_rate": self.monitoring_core.kimera_prometheus_metrics.get(
                    "geoid_creation_rate"
                ),
                "total_count": self.monitoring_core.geoid_count,
            }

        # Scar Repository metrics
        if self.config.track_scar_operations:
            self.component_metrics["scar"] = {
                "formation_rate": self.monitoring_core.kimera_prometheus_metrics.get(
                    "scar_formation_rate"
                ),
                "total_count": self.monitoring_core.scar_count,
            }

        # Contradiction Engine metrics
        if self.config.track_contradiction_events:
            self.component_metrics["contradiction"] = {
                "events": self.monitoring_core.contradiction_events,
                "severity": self.monitoring_core.kimera_prometheus_metrics.get(
                    "contradiction_severity"
                ),
            }

        # Selective Feedback metrics
        if self.config.track_selective_feedback:
            self.component_metrics["selective_feedback"] = {
                "operations": self.monitoring_core.selective_feedback_operations,
                "accuracy": self.monitoring_core.kimera_prometheus_metrics.get(
                    "selective_feedback_accuracy"
                ),
            }

        # Revolutionary Insights metrics
        if self.config.track_revolutionary_insights:
            self.component_metrics["revolutionary"] = {
                "insights": self.monitoring_core.revolutionary_insights,
                "breakthrough_score": self.monitoring_core.kimera_prometheus_metrics.get(
                    "revolutionary_breakthrough_score"
                ),
            }

        logger.info(
            f"📊 Initialized metrics for {len(self.component_metrics)} Kimera components"
        )

    @asynccontextmanager
    async def track_geoid_operation(self, operation_type: str, vault_id: str):
        """Context manager for tracking geoid operations"""

        start_time = time.time()

        try:
            yield

            # Record successful operation
            if self.component_metrics.get("geoid", {}).get("creation_rate"):
                self.component_metrics["geoid"]["creation_rate"].labels(
                    vault=vault_id
                ).inc()

        except Exception as e:
            # Record error
            self.monitoring_core.error_count.labels(
                error_type=type(e).__name__, component="geoid_manager"
            ).inc()
            raise

        finally:
            # Record latency
            duration = time.time() - start_time
            # Could add specific geoid operation latency metric

    @asynccontextmanager
    async def track_scar_operation(self, operation_type: str, scar_type: str):
        """Context manager for tracking scar operations"""

        start_time = time.time()

        try:
            yield

            # Record successful operation
            if self.component_metrics.get("scar", {}).get("formation_rate"):
                self.component_metrics["scar"]["formation_rate"].labels(
                    type=scar_type
                ).inc()

        except Exception as e:
            self.monitoring_core.error_count.labels(
                error_type=type(e).__name__, component="scar_repository"
            ).inc()
            raise

        finally:
            duration = time.time() - start_time

    @asynccontextmanager
    async def track_contradiction_event(self, source: str, severity: float):
        """Context manager for tracking contradiction events"""

        try:
            yield

            # Record contradiction event
            self.monitoring_core.contradiction_events.labels(type=source).inc()

            # Record severity
            if self.component_metrics.get("contradiction", {}).get("severity"):
                self.component_metrics["contradiction"]["severity"].labels(
                    source=source
                ).observe(severity)

        except Exception as e:
            self.monitoring_core.error_count.labels(
                error_type=type(e).__name__, component="contradiction_engine"
            ).inc()
            raise

    @asynccontextmanager
    async def track_selective_feedback(self, domain: str):
        """Context manager for tracking selective feedback operations"""

        start_time = time.time()

        try:
            yield

            # Record operation
            self.monitoring_core.selective_feedback_operations.labels(
                domain=domain
            ).inc()

        except Exception as e:
            self.monitoring_core.error_count.labels(
                error_type=type(e).__name__, component="selective_feedback"
            ).inc()
            raise

        finally:
            duration = time.time() - start_time

    def record_revolutionary_insight(self, breakthrough_score: float):
        """Record a revolutionary insight generation"""

        self.monitoring_core.revolutionary_insights.inc()

        if self.component_metrics.get("revolutionary", {}).get("breakthrough_score"):
            self.component_metrics["revolutionary"]["breakthrough_score"].observe(
                breakthrough_score
            )

    def update_component_counts(
        self, geoid_count: Optional[int] = None, scar_count: Optional[int] = None
    ):
        """Update component counts"""

        if geoid_count is not None:
            self.monitoring_core.geoid_count.set(geoid_count)

        if scar_count is not None:
            self.monitoring_core.scar_count.set(scar_count)


class MetricsIntegrationManager:
    """Main manager for integrating monitoring across all Kimera components"""

    def __init__(self, config: Optional[MetricIntegrationConfig] = None):
        self.config = config or MetricIntegrationConfig()
        self.monitoring_core = get_monitoring_core()
        self.component_integration = KimeraComponentIntegration(self.config)

        # Integration state
        self.is_initialized = False
        self.fastapi_app: Optional[FastAPI] = None
        self.prometheus_server_port: Optional[int] = None

        logger.info("🔧 Metrics Integration Manager initialized")

    def integrate_with_fastapi(self, app: FastAPI, prometheus_port: int = 9090):
        """Integrate monitoring with FastAPI application"""

        self.fastapi_app = app
        self.prometheus_server_port = prometheus_port

        # Add middleware
        app.add_middleware(KimeraMetricsMiddleware, config=self.config)

        # Configure Instrumentator
        instrumentator = Instrumentator(
            should_group_status_codes=False,
            should_ignore_untemplated=True,
            should_group_untemplated=False,
            should_instrument_requests_inprogress=True,
            should_instrument_requests_body_size=True,
            should_instrument_responses_body_size=True,
            excluded_handlers=["/metrics", "/health"],
        )

        # Instrument the app
        instrumentator.instrument(app)

        # Expose metrics endpoint
        instrumentator.expose(app, endpoint="/metrics")

        # Start Prometheus server
        if prometheus_port:
            start_http_server(prometheus_port)
            logger.info(f"📊 Prometheus server started on port {prometheus_port}")

        # Configure OpenTelemetry
        FastAPIInstrumentor.instrument_app(app)

        logger.info("🚀 FastAPI integration complete")
        self.is_initialized = True

    def create_metrics_decorator(self, **kwargs) -> MetricsDecorator:
        """Create a metrics decorator with default configuration"""

        return MetricsDecorator(**kwargs)

    def get_component_integration(self) -> KimeraComponentIntegration:
        """Get component integration instance"""

        return self.component_integration

    async def start_background_monitoring(self):
        """Start background monitoring tasks"""

        if not self.monitoring_core.is_running:
            await self.monitoring_core.start_monitoring()
            logger.info("🔄 Background monitoring started")

    async def stop_background_monitoring(self):
        """Stop background monitoring tasks"""

        if self.monitoring_core.is_running:
            await self.monitoring_core.stop_monitoring()
            logger.info("⏹️ Background monitoring stopped")

    def get_integration_status(self) -> Dict[str, Any]:
        """Get status of metrics integration"""

        return {
            "is_initialized": self.is_initialized,
            "fastapi_integrated": self.fastapi_app is not None,
            "prometheus_port": self.prometheus_server_port,
            "monitoring_status": self.monitoring_core.get_monitoring_status(),
            "config": {
                "request_metrics": self.config.enable_request_metrics,
                "performance_metrics": self.config.enable_performance_metrics,
                "kimera_tracking": {
                    "geoids": self.config.track_geoid_operations,
                    "scars": self.config.track_scar_operations,
                    "contradictions": self.config.track_contradiction_events,
                    "selective_feedback": self.config.track_selective_feedback,
                    "revolutionary_insights": self.config.track_revolutionary_insights,
                },
            },
        }


# Global integration manager
_integration_manager: Optional[MetricsIntegrationManager] = None


def get_integration_manager() -> MetricsIntegrationManager:
    """Get the global integration manager instance"""
    global _integration_manager

    if _integration_manager is None:
        _integration_manager = MetricsIntegrationManager()

    return _integration_manager


def initialize_metrics_integration(
    config: Optional[MetricIntegrationConfig] = None,
) -> MetricsIntegrationManager:
    """Initialize the metrics integration system"""

    global _integration_manager

    _integration_manager = MetricsIntegrationManager(config)
    return _integration_manager


# Convenience decorators
def track_kimera_function(component: str = None, **kwargs):
    """Decorator for tracking Kimera function calls"""
    manager = get_integration_manager()
    return manager.create_metrics_decorator(component=component, **kwargs)


def track_geoid_operation(operation_type: str, vault_id: str):
    """Context manager for tracking geoid operations"""
    manager = get_integration_manager()
    return manager.component_integration.track_geoid_operation(operation_type, vault_id)


def track_scar_operation(operation_type: str, scar_type: str):
    """Context manager for tracking scar operations"""
    manager = get_integration_manager()
    return manager.component_integration.track_scar_operation(operation_type, scar_type)


def track_contradiction_event(source: str, severity: float):
    """Context manager for tracking contradiction events"""
    manager = get_integration_manager()
    return manager.component_integration.track_contradiction_event(source, severity)


def track_selective_feedback(domain: str):
    """Context manager for tracking selective feedback"""
    manager = get_integration_manager()
    return manager.component_integration.track_selective_feedback(domain)


# Export main classes and functions
__all__ = [
    "MetricIntegrationConfig",
    "KimeraMetricsMiddleware",
    "MetricsDecorator",
    "KimeraComponentIntegration",
    "MetricsIntegrationManager",
    "get_integration_manager",
    "initialize_metrics_integration",
    "track_kimera_function",
    "track_geoid_operation",
    "track_scar_operation",
    "track_contradiction_event",
    "track_selective_feedback",
]
