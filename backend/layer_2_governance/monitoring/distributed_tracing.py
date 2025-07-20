"""
Distributed Tracing for KIMERA System
Implements OpenTelemetry-based distributed tracing
Phase 3, Week 9: Monitoring Infrastructure
"""

import logging
import json
from typing import Optional, Dict, Any, List

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor, ConsoleSpanExporter, SpanExporter
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
try:
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
except ImportError:
    HTTPXClientInstrumentor = None
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

from backend.config import get_settings, is_production

logger = logging.getLogger(__name__)


class TracingManager:
    """
    Manages distributed tracing for KIMERA
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.is_configured = False
    
    def configure_tracing(self, app: Optional[Any] = None, engine: Optional[Any] = None):
        """Configure distributed tracing for the application"""
        if not self.settings.monitoring.enabled or not self.settings.get_feature("distributed_tracing"):
            logger.info("Distributed tracing is disabled")
            return
        
        if self.is_configured:
            return
        
        # Create resource
        resource = Resource.create({
            "service.name": "kimera",
            "service.version": "0.1.0"
        })
        
        # Create tracer provider
        provider = TracerProvider(resource=resource)
        
        # Create span processor
        span_processor = BatchSpanProcessor(self._get_span_exporter())
        provider.add_span_processor(span_processor)
        
        # Set the global tracer provider
        trace.set_tracer_provider(provider)
        
        # Instrument application
        if app:
            self.instrument_fastapi(app)
        
        if engine:
            self.instrument_sqlalchemy(engine)
        
        self.instrument_httpx()
        
        self.is_configured = True
        logger.info("Distributed tracing configured")
    
    def _get_span_exporter(self) -> SpanExporter:
        """Get span exporter based on configuration"""
        if is_production():
            # Use OTLP exporter for production (e.g., to Jaeger, Grafana Tempo)
            endpoint = self.settings.monitoring.get("otlp_endpoint", "localhost:4317")
            logger.info(f"Using OTLP span exporter with endpoint: {endpoint}")
            return OTLPSpanExporter(endpoint=endpoint, insecure=True)
        else:
            # Use console exporter for development
            logger.info("Using console span exporter")
            return ConsoleSpanExporter()
    
    def instrument_fastapi(self, app: Any):
        """Instrument FastAPI application"""
        try:
            FastAPIInstrumentor.instrument_app(app)
            logger.info("FastAPI instrumentation enabled")
        except Exception as e:
            logger.error(f"Failed to instrument FastAPI: {e}")
    
    def instrument_sqlalchemy(self, engine: Any):
        """Instrument SQLAlchemy engine"""
        try:
            SQLAlchemyInstrumentor().instrument(engine=engine)
            logger.info("SQLAlchemy instrumentation enabled")
        except Exception as e:
            logger.error(f"Failed to instrument SQLAlchemy: {e}")
    
    def instrument_httpx(self):
        """Instrument HTTPX client"""
        try:
            HTTPXClientInstrumentor().instrument()
            logger.info("HTTPX client instrumentation enabled")
        except Exception as e:
            logger.error(f"Failed to instrument HTTPX: {e}")
    
    def get_tracer(self, name: str) -> trace.Tracer:
        """Get a tracer instance"""
        return trace.get_tracer(name)


# Global tracing manager
_tracing_manager: Optional[TracingManager] = None


def get_tracing_manager() -> TracingManager:
    """Get global tracing manager instance"""
    global _tracing_manager
    if _tracing_manager is None:
        _tracing_manager = TracingManager()
    return _tracing_manager


def get_tracer(name: str) -> trace.Tracer:
    """Get a tracer instance"""
    manager = get_tracing_manager()
    return manager.get_tracer(name)


# Decorator for custom spans
def with_span(name: Optional[str] = None):
    """
    Decorator to create a new span for a function
    
    Args:
        name: Optional span name (defaults to function name)
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            span_name = name or func.__name__
            tracer = get_tracer(func.__module__)
            
            with tracer.start_as_current_span(span_name) as span:
                # Add arguments as attributes
                for i, arg in enumerate(args):
                    span.set_attribute(f"arg_{i}", str(arg))
                for key, value in kwargs.items():
                    span.set_attribute(key, str(value))
                
                try:
                    result = func(*args, **kwargs)
                    span.set_status(trace.Status(trace.StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
        
        return wrapper
    
    return decorator


# Example usage
if __name__ == "__main__":
    # Configure tracing
    manager = get_tracing_manager()
    manager.configure_tracing()
    
    # Get tracer
    tracer = get_tracer("my_app")
    
    @with_span("my_function")
    def my_function(x, y):
        """Example function with tracing"""
        with tracer.start_as_current_span("inner_work") as span:
            span.set_attribute("work_type", "computation")
            result = x + y
            span.set_attribute("result", result)
            return result
    
    # Run function
    my_function(1, 2)
    
    # Example with error
    @with_span()
    def failing_function():
        raise ValueError("Test error")
    
    try:
        failing_function()
    except ValueError:
        pass