"""
KIMERA Prometheus Metrics Integration
=====================================

Provides Prometheus-compatible metrics for KIMERA system monitoring.
"""

import time
import logging
from typing import Dict, Any, Optional
from prometheus_client import Counter, Gauge, Histogram, Summary, Info
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
import asyncio

logger = logging.getLogger(__name__)

# System Information
kimera_info = Info('kimera_system', 'KIMERA system information')
kimera_info.info({
    'version': '0.1.0',
    'profile': 'development'
})

# Request Metrics
request_count = Counter(
    'kimera_requests_total',
    'Total number of requests',
    ['method', 'endpoint', 'status']
)

request_duration = Histogram(
    'kimera_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint']
)

# System Metrics
cpu_usage = Gauge('kimera_cpu_usage_percent', 'CPU usage percentage')
memory_usage = Gauge('kimera_memory_usage_bytes', 'Memory usage in bytes')
active_connections = Gauge('kimera_active_connections', 'Number of active connections')

# Component Health
component_health = Gauge(
    'kimera_component_health',
    'Component health status (1=healthy, 0=unhealthy)',
    ['component']
)

# Cognitive Metrics
geoid_count = Gauge('kimera_geoid_count', 'Total number of geoids')
scar_count = Gauge('kimera_scar_count', 'Total number of SCARs')
insight_count = Gauge('kimera_insight_count', 'Total number of insights')

contradiction_detections = Counter(
    'kimera_contradiction_detections_total',
    'Total number of contradiction detections'
)

cognitive_field_coherence = Gauge(
    'kimera_cognitive_field_coherence',
    'Cognitive field coherence level (0-1)'
)

# Performance Metrics
embedding_generation_time = Summary(
    'kimera_embedding_generation_seconds',
    'Time to generate embeddings'
)

database_query_time = Histogram(
    'kimera_database_query_seconds',
    'Database query execution time',
    ['operation']
)

# Error Metrics
error_count = Counter(
    'kimera_errors_total',
    'Total number of errors',
    ['component', 'error_type']
)

# Background collection task
_collection_task: Optional[asyncio.Task] = None

async def collect_system_metrics():
    """Collect system metrics periodically."""
    import psutil
    
    while True:
        try:
            # CPU and Memory
            cpu_usage.set(psutil.cpu_percent(interval=1))
            memory = psutil.virtual_memory()
            memory_usage.set(memory.used)
            
            # Component health checks
            try:
                from backend.core.kimera_system import get_kimera_system
                system = get_kimera_system()
                status = system.get_status()
                component_health.labels(component='kimera_system').set(
                    1 if status == 'running' else 0
                )
            except:
                component_health.labels(component='kimera_system').set(0)
            
            # Database metrics
            try:
                from backend.vault.vault_manager import VaultManager
                vault = VaultManager()
                stats = vault.get_vault_statistics()
                geoid_count.set(stats.get('total_geoids', 0))
                scar_count.set(stats.get('total_scars', 0))
                insight_count.set(stats.get('total_insights', 0))
            except:
                pass
            
            # Cognitive field metrics
            try:
                from backend.monitoring.cognitive_field_metrics import get_cognitive_field_metrics
                field_metrics = get_cognitive_field_metrics()
                summary = field_metrics.get_field_summary()
                if 'coherence' in summary:
                    cognitive_field_coherence.set(summary['coherence'])
            except:
                pass
            
            await asyncio.sleep(15)  # Collect every 15 seconds
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            await asyncio.sleep(30)

def initialize_background_collection():
    """Initialize background metrics collection."""
    global _collection_task
    
    if _collection_task is not None:
        raise ValueError("Background collection already initialized")
    
    loop = asyncio.get_event_loop()
    _collection_task = loop.create_task(collect_system_metrics())
    logger.info("Prometheus metrics background collection started")

def get_metrics():
    """Get current metrics in Prometheus format."""
    return generate_latest()

def get_metrics_content_type():
    """Get content type for Prometheus metrics."""
    return CONTENT_TYPE_LATEST

# Decorator for timing functions
def track_time(metric_name: str, labels: Dict[str, str] = None):
    """Decorator to track execution time of functions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                if metric_name == 'request':
                    request_duration.labels(**labels).observe(duration)
                elif metric_name == 'embedding':
                    embedding_generation_time.observe(duration)
                elif metric_name == 'database':
                    database_query_time.labels(**labels).observe(duration)
        return wrapper
    return decorator

def get_kimera_metrics():
    """Get KIMERA-specific metrics summary."""
    return {
        'geoids': geoid_count._value.get() if hasattr(geoid_count, '_value') else 0,
        'scars': scar_count._value.get() if hasattr(scar_count, '_value') else 0,
        'insights': insight_count._value.get() if hasattr(insight_count, '_value') else 0,
        'coherence': cognitive_field_coherence._value.get() if hasattr(cognitive_field_coherence, '_value') else 0,
        'cpu_usage': cpu_usage._value.get() if hasattr(cpu_usage, '_value') else 0,
        'memory_usage': memory_usage._value.get() if hasattr(memory_usage, '_value') else 0
    }

# Export commonly used functions
class KimeraPrometheusMetrics:
    """Legacy compatibility class for KimeraPrometheusMetrics."""
    
    def __init__(self):
        self.request_count = request_count
        self.request_duration = request_duration
        self.cpu_usage = cpu_usage
        self.memory_usage = memory_usage
        self.component_health = component_health
        self.error_count = error_count
        
    def get_metrics(self):
        return get_metrics()
        
    def get_kimera_metrics(self):
        return get_kimera_metrics()

__all__ = [
    'request_count',
    'request_duration',
    'cpu_usage',
    'memory_usage',
    'component_health',
    'error_count',
    'track_time',
    'initialize_background_collection',
    'get_metrics',
    'get_metrics_content_type',
    'get_kimera_metrics',
    'KimeraPrometheusMetrics'
]