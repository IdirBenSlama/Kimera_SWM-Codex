"""
KIMERA Monitoring Module
Provides monitoring, metrics, and observability features
"""

from .monitoring_integration import (
    MonitoringManager,
    get_monitoring_manager
)

from .metrics_and_alerting import (
    MetricsManager,
    AlertingManager,
    get_metrics_manager,
    get_alerting_manager,
    track_requests
)

from .distributed_tracing import (
    TracingManager,
    get_tracing_manager,
    get_tracer,
    with_span
)

from .kimera_monitoring_core import (
    get_monitoring_core,
    MonitoringLevel,
    AlertSeverity
)

__all__ = [
    # Monitoring integration
    "MonitoringManager",
    "get_monitoring_manager",
    
    # Metrics and alerting
    "MetricsManager",
    "AlertingManager",
    "get_metrics_manager",
    "get_alerting_manager",
    "track_requests",
    
    # Distributed tracing
    "TracingManager",
    "get_tracing_manager",
    "get_tracer",
    "with_span",
    
    # Kimera monitoring core
    "get_monitoring_core",
    "MonitoringLevel",
    "AlertSeverity",
]