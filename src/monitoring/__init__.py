"""
KIMERA Monitoring Module
========================

Comprehensive monitoring and observability for KIMERA system.
Implements patterns from aerospace and medical device monitoring.

Key Components:
- Real-time metrics collection
- Distributed tracing
- Performance profiling
- Anomaly detection
- Alert management
"""

from .alert_manager import AlertLevel, AlertManager
from .entropy_monitor import EntropyEstimator, EntropyMonitor
from .kimera_monitoring_core import (AlertSeverity, KimeraMonitoringCore,
                                     MonitoringLevel, get_monitoring_core)
from .kimera_prometheus_metrics import get_kimera_metrics
from .metrics_collector import MetricsCollector, MetricType
from .metrics_integration import get_integration_manager
from .monitoring_integration import get_monitoring_manager
from .performance_monitor import PerformanceMonitor
from .psychiatric_stability_monitor import CognitiveCoherenceMonitor
from .system_profiler import SystemProfiler
from .thermodynamic_analyzer import ThermodynamicCalculator

__all__ = [
    "MetricsCollector",
    "MetricType",
    "PerformanceMonitor",
    "AlertManager",
    "AlertLevel",
    "SystemProfiler",
    "KimeraMonitoringCore",
    "get_monitoring_core",
    "MonitoringLevel",
    "AlertSeverity",
    "get_monitoring_manager",
    "get_integration_manager",
    "get_kimera_metrics",
    "CognitiveCoherenceMonitor",
    "EntropyEstimator",
    "EntropyMonitor",
    "ThermodynamicCalculator",
]
