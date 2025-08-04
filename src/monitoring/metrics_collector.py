"""
Metrics Collector - Aerospace-Grade Telemetry System
====================================================

Implements comprehensive metrics collection based on:
- NASA Mission Control telemetry systems
- Prometheus/Grafana patterns
- Time-series database optimization
"""

import json
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics following Prometheus conventions."""

    COUNTER = "counter"  # Monotonically increasing
    GAUGE = "gauge"  # Can go up or down
    HISTOGRAM = "histogram"  # Distribution of values
    SUMMARY = "summary"  # Similar to histogram with percentiles


@dataclass
class MetricPoint:
    """Single metric data point."""

    timestamp: float
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class Metric:
    """Metric definition and storage."""

    name: str
    type: MetricType
    description: str
    unit: str = ""
    labels: List[str] = field(default_factory=list)

    # Data storage (ring buffer for memory efficiency)
    data_points: deque = field(default_factory=lambda: deque(maxlen=10000))

    # Aggregations
    total: float = 0.0
    count: int = 0
    min_value: float = float("inf")
    max_value: float = float("-inf")

    # For histograms
    buckets: Optional[List[float]] = None
    bucket_counts: Optional[Dict[float, int]] = None

    def __post_init__(self):
        if self.type == MetricType.HISTOGRAM and self.buckets:
            self.bucket_counts = {b: 0 for b in self.buckets}


class MetricsCollector:
    """
    Centralized metrics collection system.

    Implements patterns from:
    - Prometheus metrics
    - StatsD
    - NASA telemetry systems
    """

    def __init__(self, retention_hours: int = 24):
        self.metrics: Dict[str, Metric] = {}
        self.retention_hours = retention_hours
        self._lock = threading.RLock()

        # Metric families for organization
        self.families: Dict[str, List[str]] = defaultdict(list)

        # Export formats
        self.exporters = {
            "prometheus": self._export_prometheus,
            "json": self._export_json,
            "csv": self._export_csv,
        }

        # Background cleanup
        self._cleanup_interval = 3600  # 1 hour
        self._last_cleanup = time.time()

        logger.info(f"MetricsCollector initialized (retention={retention_hours}h)")

    def register_metric(
        self,
        name: str,
        metric_type: MetricType,
        description: str,
        unit: str = "",
        labels: List[str] = None,
        buckets: List[float] = None,
    ) -> Metric:
        """
        Register a new metric.

        Args:
            name: Metric name (e.g., 'kimera_requests_total')
            metric_type: Type of metric
            description: Human-readable description
            unit: Unit of measurement
            labels: Label dimensions
            buckets: For histograms, bucket boundaries

        Returns:
            Registered metric
        """
        with self._lock:
            if name in self.metrics:
                logger.warning(f"Metric {name} already registered")
                return self.metrics[name]

            metric = Metric(
                name=name,
                type=metric_type,
                description=description,
                unit=unit,
                labels=labels or [],
                buckets=buckets,
            )

            self.metrics[name] = metric

            # Add to family
            family = name.split("_")[0]
            self.families[family].append(name)

            logger.debug(f"Registered metric: {name} ({metric_type.value})")
            return metric

    def record(
        self,
        name: str,
        value: float,
        labels: Dict[str, str] = None,
        timestamp: Optional[float] = None,
    ):
        """
        Record a metric value.

        Args:
            name: Metric name
            value: Metric value
            labels: Label values
            timestamp: Optional timestamp (defaults to now)
        """
        with self._lock:
            if name not in self.metrics:
                logger.warning(f"Unregistered metric: {name}")
                return

            metric = self.metrics[name]
            ts = timestamp or time.time()

            # Create data point
            point = MetricPoint(timestamp=ts, value=value, labels=labels or {})

            # Store data point
            metric.data_points.append(point)

            # Update aggregations
            self._update_aggregations(metric, value)

            # Periodic cleanup
            if time.time() - self._last_cleanup > self._cleanup_interval:
                self._cleanup_old_data()

    def increment(self, name: str, value: float = 1.0, labels: Dict[str, str] = None):
        """Increment a counter metric."""
        with self._lock:
            if name not in self.metrics:
                return

            metric = self.metrics[name]
            if metric.type != MetricType.COUNTER:
                logger.warning(f"Increment called on non-counter: {name}")
                return

            metric.total += value
            self.record(name, metric.total, labels)

    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Set a gauge metric."""
        self.record(name, value, labels)

    def observe(self, name: str, value: float, labels: Dict[str, str] = None):
        """Observe a value for histogram/summary."""
        with self._lock:
            if name not in self.metrics:
                return

            metric = self.metrics[name]
            if metric.type not in [MetricType.HISTOGRAM, MetricType.SUMMARY]:
                logger.warning(f"Observe called on wrong type: {name}")
                return

            # Record the observation
            self.record(name, value, labels)

            # Update histogram buckets
            if metric.type == MetricType.HISTOGRAM and metric.bucket_counts:
                for bucket in sorted(metric.buckets):
                    if value <= bucket:
                        metric.bucket_counts[bucket] += 1

    def _update_aggregations(self, metric: Metric, value: float):
        """Update metric aggregations."""
        metric.count += 1
        metric.min_value = min(metric.min_value, value)
        metric.max_value = max(metric.max_value, value)

        if metric.type == MetricType.COUNTER:
            metric.total = max(metric.total, value)  # Counters only increase

    def _cleanup_old_data(self):
        """Remove data points older than retention period."""
        with self._lock:
            cutoff = time.time() - (self.retention_hours * 3600)

            for metric in self.metrics.values():
                # Remove old points
                while metric.data_points and metric.data_points[0].timestamp < cutoff:
                    metric.data_points.popleft()

            self._last_cleanup = time.time()

    def get_metric(self, name: str) -> Optional[Metric]:
        """Get a metric by name."""
        return self.metrics.get(name)

    def get_current_value(self, name: str) -> Optional[float]:
        """Get the most recent value of a metric."""
        with self._lock:
            metric = self.metrics.get(name)
            if not metric or not metric.data_points:
                return None

            return metric.data_points[-1].value

    def get_statistics(self, name: str, window_seconds: int = 300) -> Dict[str, float]:
        """
        Get statistics for a metric over a time window.

        Args:
            name: Metric name
            window_seconds: Time window in seconds

        Returns:
            Dictionary with min, max, mean, std, p50, p95, p99
        """
        with self._lock:
            metric = self.metrics.get(name)
            if not metric:
                return {}

            cutoff = time.time() - window_seconds
            recent_values = [
                p.value for p in metric.data_points if p.timestamp >= cutoff
            ]

            if not recent_values:
                return {}

            values = np.array(recent_values)

            return {
                "count": len(values),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "p50": float(np.percentile(values, 50)),
                "p95": float(np.percentile(values, 95)),
                "p99": float(np.percentile(values, 99)),
            }

    def export(self, format: str = "prometheus") -> str:
        """
        Export metrics in specified format.

        Args:
            format: Export format (prometheus, json, csv)

        Returns:
            Formatted metrics string
        """
        if format not in self.exporters:
            raise ValueError(f"Unknown format: {format}")

        return self.exporters[format]()

    def _export_prometheus(self) -> str:
        """Export in Prometheus format."""
        lines = []

        with self._lock:
            for metric in self.metrics.values():
                # HELP and TYPE
                lines.append(f"# HELP {metric.name} {metric.description}")
                lines.append(f"# TYPE {metric.name} {metric.type.value}")

                if metric.type == MetricType.COUNTER:
                    lines.append(f"{metric.name} {metric.total}")

                elif metric.type == MetricType.GAUGE:
                    if metric.data_points:
                        value = metric.data_points[-1].value
                        lines.append(f"{metric.name} {value}")

                elif metric.type == MetricType.HISTOGRAM:
                    # Bucket counts
                    if metric.bucket_counts:
                        for bucket, count in sorted(metric.bucket_counts.items()):
                            lines.append(
                                f'{metric.name}_bucket{{le="{bucket}"}} {count}'
                            )
                        lines.append(
                            f'{metric.name}_bucket{{le="+Inf"}} {metric.count}'
                        )

                    # Sum and count
                    total = sum(p.value for p in metric.data_points)
                    lines.append(f"{metric.name}_sum {total}")
                    lines.append(f"{metric.name}_count {metric.count}")

                lines.append("")  # Empty line between metrics

        return "\n".join(lines)

    def _export_json(self) -> str:
        """Export in JSON format."""
        data = {}

        with self._lock:
            for name, metric in self.metrics.items():
                data[name] = {
                    "type": metric.type.value,
                    "description": metric.description,
                    "unit": metric.unit,
                    "count": metric.count,
                    "min": (
                        metric.min_value if metric.min_value != float("inf") else None
                    ),
                    "max": (
                        metric.max_value if metric.max_value != float("-inf") else None
                    ),
                    "current": (
                        metric.data_points[-1].value if metric.data_points else None
                    ),
                    "total": (
                        metric.total if metric.type == MetricType.COUNTER else None
                    ),
                }

        return json.dumps(data, indent=2)

    def _export_csv(self) -> str:
        """Export recent data in CSV format."""
        lines = ["timestamp,metric,value,labels"]

        with self._lock:
            # Last 1000 points across all metrics
            all_points = []
            for name, metric in self.metrics.items():
                for point in list(metric.data_points)[-100:]:  # Last 100 per metric
                    labels_str = json.dumps(point.labels) if point.labels else "{}"
                    all_points.append((point.timestamp, name, point.value, labels_str))

            # Sort by timestamp
            all_points.sort(key=lambda x: x[0])

            for ts, name, value, labels in all_points[-1000:]:
                lines.append(f"{ts},{name},{value},{labels}")

        return "\n".join(lines)


# Global metrics collector
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
        _register_default_metrics()
    return _metrics_collector


def _register_default_metrics():
    """Register default system metrics."""
    collector = get_metrics_collector()

    # Request metrics
    collector.register_metric(
        "kimera_requests_total",
        MetricType.COUNTER,
        "Total number of requests",
        labels=["method", "endpoint", "status"],
    )

    collector.register_metric(
        "kimera_request_duration_seconds",
        MetricType.HISTOGRAM,
        "Request duration in seconds",
        unit="seconds",
        labels=["method", "endpoint"],
        buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    )

    # System metrics
    collector.register_metric(
        "kimera_cpu_usage_percent",
        MetricType.GAUGE,
        "CPU usage percentage",
        unit="percent",
    )

    collector.register_metric(
        "kimera_memory_usage_bytes",
        MetricType.GAUGE,
        "Memory usage in bytes",
        unit="bytes",
    )

    # Component health
    collector.register_metric(
        "kimera_component_health",
        MetricType.GAUGE,
        "Component health status (1=healthy, 0=unhealthy)",
        labels=["component"],
    )

    # Error metrics
    collector.register_metric(
        "kimera_errors_total",
        MetricType.COUNTER,
        "Total number of errors",
        labels=["component", "error_type"],
    )
