# -*- coding: utf-8 -*-
"""
System Health Monitor
--------------------
This module provides system health monitoring capabilities.
"""

import logging
import threading
from datetime import datetime
from typing import Any, Dict, Optional

import psutil

logger = logging.getLogger(__name__)
class SystemHealthMonitor:
    """Auto-generated class."""
    pass
    """Monitor system health metrics."""

    def __init__(self):
        self.metrics = {}
        self._lock = threading.Lock()
        self._update_metrics()
        logger.info("SystemHealthMonitor initialized")

    def _update_metrics(self):
        """Update system metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            with self._lock:
                self.metrics = {
                    "cpu": {"percent": cpu_percent, "count": psutil.cpu_count()},
                    "memory": {
                        "total": memory.total,
                        "available": memory.available,
                        "percent": memory.percent,
                        "used": memory.used,
                    },
                    "disk": {
                        "total": disk.total,
                        "used": disk.used,
                        "free": disk.free,
                        "percent": disk.percent,
                    },
                    "timestamp": datetime.now().isoformat(),
                }
        except Exception as e:
            logger.error(f"Failed to update system metrics: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        self._update_metrics()
        with self._lock:
            return self.metrics.copy()

    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status."""
        metrics = self.get_metrics()

        # Determine health status based on thresholds
        cpu_health = (
            "healthy"
            if metrics["cpu"]["percent"] < 80
            else "warning" if metrics["cpu"]["percent"] < 90 else "critical"
        )
        memory_health = (
            "healthy"
            if metrics["memory"]["percent"] < 80
            else "warning" if metrics["memory"]["percent"] < 90 else "critical"
        )
        disk_health = (
            "healthy"
            if metrics["disk"]["percent"] < 80
            else "warning" if metrics["disk"]["percent"] < 90 else "critical"
        )

        overall_health = "healthy"
        if any(h == "critical" for h in [cpu_health, memory_health, disk_health]):
            overall_health = "critical"
        elif any(h == "warning" for h in [cpu_health, memory_health, disk_health]):
            overall_health = "warning"

        return {
            "overall": overall_health,
            "cpu": cpu_health,
            "memory": memory_health,
            "disk": disk_health,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }


# Singleton instance
_monitor_instance = None
_monitor_lock = threading.Lock()


def get_health_monitor() -> SystemHealthMonitor:
    """Get the singleton health monitor instance."""
    global _monitor_instance

    with _monitor_lock:
        if _monitor_instance is None:
            _monitor_instance = SystemHealthMonitor()

    return _monitor_instance
