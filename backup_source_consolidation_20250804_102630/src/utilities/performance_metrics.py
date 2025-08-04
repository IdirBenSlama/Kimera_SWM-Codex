"""
Performance Metrics Utilities
=============================

Provides performance tracking and metrics collection for KIMERA SWM.

Author: KIMERA Development Team
Version: 1.0.0
"""

import time
from typing import Dict, Any, List
from datetime import datetime

class PerformanceMetrics:
    """Performance metrics collection and tracking"""

    def __init__(self):
        self.metrics = {}
        self.start_time = time.time()

    def record_metric(self, name: str, value: float, timestamp: datetime = None):
        """Record a performance metric"""
        if timestamp is None:
            timestamp = datetime.now()

        if name not in self.metrics:
            self.metrics[name] = []

        self.metrics[name].append({
            'value': value,
            'timestamp': timestamp
        })

    def get_metric_average(self, name: str) -> float:
        """Get average value for a metric"""
        if name not in self.metrics:
            return 0.0

        values = [m['value'] for m in self.metrics[name]]
        return sum(values) / len(values) if values else 0.0

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics"""
        return self.metrics.copy()
