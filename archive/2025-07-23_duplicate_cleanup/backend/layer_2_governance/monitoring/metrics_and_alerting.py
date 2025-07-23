"""
Metrics and Alerting for KIMERA System
Implements Prometheus metrics and basic alerting rules
Phase 3, Week 9: Monitoring Infrastructure
"""

import logging
import time
from typing import Optional, Dict, Any, List, Callable
import asyncio
from datetime import datetime

from prometheus_client import (
    start_http_server, Counter, Gauge, Histogram, Summary, REGISTRY
)

from backend.config import get_settings

logger = logging.getLogger(__name__)


class MetricsManager:
    """
    Manages Prometheus metrics for KIMERA
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.is_configured = False
        self.metrics: Dict[str, Any] = {}
    
    def configure_metrics(self):
        """Configure and start Prometheus metrics server"""
        if not self.settings.monitoring.enabled or not self.settings.get_feature("prometheus_metrics"):
            logger.info("Prometheus metrics are disabled")
            return
        
        if self.is_configured:
            return
        
        try:
            port = self.settings.monitoring.metrics_port
            start_http_server(port)
            logger.info(f"Prometheus metrics server started on port {port}")
            self.is_configured = True
            
            # Register default metrics
            self._register_default_metrics()
            
        except Exception as e:
            logger.error(f"Failed to start Prometheus metrics server: {e}")
    
    def _register_default_metrics(self):
        """Register default application metrics"""
        self.register_counter(
            "kimera_requests_total",
            "Total number of requests",
            ["method", "endpoint", "status_code"]
        )
        self.register_histogram(
            "kimera_request_duration_seconds",
            "Request duration in seconds",
            ["method", "endpoint"]
        )
        self.register_gauge(
            "kimera_active_requests",
            "Number of active requests"
        )
        self.register_gauge(
            "kimera_info",
            "KIMERA application information",
            ["version"]
        )
        self.metrics["kimera_info"].labels(version="0.1.0").set(1)
    
    def register_counter(self, name: str, documentation: str, labelnames: List[str] = []) -> Counter:
        """Register a counter metric"""
        if name not in self.metrics:
            self.metrics[name] = Counter(name, documentation, labelnames)
        return self.metrics[name]
    
    def register_gauge(self, name: str, documentation: str, labelnames: List[str] = []) -> Gauge:
        """Register a gauge metric"""
        if name not in self.metrics:
            self.metrics[name] = Gauge(name, documentation, labelnames)
        return self.metrics[name]
    
    def register_histogram(self, name: str, documentation: str, labelnames: List[str] = []) -> Histogram:
        """Register a histogram metric"""
        if name not in self.metrics:
            self.metrics[name] = Histogram(name, documentation, labelnames)
        return self.metrics[name]
    
    def register_summary(self, name: str, documentation: str, labelnames: List[str] = []) -> Summary:
        """Register a summary metric"""
        if name not in self.metrics:
            self.metrics[name] = Summary(name, documentation, labelnames)
        return self.metrics[name]
    
    def get_metric(self, name: str) -> Optional[Any]:
        """Get a registered metric"""
        return self.metrics.get(name)


class AlertingManager:
    """
    Manages alerting rules and notifications
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.alert_rules: List[Dict[str, Any]] = []
        self.notification_channels: List[Callable] = []
        self._alerting_task: Optional[asyncio.Task] = None
    
    def configure_alerting(self):
        """Configure alerting system"""
        if not self.settings.monitoring.enabled or not self.settings.get_feature("alerting"):
            logger.info("Alerting is disabled")
            return
        
        # Load alert rules
        self._load_alert_rules()
        
        # Configure notification channels
        self._configure_notification_channels()
        
        # Start alerting task
        if self.alert_rules:
            self._alerting_task = asyncio.create_task(self._alerting_loop())
            logger.info("Alerting system configured")
    
    def _load_alert_rules(self):
        """Load alert rules from configuration"""
        # In a real application, this would load from a file or database
        self.alert_rules = [
            {
                "name": "HighErrorRate",
                "expression": "sum(rate(kimera_requests_total{status_code=~\"5..\"}[5m])) / sum(rate(kimera_requests_total[5m])) > 0.05",
                "for": "5m",
                "labels": {"severity": "critical"},
                "annotations": {
                    "summary": "High API error rate",
                    "description": "More than 5% of API requests are failing"
                }
            },
            {
                "name": "HighLatency",
                "expression": "histogram_quantile(0.95, sum(rate(kimera_request_duration_seconds_bucket[5m])) by (le)) > 1.0",
                "for": "10m",
                "labels": {"severity": "warning"},
                "annotations": {
                    "summary": "High API latency",
                    "description": "P95 latency is over 1 second"
                }
            }
        ]
        logger.info(f"Loaded {len(self.alert_rules)} alert rules")
    
    def _configure_notification_channels(self):
        """Configure notification channels"""
        # Example: log to console
        self.add_notification_channel(self.log_notification)
        
        # In a real application, you would add channels for email, Slack, etc.
    
    def add_notification_channel(self, channel: Callable):
        """Add a notification channel"""
        self.notification_channels.append(channel)
    
    async def _alerting_loop(self):
        """Periodically check alert rules"""
        while True:
            await asyncio.sleep(60)  # Check every minute
            await self.check_alerts()
    
    async def check_alerts(self):
        """Check all alert rules"""
        # This is a simplified check. A real implementation would query Prometheus.
        for rule in self.alert_rules:
            # Simulate checking the rule
            if self._simulate_alert_check(rule):
                await self.trigger_alert(rule)
    
    def _simulate_alert_check(self, rule: Dict[str, Any]) -> bool:
        """Simulate checking an alert rule"""
        # In a real system, you would query Prometheus here
        return False  # For demonstration, no alerts are fired
    
    async def trigger_alert(self, rule: Dict[str, Any]):
        """Trigger an alert and send notifications"""
        alert_data = {
            "rule_name": rule["name"],
            "severity": rule["labels"]["severity"],
            "summary": rule["annotations"]["summary"],
            "description": rule["annotations"]["description"],
            "timestamp": datetime.now().isoformat()
        }
        
        for channel in self.notification_channels:
            try:
                await channel(alert_data)
            except Exception as e:
                logger.error(f"Failed to send notification: {e}")
    
    async def log_notification(self, alert_data: Dict[str, Any]):
        """Log alert notification to console"""
        logger.warning(
            f"ALERT: {alert_data['summary']} (severity: {alert_data['severity']})"
        )


# Global metrics and alerting managers
_metrics_manager: Optional[MetricsManager] = None
_alerting_manager: Optional[AlertingManager] = None


def get_metrics_manager() -> MetricsManager:
    """Get global metrics manager instance"""
    global _metrics_manager
    if _metrics_manager is None:
        _metrics_manager = MetricsManager()
        _metrics_manager.configure_metrics()
    return _metrics_manager


def get_alerting_manager() -> AlertingManager:
    """Get global alerting manager instance"""
    global _alerting_manager
    if _alerting_manager is None:
        _alerting_manager = AlertingManager()
        _alerting_manager.configure_alerting()
    return _alerting_manager


# Decorators for metrics
def track_requests(endpoint: str):
    """Decorator to track requests for an endpoint"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            metrics_manager = get_metrics_manager()
            requests_total = metrics_manager.get_metric("kimera_requests_total")
            request_duration = metrics_manager.get_metric("kimera_request_duration_seconds")
            active_requests = metrics_manager.get_metric("kimera_active_requests")
            
            method = kwargs.get("request").method if "request" in kwargs else "UNKNOWN"
            
            active_requests.inc()
            start_time = time.time()
            
            try:
                response = await func(*args, **kwargs)
                status_code = response.status_code
                return response
            except Exception as e:
                status_code = 500
                raise e
            finally:
                duration = time.time() - start_time
                requests_total.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
                request_duration.labels(method=method, endpoint=endpoint).observe(duration)
                active_requests.dec()
        
        return wrapper
    
    return decorator