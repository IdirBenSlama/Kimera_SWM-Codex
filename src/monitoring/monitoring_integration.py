"""
Monitoring Integration for KIMERA System
Integrates all monitoring components
Phase 3, Week 9: Monitoring Infrastructure
"""

import logging
from typing import Any, Optional

try:
    from config import get_settings
except ImportError:
    # Create placeholders for config
    def get_settings(*args, **kwargs):
        return None


from datetime import datetime

from .distributed_tracing import (
    TracingManager,
    get_tracer,
    get_tracing_manager,
    with_span,
)
from .metrics_and_alerting import (
    AlertingManager,
    MetricsManager,
    get_alerting_manager,
    get_metrics_manager,
    track_requests,
)
from .structured_logging import (
    LoggingManager,
    correlation_id_middleware,
    get_logger,
    get_logging_manager,
)

logger = logging.getLogger(__name__)


class MonitoringManager:
    """
    Central manager for all monitoring components
    """

    def __init__(self):
        self.settings = get_settings()
        self.logging_manager: Optional[LoggingManager] = None
        self.tracing_manager: Optional[TracingManager] = None
        self.metrics_manager: Optional[MetricsManager] = None
        self.alerting_manager: Optional[AlertingManager] = None
        self._initialized = False

        logger.info("MonitoringManager created")

    def initialize(self, app: Optional[Any] = None, engine: Optional[Any] = None):
        """
        Initialize all monitoring components

        Args:
            app: FastAPI application instance
            engine: SQLAlchemy engine instance
        """
        if self._initialized:
            logger.warning("MonitoringManager already initialized")
            return

        # Initialize logging
        self.logging_manager = get_logging_manager()
        self.logging_manager.configure_logging()

        # Initialize tracing
        self.tracing_manager = get_tracing_manager()
        self.tracing_manager.configure_tracing(app=app, engine=engine)

        # Initialize metrics
        self.metrics_manager = get_metrics_manager()
        self.metrics_manager.configure_metrics()

        # Initialize alerting
        self.alerting_manager = get_alerting_manager()
        self.alerting_manager.configure_alerting()

        # Add middleware to app
        if app:
            app.middleware("http")(correlation_id_middleware)
            logger.info("Correlation ID middleware added")

        self._initialized = True
        logger.info("MonitoringManager fully initialized")

    def get_logger(self, name: str):
        """Get a contextual logger"""
        return get_logger(name)

    def get_tracer(self, name: str):
        """Get a tracer"""
        return get_tracer(name)

    def get_metric(self, name: str):
        """Get a metric"""
        if self.metrics_manager:
            return self.metrics_manager.get_metric(name)
        return None


# Global monitoring manager
_monitoring_manager: Optional[MonitoringManager] = None


def get_monitoring_manager() -> MonitoringManager:
    """Get global monitoring manager instance"""
    global _monitoring_manager
    if _monitoring_manager is None:
        _monitoring_manager = MonitoringManager()
    return _monitoring_manager


# Example usage
if __name__ == "__main__":
    # Initialize monitoring
    manager = get_monitoring_manager()
    manager.initialize()

    # Get logger and tracer
    logger = manager.get_logger("my_app")
    tracer = manager.get_tracer("my_app")

    # Use logger
    logger.info("This is a test log message")

    # Use tracer
    with tracer.start_as_current_span("main_operation") as span:
        span.set_attribute("test_attribute", "test_value")
        logger.info("Inside a trace span")

    # Use metrics
    requests_total = manager.get_metric("kimera_requests_total")
    if requests_total:
        requests_total.labels(method="GET", endpoint="/test", status_code=200).inc()

    logger.info("Monitoring components used successfully")
