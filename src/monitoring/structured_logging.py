"""
Structured Logging for KIMERA System
Implements structured logging with correlation IDs and context
Phase 3, Week 9: Monitoring Infrastructure
"""

import json
import logging
import sys
import traceback
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from pythonjsonlogger import jsonlogger

from src.config import get_settings, is_development, is_production


class CorrelationIDManager:
    """Manages correlation IDs for request tracking"""

    def __init__(self):
        self._correlation_id: Optional[str] = None

    def set(self, correlation_id: Optional[str] = None) -> str:
        """Set correlation ID"""
        self._correlation_id = correlation_id or str(uuid.uuid4())
        return self._correlation_id

    def get(self) -> Optional[str]:
        """Get current correlation ID"""
        return self._correlation_id

    def clear(self):
        """Clear correlation ID"""
        self._correlation_id = None


# Global correlation ID manager
correlation_id_manager = CorrelationIDManager()


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON log formatter"""

    def add_fields(
        self,
        log_record: Dict[str, Any],
        record: logging.LogRecord,
        message_dict: Dict[str, Any],
    ):
        """Add custom fields to log record"""
        super().add_fields(log_record, record, message_dict)

        # Add correlation ID
        correlation_id = correlation_id_manager.get()
        if correlation_id:
            log_record["correlation_id"] = correlation_id

        # Add timestamp
        log_record["timestamp"] = datetime.fromtimestamp(record.created).isoformat()

        # Add exception info
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
            log_record["exc_text"] = logging.Formatter.formatException(
                self, record.exc_info
            )

        # Add context from extra
        if hasattr(record, "extra_context") and isinstance(record.extra_context, dict):
            log_record.update(record.extra_context)


class LoggingManager:
    """
    Manages logging configuration for KIMERA
    """

    def __init__(self):
        self.settings = get_settings()
        self.log_level = self.settings.logging.level.value
        self.is_configured = False

    def configure_logging(self):
        """Configure logging for the application"""
        if self.is_configured:
            return

        # Determine formatter
        if self.settings.logging.structured and is_production():
            formatter = CustomJsonFormatter(
                "%(timestamp)s %(name)s %(levelname)s %(message)s"
            )
        else:
            formatter = logging.Formatter(self.settings.logging.format)

        # Configure handlers
        handlers = self._get_handlers(formatter)

        # Configure root logger
        logging.basicConfig(
            level=self.log_level,
            handlers=handlers,
            force=True,  # Override existing config
        )

        # Set log levels for third-party libraries
        self._set_third_party_log_levels()

        self.is_configured = True
        logging.info(
            f"Logging configured (level: {self.log_level}, structured: {self.settings.logging.structured})"
        )

    def _get_handlers(self, formatter: logging.Formatter) -> List[logging.Handler]:
        """Get log handlers based on configuration"""
        handlers = []

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)

        # File handler
        if self.settings.logging.file_enabled:
            try:
                from logging.handlers import RotatingFileHandler

                log_dir = self.settings.paths.logs_dir
                log_dir.mkdir(parents=True, exist_ok=True)
                log_file = log_dir / "kimera.log"

                file_handler = RotatingFileHandler(
                    log_file, maxBytes=10 * 1024 * 1024, backupCount=5  # 10MB
                )
                file_handler.setFormatter(formatter)
                handlers.append(file_handler)

            except Exception as e:
                logging.warning(f"Failed to configure file logging: {e}")

        return handlers

    def _set_third_party_log_levels(self):
        """Set log levels for noisy third-party libraries"""
        logging.getLogger("uvicorn").setLevel(logging.WARNING)
        logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("websockets").setLevel(logging.WARNING)

    def get_logger(self, name: str) -> "ContextualLogger":
        """Get a logger with context support"""
        return ContextualLogger(logging.getLogger(name))


class ContextualLogger:
    """
    Logger wrapper for adding context to log messages
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.context: Dict[str, Any] = {}

    def with_context(self, **kwargs) -> "ContextualLogger":
        """Add context to the logger"""
        new_logger = ContextualLogger(self.logger)
        new_logger.context = {**self.context, **kwargs}
        return new_logger

    def _log(self, level: int, msg: str, *args, **kwargs):
        """Log message with context"""
        extra = kwargs.pop("extra", {})
        extra["extra_context"] = self.context
        kwargs["extra"] = extra

        self.logger.log(level, msg, *args, **kwargs)

    def debug(self, msg: str, *args, **kwargs):
        self._log(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        self._log(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        self._log(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        self._log(logging.ERROR, msg, *args, **kwargs)

    def exception(self, msg: str, *args, **kwargs):
        kwargs["exc_info"] = True
        self._log(logging.ERROR, msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs):
        self._log(logging.CRITICAL, msg, *args, **kwargs)


# Global logging manager
_logging_manager: Optional[LoggingManager] = None


def get_logging_manager() -> LoggingManager:
    """Get global logging manager instance"""
    global _logging_manager
    if _logging_manager is None:
        _logging_manager = LoggingManager()
        _logging_manager.configure_logging()
    return _logging_manager


def get_logger(name: str) -> ContextualLogger:
    """Get a contextual logger"""
    manager = get_logging_manager()
    return manager.get_logger(name)


# Middleware for correlation ID
async def correlation_id_middleware(request, call_next):
    """FastAPI middleware to manage correlation IDs"""
    # Get correlation ID from header or create new one
    correlation_id = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
    correlation_id_manager.set(correlation_id)

    # Process request
    response = await call_next(request)

    # Add correlation ID to response header
    response.headers["X-Correlation-ID"] = correlation_id_manager.get()

    # Clear correlation ID
    correlation_id_manager.clear()

    return response


# Example usage
if __name__ == "__main__":
    # Configure logging
    manager = get_logging_manager()

    # Get logger
    logger = get_logger("my_app")

    # Set correlation ID for a request
    correlation_id_manager.set("request-123")

    # Log messages
    logger.info("This is an info message")

    # Log with context
    context_logger = logger.with_context(user_id=123, operation="data_processing")
    context_logger.info("Processing user data")

    try:
        1 / 0
    except Exception:
        context_logger.exception("An error occurred")

    # Clear correlation ID
    correlation_id_manager.clear()

    logger.info("This message has no correlation ID")
