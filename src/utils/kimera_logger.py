"""
KIMERA SWM - Comprehensive Structured Logging Framework
=====================================================

Zero-debugging constraint compliant logging system with:
- Structured logging with categories
- Context tracking and correlation
- GPU operation monitoring
- Performance tracking
- Security event logging
- Error context preservation
"""

import json
import logging
import logging.handlers
import os
import sys
import threading
import time
import traceback
import uuid
import weakref
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Import portalocker but handle Windows/Unix differences gracefully
try:
    import portalocker
except ImportError:
    portalocker = None

# GPU monitoring imports
try:
    import torch

    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

try:
    import psutil

    SYSTEM_MONITORING = True
except ImportError:
    SYSTEM_MONITORING = False


class LogLevel(Enum):
    """Structured log levels for Kimera system"""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    SECURITY = "security"
    PERFORMANCE = "performance"
    GPU = "gpu"


class LogCategory(Enum):
    """Log categories for different system components"""

    SYSTEM = "system"
    COGNITIVE = "cognitive"
    TRADING = "trading"
    SECURITY = "security"
    DATABASE = "database"
    API = "api"
    GPU = "gpu"
    PERFORMANCE = "performance"
    MEMORY = "memory"
    NETWORK = "network"
    DUAL_SYSTEM = "dual_system"


@dataclass
class LogContext:
    """Structured log context for correlation and analysis"""

    operation_id: str
    component: str
    category: LogCategory
    timestamp: float
    thread_id: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    correlation_id: Optional[str] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    gpu_metrics: Optional[Dict[str, Any]] = None
    security_context: Optional[Dict[str, Any]] = None


class ProcessSafeFileHandler(logging.FileHandler):
    """
    Simple file handler that avoids rotation issues by using basic file writing
    with process-safe locking when available.
    """

    def __init__(self, filename, mode="a", encoding=None, delay=False):
        # Create logs directory if it doesn't exist
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        super().__init__(filename, mode, encoding, delay)
        self.lock_timeout = 5.0  # 5 second timeout for file locks

    def emit(self, record):
        """
        Emit a record with optional file locking for process safety
        """
        try:
            if self.stream is None:
                self.stream = self._open()

            # Try to acquire lock if portalocker is available
            if portalocker:
                try:
                    portalocker.lock(
                        self.stream, portalocker.LOCK_EX | portalocker.LOCK_NB
                    )
                    super().emit(record)
                    portalocker.unlock(self.stream)
                except portalocker.LockException:
                    # If we can't get the lock, just write anyway (better than failing)
                    super().emit(record)
                except Exception:
                    # If locking fails for any reason, just write without lock
                    super().emit(record)
            else:
                # No portalocker available, just write normally
                super().emit(record)

        except Exception:
            # If anything fails, handle gracefully
            self.handleError(record)


class KimeraStructuredLogger:
    """
    Enhanced structured logger with comprehensive context tracking and performance metrics.

    Features:
    - JSON structured logging with context correlation
    - Performance metrics collection
    - GPU monitoring integration
    - Security event tracking
    - Thread-safe operation contexts
    - Process-safe file writing (no rotation)
    """

    def __init__(self, name: str, category: LogCategory = LogCategory.SYSTEM):
        self.name = name
        self.category = category
        self.local_context = threading.local()
        self.logger = logging.getLogger(name)
        self._context_lock = threading.Lock()
        self._thread_contexts = weakref.WeakKeyDictionary()
        self._gpu_baseline = None
        self._setup_structured_logging()

    def _setup_structured_logging(self):
        """Setup structured logging with process-safe handlers"""
        if not self.logger.handlers:
            # Create logs directory
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)

            # Use simple file handler instead of rotating handler
            log_file = log_dir / f"kimera_{self.category.value}.log"
            file_handler = ProcessSafeFileHandler(
                str(log_file), mode="a", encoding="utf-8"
            )

            # Use structured formatter
            file_handler.setFormatter(StructuredFormatter())

            # Add console handler for immediate feedback
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )

            # Add handlers
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
            self.logger.setLevel(logging.DEBUG)

            # Prevent double logging
            self.logger.propagate = False

    def _get_current_context(self) -> Optional[LogContext]:
        """Get current log context from thread-local storage"""
        with self._context_lock:
            return self._thread_contexts.get(threading.current_thread())

    def _set_current_context(self, context: Optional[LogContext]):
        """Set current log context in thread-local storage"""
        with self._context_lock:
            current_thread = threading.current_thread()
            if context:
                self._thread_contexts[current_thread] = context
            elif current_thread in self._thread_contexts:
                del self._thread_contexts[current_thread]

    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics"""
        metrics = {}

        if SYSTEM_MONITORING:
            try:
                process = psutil.Process()
                metrics.update(
                    {
                        "cpu_percent": process.cpu_percent(),
                        "memory_mb": process.memory_info().rss / 1024 / 1024,
                        "memory_percent": process.memory_percent(),
                        "thread_count": process.num_threads(),
                    }
                )
            except Exception:
                pass

        return metrics

    def _collect_gpu_metrics(self) -> Dict[str, Any]:
        """Collect current GPU metrics"""
        metrics = {}

        if GPU_AVAILABLE and torch.cuda.is_available():
            try:
                current_memory = torch.cuda.memory_allocated()
                reserved_memory = torch.cuda.memory_reserved()

                metrics.update(
                    {
                        "gpu_memory_allocated_mb": current_memory / 1024 / 1024,
                        "gpu_memory_reserved_mb": reserved_memory / 1024 / 1024,
                        "gpu_memory_growth_mb": (
                            current_memory - (self._gpu_baseline or 0)
                        )
                        / 1024
                        / 1024,
                        "gpu_utilization": self._get_gpu_utilization(),
                    }
                )
            except Exception:
                pass

        return metrics

    def _get_gpu_utilization(self) -> Optional[float]:
        """Get GPU utilization percentage"""
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return util.gpu
        except Exception:
            return None

    def _log(self, level: int, message: str, **kwargs):
        """Internal logging method with structured context"""
        context = self._get_current_context()

        # Enhance kwargs with context and metrics
        enhanced_kwargs = {
            "component": self.name,
            "category": self.category.value,
            "timestamp": time.time(),
            "thread_id": threading.get_ident(),
            **kwargs,
        }

        # Add context if available
        if context:
            enhanced_kwargs.update(
                {
                    "operation_id": context.operation_id,
                    "session_id": context.session_id,
                    "correlation_id": context.correlation_id,
                }
            )

        # Add system metrics for performance/error logs
        if level >= logging.WARNING:
            enhanced_kwargs["system_metrics"] = self._collect_system_metrics()
            enhanced_kwargs["gpu_metrics"] = self._collect_gpu_metrics()

        # Create structured log record
        extra = {"structured_data": enhanced_kwargs}
        self.logger.log(level, message, extra=extra)

    def debug(self, message: str, **kwargs):
        """Log debug message with context"""
        self._log(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log info message with context"""
        self._log(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message with context"""
        self._log(logging.WARNING, message, **kwargs)

    def error(self, message: str, error: Optional[Exception] = None, **kwargs):
        """Log error message with context and exception details"""
        if error:
            kwargs.update(
                {
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "traceback": traceback.format_exc() if error else None,
                }
            )
        self._log(logging.ERROR, message, **kwargs)

    def critical(self, message: str, error: Optional[Exception] = None, **kwargs):
        """Log critical message with context and exception details"""
        if error:
            kwargs.update(
                {
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "traceback": traceback.format_exc() if error else None,
                }
            )
        self._log(logging.CRITICAL, message, **kwargs)

    def security(self, message: str, **kwargs):
        """Log security event with enhanced context"""
        kwargs["security_event"] = True
        self._log(logging.WARNING, f"SECURITY: {message}", **kwargs)

    def performance(self, message: str, duration_ms: Optional[float] = None, **kwargs):
        """Log performance event with metrics"""
        if duration_ms is not None:
            kwargs["duration_ms"] = duration_ms
        kwargs["performance_event"] = True
        self._log(logging.INFO, f"PERFORMANCE: {message}", **kwargs)

    def gpu_operation(self, message: str, **kwargs):
        """Log GPU operation with memory tracking"""
        kwargs["gpu_metrics"] = self._collect_gpu_metrics()
        kwargs["gpu_operation"] = True
        self._log(logging.INFO, f"GPU: {message}", **kwargs)

    @contextmanager
    def operation_context(self, operation_name: str, **context_kwargs):
        """Context manager for logging an operation with timing and correlation"""
        operation_id = str(uuid.uuid4())
        start_time = time.time()

        parent_context = self._get_current_context()
        correlation_id = (
            parent_context.correlation_id if parent_context else str(uuid.uuid4())
        )

        context = LogContext(
            operation_id=operation_id,
            component=f"{self.name}.{operation_name}",
            category=self.category,
            timestamp=start_time,
            thread_id=str(threading.get_ident()),
            correlation_id=correlation_id,
            **context_kwargs,
        )

        try:
            self._set_current_context(context)
            self.debug(f"BEGIN: {operation_name}", operation_id=operation_id)
            yield context
        finally:
            duration_ms = (time.time() - start_time) * 1000
            self.performance(
                f"END: {operation_name}",
                duration_ms=duration_ms,
                operation_id=operation_id,
            )
            # Restore parent context or clear it
            self._set_current_context(parent_context)


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging"""

    def format(self, record):
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add structured data if available
        if hasattr(record, "structured_data"):
            log_data.update(record.structured_data)

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, default=str)


# Global logger instances for different categories
_loggers: Dict[str, KimeraStructuredLogger] = {}


def get_logger(
    name: str, category: LogCategory = LogCategory.SYSTEM
) -> KimeraStructuredLogger:
    """Get or create a structured logger for the given name and category"""
    key = f"{name}:{category.value}"
    if key not in _loggers:
        _loggers[key] = KimeraStructuredLogger(name, category)
    return _loggers[key]


# Convenience functions for different categories
def get_cognitive_logger(name: str) -> KimeraStructuredLogger:
    """Get cognitive system logger"""
    return get_logger(name, LogCategory.COGNITIVE)


def get_trading_logger(name: str) -> KimeraStructuredLogger:
    """Get trading system logger"""
    return get_logger(name, LogCategory.TRADING)


def get_security_logger(name: str) -> KimeraStructuredLogger:
    """Get security system logger"""
    return get_logger(name, LogCategory.SECURITY)


def get_database_logger(name: str) -> KimeraStructuredLogger:
    """Get database system logger"""
    return get_logger(name, LogCategory.DATABASE)


def get_api_logger(name: str) -> KimeraStructuredLogger:
    """Get API system logger"""
    return get_logger(name, LogCategory.API)


def get_gpu_logger(name: str) -> KimeraStructuredLogger:
    """Get GPU system logger"""
    return get_logger(name, LogCategory.GPU)


def get_performance_logger(name: str) -> KimeraStructuredLogger:
    """Get performance system logger"""
    return get_logger(name, LogCategory.PERFORMANCE)


def get_system_logger(name: str) -> KimeraStructuredLogger:
    """Get system logger (alias for get_logger with SYSTEM category)"""
    return get_logger(name, LogCategory.SYSTEM)


# Configure third-party loggers to reduce noise
def configure_third_party_loggers():
    """Configure third-party loggers to reduce noise"""
    third_party_loggers = [
        "urllib3",
        "requests",
        "transformers",
        "torch",
        "sqlalchemy",
        "httpx",
        "fastapi",
        "uvicorn",
    ]

    for logger_name in third_party_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


# Compatibility function for legacy code
def setup_logger(level=logging.INFO):
    """Setup the basic logger - compatibility function for legacy code"""
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
        root_logger.setLevel(level)
    return root_logger


# Initialize logging configuration
configure_third_party_loggers()
