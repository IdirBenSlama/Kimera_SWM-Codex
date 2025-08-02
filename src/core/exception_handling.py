#!/usr/bin/env python3
"""
KIMERA Exception Handling Framework
===================================

This module provides a comprehensive exception handling framework that ensures
proper error recovery, logging, and graceful degradation throughout the system.

Scientific Principles:
- Fail-safe design: System continues operating with reduced functionality
- Error transparency: All errors are logged with full context
- Recovery strategies: Each error type has defined recovery mechanisms
- Circuit breaker pattern: Prevents cascade failures
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import TypeVar, Callable, Optional, Any, Dict, Union, Type
import asyncio
import logging
import threading
import time

from fastapi import HTTPException
from functools import wraps
from src.utils.kimera_logger import get_logger, LogCategory
import functools
import traceback
logger = get_logger("exception_handler", category=LogCategory.SYSTEM)

T = TypeVar('T')


class ErrorSeverity(Enum):
    """Error severity levels for proper handling strategies."""
    CRITICAL = "critical"  # System cannot continue
    HIGH = "high"         # Major functionality impaired
    MEDIUM = "medium"     # Some features unavailable
    LOW = "low"          # Minor issues, can be ignored


class RecoveryStrategy(Enum):
    """Recovery strategies for different error types."""
    RETRY = "retry"              # Retry the operation
    FALLBACK = "fallback"        # Use fallback mechanism
    DEGRADE = "degrade"          # Operate with reduced functionality
    CIRCUIT_BREAK = "circuit_break"  # Stop attempting operation
    IGNORE = "ignore"            # Log and continue


@dataclass
class ErrorContext:
    """Comprehensive error context for debugging and recovery."""
    error_type: Type[Exception]
    error_message: str
    severity: ErrorSeverity
    operation: str
    timestamp: datetime = field(default_factory=datetime.now)
    traceback: str = field(default_factory=lambda: traceback.format_exc())
    retry_count: int = 0
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY
    additional_context: Dict[str, Any] = field(default_factory=dict)


class CircuitBreaker:
    """Circuit breaker pattern implementation to prevent cascade failures."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.is_open = False
        self._lock = threading.Lock()
    
    def record_success(self):
        """Record a successful operation."""
        with self._lock:
            self.failure_count = 0
            self.is_open = False
    
    def record_failure(self):
        """Record a failed operation."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.failure_count >= self.failure_threshold:
                self.is_open = True
                logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
    
    def can_attempt(self) -> bool:
        """Check if operation can be attempted."""
        with self._lock:
            if not self.is_open:
                return True
            
            # Check if recovery timeout has passed
            if self.last_failure_time:
                time_since_failure = datetime.now() - self.last_failure_time
                if time_since_failure > timedelta(seconds=self.recovery_timeout):
                    logger.info("Circuit breaker attempting recovery")
                    self.is_open = False
                    self.failure_count = 0
                    return True
            
            return False


class ErrorRegistry:
    """Central registry for error handling strategies."""
    
    def __init__(self):
        self._strategies: Dict[Type[Exception], Dict[str, Any]] = {}
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._error_stats: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()
    
    def register_error_strategy(
        self,
        error_type: Type[Exception],
        severity: ErrorSeverity,
        recovery_strategy: RecoveryStrategy,
        fallback_value: Optional[Any] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """Register an error handling strategy for a specific error type."""
        with self._lock:
            self._strategies[error_type] = {
                'severity': severity,
                'recovery_strategy': recovery_strategy,
                'fallback_value': fallback_value,
                'max_retries': max_retries,
                'retry_delay': retry_delay
            }
    
    def get_strategy(self, error_type: Type[Exception]) -> Dict[str, Any]:
        """Get the strategy for handling a specific error type."""
        with self._lock:
            # Look for exact match first
            if error_type in self._strategies:
                return self._strategies[error_type]
            
            # Look for parent classes
            for registered_type, strategy in self._strategies.items():
                if issubclass(error_type, registered_type):
                    return strategy
            
            # Default strategy
            return {
                'severity': ErrorSeverity.MEDIUM,
                'recovery_strategy': RecoveryStrategy.RETRY,
                'fallback_value': None,
                'max_retries': 3,
                'retry_delay': 1.0
            }
    
    def get_circuit_breaker(self, operation: str) -> CircuitBreaker:
        """Get or create a circuit breaker for an operation."""
        with self._lock:
            if operation not in self._circuit_breakers:
                self._circuit_breakers[operation] = CircuitBreaker()
            return self._circuit_breakers[operation]
    
    def record_error(self, operation: str, error_type: Type[Exception]):
        """Record error statistics."""
        with self._lock:
            key = f"{operation}:{error_type.__name__}"
            self._error_stats[key] += 1
    
    def get_error_stats(self) -> Dict[str, int]:
        """Get error statistics."""
        with self._lock:
            return dict(self._error_stats)


# Global error registry
error_registry = ErrorRegistry()

# Register common error strategies
error_registry.register_error_strategy(
    ConnectionError,
    ErrorSeverity.HIGH,
    RecoveryStrategy.RETRY,
    max_retries=5,
    retry_delay=2.0
)

error_registry.register_error_strategy(
    TimeoutError,
    ErrorSeverity.MEDIUM,
    RecoveryStrategy.CIRCUIT_BREAK,
    max_retries=3,
    retry_delay=1.0
)

error_registry.register_error_strategy(
    ValueError,
    ErrorSeverity.LOW,
    RecoveryStrategy.FALLBACK,
    fallback_value=None
)

error_registry.register_error_strategy(
    MemoryError,
    ErrorSeverity.CRITICAL,
    RecoveryStrategy.DEGRADE
)


def safe_operation(
    operation: str,
    fallback: Optional[T] = None,
    reraise: bool = False,
    log_level: int = logging.ERROR,
    use_circuit_breaker: bool = False,
    severity: Optional[ErrorSeverity] = None,
):
    """
    A decorator to safely wrap an operation, providing logging, fallback,
    and graceful error handling.

    Args:
        operation (str): A description of the operation being performed.
        fallback (Optional[T]): A default value to return on failure.
                                 If None, an HTTPException is raised.
        reraise (bool): If True, the original exception is re-raised after logging.
                        This is useful for background tasks where the exception
                        needs to be handled by a higher-level manager.
        log_level (int): The logging level to use for the exception.
        use_circuit_breaker (bool): If True, use circuit breaker pattern for the operation.
        severity (Optional[ErrorSeverity]): Override error severity for this operation.
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        # Get circuit breaker if enabled
        circuit_breaker = error_registry.get_circuit_breaker(operation) if use_circuit_breaker else None
        
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Check circuit breaker before attempting operation
            if circuit_breaker and not circuit_breaker.can_attempt():
                logger.warning(f"Circuit breaker is open for operation '{operation}', returning fallback")
                if fallback is not None:
                    return fallback
                raise HTTPException(
                    status_code=503,
                    detail=f"Service '{operation}' is temporarily unavailable due to circuit breaker."
                )
            
            try:
                result = await func(*args, **kwargs)
                # Record success if using circuit breaker
                if circuit_breaker:
                    circuit_breaker.record_success()
                return result
            except Exception as e:
                # Record failure if using circuit breaker
                if circuit_breaker:
                    circuit_breaker.record_failure()
                
                # Record error in registry
                error_registry.record_error(operation, type(e))
                
                # Log the exception with structured details
                logger.log(
                    log_level,
                    f"Operation '{operation}' failed: {e}",
                    operation=operation,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    exc_info=True,
                    func_name=func.__name__,
                    func_args=args,
                    func_kwargs=kwargs
                )
                
                if reraise:
                    raise
                    
                if fallback is not None:
                    return fallback
                    
                # For API-facing operations, raise a graceful HTTP exception
                raise HTTPException(
                    status_code=503,
                    detail=f"Service '{operation}' is temporarily unavailable due to an internal error."
                )
        
        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Check circuit breaker before attempting operation
            if circuit_breaker and not circuit_breaker.can_attempt():
                logger.warning(f"Circuit breaker is open for operation '{operation}', returning fallback")
                if fallback is not None:
                    return fallback
                return None
            
            try:
                result = func(*args, **kwargs)
                # Record success if using circuit breaker
                if circuit_breaker:
                    circuit_breaker.record_success()
                return result
            except Exception as e:
                # Record failure if using circuit breaker
                if circuit_breaker:
                    circuit_breaker.record_failure()
                
                # Record error in registry
                error_registry.record_error(operation, type(e))
                
                logger.log(
                    log_level,
                    f"Operation '{operation}' failed: {e}",
                    operation=operation,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    exc_info=True,
                    func_name=func.__name__,
                    func_args=args,
                    func_kwargs=kwargs
                )
                
                if reraise:
                    raise
                    
                if fallback is not None:
                    return fallback
                
                # For non-API operations, we can't raise HTTPException.
                # Depending on the desired behavior, we might return None or another default.
                # For now, we will just return None if no fallback is provided.
                return None

        # Check if the wrapped function is async or sync
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
            
    return decorator


def get_error_report() -> Dict[str, Any]:
    """Get comprehensive error report for monitoring."""
    return {
        'error_stats': error_registry.get_error_stats(),
        'circuit_breakers': {
            name: {
                'is_open': cb.is_open,
                'failure_count': cb.failure_count,
                'last_failure': cb.last_failure_time.isoformat() if cb.last_failure_time else None
            }
            for name, cb in error_registry._circuit_breakers.items()
        },
        'timestamp': datetime.now().isoformat()
    }


# Example usage patterns
if __name__ == "__main__":
    # Example of using the safe_operation decorator
    
    @safe_operation(
        operation="database_query",
        fallback=[],
        use_circuit_breaker=True
    )
    async def fetch_data(query: str):
        # Simulated database operation
        if "error" in query:
            raise ConnectionError("Database connection failed")
        return [{"id": 1, "data": "example"}]
    
    @safe_operation(
        operation="external_api_call",
        fallback={"status": "unavailable"},
        severity=ErrorSeverity.HIGH
    )
    async def call_external_api(endpoint: str):
        # Simulated API call
        if "timeout" in endpoint:
            raise TimeoutError("API request timed out")
        return {"status": "success", "data": "response"}
    
    # Run examples
    async def test_error_handling():
        # Test successful operation
        result = await fetch_data("SELECT * FROM users")
        print(f"Success: {result}")
        
        # Test error with fallback
        result = await fetch_data("error query")
        print(f"Error handled with fallback: {result}")
        
        # Test circuit breaker
        for i in range(10):
            try:
                result = await call_external_api("timeout endpoint")
            except HTTPException as e:
                print(f"Request {i}: {e.detail}")
        
        # Get error report
        report = get_error_report()
        print(f"Error report: {report}")
    
    # Run the test
    asyncio.run(test_error_handling())