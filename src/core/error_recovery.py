"""
KIMERA Error Recovery System
============================

Comprehensive error handling and recovery mechanisms for the KIMERA system.
Implements circuit breaker pattern, retry logic, and graceful degradation.

Features:
- Circuit breaker pattern for preventing cascade failures
- Exponential backoff retry logic
- Graceful degradation strategies
- Error categorization and routing
- Recovery action suggestions
"""

import asyncio
import functools
import logging
import random
import time
import traceback
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ErrorSeverity(Enum):
    """Error severity levels."""

    LOW = "low"  # Can be ignored or logged
    MEDIUM = "medium"  # Should be handled but not critical
    HIGH = "high"  # Requires immediate handling
    CRITICAL = "critical"  # System-threatening errors


class ErrorCategory(Enum):
    """Error categories for specialized handling."""

    TENSOR_SHAPE = "tensor_shape"
    MEMORY = "memory"
    GPU = "gpu"
    NETWORK = "network"
    DATABASE = "database"
    VALIDATION = "validation"
    TIMEOUT = "timeout"
    AUTHENTICATION = "authentication"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Context information for error handling."""

    error: Exception
    category: ErrorCategory
    severity: ErrorSeverity
    timestamp: datetime = field(default_factory=datetime.now)
    component: str = ""
    operation: str = ""
    retry_count: int = 0
    additional_info: Dict[str, Any] = field(default_factory=dict)
    stack_trace: str = field(default_factory=lambda: traceback.format_exc())


@dataclass
class RecoveryAction:
    """Suggested recovery action for an error."""

    action_type: str
    description: str
    handler: Optional[Callable] = None
    parameters: Dict[str, Any] = field(default_factory=dict)


class CircuitBreakerState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing if recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5
    recovery_timeout: timedelta = timedelta(seconds=60)
    expected_exception: Type[Exception] = Exception
    success_threshold: int = 2


class CircuitBreaker:
    """
    Circuit breaker implementation to prevent cascade failures.

    States:
    - CLOSED: Normal operation, calls pass through
    - OPEN: Too many failures, calls are rejected
    - HALF_OPEN: Testing recovery, limited calls allowed
    """

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state_change_callbacks: List[Callable] = []

    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function through circuit breaker."""
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self._transition_to_half_open()
            else:
                raise Exception(f"Circuit breaker is OPEN for {func.__name__}")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.config.expected_exception as e:
            self._on_failure()
            raise e

    async def async_call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute async function through circuit breaker."""
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self._transition_to_half_open()
            else:
                raise Exception(f"Circuit breaker is OPEN for {func.__name__}")

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.config.expected_exception as e:
            self._on_failure()
            raise e

    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0

        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self._transition_to_closed()

    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        self.success_count = 0

        if self.failure_count >= self.config.failure_threshold:
            self._transition_to_open()

    def _should_attempt_reset(self) -> bool:
        """Check if we should try to reset from OPEN state."""
        return (
            self.last_failure_time
            and datetime.now() - self.last_failure_time >= self.config.recovery_timeout
        )

    def _transition_to_closed(self):
        """Transition to CLOSED state."""
        logger.info(f"Circuit breaker transitioning to CLOSED")
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self._notify_state_change()

    def _transition_to_open(self):
        """Transition to OPEN state."""
        logger.warning(
            f"Circuit breaker transitioning to OPEN after {self.failure_count} failures"
        )
        self.state = CircuitBreakerState.OPEN
        self._notify_state_change()

    def _transition_to_half_open(self):
        """Transition to HALF_OPEN state."""
        logger.info(f"Circuit breaker transitioning to HALF_OPEN")
        self.state = CircuitBreakerState.HALF_OPEN
        self.success_count = 0
        self._notify_state_change()

    def _notify_state_change(self):
        """Notify callbacks of state change."""
        for callback in self.state_change_callbacks:
            try:
                callback(self.state)
            except Exception as e:
                logger.error(f"Error in circuit breaker callback: {e}")

    def add_state_change_callback(self, callback: Callable):
        """Add callback for state changes."""
        self.state_change_callbacks.append(callback)


class RetryConfig:
    """Configuration for retry logic."""

    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        exceptions: tuple = (Exception,),
    ):
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.exceptions = exceptions


def retry_with_backoff(config: Optional[RetryConfig] = None):
    """
    Decorator for retry logic with exponential backoff.

    Example:
        @retry_with_backoff(RetryConfig(max_attempts=5))
        async def flaky_operation():
            # Operation that might fail
            pass
    """
    if config is None:
        config = RetryConfig()

    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(config.max_attempts):
                try:
                    return await func(*args, **kwargs)
                except config.exceptions as e:
                    last_exception = e

                    if attempt == config.max_attempts - 1:
                        logger.error(
                            f"Max retries ({config.max_attempts}) exceeded for {func.__name__}"
                        )
                        raise

                    # Calculate delay with exponential backoff
                    delay = min(
                        config.initial_delay * (config.exponential_base**attempt),
                        config.max_delay,
                    )

                    # Add jitter to prevent thundering herd
                    if config.jitter:
                        delay *= 0.5 + random.random()

                    logger.warning(
                        f"Retry {attempt + 1}/{config.max_attempts} for {func.__name__} "
                        f"after {delay:.2f}s delay. Error: {e}"
                    )

                    await asyncio.sleep(delay)

            raise last_exception

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(config.max_attempts):
                try:
                    return func(*args, **kwargs)
                except config.exceptions as e:
                    last_exception = e

                    if attempt == config.max_attempts - 1:
                        logger.error(
                            f"Max retries ({config.max_attempts}) exceeded for {func.__name__}"
                        )
                        raise

                    # Calculate delay
                    delay = min(
                        config.initial_delay * (config.exponential_base**attempt),
                        config.max_delay,
                    )

                    if config.jitter:
                        delay *= 0.5 + random.random()

                    logger.warning(
                        f"Retry {attempt + 1}/{config.max_attempts} for {func.__name__} "
                        f"after {delay:.2f}s delay. Error: {e}"
                    )

                    time.sleep(delay)

            raise last_exception

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


class ErrorRecoveryManager:
    """
    Central error recovery manager for KIMERA system.

    Provides:
    - Error categorization
    - Recovery action suggestions
    - Circuit breaker management
    - Error statistics and monitoring
    """

    def __init__(self):
        self.error_handlers: Dict[ErrorCategory, List[Callable]] = defaultdict(list)
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.error_history: deque = deque(maxlen=1000)
        self.recovery_actions: Dict[ErrorCategory, List[RecoveryAction]] = (
            self._init_recovery_actions()
        )

        # Statistics
        self.error_counts: Dict[ErrorCategory, int] = defaultdict(int)
        self.recovery_success_counts: Dict[ErrorCategory, int] = defaultdict(int)

    def _init_recovery_actions(self) -> Dict[ErrorCategory, List[RecoveryAction]]:
        """Initialize default recovery actions for each error category."""
        return {
            ErrorCategory.TENSOR_SHAPE: [
                RecoveryAction(
                    "reshape_tensor",
                    "Attempt to reshape tensor to expected dimensions",
                    handler=self._reshape_tensor_handler,
                ),
                RecoveryAction(
                    "use_fallback_shape",
                    "Use fallback tensor shape",
                    parameters={"fallback_shape": (1, 1024)},
                ),
            ],
            ErrorCategory.MEMORY: [
                RecoveryAction(
                    "clear_cache",
                    "Clear memory caches and retry",
                    handler=self._clear_memory_handler,
                ),
                RecoveryAction(
                    "reduce_batch_size",
                    "Reduce batch size to lower memory usage",
                    parameters={"reduction_factor": 0.5},
                ),
            ],
            ErrorCategory.GPU: [
                RecoveryAction(
                    "fallback_to_cpu",
                    "Fall back to CPU processing",
                    handler=self._gpu_fallback_handler,
                ),
                RecoveryAction(
                    "reset_gpu",
                    "Reset GPU and clear memory",
                    handler=self._reset_gpu_handler,
                ),
            ],
            ErrorCategory.DATABASE: [
                RecoveryAction(
                    "reconnect",
                    "Attempt to reconnect to database",
                    parameters={"max_attempts": 3},
                ),
                RecoveryAction("use_cache", "Use cached data if available"),
            ],
            ErrorCategory.NETWORK: [
                RecoveryAction(
                    "retry_request",
                    "Retry network request with backoff",
                    parameters={"backoff_strategy": "exponential"},
                ),
                RecoveryAction("use_offline_mode", "Switch to offline mode"),
            ],
            ErrorCategory.TIMEOUT: [
                RecoveryAction(
                    "increase_timeout",
                    "Increase timeout and retry",
                    parameters={"multiplier": 2.0},
                ),
                RecoveryAction("async_processing", "Switch to asynchronous processing"),
            ],
        }

    def register_handler(self, category: ErrorCategory, handler: Callable):
        """Register an error handler for a specific category."""
        self.error_handlers[category].append(handler)

    def create_circuit_breaker(
        self, name: str, config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """Create or get a circuit breaker."""
        if name not in self.circuit_breakers:
            if config is None:
                config = CircuitBreakerConfig()
            self.circuit_breakers[name] = CircuitBreaker(config)

        return self.circuit_breakers[name]

    async def handle_error(self, context: ErrorContext) -> Optional[Any]:
        """
        Handle an error with appropriate recovery strategy.

        Args:
            context: Error context information

        Returns:
            Recovery result if successful, None otherwise
        """
        # Log error
        self._log_error(context)

        # Update statistics
        self.error_counts[context.category] += 1
        self.error_history.append(context)

        # Get recovery actions
        actions = self.recovery_actions.get(context.category, [])

        # Try recovery actions
        for action in actions:
            try:
                if action.handler:
                    result = await self._execute_recovery_action(action, context)
                    if result is not None:
                        self.recovery_success_counts[context.category] += 1
                        logger.info(
                            f"✅ Recovery successful using {action.action_type}"
                        )
                        return result
                else:
                    logger.info(f"ℹ️ Suggested recovery: {action.description}")
            except Exception as e:
                logger.error(f"Recovery action {action.action_type} failed: {e}")

        # Call registered handlers
        for handler in self.error_handlers[context.category]:
            try:
                result = await handler(context)
                if result is not None:
                    return result
            except Exception as e:
                logger.error(f"Error handler failed: {e}")

        return None

    def _log_error(self, context: ErrorContext):
        """Log error with appropriate severity."""
        message = (
            f"Error in {context.component}.{context.operation}: "
            f"{context.error.__class__.__name__}: {str(context.error)}"
        )

        if context.severity == ErrorSeverity.CRITICAL:
            logger.critical(message)
        elif context.severity == ErrorSeverity.HIGH:
            logger.error(message)
        elif context.severity == ErrorSeverity.MEDIUM:
            logger.warning(message)
        else:
            logger.info(message)

    async def _execute_recovery_action(
        self, action: RecoveryAction, context: ErrorContext
    ) -> Optional[Any]:
        """Execute a recovery action."""
        if asyncio.iscoroutinefunction(action.handler):
            return await action.handler(context, **action.parameters)
        else:
            return action.handler(context, **action.parameters)

    # Default recovery handlers
    def _reshape_tensor_handler(self, context: ErrorContext, **kwargs):
        """Handle tensor reshape errors."""
        logger.info("Attempting tensor reshape recovery")
        # Implementation would depend on tensor library
        return None

    def _clear_memory_handler(self, context: ErrorContext, **kwargs):
        """Handle memory clearing."""
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Memory cleared")
        return True

    def _gpu_fallback_handler(self, context: ErrorContext, **kwargs):
        """Handle GPU fallback."""
        logger.info("Falling back to CPU processing")
        context.additional_info["device"] = "cpu"
        return True

    def _reset_gpu_handler(self, context: ErrorContext, **kwargs):
        """Handle GPU reset."""
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        logger.info("GPU reset completed")
        return True

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics."""
        total_errors = sum(self.error_counts.values())
        total_recoveries = sum(self.recovery_success_counts.values())

        category_stats = {}
        for category in ErrorCategory:
            errors = self.error_counts[category]
            recoveries = self.recovery_success_counts[category]
            recovery_rate = recoveries / errors if errors > 0 else 0

            category_stats[category.value] = {
                "errors": errors,
                "recoveries": recoveries,
                "recovery_rate": recovery_rate,
            }

        # Circuit breaker states
        cb_states = {name: cb.state.value for name, cb in self.circuit_breakers.items()}

        return {
            "total_errors": total_errors,
            "total_recoveries": total_recoveries,
            "overall_recovery_rate": (
                total_recoveries / total_errors if total_errors > 0 else 0
            ),
            "categories": category_stats,
            "circuit_breakers": cb_states,
            "recent_errors": len(self.error_history),
        }


# Global error recovery manager
_error_recovery_manager: Optional[ErrorRecoveryManager] = None


def get_error_recovery_manager() -> ErrorRecoveryManager:
    """Get the global error recovery manager."""
    global _error_recovery_manager
    if _error_recovery_manager is None:
        _error_recovery_manager = ErrorRecoveryManager()
    return _error_recovery_manager


# Convenience decorator for error recovery
def with_error_recovery(
    category: ErrorCategory = ErrorCategory.UNKNOWN,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
):
    """
    Decorator to add error recovery to a function.

    Example:
        @with_error_recovery(ErrorCategory.DATABASE, ErrorSeverity.HIGH)
        async def database_operation():
            # Database operation that might fail
            pass
    """

    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                context = ErrorContext(
                    error=e,
                    category=category,
                    severity=severity,
                    component=func.__module__,
                    operation=func.__name__,
                )

                manager = get_error_recovery_manager()
                result = await manager.handle_error(context)

                if result is not None:
                    return result
                else:
                    raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = ErrorContext(
                    error=e,
                    category=category,
                    severity=severity,
                    component=func.__module__,
                    operation=func.__name__,
                )

                # For sync functions, we can't use async recovery
                logger.error(f"Error in {func.__name__}: {e}")
                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator
