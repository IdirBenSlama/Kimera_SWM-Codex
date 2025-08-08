#!/usr/bin/env python3
"""
KIMERA SWM System - Resilience Framework
=======================================

Phase 2.1: Error Handling & Resilience Implementation
Provides comprehensive error handling, graceful degradation, and system resilience.

Features:
- Circuit breaker patterns for external dependencies
- Graceful degradation mechanisms
- Fallback strategies for critical components
- Error recovery procedures
- System health monitoring
- Automatic retry logic with exponential backoff

Author: KIMERA Development Team
Date: 2025-01-31
Phase: 2.1 - Error Handling & Resilience
"""

import asyncio
import functools
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union
from datetime import datetime, timedelta
import threading
import traceback
from contextlib import contextmanager

# Configure logging
logger = logging.getLogger(__name__)

class SystemState(Enum):
    """System operational states."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    RECOVERY = "recovery"
    MAINTENANCE = "maintenance"

class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing fast
    HALF_OPEN = "half_open"  # Testing recovery

class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    CATASTROPHIC = 5

@dataclass
class ErrorContext:
    """Context information for error handling."""
    error: Exception
    component: str
    operation: str
    timestamp: datetime
    severity: ErrorSeverity
    retry_count: int = 0
    recovery_attempted: bool = False
    additional_info: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HealthMetrics:
    """System health metrics."""
    component: str
    state: SystemState
    error_rate: float
    response_time: float
    success_rate: float
    last_error: Optional[str] = None
    last_check: datetime = field(default_factory=datetime.now)
    consecutive_failures: int = 0
    uptime: timedelta = field(default_factory=lambda: timedelta(0))

class FallbackStrategy(ABC):
    """Abstract base class for fallback strategies."""
    
    @abstractmethod
    def execute(self, context: ErrorContext) -> Any:
        """Execute the fallback strategy."""
        pass
    
    @abstractmethod
    def is_applicable(self, context: ErrorContext) -> bool:
        """Check if this fallback strategy is applicable."""
        pass

class DefaultFallbackStrategy(FallbackStrategy):
    """Default fallback strategy that returns safe defaults."""
    
    def __init__(self, default_value: Any = None):
        self.default_value = default_value
    
    def execute(self, context: ErrorContext) -> Any:
        """Return the default value."""
        logger.warning(f"Using default fallback for {context.component}: {self.default_value}")
        return self.default_value
    
    def is_applicable(self, context: ErrorContext) -> bool:
        """Always applicable as last resort."""
        return True

class CachedFallbackStrategy(FallbackStrategy):
    """Fallback strategy that uses cached values."""
    
    def __init__(self, cache_manager):
        self.cache_manager = cache_manager
    
    def execute(self, context: ErrorContext) -> Any:
        """Return cached value if available."""
        cache_key = f"{context.component}_{context.operation}"
        cached_value = self.cache_manager.get(cache_key)
        
        if cached_value is not None:
            logger.info(f"Using cached fallback for {context.component}")
            return cached_value
        
        raise ValueError("No cached value available")
    
    def is_applicable(self, context: ErrorContext) -> bool:
        """Applicable if cached value exists."""
        cache_key = f"{context.component}_{context.operation}"
        return self.cache_manager.exists(cache_key)

class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: Type[Exception] = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self._failure_count = 0
        self._last_failure_time = None
        self._state = CircuitBreakerState.CLOSED
        self._lock = threading.Lock()
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap functions with circuit breaker."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self._call(func, *args, **kwargs)
        return wrapper
    
    def _call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker logic."""
        with self._lock:
            if self._state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self._state = CircuitBreakerState.HALF_OPEN
                else:
                    raise Exception(f"Circuit breaker is OPEN for {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        return (
            self._last_failure_time and
            time.time() - self._last_failure_time >= self.recovery_timeout
        )
    
    def _on_success(self):
        """Handle successful execution."""
        with self._lock:
            self._failure_count = 0
            self._state = CircuitBreakerState.CLOSED
    
    def _on_failure(self):
        """Handle failed execution."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            if self._failure_count >= self.failure_threshold:
                self._state = CircuitBreakerState.OPEN

class RetryManager:
    """Manages retry logic with exponential backoff."""
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to add retry logic to functions."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self._retry_call(func, *args, **kwargs)
        return wrapper
    
    def _retry_call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    delay = self._calculate_delay(attempt)
                    logger.warning(
                        f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                        f"Retrying in {delay:.2f}s"
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"All {self.max_retries + 1} attempts failed for {func.__name__}")
        
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for exponential backoff."""
        delay = self.base_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)  # Add 0-50% jitter
        
        return delay

class HealthMonitor:
    """Monitors system health and component status."""
    
    def __init__(self):
        self.metrics: Dict[str, HealthMetrics] = {}
        self._lock = threading.Lock()
        self._monitoring = False
        self._monitor_thread = None
    
    def register_component(self, component: str):
        """Register a component for health monitoring."""
        with self._lock:
            if component not in self.metrics:
                self.metrics[component] = HealthMetrics(
                    component=component,
                    state=SystemState.HEALTHY,
                    error_rate=0.0,
                    response_time=0.0,
                    success_rate=1.0
                )
    
    def record_success(self, component: str, response_time: float):
        """Record successful operation."""
        with self._lock:
            if component in self.metrics:
                metrics = self.metrics[component]
                metrics.consecutive_failures = 0
                metrics.last_check = datetime.now()
                
                # Update response time (exponential moving average)
                alpha = 0.1
                metrics.response_time = (
                    alpha * response_time + (1 - alpha) * metrics.response_time
                )
                
                # Update success rate
                metrics.success_rate = min(1.0, metrics.success_rate + 0.01)
                
                # Update state based on metrics
                self._update_component_state(component)
    
    def record_failure(self, component: str, error: Exception):
        """Record failed operation."""
        with self._lock:
            if component in self.metrics:
                metrics = self.metrics[component]
                metrics.consecutive_failures += 1
                metrics.last_error = str(error)
                metrics.last_check = datetime.now()
                
                # Update error rate
                metrics.error_rate = min(1.0, metrics.error_rate + 0.1)
                
                # Update success rate
                metrics.success_rate = max(0.0, metrics.success_rate - 0.1)
                
                # Update state based on metrics
                self._update_component_state(component)
    
    def _update_component_state(self, component: str):
        """Update component state based on current metrics."""
        metrics = self.metrics[component]
        
        if metrics.consecutive_failures >= 10:
            metrics.state = SystemState.CRITICAL
        elif metrics.consecutive_failures >= 5:
            metrics.state = SystemState.DEGRADED
        elif metrics.error_rate > 0.5:
            metrics.state = SystemState.DEGRADED
        elif metrics.success_rate < 0.5:
            metrics.state = SystemState.DEGRADED
        else:
            metrics.state = SystemState.HEALTHY
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        with self._lock:
            total_components = len(self.metrics)
            if total_components == 0:
                return {"status": "unknown", "components": {}}
            
            healthy_count = sum(
                1 for m in self.metrics.values() 
                if m.state == SystemState.HEALTHY
            )
            
            degraded_count = sum(
                1 for m in self.metrics.values() 
                if m.state == SystemState.DEGRADED
            )
            
            critical_count = sum(
                1 for m in self.metrics.values() 
                if m.state == SystemState.CRITICAL
            )
            
            if critical_count > 0:
                overall_status = SystemState.CRITICAL
            elif degraded_count > total_components * 0.5:
                overall_status = SystemState.DEGRADED
            else:
                overall_status = SystemState.HEALTHY
            
            return {
                "status": overall_status.value,
                "total_components": total_components,
                "healthy": healthy_count,
                "degraded": degraded_count,
                "critical": critical_count,
                "components": {
                    name: {
                        "state": metrics.state.value,
                        "error_rate": metrics.error_rate,
                        "response_time": metrics.response_time,
                        "success_rate": metrics.success_rate,
                        "consecutive_failures": metrics.consecutive_failures,
                        "last_error": metrics.last_error
                    }
                    for name, metrics in self.metrics.items()
                }
            }

class ResilienceManager:
    """Main resilience management system."""
    
    def __init__(self):
        self.health_monitor = HealthMonitor()
        self.fallback_strategies: Dict[str, List[FallbackStrategy]] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.error_handlers: Dict[Type[Exception], Callable] = {}
        self._recovery_procedures: Dict[str, Callable] = {}
    
    def register_component(self, component: str):
        """Register a component with the resilience system."""
        self.health_monitor.register_component(component)
        logger.info(f"Registered component for resilience monitoring: {component}")
    
    def add_fallback_strategy(self, component: str, strategy: FallbackStrategy):
        """Add a fallback strategy for a component."""
        if component not in self.fallback_strategies:
            self.fallback_strategies[component] = []
        
        self.fallback_strategies[component].append(strategy)
        logger.info(f"Added fallback strategy for {component}: {type(strategy).__name__}")
    
    def add_circuit_breaker(
        self, 
        component: str, 
        failure_threshold: int = 5,
        recovery_timeout: int = 60
    ):
        """Add circuit breaker for a component."""
        self.circuit_breakers[component] = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout
        )
        logger.info(f"Added circuit breaker for {component}")
    
    def register_error_handler(
        self, 
        exception_type: Type[Exception], 
        handler: Callable
    ):
        """Register custom error handler for specific exception types."""
        self.error_handlers[exception_type] = handler
        logger.info(f"Registered error handler for {exception_type.__name__}")
    
    def register_recovery_procedure(self, component: str, procedure: Callable):
        """Register recovery procedure for a component."""
        self._recovery_procedures[component] = procedure
        logger.info(f"Registered recovery procedure for {component}")
    
    @contextmanager
    def resilient_operation(self, component: str, operation: str):
        """Context manager for resilient operations."""
        start_time = time.time()
        
        try:
            yield
            
            # Record success
            response_time = time.time() - start_time
            self.health_monitor.record_success(component, response_time)
            
        except Exception as e:
            # Record failure
            self.health_monitor.record_failure(component, e)
            
            # Create error context
            context = ErrorContext(
                error=e,
                component=component,
                operation=operation,
                timestamp=datetime.now(),
                severity=self._determine_severity(e),
                additional_info={"response_time": time.time() - start_time}
            )
            
            # Handle error
            self._handle_error(context)
    
    def _determine_severity(self, error: Exception) -> ErrorSeverity:
        """Determine error severity based on exception type."""
        if isinstance(error, (ConnectionError, TimeoutError)):
            return ErrorSeverity.HIGH
        elif isinstance(error, (ValueError, TypeError)):
            return ErrorSeverity.MEDIUM
        elif isinstance(error, KeyboardInterrupt):
            return ErrorSeverity.CRITICAL
        else:
            return ErrorSeverity.MEDIUM
    
    def _handle_error(self, context: ErrorContext):
        """Handle error using registered handlers and fallback strategies."""
        # Try custom error handler first
        exception_type = type(context.error)
        if exception_type in self.error_handlers:
            try:
                self.error_handlers[exception_type](context)
                return
            except Exception as handler_error:
                logger.error(f"Error handler failed: {handler_error}")
        
        # Try fallback strategies
        if context.component in self.fallback_strategies:
            for strategy in self.fallback_strategies[context.component]:
                if strategy.is_applicable(context):
                    try:
                        result = strategy.execute(context)
                        logger.info(f"Fallback strategy succeeded for {context.component}")
                        return result
                    except Exception as fallback_error:
                        logger.warning(f"Fallback strategy failed: {fallback_error}")
        
        # Try recovery procedure
        if context.component in self._recovery_procedures:
            try:
                self._recovery_procedures[context.component](context)
                logger.info(f"Recovery procedure executed for {context.component}")
            except Exception as recovery_error:
                logger.error(f"Recovery procedure failed: {recovery_error}")
        
        # Log error for monitoring
        logger.error(
            f"Resilience handling completed for {context.component}.{context.operation}: "
            f"{context.error}"
        )
    
    def get_resilience_status(self) -> Dict[str, Any]:
        """Get comprehensive resilience status."""
        health_status = self.health_monitor.get_system_health()
        
        circuit_breaker_status = {
            component: {
                "state": breaker._state.value,
                "failure_count": breaker._failure_count,
                "last_failure": breaker._last_failure_time
            }
            for component, breaker in self.circuit_breakers.items()
        }
        
        return {
            "health": health_status,
            "circuit_breakers": circuit_breaker_status,
            "fallback_strategies": {
                component: len(strategies)
                for component, strategies in self.fallback_strategies.items()
            },
            "recovery_procedures": list(self._recovery_procedures.keys()),
            "timestamp": datetime.now().isoformat()
        }

# Global resilience manager instance
resilience_manager = ResilienceManager()

# Convenience decorators
def resilient(component: str, operation: str = "operation"):
    """Decorator to make functions resilient."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with resilience_manager.resilient_operation(component, operation):
                return func(*args, **kwargs)
        return wrapper
    return decorator

def with_circuit_breaker(
    component: str,
    failure_threshold: int = 5,
    recovery_timeout: int = 60
):
    """Decorator to add circuit breaker to functions."""
    def decorator(func: Callable) -> Callable:
        # Ensure circuit breaker exists
        if component not in resilience_manager.circuit_breakers:
            resilience_manager.add_circuit_breaker(
                component, failure_threshold, recovery_timeout
            )
        
        circuit_breaker = resilience_manager.circuit_breakers[component]
        return circuit_breaker(func)
    
    return decorator

def with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0
):
    """Decorator to add retry logic to functions."""
    retry_manager = RetryManager(max_retries, base_delay, max_delay)
    return retry_manager

# Example usage and setup functions
def setup_consciousness_resilience():
    """Setup resilience for consciousness detection components."""
    # Register components
    resilience_manager.register_component("consciousness_detector")
    resilience_manager.register_component("phi_calculator")
    resilience_manager.register_component("integration_analyzer")
    
    # Add circuit breakers
    resilience_manager.add_circuit_breaker("consciousness_detector", 3, 30)
    resilience_manager.add_circuit_breaker("phi_calculator", 5, 60)
    
    # Add fallback strategies
    resilience_manager.add_fallback_strategy(
        "consciousness_detector",
        DefaultFallbackStrategy({
            "consciousness_level": "unknown",
            "confidence": 0.0,
            "phi": 0.0
        })
    )
    
    logger.info("Consciousness resilience setup completed")

def setup_thermodynamic_resilience():
    """Setup resilience for thermodynamic system components."""
    # Register components
    resilience_manager.register_component("thermodynamic_engine")
    resilience_manager.register_component("carnot_cycle")
    resilience_manager.register_component("heat_pump")
    resilience_manager.register_component("vortex_battery")
    
    # Add circuit breakers
    resilience_manager.add_circuit_breaker("thermodynamic_engine", 3, 45)
    resilience_manager.add_circuit_breaker("vortex_battery", 2, 30)
    
    # Add fallback strategies
    resilience_manager.add_fallback_strategy(
        "thermodynamic_engine",
        DefaultFallbackStrategy({
            "efficiency": 0.5,
            "temperature": 298.15,
            "entropy": 0.0
        })
    )
    
    logger.info("Thermodynamic resilience setup completed")

def initialize_resilience_framework():
    """Initialize the complete resilience framework."""
    logger.info("Initializing KIMERA resilience framework...")
    
    # Setup component-specific resilience
    setup_consciousness_resilience()
    setup_thermodynamic_resilience()
    
    # Register global error handlers
    def handle_connection_error(context: ErrorContext):
        logger.warning(f"Connection error in {context.component}: {context.error}")
        # Implement connection recovery logic here
    
    def handle_timeout_error(context: ErrorContext):
        logger.warning(f"Timeout error in {context.component}: {context.error}")
        # Implement timeout recovery logic here
    
    resilience_manager.register_error_handler(ConnectionError, handle_connection_error)
    resilience_manager.register_error_handler(TimeoutError, handle_timeout_error)
    
    logger.info("Resilience framework initialization completed")
    
    return resilience_manager

if __name__ == "__main__":
    # Example usage
    initialize_resilience_framework()
    
    # Print current status
    status = resilience_manager.get_resilience_status()
    print("Resilience Framework Status:")
    print(f"Health: {status['health']['status']}")
    print(f"Components: {status['health']['total_components']}")
    print(f"Circuit Breakers: {len(status['circuit_breakers'])}")
    print(f"Fallback Strategies: {sum(status['fallback_strategies'].values())}") 