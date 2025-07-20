"""
KIMERA SWM - Comprehensive Exception Handling Framework
======================================================

Zero-debugging constraint compliant exception system with:
- Specific exception types for different error categories
- Context-aware error information with recovery suggestions
- Mapping from generic exceptions to Kimera-specific ones
- Integration with logging framework
"""

from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from enum import Enum
import traceback
import time


class ErrorSeverity(Enum):
    """Error severity levels for categorization"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categories of errors in Kimera system"""
    COGNITIVE = "cognitive"
    DATABASE = "database"
    GPU = "gpu"
    NETWORK = "network"
    VALIDATION = "validation"
    SECURITY = "security"
    PERFORMANCE = "performance"
    RESOURCE = "resource"
    CONFIGURATION = "configuration"


@dataclass
class ErrorContext:
    """Comprehensive error context for debugging and recovery"""
    error_id: str
    timestamp: float
    component: str
    operation: str
    category: ErrorCategory
    severity: ErrorSeverity
    context_data: Dict[str, Any] = field(default_factory=dict)
    recovery_suggestions: List[str] = field(default_factory=list)
    related_errors: List[str] = field(default_factory=list)
    system_state: Dict[str, Any] = field(default_factory=dict)


class KimeraBaseException(Exception):
    """
    Base exception class for all Kimera-specific exceptions.
    
    Provides structured error information with context and recovery suggestions.
    """
    
    def __init__(
        self, 
        message: str, 
        category: ErrorCategory = ErrorCategory.COGNITIVE,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
        recovery_suggestions: Optional[List[str]] = None,
        original_exception: Optional[Exception] = None
    ):
        super().__init__(message)
        
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context or {}
        self.recovery_suggestions = recovery_suggestions or []
        self.original_exception = original_exception
        self.timestamp = time.time()
        
        # Generate unique error ID
        import uuid
        self.error_id = f"{category.value}_{uuid.uuid4().hex[:8]}"
        
        # Capture stack trace
        self.stack_trace = traceback.format_exc()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization"""
        return {
            'error_id': self.error_id,
            'message': self.message,
            'category': self.category.value,
            'severity': self.severity.value,
            'context': self.context,
            'recovery_suggestions': self.recovery_suggestions,
            'timestamp': self.timestamp,
            'stack_trace': self.stack_trace,
            'original_exception': str(self.original_exception) if self.original_exception else None,
        }
    
    def add_context(self, key: str, value: Any) -> 'KimeraBaseException':
        """Add context information to the exception"""
        self.context[key] = value
        return self
    
    def add_recovery_suggestion(self, suggestion: str) -> 'KimeraBaseException':
        """Add a recovery suggestion"""
        self.recovery_suggestions.append(suggestion)
        return self


class KimeraCognitiveError(KimeraBaseException):
    """
    Exception for cognitive processing errors.
    
    Raised when cognitive field dynamics, contradiction detection,
    or other cognitive operations fail.
    """
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message, 
            category=ErrorCategory.COGNITIVE,
            **kwargs
        )
        
        # Add default recovery suggestions
        if not self.recovery_suggestions:
            self.recovery_suggestions = [
                "Check cognitive field initialization",
                "Verify geoid data integrity",
                "Review contradiction detection parameters",
                "Ensure sufficient system resources"
            ]


class KimeraDatabaseError(KimeraBaseException):
    """
    Exception for database operation errors.
    
    Raised when database connections, queries, or transactions fail.
    """
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message, 
            category=ErrorCategory.DATABASE,
            **kwargs
        )
        
        # Add default recovery suggestions
        if not self.recovery_suggestions:
            self.recovery_suggestions = [
                "Check database connection status",
                "Verify database credentials",
                "Review database schema integrity",
                "Check available disk space",
                "Retry operation with backoff"
            ]


class KimeraGPUError(KimeraBaseException):
    """
    Exception for GPU operation errors.
    
    Raised when GPU memory allocation, CUDA operations, or
    GPU resource management fails.
    """
    
    def __init__(self, message: str, device: Optional[str] = None, **kwargs):
        super().__init__(
            message, 
            category=ErrorCategory.GPU,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )
        
        if device:
            self.add_context('device', device)
        
        # Add default recovery suggestions
        if not self.recovery_suggestions:
            self.recovery_suggestions = [
                "Clear GPU memory cache with torch.cuda.empty_cache()",
                "Reduce batch size or model dimensions",
                "Check GPU memory availability",
                "Verify CUDA installation and drivers",
                "Fall back to CPU processing if necessary"
            ]


class KimeraNetworkError(KimeraBaseException):
    """
    Exception for network operation errors.
    
    Raised when API calls, data fetching, or network
    communication fails.
    """
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message, 
            category=ErrorCategory.NETWORK,
            **kwargs
        )
        
        # Add default recovery suggestions
        if not self.recovery_suggestions:
            self.recovery_suggestions = [
                "Check network connectivity",
                "Verify API endpoint availability",
                "Review authentication credentials",
                "Implement retry with exponential backoff",
                "Check rate limiting and quotas"
            ]


class KimeraValidationError(KimeraBaseException):
    """
    Exception for input validation errors.
    
    Raised when input data fails validation checks
    or type requirements.
    """
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message, 
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )
        
        # Add default recovery suggestions
        if not self.recovery_suggestions:
            self.recovery_suggestions = [
                "Verify input data format and types",
                "Check parameter value ranges",
                "Review API documentation for requirements",
                "Validate data schema compliance"
            ]


class KimeraSecurityError(KimeraBaseException):
    """
    Exception for security-related errors.
    
    Raised when authentication, authorization, or
    security validation fails.
    """
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message, 
            category=ErrorCategory.SECURITY,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )
        
        # Add default recovery suggestions
        if not self.recovery_suggestions:
            self.recovery_suggestions = [
                "Verify authentication credentials",
                "Check user permissions and roles",
                "Review security configuration",
                "Audit access logs for suspicious activity",
                "Consider implementing additional security measures"
            ]


class KimeraPerformanceError(KimeraBaseException):
    """
    Exception for performance-related errors.
    
    Raised when operations exceed performance thresholds
    or resource limits.
    """
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message, 
            category=ErrorCategory.PERFORMANCE,
            **kwargs
        )
        
        # Add default recovery suggestions
        if not self.recovery_suggestions:
            self.recovery_suggestions = [
                "Optimize algorithm or data structures",
                "Increase system resources (CPU, memory)",
                "Implement caching or memoization",
                "Consider parallel processing",
                "Review performance bottlenecks"
            ]


class KimeraResourceError(KimeraBaseException):
    """
    Exception for resource allocation errors.
    
    Raised when system resources (memory, disk, CPU)
    are insufficient or unavailable.
    """
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message, 
            category=ErrorCategory.RESOURCE,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )
        
        # Add default recovery suggestions
        if not self.recovery_suggestions:
            self.recovery_suggestions = [
                "Free up system memory",
                "Clear temporary files and caches",
                "Reduce processing load",
                "Scale up system resources",
                "Implement resource pooling"
            ]


class KimeraConfigurationError(KimeraBaseException):
    """
    Exception for configuration errors.
    
    Raised when system configuration is invalid
    or missing required settings.
    """
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message, 
            category=ErrorCategory.CONFIGURATION,
            **kwargs
        )
        
        # Add default recovery suggestions
        if not self.recovery_suggestions:
            self.recovery_suggestions = [
                "Review configuration file syntax",
                "Verify all required settings are provided",
                "Check environment variable values",
                "Validate configuration against schema",
                "Restore from known good configuration"
            ]


# Exception mapping for converting generic exceptions to Kimera-specific ones
EXCEPTION_MAPPING = {
    # Database exceptions
    'SQLAlchemyError': KimeraDatabaseError,
    'OperationalError': KimeraDatabaseError,
    'IntegrityError': KimeraDatabaseError,
    'DatabaseError': KimeraDatabaseError,
    'ConnectionError': KimeraDatabaseError,
    
    # GPU/CUDA exceptions
    'RuntimeError': lambda msg, **kwargs: (
        KimeraGPUError(msg, **kwargs) if 'cuda' in msg.lower() or 'gpu' in msg.lower() 
        else KimeraResourceError(msg, **kwargs)
    ),
    'OutOfMemoryError': KimeraResourceError,
    'CudaError': KimeraGPUError,
    
    # Network exceptions
    'HTTPException': KimeraNetworkError,
    'RequestException': KimeraNetworkError,
    'TimeoutError': KimeraNetworkError,
    'URLError': KimeraNetworkError,
    
    # Validation exceptions
    'ValueError': KimeraValidationError,
    'TypeError': KimeraValidationError,
    'ValidationError': KimeraValidationError,
    
    # Security exceptions
    'PermissionError': KimeraSecurityError,
    'AuthenticationError': KimeraSecurityError,
    'AuthorizationError': KimeraSecurityError,
    
    # Configuration exceptions
    'ConfigurationError': KimeraConfigurationError,
    'EnvironmentError': KimeraConfigurationError,
    'KeyError': KimeraConfigurationError,
    'AttributeError': KimeraConfigurationError,
    
    # Performance exceptions
    'MemoryError': KimeraPerformanceError,
    'RecursionError': KimeraPerformanceError,
}


def map_exception(
    exception: Exception, 
    context: Optional[Dict[str, Any]] = None,
    recovery_suggestions: Optional[List[str]] = None
) -> KimeraBaseException:
    """
    Map a generic exception to a Kimera-specific exception.
    
    Args:
        exception: The original exception
        context: Additional context information
        recovery_suggestions: Specific recovery suggestions
        
    Returns:
        Mapped Kimera-specific exception
    """
    exception_name = type(exception).__name__
    message = str(exception)
    
    # Try to find specific mapping
    if exception_name in EXCEPTION_MAPPING:
        mapper = EXCEPTION_MAPPING[exception_name]
        
        # Handle callable mappers (for conditional mapping)
        if callable(mapper) and not issubclass(mapper, KimeraBaseException):
            return mapper(message, context=context, recovery_suggestions=recovery_suggestions, original_exception=exception)
        else:
            return mapper(message, context=context, recovery_suggestions=recovery_suggestions, original_exception=exception)
    
    # Default to cognitive error for unmapped exceptions
    return KimeraCognitiveError(
        f"Unmapped exception: {exception_name}: {message}",
        context=context,
        recovery_suggestions=recovery_suggestions or ["Review system logs for more details"],
        original_exception=exception
    )


def handle_exception(
    exception: Exception,
    component: str,
    operation: str,
    context: Optional[Dict[str, Any]] = None,
    logger=None
) -> KimeraBaseException:
    """
    Comprehensive exception handler that maps, logs, and enriches exceptions.
    
    Args:
        exception: The original exception
        component: Component where the exception occurred
        operation: Operation that failed
        context: Additional context information
        logger: Logger instance for recording the exception
        
    Returns:
        Enriched Kimera-specific exception
    """
    # Map to Kimera-specific exception
    kimera_exception = map_exception(exception, context)
    
    # Enrich with component and operation context
    kimera_exception.add_context('component', component)
    kimera_exception.add_context('operation', operation)
    
    # Add system state if available
    try:
        import psutil
        process = psutil.Process()
        system_state = {
            'cpu_percent': process.cpu_percent(),
            'memory_percent': process.memory_percent(),
            'thread_count': process.num_threads(),
        }
        kimera_exception.add_context('system_state', system_state)
    except Exception:
        pass  # System monitoring is optional
    
    # Log the exception if logger provided
    if logger:
        logger.error(
            f"Exception in {component}.{operation}: {kimera_exception.message}",
            error=kimera_exception,
            **kimera_exception.context
        )
    
    return kimera_exception


# Context manager for automatic exception handling
class KimeraExceptionHandler:
    """
    Context manager for automatic exception handling and mapping.
    """
    
    def __init__(
        self, 
        component: str, 
        operation: str, 
        context: Optional[Dict[str, Any]] = None,
        logger=None,
        reraise: bool = True
    ):
        self.component = component
        self.operation = operation
        self.context = context or {}
        self.logger = logger
        self.reraise = reraise
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # Handle the exception
            kimera_exception = handle_exception(
                exc_val, 
                self.component, 
                self.operation, 
                self.context, 
                self.logger
            )
            
            if self.reraise:
                raise kimera_exception from exc_val
            else:
                # Exception handled, don't propagate
                return True
        
        return False


# Decorator for automatic exception handling
def kimera_exception_handler(
    component: str,
    operation: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    logger=None
):
    """
    Decorator for automatic exception handling in functions.
    
    Args:
        component: Component name
        operation: Operation name (defaults to function name)
        context: Additional context
        logger: Logger instance
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            op_name = operation or func.__name__
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                kimera_exception = handle_exception(
                    e, component, op_name, context, logger
                )
                raise kimera_exception from e
        
        return wrapper
    return decorator 