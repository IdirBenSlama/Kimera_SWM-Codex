#!/usr/bin/env python3
"""
KIMERA Router Exception Handler
===============================

This module provides standardized exception handling for all FastAPI routers
in the KIMERA system. It ensures consistent error responses, proper logging,
and graceful degradation across all API endpoints.

Scientific Principles:
- Consistent error response format
- Detailed logging for debugging
- User-friendly error messages
- Security-conscious error exposure
"""

from datetime import datetime
from fastapi.responses import JSONResponse
from typing import Any, Callable, Dict, Optional, Type, Union
import asyncio
import logging

from fastapi import HTTPException, Request, Response
from functools import wraps
from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError
import traceback

from .exception_handling import (
    ErrorContext, ErrorSeverity, RecoveryStrategy,
    error_registry, safe_operation
)
from ..utils.kimera_exceptions import (
    KimeraBaseException,
    KimeraValidationError,
    KimeraResourceError,
    KimeraSecurityError,
    KimeraCognitiveError
)

logger = logging.getLogger(__name__)


class ErrorResponseBuilder:
    """Builds consistent error responses for API endpoints."""
    
    @staticmethod
    def build_error_response(
        error: Exception,
        operation: str,
        request_id: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> JSONResponse:
        """Build a standardized error response."""
        
        # Determine error type and severity
        if isinstance(error, KimeraSecurityError):
            status_code = 403
            error_type = "security_error"
            severity = ErrorSeverity.CRITICAL
            user_message = "Security validation failed"
            
        elif isinstance(error, KimeraValidationError):
            status_code = 400
            error_type = "validation_error"
            severity = ErrorSeverity.LOW
            user_message = str(error)
            
        elif isinstance(error, KimeraResourceError):
            status_code = 503
            error_type = "resource_error"
            severity = ErrorSeverity.HIGH
            user_message = "Resource temporarily unavailable"
            
        elif isinstance(error, KimeraCognitiveError):
            status_code = 500
            error_type = "cognitive_error"
            severity = ErrorSeverity.HIGH
            user_message = "Cognitive processing error"
            
        elif isinstance(error, ValidationError):
            status_code = 422
            error_type = "validation_error"
            severity = ErrorSeverity.LOW
            user_message = "Invalid request data"
            
        elif isinstance(error, SQLAlchemyError):
            status_code = 503
            error_type = "database_error"
            severity = ErrorSeverity.HIGH
            user_message = "Database operation failed"
            
        elif isinstance(error, HTTPException):
            status_code = error.status_code
            error_type = "http_error"
            severity = ErrorSeverity.MEDIUM
            user_message = error.detail
            
        else:
            status_code = 500
            error_type = "internal_error"
            severity = ErrorSeverity.HIGH
            user_message = "An unexpected error occurred"
        
        # Build error response
        error_response = {
            "error": {
                "type": error_type,
                "message": user_message,
                "operation": operation,
                "timestamp": datetime.now().isoformat(),
                "request_id": request_id
            },
            "status": "error"
        }
        
        # Add debug information in development mode
        if logger.isEnabledFor(logging.DEBUG):
            error_response["error"]["debug"] = {
                "exception_type": type(error).__name__,
                "exception_message": str(error),
                "severity": severity.value
            }
            
            # Add validation errors detail
            if isinstance(error, ValidationError):
                error_response["error"]["validation_errors"] = error.errors()
        
        # Add additional context if provided
        if additional_context:
            error_response["error"]["context"] = additional_context
        
        # Log the error
        logger.error(
            f"API Error in {operation}",
            exc_info=True,
            extra={
                "error_type": error_type,
                "status_code": status_code,
                "severity": severity.value,
                "request_id": request_id,
                "operation": operation
            }
        )
        
        return JSONResponse(
            status_code=status_code,
            content=error_response
        )


def handle_router_errors(
    operation: str,
    severity: Optional[ErrorSeverity] = None,
    fallback_response: Optional[Any] = None,
    include_request_id: bool = True
):
    """
    Decorator for handling errors in router endpoints.
    
    Args:
        operation: Name of the operation for logging
        severity: Override error severity
        fallback_response: Response to return on non-critical errors
        include_request_id: Whether to include request ID in response
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(request: Request, *args, **kwargs):
            request_id = None
            
            try:
                # Extract request ID if available
                if include_request_id:
                    request_id = request.headers.get("X-Request-ID") or \
                                request.headers.get("X-Correlation-ID")
                
                # Execute the endpoint function
                result = await func(request, *args, **kwargs)
                return result
                
            except Exception as e:
                # Check if we should use fallback
                if fallback_response is not None and not isinstance(e, (
                    KimeraSecurityError,
                    HTTPException
                )):
                    logger.warning(
                        f"Using fallback response for {operation} due to error: {e}"
                    )
                    return fallback_response
                
                # Build error response
                return ErrorResponseBuilder.build_error_response(
                    error=e,
                    operation=operation,
                    request_id=request_id,
                    additional_context={
                        "path": str(request.url.path),
                        "method": request.method
                    }
                )
        
        @wraps(func)
        def sync_wrapper(request: Request, *args, **kwargs):
            request_id = None
            
            try:
                # Extract request ID if available
                if include_request_id:
                    request_id = request.headers.get("X-Request-ID") or \
                                request.headers.get("X-Correlation-ID")
                
                # Execute the endpoint function
                result = func(request, *args, **kwargs)
                return result
                
            except Exception as e:
                # Check if we should use fallback
                if fallback_response is not None and not isinstance(e, (
                    KimeraSecurityError,
                    HTTPException
                )):
                    logger.warning(
                        f"Using fallback response for {operation} due to error: {e}"
                    )
                    return fallback_response
                
                # Build error response
                return ErrorResponseBuilder.build_error_response(
                    error=e,
                    operation=operation,
                    request_id=request_id,
                    additional_context={
                        "path": str(request.url.path),
                        "method": request.method
                    }
                )
        
        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def register_router_error_handlers(app):
    """Register global exception handlers for the FastAPI app."""
    
    @app.exception_handler(KimeraBaseException)
    async def kimera_exception_handler(request: Request, exc: KimeraBaseException):
        """Handle KIMERA-specific exceptions."""
        return ErrorResponseBuilder.build_error_response(
            error=exc,
            operation="global_handler",
            request_id=request.headers.get("X-Request-ID"),
            additional_context={
                "path": str(request.url.path),
                "method": request.method
            }
        )
    
    @app.exception_handler(ValidationError)
    async def validation_exception_handler(request: Request, exc: ValidationError):
        """Handle Pydantic validation errors."""
        return ErrorResponseBuilder.build_error_response(
            error=exc,
            operation="validation",
            request_id=request.headers.get("X-Request-ID"),
            additional_context={
                "path": str(request.url.path),
                "method": request.method
            }
        )
    
    @app.exception_handler(SQLAlchemyError)
    async def database_exception_handler(request: Request, exc: SQLAlchemyError):
        """Handle database errors."""
        return ErrorResponseBuilder.build_error_response(
            error=exc,
            operation="database_operation",
            request_id=request.headers.get("X-Request-ID"),
            additional_context={
                "path": str(request.url.path),
                "method": request.method
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle all other exceptions."""
        # Log unexpected errors with full traceback
        logger.error(
            f"Unexpected error: {exc}",
            exc_info=True,
            extra={
                "path": str(request.url.path),
                "method": request.method,
                "traceback": traceback.format_exc()
            }
        )
        
        return ErrorResponseBuilder.build_error_response(
            error=exc,
            operation="unexpected_error",
            request_id=request.headers.get("X-Request-ID"),
            additional_context={
                "path": str(request.url.path),
                "method": request.method
            }
        )


# Utility functions for common error scenarios

def validate_required_component(component: Any, component_name: str) -> None:
    """Validate that a required component is available."""
    if component is None:
        raise KimeraResourceError(
            f"{component_name} is not initialized or unavailable",
            resource_type=component_name,
            required_action="initialization"
        )


def validate_request_data(data: Dict[str, Any], required_fields: list) -> None:
    """Validate that required fields are present in request data."""
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        raise KimeraValidationError(
            f"Missing required fields: {', '.join(missing_fields)}",
            validation_errors={"missing_fields": missing_fields}
        )


def handle_cognitive_operation(
    operation_name: str,
    operation_func: Callable,
    *args,
    **kwargs
) -> Any:
    """Handle cognitive operations with proper error handling."""
    try:
        return operation_func(*args, **kwargs)
    except Exception as e:
        raise KimeraCognitiveError(
            f"Cognitive operation '{operation_name}' failed: {str(e)}",
            operation=operation_name,
            original_error=str(e)
        )


# Example usage in routers
"""
from fastapi import APIRouter, Request
from .router_exception_handler import handle_router_errors, validate_required_component

router = APIRouter()

@router.post("/process")
@handle_router_errors(
    operation="process_data",
    fallback_response={"status": "degraded", "message": "Processing with limited functionality"}
)
async def process_data(request: Request, data: ProcessRequest):
    # Validate components
    validate_required_component(kimera_system.get_vault_manager(), "VaultManager")
    
    # Process data
    result = await process_with_cognitive_engine(data)
    
    return {"status": "success", "result": result}
"""