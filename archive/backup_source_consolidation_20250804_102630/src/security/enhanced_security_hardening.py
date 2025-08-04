"""
Enhanced Security Hardening
===========================

Implements defense-in-depth security middleware based on:
- OWASP Top 10 protection
- NIST Cybersecurity Framework
- Zero Trust principles
"""

import time
import logging
import hashlib
import secrets
from typing import Dict, Any, Optional, Callable
from datetime import datetime, timedelta
import json
from collections import defaultdict

from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(self, rate: int = 100, per: int = 60):
        self.rate = rate  # requests
        self.per = per    # seconds
        self.allowance = defaultdict(lambda: rate)
        self.last_check = defaultdict(time.time)
    
    def is_allowed(self, key: str) -> bool:
        """Check if request is allowed."""
        current = time.time()
        time_passed = current - self.last_check[key]
        self.last_check[key] = current
        
        self.allowance[key] += time_passed * (self.rate / self.per)
        if self.allowance[key] > self.rate:
            self.allowance[key] = self.rate
        
        if self.allowance[key] < 1.0:
            return False
        
        self.allowance[key] -= 1.0
        return True

class SecurityHeaders:
    """Security headers configuration."""
    
    HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Content-Security-Policy": "default-src 'self'",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
    }

async def security_middleware(request: Request, call_next: Callable) -> Response:
    """
    Enhanced security middleware.
    
    Implements:
    - Rate limiting
    - Security headers
    - Request validation
    - Audit logging
    """
    start_time = time.time()
    
    # Extract client identifier
    client_ip = request.client.host if request.client else "unknown"
    
    # Rate limiting
    rate_limiter = getattr(request.app.state, 'rate_limiter', None)
    if not rate_limiter:
        rate_limiter = RateLimiter()
        request.app.state.rate_limiter = rate_limiter
    
    if not rate_limiter.is_allowed(client_ip):
        logger.warning(f"Rate limit exceeded for {client_ip}")
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded"},
            headers={"Retry-After": "60"}
        )
    
    # Input validation
    if request.method in ["POST", "PUT", "PATCH"]:
        # Check content length
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > 10 * 1024 * 1024:  # 10MB limit
            return JSONResponse(
                status_code=413,
                content={"detail": "Request entity too large"}
            )
    
    # Process request
    try:
        response = await call_next(request)
    except Exception as e:
        logger.error(f"Request processing error: {e}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        )
    
    # Add security headers
    for header, value in SecurityHeaders.HEADERS.items():
        response.headers[header] = value
    
    # Add request ID for tracing
    request_id = request.headers.get("X-Request-ID", secrets.token_urlsafe(16))
    response.headers["X-Request-ID"] = request_id
    
    # Log request
    process_time = time.time() - start_time
    logger.info(
        f"Security middleware: {request.method} {request.url.path} "
        f"- {response.status_code} - {process_time:.3f}s - {client_ip}"
    )
    
    return response

class SecurityMiddleware(BaseHTTPMiddleware):
    """Enhanced security middleware class."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        return await security_middleware(request, call_next)