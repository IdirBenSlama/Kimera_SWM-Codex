"""
KIMERA Security Module
======================

Implements defense-in-depth security patterns from:
- NIST Cybersecurity Framework
- ISO 27001/27002
- OWASP Security Guidelines
- Zero Trust Architecture

Key Components:
- Authentication and authorization
- Encryption and key management
- Input validation and sanitization
- Security monitoring and incident response
"""

from .authentication import Authenticator, AuthenticationMethod
from .authorization import Authorizer, Permission
from .encryption import EncryptionManager
from .validator import InputValidator, ValidationRule

__all__ = [
    'Authenticator',
    'AuthenticationMethod',
    'Authorizer',
    'Permission',
    'EncryptionManager',
    'InputValidator',
    'ValidationRule'
]