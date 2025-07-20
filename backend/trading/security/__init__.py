"""Security module for the trading system.

Provides core security functionality including:
- Authentication and authorization
- Data encryption/decryption
- Security logging
- Configuration management
"""

from typing import Optional
import logging
from cryptography.fernet import Fernet
from ..config.config import get_security_config

logger = logging.getLogger(__name__)

class SecurityManager:
    """Core security operations for the trading system.
    
    Handles:
    - Data encryption/decryption
    - Security configuration
    - Audit logging
    """
    
    def __init__(self):
        self.config = get_security_config()
        self.cipher = Fernet(self.config.encryption_key)
        
    def encrypt_data(self, data: bytes) -> bytes:
        """Encrypt sensitive data using Fernet symmetric encryption."""
        if not isinstance(data, bytes):
            raise TypeError("Data must be bytes for encryption")
        return self.cipher.encrypt(data)
        
    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt data using Fernet symmetric encryption."""
        if not isinstance(encrypted_data, bytes):
            raise TypeError("Encrypted data must be bytes")
        return self.cipher.decrypt(encrypted_data)
