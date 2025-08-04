"""
Vault module for Kimera SWM

This module provides persistent storage functionality for the Kimera SWM system.
It implements lazy initialization to prevent import-time database connection failures.
"""

import logging
from importlib import import_module

logger = logging.getLogger(__name__)

# Import core components that don't require database connection
from .vault_manager import VaultManager

# Define initialization function for lazy loading
def initialize_vault():
    """
    Initialize the vault module.
    
    This function should be called explicitly when the vault functionality is needed.
    It initializes the database connection and creates necessary tables.
    
    Returns:
        bool: True if initialization was successful, False otherwise
    """
    try:
        # Import database module
        from .database import initialize_database
        
        # Initialize database
        if not initialize_database():
            logger.error("Failed to initialize database")
            return False
        
        # Import enhanced database schema
        from .enhanced_database_schema import create_tables
        
        logger.info("Vault module initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize vault module: {e}")
        return False

# Export public API
__all__ = [
    'VaultManager',
    'initialize_vault'
]
