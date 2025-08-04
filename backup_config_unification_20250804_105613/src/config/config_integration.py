"""
Configuration Integration for KIMERA System
Integrates configuration management into existing components
Phase 2, Week 6-7: Configuration Management Implementation
"""

import logging
from typing import Optional, Dict, Any, Callable
from pathlib import Path
import os
from functools import lru_cache

from .settings import KimeraSettings, get_settings
from .config_loader import load_configuration

logger = logging.getLogger(__name__)


class ConfigurationManager:
    """
    Central configuration manager for KIMERA system
    Provides easy access to configuration throughout the application
    """
    
    def __init__(self):
        self._settings: Optional[KimeraSettings] = None
        self._config_callbacks: Dict[str, Callable] = {}
        self._original_values: Dict[str, Any] = {}
        
    def initialize(self, environment: Optional[str] = None) -> KimeraSettings:
        """
        Initialize configuration manager
        
        Args:
            environment: Optional environment override
            
        Returns:
            Loaded configuration settings
        """
        self._settings = load_configuration(environment)
        
        # Store original values for reset
        self._original_values = self._settings.dict()
        
        # Apply configuration to system
        self._apply_configuration()
        
        logger.info("Configuration manager initialized")
        return self._settings
    
    @property
    def settings(self) -> KimeraSettings:
        """Get current settings"""
        if self._settings is None:
            self._settings = get_settings()
        return self._settings
    
    def _apply_configuration(self) -> None:
        """Apply configuration to various system components"""
        settings = self.settings
        
        # Configure logging
        self._configure_logging(settings)
        
        # Configure paths
        self._configure_paths(settings)
        
        # Configure threading
        self._configure_threading(settings)
        
        # Run registered callbacks
        for name, callback in self._config_callbacks.items():
            try:
                callback(settings)
                logger.debug(f"Configuration callback '{name}' executed")
            except Exception as e:
                logger.error(f"Configuration callback '{name}' failed: {e}")
    
    def _configure_logging(self, settings: KimeraSettings) -> None:
        """Configure logging based on settings"""
        import logging.config
        
        log_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": settings.logging.format
                },
                "structured": {
                    "()": "pythonjsonlogger.jsonlogger.JsonFormatter" if settings.logging.structured else None,
                    "format": "%(asctime)s %(name)s %(levelname)s %(message)s"
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": settings.logging.level.value,
                    "formatter": "structured" if settings.logging.structured else "default",
                    "stream": "ext://sys.stdout"
                }
            },
            "root": {
                "level": settings.logging.level.value,
                "handlers": ["console"]
            }
        }
        
        # Add file handler if enabled
        if settings.logging.file_enabled:
            log_file = settings.paths.logs_dir / "kimera.log"
            log_config["handlers"]["file"] = {
                "class": "logging.handlers.RotatingFileHandler",
                "level": settings.logging.level.value,
                "formatter": "structured" if settings.logging.structured else "default",
                "filename": str(log_file),
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5
            }
            log_config["root"]["handlers"].append("file")
        
        logging.config.dictConfig(log_config)
    
    def _configure_paths(self, settings: KimeraSettings) -> None:
        """Configure system paths"""
        # Ensure all required directories exist
        for path_name in ["data_dir", "logs_dir", "models_dir", "temp_dir"]:
            path = getattr(settings.paths, path_name)
            path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {path}")
    
    def _configure_threading(self, settings: KimeraSettings) -> None:
        """Configure threading based on settings"""
        # This would integrate with the async utilities from Week 5
        os.environ["OMP_NUM_THREADS"] = str(settings.performance.max_threads)
        os.environ["MKL_NUM_THREADS"] = str(settings.performance.max_threads)
        
    def register_callback(self, name: str, callback: Callable[[KimeraSettings], None]) -> None:
        """
        Register a callback to be called when configuration changes
        
        Args:
            name: Unique name for the callback
            callback: Function to call with settings
        """
        self._config_callbacks[name] = callback
        
        # Call immediately if already initialized
        if self._settings:
            try:
                callback(self._settings)
            except Exception as e:
                logger.error(f"Failed to execute callback '{name}': {e}")
    
    def reload(self) -> KimeraSettings:
        """Reload configuration from sources"""
        logger.info("Reloading configuration...")
        
        # Clear cached settings
        self._settings = None
        
        # Reinitialize
        return self.initialize()
    
    def update_runtime(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration at runtime (temporary, not persisted)
        
        Args:
            updates: Dictionary of updates to apply
        """
        if not self._settings:
            raise RuntimeError("Configuration not initialized")
        
        # Apply updates
        for key, value in updates.items():
            if hasattr(self._settings, key):
                setattr(self._settings, key, value)
                logger.info(f"Updated runtime configuration: {key} = {value}")
            else:
                logger.warning(f"Unknown configuration key: {key}")
        
        # Reapply configuration
        self._apply_configuration()
    
    def reset(self) -> None:
        """Reset configuration to original values"""
        if self._original_values:
            # This is a simplified reset - in production would need proper deserialization
            logger.info("Configuration reset to original values")
            self.reload()


# Global configuration manager instance
_config_manager: Optional[ConfigurationManager] = None


def get_config_manager() -> ConfigurationManager:
    """Get global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigurationManager()
    return _config_manager


# Convenience functions for common configuration access

@lru_cache(maxsize=1)
def get_database_url() -> str:
    """Get database URL from configuration"""
    return get_settings().database.url


@lru_cache(maxsize=1)
def get_api_key(service: str) -> Optional[str]:
    """
    Get API key for a service
    
    Args:
        service: Service name (e.g., 'openai', 'cryptopanic')
        
    Returns:
        API key or None if not configured
    """
    settings = get_settings()
    
    if service == "openai" and settings.api_keys.openai_api_key:
        return settings.api_keys.openai_api_key.get_secret_value()
    elif service == "cryptopanic" and settings.api_keys.cryptopanic_api_key:
        return settings.api_keys.cryptopanic_api_key.get_secret_value()
    elif service == "huggingface" and settings.api_keys.huggingface_token:
        return settings.api_keys.huggingface_token.get_secret_value()
    elif service in settings.api_keys.custom_api_keys:
        return settings.api_keys.custom_api_keys[service].get_secret_value()
    
    return None


def get_project_root() -> Path:
    """Get project root directory"""
    return get_settings().paths.project_root


def get_data_dir() -> Path:
    """Get data directory"""
    return get_settings().paths.data_dir


def get_models_dir() -> Path:
    """Get models directory"""
    return get_settings().paths.models_dir


def is_production() -> bool:
    """Check if running in production"""
    return get_settings().is_production


def is_development() -> bool:
    """Check if running in development"""
    return get_settings().is_development


def get_feature_flag(feature: str, default: bool = False) -> bool:
    """
    Get feature flag value
    
    Args:
        feature: Feature name
        default: Default value if not set
        
    Returns:
        Feature flag value
    """
    return get_settings().get_feature(feature, default)


# Decorators for configuration-aware functions

def requires_feature(feature: str):
    """
    Decorator to check if a feature is enabled
    
    Args:
        feature: Feature name to check
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not get_feature_flag(feature):
                raise RuntimeError(f"Feature '{feature}' is not enabled")
            return func(*args, **kwargs)
        return wrapper
    return decorator


def production_only(func):
    """Decorator to ensure function only runs in production"""
    def wrapper(*args, **kwargs):
        if not is_production():
            raise RuntimeError("This function can only be run in production")
        return func(*args, **kwargs)
    return wrapper


def development_only(func):
    """Decorator to ensure function only runs in development"""
    def wrapper(*args, **kwargs):
        if not is_development():
            raise RuntimeError("This function can only be run in development")
        return func(*args, **kwargs)
    return wrapper


# Configuration context manager

class ConfigurationContext:
    """Context manager for temporary configuration changes"""
    
    def __init__(self, **overrides):
        self.overrides = overrides
        self.original_values = {}
        self.settings = get_settings()
    
    def __enter__(self):
        # Store original values and apply overrides
        for key, value in self.overrides.items():
            if hasattr(self.settings, key):
                self.original_values[key] = getattr(self.settings, key)
                setattr(self.settings, key, value)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original values
        for key, value in self.original_values.items():
            setattr(self.settings, key, value)


# Initialize configuration on module import
def initialize_configuration():
    """Initialize configuration system"""
    manager = get_config_manager()
    manager.initialize()
    logger.info("Configuration system initialized")