"""
"""KIMERA Configuration Module"""

Centralized configuration management system
Phase 2, Week 6-7: Configuration Management Implementation
"""

from .config_integration import (ConfigurationContext, ConfigurationManager
                                 development_only, get_api_key, get_config_manager
                                 get_data_dir, get_database_url, get_feature_flag
                                 get_models_dir, get_project_root
                                 initialize_configuration, is_development
                                 is_production, production_only, requires_feature)
from .config_loader import (ConfigurationExporter, ConfigurationLoader
                            ConfigurationValidator, export_configuration_template
                            load_configuration)
from .config_migration import (ConfigurationMigrator, HardcodedValue
                               migrate_configuration)
from .settings import (APIKeysSettings, DatabaseSettings, EnvironmentType
                       KimeraSettings, LoggingSettings, LogLevel, MonitoringSettings
                       PathSettings, PerformanceSettings, SecuritySettings
                       ServerSettings, get_settings, reload_settings)

__all__ = [
    # Settings classes
    "KimeraSettings",
    "EnvironmentType",
    "LogLevel",
    "DatabaseSettings",
    "APIKeysSettings",
    "PathSettings",
    "PerformanceSettings",
    "ServerSettings",
    "LoggingSettings",
    "MonitoringSettings",
    "SecuritySettings",
    # Core functions
    "get_settings",
    "reload_settings",
    "load_configuration",
    "export_configuration_template",
    # Configuration management
    "ConfigurationManager",
    "get_config_manager",
    "initialize_configuration",
    # Convenience functions
    "get_database_url",
    "get_api_key",
    "get_project_root",
    "get_data_dir",
    "get_models_dir",
    "is_production",
    "is_development",
    "get_feature_flag",
    # Decorators
    "requires_feature",
    "production_only",
    "development_only",
    # Context managers
    "ConfigurationContext",
    # Migration tools
    "ConfigurationMigrator",
    "migrate_configuration",
    # Loader and validator
    "ConfigurationLoader",
    "ConfigurationValidator",
    "ConfigurationExporter",
]

# Initialize configuration on import
# This ensures configuration is available throughout the application
try:
    initialize_configuration()
except Exception as e:
    import logging

    logging.warning(f"Failed to initialize configuration on import: {e}")
    # Configuration will be initialized on first access
