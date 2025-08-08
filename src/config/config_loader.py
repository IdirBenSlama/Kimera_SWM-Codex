"""
Configuration Loader for KIMERA System
Handles loading configuration from multiple sources
Phase 2, Week 6-7: Configuration Management Implementation
"""

import hashlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from dotenv import load_dotenv

from .settings import EnvironmentType, KimeraSettings

logger = logging.getLogger(__name__)
class ConfigurationLoader:
    """Auto-generated class."""
    pass
    """
    Loads configuration from multiple sources in priority order:
    1. Environment variables (highest priority)
    2. .env file
    3. Configuration files (JSON/YAML)
    4. Default values (lowest priority)
    """

    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path.cwd() / "config"
        self.loaded_sources: List[str] = []
        self._config_hash: Optional[str] = None

    def load(self, environment: Optional[str] = None) -> KimeraSettings:
        """
        Load configuration for the specified environment

        Args:
            environment: Environment name (development, staging, production)

        Returns:
            Loaded configuration settings
        """
        # Determine environment
        env = environment or os.getenv("KIMERA_ENV", "development")

        # Load .env file first
        self._load_env_file(env)

        # Load configuration files
        self._load_config_files(env)

        # Create settings instance (will read from environment)
        settings = KimeraSettings()

        # Calculate configuration hash
        self._config_hash = self._calculate_config_hash(settings)

        # Log loaded configuration sources
        logger.info(
            f"Configuration loaded from sources: {', '.join(self.loaded_sources)}"
        )
        logger.info(f"Environment: {settings.environment}")
        logger.info(f"Configuration hash: {self._config_hash[:8]}...")

        return settings

    def _load_env_file(self, environment: str) -> None:
        """Load .env file for the environment"""
        env_files = [
            ".env",  # Default
            f".env.{environment}",  # Environment specific
            ".env.local",  # Local overrides
            f".env.{environment}.local",  # Environment specific local overrides
        ]

        for env_file in env_files:
            env_path = Path(env_file)
            if env_path.exists():
                load_dotenv(env_path, override=True)
                self.loaded_sources.append(f"env:{env_file}")
                logger.debug(f"Loaded environment file: {env_file}")

    def _load_config_files(self, environment: str) -> None:
        """Load configuration files"""
        if not self.config_dir.exists():
            logger.debug(f"Config directory not found: {self.config_dir}")
            return

        # Configuration file patterns to load
        config_patterns = [
            "default",  # Default configuration
            environment,  # Environment specific
            "local",  # Local overrides
            f"{environment}.local",  # Environment specific local overrides
        ]

        for pattern in config_patterns:
            # Try JSON
            json_path = self.config_dir / f"{pattern}.json"
            if json_path.exists():
                self._load_json_config(json_path)

            # Try YAML
            yaml_path = self.config_dir / f"{pattern}.yaml"
            if yaml_path.exists():
                self._load_yaml_config(yaml_path)

            yml_path = self.config_dir / f"{pattern}.yml"
            if yml_path.exists():
                self._load_yaml_config(yml_path)

    def _load_json_config(self, path: Path) -> None:
        """Load JSON configuration file"""
        try:
            with open(path, "r") as f:
                config = json.load(f)
            self._apply_config_to_env(config)
            self.loaded_sources.append(f"json:{path.name}")
            logger.debug(f"Loaded JSON config: {path}")
        except Exception as e:
            logger.error(f"Failed to load JSON config {path}: {e}")

    def _load_yaml_config(self, path: Path) -> None:
        """Load YAML configuration file"""
        try:
            with open(path, "r") as f:
                config = yaml.safe_load(f)
            if config:
                self._apply_config_to_env(config)
                self.loaded_sources.append(f"yaml:{path.name}")
                logger.debug(f"Loaded YAML config: {path}")
        except Exception as e:
            logger.error(f"Failed to load YAML config {path}: {e}")

    def _apply_config_to_env(
        self, config: Dict[str, Any], prefix: str = "KIMERA"
    ) -> None:
        """
        Apply configuration dictionary to environment variables

        Args:
            config: Configuration dictionary
            prefix: Environment variable prefix
        """
        for key, value in config.items():
            env_key = f"{prefix}_{key.upper()}"

            if isinstance(value, dict):
                # Handle nested configuration
                self._apply_config_to_env(value, env_key)
            elif isinstance(value, (list, tuple)):
                # Convert lists to comma-separated strings
                os.environ[env_key] = ",".join(str(v) for v in value)
            elif value is not None:
                # Set environment variable
                os.environ[env_key] = str(value)

    def _calculate_config_hash(self, settings: KimeraSettings) -> str:
        """Calculate hash of configuration for change detection"""
        # Get non-sensitive configuration as dict
        config_dict = settings.dict(exclude={"api_keys", "security"})
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()

    @property
    def config_hash(self) -> Optional[str]:
        """Get configuration hash"""
        return self._config_hash
class ConfigurationValidator:
    """Auto-generated class."""
    pass
    """Validates configuration settings"""

    @staticmethod
    def validate(settings: KimeraSettings) -> List[str]:
        """
        Validate configuration settings

        Args:
            settings: Configuration settings to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Validate paths exist and are accessible
        if not settings.paths.project_root.exists():
            errors.append(f"Project root does not exist: {settings.paths.project_root}")

        # Validate database URL
        if not settings.database.url:
            errors.append("Database URL is not configured")

        # Validate production settings
        if settings.is_production:
            # Production must have proper security
            if not settings.security.secret_key.get_secret_value():
                errors.append("Secret key must be set in production")

            # Production should not have debug/reload enabled
            if settings.server.reload:
                errors.append("Auto-reload must be disabled in production")

            if settings.database.echo:
                errors.append("Database echo must be disabled in production")

            if settings.logging.level.value == "DEBUG":
                errors.append("Debug logging should be disabled in production")

        # Validate API keys if features require them
        if (
            settings.get_feature("openai_integration")
            and not settings.api_keys.openai_api_key
        ):
            errors.append(
                "OpenAI API key required when openai_integration feature is enabled"
            )

        # Validate performance settings
        if settings.performance.gpu_memory_fraction > 0.9:
            errors.append("GPU memory fraction should not exceed 0.9 to prevent OOM")

        return errors
class ConfigurationExporter:
    """Auto-generated class."""
    pass
    """Export configuration for different purposes"""

    @staticmethod
    def export_env_template(settings: KimeraSettings, output_path: Path) -> None:
        """
        Export environment variable template

        Args:
            settings: Configuration settings
            output_path: Path to write template
        """
        template_lines = [
            "# KIMERA Configuration Template",
            f"# Generated: {datetime.now().isoformat()}",
            "# Copy this file to .env and fill in the values",
            "",
            "# Environment",
            "KIMERA_ENV=development",
            "",
            "# Database",
            "KIMERA_DATABASE_URL=sqlite:///kimera_swm.db",
            "KIMERA_DB_POOL_SIZE=20",
            "",
            "# API Keys (obtain from respective services)",
            "OPENAI_API_KEY=",
            "CRYPTOPANIC_API_KEY=",
            "HUGGINGFACE_TOKEN=",
            "",
            "# Paths",
            "KIMERA_PROJECT_ROOT=",
            "KIMERA_DATA_DIR=./data",
            "KIMERA_LOGS_DIR=./logs",
            "",
            "# Performance",
            "KIMERA_MAX_THREADS=32",
            "KIMERA_GPU_MEMORY_FRACTION=0.8",
            "",
            "# Server",
            "KIMERA_HOST=127.0.0.1",
            "KIMERA_PORT=8000",
            "",
            "# Logging",
            "KIMERA_LOG_LEVEL=INFO",
            "",
            "# Security",
            "KIMERA_SECRET_KEY=",  # Will be generated
            "",
            "# Features (JSON format)",
            'KIMERA_FEATURES={"openai_integration": false, "advanced_monitoring": true}',
        ]

        with open(output_path, "w") as f:
            f.write("\n".join(template_lines))

        logger.info(f"Environment template exported to: {output_path}")

    @staticmethod
    def export_docker_env(settings: KimeraSettings, output_path: Path) -> None:
        """
        Export Docker environment file

        Args:
            settings: Configuration settings
            output_path: Path to write Docker env file
        """
        # Export only non-sensitive production settings
        docker_env = {
            "KIMERA_ENV": "production",
            "KIMERA_HOST": "0.0.0.0",
            "KIMERA_PORT": "8000",
            "KIMERA_WORKERS": "4",
            "KIMERA_LOG_LEVEL": "INFO",
            "KIMERA_DATABASE_URL": "${DATABASE_URL}",  # Injected by Docker
            "KIMERA_SECRET_KEY": "${SECRET_KEY}",  # Injected by Docker
        }

        lines = [f"{k}={v}" for k, v in docker_env.items()]

        with open(output_path, "w") as f:
            f.write("\n".join(lines))

        logger.info(f"Docker environment file exported to: {output_path}")


# Configuration management functions


def load_configuration(environment: Optional[str] = None) -> KimeraSettings:
    """
    Load and validate configuration

    Args:
        environment: Optional environment override

    Returns:
        Validated configuration settings

    Raises:
        ValueError: If configuration is invalid
    """
    loader = ConfigurationLoader()
    settings = loader.load(environment)

    # Validate configuration
    validator = ConfigurationValidator()
    errors = validator.validate(settings)

    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(
            f"  - {e}" for e in errors
        )
        raise ValueError(error_msg)

    return settings


def get_config(environment: Optional[str] = None) -> KimeraSettings:
    """
    Convenient function to get configuration

    Args:
        environment: Optional environment override

    Returns:
        Configuration settings
    """
    return load_configuration(environment)


def export_configuration_template(output_dir: Path = Path.cwd()) -> None:
    """Export configuration templates"""
    settings = KimeraSettings()
    exporter = ConfigurationExporter()

    # Export .env template
    exporter.export_env_template(settings, output_dir / ".env.template")

    # Export Docker env
    exporter.export_docker_env(settings, output_dir / "docker.env")

    # Export example JSON config
    example_config = {
        "environment": "development",
        "database": {"pool_size": 20, "echo": False},
        "performance": {"max_threads": 32, "batch_size": 100},
        "features": {"advanced_monitoring": True, "experimental_features": False},
    }

    with open(output_dir / "config" / "example.json", "w") as f:
        json.dump(example_config, f, indent=2)

    logger.info("Configuration templates exported")
