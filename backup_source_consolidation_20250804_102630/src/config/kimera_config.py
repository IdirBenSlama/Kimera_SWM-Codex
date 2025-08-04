"""
KIMERA Configuration Management System
=====================================

Centralized configuration management with environment variable support,
validation, and dynamic updates.

Features:
- Environment-based configuration
- Type validation and defaults
- Dynamic configuration updates
- Secrets management
- Configuration profiles (dev, staging, prod)
"""

import os
import json
import logging
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum
import yaml

logger = logging.getLogger(__name__)

class ConfigProfile(Enum):
    """Configuration profiles for different environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"

@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    url: str = field(default_factory=lambda: os.getenv(
        "DATABASE_URL", 
        "sqlite:///./kimera_swm.db"
    ))
    pool_size: int = field(default_factory=lambda: int(os.getenv("DB_POOL_SIZE", "10")))
    max_overflow: int = field(default_factory=lambda: int(os.getenv("DB_MAX_OVERFLOW", "20")))
    echo: bool = field(default_factory=lambda: os.getenv("DB_ECHO", "false").lower() == "true")
    
    def __post_init__(self):
        # Validate PostgreSQL URL if provided
        if "postgresql" in self.url and "pgvector" not in self.url:
            logger.warning("PostgreSQL URL detected but pgvector extension not mentioned")

@dataclass
class ServerConfig:
    """Server configuration settings."""
    host: str = field(default_factory=lambda: os.getenv("SERVER_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("SERVER_PORT", "8000")))
    reload: bool = field(default_factory=lambda: os.getenv("SERVER_RELOAD", "false").lower() == "true")
    workers: int = field(default_factory=lambda: int(os.getenv("SERVER_WORKERS", "1")))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "info"))

@dataclass
class GPUConfig:
    """GPU configuration settings."""
    enabled: bool = field(default_factory=lambda: os.getenv("GPU_ENABLED", "true").lower() == "true")
    device_id: int = field(default_factory=lambda: int(os.getenv("CUDA_VISIBLE_DEVICES", "0")))
    memory_fraction: float = field(default_factory=lambda: float(os.getenv("GPU_MEMORY_FRACTION", "0.9")))
    mixed_precision: bool = field(default_factory=lambda: os.getenv("GPU_MIXED_PRECISION", "true").lower() == "true")

@dataclass
class DiffusionConfig:
    """Diffusion engine configuration."""
    num_steps: int = field(default_factory=lambda: int(os.getenv("DIFFUSION_STEPS", "20")))
    noise_schedule: str = field(default_factory=lambda: os.getenv("DIFFUSION_NOISE_SCHEDULE", "cosine"))
    embedding_dim: int = field(default_factory=lambda: int(os.getenv("DIFFUSION_EMBEDDING_DIM", "1024")))
    max_length: int = field(default_factory=lambda: int(os.getenv("DIFFUSION_MAX_LENGTH", "512")))
    temperature: float = field(default_factory=lambda: float(os.getenv("DIFFUSION_TEMPERATURE", "0.8")))
    top_k: int = field(default_factory=lambda: int(os.getenv("DIFFUSION_TOP_K", "50")))
    top_p: float = field(default_factory=lambda: float(os.getenv("DIFFUSION_TOP_P", "0.9")))

@dataclass
class SecurityConfig:
    """Security configuration settings."""
    secret_key: str = field(default_factory=lambda: os.getenv(
        "SECRET_KEY",
        "CHANGE_THIS_IN_PRODUCTION_" + os.urandom(32).hex()
    ))
    jwt_algorithm: str = field(default_factory=lambda: os.getenv("JWT_ALGORITHM", "HS256"))
    jwt_expiration_hours: int = field(default_factory=lambda: int(os.getenv("JWT_EXPIRATION_HOURS", "24")))
    cors_origins: List[str] = field(default_factory=lambda: os.getenv(
        "CORS_ORIGINS",
        "http://localhost,http://localhost:8080,http://localhost:3000"
    ).split(","))
    
    def __post_init__(self):
        if self.secret_key.startswith("CHANGE_THIS"):
            logger.warning("⚠️ Using default secret key - CHANGE THIS IN PRODUCTION!")

@dataclass
class MonitoringConfig:
    """Monitoring and metrics configuration."""
    prometheus_enabled: bool = field(default_factory=lambda: os.getenv("PROMETHEUS_ENABLED", "true").lower() == "true")
    metrics_port: int = field(default_factory=lambda: int(os.getenv("METRICS_PORT", "9090")))
    log_format: str = field(default_factory=lambda: os.getenv("LOG_FORMAT", "json"))
    trace_enabled: bool = field(default_factory=lambda: os.getenv("TRACE_ENABLED", "false").lower() == "true")

@dataclass
class CacheConfig:
    """Cache configuration settings."""
    redis_url: Optional[str] = field(default_factory=lambda: os.getenv("REDIS_URL"))
    cache_ttl: int = field(default_factory=lambda: int(os.getenv("CACHE_TTL", "3600")))
    max_cache_size: int = field(default_factory=lambda: int(os.getenv("MAX_CACHE_SIZE", "1000")))

@dataclass
class MCPConfig:
    """MCP (Model Context Protocol) configuration."""
    config_path: str = field(default_factory=lambda: os.getenv(
        "MCP_CONFIG_PATH",
        str(Path.home() / ".cursor" / "mcp.json")
    ))
    enabled_servers: List[str] = field(default_factory=lambda: os.getenv(
        "MCP_ENABLED_SERVERS",
        "kimera-cognitive,kimera-enhanced,fetch,sqlite-kimera"
    ).split(","))
    
    def get_server_config(self, server_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific MCP server."""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                return config.get("mcpServers", {}).get(server_name)
        except Exception as e:
            logger.error(f"Failed to load MCP config: {e}")
            return None

@dataclass
class KimeraConfig:
    """Main KIMERA configuration container."""
    profile: ConfigProfile = field(default_factory=lambda: ConfigProfile(
        os.getenv("KIMERA_PROFILE", "development").lower()
    ))
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    gpu: GPUConfig = field(default_factory=GPUConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)
    
    # Runtime configuration
    _overrides: Dict[str, Any] = field(default_factory=dict, init=False)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        logger.info(f"🔧 KIMERA Configuration initialized for {self.profile.value} profile")
        
        # Apply profile-specific settings
        self._apply_profile_settings()
        
        # Load additional config from file if exists
        self._load_config_file()
    
    def _apply_profile_settings(self):
        """Apply profile-specific configuration adjustments."""
        if self.profile == ConfigProfile.PRODUCTION:
            # Production settings
            self.server.reload = False
            self.server.workers = max(4, self.server.workers)
            self.database.echo = False
            self.monitoring.trace_enabled = False
            
            # Ensure security
            if self.security.secret_key.startswith("CHANGE_THIS"):
                raise ValueError("Secret key must be set for production!")
                
        elif self.profile == ConfigProfile.DEVELOPMENT:
            # Development settings
            self.server.reload = True
            self.server.workers = 1
            self.monitoring.trace_enabled = True
            
        elif self.profile == ConfigProfile.TEST:
            # Test settings
            self.database.url = "sqlite:///:memory:"
            self.gpu.enabled = False
            self.monitoring.prometheus_enabled = False
    
    def _load_config_file(self):
        """Load additional configuration from file."""
        config_files = [
            f"config.{self.profile.value}.yaml",
            f"config.{self.profile.value}.json",
            "config.yaml",
            "config.json"
        ]
        
        for config_file in config_files:
            config_path = Path("config") / config_file
            if config_path.exists():
                logger.info(f"Loading configuration from {config_path}")
                try:
                    with open(config_path, 'r') as f:
                        if config_file.endswith('.yaml'):
                            additional_config = yaml.safe_load(f)
                        else:
                            additional_config = json.load(f)
                    
                    # Merge with existing config
                    self._merge_config(additional_config)
                    break
                except Exception as e:
                    logger.error(f"Failed to load config file {config_path}: {e}")
    
    def _merge_config(self, additional_config: Dict[str, Any]):
        """Merge additional configuration with existing."""
        for key, value in additional_config.items():
            if hasattr(self, key) and isinstance(value, dict):
                # Merge nested configs
                existing = getattr(self, key)
                for sub_key, sub_value in value.items():
                    if hasattr(existing, sub_key):
                        setattr(existing, sub_key, sub_value)
            else:
                # Store as override
                self._overrides[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with dot notation support."""
        # Check overrides first
        if key in self._overrides:
            return self._overrides[key]
        
        # Support dot notation
        parts = key.split('.')
        obj = self
        
        try:
            for part in parts:
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    return default
            return obj
        except AttributeError:
            return default
    
    def set(self, key: str, value: Any):
        """Set configuration value dynamically."""
        parts = key.split('.')
        
        if len(parts) == 1:
            # Top-level override
            self._overrides[key] = value
        else:
            # Nested setting
            obj = self
            for part in parts[:-1]:
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    raise ValueError(f"Invalid configuration key: {key}")
            
            if hasattr(obj, parts[-1]):
                setattr(obj, parts[-1], value)
            else:
                raise ValueError(f"Invalid configuration key: {key}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        config_dict = asdict(self)
        # Remove private fields
        config_dict.pop('_overrides', None)
        # Add overrides
        config_dict.update(self._overrides)
        return config_dict
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Database validation
        if "postgresql" in self.database.url and self.database.pool_size < 5:
            issues.append("PostgreSQL pool size should be at least 5")
        
        # GPU validation
        if self.gpu.enabled and self.gpu.memory_fraction > 0.95:
            issues.append("GPU memory fraction > 0.95 may cause OOM errors")
        
        # Security validation
        if self.profile == ConfigProfile.PRODUCTION:
            if len(self.security.secret_key) < 32:
                issues.append("Secret key should be at least 32 characters")
            if "localhost" in self.security.cors_origins:
                issues.append("localhost should not be in CORS origins for production")
        
        # Server validation
        if self.server.workers > 1 and self.server.reload:
            issues.append("Auto-reload should be disabled when using multiple workers")
        
        return issues
    
    def save_to_file(self, path: Union[str, Path]):
        """Save current configuration to file."""
        path = Path(path)
        config_dict = self.to_dict()
        
        with open(path, 'w') as f:
            if path.suffix == '.yaml':
                yaml.dump(config_dict, f, default_flow_style=False)
            else:
                json.dump(config_dict, f, indent=2)
        
        logger.info(f"Configuration saved to {path}")

# Global configuration instance
_config: Optional[KimeraConfig] = None

def get_config() -> KimeraConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = KimeraConfig()
        
        # Validate configuration
        issues = _config.validate()
        if issues:
            logger.warning("Configuration validation issues:")
            for issue in issues:
                logger.warning(f"  - {issue}")
    
    return _config

def reload_config():
    """Reload configuration from environment and files."""
    global _config
    _config = None
    return get_config()

# Convenience exports
config = get_config()