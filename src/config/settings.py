"""
KIMERA Configuration Settings
Pydantic-based configuration management with environment variable support
Phase 2, Week 6-7: Configuration Management Implementation
"""

import json
import os
import sys
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import Field, SecretStr, validator
from pydantic_settings import BaseSettings


class EnvironmentType(str, Enum):
    """Application environment types"""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(str, Enum):
    """Logging levels"""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class DatabaseSettings(BaseSettings):
    """Database configuration settings"""

    url: str = Field(
        default="postgresql+psycopg2://kimera:kimera_secure_pass_2025@localhost:5432/kimera_swm",
        env="KIMERA_DATABASE_URL",
        description="Database connection URL",
    )

    pool_size: int = Field(
        default=20,
        env="KIMERA_DB_POOL_SIZE",
        ge=1,
        le=100,
        description="Database connection pool size",
    )
    pool_timeout: float = Field(
        default=30.0,
        env="KIMERA_DB_POOL_TIMEOUT",
        ge=1.0,
        description="Database connection pool timeout in seconds",
    )
    echo: bool = Field(
        default=False, env="KIMERA_DB_ECHO", description="Echo SQL statements"
    )

    class Config:
        env_prefix = "KIMERA_DB_"


class APIKeysSettings(BaseSettings):
    """API keys and secrets configuration"""

    openai_api_key: Optional[SecretStr] = Field(
        default=None, env="OPENAI_API_KEY", description="OpenAI API key"
    )
    cryptopanic_api_key: Optional[SecretStr] = Field(
        default=None, env="CRYPTOPANIC_API_KEY", description="CryptoPanic API key"
    )
    huggingface_token: Optional[SecretStr] = Field(
        default=None, env="HUGGINGFACE_TOKEN", description="HuggingFace API token"
    )

    # Add more API keys as needed
    custom_api_keys: Dict[str, SecretStr] = Field(
        default_factory=dict,
        env="KIMERA_CUSTOM_API_KEYS",
        description="Custom API keys as JSON",
    )

    @validator("custom_api_keys", pre=True)
    def parse_custom_api_keys(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return {}
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"  # Allow extra fields from environment


class PathSettings(BaseSettings):
    """Path configuration settings"""

    project_root: Path = Field(
        default_factory=lambda: Path.cwd(),
        env="KIMERA_PROJECT_ROOT",
        description="Project root directory",
    )
    data_dir: Path = Field(
        default_factory=lambda: Path.cwd() / "data",
        env="KIMERA_DATA_DIR",
        description="Data directory",
    )
    logs_dir: Path = Field(
        default_factory=lambda: Path.cwd() / "logs",
        env="KIMERA_LOGS_DIR",
        description="Logs directory",
    )
    models_dir: Path = Field(
        default_factory=lambda: Path.cwd() / "models",
        env="KIMERA_MODELS_DIR",
        description="Models directory",
    )
    temp_dir: Path = Field(
        default_factory=lambda: Path.cwd() / "temp",
        env="KIMERA_TEMP_DIR",
        description="Temporary files directory",
    )

    @validator("project_root", "data_dir", "logs_dir", "models_dir", "temp_dir")
    def resolve_path(cls, v):
        """Resolve paths to absolute paths"""
        return Path(v).resolve()

    @validator("data_dir", "logs_dir", "models_dir", "temp_dir")
    def create_directories(cls, v):
        """Create directories if they don't exist"""
        v.mkdir(parents=True, exist_ok=True)
        return v

    class Config:
        env_prefix = "KIMERA_"


class PerformanceSettings(BaseSettings):
    """Performance and resource configuration"""

    max_threads: int = Field(
        default=32,
        env="KIMERA_MAX_THREADS",
        ge=1,
        le=128,
        description="Maximum number of threads",
    )
    max_processes: int = Field(
        default=4,
        env="KIMERA_MAX_PROCESSES",
        ge=1,
        le=16,
        description="Maximum number of processes",
    )
    gpu_memory_fraction: float = Field(
        default=0.8,
        env="KIMERA_GPU_MEMORY_FRACTION",
        ge=0.1,
        le=1.0,
        description="GPU memory fraction to use",
    )
    batch_size: int = Field(
        default=100,
        env="KIMERA_BATCH_SIZE",
        ge=1,
        le=10000,
        description="Default batch size for processing",
    )
    cache_size: int = Field(
        default=1000, env="KIMERA_CACHE_SIZE", ge=0, description="Default cache size"
    )
    request_timeout: float = Field(
        default=30.0,
        env="KIMERA_REQUEST_TIMEOUT",
        ge=1.0,
        description="Default request timeout in seconds",
    )

    class Config:
        env_prefix = "KIMERA_"


class ServerSettings(BaseSettings):
    """Server configuration settings"""

    host: str = Field(default="127.0.0.1", env="KIMERA_HOST", description="Server host")
    port: int = Field(
        default=8000, env="KIMERA_PORT", ge=1, le=65535, description="Server port"
    )
    reload: bool = Field(
        default=False,
        env="KIMERA_RELOAD",
        description="Enable auto-reload (development only)",
    )
    workers: int = Field(
        default=1, env="KIMERA_WORKERS", ge=1, description="Number of worker processes"
    )
    cors_origins: List[str] = Field(
        default=["*"], env="KIMERA_CORS_ORIGINS", description="CORS allowed origins"
    )

    @validator("cors_origins", pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    @validator("reload")
    def validate_reload(cls, v, values):
        """Ensure reload is only enabled in development"""
        if v and os.getenv("KIMERA_ENV", "development") == "production":
            raise ValueError("Auto-reload cannot be enabled in production")
        return v

    class Config:
        env_prefix = "KIMERA_"


class LoggingSettings(BaseSettings):
    """Logging configuration settings"""

    level: LogLevel = Field(
        default=LogLevel.INFO, env="KIMERA_LOG_LEVEL", description="Logging level"
    )
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="KIMERA_LOG_FORMAT",
        description="Log message format",
    )
    file_enabled: bool = Field(
        default=True, env="KIMERA_LOG_FILE_ENABLED", description="Enable file logging"
    )
    file_rotation: str = Field(
        default="1 day",
        env="KIMERA_LOG_FILE_ROTATION",
        description="Log file rotation interval",
    )
    file_retention: str = Field(
        default="30 days",
        env="KIMERA_LOG_FILE_RETENTION",
        description="Log file retention period",
    )
    structured: bool = Field(
        default=True, env="KIMERA_LOG_STRUCTURED", description="Use structured logging"
    )

    class Config:
        env_prefix = "KIMERA_LOG_"


class ThresholdSettings(BaseSettings):
    """System health monitoring thresholds"""

    cpu_warning: float = Field(80.0, description="CPU usage warning threshold (%)")
    cpu_critical: float = Field(95.0, description="CPU usage critical threshold (%)")
    memory_warning: float = Field(
        80.0, description="Memory usage warning threshold (%)"
    )
    memory_critical: float = Field(
        95.0, description="Memory usage critical threshold (%)"
    )
    gpu_memory_warning: float = Field(
        85.0, description="GPU memory usage warning threshold (%)"
    )
    gpu_memory_critical: float = Field(
        95.0, description="GPU memory usage critical threshold (%)"
    )
    disk_warning: float = Field(85.0, description="Disk usage warning threshold (%)")
    disk_critical: float = Field(95.0, description="Disk usage critical threshold (%)")
    response_time_warning: float = Field(
        2.0, description="Response time warning threshold (s)"
    )
    response_time_critical: float = Field(
        5.0, description="Response time critical threshold (s)"
    )

    class Config:
        env_prefix = "KIMERA_MONITORING_THRESHOLDS_"


class MonitoringSettings(BaseSettings):
    """Monitoring and metrics configuration"""

    enabled: bool = Field(
        default=True, env="KIMERA_MONITORING_ENABLED", description="Enable monitoring"
    )
    metrics_port: int = Field(
        default=9090,
        env="KIMERA_METRICS_PORT",
        ge=1,
        le=65535,
        description="Metrics server port",
    )
    health_check_interval: float = Field(
        default=30.0,
        env="KIMERA_HEALTH_CHECK_INTERVAL",
        ge=1.0,
        description="Health check interval in seconds",
    )
    performance_tracking: bool = Field(
        default=True,
        env="KIMERA_PERFORMANCE_TRACKING",
        description="Enable performance tracking",
    )
    memory_tracking: bool = Field(
        default=True, env="KIMERA_MEMORY_TRACKING", description="Enable memory tracking"
    )

    thresholds: ThresholdSettings = Field(
        default_factory=ThresholdSettings, description="Health monitoring thresholds"
    )

    class Config:
        env_prefix = "KIMERA_"


class SecuritySettings(BaseSettings):
    """Security configuration settings"""

    secret_key: SecretStr = Field(
        default_factory=lambda: SecretStr(os.urandom(32).hex()),
        env="KIMERA_SECRET_KEY",
        description="Application secret key",
    )
    jwt_algorithm: str = Field(
        default="HS256", env="KIMERA_JWT_ALGORITHM", description="JWT signing algorithm"
    )
    jwt_expiration: timedelta = Field(
        default=timedelta(hours=24),
        env="KIMERA_JWT_EXPIRATION",
        description="JWT token expiration time",
    )
    rate_limit_enabled: bool = Field(
        default=True,
        env="KIMERA_RATE_LIMIT_ENABLED",
        description="Enable rate limiting",
    )
    rate_limit_requests: int = Field(
        default=100,
        env="KIMERA_RATE_LIMIT_REQUESTS",
        ge=1,
        description="Rate limit requests per period",
    )
    rate_limit_period: int = Field(
        default=60,
        env="KIMERA_RATE_LIMIT_PERIOD",
        ge=1,
        description="Rate limit period in seconds",
    )

    @validator("jwt_expiration", pre=True)
    def parse_jwt_expiration(cls, v):
        if isinstance(v, str):
            # Parse string like "24h", "7d", "30m"
            if v.endswith("h"):
                return timedelta(hours=int(v[:-1]))
            elif v.endswith("d"):
                return timedelta(days=int(v[:-1]))
            elif v.endswith("m"):
                return timedelta(minutes=int(v[:-1]))
        return v

    class Config:
        env_prefix = "KIMERA_"


class RedisSettings(BaseSettings):
    """Redis configuration settings"""

    # Connection settings
    host: str = Field(
        default="localhost", env="REDIS_HOST", description="Redis server host"
    )
    port: int = Field(
        default=6379, env="REDIS_PORT", ge=1, le=65535, description="Redis server port"
    )
    username: Optional[str] = Field(
        default=None, env="REDIS_USERNAME", description="Redis username (Redis 6.0+)"
    )
    password: Optional[SecretStr] = Field(
        default=None, env="REDIS_PASSWORD", description="Redis password"
    )

    # Database settings
    db: int = Field(
        default=0, env="REDIS_DB", ge=0, le=15, description="Redis database number"
    )

    # SSL/TLS settings
    ssl: bool = Field(
        default=False, env="REDIS_SSL", description="Enable SSL/TLS connection"
    )
    ssl_keyfile: Optional[str] = Field(
        default=None,
        env="REDIS_SSL_KEYFILE",
        description="Path to SSL private key file",
    )
    ssl_certfile: Optional[str] = Field(
        default=None,
        env="REDIS_SSL_CERTFILE",
        description="Path to SSL certificate file",
    )
    ssl_ca_path: Optional[str] = Field(
        default=None, env="REDIS_CA_PATH", description="Path to SSL CA certificate file"
    )
    ssl_ca_certs: Optional[str] = Field(
        default=None,
        env="REDIS_CA_CERTS",
        description="Path to SSL CA certificates bundle",
    )
    ssl_cert_reqs: str = Field(
        default="required",
        env="REDIS_CERT_REQS",
        pattern="^(none|optional|required)$",
        description="SSL certificate requirements (none/optional/required)",
    )

    # Connection pool settings
    max_connections: int = Field(
        default=50,
        env="REDIS_MAX_CONNECTIONS",
        ge=1,
        description="Maximum number of connections in the pool",
    )
    retry_on_timeout: bool = Field(
        default=True,
        env="REDIS_RETRY_ON_TIMEOUT",
        description="Retry commands on timeout",
    )
    health_check_interval: int = Field(
        default=30,
        env="REDIS_HEALTH_CHECK_INTERVAL",
        ge=0,
        description="Health check interval in seconds (0 to disable)",
    )

    # Timeouts
    socket_timeout: float = Field(
        default=5.0,
        env="REDIS_SOCKET_TIMEOUT",
        ge=0.1,
        description="Socket timeout in seconds",
    )
    socket_connect_timeout: float = Field(
        default=5.0,
        env="REDIS_SOCKET_CONNECT_TIMEOUT",
        ge=0.1,
        description="Socket connection timeout in seconds",
    )

    # Cluster settings
    cluster_mode: bool = Field(
        default=False, env="REDIS_CLUSTER_MODE", description="Enable Redis Cluster mode"
    )
    cluster_nodes: List[str] = Field(
        default_factory=list,
        env="REDIS_CLUSTER_NODES",
        description="Redis cluster nodes (host:port format)",
    )
    skip_full_coverage_check: bool = Field(
        default=False,
        env="REDIS_SKIP_FULL_COVERAGE_CHECK",
        description="Skip full coverage check in cluster mode",
    )

    # Performance settings
    decode_responses: bool = Field(
        default=True,
        env="REDIS_DECODE_RESPONSES",
        description="Automatically decode responses to strings",
    )
    encoding: str = Field(
        default="utf-8",
        env="REDIS_ENCODING",
        description="Character encoding for Redis responses",
    )

    @validator("cluster_nodes", pre=True)
    def parse_cluster_nodes(cls, v):
        """Parse cluster nodes from comma-separated string"""
        if isinstance(v, str):
            return [node.strip() for node in v.split(",") if node.strip()]
        return v

    @validator("ssl_keyfile", "ssl_certfile", "ssl_ca_path", "ssl_ca_certs")
    def validate_ssl_files(cls, v):
        """Validate SSL file paths exist if provided"""
        if v and not Path(v).exists():
            raise ValueError(f"SSL file not found: {v}")
        return v

    @property
    def connection_url(self) -> str:
        """Generate Redis connection URL"""
        scheme = "rediss" if self.ssl else "redis"
        auth = ""

        if self.username and self.password:
            auth = f"{self.username}:{self.password.get_secret_value()}@"
        elif self.password:
            auth = f":{self.password.get_secret_value()}@"

        return f"{scheme}://{auth}{self.host}:{self.port}/{self.db}"

    @property
    def connection_kwargs(self) -> Dict[str, Any]:
        """Generate connection kwargs for redis client"""
        kwargs = {
            "host": self.host,
            "port": self.port,
            "db": self.db,
            "socket_timeout": self.socket_timeout,
            "socket_connect_timeout": self.socket_connect_timeout,
            "retry_on_timeout": self.retry_on_timeout,
            "decode_responses": self.decode_responses,
            "encoding": self.encoding,
            "max_connections": self.max_connections,
        }

        # Authentication
        if self.username:
            kwargs["username"] = self.username
        if self.password:
            kwargs["password"] = self.password.get_secret_value()

        # SSL settings
        if self.ssl:
            kwargs["ssl"] = True
            if self.ssl_keyfile:
                kwargs["ssl_keyfile"] = self.ssl_keyfile
            if self.ssl_certfile:
                kwargs["ssl_certfile"] = self.ssl_certfile
            if self.ssl_ca_path:
                kwargs["ssl_ca_certs"] = self.ssl_ca_path
            elif self.ssl_ca_certs:
                kwargs["ssl_ca_certs"] = self.ssl_ca_certs

            # Map cert_reqs string to ssl module constants
            cert_reqs_map = {
                "none": 0,  # ssl.CERT_NONE
                "optional": 1,  # ssl.CERT_OPTIONAL
                "required": 2,  # ssl.CERT_REQUIRED
            }
            kwargs["ssl_cert_reqs"] = cert_reqs_map.get(self.ssl_cert_reqs, 2)

        # Health check
        if self.health_check_interval > 0:
            kwargs["health_check_interval"] = self.health_check_interval

        return kwargs

    class Config:
        env_prefix = "REDIS_"


class KimeraSettings(BaseSettings):
    """Main KIMERA configuration settings"""

    # Environment
    environment: EnvironmentType = Field(
        default=EnvironmentType.DEVELOPMENT,
        env="KIMERA_ENV",
        description="Application environment",
    )

    # Sub-configurations
    database: DatabaseSettings = Field(
        default_factory=DatabaseSettings, description="Database settings"
    )
    api_keys: APIKeysSettings = Field(
        default_factory=APIKeysSettings, description="API keys settings"
    )
    paths: PathSettings = Field(
        default_factory=PathSettings, description="Path settings"
    )
    performance: PerformanceSettings = Field(
        default_factory=PerformanceSettings, description="Performance settings"
    )
    server: ServerSettings = Field(
        default_factory=ServerSettings, description="Server settings"
    )
    logging: LoggingSettings = Field(
        default_factory=LoggingSettings, description="Logging settings"
    )
    monitoring: MonitoringSettings = Field(
        default_factory=MonitoringSettings, description="Monitoring settings"
    )
    security: SecuritySettings = Field(
        default_factory=SecuritySettings, description="Security settings"
    )
    redis: RedisSettings = Field(
        default_factory=RedisSettings, description="Redis settings"
    )

    # Feature flags
    features: Dict[str, bool] = Field(
        default_factory=dict, env="KIMERA_FEATURES", description="Feature flags as JSON"
    )

    @validator("features", pre=True)
    def parse_features(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return {}
        return v

    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment == EnvironmentType.PRODUCTION

    @property
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.environment == EnvironmentType.DEVELOPMENT

    def get_feature(self, feature_name: str, default: bool = False) -> bool:
        """Get feature flag value"""
        return self.features.get(feature_name, default)

    class Config:
        # Pydantic-settings configuration
        env_prefix = "KIMERA_"
        if os.getenv("KIMERA_ENV", "development") == "development":
            env_file = ".env.dev"
        else:
            env_file = None
        env_file_encoding = "utf-8"
        env_nested_delimiter = "__"
        extra = "allow"

    def dict(self, **kwargs) -> Dict[str, Any]:
        """Override dict to exclude secrets in non-production"""
        # Convert Path objects to strings for JSON serialization
        data = super().dict(**kwargs)

        # Convert Path objects to strings
        if "paths" in data:
            for key, value in data["paths"].items():
                if isinstance(value, Path):
                    data["paths"][key] = str(value)

        if not self.is_production:
            # Mask sensitive data in non-production logs
            if "api_keys" in data:
                for key in data["api_keys"]:
                    if (
                        isinstance(data["api_keys"][key], dict)
                        and "secret_value" in data["api_keys"][key]
                    ):
                        data["api_keys"][key] = "***MASKED***"
        return data


# Global settings instance
_settings: Optional[KimeraSettings] = None


def get_settings(force_reload: bool = False) -> KimeraSettings:
    """
    Get the global settings instance.

    This function uses a cached instance for performance.
    """
    global _settings
    if _settings is None or force_reload:
        _settings = KimeraSettings()
    return _settings


def reload_settings() -> KimeraSettings:
    """
    Reload the settings from all sources.

    This is useful for testing when environment variables are changed.
    """
    return get_settings(force_reload=True)
