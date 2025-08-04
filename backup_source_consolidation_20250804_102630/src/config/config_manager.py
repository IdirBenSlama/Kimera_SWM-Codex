import os
import json
import logging
from typing import Dict, Optional, Any
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Enhanced configuration manager for Kimera SWM
    Securely manages API keys and credentials from environment variables
    """
    
    # Dictionary of supported API services and their environment variable names
    API_SERVICES = {
        "openai": "OPENAI_API_KEY",
        "cryptopanic": "CRYPTOPANIC_API_KEY",
        "huggingface": "HUGGINGFACE_TOKEN",
        "alpha_vantage": "ALPHA_VANTAGE_API_KEY",
        "finnhub": "FINNHUB_API_KEY",
        "twelve_data": "TWELVE_DATA_API_KEY"
    }
    
    @staticmethod
    def get_database_url() -> str:
        """Get database URL from environment variables"""
        return os.getenv('DATABASE_URL', 'sqlite:///kimera_swm.db')

    @staticmethod
    def get_api_key(service: str) -> Optional[str]:
        """
        Get API key for a specific service
        
        Args:
            service: Service name (e.g., 'openai', 'cryptopanic')
            
        Returns:
            API key or None if not configured
        """
        # Check if service is in our known services
        if service in ConfigManager.API_SERVICES:
            env_var = ConfigManager.API_SERVICES[service]
            api_key = os.getenv(env_var)
            if not api_key:
                logger.warning(f"API key for {service} not found in environment variables")
            return api_key
        
        # Check custom API keys
        custom_keys = ConfigManager.get_custom_api_keys()
        if service in custom_keys:
            return custom_keys[service]
            
        # Legacy fallback for generic API_KEY
        if service == "generic":
            return os.getenv('API_KEY', '')
            
        logger.warning(f"Unknown API service: {service}")
        return None
        
    @staticmethod
    def get_custom_api_keys() -> Dict[str, str]:
        """Get custom API keys from environment variables"""
        custom_keys_json = os.getenv('KIMERA_CUSTOM_API_KEYS', '{}')
        try:
            return json.loads(custom_keys_json)
        except json.JSONDecodeError:
            logger.error("Failed to parse KIMERA_CUSTOM_API_KEYS as JSON")
            return {}
            
    @staticmethod
    def get_secret(name: str, default: str = '') -> str:
        """
        Get a secret value from environment variables
        
        Args:
            name: Secret name
            default: Default value if not found
            
        Returns:
            Secret value or default
        """
        return os.getenv(name, default)
        
    @staticmethod
    def get_feature_flag(feature: str, default: bool = False) -> bool:
        """
        Get feature flag from environment variables
        
        Args:
            feature: Feature name
            default: Default value if not found
            
        Returns:
            Boolean feature flag value
        """
        features_json = os.getenv('KIMERA_FEATURES', '{}')
        try:
            features = json.loads(features_json)
            return features.get(feature, default)
        except json.JSONDecodeError:
            return default
            
    @staticmethod
    def get_env() -> str:
        """Get current environment name"""
        return os.getenv('KIMERA_ENV', 'development')
        
    @staticmethod
    def is_production() -> bool:
        """Check if running in production environment"""
        return ConfigManager.get_env().lower() == 'production'
