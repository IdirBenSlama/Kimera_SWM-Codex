"""
KIMERA SWM System - Unified Configuration Loader
===============================================

Centralized configuration management for all KIMERA components.
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional

class KimeraConfigLoader:
    """Unified configuration loader for KIMERA SWM System"""
    
    def __init__(self, config_root: str = "config_unified"):
        self.config_root = Path(config_root)
        self.environment = os.getenv("KIMERA_ENV", "development")
        self._cache = {}
    
    def load_environment_config(self, env: Optional[str] = None) -> Dict[str, Any]:
        """Load environment-specific configuration"""
        env = env or self.environment
        cache_key = f"env_{env}"
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        config = {}
        env_dir = self.config_root / "environments" / env
        
        if env_dir.exists():
            for config_file in env_dir.glob("*.yaml"):
                with open(config_file) as f:
                    file_config = yaml.safe_load(f)
                    if file_config:
                        config.update(file_config)
        
        self._cache[cache_key] = config
        return config
    
    def load_component_config(self, component: str) -> Dict[str, Any]:
        """Load component-specific configuration"""
        cache_key = f"component_{component}"
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        config = {}
        component_paths = [
            self.config_root / "shared" / component,
            self.config_root / "environments" / self.environment / f"{component}.yaml"
        ]
        
        for path in component_paths:
            if path.is_dir():
                for config_file in path.glob("*.yaml"):
                    with open(config_file) as f:
                        file_config = yaml.safe_load(f)
                        if file_config:
                            config.update(file_config)
            elif path.is_file():
                with open(path) as f:
                    file_config = yaml.safe_load(f)
                    if file_config:
                        config.update(file_config)
        
        self._cache[cache_key] = config
        return config
    
    def get_config(self, key: str = None) -> Any:
        """Get configuration value by key path (e.g. 'database.host')"""
        env_config = self.load_environment_config()
        
        if not key:
            return env_config
        
        keys = key.split('.')
        value = env_config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return None
        
        return value
    
    def clear_cache(self):
        """Clear configuration cache"""
        self._cache.clear()

# Global instance
config_loader = KimeraConfigLoader()

# Convenience functions
def load_config(component: str = None) -> Dict[str, Any]:
    """Load configuration for component or environment"""
    if component:
        return config_loader.load_component_config(component)
    return config_loader.load_environment_config()

def get_config(key: str) -> Any:
    """Get configuration value by key"""
    return config_loader.get_config(key) 