#!/usr/bin/env python3
"""
KIMERA SWM Configuration Consolidator
====================================

Consolidates scattered configuration files into a unified, hierarchical configuration system.
Merges environment files, validates settings, and creates a centralized config management system.

Author: Kimera SWM Autonomous Architect
Date: January 31, 2025
Version: 1.0.0
"""

import os
import sys
import json
import yaml
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
import re

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ConfigConsolidator:
    """
    Consolidates configuration files into a unified hierarchical system.
    """
    
    def __init__(self, kimera_root: str):
        self.kimera_root = Path(kimera_root)
        self.config_dir = self.kimera_root / "config"
        self.consolidated_dir = self.kimera_root / "configs_consolidated"
        self.backup_dir = self.kimera_root / "archive" / f"2025-07-31_config_backup"
        
        self.config_files = {}
        self.env_files = {}
        self.merged_configs = {}
        self.conflicts = []
    
    def create_backup(self):
        """Create backup of existing configuration files."""
        logger.info("Creating backup of existing configuration files...")
        
        os.makedirs(self.backup_dir, exist_ok=True)
        
        # Backup config directory
        if self.config_dir.exists():
            config_backup = self.backup_dir / "config"
            shutil.copytree(self.config_dir, config_backup, dirs_exist_ok=True)
            logger.info(f"Config directory backed up to: {config_backup}")
        
        # Backup configs directory if it exists
        configs_dir = self.kimera_root / "configs"
        if configs_dir.exists():
            configs_backup = self.backup_dir / "configs"
            shutil.copytree(configs_dir, configs_backup, dirs_exist_ok=True)
            logger.info(f"Configs directory backed up to: {configs_backup}")
        
        # Create backup manifest
        manifest_path = self.backup_dir / "BACKUP_MANIFEST.md"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            f.write(f"# Configuration Consolidator Backup\n")
            f.write(f"**Created**: {datetime.now().isoformat()}\n")
            f.write(f"**Purpose**: Backup before configuration consolidation\n\n")
            f.write(f"## Backup Contents\n")
            f.write(f"- `config/` - Original configuration directory\n")
            f.write(f"- `configs/` - Secondary configs directory\n\n")
            f.write(f"## Restoration\n")
            f.write(f"To restore, copy contents back to original locations.\n")
    
    def discover_config_files(self):
        """Discover all configuration files in the system."""
        logger.info("Discovering configuration files...")
        
        # Common config file patterns
        config_patterns = [
            "*.yaml", "*.yml", "*.json", "*.env", "*.ini", "*.toml", "*.conf"
        ]
        
        # Search directories
        search_dirs = [
            self.config_dir,
            self.kimera_root / "configs"
        ]
        
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            
            for pattern in config_patterns:
                for config_file in search_dir.rglob(pattern):
                    if config_file.is_file():
                        relative_path = config_file.relative_to(search_dir)
                        category = self._categorize_config_file(config_file)
                        
                        if category not in self.config_files:
                            self.config_files[category] = []
                        
                        self.config_files[category].append({
                            'path': config_file,
                            'relative_path': relative_path,
                            'size': config_file.stat().st_size,
                            'type': config_file.suffix,
                            'content': self._load_config_content(config_file)
                        })
        
        total_files = sum(len(files) for files in self.config_files.values())
        logger.info(f"Discovered {total_files} configuration files in {len(self.config_files)} categories")
    
    def _categorize_config_file(self, config_path: Path) -> str:
        """Categorize configuration file based on name and location."""
        name = config_path.name.lower()
        
        if 'trading' in name or 'binance' in name or 'cdp' in name:
            return 'trading'
        elif 'docker' in name or 'compose' in name:
            return 'docker'
        elif 'grafana' in name or 'prometheus' in name:
            return 'monitoring'
        elif 'development' in name or 'dev' in name:
            return 'development'
        elif 'production' in name or 'prod' in name:
            return 'production'
        elif 'test' in name:
            return 'testing'
        elif 'redis' in name:
            return 'database'
        elif 'ai' in name or 'fine_tuning' in name:
            return 'ai_ml'
        elif name.endswith('.env'):
            return 'environment'
        else:
            return 'general'
    
    def _load_config_content(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration file content."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if config_path.suffix in ['.yaml', '.yml']:
                return yaml.safe_load(content) or {}
            elif config_path.suffix == '.json':
                return json.loads(content) or {}
            elif config_path.suffix == '.env':
                return self._parse_env_file(content)
            else:
                # Return raw content for other types
                return {'raw_content': content}
        
        except Exception as e:
            logger.warning(f"Could not load config file {config_path}: {e}")
            return {}
    
    def _parse_env_file(self, content: str) -> Dict[str, str]:
        """Parse environment file content."""
        env_vars = {}
        for line in content.splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                # Remove quotes if present
                value = value.strip().strip('"').strip("'")
                env_vars[key.strip()] = value
        return env_vars
    
    def analyze_conflicts(self):
        """Analyze configuration conflicts and overlaps."""
        logger.info("Analyzing configuration conflicts...")
        
        # Track configuration keys across files
        all_keys = {}  # key -> [(file_path, value, category)]
        
        for category, files in self.config_files.items():
            for file_info in files:
                content = file_info['content']
                if isinstance(content, dict):
                    for key, value in content.items():
                        if key not in all_keys:
                            all_keys[key] = []
                        all_keys[key].append((file_info['path'], value, category))
        
        # Find conflicts
        for key, occurrences in all_keys.items():
            if len(occurrences) > 1:
                # Check if values are different
                values = [str(occ[1]) for occ in occurrences]
                if len(set(values)) > 1:
                    self.conflicts.append({
                        'key': key,
                        'occurrences': occurrences,
                        'conflict_type': 'value_mismatch'
                    })
        
        logger.info(f"Found {len(self.conflicts)} configuration conflicts")
    
    def create_unified_config_structure(self):
        """Create unified configuration structure."""
        logger.info("Creating unified configuration structure...")
        
        os.makedirs(self.consolidated_dir, exist_ok=True)
        
        # Create environment-based structure
        environments = ['development', 'production', 'testing']
        
        for env in environments:
            env_dir = self.consolidated_dir / env
            os.makedirs(env_dir, exist_ok=True)
            
            # Create base config for this environment
            base_config = self._create_base_config(env)
            
            # Add environment-specific overrides
            env_config = self._merge_environment_configs(env)
            
            # Merge configs
            merged_config = {**base_config, **env_config}
            
            # Save as YAML
            config_path = env_dir / "config.yaml"
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(merged_config, f, default_flow_style=False, indent=2)
            
            logger.info(f"Created {env} environment config: {config_path}")
            
            # Create category-specific configs
            self._create_category_configs(env_dir, env)
        
        # Create shared configurations
        self._create_shared_configs()
        
        # Create environment files
        self._create_consolidated_env_files()
    
    def _create_base_config(self, environment: str) -> Dict[str, Any]:
        """Create base configuration for an environment."""
        return {
            'app': {
                'name': 'KIMERA SWM',
                'version': '2.0.0',
                'environment': environment,
                'debug': environment == 'development'
            },
            'server': {
                'host': '127.0.0.1',
                'port': 8000,
                'auto_port_discovery': True,
                'cors_enabled': True
            },
            'logging': {
                'level': 'DEBUG' if environment == 'development' else 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
            'features': {
                'monitoring': True,
                'metrics': True,
                'health_checks': True,
                'api_documentation': True
            }
        }
    
    def _merge_environment_configs(self, environment: str) -> Dict[str, Any]:
        """Merge configs specific to an environment."""
        merged = {}
        
        # Get configs for this environment
        env_files = self.config_files.get(environment, [])
        env_files.extend(self.config_files.get('general', []))
        
        for file_info in env_files:
            content = file_info['content']
            if isinstance(content, dict):
                merged = self._deep_merge(merged, content)
        
        return merged
    
    def _deep_merge(self, base: Dict, overlay: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in overlay.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _create_category_configs(self, env_dir: Path, environment: str):
        """Create category-specific configuration files."""
        categories = ['trading', 'monitoring', 'database', 'ai_ml']
        
        for category in categories:
            if category in self.config_files:
                category_config = {}
                
                for file_info in self.config_files[category]:
                    content = file_info['content']
                    if isinstance(content, dict):
                        category_config = self._deep_merge(category_config, content)
                
                if category_config:
                    category_path = env_dir / f"{category}.yaml"
                    with open(category_path, 'w', encoding='utf-8') as f:
                        yaml.dump(category_config, f, default_flow_style=False, indent=2)
                    
                    logger.info(f"Created {category} config for {environment}: {category_path}")
    
    def _create_shared_configs(self):
        """Create shared configuration files."""
        shared_dir = self.consolidated_dir / "shared"
        os.makedirs(shared_dir, exist_ok=True)
        
        # Docker configurations
        if 'docker' in self.config_files:
            docker_dir = shared_dir / "docker"
            os.makedirs(docker_dir, exist_ok=True)
            
            for file_info in self.config_files['docker']:
                target_path = docker_dir / file_info['relative_path']
                os.makedirs(target_path.parent, exist_ok=True)
                shutil.copy2(file_info['path'], target_path)
        
        # Monitoring configurations
        if 'monitoring' in self.config_files:
            monitoring_dir = shared_dir / "monitoring"
            os.makedirs(monitoring_dir, exist_ok=True)
            
            for file_info in self.config_files['monitoring']:
                target_path = monitoring_dir / file_info['relative_path']
                os.makedirs(target_path.parent, exist_ok=True)
                shutil.copy2(file_info['path'], target_path)
    
    def _create_consolidated_env_files(self):
        """Create consolidated environment files."""
        env_dir = self.consolidated_dir / "env"
        os.makedirs(env_dir, exist_ok=True)
        
        # Consolidate all environment variables
        all_env_vars = {}
        
        for category, files in self.config_files.items():
            for file_info in files:
                if file_info['type'] == '.env':
                    content = file_info['content']
                    if isinstance(content, dict):
                        all_env_vars.update(content)
        
        # Create environment-specific .env files
        environments = ['development', 'production', 'testing']
        
        for env in environments:
            env_file_path = env_dir / f"{env}.env"
            
            with open(env_file_path, 'w', encoding='utf-8') as f:
                f.write(f"# KIMERA SWM Environment Variables - {env.upper()}\n")
                f.write(f"# Generated: {datetime.now().isoformat()}\n\n")
                
                # Environment-specific settings
                f.write(f"KIMERA_MODE={'progressive' if env == 'development' else 'full'}\n")
                f.write(f"DEBUG={'true' if env == 'development' else 'false'}\n")
                f.write(f"ENVIRONMENT={env}\n\n")
                
                # Add other environment variables
                for key, value in all_env_vars.items():
                    if not key.startswith(('KIMERA_MODE', 'DEBUG', 'ENVIRONMENT')):
                        f.write(f"{key}={value}\n")
            
            logger.info(f"Created consolidated env file: {env_file_path}")
    
    def generate_consolidation_report(self) -> str:
        """Generate detailed consolidation report."""
        total_files = sum(len(files) for files in self.config_files.values())
        
        report_lines = [
            "# KIMERA SWM Configuration Consolidation Report",
            f"**Generated**: {datetime.now().isoformat()}",
            f"**Configuration Files Processed**: {total_files}",
            f"**Categories**: {len(self.config_files)}",
            f"**Conflicts Found**: {len(self.conflicts)}",
            "",
            "## Configuration Categories",
            ""
        ]
        
        for category, files in self.config_files.items():
            report_lines.extend([
                f"### {category.title()}",
                f"**Files**: {len(files)}",
                ""
            ])
            
            for file_info in files:
                report_lines.append(f"- `{file_info['relative_path']}` ({file_info['size']} bytes)")
            
            report_lines.extend(["", "---", ""])
        
        if self.conflicts:
            report_lines.extend([
                "## Configuration Conflicts",
                ""
            ])
            
            for conflict in self.conflicts:
                key = conflict['key']
                report_lines.extend([
                    f"### {key}",
                    "**Conflicting values:**",
                    ""
                ])
                
                for path, value, category in conflict['occurrences']:
                    report_lines.append(f"- `{category}`: `{value}` (from `{path.name}`)")
                
                report_lines.extend(["", "---", ""])
        
        report_lines.extend([
            "## Consolidated Structure",
            "",
            "```",
            "configs_consolidated/",
            "├── development/",
            "│   ├── config.yaml",
            "│   ├── trading.yaml",
            "│   ├── monitoring.yaml",
            "│   └── database.yaml",
            "├── production/",
            "│   ├── config.yaml",
            "│   ├── trading.yaml",
            "│   ├── monitoring.yaml",
            "│   └── database.yaml",
            "├── testing/",
            "│   └── config.yaml",
            "├── shared/",
            "│   ├── docker/",
            "│   └── monitoring/",
            "└── env/",
            "    ├── development.env",
            "    ├── production.env",
            "    └── testing.env",
            "```",
            "",
            "## Usage",
            "",
            "### Loading Configuration",
            "```python",
            "from src.config.unified_config import load_config",
            "",
            "# Load environment-specific config",
            "config = load_config('development')",
            "",
            "# Load category-specific config",
            "trading_config = load_config('development', 'trading')",
            "```",
            "",
            "### Environment Variables",
            "```bash",
            "# Load environment variables",
            "source configs_consolidated/env/development.env",
            "```",
            "",
            "## Migration Notes",
            "",
            "1. **Update import statements** to use unified config loader",
            "2. **Test all configuration loading** in different environments",
            "3. **Verify environment variable resolution**",
            "4. **Update deployment scripts** to use consolidated configs",
            "5. **Remove old configuration files** after verification",
            "",
            f"## Backup Location",
            f"Original configurations backed up to: `{self.backup_dir.relative_to(self.kimera_root)}`"
        ])
        
        return "\n".join(report_lines)
    
    def save_report(self, report_content: str):
        """Save consolidation report."""
        date_str = datetime.now().strftime('%Y-%m-%d')
        report_path = self.kimera_root / "docs" / "reports" / "analysis" / f"{date_str}_config_consolidation_report.md"
        
        # Ensure directory exists
        os.makedirs(report_path.parent, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"Consolidation report saved to: {report_path}")
    
    def create_config_loader(self):
        """Create unified configuration loader."""
        loader_path = self.kimera_root / "src" / "config" / "unified_config.py"
        
        # Ensure directory exists
        os.makedirs(loader_path.parent, exist_ok=True)
        
        loader_content = f'''#!/usr/bin/env python3
"""
KIMERA SWM Unified Configuration Loader
=======================================

Unified configuration loading system for all environments and categories.

Generated by: Kimera SWM Autonomous Architect
Date: {datetime.now().isoformat()}
Version: 1.0.0
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class UnifiedConfigLoader:
    """Unified configuration loader for KIMERA SWM."""
    
    def __init__(self, config_root: Optional[Path] = None):
        if config_root is None:
            # Default to configs_consolidated directory
            current_dir = Path(__file__).parent.parent.parent
            config_root = current_dir / "configs_consolidated"
        
        self.config_root = Path(config_root)
        self._cache = {{}}
    
    def load_config(self, environment: str = None, category: str = None) -> Dict[str, Any]:
        """
        Load configuration for specified environment and category.
        
        Args:
            environment: Environment name (development, production, testing)
            category: Config category (trading, monitoring, database, ai_ml)
            
        Returns:
            Dictionary containing configuration
        """
        # Default environment from ENV var or development
        if environment is None:
            environment = os.getenv('ENVIRONMENT', 'development')
        
        cache_key = f"{{environment}}_{{category or 'main'}}"
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        config = {{}}
        
        try:
            if category is None:
                # Load main environment config
                config_path = self.config_root / environment / "config.yaml"
            else:
                # Load category-specific config
                config_path = self.config_root / environment / f"{{category}}.yaml"
            
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f) or {{}}
                
                logger.info(f"Loaded config: {{config_path}}")
            else:
                logger.warning(f"Config file not found: {{config_path}}")
        
        except Exception as e:
            logger.error(f"Error loading config {{cache_key}}: {{e}}")
        
        # Cache the result
        self._cache[cache_key] = config
        
        return config
    
    def get_env_file_path(self, environment: str = None) -> Path:
        """Get path to environment file."""
        if environment is None:
            environment = os.getenv('ENVIRONMENT', 'development')
        
        return self.config_root / "env" / f"{{environment}}.env"
    
    def load_env_vars(self, environment: str = None):
        """Load environment variables from file."""
        env_file = self.get_env_file_path(environment)
        
        if env_file.exists():
            from dotenv import load_dotenv
            load_dotenv(env_file)
            logger.info(f"Loaded environment variables from: {{env_file}}")
        else:
            logger.warning(f"Environment file not found: {{env_file}}")


# Global loader instance
_loader = UnifiedConfigLoader()

def load_config(environment: str = None, category: str = None) -> Dict[str, Any]:
    """Load configuration using global loader."""
    return _loader.load_config(environment, category)

def load_env_vars(environment: str = None):
    """Load environment variables using global loader."""
    return _loader.load_env_vars(environment)

def get_config_value(key: str, default: Any = None, environment: str = None, category: str = None) -> Any:
    """Get specific configuration value with dot notation support."""
    config = load_config(environment, category)
    
    # Support dot notation (e.g., "app.name")
    keys = key.split('.')
    value = config
    
    try:
        for k in keys:
            value = value[k]
        return value
    except (KeyError, TypeError):
        return default
'''
        
        with open(loader_path, 'w', encoding='utf-8') as f:
            f.write(loader_content)
        
        logger.info(f"Created unified config loader: {loader_path}")
    
    def run_consolidation(self):
        """Execute the complete configuration consolidation process."""
        logger.info("Starting KIMERA SWM configuration consolidation...")
        
        # Create backup
        self.create_backup()
        
        # Discover configuration files
        self.discover_config_files()
        
        # Analyze conflicts
        self.analyze_conflicts()
        
        # Create unified structure
        self.create_unified_config_structure()
        
        # Create config loader
        self.create_config_loader()
        
        # Generate and save report
        report = self.generate_consolidation_report()
        self.save_report(report)
        
        logger.info("✅ Configuration consolidation completed successfully!")
        logger.info(f"📁 Consolidated configs: {self.consolidated_dir}")
        logger.info(f"📁 Backup location: {self.backup_dir}")
        
        total_files = sum(len(files) for files in self.config_files.values())
        
        return {
            'config_files_processed': total_files,
            'categories': len(self.config_files),
            'conflicts_found': len(self.conflicts),
            'consolidation_successful': True
        }


def main():
    """Main execution function."""
    if len(sys.argv) > 1:
        kimera_root = sys.argv[1]
    else:
        # Default to current directory
        kimera_root = os.getcwd()
    
    if not os.path.exists(kimera_root):
        logger.error(f"Kimera root does not exist: {kimera_root}")
        sys.exit(1)
    
    consolidator = ConfigConsolidator(kimera_root)
    result = consolidator.run_consolidation()
    
    if result['consolidation_successful']:
        logger.info(f"✅ Successfully processed {result['config_files_processed']} configuration files")
        logger.info(f"📊 Organized into {result['categories']} categories")
        logger.info(f"⚠️ Found {result['conflicts_found']} conflicts")
        logger.info(f"📁 Consolidated configs: {consolidator.consolidated_dir}")
        logger.info(f"📁 Backup available at: {consolidator.backup_dir}")
    else:
        logger.info("❌ Configuration consolidation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()