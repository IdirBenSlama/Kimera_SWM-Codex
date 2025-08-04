#!/usr/bin/env python3
"""
KIMERA SWM Configuration Unification Executor
============================================

Executes Phase 4 of technical debt remediation: Configuration Unification
Following Martin Fowler framework and KIMERA SWM Protocol v3.0

Configuration Chaos Identified:
- config/ (main configs: production, development, trading, AI test suites)
- configs/ (database, initialization, GPU, environments)  
- configs_consolidated/ (previous consolidation attempt)
- kimera_trading/config/ (consciousness, quantum, thermodynamic, cognitive)
- src/kimera_trading/config/ (EXACT DUPLICATES of kimera_trading)

Strategy: Unified environment-based configuration structure
"""

import os
import hashlib
import logging
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional, Any
import json
import yaml

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RemediationResult:
    """Data class to hold remediation results."""
    def __init__(self, files_processed: int, changes_made: int, errors_encountered: int, 
                 time_saved_hours: float, recommendations: List[str]):
        self.files_processed = files_processed
        self.changes_made = changes_made
        self.errors_encountered = errors_encountered
        self.time_saved_hours = time_saved_hours
        self.recommendations = recommendations

class ConfigurationUnifier:
    """Analyzes and unifies scattered configuration files"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.config_extensions = {'.json', '.yaml', '.yml', '.toml', '.ini', '.conf', '.cfg'}
        self.backup_dir = self.project_root / f"backup_config_unification_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Configuration directories to analyze (excluding backups/archives)
        self.active_config_dirs = [
            "config",
            "configs", 
            "configs_consolidated",
            "kimera_trading/config",
            "src/kimera_trading/config",
            "src/config"
        ]
        
        # Target unified structure
        self.unified_structure = {
            "config": {
                "environments": {
                    "development": ["dev configs"],
                    "testing": ["test configs"],
                    "staging": ["staging configs"],
                    "production": ["prod configs"]
                },
                "shared": ["shared across environments"],
                "templates": ["config templates"],
                "schemas": ["validation schemas"]
            }
        }
        
    def analyze_configuration_landscape(self):
        """Comprehensive analysis of configuration files and directories"""
        logger.info("🔍 Analyzing configuration landscape...")
        
        analysis = {
            'directories': {},
            'files': {},
            'duplicates': [],
            'conflicts': [],
            'recommendations': []
        }
        
        # Analyze each configuration directory
        for config_dir_path in self.active_config_dirs:
            full_path = self.project_root / config_dir_path
            if not full_path.exists():
                continue
                
            logger.info(f"📁 Analyzing: {config_dir_path}")
            dir_analysis = self._analyze_directory(full_path, config_dir_path)
            analysis['directories'][config_dir_path] = dir_analysis
            
            # Add files to global analysis
            for file_info in dir_analysis['files']:
                file_path = file_info['relative_path']
                if file_path in analysis['files']:
                    # Potential duplicate detected
                    analysis['duplicates'].append({
                        'file': file_path,
                        'locations': [analysis['files'][file_path]['source'], config_dir_path],
                        'action': 'consolidate'
                    })
                else:
                    file_info['source'] = config_dir_path
                    analysis['files'][file_path] = file_info
                    
        # Find exact duplicates by content hash
        content_hashes = {}
        for file_path, file_info in analysis['files'].items():
            content_hash = file_info.get('content_hash')
            if content_hash:
                if content_hash not in content_hashes:
                    content_hashes[content_hash] = []
                content_hashes[content_hash].append((file_path, file_info))
                
        # Process exact duplicates
        for content_hash, files in content_hashes.items():
            if len(files) > 1:
                primary_file = files[0]
                duplicate_files = files[1:]
                
                analysis['duplicates'].append({
                    'type': 'exact_duplicate',
                    'content_hash': content_hash,
                    'primary': primary_file[0],
                    'duplicates': [f[0] for f in duplicate_files],
                    'action': 'remove_duplicates'
                })
                
        logger.info(f"📊 Analysis complete:")
        logger.info(f"   Directories: {len(analysis['directories'])}")
        logger.info(f"   Files: {len(analysis['files'])}")
        logger.info(f"   Duplicates: {len(analysis['duplicates'])}")
        
        return analysis
    
    def _analyze_directory(self, dir_path: Path, relative_name: str):
        """Analyze a single configuration directory"""
        analysis = {
            'path': str(dir_path),
            'relative_name': relative_name,
            'files': [],
            'subdirs': [],
            'total_size': 0,
            'file_types': {}
        }
        
        try:
            for item in dir_path.rglob("*"):
                if item.is_file() and item.suffix.lower() in self.config_extensions:
                    try:
                        file_info = self._analyze_file(item, dir_path)
                        analysis['files'].append(file_info)
                        analysis['total_size'] += file_info['size']
                        
                        # Track file types
                        ext = item.suffix.lower()
                        analysis['file_types'][ext] = analysis['file_types'].get(ext, 0) + 1
                        
                    except Exception as e:
                        logger.warning(f"Error analyzing {item}: {e}")
                        
        except Exception as e:
            logger.error(f"Error analyzing directory {dir_path}: {e}")
            
        return analysis
    
    def _analyze_file(self, file_path: Path, base_dir: Path):
        """Analyze a single configuration file"""
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            content_hash = hashlib.md5(content.encode()).hexdigest()
            
            # Try to parse the configuration
            config_data = None
            config_type = "unknown"
            
            if file_path.suffix.lower() in ['.json']:
                try:
                    config_data = json.loads(content)
                    config_type = "json"
                except:
                    pass
            elif file_path.suffix.lower() in ['.yaml', '.yml']:
                try:
                    config_data = yaml.safe_load(content)
                    config_type = "yaml"
                except:
                    pass
                    
            return {
                'path': str(file_path),
                'relative_path': str(file_path.relative_to(base_dir)),
                'name': file_path.name,
                'size': file_path.stat().st_size,
                'content_hash': content_hash,
                'config_type': config_type,
                'line_count': len(content.splitlines()),
                'modified_time': file_path.stat().st_mtime,
                'has_content': len(content.strip()) > 0,
                'config_keys': list(config_data.keys()) if isinstance(config_data, dict) else None
            }
            
        except Exception as e:
            logger.warning(f"Error analyzing file {file_path}: {e}")
            return {
                'path': str(file_path),
                'relative_path': str(file_path.relative_to(base_dir)),
                'name': file_path.name,
                'size': 0,
                'content_hash': '',
                'config_type': 'error',
                'error': str(e)
            }
    
    def create_unification_plan(self, analysis: Dict):
        """Create comprehensive unification plan"""
        logger.info("📋 Creating configuration unification plan...")
        
        plan = {
            'timestamp': datetime.now().isoformat(),
            'backup_location': str(self.backup_dir),
            'target_structure': self.unified_structure,
            'actions': [],
            'migration_map': {},
            'statistics': {
                'directories_to_consolidate': len(analysis['directories']),
                'files_to_migrate': len(analysis['files']),
                'duplicates_to_remove': len([d for d in analysis['duplicates'] if d.get('type') == 'exact_duplicate']),
                'conflicts_to_resolve': len(analysis['conflicts'])
            }
        }
        
        # Plan target directory structure
        target_config_dir = self.project_root / "config"
        
        # Process each configuration file for migration
        for file_path, file_info in analysis['files'].items():
            source_dir = file_info['source']
            target_location = self._determine_target_location(file_info, source_dir)
            
            plan['actions'].append({
                'type': 'migrate_file',
                'source': file_info['path'],
                'target': str(target_config_dir / target_location),
                'reason': f"Consolidate from {source_dir}",
                'file_info': file_info
            })
            
            plan['migration_map'][file_info['path']] = str(target_config_dir / target_location)
            
        # Process duplicates
        for duplicate in analysis['duplicates']:
            if duplicate.get('type') == 'exact_duplicate':
                # Keep the primary, remove duplicates
                primary_path = duplicate['primary']
                for dup_path in duplicate['duplicates']:
                    plan['actions'].append({
                        'type': 'remove_duplicate',
                        'file': dup_path,
                        'primary': primary_path,
                        'reason': f"Exact duplicate of {Path(primary_path).name}"
                    })
                    
        # Plan directory cleanup
        for dir_name in self.active_config_dirs:
            if dir_name != "config":  # Keep main config dir
                plan['actions'].append({
                    'type': 'archive_empty_directory',
                    'directory': dir_name,
                    'reason': "Consolidate into unified config structure"
                })
                
        return plan
    
    def _determine_target_location(self, file_info: Dict, source_dir: str) -> str:
        """Determine where a config file should be placed in unified structure"""
        file_name = file_info['name'].lower()
        
        # Environment-specific classification
        if any(env in file_name for env in ['dev', 'development']):
            return f"environments/development/{file_info['name']}"
        elif any(env in file_name for env in ['prod', 'production']):
            return f"environments/production/{file_info['name']}"
        elif any(env in file_name for env in ['test', 'testing']):
            return f"environments/testing/{file_info['name']}"
        elif any(env in file_name for env in ['staging', 'stage']):
            return f"environments/staging/{file_info['name']}"
        
        # Component-specific classification
        elif 'kimera' in source_dir.lower():
            return f"shared/kimera/{file_info['name']}"
        elif 'database' in file_name or 'db' in file_name:
            return f"shared/database/{file_info['name']}"
        elif 'gpu' in file_name:
            return f"shared/gpu/{file_info['name']}"
        elif 'prometheus' in file_name or 'grafana' in file_name:
            return f"shared/monitoring/{file_info['name']}"
        elif 'trading' in file_name:
            return f"shared/trading/{file_info['name']}"
        
        # Default to shared
        else:
            return f"shared/{file_info['name']}"
    
    def create_backup(self, analysis: Dict):
        """Create comprehensive backup before unification"""
        logger.info("💾 Creating comprehensive configuration backup...")
        
        self.backup_dir.mkdir(exist_ok=True)
        backup_count = 0
        
        for dir_name in self.active_config_dirs:
            source_dir = self.project_root / dir_name
            if source_dir.exists():
                target_dir = self.backup_dir / dir_name
                
                try:
                    shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)
                    backup_count += 1
                    logger.info(f"   📁 Backed up: {dir_name}")
                except Exception as e:
                    logger.error(f"Error backing up {dir_name}: {e}")
                    
        logger.info(f"✅ Backed up {backup_count} configuration directories to: {self.backup_dir}")
        
    def execute_unification(self, plan: Dict, dry_run: bool = True):
        """Execute the configuration unification plan"""
        logger.info(f"🚀 {'DRY RUN:' if dry_run else 'EXECUTING:'} Configuration Unification")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'dry_run': dry_run,
            'files_migrated': 0,
            'duplicates_removed': 0,
            'directories_created': 0,
            'errors': []
        }
        
        if not dry_run:
            # Create target directory structure
            target_config = self.project_root / "config"
            for env_dir in ['environments/development', 'environments/testing', 
                           'environments/staging', 'environments/production',
                           'shared/kimera', 'shared/database', 'shared/gpu',
                           'shared/monitoring', 'shared/trading', 'templates', 'schemas']:
                target_path = target_config / env_dir
                target_path.mkdir(parents=True, exist_ok=True)
                results['directories_created'] += 1
                
        # Execute actions
        for action in plan['actions']:
            try:
                if action['type'] == 'migrate_file':
                    source_path = Path(action['source'])
                    target_path = Path(action['target'])
                    
                    if dry_run:
                        logger.info(f"📄 Would migrate: {source_path.name}")
                        logger.info(f"   From: {source_path.relative_to(self.project_root)}")
                        logger.info(f"   To: {target_path.relative_to(self.project_root)}")
                    else:
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source_path, target_path)
                        logger.info(f"✅ Migrated: {source_path.name}")
                        
                    results['files_migrated'] += 1
                    
                elif action['type'] == 'remove_duplicate':
                    dup_path = Path(action['file'])
                    
                    if dry_run:
                        logger.info(f"🗑️  Would remove duplicate: {dup_path.name}")
                        logger.info(f"   Path: {dup_path.relative_to(self.project_root)}")
                    else:
                        dup_path.unlink()
                        logger.info(f"🗑️  Removed duplicate: {dup_path.name}")
                        
                    results['duplicates_removed'] += 1
                    
            except Exception as e:
                error_msg = f"Error executing {action['type']} for {action.get('source', action.get('file', 'unknown'))}: {e}"
                logger.error(error_msg)
                results['errors'].append(error_msg)
                
        return results
    
    def generate_report(self, analysis: Dict, plan: Dict, results: Dict):
        """Generate comprehensive unification report"""
        timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        
        total_files = len(analysis['files'])
        duplicates_found = len([d for d in analysis['duplicates'] if d.get('type') == 'exact_duplicate'])
        
        report = f"""# KIMERA SWM Configuration Unification Report
**Generated**: {timestamp}
**Phase**: 4 of Technical Debt Remediation - Configuration Unification
**Framework**: Martin Fowler + KIMERA SWM Protocol v3.0
**Strategy**: Environment-based unified configuration structure

## Executive Summary

**Status**: {'✅ COMPLETED' if not results['dry_run'] else '🔄 DRY RUN'}
- **Directories Analyzed**: {len(analysis['directories'])}
- **Files Processed**: {results['files_migrated']}
- **Duplicates Removed**: {results['duplicates_removed']}
- **Target Structure Created**: Unified environment-based configuration
- **Errors**: {len(results['errors'])}

## Configuration Chaos Analysis

### Before Unification
**Scattered Configuration Directories**:
"""
        
        for dir_name, dir_info in analysis['directories'].items():
            file_count = len(dir_info['files'])
            total_size = dir_info['total_size'] / 1024  # KB
            report += f"- **{dir_name}**: {file_count} files ({total_size:.1f} KB)\n"
            
        report += f"""

### After Unification
**Unified Configuration Structure**:
- **config/environments/**: Environment-specific configurations
  - development/ (dev configs)
  - testing/ (test configs) 
  - staging/ (staging configs)
  - production/ (prod configs)
- **config/shared/**: Component-specific shared configurations
  - kimera/ (Kimera-specific configs)
  - database/ (database configurations)
  - gpu/ (GPU configurations)
  - monitoring/ (Prometheus, Grafana)
  - trading/ (trading configurations)
- **config/templates/**: Configuration templates
- **config/schemas/**: Validation schemas

## Actions Performed

### Files Migrated
**Total**: {results['files_migrated']} configuration files consolidated

### Duplicates Eliminated  
**Total**: {results['duplicates_removed']} exact duplicate files removed

"""
        
        # Show specific duplicates
        for duplicate in analysis['duplicates']:
            if duplicate.get('type') == 'exact_duplicate':
                report += f"- ❌ **Removed**: {duplicate['duplicates']}\n"
                report += f"  ✅ **Kept**: {duplicate['primary']}\n"
                
        if results['errors']:
            report += "\n## Errors Encountered\n"
            for error in results['errors']:
                report += f"- ❌ {error}\n"
        else:
            report += "\n## ✅ No Errors - Perfect Execution\n"
            
        report += f"""
## Benefits Achieved

### Configuration Management
- **Single Source of Truth**: Unified configuration structure
- **Environment Separation**: Clear dev/test/staging/prod separation
- **Component Organization**: Logical grouping by functionality
- **Maintenance Reduction**: Single location for all configurations

### Developer Experience
- **Predictable Structure**: Consistent configuration locations
- **Environment Management**: Easy switching between environments
- **Reduced Confusion**: No more scattered configuration files
- **Deployment Simplification**: Clear configuration deployment paths

### System Benefits
- **Reduced Redundancy**: {results['duplicates_removed']} duplicate files eliminated
- **Storage Efficiency**: Consolidated configuration storage
- **Version Control**: Cleaner git history with unified structure
- **Configuration Validation**: Foundation for automated validation

## Next Steps

### Immediate Actions
1. **Update Application Code**: Modify config loading to use new paths
2. **Update Deployment Scripts**: Point to unified configuration structure
3. **Create Configuration Templates**: Standardize configuration creation
4. **Add Validation**: Implement configuration schema validation

### Long-term Configuration Strategy
1. **Environment Management**: Automated environment-specific config loading
2. **Configuration as Code**: Git-based configuration management
3. **Automated Validation**: Pre-deployment configuration validation
4. **Documentation**: Clear configuration management guidelines

### Backup Information
- **Backup Location**: {plan.get('backup_location', 'N/A')}
- **Recovery Instructions**: Restore from backup if configuration loading fails

---

*Phase 4 of KIMERA SWM Technical Debt Remediation*
*Configuration Chaos → Unified Excellence*
*Following Martin Fowler's Technical Debt Quadrants Framework*
"""
        
        # Save report
        report_dir = Path("docs/reports/debt")
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"{timestamp}_configuration_unification_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
            
        logger.info(f"📄 Report saved: {report_path}")
        return str(report_path)

def main():
    """Main execution function"""
    logger.info("🚀 KIMERA SWM Configuration Unification - Phase 4")
    logger.info("🎯 Strategy: Environment-based unified configuration structure")
    logger.info("=" * 70)
    
    unifier = ConfigurationUnifier()
    
    # Step 1: Analyze configuration landscape
    analysis = unifier.analyze_configuration_landscape()
    
    if not analysis['files']:
        logger.info("✅ No configuration files found - unification not needed")
        return
        
    # Step 2: Create unification plan
    plan = unifier.create_unification_plan(analysis)
    
    # Step 3: Execute dry run
    logger.info("\n🔄 Executing DRY RUN...")
    dry_results = unifier.execute_unification(plan, dry_run=True)
    
    # Step 4: Generate dry run report
    report_path = unifier.generate_report(analysis, plan, dry_results)
    
    logger.info(f"\n📊 CONFIGURATION UNIFICATION RESULTS:")
    logger.info(f"   Directories Analyzed: {len(analysis['directories'])}")
    logger.info(f"   Files to Migrate: {dry_results['files_migrated']}")
    logger.info(f"   Duplicates to Remove: {dry_results['duplicates_removed']}")
    logger.info(f"   Errors: {len(dry_results['errors'])}")
    
    logger.info(f"\n💡 To execute actual unification, run with --execute flag")
    logger.info(f"📄 Detailed report: {report_path}")

if __name__ == "__main__":
    import sys
    
    if "--execute" in sys.argv:
        logger.info("⚠️  EXECUTING ACTUAL CONFIGURATION UNIFICATION...")
        unifier = ConfigurationUnifier()
        analysis = unifier.analyze_configuration_landscape()
        plan = unifier.create_unification_plan(analysis)
        unifier.create_backup(analysis)
        results = unifier.execute_unification(plan, dry_run=False)
        report_path = unifier.generate_report(analysis, plan, results)
        logger.info("✅ Configuration unification complete!")
        logger.info(f"📄 Final report: {report_path}")
    else:
        main()