#!/usr/bin/env python3
"""
KIMERA SWM - COMPREHENSIVE SYSTEM VERIFICATION
==============================================

Complete system verification script that checks:
- All Python dependencies and requirements
- Database systems and schemas
- Vault and storage systems
- GPU acceleration components  
- Core system architecture
- Configuration files
- File integrity and paths

This script ensures everything is properly set up before core system integration.
"""

import os
import sys
import subprocess
import logging
import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import importlib.util

# Add src to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
src_dir = project_root / "src"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_dir))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SystemVerifier:
    """Comprehensive system verification and validation"""
    
    def __init__(self):
        self.project_root = project_root
        self.src_dir = src_dir
        self.verification_results = {
            'dependencies': {},
            'databases': {},
            'vault_systems': {},
            'gpu_components': {},
            'core_architecture': {},
            'configuration': {},
            'file_integrity': {},
            'overall_status': 'unknown'
        }
        self.critical_failures = []
        self.warnings = []
    
    def verify_python_dependencies(self) -> Dict[str, Any]:
        """Verify all Python package dependencies"""
        logger.info("🔍 Verifying Python Dependencies...")
        
        results = {
            'base_packages': {},
            'gpu_packages': {},
            'data_packages': {},
            'web_packages': {},
            'database_packages': {},
            'missing_packages': [],
            'version_conflicts': []
        }
        
        # Critical packages categorized
        package_categories = {
            'base_packages': [
                'fastapi', 'uvicorn', 'pydantic', 'sqlalchemy', 'aiohttp',
                'httpx', 'python-dotenv', 'PyYAML', 'requests', 'tqdm'
            ],
            'gpu_packages': [
                'torch', 'torchvision', 'torchaudio', 'cupy', 'numba', 
                'pynvml', 'GPUtil'
            ],
            'data_packages': [
                'numpy', 'pandas', 'scipy', 'matplotlib', 'networkx',
                'pillow', 'joblib'
            ],
            'web_packages': [
                'starlette', 'anyio', 'h11', 'httptools', 'click'
            ],
            'database_packages': [
                'neo4j', 'psycopg2', 'alembic', 'aiosqlite'
            ]
        }
        
        for category, packages in package_categories.items():
            results[category] = {}
            for package in packages:
                try:
                    # Try to import the package
                    spec = importlib.util.find_spec(package)
                    if spec is not None:
                        module = importlib.import_module(package)
                        version = getattr(module, '__version__', 'unknown')
                        results[category][package] = {
                            'status': 'available',
                            'version': version,
                            'path': str(spec.origin) if spec.origin else 'built-in'
                        }
                        logger.debug(f"✅ {package} {version}")
                    else:
                        results[category][package] = {'status': 'missing'}
                        results['missing_packages'].append(package)
                        logger.warning(f"❌ {package}: Not found")
                        
                except ImportError as e:
                    results[category][package] = {
                        'status': 'import_error',
                        'error': str(e)
                    }
                    results['missing_packages'].append(package)
                    logger.warning(f"❌ {package}: Import error - {e}")
                    
                except Exception as e:
                    results[category][package] = {
                        'status': 'error', 
                        'error': str(e)
                    }
                    logger.error(f"❌ {package}: Unexpected error - {e}")
        
        # Summary
        total_packages = sum(len(cat) for cat in package_categories.values())
        available_packages = sum(
            len([p for p in cat.values() if p.get('status') == 'available'])
            for cat in results.values() if isinstance(cat, dict) and cat
        )
        
        results['summary'] = {
            'total_packages': total_packages,
            'available_packages': available_packages,
            'missing_packages_count': len(results['missing_packages']),
            'success_rate': (available_packages / total_packages) * 100 if total_packages > 0 else 0
        }
        
        logger.info(f"Dependencies: {available_packages}/{total_packages} available ({results['summary']['success_rate']:.1f}%)")
        
        if results['missing_packages']:
            self.warnings.append(f"Missing packages: {', '.join(results['missing_packages'])}")
        
        return results
    
    def verify_database_systems(self) -> Dict[str, Any]:
        """Verify database systems and schemas"""
        logger.info("🗄️ Verifying Database Systems...")
        
        results = {
            'sqlite': {'status': 'unknown'},
            'postgresql': {'status': 'unknown'},
            'neo4j': {'status': 'unknown'},
            'schemas': {},
            'data_files': {}
        }
        
        # Check SQLite
        try:
            import sqlite3
            results['sqlite'] = {
                'status': 'available',
                'version': sqlite3.sqlite_version,
                'module_version': sqlite3.version
            }
            logger.info(f"✅ SQLite {sqlite3.sqlite_version}")
            
            # Check existing database files
            db_dir = self.project_root / "data" / "database"
            if db_dir.exists():
                for db_file in db_dir.glob("*.db"):
                    try:
                        conn = sqlite3.connect(str(db_file))
                        cursor = conn.cursor()
                        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                        tables = [row[0] for row in cursor.fetchall()]
                        conn.close()
                        
                        results['data_files'][db_file.name] = {
                            'status': 'accessible',
                            'tables': tables,
                            'size_mb': db_file.stat().st_size / (1024 * 1024)
                        }
                        logger.info(f"✅ Database {db_file.name}: {len(tables)} tables")
                        
                    except Exception as e:
                        results['data_files'][db_file.name] = {
                            'status': 'error',
                            'error': str(e)
                        }
                        logger.warning(f"❌ Database {db_file.name}: {e}")
            
        except Exception as e:
            results['sqlite'] = {'status': 'error', 'error': str(e)}
            logger.error(f"❌ SQLite error: {e}")
        
        # Check PostgreSQL
        try:
            import psycopg2
            results['postgresql'] = {
                'status': 'driver_available',
                'version': psycopg2.__version__
            }
            logger.info(f"✅ PostgreSQL driver {psycopg2.__version__}")
            
            # Try to connect (will likely fail in development)
            try:
                import psycopg2
                conn = psycopg2.connect(
                    host="localhost",
                    database="kimera",
                    user="kimera", 
                    password="kimera_secure_pass"
                )
                conn.close()
                results['postgresql']['connection'] = 'success'
                logger.info("✅ PostgreSQL connection successful")
            except psycopg2.OperationalError:
                results['postgresql']['connection'] = 'failed_expected'
                logger.debug("⚠️ PostgreSQL connection failed (expected in development)")
            
        except ImportError:
            results['postgresql'] = {'status': 'driver_missing'}
            logger.warning("❌ PostgreSQL driver not available")
        
        # Check Neo4j
        try:
            import neo4j
            results['neo4j'] = {
                'status': 'driver_available',
                'version': neo4j.__version__
            }
            logger.info(f"✅ Neo4j driver {neo4j.__version__}")
            
        except ImportError:
            results['neo4j'] = {'status': 'driver_missing'}
            logger.warning("❌ Neo4j driver not available")
        
        # Check database schemas
        try:
            from src.vault.enhanced_database_schema import Base
            from sqlalchemy import create_engine
            
            # Create temporary in-memory database to test schema
            engine = create_engine('sqlite:///:memory:')
            Base.metadata.create_all(engine)
            
            table_names = list(Base.metadata.tables.keys())
            results['schemas']['enhanced_schema'] = {
                'status': 'valid',
                'tables': table_names
            }
            logger.info(f"✅ Database schema: {len(table_names)} tables defined")
            
        except Exception as e:
            results['schemas']['enhanced_schema'] = {
                'status': 'error',
                'error': str(e)
            }
            logger.error(f"❌ Database schema error: {e}")
        
        return results
    
    def verify_vault_systems(self) -> Dict[str, Any]:
        """Verify vault and storage systems"""
        logger.info("🔐 Verifying Vault Systems...")
        
        results = {
            'vault_manager': {'status': 'unknown'},
            'secure_files': {},
            'storage_paths': {},
            'vault_database': {'status': 'unknown'}
        }
        
        # Check vault manager
        try:
            from src.vault.vault_manager import VaultManager
            vault_manager = VaultManager()
            
            results['vault_manager'] = {
                'status': 'available',
                'db_initialized': vault_manager.db_initialized,
                'neo4j_available': vault_manager.neo4j_available
            }
            logger.info(f"✅ VaultManager: DB={vault_manager.db_initialized}, Neo4j={vault_manager.neo4j_available}")
            
        except Exception as e:
            results['vault_manager'] = {
                'status': 'error',
                'error': str(e)
            }
            logger.error(f"❌ VaultManager error: {e}")
        
        # Check secure vault files
        data_dir = self.project_root / "data"
        if data_dir.exists():
            for secure_file in data_dir.glob("*.secure"):
                try:
                    file_size = secure_file.stat().st_size
                    results['secure_files'][secure_file.name] = {
                        'status': 'exists',
                        'size_bytes': file_size,
                        'modified': datetime.fromtimestamp(secure_file.stat().st_mtime).isoformat()
                    }
                    logger.info(f"✅ Vault file {secure_file.name}: {file_size} bytes")
                    
                except Exception as e:
                    results['secure_files'][secure_file.name] = {
                        'status': 'error',
                        'error': str(e)
                    }
        
        # Check storage paths
        storage_paths = [
            'data/logs',
            'data/sessions', 
            'data/exports',
            'data/reports',
            'data/analysis',
            'vault_data',
            'cache'
        ]
        
        for path_str in storage_paths:
            path = self.project_root / path_str
            results['storage_paths'][path_str] = {
                'exists': path.exists(),
                'is_dir': path.is_dir() if path.exists() else False,
                'writable': os.access(path.parent, os.W_OK) if not path.exists() else os.access(path, os.W_OK)
            }
            
            status = "✅" if path.exists() else "⚠️"
            logger.info(f"{status} Storage path {path_str}: {'exists' if path.exists() else 'missing'}")
        
        # Check vault database
        vault_db_path = self.project_root / "vault_data" / "kimera_vault.db"
        if vault_db_path.exists():
            try:
                conn = sqlite3.connect(str(vault_db_path))
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = [row[0] for row in cursor.fetchall()]
                conn.close()
                
                results['vault_database'] = {
                    'status': 'accessible',
                    'tables': tables,
                    'size_mb': vault_db_path.stat().st_size / (1024 * 1024)
                }
                logger.info(f"✅ Vault database: {len(tables)} tables")
                
            except Exception as e:
                results['vault_database'] = {
                    'status': 'error',
                    'error': str(e)
                }
                logger.warning(f"❌ Vault database error: {e}")
        else:
            results['vault_database'] = {'status': 'missing'}
            logger.warning("⚠️ Vault database not found")
        
        return results
    
    def verify_gpu_components(self) -> Dict[str, Any]:
        """Verify GPU acceleration components"""
        logger.info("⚡ Verifying GPU Components...")
        
        results = {
            'hardware': {'status': 'unknown'},
            'software': {},
            'kimera_gpu': {},
            'performance': {}
        }
        
        # Hardware check
        try:
            import torch
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
                
                results['hardware'] = {
                    'status': 'available',
                    'device_count': device_count,
                    'current_device': device_name,
                    'cuda_version': torch.version.cuda,
                    'pytorch_version': torch.__version__
                }
                logger.info(f"✅ GPU Hardware: {device_name}")
                
            else:
                results['hardware'] = {'status': 'cuda_unavailable'}
                logger.warning("⚠️ CUDA not available")
                
        except Exception as e:
            results['hardware'] = {
                'status': 'error',
                'error': str(e)
            }
            logger.error(f"❌ GPU hardware check failed: {e}")
        
        # Software components
        gpu_packages = ['torch', 'cupy', 'numba', 'pynvml']
        for package in gpu_packages:
            try:
                module = importlib.import_module(package)
                version = getattr(module, '__version__', 'unknown')
                results['software'][package] = {
                    'status': 'available',
                    'version': version
                }
                logger.info(f"✅ {package} {version}")
                
            except ImportError:
                results['software'][package] = {'status': 'missing'}
                logger.warning(f"❌ {package} not available")
        
        # Kimera GPU components
        gpu_components = [
            'src.core.gpu.gpu_manager',
            'src.core.gpu.gpu_integration', 
            'src.engines.gpu.gpu_geoid_processor',
            'src.engines.gpu.gpu_thermodynamic_engine'
        ]
        
        for component in gpu_components:
            try:
                module = importlib.import_module(component)
                results['kimera_gpu'][component] = {'status': 'importable'}
                logger.info(f"✅ {component}")
                
            except ImportError as e:
                results['kimera_gpu'][component] = {
                    'status': 'import_error',
                    'error': str(e)
                }
                logger.warning(f"❌ {component}: {e}")
        
        # Performance test
        if results['hardware'].get('status') == 'available':
            try:
                import torch
                import time
                
                # Simple matrix multiplication test
                size = 1000
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                
                a = torch.randn(size, size, device=device)
                b = torch.randn(size, size, device=device)
                
                start_time = time.time()
                c = torch.matmul(a, b)
                if device == 'cuda':
                    torch.cuda.synchronize()
                elapsed = time.time() - start_time
                
                gflops = (2 * size**3) / elapsed / 1e9
                
                results['performance'] = {
                    'status': 'tested',
                    'device': device,
                    'matrix_size': size,
                    'elapsed_seconds': elapsed,
                    'gflops': gflops
                }
                logger.info(f"✅ GPU Performance: {gflops:.0f} GFLOPS")
                
            except Exception as e:
                results['performance'] = {
                    'status': 'test_failed',
                    'error': str(e)
                }
                logger.warning(f"⚠️ GPU performance test failed: {e}")
        
        return results
    
    def verify_core_architecture(self) -> Dict[str, Any]:
        """Verify core architecture components"""
        logger.info("🏗️ Verifying Core Architecture...")
        
        results = {
            'kimera_system': {'status': 'unknown'},
            'orchestrator': {'status': 'unknown'},
            'engines': {},
            'api_routers': {},
            'data_structures': {}
        }
        
        # Core system
        try:
            from src.core.kimera_system import get_kimera_system
            system = get_kimera_system()
            
            results['kimera_system'] = {
                'status': 'importable',
                'singleton': True
            }
            logger.info("✅ KimeraSystem core")
            
        except Exception as e:
            results['kimera_system'] = {
                'status': 'error',
                'error': str(e)
            }
            logger.error(f"❌ KimeraSystem: {e}")
            self.critical_failures.append("KimeraSystem core unavailable")
        
        # Orchestrator  
        try:
            from src.orchestration.kimera_orchestrator import EngineCoordinator
            coordinator = EngineCoordinator()
            
            results['orchestrator'] = {
                'status': 'available',
                'total_engines': len(coordinator.engines),
                'gpu_available': coordinator.gpu_available
            }
            logger.info(f"✅ Orchestrator: {len(coordinator.engines)} engines")
            
        except Exception as e:
            results['orchestrator'] = {
                'status': 'error',
                'error': str(e)
            }
            logger.error(f"❌ Orchestrator: {e}")
        
        # Core engines
        core_engines = [
            'src.core.processing.geoid_processor',
            'src.engines.thermodynamic.thermodynamic_evolution_engine',
            'src.engines.transformation.mirror_portal_engine',
            'src.engines.field_dynamics.cognitive_field_engine'
        ]
        
        for engine in core_engines:
            try:
                module = importlib.import_module(engine)
                results['engines'][engine] = {'status': 'importable'}
                logger.info(f"✅ {engine}")
                
            except ImportError as e:
                results['engines'][engine] = {
                    'status': 'import_error',
                    'error': str(e)
                }
                logger.warning(f"❌ {engine}: {e}")
        
        # API routers
        api_routers = [
            'src.api.routers.gpu_router',
            'src.api.core_actions_routes',
            'src.api.routers.thermodynamic_router'
        ]
        
        for router in api_routers:
            try:
                module = importlib.import_module(router)
                results['api_routers'][router] = {'status': 'importable'}
                logger.info(f"✅ {router}")
                
            except ImportError as e:
                results['api_routers'][router] = {
                    'status': 'import_error',
                    'error': str(e)
                }
                logger.warning(f"❌ {router}: {e}")
        
        # Data structures
        data_structures = [
            'src.core.data_structures.geoid_state',
            'src.core.data_structures.scar_state',
            'src.core.processing.processing_result'
        ]
        
        for ds in data_structures:
            try:
                module = importlib.import_module(ds)
                results['data_structures'][ds] = {'status': 'importable'}
                logger.info(f"✅ {ds}")
                
            except ImportError as e:
                results['data_structures'][ds] = {
                    'status': 'import_error',
                    'error': str(e)
                }
                logger.warning(f"❌ {ds}: {e}")
        
        return results
    
    def verify_configuration(self) -> Dict[str, Any]:
        """Verify configuration files and settings"""
        logger.info("⚙️ Verifying Configuration...")
        
        results = {
            'config_files': {},
            'requirements_files': {},
            'environment': {}
        }
        
        # Configuration files
        config_files = [
            'config/development.yaml',
            'config/production.yaml',
            'config/config.json'
        ]
        
        for config_file in config_files:
            config_path = self.project_root / config_file
            if config_path.exists():
                try:
                    if config_path.suffix == '.yaml':
                        import yaml
                        with open(config_path, 'r') as f:
                            data = yaml.safe_load(f)
                            
                        results['config_files'][config_file] = {
                            'status': 'valid',
                            'keys': list(data.keys()) if isinstance(data, dict) else [],
                            'size_kb': config_path.stat().st_size / 1024
                        }
                        logger.info(f"✅ Config {config_file}")
                        
                    elif config_path.suffix == '.json':
                        with open(config_path, 'r') as f:
                            data = json.load(f)
                            
                        results['config_files'][config_file] = {
                            'status': 'valid',
                            'keys': list(data.keys()) if isinstance(data, dict) else [],
                            'size_kb': config_path.stat().st_size / 1024
                        }
                        logger.info(f"✅ Config {config_file}")
                        
                except Exception as e:
                    results['config_files'][config_file] = {
                        'status': 'parse_error',
                        'error': str(e)
                    }
                    logger.warning(f"❌ Config {config_file}: {e}")
            else:
                results['config_files'][config_file] = {'status': 'missing'}
        
        # Requirements files
        req_dir = self.project_root / "requirements"
        if req_dir.exists():
            for req_file in req_dir.glob("*.txt"):
                try:
                    with open(req_file, 'r') as f:
                        lines = f.readlines()
                        
                    packages = [line.strip() for line in lines 
                              if line.strip() and not line.strip().startswith('#')]
                    
                    results['requirements_files'][req_file.name] = {
                        'status': 'readable',
                        'package_count': len(packages),
                        'size_kb': req_file.stat().st_size / 1024
                    }
                    logger.info(f"✅ Requirements {req_file.name}: {len(packages)} packages")
                    
                except Exception as e:
                    results['requirements_files'][req_file.name] = {
                        'status': 'error',
                        'error': str(e)
                    }
        
        # Environment variables
        env_vars = [
            'PYTHONPATH', 'CUDA_PATH', 'CUDA_HOME', 'PATH'
        ]
        
        for var in env_vars:
            value = os.environ.get(var)
            results['environment'][var] = {
                'set': value is not None,
                'value_length': len(value) if value else 0
            }
            
            status = "✅" if value else "⚠️"
            logger.info(f"{status} Environment {var}: {'set' if value else 'not set'}")
        
        return results
    
    def verify_file_integrity(self) -> Dict[str, Any]:
        """Verify file integrity and critical paths"""
        logger.info("📁 Verifying File Integrity...")
        
        results = {
            'critical_files': {},
            'directory_structure': {},
            'file_permissions': {}
        }
        
        # Critical files
        critical_files = [
            'kimera.py',
            'src/main.py',
            'src/core/kimera_system.py',
            'src/orchestration/kimera_orchestrator.py',
            'src/core/gpu/gpu_manager.py',
            'src/vault/vault_manager.py'
        ]
        
        for file_path in critical_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                try:
                    size = full_path.stat().st_size
                    # Try to parse as Python to check syntax
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        compile(content, str(full_path), 'exec')
                    
                    results['critical_files'][file_path] = {
                        'status': 'valid',
                        'size_bytes': size,
                        'lines': len(content.splitlines())
                    }
                    logger.info(f"✅ File {file_path}")
                    
                except SyntaxError as e:
                    results['critical_files'][file_path] = {
                        'status': 'syntax_error',
                        'error': str(e)
                    }
                    logger.error(f"❌ File {file_path}: Syntax error - {e}")
                    self.critical_failures.append(f"Syntax error in {file_path}")
                    
                except Exception as e:
                    results['critical_files'][file_path] = {
                        'status': 'error',
                        'error': str(e)
                    }
                    logger.warning(f"⚠️ File {file_path}: {e}")
            else:
                results['critical_files'][file_path] = {'status': 'missing'}
                logger.error(f"❌ File {file_path}: Missing")
                self.critical_failures.append(f"Missing critical file: {file_path}")
        
        # Directory structure
        critical_dirs = [
            'src', 'src/core', 'src/engines', 'src/api', 'src/vault',
            'data', 'config', 'requirements', 'scripts'
        ]
        
        for dir_path in critical_dirs:
            full_path = self.project_root / dir_path
            results['directory_structure'][dir_path] = {
                'exists': full_path.exists(),
                'is_dir': full_path.is_dir() if full_path.exists() else False,
                'file_count': len(list(full_path.iterdir())) if full_path.exists() and full_path.is_dir() else 0
            }
            
            status = "✅" if full_path.exists() else "❌"
            logger.info(f"{status} Directory {dir_path}")
            
            if not full_path.exists():
                self.critical_failures.append(f"Missing directory: {dir_path}")
        
        return results
    
    def run_full_verification(self) -> Dict[str, Any]:
        """Run complete system verification"""
        logger.info("🚀 Starting Comprehensive System Verification")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        
        # Run all verification steps
        self.verification_results['dependencies'] = self.verify_python_dependencies()
        self.verification_results['databases'] = self.verify_database_systems()
        self.verification_results['vault_systems'] = self.verify_vault_systems()
        self.verification_results['gpu_components'] = self.verify_gpu_components()
        self.verification_results['core_architecture'] = self.verify_core_architecture()
        self.verification_results['configuration'] = self.verify_configuration()
        self.verification_results['file_integrity'] = self.verify_file_integrity()
        
        end_time = datetime.now()
        verification_duration = (end_time - start_time).total_seconds()
        
        # Calculate overall status
        critical_failures_count = len(self.critical_failures)
        warnings_count = len(self.warnings)
        
        if critical_failures_count == 0:
            if warnings_count == 0:
                overall_status = 'excellent'
            elif warnings_count <= 3:
                overall_status = 'good' 
            else:
                overall_status = 'acceptable'
        else:
            if critical_failures_count <= 2:
                overall_status = 'needs_attention'
            else:
                overall_status = 'critical_issues'
        
        self.verification_results['overall_status'] = overall_status
        self.verification_results['summary'] = {
            'verification_time': verification_duration,
            'timestamp': end_time.isoformat(),
            'critical_failures': critical_failures_count,
            'warnings': warnings_count,
            'status': overall_status
        }
        
        return self.verification_results
    
    def generate_report(self) -> str:
        """Generate comprehensive verification report"""
        report = []
        report.append("# KIMERA SWM - COMPREHENSIVE SYSTEM VERIFICATION REPORT")
        report.append("=" * 70)
        report.append(f"**Timestamp**: {datetime.now().isoformat()}")
        report.append(f"**Overall Status**: {self.verification_results['overall_status'].upper()}")
        report.append("")
        
        # Summary
        summary = self.verification_results.get('summary', {})
        report.append("## Summary")
        report.append(f"- Verification Time: {summary.get('verification_time', 0):.2f} seconds")
        report.append(f"- Critical Failures: {summary.get('critical_failures', 0)}")
        report.append(f"- Warnings: {summary.get('warnings', 0)}")
        report.append("")
        
        # Dependencies
        deps = self.verification_results.get('dependencies', {}).get('summary', {})
        if deps:
            report.append("## Dependencies")
            report.append(f"- Available Packages: {deps.get('available_packages', 0)}/{deps.get('total_packages', 0)}")
            report.append(f"- Success Rate: {deps.get('success_rate', 0):.1f}%")
            report.append("")
        
        # Databases
        db_status = self.verification_results.get('databases', {})
        report.append("## Database Systems")
        report.append(f"- SQLite: {db_status.get('sqlite', {}).get('status', 'unknown')}")
        report.append(f"- PostgreSQL: {db_status.get('postgresql', {}).get('status', 'unknown')}")
        report.append(f"- Neo4j: {db_status.get('neo4j', {}).get('status', 'unknown')}")
        report.append("")
        
        # GPU
        gpu_status = self.verification_results.get('gpu_components', {})
        report.append("## GPU Acceleration")
        report.append(f"- Hardware: {gpu_status.get('hardware', {}).get('status', 'unknown')}")
        if gpu_status.get('hardware', {}).get('status') == 'available':
            hw = gpu_status['hardware']
            report.append(f"- Device: {hw.get('current_device', 'Unknown')}")
            report.append(f"- CUDA Version: {hw.get('cuda_version', 'Unknown')}")
        report.append("")
        
        # Critical failures
        if self.critical_failures:
            report.append("## Critical Failures")
            for failure in self.critical_failures:
                report.append(f"- ❌ {failure}")
            report.append("")
        
        # Warnings  
        if self.warnings:
            report.append("## Warnings")
            for warning in self.warnings:
                report.append(f"- ⚠️ {warning}")
            report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        if self.verification_results['overall_status'] == 'excellent':
            report.append("✅ System is ready for production deployment")
        elif self.verification_results['overall_status'] == 'good':
            report.append("✅ System is ready with minor optimizations recommended")
        elif self.verification_results['overall_status'] == 'acceptable':
            report.append("⚠️ System functional but improvements recommended")
        else:
            report.append("❌ Critical issues must be resolved before deployment")
        
        report.append("")
        report.append("---")
        report.append("*Report generated by Kimera SWM Comprehensive System Verifier*")
        
        return "\n".join(report)


def main():
    """Main verification function"""
    try:
        verifier = SystemVerifier()
        results = verifier.run_full_verification()
        
        # Generate and save report
        report = verifier.generate_report()
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Ensure reports directory exists
        reports_dir = verifier.project_root / "docs" / "reports" / "verification"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON results
        json_path = reports_dir / f"{timestamp}_system_verification.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save markdown report
        md_path = reports_dir / f"{timestamp}_system_verification.md"
        with open(md_path, 'w') as f:
            f.write(report)
        
        # Print summary
        logger.info("\n" + "=" * 70)
        logger.info("KIMERA SWM - SYSTEM VERIFICATION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Overall Status: {results['overall_status'].upper()}")
        logger.info(f"Critical Failures: {len(verifier.critical_failures)}")
        logger.info(f"Warnings: {len(verifier.warnings)}")
        logger.info(f"Detailed Report: {md_path}")
        logger.info(f"Raw Data: {json_path}")
        
        if results['overall_status'] in ['excellent', 'good']:
            logger.info("\n🎉 SYSTEM READY FOR CORE INTEGRATION! 🎉")
            return 0
        else:
            logger.info("\n⚠️ Issues detected - review report before integration")
            return 1
            
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 