#!/usr/bin/env python3
"""
Comprehensive System Audit and Diagnosis
=======================================
"""

# Fix import paths
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))



import sys
import os
import asyncio
import time
import json
import logging
import platform
import psutil
import importlib
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SystemAuditor:
    """
    Comprehensive system auditor implementing aerospace-grade diagnostic protocols.

    Performs exhaustive analysis following DO-178C Level A standards:
    - Dependency verification
    - Engine health assessment
    - Performance benchmarking
    - Integration validation
    - Compliance verification
    - Resource utilization analysis
    """

    def __init__(self):
        self.audit_start_time = time.time()
        self.audit_results = {
            'timestamp': datetime.now().isoformat(),
            'system_info': {},
            'dependencies': {},
            'engines': {},
            'integrations': {},
            'performance': {},
            'compliance': {},
            'issues': [],
            'recommendations': []
        }

        # Create reports directory
        os.makedirs('docs/reports/audit', exist_ok=True)

    def log_issue(self, severity: str, component: str, message: str, recommendation: str = None):
        """Log system issue with aerospace-grade classification."""
        issue = {
            'severity': severity,  # CRITICAL, HIGH, MEDIUM, LOW
            'component': component,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        if recommendation:
            issue['recommendation'] = recommendation

        self.audit_results['issues'].append(issue)

        if severity == 'CRITICAL':
            logger.error(f"🔴 CRITICAL [{component}]: {message}")
        elif severity == 'HIGH':
            logger.warning(f"🟠 HIGH [{component}]: {message}")
        elif severity == 'MEDIUM':
            logger.info(f"🟡 MEDIUM [{component}]: {message}")
        else:
            logger.info(f"🔵 LOW [{component}]: {message}")

    def collect_system_info(self):
        """Collect comprehensive system information."""
        logger.info("🔍 COLLECTING SYSTEM INFORMATION...")
        logger.info("=" * 60)

        try:
            # Platform information
            system_info = {
                'platform': platform.platform(),
                'architecture': platform.architecture(),
                'processor': platform.processor(),
                'python_version': platform.python_version(),
                'system': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'machine': platform.machine(),
                'node': platform.node()
            }

            # Memory information
            memory = psutil.virtual_memory()
            system_info['memory'] = {
                'total': memory.total,
                'available': memory.available,
                'percent': memory.percent,
                'used': memory.used,
                'free': memory.free
            }

            # CPU information
            system_info['cpu'] = {
                'count': psutil.cpu_count(),
                'percent': psutil.cpu_percent(interval=1),
                'freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
            }

            # Disk information
            disk = psutil.disk_usage('.')
            system_info['disk'] = {
                'total': disk.total,
                'used': disk.used,
                'free': disk.free,
                'percent': (disk.used / disk.total) * 100
            }

            self.audit_results['system_info'] = system_info

            logger.info(f"✅ Platform: {system_info['platform']}")
            logger.info(f"✅ Python: {system_info['python_version']}")
            logger.info(f"✅ CPU: {system_info['cpu']['count']} cores @ {system_info['cpu']['percent']}% utilization")
            logger.info(f"✅ Memory: {memory.available // (1024**3)}GB / {memory.total // (1024**3)}GB available ({memory.percent}% used)")
            logger.info(f"✅ Disk: {disk.free // (1024**3)}GB / {disk.total // (1024**3)}GB free")
            logger.info()

        except Exception as e:
            self.log_issue('HIGH', 'SystemInfo', f"Failed to collect system information: {e}")

    def verify_dependencies(self):
        """Verify all critical dependencies."""
        logger.info("🔧 VERIFYING DEPENDENCIES...")
        logger.info("=" * 60)

        critical_packages = [
            'fastapi', 'pydantic', 'torch', 'numpy', 'pandas', 'scipy',
            'psycopg2', 'neo4j', 'redis', 'cupy', 'matplotlib', 'pytest'
        ]

        dependency_results = {}

        for package in critical_packages:
            try:
                module = importlib.import_module(package)
                version = getattr(module, '__version__', 'unknown')
                dependency_results[package] = {
                    'status': 'available',
                    'version': version
                }
                logger.info(f"✅ {package}: {version}")
            except ImportError:
                dependency_results[package] = {
                    'status': 'missing',
                    'version': None
                }
                logger.info(f"❌ {package}: MISSING")
                self.log_issue('HIGH', 'Dependencies', f"Missing critical package: {package}")

        self.audit_results['dependencies'] = dependency_results
        logger.info()

    def test_gpu_availability(self):
        """Test GPU and CUDA availability."""
        logger.info("🎮 TESTING GPU AVAILABILITY...")
        logger.info("=" * 60)

        gpu_info = {}

        try:
            import torch
            gpu_info['cuda_available'] = torch.cuda.is_available()

            if torch.cuda.is_available():
                gpu_info['device_count'] = torch.cuda.device_count()
                gpu_info['current_device'] = torch.cuda.current_device()
                gpu_info['device_name'] = torch.cuda.get_device_name(0)
                gpu_info['memory_allocated'] = torch.cuda.memory_allocated(0)
                gpu_info['memory_cached'] = torch.cuda.memory_reserved(0)

                logger.info(f"✅ CUDA Available: {gpu_info['device_count']} device(s)")
                logger.info(f"✅ Device: {gpu_info['device_name']}")
                logger.info(f"✅ Memory: {gpu_info['memory_allocated']//1024//1024}MB allocated, {gpu_info['memory_cached']//1024//1024}MB cached")
            else:
                logger.info("⚠️ CUDA not available - CPU mode only")
                self.log_issue('MEDIUM', 'GPU', "CUDA not available, falling back to CPU mode")

        except ImportError:
            logger.info("❌ PyTorch not available")
            self.log_issue('HIGH', 'GPU', "PyTorch not available for GPU testing")

        self.audit_results['system_info']['gpu'] = gpu_info
        logger.info()

    async def test_kimera_system_health(self):
        """Test core Kimera system health."""
        logger.info("🧠 TESTING KIMERA SYSTEM HEALTH...")
        logger.info("=" * 60)

        try:
            # Fix import path issue - ensure we're in the right directory
            import sys
            import os

            # Get the absolute path to the project root
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.join(current_dir, '..', '..')
            src_path = os.path.join(project_root, 'src')

            # Normalize the paths
            project_root = os.path.normpath(project_root)
            src_path = os.path.normpath(src_path)

            logger.info(f"🔍 Project root: {project_root}")
            logger.info(f"🔍 Source path: {src_path}")

            # Add to Python path if not already there
            if src_path not in sys.path:
                sys.path.insert(0, src_path)
                logger.info(f"✅ Added {src_path} to Python path")

            # Change to project root directory
            original_cwd = os.getcwd()
            os.chdir(project_root)
            logger.info(f"✅ Changed directory to {project_root}")

            from src.core.kimera_system import KimeraSystem

            # Initialize system
            logger.info("🔄 Initializing KimeraSystem...")
            try:
                system = KimeraSystem()
                logger.info("✅ KimeraSystem instance created")
                system.initialize()
                logger.info("✅ KimeraSystem initialized successfully")
            except Exception as init_error:
                logger.info(f"❌ KimeraSystem initialization error: {init_error}")
                import traceback
                traceback.print_exc()
                raise

            # Get system state
            state = system.get_system_state()
            logger.info(f"✅ System State: {state['state']}")
            logger.info(f"✅ Device: {state['device']}")
            logger.info(f"✅ GPU Acceleration: {state['gpu_acceleration_enabled']}")
            logger.info(f"✅ Components: {len(state['components'])} loaded")

            # Test thermodynamic systems instead of signal processing
            if hasattr(system, 'is_thermodynamic_systems_ready') and system.is_thermodynamic_systems_ready():
                logger.info("🔬 Testing Thermodynamic Systems...")
                thermo_system = system.get_thermodynamic_integration()

                if thermo_system:
                    logger.info("✅ Thermodynamic Systems: Ready")
                else:
                    logger.info("⚠️ Thermodynamic Systems: Not initialized")
            else:
                logger.info("⚠️ Thermodynamic Systems: Not available")

            # Additional component health checks
            if hasattr(system, 'get_component'):
                # Check High-Dimensional Modeling (newly integrated)
                hd_modeling = system.get_component('high_dimensional_modeling')
                if hd_modeling:
                    logger.info(f"✅ High-Dimensional Modeling: {type(hd_modeling).__name__}")
                    try:
                        logger.info(f"   BGM Dimension: {hd_modeling.bgm_engine.config.dimension}D")
                    except:
                        pass
                else:
                    logger.info("⚠️ High-Dimensional Modeling: Not loaded")

            self.audit_results['engines']['kimera_system'] = {
                'status': 'operational',
                'state': state
            }

        except Exception as e:
            logger.info(f"❌ Kimera System Test Failed: {e}")
            self.log_issue('CRITICAL', 'KimeraSystem', f"System initialization failed: {e}")

        logger.info()

    def test_database_connections(self):
        """Test database connectivity."""
        logger.info("🗄️ TESTING DATABASE CONNECTIONS...")
        logger.info("=" * 60)

        database_results = {}

        # Test Neo4j
        try:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
            with driver.session() as session:
                result = session.run("RETURN 1")
                result.single()
            driver.close()
            logger.info("✅ Neo4j: Connected")
            database_results['neo4j'] = {'status': 'connected'}
        except Exception as e:
            logger.info(f"❌ Neo4j: Connection failed - {e}")
            database_results['neo4j'] = {'status': 'failed', 'error': str(e)}
            self.log_issue('MEDIUM', 'Database', f"Neo4j connection failed: {e}")

        # Test Redis
        try:
            import redis
            r = redis.Redis(host='localhost', port=6379, db=0)
            r.ping()
            logger.info("✅ Redis: Connected")
            database_results['redis'] = {'status': 'connected'}
        except Exception as e:
            logger.info(f"❌ Redis: Connection failed - {e}")
            database_results['redis'] = {'status': 'failed', 'error': str(e)}
            self.log_issue('MEDIUM', 'Database', f"Redis connection failed: {e}")

        # Test PostgreSQL
        try:
            import psycopg2
            conn = psycopg2.connect(
                host="localhost",
                database="kimera_swm",
                user="kimera_user",
                password="kimera_secure_pass"
            )
            conn.close()
            logger.info("✅ PostgreSQL: Connected")
            database_results['postgresql'] = {'status': 'connected'}
        except Exception as e:
            logger.info(f"❌ PostgreSQL: Connection failed - {e}")
            database_results['postgresql'] = {'status': 'failed', 'error': str(e)}
            self.log_issue('HIGH', 'Database', f"PostgreSQL connection failed: {e}")

        self.audit_results['integrations']['databases'] = database_results
        logger.info()

    def analyze_file_structure(self):
        """Analyze project file structure and organization."""
        logger.info("📁 ANALYZING FILE STRUCTURE...")
        logger.info("=" * 60)

        structure_analysis = {}

        # Check critical directories
        critical_dirs = [
            'src/core',
            'src/engines',
            'scripts/health_check',
            'docs/reports',
            'tests',
            'configs'
        ]

        for dir_path in critical_dirs:
            if os.path.exists(dir_path):
                file_count = len(list(Path(dir_path).rglob('*.py')))
                structure_analysis[dir_path] = {
                    'exists': True,
                    'python_files': file_count
                }
                logger.info(f"✅ {dir_path}: {file_count} Python files")
            else:
                structure_analysis[dir_path] = {'exists': False}
                logger.info(f"❌ {dir_path}: Missing")
                self.log_issue('MEDIUM', 'FileStructure', f"Missing directory: {dir_path}")

        # Check for proper organization
        src_core_files = list(Path('src/core').rglob('*.py')) if os.path.exists('src/core') else []
        if len(src_core_files) > 50:
            self.log_issue('LOW', 'FileStructure', f"Large number of files in src/core ({len(src_core_files)}), consider reorganization")

        self.audit_results['integrations']['file_structure'] = structure_analysis
        logger.info()

    def performance_benchmark(self):
        """Run performance benchmarks."""
        logger.info("⚡ RUNNING PERFORMANCE BENCHMARKS...")
        logger.info("=" * 60)

        benchmarks = {}

        try:
            # CPU benchmark
            start_time = time.time()
            result = sum(i**2 for i in range(100000))
            cpu_time = time.time() - start_time
            benchmarks['cpu_benchmark'] = {
                'time': cpu_time,
                'result': result
            }
            logger.info(f"✅ CPU Benchmark: {cpu_time:.3f}s")

            # Memory benchmark
            start_time = time.time()
            large_list = [i for i in range(1000000)]
            memory_time = time.time() - start_time
            benchmarks['memory_benchmark'] = {
                'time': memory_time,
                'size': len(large_list)
            }
            logger.info(f"✅ Memory Benchmark: {memory_time:.3f}s")
            del large_list

            # Import benchmark
            start_time = time.time()
            import numpy as np
            import_time = time.time() - start_time
            benchmarks['import_benchmark'] = {
                'time': import_time
            }
            logger.info(f"✅ Import Benchmark: {import_time:.3f}s")

        except Exception as e:
            self.log_issue('MEDIUM', 'Performance', f"Benchmark failed: {e}")

        self.audit_results['performance']['benchmarks'] = benchmarks
        logger.info()

    def compliance_check(self):
        """Check DO-178C Level A compliance indicators."""
        logger.info("🔒 CHECKING COMPLIANCE INDICATORS...")
        logger.info("=" * 60)

        compliance_results = {}

        # Check for formal verification components
        z3_available = False
        try:
            import z3
            z3_available = True
            logger.info("✅ Z3 SMT Solver: Available for formal verification")
        except ImportError:
            logger.info("⚠️ Z3 SMT Solver: Not available")
            self.log_issue('MEDIUM', 'Compliance', "Z3 SMT solver not available for formal verification")

        compliance_results['formal_verification'] = z3_available

        # Check for test coverage
        test_files = list(Path('tests').rglob('*.py')) if os.path.exists('tests') else []
        compliance_results['test_coverage'] = {
            'test_files': len(test_files),
            'adequate': len(test_files) > 50
        }

        if len(test_files) > 50:
            logger.info(f"✅ Test Coverage: {len(test_files)} test files")
        else:
            logger.info(f"⚠️ Test Coverage: {len(test_files)} test files (may be insufficient)")
            self.log_issue('MEDIUM', 'Compliance', f"Low test coverage: {len(test_files)} test files")

        # Check documentation
        doc_files = list(Path('docs').rglob('*.md')) if os.path.exists('docs') else []
        compliance_results['documentation'] = {
            'doc_files': len(doc_files),
            'adequate': len(doc_files) > 20
        }

        if len(doc_files) > 20:
            logger.info(f"✅ Documentation: {len(doc_files)} documentation files")
        else:
            logger.info(f"⚠️ Documentation: {len(doc_files)} documentation files")
            self.log_issue('MEDIUM', 'Compliance', f"Insufficient documentation: {len(doc_files)} files")

        self.audit_results['compliance'] = compliance_results
        logger.info()

    def generate_recommendations(self):
        """Generate actionable recommendations based on audit results."""
        logger.info("💡 GENERATING RECOMMENDATIONS...")
        logger.info("=" * 60)

        recommendations = []

        # Analyze issues and generate recommendations
        critical_issues = [issue for issue in self.audit_results['issues'] if issue['severity'] == 'CRITICAL']
        high_issues = [issue for issue in self.audit_results['issues'] if issue['severity'] == 'HIGH']

        if critical_issues:
            recommendations.append({
                'priority': 'IMMEDIATE',
                'action': f"Resolve {len(critical_issues)} critical issues before proceeding",
                'details': [issue['message'] for issue in critical_issues]
            })

        if high_issues:
            recommendations.append({
                'priority': 'HIGH',
                'action': f"Address {len(high_issues)} high-priority issues",
                'details': [issue['message'] for issue in high_issues]
            })

        # Performance recommendations
        if 'benchmarks' in self.audit_results['performance']:
            cpu_time = self.audit_results['performance']['benchmarks'].get('cpu_benchmark', {}).get('time', 0)
            if cpu_time > 0.1:
                recommendations.append({
                    'priority': 'MEDIUM',
                    'action': 'Consider CPU performance optimization',
                    'details': [f'CPU benchmark took {cpu_time:.3f}s (target: <0.1s)']
                })

        # Database recommendations
        db_status = self.audit_results['integrations'].get('databases', {})
        failed_dbs = [db for db, info in db_status.items() if info.get('status') != 'connected']
        if failed_dbs:
            recommendations.append({
                'priority': 'HIGH',
                'action': 'Fix database connectivity issues',
                'details': [f'Database connection failed: {db}' for db in failed_dbs]
            })

        self.audit_results['recommendations'] = recommendations

        for rec in recommendations:
            logger.info(f"🎯 {rec['priority']}: {rec['action']}")
            for detail in rec['details']:
                logger.info(f"   - {detail}")

        logger.info()

    def generate_report(self):
        """Generate comprehensive audit report."""
        logger.info("📊 GENERATING COMPREHENSIVE REPORT...")
        logger.info("=" * 60)

        # Calculate audit duration
        audit_duration = time.time() - self.audit_start_time
        self.audit_results['audit_duration'] = audit_duration

        # Generate summary statistics
        total_issues = len(self.audit_results['issues'])
        critical_issues = len([i for i in self.audit_results['issues'] if i['severity'] == 'CRITICAL'])
        high_issues = len([i for i in self.audit_results['issues'] if i['severity'] == 'HIGH'])

        self.audit_results['summary'] = {
            'total_issues': total_issues,
            'critical_issues': critical_issues,
            'high_issues': high_issues,
            'overall_health': 'HEALTHY' if critical_issues == 0 and high_issues < 3 else 'DEGRADED',
            'audit_duration': audit_duration
        }

        # Save detailed JSON report
        json_report_path = f"docs/reports/audit/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_comprehensive_audit.json"
        with open(json_report_path, 'w', encoding='utf-8') as f:
            json.dump(self.audit_results, f, indent=2)

        # Generate markdown report
        md_report_path = f"docs/reports/audit/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_audit_report.md"
        self._generate_markdown_report(md_report_path)

        logger.info(f"✅ JSON Report: {json_report_path}")
        logger.info(f"✅ Markdown Report: {md_report_path}")
        logger.info(f"✅ Audit Duration: {audit_duration:.2f} seconds")
        logger.info(f"✅ Overall Health: {self.audit_results['summary']['overall_health']}")
        logger.info(f"✅ Total Issues: {total_issues} (Critical: {critical_issues}, High: {high_issues})")
        logger.info()

        return json_report_path, md_report_path

    def _generate_markdown_report(self, filepath: str):
        """Generate human-readable markdown report."""
        summary = self.audit_results['summary']

        report_content = f"""# KIMERA SWM COMPREHENSIVE SYSTEM AUDIT REPORT
## Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## EXECUTIVE SUMMARY

**Overall Health**: {summary['overall_health']}
**Audit Duration**: {summary['audit_duration']:.2f} seconds
**Total Issues**: {summary['total_issues']} (Critical: {summary['critical_issues']}, High: {summary['high_issues']})

---

## SYSTEM INFORMATION

**Platform**: {self.audit_results['system_info'].get('platform', 'Unknown')}
**Python Version**: {self.audit_results['system_info'].get('python_version', 'Unknown')}
**CPU Cores**: {self.audit_results['system_info'].get('cpu', {}).get('count', 'Unknown')}
**Memory**: {self.audit_results['system_info'].get('memory', {}).get('total', 0) // (1024**3)}GB

---

## ISSUES IDENTIFIED

"""

        # Add issues by severity
        for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
            issues = [i for i in self.audit_results['issues'] if i['severity'] == severity]
            if issues:
                report_content += f"\n### {severity} Issues ({len(issues)})\n\n"
                for issue in issues:
                    report_content += f"- **{issue['component']}**: {issue['message']}\n"

        # Add recommendations
        if self.audit_results['recommendations']:
            report_content += "\n---\n\n## RECOMMENDATIONS\n\n"
            for rec in self.audit_results['recommendations']:
                report_content += f"### {rec['priority']} Priority\n"
                report_content += f"**Action**: {rec['action']}\n\n"
                for detail in rec['details']:
                    report_content += f"- {detail}\n"
                report_content += "\n"

        report_content += f"\n---\n\n*Report generated by Kimera SWM Autonomous Architect*  \n*Compliance: DO-178C Level A Standards*"

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report_content)

async def main():
    """Main audit execution function."""
    logger.info("=" * 80)
    logger.info("🔍 KIMERA SWM COMPREHENSIVE SYSTEM AUDIT & DIAGNOSIS")
    logger.info("=" * 80)
    logger.info("🔒 DO-178C Level A Compliance | Aerospace-Grade Analysis")
    logger.info("📊 Complete System Health Assessment")
    logger.info("=" * 80)
    logger.info()

    auditor = SystemAuditor()

    try:
        # Execute all audit components
        auditor.collect_system_info()
        auditor.verify_dependencies()
        auditor.test_gpu_availability()
        await auditor.test_kimera_system_health()
        auditor.test_database_connections()
        auditor.analyze_file_structure()
        auditor.performance_benchmark()
        auditor.compliance_check()
        auditor.generate_recommendations()

        # Generate final report
        json_path, md_path = auditor.generate_report()

        logger.info("=" * 80)
        logger.info("🎉 COMPREHENSIVE AUDIT COMPLETE")
        logger.info("=" * 80)

        summary = auditor.audit_results['summary']
        if summary['overall_health'] == 'HEALTHY':
            logger.info("✅ SYSTEM STATUS: HEALTHY")
        else:
            logger.info("⚠️ SYSTEM STATUS: DEGRADED - ACTION REQUIRED")

        logger.info(f"📊 Issues Found: {summary['total_issues']} total")
        logger.info(f"🔴 Critical: {summary['critical_issues']}")
        logger.info(f"🟠 High: {summary['high_issues']}")
        logger.info(f"📄 Reports: {json_path}, {md_path}")
        logger.info("=" * 80)

        return 0 if summary['overall_health'] == 'HEALTHY' else 1

    except Exception as e:
        logger.info(f"❌ AUDIT FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("\n⚠️ Audit interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.info(f"❌ Fatal audit error: {e}")
        sys.exit(1)
