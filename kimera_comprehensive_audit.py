#!/usr/bin/env python3
"""
Kimera SWM Comprehensive System Audit
=====================================
This script performs a thorough audit of the Kimera system, looking for:
- Holes (missing components, functions, configurations)
- Misalignments (configuration vs code mismatches)
- Disconnections (broken imports, API endpoints)
- Mocks (placeholder implementations)
- Fluidity issues (performance, integration problems)
- Coherence problems (inconsistent states, data flow)
"""

import os
import sys
import json
import logging
import traceback
import importlib
import inspect
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import psutil
import ast

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'kimera_audit_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class KimeraAuditor:
    """Comprehensive system auditor for Kimera SWM"""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "holes": [],
            "misalignments": [],
            "disconnections": [],
            "mocks": [],
            "fluidity_issues": [],
            "coherence_issues": [],
            "critical_issues": [],
            "warnings": [],
            "successes": [],
            "statistics": {}
        }
        self.project_root = Path(__file__).parent.absolute()
        
    def log_issue(self, category: str, issue: str, severity: str = "warning", details: Dict = None):
        """Log an issue to the appropriate category"""
        entry = {
            "issue": issue,
            "severity": severity,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }
        
        if severity == "critical":
            self.results["critical_issues"].append(entry)
            logger.error(f"CRITICAL: {issue}")
        elif category in self.results:
            self.results[category].append(entry)
            logger.warning(f"{category.upper()}: {issue}")
            
    def log_success(self, message: str, details: Dict = None):
        """Log a successful check"""
        entry = {
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }
        self.results["successes"].append(entry)
        logger.info(f"✓ {message}")
        
    def check_environment_setup(self):
        """Check environment configuration and setup"""
        logger.info("\n=== Checking Environment Setup ===")
        
        # Check for .env files
        env_files = [".env", ".env.dev", ".env.prod"]
        env_found = False
        for env_file in env_files:
            if (self.project_root / env_file).exists():
                self.log_success(f"Environment file found: {env_file}")
                env_found = True
                
        if not env_found:
            self.log_issue("holes", "No environment files found (.env, .env.dev, .env.prod)", 
                          severity="critical")
            
        # Check critical environment variables
        critical_vars = ["DATABASE_URL", "KIMERA_DATABASE_URL", "OPENAI_API_KEY"]
        for var in critical_vars:
            if os.getenv(var):
                self.log_success(f"Environment variable set: {var}")
            else:
                self.log_issue("holes", f"Missing environment variable: {var}")
                
    def check_database_configuration(self):
        """Check database configuration alignment"""
        logger.info("\n=== Checking Database Configuration ===")
        
        try:
            # Check settings.py configuration
            from backend.config.settings import get_settings
            settings = get_settings()
            configured_db = settings.database.url
            self.log_success(f"Settings database URL: {configured_db}")
            
            # Check vault/database.py hardcoded value
            db_file = self.project_root / "backend" / "vault" / "database.py"
            if db_file.exists():
                content = db_file.read_text()
                match = re.search(r'DATABASE_URL = os\.getenv\("DATABASE_URL", "(.+?)"\)', content)
                if match:
                    hardcoded_db = match.group(1)
                    
                    # Check for misalignment
                    if "sqlite" in configured_db and "postgresql" in hardcoded_db:
                        self.log_issue("misalignments", 
                                     f"Database mismatch: settings.py uses SQLite ({configured_db}) "
                                     f"but database.py defaults to PostgreSQL ({hardcoded_db})",
                                     severity="critical")
                    else:
                        self.log_success("Database configuration aligned")
                        
        except Exception as e:
            self.log_issue("disconnections", f"Failed to check database configuration: {str(e)}")
            
    def check_imports_and_modules(self):
        """Check all critical imports and module availability"""
        logger.info("\n=== Checking Imports and Modules ===")
        
        critical_modules = [
            "backend.core.kimera_system",
            "backend.api.main",
            "backend.vault.vault_manager",
            "backend.engines.contradiction_engine",
            "backend.engines.geoid_scar_manager",
            "backend.monitoring.system_monitor",
            "backend.governance.ethical_governor",
            "backend.utils.config",
            "backend.security.enhanced_security_hardening"
        ]
        
        for module_name in critical_modules:
            try:
                module = importlib.import_module(module_name)
                self.log_success(f"Module imported: {module_name}")
                
                # Check for specific functions/classes
                if module_name == "backend.utils.config":
                    if hasattr(module, "get_config"):
                        self.log_success("get_config function found in backend.utils.config")
                    else:
                        self.log_issue("holes", "get_config function missing from backend.utils.config")
                        
            except ImportError as e:
                self.log_issue("disconnections", f"Failed to import {module_name}: {str(e)}", 
                             severity="critical")
            except Exception as e:
                self.log_issue("disconnections", f"Error checking {module_name}: {str(e)}")
                
    def check_api_endpoints(self):
        """Check API endpoint definitions and routers"""
        logger.info("\n=== Checking API Endpoints ===")
        
        router_files = [
            "backend/api/routers/geoid_scar_router.py",
            "backend/api/routers/contradiction_router.py",
            "backend/api/routers/vault_router.py",
            "backend/api/routers/system_router.py"
        ]
        
        expected_endpoints = {
            "geoid_scar_router": ["/scars", "/geoids"],
            "contradiction_router": ["/contradictions/detect", "/contradictions/resolve"],
            "vault_router": ["/vault/store", "/vault/retrieve"],
            "system_router": ["/status", "/health"]
        }
        
        for router_file in router_files:
            router_path = self.project_root / router_file
            if router_path.exists():
                content = router_path.read_text()
                router_name = router_path.stem
                
                # Check for expected endpoints
                if router_name in expected_endpoints:
                    for endpoint in expected_endpoints[router_name]:
                        if endpoint in content:
                            self.log_success(f"Endpoint found: {endpoint} in {router_name}")
                        else:
                            self.log_issue("holes", f"Missing endpoint: {endpoint} in {router_name}")
                            
                # Check for mock implementations
                if "# TODO" in content or "pass" in content or "NotImplementedError" in content:
                    self.log_issue("mocks", f"Possible mock implementation in {router_name}")
                    
            else:
                self.log_issue("holes", f"Router file not found: {router_file}")
                
    def check_mock_implementations(self):
        """Scan for mock implementations and placeholders"""
        logger.info("\n=== Checking for Mock Implementations ===")
        
        mock_patterns = [
            r"#\s*TODO",
            r"#\s*FIXME",
            r"#\s*HACK",
            r"#\s*MOCK",
            r"raise\s+NotImplementedError",
            r"pass\s*#.*mock",
            r"return\s+None\s*#.*placeholder",
            r"mock_\w+",
            r"fake_\w+",
            r"dummy_\w+"
        ]
        
        python_files = list(Path(self.project_root / "backend").rglob("*.py"))
        
        for py_file in python_files[:50]:  # Limit to first 50 files for performance
            try:
                content = py_file.read_text()
                for pattern in mock_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        self.log_issue("mocks", 
                                     f"Mock pattern found in {py_file.relative_to(self.project_root)}: {matches[0]}")
            except Exception as e:
                logger.debug(f"Error reading {py_file}: {e}")
                
    def check_system_coherence(self):
        """Check system coherence and integration"""
        logger.info("\n=== Checking System Coherence ===")
        
        try:
            # Try to initialize the core system
            from backend.core.kimera_system import KimeraSystem
            
            # Check if system can be instantiated
            system = KimeraSystem()
            self.log_success("KimeraSystem instantiated successfully")
            
            # Check component registration
            expected_components = [
                "geoid_scar_manager",
                "contradiction_engine",
                "vault_manager",
                "system_monitor",
                "ethical_governor"
            ]
            
            for component in expected_components:
                if hasattr(system, component) or component in getattr(system, '_components', {}):
                    self.log_success(f"Component registered: {component}")
                else:
                    self.log_issue("coherence_issues", f"Component not registered: {component}")
                    
        except Exception as e:
            self.log_issue("coherence_issues", f"Failed to check system coherence: {str(e)}", 
                         severity="critical")
            
    def check_performance_and_resources(self):
        """Check system performance and resource usage"""
        logger.info("\n=== Checking Performance and Resources ===")
        
        # Check system resources
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        self.results["statistics"]["resources"] = {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "disk_percent": disk.percent
        }
        
        # Check for performance issues
        if cpu_percent > 80:
            self.log_issue("fluidity_issues", f"High CPU usage: {cpu_percent}%")
        else:
            self.log_success(f"CPU usage normal: {cpu_percent}%")
            
        if memory.percent > 80:
            self.log_issue("fluidity_issues", f"High memory usage: {memory.percent}%")
        else:
            self.log_success(f"Memory usage normal: {memory.percent}%")
            
        if disk.percent > 90:
            self.log_issue("fluidity_issues", f"High disk usage: {disk.percent}%", severity="critical")
        else:
            self.log_success(f"Disk usage normal: {disk.percent}%")
            
    def check_configuration_files(self):
        """Check for required configuration files"""
        logger.info("\n=== Checking Configuration Files ===")
        
        required_files = [
            "requirements.txt",
            "README.md",
            ".gitignore",
            "backend/__init__.py",
            "backend/core/__init__.py",
            "backend/config/settings.py"
        ]
        
        for file_path in required_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                self.log_success(f"Configuration file exists: {file_path}")
            else:
                self.log_issue("holes", f"Missing configuration file: {file_path}")
                
    def check_logging_configuration(self):
        """Check logging setup and configuration"""
        logger.info("\n=== Checking Logging Configuration ===")
        
        # Check if logs directory exists
        logs_dir = self.project_root / "logs"
        if logs_dir.exists():
            self.log_success("Logs directory exists")
        else:
            self.log_issue("holes", "Logs directory missing")
            
        # Check for print statements (should use logging instead)
        python_files = list(Path(self.project_root / "backend").rglob("*.py"))
        
        print_count = 0
        for py_file in python_files[:20]:  # Check first 20 files
            try:
                content = py_file.read_text()
                # Look for print statements not in docstrings or comments
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'print':
                        print_count += 1
                        if print_count == 1:  # Only log first occurrence
                            self.log_issue("fluidity_issues", 
                                         f"Using print() instead of logging in {py_file.relative_to(self.project_root)}")
            except:
                pass
                
        if print_count == 0:
            self.log_success("No print statements found (good - using logging)")
            
    def generate_report(self):
        """Generate comprehensive audit report"""
        logger.info("\n=== Generating Audit Report ===")
        
        # Calculate statistics
        self.results["statistics"]["total_issues"] = sum([
            len(self.results["holes"]),
            len(self.results["misalignments"]),
            len(self.results["disconnections"]),
            len(self.results["mocks"]),
            len(self.results["fluidity_issues"]),
            len(self.results["coherence_issues"]),
            len(self.results["critical_issues"])
        ])
        
        self.results["statistics"]["total_successes"] = len(self.results["successes"])
        
        # Save report
        report_file = f"kimera_audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        logger.info(f"\nAudit report saved to: {report_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("KIMERA SYSTEM AUDIT SUMMARY")
        print("="*60)
        print(f"Total Issues Found: {self.results['statistics']['total_issues']}")
        print(f"  - Critical Issues: {len(self.results['critical_issues'])}")
        print(f"  - Holes: {len(self.results['holes'])}")
        print(f"  - Misalignments: {len(self.results['misalignments'])}")
        print(f"  - Disconnections: {len(self.results['disconnections'])}")
        print(f"  - Mocks: {len(self.results['mocks'])}")
        print(f"  - Fluidity Issues: {len(self.results['fluidity_issues'])}")
        print(f"  - Coherence Issues: {len(self.results['coherence_issues'])}")
        print(f"\nTotal Successes: {self.results['statistics']['total_successes']}")
        print("="*60)
        
    def run_audit(self):
        """Run the complete system audit"""
        logger.info("Starting Kimera SWM Comprehensive System Audit...")
        
        try:
            # Add project root to path
            sys.path.insert(0, str(self.project_root))
            
            # Run all checks
            self.check_environment_setup()
            self.check_database_configuration()
            self.check_imports_and_modules()
            self.check_api_endpoints()
            self.check_mock_implementations()
            self.check_system_coherence()
            self.check_performance_and_resources()
            self.check_configuration_files()
            self.check_logging_configuration()
            
            # Generate report
            self.generate_report()
            
        except Exception as e:
            logger.error(f"Audit failed with error: {str(e)}")
            traceback.print_exc()
            
        return self.results


if __name__ == "__main__":
    auditor = KimeraAuditor()
    results = auditor.run_audit() 