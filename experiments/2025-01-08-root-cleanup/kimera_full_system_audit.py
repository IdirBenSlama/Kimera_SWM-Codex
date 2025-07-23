#!/usr/bin/env python3
"""
Kimera SWM Full System Audit
============================
This script performs a comprehensive audit of the Kimera system to identify:
- Missing components or modules
- Broken imports
- Configuration issues
- Database connectivity problems
- API endpoint issues
- Integration misalignments
- Performance bottlenecks
"""

import os
import sys
import logging
import json
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import asyncio
import importlib
import inspect

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class KimeraSystemAuditor:
    """Comprehensive system auditor for Kimera SWM"""
    
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.successes = []
        self.audit_results = {
            "timestamp": datetime.now().isoformat(),
            "issues": [],
            "warnings": [],
            "successes": [],
            "component_status": {},
            "api_endpoints": {},
            "performance_metrics": {}
        }
    
    def log_issue(self, component: str, issue: str, severity: str = "ERROR"):
        """Log a critical issue"""
        entry = {
            "component": component,
            "issue": issue,
            "severity": severity,
            "timestamp": datetime.now().isoformat()
        }
        self.issues.append(entry)
        self.audit_results["issues"].append(entry)
        logger.error(f"[{component}] {issue}")
    
    def log_warning(self, component: str, warning: str):
        """Log a warning"""
        entry = {
            "component": component,
            "warning": warning,
            "timestamp": datetime.now().isoformat()
        }
        self.warnings.append(entry)
        self.audit_results["warnings"].append(entry)
        logger.warning(f"[{component}] {warning}")
    
    def log_success(self, component: str, message: str):
        """Log a success"""
        entry = {
            "component": component,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        self.successes.append(entry)
        self.audit_results["successes"].append(entry)
        logger.info(f"✅ [{component}] {message}")
    
    def audit_imports(self):
        """Audit all critical imports"""
        logger.info("=== AUDITING IMPORTS ===")
        
        critical_imports = [
            # Core modules
            ("backend.core.kimera_system", "KimeraSystem"),
            ("backend.core.geoid", "GeoidState"),
            ("backend.core.scar", "ScarRecord"),
            ("backend.core.insight", "InsightScar"),
            ("backend.core.ethical_governor", "EthicalGovernor"),
            
            # Engine modules
            ("backend.engines.contradiction_engine", "ContradictionEngine"),
            ("backend.engines.foundational_thermodynamic_engine", "FoundationalThermodynamicEngine"),
            ("backend.engines.spde_engine", "create_spde_engine"),
            ("backend.engines.cognitive_cycle_engine", "create_cognitive_cycle_engine"),
            ("backend.engines.meta_insight_engine", "create_meta_insight_engine"),
            ("backend.engines.proactive_detector", "create_proactive_detector"),
            ("backend.engines.revolutionary_intelligence_engine", "create_revolutionary_intelligence_engine"),
            ("backend.engines.universal_translator_hub", "create_universal_translator_hub"),
            
            # Vault and database
            ("backend.vault.vault_manager", "VaultManager"),
            ("backend.vault.database", "SessionLocal", "initialize_database", "get_db_status"),
            
            # API and routers
            ("backend.api.main", "create_app"),
            ("backend.api.routers.geoid_scar_router", "router"),
            ("backend.api.routers.system_router", "router"),
            ("backend.api.routers.contradiction_router", "router"),
            
            # Utils
            ("backend.utils.gpu_foundation", "GPUFoundation"),
            ("backend.utils.config", "get_config"),
            ("backend.utils.kimera_logger", "get_logger"),
            
            # Config
            ("backend.config.settings", "get_settings", "KimeraSettings"),
        ]
        
        for module_path, *items in critical_imports:
            try:
                module = importlib.import_module(module_path)
                self.audit_results["component_status"][module_path] = "imported"
                
                # Check if specific items exist in the module
                missing_items = []
                for item in items:
                    if not hasattr(module, item):
                        missing_items.append(item)
                
                if missing_items:
                    self.log_warning(module_path, f"Missing items: {missing_items}")
                else:
                    self.log_success(module_path, "Import successful")
                    
            except ImportError as e:
                self.log_issue(module_path, f"Import failed: {str(e)}")
                self.audit_results["component_status"][module_path] = "failed"
            except Exception as e:
                self.log_issue(module_path, f"Unexpected error: {str(e)}")
                self.audit_results["component_status"][module_path] = "error"
    
    def audit_configuration(self):
        """Audit configuration and environment"""
        logger.info("=== AUDITING CONFIGURATION ===")
        
        try:
            from src.config.settings import get_settings
            settings = get_settings()
            
            # Check critical settings
            if not settings.database.url:
                self.log_issue("Configuration", "DATABASE_URL not configured")
            else:
                self.log_success("Configuration", f"Database URL: {settings.database.url}")
            
            # Check API keys
            if not settings.api_keys.openai_api_key:
                self.log_warning("Configuration", "OpenAI API key not configured")
            
            # Check paths
            for path_name in ["data_dir", "logs_dir", "models_dir", "temp_dir"]:
                path = getattr(settings.paths, path_name)
                if not path.exists():
                    self.log_warning("Configuration", f"{path_name} does not exist: {path}")
                else:
                    self.log_success("Configuration", f"{path_name} exists: {path}")
            
            self.audit_results["component_status"]["configuration"] = "ok"
            
        except Exception as e:
            self.log_issue("Configuration", f"Failed to load settings: {str(e)}")
            self.audit_results["component_status"]["configuration"] = "failed"
    
    def audit_database(self):
        """Audit database connectivity and schema"""
        logger.info("=== AUDITING DATABASE ===")
        
        try:
            from src.vault.database import get_db_status, SessionLocal, GeoidDB, ScarDB
            
            # Check database status
            db_status = get_db_status()
            if db_status["status"] == "connected":
                self.log_success("Database", f"Connected to {db_status['type']} {db_status.get('version', '')}")
                self.audit_results["component_status"]["database"] = "connected"
            else:
                self.log_issue("Database", f"Connection failed: {db_status.get('error', 'Unknown error')}")
                self.audit_results["component_status"]["database"] = "disconnected"
            
            # Test database operations
            try:
                with SessionLocal() as db:
                    # Test query
                    geoid_count = db.query(GeoidDB).count()
                    scar_count = db.query(ScarDB).count()
                    self.log_success("Database", f"Geoids: {geoid_count}, Scars: {scar_count}")
            except Exception as e:
                self.log_issue("Database", f"Query failed: {str(e)}")
                
        except Exception as e:
            self.log_issue("Database", f"Database audit failed: {str(e)}")
            self.audit_results["component_status"]["database"] = "failed"
    
    def audit_kimera_system(self):
        """Audit the core KimeraSystem"""
        logger.info("=== AUDITING KIMERA SYSTEM ===")
        
        try:
            from src.core.kimera_system import get_kimera_system
            
            kimera = get_kimera_system()
            
            # Check system state
            status = kimera.get_status()
            self.log_success("KimeraSystem", f"Status: {status}")
            
            # Check if system can initialize
            if status == "stopped":
                self.log_warning("KimeraSystem", "System not initialized")
                # Try to initialize
                try:
                    kimera.initialize()
                    self.log_success("KimeraSystem", "System initialized successfully")
                except Exception as e:
                    self.log_issue("KimeraSystem", f"Initialization failed: {str(e)}")
            
            # Check components
            components = [
                "vault_manager", "gpu_foundation", "contradiction_engine",
                "thermodynamics_engine", "embedding_model", "spde_engine",
                "cognitive_cycle_engine", "meta_insight_engine", "proactive_detector",
                "revolutionary_intelligence_engine"
            ]
            
            for component_name in components:
                component = kimera.get_component(component_name)
                if component is None:
                    self.log_warning("KimeraSystem", f"Component not initialized: {component_name}")
                else:
                    self.log_success("KimeraSystem", f"Component ready: {component_name}")
                    self.audit_results["component_status"][f"kimera.{component_name}"] = "ready"
            
        except Exception as e:
            self.log_issue("KimeraSystem", f"System audit failed: {str(e)}")
            self.audit_results["component_status"]["kimera_system"] = "failed"
    
    def audit_api_endpoints(self):
        """Audit API endpoints"""
        logger.info("=== AUDITING API ENDPOINTS ===")
        
        try:
            from src.api.main import create_app
            app = create_app()
            
            # Get all routes
            routes = []
            for route in app.routes:
                if hasattr(route, "path"):
                    routes.append({
                        "path": route.path,
                        "methods": list(route.methods) if hasattr(route, "methods") else [],
                        "name": route.name if hasattr(route, "name") else "unknown"
                    })
            
            self.log_success("API", f"Found {len(routes)} endpoints")
            self.audit_results["api_endpoints"]["total"] = len(routes)
            self.audit_results["api_endpoints"]["routes"] = routes
            
            # Check for critical endpoints
            critical_paths = [
                "/", "/health", "/kimera/status", "/kimera/geoids",
                "/kimera/scars", "/kimera/contradictions/detect"
            ]
            
            route_paths = [r["path"] for r in routes]
            for path in critical_paths:
                if path in route_paths:
                    self.log_success("API", f"Critical endpoint found: {path}")
                else:
                    self.log_warning("API", f"Critical endpoint missing: {path}")
            
        except Exception as e:
            self.log_issue("API", f"API audit failed: {str(e)}")
            self.audit_results["component_status"]["api"] = "failed"
    
    def audit_engines(self):
        """Audit individual engine components"""
        logger.info("=== AUDITING ENGINES ===")
        
        engines = [
            ("ContradictionEngine", "backend.engines.contradiction_engine", "ContradictionEngine"),
            ("ThermodynamicEngine", "backend.engines.foundational_thermodynamic_engine", "FoundationalThermodynamicEngine"),
            ("UniversalTranslatorHub", "backend.engines.universal_translator_hub", "create_universal_translator_hub"),
        ]
        
        for engine_name, module_path, class_name in engines:
            try:
                module = importlib.import_module(module_path)
                if hasattr(module, class_name):
                    self.log_success(engine_name, "Engine module loaded")
                    
                    # Try to instantiate if it's a class
                    if class_name.startswith("create_"):
                        # It's a factory function
                        self.log_success(engine_name, "Factory function available")
                    else:
                        # It's a class, try to instantiate
                        cls = getattr(module, class_name)
                        if inspect.isclass(cls):
                            # Try basic instantiation
                            try:
                                if engine_name == "ContradictionEngine":
                                    instance = cls(tension_threshold=0.4)
                                else:
                                    instance = cls()
                                self.log_success(engine_name, "Engine instantiated successfully")
                            except Exception as e:
                                self.log_warning(engine_name, f"Instantiation failed: {str(e)}")
                else:
                    self.log_issue(engine_name, f"Class {class_name} not found in module")
                    
            except Exception as e:
                self.log_issue(engine_name, f"Engine audit failed: {str(e)}")
    
    def audit_integration(self):
        """Audit system integration and connections"""
        logger.info("=== AUDITING INTEGRATION ===")
        
        # Check if components can work together
        try:
            from src.core.kimera_system import get_kimera_system
            from src.core.geoid import GeoidState
            from src.core.scar import ScarRecord
            
            kimera = get_kimera_system()
            
            # Test vault manager integration
            vault_manager = kimera.get_vault_manager()
            if vault_manager:
                try:
                    geoids = vault_manager.get_all_geoids()
                    self.log_success("Integration", f"Vault manager can retrieve geoids: {len(geoids)}")
                except Exception as e:
                    self.log_issue("Integration", f"Vault manager integration failed: {str(e)}")
            
            # Test contradiction engine integration
            contradiction_engine = kimera.get_contradiction_engine()
            if contradiction_engine and vault_manager:
                try:
                    # Create test geoids
                    test_geoids = [
                        GeoidState(
                            geoid_id=f"TEST_{i}",
                            semantic_state={"test": i},
                            symbolic_state={"symbol": i},
                            embedding_vector=[float(i)] * 384
                        )
                        for i in range(3)
                    ]
                    
                    # Test tension detection
                    tensions = contradiction_engine.detect_tension_gradients(test_geoids)
                    self.log_success("Integration", f"Contradiction engine detected {len(tensions)} tensions")
                except Exception as e:
                    self.log_issue("Integration", f"Contradiction engine integration failed: {str(e)}")
            
        except Exception as e:
            self.log_issue("Integration", f"Integration audit failed: {str(e)}")
    
    def audit_performance(self):
        """Basic performance checks"""
        logger.info("=== AUDITING PERFORMANCE ===")
        
        try:
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.audit_results["performance_metrics"]["cpu_percent"] = cpu_percent
            if cpu_percent > 80:
                self.log_warning("Performance", f"High CPU usage: {cpu_percent}%")
            else:
                self.log_success("Performance", f"CPU usage: {cpu_percent}%")
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.audit_results["performance_metrics"]["memory_percent"] = memory.percent
            if memory.percent > 80:
                self.log_warning("Performance", f"High memory usage: {memory.percent}%")
            else:
                self.log_success("Performance", f"Memory usage: {memory.percent}%")
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.audit_results["performance_metrics"]["disk_percent"] = disk.percent
            if disk.percent > 90:
                self.log_warning("Performance", f"High disk usage: {disk.percent}%")
            else:
                self.log_success("Performance", f"Disk usage: {disk.percent}%")
                
        except ImportError:
            self.log_warning("Performance", "psutil not installed, skipping performance metrics")
        except Exception as e:
            self.log_issue("Performance", f"Performance audit failed: {str(e)}")
    
    def generate_report(self):
        """Generate comprehensive audit report"""
        logger.info("=== GENERATING AUDIT REPORT ===")
        
        # Summary statistics
        self.audit_results["summary"] = {
            "total_issues": len(self.issues),
            "total_warnings": len(self.warnings),
            "total_successes": len(self.successes),
            "critical_issues": len([i for i in self.issues if i.get("severity") == "CRITICAL"]),
            "audit_duration": (datetime.now() - datetime.fromisoformat(self.audit_results["timestamp"])).total_seconds()
        }
        
        # Save report
        report_path = Path("kimera_audit_report.json")
        with open(report_path, "w") as f:
            json.dump(self.audit_results, f, indent=2)
        
        logger.info(f"Audit report saved to: {report_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("KIMERA SYSTEM AUDIT SUMMARY")
        print("="*60)
        print(f"✅ Successes: {len(self.successes)}")
        print(f"⚠️  Warnings: {len(self.warnings)}")
        print(f"❌ Issues: {len(self.issues)}")
        print(f"🔴 Critical Issues: {self.audit_results['summary']['critical_issues']}")
        print(f"⏱️  Audit Duration: {self.audit_results['summary']['audit_duration']:.2f}s")
        print("="*60)
        
        if self.issues:
            print("\nCRITICAL ISSUES TO FIX:")
            for issue in self.issues[:10]:  # Show top 10 issues
                print(f"  - [{issue['component']}] {issue['issue']}")
        
        return self.audit_results

def main():
    """Run the full system audit"""
    auditor = KimeraSystemAuditor()
    
    # Run all audits
    auditor.audit_imports()
    auditor.audit_configuration()
    auditor.audit_database()
    auditor.audit_kimera_system()
    auditor.audit_api_endpoints()
    auditor.audit_engines()
    auditor.audit_integration()
    auditor.audit_performance()
    
    # Generate report
    report = auditor.generate_report()
    
    # Return exit code based on issues
    if auditor.issues:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main() 