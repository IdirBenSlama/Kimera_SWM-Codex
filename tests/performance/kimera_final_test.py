#!/usr/bin/env python3
"""
Kimera SWM Final Test
====================
Tests the system after applying all fixes to ensure everything is working correctly.
"""

import os
import sys
import logging
import asyncio
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KimeraFinalTester:
    def __init__(self):
        self.project_root = Path(__file__).parent.absolute()
        self.test_results = {
            "timestamp": datetime.now().isoformat(),
            "passed": [],
            "failed": [],
            "warnings": []
        }
        
    def log_test(self, test_name: str, passed: bool, details: str = ""):
        """Log test result"""
        if passed:
            self.test_results["passed"].append({"test": test_name, "details": details})
            logger.info(f"✓ PASSED: {test_name} - {details}")
        else:
            self.test_results["failed"].append({"test": test_name, "details": details})
            logger.error(f"✗ FAILED: {test_name} - {details}")
            
    def log_warning(self, test_name: str, details: str):
        """Log test warning"""
        self.test_results["warnings"].append({"test": test_name, "details": details})
        logger.warning(f"⚠ WARNING: {test_name} - {details}")
        
    def test_environment_variables(self):
        """Test environment variables are set"""
        logger.info("\n=== Testing Environment Variables ===")
        
        # Check .env file exists
        env_file = self.project_root / ".env"
        if env_file.exists():
            self.log_test("env_file_exists", True, ".env file found")
            
            # Load environment variables
            from dotenv import load_dotenv
            load_dotenv(env_file)
            
            # Check critical variables
            vars_to_check = ["DATABASE_URL", "KIMERA_DATABASE_URL", "OPENAI_API_KEY"]
            for var in vars_to_check:
                value = os.getenv(var)
                if value:
                    self.log_test(f"env_var_{var}", True, f"{var} is set")
                else:
                    self.log_test(f"env_var_{var}", False, f"{var} not found")
        else:
            self.log_test("env_file_exists", False, ".env file not found")
            
    def test_module_imports(self):
        """Test all critical modules can be imported"""
        logger.info("\n=== Testing Module Imports ===")
        
        # Add project root to path
        sys.path.insert(0, str(self.project_root))
        
        modules_to_test = [
            "backend.core.kimera_system",
            "backend.engines.geoid_scar_manager",
            "backend.monitoring.system_monitor",
            "backend.governance.ethical_governor",
            "backend.vault.vault_manager",
            "backend.engines.contradiction_engine",
            "backend.utils.config"
        ]
        
        for module_name in modules_to_test:
            try:
                __import__(module_name)
                self.log_test(f"import_{module_name}", True, "Module imported successfully")
            except Exception as e:
                self.log_test(f"import_{module_name}", False, str(e))
                
    def test_database_configuration(self):
        """Test database configuration alignment"""
        logger.info("\n=== Testing Database Configuration ===")
        
        try:
            # Check database.py uses environment variable
            db_file = self.project_root / "backend" / "vault" / "database.py"
            if db_file.exists():
                content = db_file.read_text()
                if 'os.getenv("KIMERA_DATABASE_URL", "sqlite:///kimera_swm.db")' in content:
                    self.log_test("database_config_aligned", True, "Database configuration uses environment variable")
                else:
                    self.log_test("database_config_aligned", False, "Database configuration still hardcoded")
            else:
                self.log_test("database_config_aligned", False, "database.py not found")
                
        except Exception as e:
            self.log_test("database_config_aligned", False, str(e))
            
    def test_kimera_system_initialization(self):
        """Test KimeraSystem can be initialized with all components"""
        logger.info("\n=== Testing KimeraSystem Initialization ===")
        
        try:
            from src.core.kimera_system import KimeraSystem
            
            # Create instance
            system = KimeraSystem()
            self.log_test("kimera_system_create", True, "KimeraSystem instance created")
            
            # Initialize system
            system.initialize()
            self.log_test("kimera_system_initialize", True, "KimeraSystem initialized")
            
            # Check component registration
            components_to_check = [
                "geoid_scar_manager",
                "system_monitor",
                "ethical_governor",
                "vault_manager",
                "contradiction_engine"
            ]
            
            for component in components_to_check:
                comp_instance = system.get_component(component)
                if comp_instance is not None:
                    self.log_test(f"component_{component}", True, f"{component} registered and available")
                else:
                    self.log_warning(f"component_{component}", f"{component} not available (may have failed to initialize)")
                    
            # Get system status
            status = system.get_system_status()
            self.log_test("system_status", True, f"System state: {status['state']}")
            
        except Exception as e:
            self.log_test("kimera_system_initialization", False, str(e))
            
    def test_api_endpoints(self):
        """Test API endpoint definitions"""
        logger.info("\n=== Testing API Endpoints ===")
        
        # Check contradiction router endpoints
        try:
            router_file = self.project_root / "backend" / "api" / "routers" / "contradiction_router.py"
            if router_file.exists():
                content = router_file.read_text()
                
                endpoints = ["/contradictions/detect", "/contradictions/resolve"]
                for endpoint in endpoints:
                    if endpoint in content:
                        self.log_test(f"endpoint_contradiction_{endpoint}", True, f"Endpoint {endpoint} found")
                    else:
                        self.log_test(f"endpoint_contradiction_{endpoint}", False, f"Endpoint {endpoint} missing")
            else:
                self.log_test("contradiction_router", False, "Router file not found")
                
        except Exception as e:
            self.log_test("contradiction_endpoints", False, str(e))
            
        # Check vault router endpoints
        try:
            vault_router_file = self.project_root / "backend" / "api" / "routers" / "vault_router.py"
            if vault_router_file.exists():
                content = vault_router_file.read_text()
                
                endpoints = ["/vault/store", "/vault/retrieve"]
                for endpoint in endpoints:
                    if endpoint in content:
                        self.log_test(f"endpoint_vault_{endpoint}", True, f"Endpoint {endpoint} found")
                    else:
                        self.log_test(f"endpoint_vault_{endpoint}", False, f"Endpoint {endpoint} missing")
            else:
                self.log_test("vault_router", False, "Router file not found")
                
        except Exception as e:
            self.log_test("vault_endpoints", False, str(e))
            
    async def test_component_functionality(self):
        """Test basic functionality of components"""
        logger.info("\n=== Testing Component Functionality ===")
        
        try:
            # Test GeoidScarManager
            from src.engines.geoid_scar_manager import GeoidScarManager
            manager = GeoidScarManager()
            
            # Create a geoid
            geoid_id = await manager.create_geoid({"test": "data"})
            if geoid_id:
                self.log_test("geoid_creation", True, f"Created geoid: {geoid_id}")
            else:
                self.log_test("geoid_creation", False, "Failed to create geoid")
                
            # Create a SCAR
            scar_id = await manager.create_scar([geoid_id], "Test SCAR")
            if scar_id:
                self.log_test("scar_creation", True, f"Created SCAR: {scar_id}")
            else:
                self.log_test("scar_creation", False, "Failed to create SCAR")
                
        except Exception as e:
            self.log_test("geoid_scar_functionality", False, str(e))
            
        try:
            # Test SystemMonitor
            from src.monitoring.system_monitor import SystemMonitor
            monitor = SystemMonitor()
            
            # Get health check
            health = await monitor.get_health_check()
            if health and "status" in health:
                self.log_test("system_monitor_health", True, f"Health status: {health['status']}")
            else:
                self.log_test("system_monitor_health", False, "Failed to get health check")
                
        except Exception as e:
            self.log_test("system_monitor_functionality", False, str(e))
            
        try:
            # Test EthicalGovernor
            from src.governance.ethical_governor import EthicalGovernor
            governor = EthicalGovernor()
            
            # Evaluate an action
            action = {"id": "test_action", "risk_level": 0.3, "transparency": 0.9}
            evaluation = await governor.evaluate_action(action)
            if evaluation and "approved" in evaluation:
                self.log_test("ethical_evaluation", True, f"Action approved: {evaluation['approved']}")
            else:
                self.log_test("ethical_evaluation", False, "Failed to evaluate action")
                
        except Exception as e:
            self.log_test("ethical_governor_functionality", False, str(e))
            
    def generate_report(self):
        """Generate test report"""
        total_tests = len(self.test_results["passed"]) + len(self.test_results["failed"])
        pass_rate = len(self.test_results["passed"]) / total_tests * 100 if total_tests > 0 else 0
        
        logger.info("\n" + "="*60)
        logger.info("KIMERA SYSTEM FINAL TEST REPORT")
        logger.info("="*60)
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {len(self.test_results['passed'])}")
        logger.info(f"Failed: {len(self.test_results['failed'])}")
        logger.info(f"Warnings: {len(self.test_results['warnings'])}")
        logger.info(f"Pass Rate: {pass_rate:.1f}%")
        logger.info("="*60)
        
        if self.test_results["failed"]:
            logger.info("\nFailed Tests:")
            for failure in self.test_results["failed"]:
                logger.info(f"  - {failure['test']}: {failure['details']}")
                
        if self.test_results["warnings"]:
            logger.info("\nWarnings:")
            for warning in self.test_results["warnings"]:
                logger.info(f"  - {warning['test']}: {warning['details']}")
                
        # Save detailed report
        import json
        report_file = f"kimera_final_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        logger.info(f"\nDetailed report saved to: {report_file}")
        
    async def run_tests(self):
        """Run all tests"""
        logger.info("Starting Kimera System Final Tests...")
        
        try:
            # Run synchronous tests
            self.test_environment_variables()
            self.test_module_imports()
            self.test_database_configuration()
            self.test_kimera_system_initialization()
            self.test_api_endpoints()
            
            # Run async tests
            await self.test_component_functionality()
            
            # Generate report
            self.generate_report()
            
        except Exception as e:
            logger.error(f"Test suite failed: {str(e)}")
            
        return self.test_results


if __name__ == "__main__":
    tester = KimeraFinalTester()
    results = asyncio.run(tester.run_tests()) 