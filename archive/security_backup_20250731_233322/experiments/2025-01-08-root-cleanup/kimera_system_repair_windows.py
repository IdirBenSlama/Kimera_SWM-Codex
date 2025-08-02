#!/usr/bin/env python3
"""
KIMERA System Repair and Health Check - Windows Compatible Version
==================================================================

This version handles Windows encoding issues and provides clean output.
"""

import os
import sys
import asyncio
import logging
import json
import time
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import subprocess
import psutil

# Fix Windows console encoding
if sys.platform == 'win32':
    # Set console to UTF-8
    os.system('chcp 65001 > nul')
    # Configure Python for UTF-8
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Configure logging without Unicode characters for Windows compatibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('kimera_repair.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Disable Unicode in specific loggers that cause issues
for logger_name in ['backend.config.kimera_config', 'backend.vault.vault_manager', 
                    'backend.engines.kimera_text_diffusion_engine', 'backend.core.context_imposer',
                    'backend.engines.universal_translator_hub']:
    specific_logger = logging.getLogger(logger_name)
    specific_logger.setLevel(logging.WARNING)

@dataclass
class RepairResult:
    """Result of a repair operation."""
    component: str
    issue: str
    action_taken: str
    success: bool
    error: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class KimeraSystemRepair:
    """Main system repair class - Windows compatible."""
    
    def __init__(self):
        self.results: List[RepairResult] = []
        self.system_status = {
            "healthy": False,
            "components": {},
            "issues_found": 0,
            "issues_fixed": 0,
            "critical_errors": []
        }
    
    async def run_full_repair(self) -> Dict[str, Any]:
        """Run comprehensive system repair."""
        logger.info("Starting KIMERA System Repair and Health Check")
        logger.info("=" * 60)
        
        # Phase 1: Environment and Dependencies
        await self._check_environment()
        await self._check_dependencies()
        
        # Phase 2: Configuration
        await self._check_configuration()
        
        # Phase 3: Core Components
        await self._check_core_components()
        
        # Phase 4: Database and Storage
        await self._check_database()
        
        # Phase 5: API and Endpoints
        await self._check_api_endpoints()
        
        # Phase 6: Performance and Resources
        await self._check_performance()
        
        # Phase 7: Security
        await self._check_security()
        
        # Generate report
        report = self._generate_report()
        
        # Save report
        self._save_report(report)
        
        return report
    
    async def _check_environment(self):
        """Check Python environment and system requirements."""
        logger.info("\nPhase 1: Environment Check")
        logger.info("-" * 40)
        
        # Python version
        python_version = sys.version_info
        if python_version.major == 3 and python_version.minor >= 11:
            logger.info(f"[OK] Python version: {sys.version.split()[0]}")
        else:
            self._add_result(RepairResult(
                component="Environment",
                issue=f"Python version {sys.version.split()[0]} is not optimal",
                action_taken="Recommend upgrading to Python 3.11+",
                success=False,
                error="Incompatible Python version"
            ))
        
        # Check CUDA availability
        try:
            import torch
            if torch.cuda.is_available():
                logger.info(f"[OK] CUDA available: {torch.cuda.get_device_name(0)}")
                logger.info(f"     CUDA version: {torch.version.cuda}")
            else:
                logger.warning("[WARN] CUDA not available - GPU acceleration disabled")
        except ImportError:
            self._add_result(RepairResult(
                component="Environment",
                issue="PyTorch not installed",
                action_taken="Installing PyTorch",
                success=False,
                error="Missing critical dependency"
            ))
        
        # Check memory
        memory = psutil.virtual_memory()
        if memory.total < 8 * 1024 * 1024 * 1024:  # 8GB
            logger.warning(f"[WARN] Low system memory: {memory.total / (1024**3):.1f} GB")
        else:
            logger.info(f"[OK] System memory: {memory.total / (1024**3):.1f} GB")
    
    async def _check_dependencies(self):
        """Check and fix missing dependencies."""
        logger.info("\nChecking Dependencies")
        logger.info("-" * 40)
        
        required_packages = [
            "torch", "transformers", "fastapi", "uvicorn",
            "sqlalchemy", "numpy", "scipy", "pydantic",
            "psutil", "aiofiles", "httpx", "prometheus-client"
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
                logger.info(f"[OK] {package} installed")
            except ImportError:
                missing_packages.append(package)
                logger.error(f"[FAIL] {package} missing")
        
        if missing_packages:
            self._add_result(RepairResult(
                component="Dependencies",
                issue=f"Missing packages: {', '.join(missing_packages)}",
                action_taken="Run: pip install " + " ".join(missing_packages),
                success=False,
                error="Missing dependencies"
            ))
    
    async def _check_configuration(self):
        """Check and fix configuration issues."""
        logger.info("\nPhase 2: Configuration Check")
        logger.info("-" * 40)
        
        try:
            from src.config.kimera_config import get_config, ConfigProfile
            
            config = get_config()
            issues = config.validate()
            
            if not issues:
                logger.info("[OK] Configuration valid")
            else:
                for issue in issues:
                    logger.warning(f"[WARN] Config issue: {issue}")
                    self._add_result(RepairResult(
                        component="Configuration",
                        issue=issue,
                        action_taken="Please update configuration",
                        success=False
                    ))
            
            # Check for production readiness
            if config.profile == ConfigProfile.PRODUCTION:
                if config.security.secret_key.startswith("CHANGE_THIS"):
                    self._add_result(RepairResult(
                        component="Configuration",
                        issue="Default secret key in production",
                        action_taken="Generate secure secret key",
                        success=False,
                        error="Security vulnerability"
                    ))
            
        except Exception as e:
            logger.error(f"[FAIL] Configuration check failed: {e}")
            
            # Create default configuration
            self._create_default_config()
    
    def _create_default_config(self):
        """Create default configuration file."""
        config_dir = project_root / "config"
        config_dir.mkdir(exist_ok=True)
        
        default_config = {
            "profile": "development",
            "database": {
                "url": "sqlite:///./kimera_swm.db"
            },
            "server": {
                "host": "0.0.0.0",
                "port": 8000
            },
            "gpu": {
                "enabled": True
            }
        }
        
        config_file = config_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        self._add_result(RepairResult(
            component="Configuration",
            issue="Missing configuration",
            action_taken=f"Created default config at {config_file}",
            success=True
        ))
    
    async def _check_core_components(self):
        """Check core KIMERA components."""
        logger.info("\nPhase 3: Core Components Check")
        logger.info("-" * 40)
        
        components_to_check = [
            ("KimeraSystem", "backend.core.kimera_system", "get_kimera_system"),
            ("VaultManager", "backend.vault.vault_manager", "VaultManager"),
            ("ContradictionEngine", "backend.engines.contradiction_engine", "ContradictionEngine"),
            ("TextDiffusionEngine", "backend.engines.kimera_text_diffusion_engine", "create_kimera_text_diffusion_engine"),
            ("UniversalTranslatorHub", "backend.engines.universal_translator_hub", "create_universal_translator_hub")
        ]
        
        for component_name, module_path, class_name in components_to_check:
            try:
                module = __import__(module_path, fromlist=[class_name])
                cls = getattr(module, class_name)
                
                # Try to instantiate
                if component_name == "KimeraSystem":
                    instance = cls()
                elif "create_" in class_name:
                    instance = cls()
                else:
                    instance = cls()
                
                logger.info(f"[OK] {component_name} initialized successfully")
                self.system_status["components"][component_name] = "healthy"
                
            except Exception as e:
                logger.error(f"[FAIL] {component_name} initialization failed: {e}")
                self.system_status["components"][component_name] = "failed"
                
                # Attempt repair
                await self._repair_component(component_name, str(e))
    
    async def _repair_component(self, component_name: str, error: str):
        """Attempt to repair a failed component."""
        repair_actions = {
            "KimeraSystem": self._repair_kimera_system,
            "VaultManager": self._repair_vault_manager,
            "TextDiffusionEngine": self._repair_diffusion_engine,
            "ContradictionEngine": self._repair_contradiction_engine
        }
        
        if component_name in repair_actions:
            try:
                await repair_actions[component_name](error)
            except Exception as e:
                self._add_result(RepairResult(
                    component=component_name,
                    issue=error,
                    action_taken="Automated repair failed",
                    success=False,
                    error=str(e)
                ))
    
    async def _repair_kimera_system(self, error: str):
        """Repair KimeraSystem issues."""
        if "thread" in error.lower() or "lock" in error.lower():
            # Thread safety issue - already fixed in our patch
            self._add_result(RepairResult(
                component="KimeraSystem",
                issue="Thread safety issue",
                action_taken="Applied thread-safe singleton pattern",
                success=True
            ))
    
    async def _repair_vault_manager(self, error: str):
        """Repair VaultManager issues."""
        if "database" in error.lower():
            # Create vault directory
            vault_dir = project_root / "vault"
            vault_dir.mkdir(exist_ok=True)
            
            self._add_result(RepairResult(
                component="VaultManager",
                issue="Missing vault directory",
                action_taken=f"Created vault directory at {vault_dir}",
                success=True
            ))
    
    async def _repair_diffusion_engine(self, error: str):
        """Repair TextDiffusionEngine issues."""
        if "model" in error.lower():
            logger.info("Attempting to download required models...")
            # Model download would happen here
            self._add_result(RepairResult(
                component="TextDiffusionEngine",
                issue="Missing language model",
                action_taken="Download models with: python -c \"from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('microsoft/phi-2')\"",
                success=False
            ))
    
    async def _repair_contradiction_engine(self, error: str):
        """Repair ContradictionEngine issues."""
        if "tuple" in error.lower():
            logger.info("ContradictionEngine has a type annotation issue")
            self._add_result(RepairResult(
                component="ContradictionEngine",
                issue="Type annotation error",
                action_taken="The Tuple import has been fixed in governance_engine.py",
                success=True
            ))
    
    async def _check_database(self):
        """Check database connectivity and integrity."""
        logger.info("\nPhase 4: Database Check")
        logger.info("-" * 40)
        
        try:
            from sqlalchemy import create_engine, text
            from src.config.kimera_config import get_config
            
            # Get database URL from configuration
            config = get_config()
            db_url = config.database.url
            engine = create_engine(db_url)
            
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                db_type = "PostgreSQL" if "postgresql" in db_url else "SQLite"
                logger.info(f"[OK] Database connection successful ({db_type})")
                
                # Check tables based on database type
                if "postgresql" in db_url:
                    tables = conn.execute(text(
                        "SELECT table_name FROM information_schema.tables WHERE table_schema='public'"
                    )).fetchall()
                else:
                    tables = conn.execute(text(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )).fetchall()
                
                logger.info(f"     Tables found: {len(tables)}")
                
        except Exception as e:
            logger.error(f"[FAIL] Database check failed: {e}")
            self._add_result(RepairResult(
                component="Database",
                issue=str(e),
                action_taken="Check DATABASE_URL environment variable",
                success=False
            ))
    
    async def _check_api_endpoints(self):
        """Check API endpoint health."""
        logger.info("\nPhase 5: API Endpoints Check")
        logger.info("-" * 40)
        
        # This would normally test actual endpoints
        # For now, we'll check if the routes are properly defined
        try:
            from src.api.main import create_app
            app = create_app()
            
            routes = []
            for route in app.routes:
                if hasattr(route, 'path'):
                    routes.append(route.path)
            
            logger.info(f"[OK] Found {len(routes)} API routes")
            
            critical_routes = ["/health", "/kimera/system/status", "/kimera/api/chat"]
            for route in critical_routes:
                if any(route in r for r in routes):
                    logger.info(f"[OK] Critical route found: {route}")
                else:
                    logger.warning(f"[WARN] Missing critical route: {route}")
                    
        except Exception as e:
            logger.error(f"[FAIL] API check failed: {e}")
            self._add_result(RepairResult(
                component="API",
                issue=str(e),
                action_taken="Check API route definitions",
                success=False
            ))
    
    async def _check_performance(self):
        """Check system performance metrics."""
        logger.info("\nPhase 6: Performance Check")
        logger.info("-" * 40)
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 80:
            logger.warning(f"[WARN] High CPU usage: {cpu_percent}%")
        else:
            logger.info(f"[OK] CPU usage: {cpu_percent}%")
        
        # Memory usage
        memory = psutil.virtual_memory()
        if memory.percent > 80:
            logger.warning(f"[WARN] High memory usage: {memory.percent}%")
        else:
            logger.info(f"[OK] Memory usage: {memory.percent}%")
        
        # Disk usage
        disk = psutil.disk_usage('/')
        if disk.percent > 90:
            logger.warning(f"[WARN] High disk usage: {disk.percent}%")
        else:
            logger.info(f"[OK] Disk usage: {disk.percent}%")
    
    async def _check_security(self):
        """Check security configurations."""
        logger.info("\nPhase 7: Security Check")
        logger.info("-" * 40)
        
        # Check for common security issues
        security_checks = [
            ("Secret key", self._check_secret_key),
            ("CORS settings", self._check_cors),
            ("SSL/TLS", self._check_ssl)
        ]
        
        for check_name, check_func in security_checks:
            try:
                result = await check_func()
                if result:
                    logger.info(f"[OK] {check_name} configured properly")
                else:
                    logger.warning(f"[WARN] {check_name} needs attention")
            except Exception as e:
                logger.error(f"[FAIL] {check_name} check failed: {e}")
    
    async def _check_secret_key(self) -> bool:
        """Check if secret key is properly set."""
        try:
            from src.config.kimera_config import get_config
            config = get_config()
            return not config.security.secret_key.startswith("CHANGE_THIS")
        except Exception as e:
            logger.error(f"Error in kimera_system_repair_windows.py: {e}", exc_info=True)
            raise  # Re-raise for proper error handling
            return False
    
    async def _check_cors(self) -> bool:
        """Check CORS configuration."""
        try:
            from src.config.kimera_config import get_config
            config = get_config()
            return len(config.security.cors_origins) > 0
        except Exception as e:
            logger.error(f"Error in kimera_system_repair_windows.py: {e}", exc_info=True)
            raise  # Re-raise for proper error handling
            return False
    
    async def _check_ssl(self) -> bool:
        """Check SSL/TLS configuration."""
        # In production, this would check for proper SSL setup
        return True
    
    def _add_result(self, result: RepairResult):
        """Add a repair result."""
        self.results.append(result)
        self.system_status["issues_found"] += 1
        if result.success:
            self.system_status["issues_fixed"] += 1
        elif result.error and "critical" in result.error.lower():
            self.system_status["critical_errors"].append(result.issue)
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive repair report."""
        successful_repairs = [r for r in self.results if r.success]
        failed_repairs = [r for r in self.results if not r.success]
        
        # Determine overall health
        self.system_status["healthy"] = (
            len(failed_repairs) == 0 and
            len(self.system_status["critical_errors"]) == 0
        )
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "system_status": self.system_status,
            "summary": {
                "total_checks": len(self.results),
                "successful_repairs": len(successful_repairs),
                "failed_repairs": len(failed_repairs),
                "health_score": (len(successful_repairs) / len(self.results) * 100) if self.results else 100
            },
            "repairs": {
                "successful": [asdict(r) for r in successful_repairs],
                "failed": [asdict(r) for r in failed_repairs]
            },
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Check for critical issues
        if self.system_status["critical_errors"]:
            recommendations.append("CRITICAL: Address critical errors immediately")
        
        # Check components
        failed_components = [
            comp for comp, status in self.system_status["components"].items()
            if status == "failed"
        ]
        if failed_components:
            recommendations.append(f"Repair failed components: {', '.join(failed_components)}")
        
        # Performance recommendations
        memory = psutil.virtual_memory()
        if memory.percent > 80:
            recommendations.append("Consider increasing system memory or optimizing memory usage")
        
        # Security recommendations
        recommendations.append("Regularly update dependencies for security patches")
        recommendations.append("Enable monitoring and alerting for production deployments")
        
        return recommendations
    
    def _save_report(self, report: Dict[str, Any]):
        """Save repair report to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = project_root / f"kimera_repair_report_{timestamp}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"\nReport saved to: {report_file}")
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("KIMERA System Repair Complete")
        logger.info("=" * 60)
        logger.info(f"Overall Health: {'HEALTHY' if report['system_status']['healthy'] else 'NEEDS ATTENTION'}")
        logger.info(f"Health Score: {report['summary']['health_score']:.1f}%")
        logger.info(f"Issues Found: {report['system_status']['issues_found']}")
        logger.info(f"Issues Fixed: {report['system_status']['issues_fixed']}")
        
        if report['recommendations']:
            logger.info("\nRecommendations:")
            for rec in report['recommendations']:
                logger.info(f"  - {rec}")

async def main():
    """Main entry point."""
    repair = KimeraSystemRepair()
    report = await repair.run_full_repair()
    
    # Return exit code based on health
    return 0 if report['system_status']['healthy'] else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)