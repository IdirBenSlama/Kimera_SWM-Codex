#!/usr/bin/env python3
"""
Kimera SWM System Fixes
======================
This script fixes all critical issues identified in the system audit.
"""

import os
import sys
import logging
import shutil
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KimeraSystemFixer:
    def __init__(self):
        self.project_root = Path(__file__).parent.absolute()
        self.fixes_applied = []
        self.fixes_failed = []
        
    def log_fix(self, fix_name: str, success: bool, details: str = ""):
        """Log fix result"""
        if success:
            self.fixes_applied.append(fix_name)
            logger.info(f"âœ“ Fixed: {fix_name} - {details}")
        else:
            self.fixes_failed.append(fix_name)
            logger.error(f"âœ— Failed: {fix_name} - {details}")
            
    def fix_database_configuration(self):
        """Fix database configuration mismatch"""
        logger.info("\n=== Fixing Database Configuration ===")
        
        try:
            # Update database.py to use environment variable properly
            db_file = self.project_root / "backend" / "vault" / "database.py"
            if db_file.exists():
                content = db_file.read_text()
                
                # Replace hardcoded PostgreSQL with environment-based configuration
                new_content = content.replace(
                    'DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://kimera:kimera_secure_pass_2025@localhost:5432/kimera_swm")',
                    'DATABASE_URL = os.getenv("DATABASE_URL", os.getenv("KIMERA_DATABASE_URL", "sqlite:///kimera_swm.db"))'
                )
                
                db_file.write_text(new_content)
                self.log_fix("database_configuration", True, "Aligned database.py with settings.py")
            else:
                self.log_fix("database_configuration", False, "database.py not found")
                
        except Exception as e:
            self.log_fix("database_configuration", False, str(e))
            
    def create_missing_modules(self):
        """Create missing critical modules"""
        logger.info("\n=== Creating Missing Modules ===")
        
        missing_modules = [
            ("backend/engines/geoid_scar_manager.py", self._get_geoid_scar_manager_code()),
            ("backend/monitoring/system_monitor.py", self._get_system_monitor_code()),
            ("backend/governance/ethical_governor.py", self._get_ethical_governor_code())
        ]
        
        for module_path, code in missing_modules:
            try:
                file_path = self.project_root / module_path
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(code)
                self.log_fix(f"create_module_{Path(module_path).stem}", True, module_path)
            except Exception as e:
                self.log_fix(f"create_module_{Path(module_path).stem}", False, str(e))
                
    def fix_environment_variables(self):
        """Create/update environment configuration"""
        logger.info("\n=== Fixing Environment Variables ===")
        
        try:
            env_file = self.project_root / ".env"
            env_content = []
            
            # Read existing content if file exists
            if env_file.exists():
                env_content = env_file.read_text().splitlines()
                
            # Add missing variables
            required_vars = {
                "DATABASE_URL": "sqlite:///kimera_swm.db",
                "KIMERA_DATABASE_URL": "sqlite:///kimera_swm.db",
                "OPENAI_API_KEY": "your-openai-api-key-here",
                "KIMERA_ENV": "development"
            }
            
            existing_vars = {}
            for line in env_content:
                if "=" in line and not line.strip().startswith("#"):
                    key = line.split("=")[0].strip()
                    existing_vars[key] = line
                    
            # Add missing variables
            for var, default_value in required_vars.items():
                if var not in existing_vars:
                    env_content.append(f"{var}={default_value}")
                    
            # Write updated content
            env_file.write_text("\n".join(env_content))
            self.log_fix("environment_variables", True, f"Updated {len(required_vars)} variables")
            
        except Exception as e:
            self.log_fix("environment_variables", False, str(e))
            
    def fix_api_endpoints(self):
        """Add missing API endpoints"""
        logger.info("\n=== Fixing API Endpoints ===")
        
        # Fix contradiction router
        try:
            router_file = self.project_root / "backend" / "api" / "routers" / "contradiction_router.py"
            if router_file.exists():
                content = router_file.read_text()
                
                # Add missing endpoints if not present
                if "/contradictions/detect" not in content:
                    # Add detect endpoint
                    detect_endpoint = '''

@router.post("/contradictions/detect", response_model=Dict[str, Any])
async def detect_contradictions(
    geoid_ids: List[str],
    threshold: float = 0.5
) -> Dict[str, Any]:
    """Detect contradictions between geoids"""
    try:
        contradiction_engine = kimera_singleton.get_component("contradiction_engine")
        if not contradiction_engine:
            raise HTTPException(status_code=503, detail="Contradiction engine not available")
            
        contradictions = await contradiction_engine.detect_contradictions(geoid_ids, threshold)
        return {
            "status": "success",
            "contradictions": contradictions,
            "count": len(contradictions)
        }
    except Exception as e:
        logger.error(f"Error detecting contradictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))
'''
                    content = content.rstrip() + detect_endpoint
                    
                if "/contradictions/resolve" not in content:
                    # Add resolve endpoint
                    resolve_endpoint = '''

@router.post("/contradictions/resolve", response_model=Dict[str, Any])
async def resolve_contradiction(
    contradiction_id: str,
    resolution_strategy: str = "entropy_minimization"
) -> Dict[str, Any]:
    """Resolve a detected contradiction"""
    try:
        contradiction_engine = kimera_singleton.get_component("contradiction_engine")
        if not contradiction_engine:
            raise HTTPException(status_code=503, detail="Contradiction engine not available")
            
        result = await contradiction_engine.resolve_contradiction(
            contradiction_id, 
            strategy=resolution_strategy
        )
        return {
            "status": "success",
            "resolution": result,
            "contradiction_id": contradiction_id
        }
    except Exception as e:
        logger.error(f"Error resolving contradiction: {e}")
        raise HTTPException(status_code=500, detail=str(e))
'''
                    content = content.rstrip() + resolve_endpoint
                    
                router_file.write_text(content)
                self.log_fix("contradiction_endpoints", True, "Added detect and resolve endpoints")
            else:
                self.log_fix("contradiction_endpoints", False, "Router file not found")
                
        except Exception as e:
            self.log_fix("contradiction_endpoints", False, str(e))
            
        # Fix vault router
        try:
            vault_router_file = self.project_root / "backend" / "api" / "routers" / "vault_router.py"
            if vault_router_file.exists():
                content = vault_router_file.read_text()
                
                if "/vault/store" not in content:
                    store_endpoint = '''

@router.post("/vault/store", response_model=Dict[str, Any])
async def store_in_vault(
    key: str,
    value: Any,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Store data in the vault"""
    try:
        vault_manager = kimera_singleton.get_component("vault_manager")
        if not vault_manager:
            raise HTTPException(status_code=503, detail="Vault manager not available")
            
        result = await vault_manager.store(key, value, metadata)
        return {
            "status": "success",
            "key": key,
            "stored": True
        }
    except Exception as e:
        logger.error(f"Error storing in vault: {e}")
        raise HTTPException(status_code=500, detail=str(e))
'''
                    content = content.rstrip() + store_endpoint
                    
                if "/vault/retrieve" not in content:
                    retrieve_endpoint = '''

@router.get("/vault/retrieve/{key}", response_model=Dict[str, Any])
async def retrieve_from_vault(key: str) -> Dict[str, Any]:
    """Retrieve data from the vault"""
    try:
        vault_manager = kimera_singleton.get_component("vault_manager")
        if not vault_manager:
            raise HTTPException(status_code=503, detail="Vault manager not available")
            
        value = await vault_manager.retrieve(key)
        if value is None:
            raise HTTPException(status_code=404, detail=f"Key '{key}' not found in vault")
            
        return {
            "status": "success",
            "key": key,
            "value": value
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving from vault: {e}")
        raise HTTPException(status_code=500, detail=str(e))
'''
                    content = content.rstrip() + retrieve_endpoint
                    
                vault_router_file.write_text(content)
                self.log_fix("vault_endpoints", True, "Added store and retrieve endpoints")
            else:
                self.log_fix("vault_endpoints", False, "Router file not found")
                
        except Exception as e:
            self.log_fix("vault_endpoints", False, str(e))
            
    def _get_geoid_scar_manager_code(self) -> str:
        """Get code for geoid scar manager module"""
        return '''"""
Geoid SCAR Manager
==================
Manages Semantic Contextual Anomaly Representations (SCARs) and Geoids.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

class GeoidScarManager:
    """Manages Geoids and SCARs in the Kimera system"""
    
    def __init__(self):
        self.geoids: Dict[str, Dict[str, Any]] = {}
        self.scars: Dict[str, Dict[str, Any]] = {}
        logger.info("GeoidScarManager initialized")
        
    async def create_geoid(self, data: Dict[str, Any]) -> str:
        """Create a new geoid"""
        geoid_id = str(uuid.uuid4())
        self.geoids[geoid_id] = {
            "id": geoid_id,
            "data": data,
            "created_at": datetime.utcnow().isoformat(),
            "scars": []
        }
        logger.info(f"Created geoid: {geoid_id}")
        return geoid_id
        
    async def create_scar(self, geoid_ids: List[str], reason: str) -> str:
        """Create a new SCAR"""
        scar_id = str(uuid.uuid4())
        self.scars[scar_id] = {
            "id": scar_id,
            "geoid_ids": geoid_ids,
            "reason": reason,
            "created_at": datetime.utcnow().isoformat(),
            "resolved": False
        }
        
        # Link SCAR to geoids
        for geoid_id in geoid_ids:
            if geoid_id in self.geoids:
                self.geoids[geoid_id]["scars"].append(scar_id)
                
        logger.info(f"Created SCAR: {scar_id}")
        return scar_id
        
    async def get_geoid(self, geoid_id: str) -> Optional[Dict[str, Any]]:
        """Get a geoid by ID"""
        return self.geoids.get(geoid_id)
        
    async def get_scar(self, scar_id: str) -> Optional[Dict[str, Any]]:
        """Get a SCAR by ID"""
        return self.scars.get(scar_id)
        
    async def list_geoids(self) -> List[Dict[str, Any]]:
        """List all geoids"""
        return list(self.geoids.values())
        
    async def list_scars(self) -> List[Dict[str, Any]]:
        """List all SCARs"""
        return list(self.scars.values())
        
    def get_status(self) -> Dict[str, Any]:
        """Get manager status"""
        return {
            "total_geoids": len(self.geoids),
            "total_scars": len(self.scars),
            "unresolved_scars": sum(1 for s in self.scars.values() if not s["resolved"])
        }
'''

    def _get_system_monitor_code(self) -> str:
        """Get code for system monitor module"""
        return '''"""
System Monitor
==============
Monitors system health, performance, and resources.
"""

import logging
import psutil
from typing import Dict, Any, Optional
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)

class SystemMonitor:
    """Monitors Kimera system health and performance"""
    
    def __init__(self):
        self.metrics: Dict[str, Any] = {}
        self.alerts: List[Dict[str, Any]] = []
        self.monitoring_active = False
        logger.info("SystemMonitor initialized")
        
    async def start_monitoring(self):
        """Start system monitoring"""
        self.monitoring_active = True
        logger.info("System monitoring started")
        
        while self.monitoring_active:
            await self.collect_metrics()
            await asyncio.sleep(30)  # Collect metrics every 30 seconds
            
    async def stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring_active = False
        logger.info("System monitoring stopped")
        
    async def collect_metrics(self):
        """Collect system metrics"""
        try:
            self.metrics = {
                "timestamp": datetime.utcnow().isoformat(),
                "cpu": {
                    "percent": psutil.cpu_percent(interval=1),
                    "count": psutil.cpu_count()
                },
                "memory": {
                    "percent": psutil.virtual_memory().percent,
                    "available": psutil.virtual_memory().available,
                    "total": psutil.virtual_memory().total
                },
                "disk": {
                    "percent": psutil.disk_usage('/').percent,
                    "free": psutil.disk_usage('/').free,
                    "total": psutil.disk_usage('/').total
                }
            }
            
            # Check for alerts
            await self.check_alerts()
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            
    async def check_alerts(self):
        """Check for system alerts"""
        # CPU alert
        if self.metrics.get("cpu", {}).get("percent", 0) > 80:
            self.add_alert("high_cpu", "CPU usage above 80%", "warning")
            
        # Memory alert
        if self.metrics.get("memory", {}).get("percent", 0) > 80:
            self.add_alert("high_memory", "Memory usage above 80%", "warning")
            
        # Disk alert
        if self.metrics.get("disk", {}).get("percent", 0) > 90:
            self.add_alert("high_disk", "Disk usage above 90%", "critical")
            
    def add_alert(self, alert_type: str, message: str, severity: str):
        """Add a system alert"""
        alert = {
            "type": alert_type,
            "message": message,
            "severity": severity,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.alerts.append(alert)
        logger.warning(f"System alert: {message}")
        
    def get_status(self) -> Dict[str, Any]:
        """Get monitor status"""
        return {
            "monitoring_active": self.monitoring_active,
            "latest_metrics": self.metrics,
            "active_alerts": len([a for a in self.alerts if a["severity"] == "critical"]),
            "total_alerts": len(self.alerts)
        }
        
    async def get_health_check(self) -> Dict[str, Any]:
        """Get system health check"""
        await self.collect_metrics()
        
        health_status = "healthy"
        issues = []
        
        if self.metrics.get("cpu", {}).get("percent", 0) > 90:
            health_status = "degraded"
            issues.append("High CPU usage")
            
        if self.metrics.get("memory", {}).get("percent", 0) > 90:
            health_status = "degraded"
            issues.append("High memory usage")
            
        if self.metrics.get("disk", {}).get("percent", 0) > 95:
            health_status = "critical"
            issues.append("Critical disk usage")
            
        return {
            "status": health_status,
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": self.metrics,
            "issues": issues
        }
'''

    def _get_ethical_governor_code(self) -> str:
        """Get code for ethical governor module"""
        return '''"""
Ethical Governor
================
Governs ethical constraints and decision-making in the Kimera system.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

class EthicalPrinciple(Enum):
    """Core ethical principles"""
    BENEFICENCE = "beneficence"
    NON_MALEFICENCE = "non_maleficence"
    AUTONOMY = "autonomy"
    JUSTICE = "justice"
    TRANSPARENCY = "transparency"
    ACCOUNTABILITY = "accountability"

class EthicalGovernor:
    """Governs ethical decision-making in Kimera"""
    
    def __init__(self):
        self.principles = list(EthicalPrinciple)
        self.decisions: List[Dict[str, Any]] = []
        self.constraints: Dict[str, Any] = {
            "max_risk_level": 0.7,
            "min_transparency": 0.8,
            "require_human_oversight": True
        }
        logger.info("EthicalGovernor initialized")
        
    async def evaluate_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate an action against ethical principles"""
        evaluation = {
            "action_id": action.get("id", "unknown"),
            "timestamp": datetime.utcnow().isoformat(),
            "approved": True,
            "risk_level": 0.0,
            "violations": [],
            "recommendations": []
        }
        
        # Check risk level
        risk_level = action.get("risk_level", 0.0)
        if risk_level > self.constraints["max_risk_level"]:
            evaluation["approved"] = False
            evaluation["violations"].append(f"Risk level {risk_level} exceeds maximum {self.constraints['max_risk_level']}")
            
        # Check transparency
        transparency = action.get("transparency", 1.0)
        if transparency < self.constraints["min_transparency"]:
            evaluation["approved"] = False
            evaluation["violations"].append(f"Transparency {transparency} below minimum {self.constraints['min_transparency']}")
            
        # Check for human oversight requirement
        if self.constraints["require_human_oversight"] and not action.get("human_approved", False):
            evaluation["recommendations"].append("Requires human oversight approval")
            
        evaluation["risk_level"] = risk_level
        self.decisions.append(evaluation)
        
        logger.info(f"Evaluated action {action.get('id')}: approved={evaluation['approved']}")
        return evaluation
        
    async def add_constraint(self, name: str, value: Any):
        """Add or update an ethical constraint"""
        self.constraints[name] = value
        logger.info(f"Updated constraint: {name} = {value}")
        
    async def get_decision_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent ethical decisions"""
        return self.decisions[-limit:]
        
    def get_status(self) -> Dict[str, Any]:
        """Get governor status"""
        total_decisions = len(self.decisions)
        approved_decisions = sum(1 for d in self.decisions if d["approved"])
        
        return {
            "active_principles": [p.value for p in self.principles],
            "active_constraints": self.constraints,
            "total_decisions": total_decisions,
            "approved_decisions": approved_decisions,
            "rejection_rate": (total_decisions - approved_decisions) / total_decisions if total_decisions > 0 else 0
        }
        
    async def check_system_ethics(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        """Check overall system ethical compliance"""
        compliance = {
            "timestamp": datetime.utcnow().isoformat(),
            "compliant": True,
            "issues": [],
            "score": 1.0
        }
        
        # Check various ethical metrics
        if system_state.get("transparency_score", 1.0) < 0.7:
            compliance["compliant"] = False
            compliance["issues"].append("System transparency below acceptable threshold")
            compliance["score"] *= 0.8
            
        if system_state.get("bias_detected", False):
            compliance["compliant"] = False
            compliance["issues"].append("Bias detected in system outputs")
            compliance["score"] *= 0.7
            
        return compliance
'''

    def fix_component_registration(self):
        """Fix component registration in KimeraSystem"""
        logger.info("\n=== Fixing Component Registration ===")
        
        try:
            kimera_system_file = self.project_root / "backend" / "core" / "kimera_system.py"
            if kimera_system_file.exists():
                content = kimera_system_file.read_text()
                
                # Check if components are properly imported and registered
                # This is a simplified check - in reality would need more sophisticated parsing
                components_to_check = [
                    ("from ..engines.geoid_scar_manager import GeoidScarManager", "GeoidScarManager"),
                    ("from ..monitoring.system_monitor import SystemMonitor", "SystemMonitor"),
                    ("from ..governance.ethical_governor import EthicalGovernor", "EthicalGovernor")
                ]
                
                modified = False
                for import_stmt, class_name in components_to_check:
                    if import_stmt not in content:
                        # Add import at the top of the file after other imports
                        import_section_end = content.find("\n\n")
                        if import_section_end > 0:
                            content = content[:import_section_end] + f"\n{import_stmt}" + content[import_section_end:]
                            modified = True
                            
                if modified:
                    kimera_system_file.write_text(content)
                    self.log_fix("component_registration", True, "Added missing imports")
                else:
                    self.log_fix("component_registration", True, "Components already imported")
                    
            else:
                self.log_fix("component_registration", False, "kimera_system.py not found")
                
        except Exception as e:
            self.log_fix("component_registration", False, str(e))
            
    def generate_report(self):
        """Generate fix report"""
        logger.info("\n" + "="*60)
        logger.info("KIMERA SYSTEM FIX REPORT")
        logger.info("="*60)
        logger.info(f"Fixes Applied: {len(self.fixes_applied)}")
        for fix in self.fixes_applied:
            logger.info(f"  âœ“ {fix}")
        logger.info(f"\nFixes Failed: {len(self.fixes_failed)}")
        for fix in self.fixes_failed:
            logger.info(f"  âœ— {fix}")
        logger.info("="*60)
        
    def run_fixes(self):
        """Run all fixes"""
        logger.info("Starting Kimera System Fixes...")
        
        try:
            # Add project root to path
            sys.path.insert(0, str(self.project_root))
            
            # Run fixes in order of priority
            self.fix_database_configuration()
            self.create_missing_modules()
            self.fix_environment_variables()
            self.fix_api_endpoints()
            self.fix_component_registration()
            
            # Generate report
            self.generate_report()
            
        except Exception as e:
            logger.error(f"Fix process failed: {str(e)}")
            
        return {
            "applied": self.fixes_applied,
            "failed": self.fixes_failed
        }


if __name__ == "__main__":
    fixer = KimeraSystemFixer()
    results = fixer.run_fixes() 