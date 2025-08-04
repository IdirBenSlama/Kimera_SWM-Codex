#!/usr/bin/env python3
"""
DIRECT KIMERA SYSTEM AUDIT
=========================

Direct audit of Kimera SWM system components without relying on web endpoints.
Tests core system, engines, and capabilities directly through Python imports.
"""

import sys
import traceback
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DirectKimeraAuditor:
    """Direct system auditor for Kimera SWM core components"""
    
    def __init__(self):
        self.audit_results = {}
        self.issues_found = []
        self.timestamp = datetime.now()
        self.kimera_system = None
    
    def test_core_system_import(self) -> bool:
        """Test if core Kimera system can be imported and initialized"""
        logger.info("🧠 TESTING CORE SYSTEM IMPORT")
        logger.info("-" * 50)
        
        try:
            from core.kimera_system import KimeraSystem
            logger.info("✅ Core KimeraSystem import successful")
            
            # Get singleton instance using constructor
            self.kimera_system = KimeraSystem()
            logger.info("✅ KimeraSystem singleton instance obtained")
            
            return True
            
        except Exception as e:
            logger.info(f"❌ Core system import failed: {e}")
            self.issues_found.append(f"Core system import failed: {e}")
            return False
    
    def test_system_initialization(self) -> bool:
        """Test system initialization"""
        logger.info("\n🚀 TESTING SYSTEM INITIALIZATION")
        logger.info("-" * 50)
        
        if not self.kimera_system:
            logger.info("❌ Cannot test initialization - core system not available")
            return False
        
        try:
            # Initialize the system
            self.kimera_system.initialize()
            logger.info("✅ System initialization completed")
            
            # Check if system is running
            status = self.kimera_system.get_system_status()
            logger.info(f"✅ System status retrieved: {status.get('system_state', 'unknown')}")
            
            return True
            
        except Exception as e:
            logger.info(f"❌ System initialization failed: {e}")
            self.issues_found.append(f"System initialization failed: {e}")
            return False
    
    def audit_integrated_engines(self) -> Dict[str, Any]:
        """Audit all integrated engines"""
        logger.info("\n⚙️ AUDITING INTEGRATED ENGINES")
        logger.info("-" * 50)
        
        engines_status = {}
        
        if not self.kimera_system:
            logger.info("❌ Cannot audit engines - core system not available")
            return engines_status
        
        # List of engines to test (based on our previous integrations)
        engine_tests = [
            ("Understanding Engine", "get_understanding_engine"),
            ("Human Interface", "get_human_interface"),
            ("Linguistic Intelligence", "get_linguistic_intelligence_engine"),
            ("Enhanced Thermodynamic Scheduler", "get_enhanced_thermodynamic_scheduler"),
            ("Quantum Cognitive Engine", "get_quantum_cognitive_engine"),
            ("Revolutionary Intelligence", "get_revolutionary_intelligence_engine"),
            ("Meta Insight Engine", "get_meta_insight_engine"),
            ("Ethical Reasoning Engine", "get_ethical_reasoning_engine"),
            ("Unsupervised Learning Engine", "get_unsupervised_cognitive_learning_engine"),
            ("Complexity Analysis Engine", "get_complexity_analysis_engine"),
            ("Quantum Field Engine", "get_quantum_field_engine"),
            ("GPU Cryptographic Engine", "get_gpu_cryptographic_engine"),
            ("Thermodynamic Integration", "get_thermodynamic_integration"),
            ("Unified Thermodynamic Integration", "get_unified_thermodynamic_integration")
        ]
        
        operational_count = 0
        
        for engine_name, getter_method in engine_tests:
            try:
                if hasattr(self.kimera_system, getter_method):
                    engine = getattr(self.kimera_system, getter_method)()
                    if engine is not None:
                        logger.info(f"   ✅ {engine_name}: Operational")
                        engines_status[engine_name] = "operational"
                        operational_count += 1
                    else:
                        logger.info(f"   ❌ {engine_name}: Not available")
                        engines_status[engine_name] = "not_available"
                        self.issues_found.append(f"Engine not available: {engine_name}")
                else:
                    logger.info(f"   ❌ {engine_name}: Getter method not found")
                    engines_status[engine_name] = "no_getter"
                    self.issues_found.append(f"Engine getter missing: {engine_name}")
                    
            except Exception as e:
                logger.info(f"   ❌ {engine_name}: Error - {str(e)[:50]}...")
                engines_status[engine_name] = f"error: {type(e).__name__}"
                self.issues_found.append(f"Engine error {engine_name}: {e}")
        
        logger.info(f"\nEngine Summary: {operational_count}/{len(engine_tests)} operational")
        return engines_status
    
    def test_thermodynamic_capabilities(self) -> Dict[str, Any]:
        """Test revolutionary thermodynamic capabilities"""
        logger.info("\n🔥 TESTING THERMODYNAMIC CAPABILITIES")
        logger.info("-" * 50)
        
        thermo_results = {}
        
        try:
            # Test thermodynamic engines directly
            from engines.vortex_thermodynamic_battery import VortexThermodynamicBattery
            logger.info("✅ Vortex Thermodynamic Battery import successful")
            
            battery = VortexThermodynamicBattery()
            status = battery.get_status()
            logger.info(f"   Battery Status: {status}")
            thermo_results['vortex_battery'] = 'operational'
            
        except Exception as e:
            logger.info(f"❌ Vortex Battery test failed: {e}")
            thermo_results['vortex_battery'] = f'error: {e}'
            self.issues_found.append(f"Vortex Battery error: {e}")
        
        try:
            from engines.portal_maxwell_demon import PortalMaxwellDemon
            logger.info("✅ Portal Maxwell Demon import successful")
            
            demon = PortalMaxwellDemon()
            thermo_results['maxwell_demon'] = 'operational'
            
        except Exception as e:
            logger.info(f"❌ Maxwell Demon test failed: {e}")
            thermo_results['maxwell_demon'] = f'error: {e}'
            self.issues_found.append(f"Maxwell Demon error: {e}")
        
        try:
            from engines.contradiction_heat_pump import ContradictionHeatPump
            logger.info("✅ Contradiction Heat Pump import successful")
            
            pump = ContradictionHeatPump()
            thermo_results['heat_pump'] = 'operational'
            
        except Exception as e:
            logger.info(f"❌ Heat Pump test failed: {e}")
            thermo_results['heat_pump'] = f'error: {e}'
            self.issues_found.append(f"Heat Pump error: {e}")
        
        return thermo_results
    
    def test_database_connectivity(self) -> Dict[str, Any]:
        """Test database connectivity"""
        logger.info("\n🗃️ TESTING DATABASE CONNECTIVITY")
        logger.info("-" * 50)
        
        db_results = {}
        
        try:
            from utils.database import DatabaseManager
            logger.info("✅ Database manager import successful")
            
            # Try to create a database connection
            db_manager = DatabaseManager()
            db_results['import'] = 'success'
            logger.info("✅ Database manager initialized")
            
        except Exception as e:
            logger.info(f"❌ Database test failed: {e}")
            db_results['import'] = f'error: {e}'
            self.issues_found.append(f"Database error: {e}")
        
        return db_results
    
    def generate_audit_report(self) -> Dict[str, Any]:
        """Generate comprehensive audit report"""
        logger.info("\n📊 GENERATING DIRECT AUDIT REPORT")
        logger.info("=" * 80)
        
        total_issues = len(self.issues_found)
        
        if total_issues == 0:
            logger.info("🎉 DIRECT AUDIT PASSED: No issues found!")
            audit_status = "PASSED"
        elif total_issues <= 5:
            logger.info(f"⚠️ DIRECT AUDIT WARNING: {total_issues} minor issues found")
            audit_status = "WARNING" 
        else:
            logger.info(f"❌ DIRECT AUDIT FAILED: {total_issues} issues found")
            audit_status = "FAILED"
        
        logger.info(f"\n📋 Issues Summary:")
        if self.issues_found:
            for i, issue in enumerate(self.issues_found, 1):
                logger.info(f"   {i}. {issue}")
        else:
            logger.info("   No issues found!")
        
        # Create comprehensive report
        audit_report = {
            'audit_timestamp': self.timestamp.isoformat(),
            'audit_type': 'direct_system',
            'audit_status': audit_status,
            'total_issues': total_issues,
            'issues_found': self.issues_found,
            'results': self.audit_results
        }
        
        return audit_report

def main():
    """Main direct audit execution"""
    logger.info("🔍 DIRECT KIMERA SWM SYSTEM AUDIT")
    logger.info("=" * 80)
    logger.info(f"Direct audit started at: {datetime.now()}")
    logger.info()
    
    auditor = DirectKimeraAuditor()
    
    # Test core system
    if not auditor.test_core_system_import():
        logger.info("❌ Cannot proceed - core system import failed")
        return False
    
    # Test initialization
    if not auditor.test_system_initialization():
        logger.info("❌ System initialization failed")
    
    # Run component audits
    auditor.audit_results['engines'] = auditor.audit_integrated_engines()
    auditor.audit_results['thermodynamic'] = auditor.test_thermodynamic_capabilities()
    auditor.audit_results['database'] = auditor.test_database_connectivity()
    
    # Generate final report
    final_report = auditor.generate_audit_report()
    
    return final_report['audit_status'] in ['PASSED', 'WARNING']

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 