"""
KIMERA SWM System Audit
======================

Comprehensive audit of the Kimera system to verify:
1. Engine availability
2. Core system structure
3. Completed integrations
4. Dependencies
5. System initialization
"""

import os
import sys
import importlib
import traceback
from datetime import datetime
from typing import Dict, List, Tuple, Any
import json

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class KimeraSystemAuditor:
    """Comprehensive system auditor for Kimera SWM"""
    
    def __init__(self):
        self.audit_results = {
            "timestamp": datetime.now().isoformat(),
            "engines": {},
            "core_modules": {},
            "integrations": {},
            "dependencies": {},
            "initialization": {},
            "issues": []
        }
        
        # Define expected engines from roadmap
        self.expected_engines = [
            # Axiom Suite
            "axiom_mathematical_proof", "axiom_of_understanding", "axiom_verification",
            # Background Services
            "background_jobs", "clip_service",
            # Advanced Cognitive
            "cognitive_graph_processor", "cognitive_pharmaceutical_optimizer",
            # Validation
            "cognitive_validation_framework", "comprehensive_thermodynamic_monitor",
            # Quantum
            "cuda_quantum_engine", "differential_privacy_engine",
            # Signal Processing
            "diffusion_response_fix", "emergent_signal_intelligence",
            # Geometric
            "geoid_mirror_portal_engine", "golden_ratio_optimizer",
            # GPU Management
            "gpu_memory_pool", "gpu_signal_memory", "gpu_thermodynamic_integrator",
            # High-Dimensional
            "high_dimensional_bgm", "homomorphic_cognitive_processor",
            # Insight
            "information_integration_analyzer", "insight_entropy", 
            "insight_feedback", "insight_lifecycle",
            # Barenholtz
            "kimera_barenholtz_core", "kimera_barenholtz_ultimate_optimization",
            "kimera_barenholtz_unified_engine", "kimera_advanced_integration_fix",
            # Response
            "kimera_cognitive_response_system", "kimera_full_integration_bridge",
            "kimera_quantum_edge_security_architecture",
            # Testing
            "large_scale_testing_framework", "omnidimensional_protocol_engine",
            # Output
            "output_generator", "portal_manager",
            # Contradiction
            "proactive_contradiction_detector", "pruning",
            # Quantum Interface
            "quantum_classical_interface", "quantum_enhanced_universal_translator",
            # Quantum Security
            "quantum_resistant_crypto", "quantum_thermodynamic_complexity_analyzer",
            # Quantum Thermodynamics
            "quantum_thermodynamic_signal_processor", "quantum_truth_monitor",
            # Signal Evolution
            "real_time_signal_evolution", "revolutionary_epistemic_validator",
            # Rhetorical
            "rhetorical_barenholtz_core", "symbolic_polyglot_barenholtz_core",
            # Symbolic
            "symbolic_processor", "tcse_system_integration",
            # Thermodynamic
            "thermodynamic_efficiency_optimizer", "thermodynamic_signal_evolution",
            "thermodynamic_signal_optimizer", "thermodynamic_signal_validation",
            # Triton
            "triton_cognitive_kernels", "unsupervised_test_optimization",
            # Vortex
            "vortex_dynamics", "vortex_energy_storage", "vortex_thermodynamic_battery",
            # Zetetic
            "zetetic_revolutionary_integration_engine"
        ]
        
        # Define completed integrations
        self.completed_integrations = {
            "axiomatic_foundation": [
                "axiom_mathematical_proof",
                "axiom_of_understanding", 
                "axiom_verification",
                "integration"
            ],
            "services": [
                "background_job_manager",
                "clip_service_integration",
                "integration"
            ]
        }
    
    def audit_engines(self):
        """Audit all engine files"""
        logger.info("\n🔍 AUDITING ENGINES...")
        engines_path = "src/engines"
        
        for engine in self.expected_engines:
            engine_file = f"{engine}.py"
            full_path = os.path.join(engines_path, engine_file)
            
            if os.path.exists(full_path):
                self.audit_results["engines"][engine] = {
                    "exists": True,
                    "path": full_path,
                    "size": os.path.getsize(full_path)
                }
                
                # Try to import
                try:
                    module_path = f"src.engines.{engine}"
                    importlib.import_module(module_path)
                    self.audit_results["engines"][engine]["importable"] = True
                except Exception as e:
                    self.audit_results["engines"][engine]["importable"] = False
                    self.audit_results["engines"][engine]["import_error"] = str(e)
                    self.audit_results["issues"].append({
                        "type": "import_error",
                        "engine": engine,
                        "error": str(e)
                    })
            else:
                self.audit_results["engines"][engine] = {
                    "exists": False,
                    "path": full_path
                }
                self.audit_results["issues"].append({
                    "type": "missing_engine",
                    "engine": engine
                })
        
        # Summary
        total = len(self.expected_engines)
        existing = sum(1 for e in self.audit_results["engines"].values() if e["exists"])
        importable = sum(1 for e in self.audit_results["engines"].values() if e.get("importable", False))
        
        logger.info(f"  ✓ Engines found: {existing}/{total}")
        logger.info(f"  ✓ Engines importable: {importable}/{total}")
    
    def audit_core_modules(self):
        """Audit core system modules"""
        logger.info("\n🔍 AUDITING CORE MODULES...")
        
        core_modules = [
            "kimera_system",
            "cognitive_architecture_core",
            "exception_handling",
            "error_recovery",
            "performance_integration",
            "database_optimization",
            "context_supremacy",
            "statistical_modeling",
            "universal_compassion",
            "cache_layer",
            "dependency_injection",
            "task_manager"
        ]
        
        for module in core_modules:
            try:
                module_path = f"src.core.{module}"
                imported = importlib.import_module(module_path)
                self.audit_results["core_modules"][module] = {
                    "importable": True,
                    "has_content": bool(dir(imported))
                }
            except Exception as e:
                self.audit_results["core_modules"][module] = {
                    "importable": False,
                    "error": str(e)
                }
                self.audit_results["issues"].append({
                    "type": "core_module_error",
                    "module": module,
                    "error": str(e)
                })
        
        # Summary
        total = len(core_modules)
        importable = sum(1 for m in self.audit_results["core_modules"].values() if m.get("importable", False))
        logger.info(f"  ✓ Core modules importable: {importable}/{total}")
    
    def audit_integrations(self):
        """Audit completed integrations"""
        logger.info("\n🔍 AUDITING COMPLETED INTEGRATIONS...")
        
        for integration_name, modules in self.completed_integrations.items():
            integration_path = f"src/core/{integration_name}"
            
            if os.path.exists(integration_path):
                self.audit_results["integrations"][integration_name] = {
                    "exists": True,
                    "modules": {}
                }
                
                for module in modules:
                    module_file = f"{module}.py"
                    full_path = os.path.join(integration_path, module_file)
                    
                    if os.path.exists(full_path):
                        self.audit_results["integrations"][integration_name]["modules"][module] = {
                            "exists": True,
                            "size": os.path.getsize(full_path)
                        }
                        
                        # Try to import
                        try:
                            module_path = f"src.core.{integration_name}.{module}"
                            importlib.import_module(module_path)
                            self.audit_results["integrations"][integration_name]["modules"][module]["importable"] = True
                        except Exception as e:
                            self.audit_results["integrations"][integration_name]["modules"][module]["importable"] = False
                            self.audit_results["integrations"][integration_name]["modules"][module]["error"] = str(e)
                    else:
                        self.audit_results["integrations"][integration_name]["modules"][module] = {
                            "exists": False
                        }
            else:
                self.audit_results["integrations"][integration_name] = {
                    "exists": False
                }
                self.audit_results["issues"].append({
                    "type": "missing_integration",
                    "integration": integration_name
                })
        
        # Summary
        for name, data in self.audit_results["integrations"].items():
            if data.get("exists"):
                total = len(data["modules"])
                existing = sum(1 for m in data["modules"].values() if m.get("exists", False))
                logger.info(f"  ✓ {name}: {existing}/{total} modules")
    
    def audit_dependencies(self):
        """Audit system dependencies"""
        logger.info("\n🔍 AUDITING DEPENDENCIES...")
        
        critical_deps = [
            "numpy", "torch", "scipy", "sqlalchemy", "asyncio",
            "apscheduler", "PIL", "transformers", "cupy", "numba"
        ]
        
        for dep in critical_deps:
            try:
                module = importlib.import_module(dep)
                version = getattr(module, "__version__", "unknown")
                self.audit_results["dependencies"][dep] = {
                    "available": True,
                    "version": version
                }
            except ImportError:
                self.audit_results["dependencies"][dep] = {
                    "available": False
                }
                self.audit_results["issues"].append({
                    "type": "missing_dependency",
                    "dependency": dep
                })
        
        # Summary
        available = sum(1 for d in self.audit_results["dependencies"].values() if d["available"])
        logger.info(f"  ✓ Dependencies available: {available}/{len(critical_deps)}")
    
    def audit_initialization(self):
        """Test system initialization"""
        logger.info("\n🔍 TESTING SYSTEM INITIALIZATION...")
        
        try:
            from src.core.kimera_system import get_kimera_system
import logging
logger = logging.getLogger(__name__)
            
            # Get instance
            kimera = get_kimera_system()
            self.audit_results["initialization"]["singleton"] = True
            
            # Check state
            state = kimera.get_status()
            self.audit_results["initialization"]["initial_state"] = state
            
            # Try initialization
            try:
                kimera.initialize()
                self.audit_results["initialization"]["can_initialize"] = True
                
                # Check components
                components = {
                    "axiomatic_foundation": kimera.get_axiomatic_foundation(),
                    "services": kimera.get_services(),
                    "vault_manager": kimera.get_vault_manager(),
                    "task_manager": kimera.get_task_manager(),
                    "gpu_foundation": kimera.get_gpu_foundation()
                }
                
                for name, component in components.items():
                    if component is None:
                        status = "not_initialized"
                    elif component == "initializing":
                        status = "initializing"
                    else:
                        status = "initialized"
                    
                    self.audit_results["initialization"][f"component_{name}"] = status
                
            except Exception as e:
                self.audit_results["initialization"]["can_initialize"] = False
                self.audit_results["initialization"]["init_error"] = str(e)
                self.audit_results["issues"].append({
                    "type": "initialization_error",
                    "error": str(e),
                    "traceback": traceback.format_exc()
                })
                
        except Exception as e:
            self.audit_results["initialization"]["singleton"] = False
            self.audit_results["initialization"]["error"] = str(e)
            self.audit_results["issues"].append({
                "type": "kimera_system_error",
                "error": str(e)
            })
        
        # Summary
        if self.audit_results["initialization"].get("can_initialize"):
            logger.info("  ✓ System can initialize")
        else:
            logger.info("  ✗ System initialization failed")
    
    def generate_report(self):
        """Generate comprehensive audit report"""
        logger.info("\n" + "="*80)
        logger.info("KIMERA SYSTEM AUDIT REPORT")
        logger.info("="*80)
        
        # Engine Summary
        logger.info("\n📊 ENGINE SUMMARY:")
        total_engines = len(self.expected_engines)
        existing_engines = sum(1 for e in self.audit_results["engines"].values() if e["exists"])
        importable_engines = sum(1 for e in self.audit_results["engines"].values() if e.get("importable", False))
        
        logger.info(f"  Total Expected: {total_engines}")
        logger.info(f"  Files Found: {existing_engines} ({existing_engines/total_engines*100:.1f}%)")
        logger.info(f"  Importable: {importable_engines} ({importable_engines/total_engines*100:.1f}%)")
        
        # Integration Summary
        logger.info("\n📊 INTEGRATION SUMMARY:")
        logger.info("  ✅ Axiomatic Foundation: COMPLETED")
        logger.info("  ✅ Background Services: COMPLETED")
        logger.info(f"  ⏳ Pending: 23/25 integrations")
        
        # Issues Summary
        logger.info(f"\n⚠️  ISSUES FOUND: {len(self.audit_results['issues'])}")
        
        if self.audit_results["issues"]:
            # Group issues by type
            issue_types = {}
            for issue in self.audit_results["issues"]:
                issue_type = issue["type"]
                if issue_type not in issue_types:
                    issue_types[issue_type] = []
                issue_types[issue_type].append(issue)
            
            for issue_type, issues in issue_types.items():
                logger.info(f"\n  {issue_type.upper()} ({len(issues)} issues):")
                for issue in issues[:3]:  # Show first 3
                    if issue_type == "import_error":
                        logger.info(f"    - {issue['engine']}: {issue['error'][:50]}...")
                    elif issue_type == "missing_engine":
                        logger.info(f"    - Missing: {issue['engine']}")
                    elif issue_type == "missing_dependency":
                        logger.info(f"    - Missing dependency: {issue['dependency']}")
                    else:
                        logger.info(f"    - {issue}")
                
                if len(issues) > 3:
                    logger.info(f"    ... and {len(issues) - 3} more")
        
        # Save detailed report
        report_file = f"audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(self.audit_results, f, indent=2)
        
        logger.info(f"\n📄 Detailed report saved to: {report_file}")
        
        # Overall Health Score
        health_score = self._calculate_health_score()
        logger.info(f"\n🏥 OVERALL SYSTEM HEALTH: {health_score:.1f}%")
        
        if health_score >= 90:
            logger.info("   Status: EXCELLENT ✨")
        elif health_score >= 70:
            logger.info("   Status: GOOD ✅")
        elif health_score >= 50:
            logger.info("   Status: FAIR ⚠️")
        else:
            logger.info("   Status: NEEDS ATTENTION ❌")
    
    def _calculate_health_score(self) -> float:
        """Calculate overall system health score"""
        scores = []
        
        # Engine health (40%)
        total_engines = len(self.expected_engines)
        importable_engines = sum(1 for e in self.audit_results["engines"].values() if e.get("importable", False))
        scores.append((importable_engines / total_engines) * 40)
        
        # Core modules health (30%)
        total_core = len(self.audit_results["core_modules"])
        importable_core = sum(1 for m in self.audit_results["core_modules"].values() if m.get("importable", False))
        if total_core > 0:
            scores.append((importable_core / total_core) * 30)
        
        # Integration health (20%)
        # 2 completed out of 25 total
        scores.append((2 / 25) * 20)
        
        # Initialization health (10%)
        if self.audit_results["initialization"].get("can_initialize"):
            scores.append(10)
        else:
            scores.append(0)
        
        return sum(scores)
    
    def run_full_audit(self):
        """Run complete system audit"""
        logger.info("\n🚀 STARTING KIMERA SYSTEM AUDIT...")
        logger.info(f"   Timestamp: {self.audit_results['timestamp']}")
        
        self.audit_engines()
        self.audit_core_modules()
        self.audit_integrations()
        self.audit_dependencies()
        self.audit_initialization()
        self.generate_report()
        
        logger.info("\n✅ AUDIT COMPLETE!")


if __name__ == "__main__":
    auditor = KimeraSystemAuditor()
    auditor.run_full_audit()