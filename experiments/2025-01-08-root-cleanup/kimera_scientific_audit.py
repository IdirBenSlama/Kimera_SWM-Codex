#!/usr/bin/env python3
"""
Kimera Scientific System Audit
==============================
Deep scientific audit of Kimera's core features:
- PostgreSQL optimization
- Entropy calculations
- Thermodynamic engine
- Quantum behaviors
- Diffusion module
- Portal/Vortex mechanics
- Semantic coherence
- Algorithm optimization
"""

import os
import sys
import json
import time
import asyncio
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import psutil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class KimeraScientificAuditor:
    """Scientific auditor for Kimera's core quantum-thermodynamic features"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.absolute()
        sys.path.insert(0, str(self.project_root))
        
        self.audit_results = {
            "timestamp": datetime.now().isoformat(),
            "postgresql_issues": [],
            "entropy_issues": [],
            "thermodynamic_issues": [],
            "quantum_issues": [],
            "diffusion_issues": [],
            "portal_vortex_issues": [],
            "semantic_issues": [],
            "performance_issues": [],
            "optimization_issues": [],
            "critical_findings": [],
            "measurements": {}
        }
        
    def log_issue(self, category: str, issue: str, severity: str = "warning", data: Dict = None):
        """Log a scientific finding"""
        entry = {
            "issue": issue,
            "severity": severity,
            "timestamp": datetime.now().isoformat(),
            "data": data or {}
        }
        
        if severity == "critical":
            self.audit_results["critical_findings"].append(entry)
            logger.error(f"CRITICAL {category}: {issue}")
        else:
            if category in self.audit_results:
                self.audit_results[category].append(entry)
            logger.warning(f"{category.upper()}: {issue}")
            
    async def audit_postgresql_configuration(self):
        """Audit PostgreSQL configuration and performance"""
        logger.info("\n=== Auditing PostgreSQL Configuration ===")
        
        try:
            # Check current database configuration
            from src.config.settings import get_settings
            settings = get_settings()
            
            db_url = settings.database.url
            if "sqlite" in db_url.lower():
                self.log_issue("postgresql_issues", 
                             "System configured for SQLite instead of PostgreSQL", 
                             severity="critical",
                             data={"current_db": db_url})
                
            # Check database.py configuration
            db_file = self.project_root / "backend" / "vault" / "database.py"
            content = db_file.read_text()
            
            # Check for PostgreSQL optimizations
            optimizations = [
                "pool_size", "max_overflow", "pool_timeout",
                "pool_pre_ping", "pool_recycle"
            ]
            
            missing_opts = []
            for opt in optimizations:
                if opt not in content:
                    missing_opts.append(opt)
                    
            if missing_opts:
                self.log_issue("postgresql_issues",
                             f"Missing PostgreSQL optimizations: {missing_opts}",
                             data={"missing": missing_opts})
                
            # Check for vector extension
            if "pgvector" in content:
                logger.info("✓ pgvector extension configured")
            else:
                self.log_issue("postgresql_issues",
                             "pgvector extension not properly configured",
                             severity="critical")
                
        except Exception as e:
            self.log_issue("postgresql_issues", 
                         f"Failed to audit PostgreSQL: {str(e)}", 
                         severity="critical")
            
    async def audit_entropy_calculations(self):
        """Audit entropy calculation implementations"""
        logger.info("\n=== Auditing Entropy Calculations ===")
        
        try:
            # Import and test thermodynamic engine
            from src.engines.foundational_thermodynamic_engine import FoundationalThermodynamicEngine
            
            engine = FoundationalThermodynamicEngine()
            
            # Test entropy calculations
            test_states = [
                np.random.rand(100),
                np.ones(100),
                np.zeros(100),
                np.linspace(0, 1, 100)
            ]
            
            for i, state in enumerate(test_states):
                try:
                    entropy = engine.calculate_entropy(state)
                    self.audit_results["measurements"][f"entropy_test_{i}"] = float(entropy)
                    
                    # Verify entropy bounds
                    if entropy < 0:
                        self.log_issue("entropy_issues",
                                     f"Negative entropy detected: {entropy}",
                                     severity="critical")
                    elif entropy > np.log(len(state)):
                        self.log_issue("entropy_issues",
                                     f"Entropy exceeds theoretical maximum: {entropy}",
                                     severity="warning")
                                     
                except Exception as e:
                    self.log_issue("entropy_issues",
                                 f"Entropy calculation failed: {str(e)}",
                                 severity="critical")
                                 
            # Check for entropy gradient calculations
            if hasattr(engine, 'calculate_entropy_gradient'):
                logger.info("✓ Entropy gradient calculations available")
            else:
                self.log_issue("entropy_issues",
                             "Missing entropy gradient calculations")
                             
        except ImportError as e:
            self.log_issue("entropy_issues",
                         f"Cannot import thermodynamic engine: {str(e)}",
                         severity="critical")
                         
    async def audit_quantum_behaviors(self):
        """Audit quantum mechanical implementations"""
        logger.info("\n=== Auditing Quantum Behaviors ===")
        
        try:
            # Check for quantum modules
            quantum_modules = [
                "backend.engines.quantum_field_engine",
                "backend.engines.quantum_consciousness_bridge",
                "backend.core.quantum_operators"
            ]
            
            for module_name in quantum_modules:
                try:
                    module = __import__(module_name, fromlist=[''])
                    logger.info(f"✓ Quantum module found: {module_name}")
                    
                    # Check for key quantum functions
                    quantum_functions = [
                        "superposition", "entanglement", "collapse",
                        "decoherence", "measurement", "wave_function"
                    ]
                    
                    for func in quantum_functions:
                        if any(func in str(getattr(module, attr, '')) 
                              for attr in dir(module)):
                            logger.info(f"  ✓ Quantum behavior: {func}")
                        else:
                            self.log_issue("quantum_issues",
                                         f"Missing quantum behavior: {func} in {module_name}")
                                         
                except ImportError:
                    self.log_issue("quantum_issues",
                                 f"Missing quantum module: {module_name}",
                                 severity="warning")
                                 
            # Test quantum field dynamics
            try:
                from src.engines.cognitive_field_dynamics import CognitiveFieldDynamics
                cfd = CognitiveFieldDynamics(dimension=384)  # Standard embedding dimension
                
                # Test quantum coherence calculation
                if hasattr(cfd, 'calculate_quantum_coherence'):
                    test_field = np.random.rand(10, 10)
                    coherence = cfd.calculate_quantum_coherence(test_field)
                    self.audit_results["measurements"]["quantum_coherence"] = float(coherence)
                    
                    if coherence < 0 or coherence > 1:
                        self.log_issue("quantum_issues",
                                     f"Quantum coherence out of bounds: {coherence}",
                                     severity="critical")
                else:
                    self.log_issue("quantum_issues",
                                 "CognitiveFieldDynamics missing quantum coherence calculation")
                                 
            except Exception as e:
                self.log_issue("quantum_issues",
                             f"Quantum field dynamics error: {str(e)}")
                             
        except Exception as e:
            self.log_issue("quantum_issues",
                         f"Quantum audit failed: {str(e)}",
                         severity="critical")
                         
    async def audit_diffusion_module(self):
        """Audit diffusion and field propagation"""
        logger.info("\n=== Auditing Diffusion Module ===")
        
        try:
            # Check for diffusion implementations
            from src.engines.spde_engine import SPDEEngine
            
            spde = SPDEEngine()
            
            # Test diffusion parameters
            if hasattr(spde, 'diffusion_coefficient'):
                diff_coeff = spde.diffusion_coefficient
                if diff_coeff <= 0:
                    self.log_issue("diffusion_issues",
                                 f"Invalid diffusion coefficient: {diff_coeff}",
                                 severity="critical")
                else:
                    self.audit_results["measurements"]["diffusion_coefficient"] = float(diff_coeff)
                    
            # Test field evolution
            test_field = np.random.rand(50, 50)
            try:
                evolved_field = spde.evolve(test_field, dt=0.01, steps=10)
                
                # Check conservation laws
                initial_sum = np.sum(test_field)
                final_sum = np.sum(evolved_field)
                conservation_error = abs(final_sum - initial_sum) / initial_sum
                
                if conservation_error > 0.01:  # 1% tolerance
                    self.log_issue("diffusion_issues",
                                 f"Conservation law violation: {conservation_error:.2%}",
                                 severity="warning")
                                 
            except Exception as e:
                self.log_issue("diffusion_issues",
                             f"Field evolution failed: {str(e)}",
                             severity="critical")
                             
        except ImportError:
            self.log_issue("diffusion_issues",
                         "SPDE engine not available",
                         severity="critical")
                         
    async def audit_portal_vortex_mechanics(self):
        """Audit portal and vortex implementations"""
        logger.info("\n=== Auditing Portal/Vortex Mechanics ===")
        
        try:
            # Look for portal/vortex modules
            portal_files = list(self.project_root.rglob("*portal*.py"))
            vortex_files = list(self.project_root.rglob("*vortex*.py"))
            
            if not portal_files:
                self.log_issue("portal_vortex_issues",
                             "No portal implementation found",
                             severity="critical")
                             
            if not vortex_files:
                self.log_issue("portal_vortex_issues",
                             "No vortex implementation found",
                             severity="critical")
                             
            # Check for portal manager
            try:
                from src.engines.portal_manager import PortalManager
                portal_mgr = PortalManager()
                
                # Test portal creation
                portal_id = portal_mgr.create_portal(
                    source_dim=3,
                    target_dim=4,
                    stability=0.8
                )
                
                if portal_id:
                    logger.info(f"✓ Portal creation successful: {portal_id}")
                else:
                    self.log_issue("portal_vortex_issues",
                                 "Portal creation failed")
                                 
            except ImportError:
                self.log_issue("portal_vortex_issues",
                             "Portal manager not implemented",
                             severity="critical")
                             
            # Check vortex dynamics
            try:
                from src.engines.vortex_dynamics import VortexField
                vortex = VortexField()
                
                # Test vortex properties
                circulation = vortex.calculate_circulation()
                if circulation < 0:
                    self.log_issue("portal_vortex_issues",
                                 f"Negative vortex circulation: {circulation}",
                                 severity="warning")
                                 
            except ImportError:
                self.log_issue("portal_vortex_issues",
                             "Vortex dynamics not implemented",
                             severity="critical")
                             
        except Exception as e:
            self.log_issue("portal_vortex_issues",
                         f"Portal/vortex audit failed: {str(e)}",
                         severity="critical")
                         
    async def audit_semantic_coherence(self):
        """Audit semantic processing and coherence"""
        logger.info("\n=== Auditing Semantic Coherence ===")
        
        try:
            # Test embedding consistency
            from src.core.embedding_utils import get_embedding_model
            
            model = get_embedding_model()
            
            # Test semantic similarity
            test_phrases = [
                ("quantum entanglement", "quantum superposition"),
                ("thermodynamic entropy", "information entropy"),
                ("cognitive resonance", "neural synchronization")
            ]
            
            for phrase1, phrase2 in test_phrases:
                try:
                    emb1 = model.encode(phrase1)
                    emb2 = model.encode(phrase2)
                    
                    # Calculate cosine similarity
                    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                    
                    self.audit_results["measurements"][f"semantic_sim_{phrase1[:10]}_{phrase2[:10]}"] = float(similarity)
                    
                    if similarity < 0.3:
                        self.log_issue("semantic_issues",
                                     f"Low semantic similarity between related concepts: {phrase1} <-> {phrase2}",
                                     severity="warning")
                                     
                except Exception as e:
                    self.log_issue("semantic_issues",
                                 f"Semantic embedding failed: {str(e)}",
                                 severity="critical")
                                 
            # Check semantic vector dimensions
            test_emb = model.encode("test")
            if len(test_emb) != 384:  # Expected dimension
                self.log_issue("semantic_issues",
                             f"Unexpected embedding dimension: {len(test_emb)} (expected 384)",
                             severity="warning")
                             
        except Exception as e:
            self.log_issue("semantic_issues",
                         f"Semantic audit failed: {str(e)}",
                         severity="critical")
                         
    async def audit_performance_optimization(self):
        """Audit system performance and optimization"""
        logger.info("\n=== Auditing Performance & Optimization ===")
        
        try:
            # Measure latency for key operations
            operations = {
                "geoid_creation": self._measure_geoid_creation,
                "entropy_calculation": self._measure_entropy_calculation,
                "embedding_generation": self._measure_embedding_generation,
                "database_query": self._measure_database_query
            }
            
            for op_name, op_func in operations.items():
                try:
                    latency = await op_func()
                    self.audit_results["measurements"][f"{op_name}_latency_ms"] = latency
                    
                    # Check against thresholds
                    if latency > 100:  # 100ms threshold
                        self.log_issue("performance_issues",
                                     f"High latency for {op_name}: {latency:.2f}ms",
                                     severity="warning" if latency < 500 else "critical")
                                     
                except Exception as e:
                    self.log_issue("performance_issues",
                                 f"Failed to measure {op_name}: {str(e)}")
                                 
            # Check GPU utilization
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
                    gpu_utilization = torch.cuda.utilization()
                    
                    self.audit_results["measurements"]["gpu_memory_gb"] = gpu_memory
                    self.audit_results["measurements"]["gpu_utilization"] = gpu_utilization
                    
                    if gpu_memory > 8:  # 8GB threshold
                        self.log_issue("performance_issues",
                                     f"High GPU memory usage: {gpu_memory:.2f}GB",
                                     severity="warning")
                                     
            except ImportError:
                self.log_issue("performance_issues",
                             "PyTorch not available for GPU monitoring")
                             
        except Exception as e:
            self.log_issue("performance_issues",
                         f"Performance audit failed: {str(e)}",
                         severity="critical")
                         
    async def _measure_geoid_creation(self) -> float:
        """Measure geoid creation latency"""
        from src.engines.geoid_scar_manager import GeoidScarManager
        manager = GeoidScarManager()
        
        start = time.time()
        await manager.create_geoid({"test": "data"})
        return (time.time() - start) * 1000
        
    async def _measure_entropy_calculation(self) -> float:
        """Measure entropy calculation latency"""
        from src.engines.foundational_thermodynamic_engine import FoundationalThermodynamicEngine
        engine = FoundationalThermodynamicEngine()
        
        test_data = np.random.rand(1000)
        start = time.time()
        engine.calculate_entropy(test_data)
        return (time.time() - start) * 1000
        
    async def _measure_embedding_generation(self) -> float:
        """Measure embedding generation latency"""
        from src.core.embedding_utils import get_embedding_model
        model = get_embedding_model()
        
        start = time.time()
        model.encode("test phrase for latency measurement")
        return (time.time() - start) * 1000
        
    async def _measure_database_query(self) -> float:
        """Measure database query latency"""
        from src.vault.database import engine
        
        start = time.time()
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        return (time.time() - start) * 1000
        
    def generate_report(self):
        """Generate comprehensive scientific audit report"""
        
        # Calculate totals
        total_issues = sum(len(v) for k, v in self.audit_results.items() 
                          if k.endswith("_issues") and isinstance(v, list))
        critical_count = len(self.audit_results["critical_findings"])
        
        # Save detailed report
        report_file = f"kimera_scientific_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(self.audit_results, f, indent=2)
            
        # Print summary
        print("\n" + "="*70)
        print("KIMERA SCIENTIFIC AUDIT SUMMARY")
        print("="*70)
        print(f"Total Issues Found: {total_issues}")
        print(f"Critical Issues: {critical_count}")
        print("\nIssue Breakdown:")
        
        for category in ["postgresql", "entropy", "thermodynamic", "quantum", 
                        "diffusion", "portal_vortex", "semantic", "performance"]:
            key = f"{category}_issues"
            if key in self.audit_results:
                count = len(self.audit_results[key])
                if count > 0:
                    print(f"  - {category.replace('_', ' ').title()}: {count}")
                    
        print("\nKey Measurements:")
        for key, value in self.audit_results["measurements"].items():
            if isinstance(value, (int, float)):
                print(f"  - {key}: {value:.4f}")
                
        print("="*70)
        print(f"\nDetailed report saved to: {report_file}")
        
        return self.audit_results
        
    async def run_audit(self):
        """Run complete scientific audit"""
        logger.info("Starting Kimera Scientific System Audit...")
        
        try:
            await self.audit_postgresql_configuration()
            await self.audit_entropy_calculations()
            await self.audit_quantum_behaviors()
            await self.audit_diffusion_module()
            await self.audit_portal_vortex_mechanics()
            await self.audit_semantic_coherence()
            await self.audit_performance_optimization()
            
            return self.generate_report()
            
        except Exception as e:
            logger.error(f"Audit failed: {str(e)}")
            raise


if __name__ == "__main__":
    auditor = KimeraScientificAuditor()
    results = asyncio.run(auditor.run_audit()) 