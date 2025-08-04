#!/usr/bin/env python3
"""
Kimera Final Scientific Test
============================
Comprehensive test of all Kimera systems after optimization.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class KimeraScientificTester:
    def __init__(self):
        self.project_root = Path(__file__).parent.absolute()
        sys.path.insert(0, str(self.project_root))

        self.test_results = {
            "timestamp": datetime.now().isoformat(),
            "tests_passed": 0,
            "tests_failed": 0,
            "components": {},
            "scientific_metrics": {},
            "performance_metrics": {},
        }

    def test_postgresql_connection(self):
        """Test PostgreSQL connection and optimization"""
        logger.info("\n=== Testing PostgreSQL Connection ===")

        try:
            from src.vault.database import engine

            # Test connection
            with engine.connect() as conn:
                result = conn.execute("SELECT version()")
                version = result.scalar()
                logger.info(f"✓ PostgreSQL connected: {version}")

                # Check pgvector extension
                result = conn.execute(
                    "SELECT extname FROM pg_extension WHERE extname = 'vector'"
                )
                if result.fetchone():
                    logger.info("✓ pgvector extension installed")
                    self.test_results["components"]["postgresql"] = "operational"
                    self.test_results["tests_passed"] += 1
                else:
                    logger.warning("⚠ pgvector extension not found")
                    self.test_results["components"]["postgresql"] = "partial"

        except Exception as e:
            logger.error(f"✗ PostgreSQL connection failed: {e}")
            self.test_results["components"]["postgresql"] = "failed"
            self.test_results["tests_failed"] += 1

    def test_thermodynamic_physics(self):
        """Test thermodynamic physics compliance"""
        logger.info("\n=== Testing Thermodynamic Physics ===")

        try:
            from src.engines.foundational_thermodynamic_engine import (
                FoundationalThermodynamicEngine,
            )

            engine = FoundationalThermodynamicEngine()

            # Test 1: Entropy calculation
            test_data = np.random.rand(1000)
            entropy = engine.calculate_entropy(test_data)

            if entropy >= 0:
                logger.info(f"✓ Entropy calculation valid: {entropy:.4f}")
                self.test_results["scientific_metrics"]["entropy"] = float(entropy)
            else:
                logger.error(f"✗ Negative entropy detected: {entropy}")

            # Test 2: Carnot efficiency
            hot_temp = 500.0  # K
            cold_temp = 300.0  # K

            hot_fields = [np.random.rand(100) * hot_temp / 100 for _ in range(5)]
            cold_fields = [np.random.rand(100) * cold_temp / 100 for _ in range(5)]

            cycle = engine.run_zetetic_carnot_engine(hot_fields, cold_fields)

            theoretical_max = 1 - (cold_temp / hot_temp)

            logger.info(
                f"Hot temperature: {cycle.hot_temperature.physical_temperature:.2f} K"
            )
            logger.info(
                f"Cold temperature: {cycle.cold_temperature.physical_temperature:.2f} K"
            )
            logger.info(f"Theoretical efficiency: {cycle.theoretical_efficiency:.4f}")
            logger.info(f"Actual efficiency: {cycle.actual_efficiency:.4f}")
            logger.info(f"Physics compliant: {cycle.physics_compliant}")

            self.test_results["scientific_metrics"]["carnot_efficiency"] = float(
                cycle.actual_efficiency
            )
            self.test_results["scientific_metrics"][
                "physics_compliant"
            ] = cycle.physics_compliant

            if cycle.physics_compliant:
                logger.info("✓ Thermodynamics physics compliant")
                self.test_results["tests_passed"] += 1
            else:
                logger.error("✗ Thermodynamics physics violation")
                self.test_results["tests_failed"] += 1

            self.test_results["components"]["thermodynamics"] = (
                "operational" if cycle.physics_compliant else "violated"
            )

        except Exception as e:
            logger.error(f"✗ Thermodynamic test failed: {e}")
            self.test_results["components"]["thermodynamics"] = "failed"
            self.test_results["tests_failed"] += 1

    def test_quantum_mechanics(self):
        """Test quantum mechanical implementations"""
        logger.info("\n=== Testing Quantum Mechanics ===")

        try:
            from src.engines.quantum_field_engine import QuantumFieldEngine

            qfe = QuantumFieldEngine(dimension=8)

            # Test 1: Superposition
            state1 = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=complex)
            state2 = np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=complex)

            superposed = qfe.create_superposition(
                [state1, state2], [1 / np.sqrt(2), 1 / np.sqrt(2)]
            )

            logger.info(f"✓ Superposition created")
            logger.info(f"  Coherence: {superposed.coherence:.4f}")
            logger.info(f"  Purity: {superposed.purity:.4f}")

            # Test 2: Entanglement
            entangled = qfe.create_entangled_state([2, 2])

            logger.info(f"✓ Entangled state created")
            logger.info(f"  Entanglement entropy: {entangled.entanglement_measure:.4f}")

            # Test 3: Measurement
            observable = (
                qfe.operators["sigma_z"]
                if "sigma_z" in qfe.operators
                else np.array([[1, 0], [0, -1]])
            )

            # For 2-qubit measurement, need to adjust
            if entangled.dimension == 4:
                # Create 4x4 observable for 2-qubit system
                observable = np.kron(observable, np.eye(2))

            outcome, collapsed = qfe.apply_measurement(entangled, observable)

            logger.info(f"✓ Measurement performed")
            logger.info(f"  Outcome: {outcome:.4f}")
            logger.info(f"  Post-measurement purity: {collapsed.purity:.4f}")

            self.test_results["scientific_metrics"]["quantum_coherence"] = float(
                superposed.coherence
            )
            self.test_results["scientific_metrics"]["entanglement_entropy"] = float(
                entangled.entanglement_measure
            )
            self.test_results["components"]["quantum"] = "operational"
            self.test_results["tests_passed"] += 1

        except Exception as e:
            logger.error(f"✗ Quantum test failed: {e}")
            self.test_results["components"]["quantum"] = "failed"
            self.test_results["tests_failed"] += 1

    def test_diffusion_dynamics(self):
        """Test SPDE diffusion dynamics"""
        logger.info("\n=== Testing Diffusion Dynamics ===")

        try:
            from src.engines.spde_engine import SPDEEngine

            device = "cuda" if torch.cuda.is_available() else "cpu"
            spde = SPDEEngine(device=device)

            # Create test field
            size = 32
            test_field = np.exp(
                -(
                    (np.linspace(-1, 1, size)[:, None]) ** 2
                    + (np.linspace(-1, 1, size)[None, :]) ** 2
                )
                / 0.1
            )

            # Evolve field
            start = time.time()
            evolved = spde.evolve(test_field, dt=0.001, steps=50)
            evolution_time = time.time() - start

            # Check conservation
            initial_integral = np.sum(test_field)
            final_integral = np.sum(evolved)
            conservation_error = (
                abs(final_integral - initial_integral) / initial_integral
            )

            logger.info(f"✓ Field evolution completed")
            logger.info(f"  Evolution time: {evolution_time:.4f}s")
            logger.info(f"  Conservation error: {conservation_error:.6f}")
            logger.info(f"  Device: {device}")

            self.test_results["scientific_metrics"]["diffusion_conservation"] = float(
                conservation_error
            )
            self.test_results["performance_metrics"]["diffusion_time"] = float(
                evolution_time
            )

            if conservation_error < 0.01:
                logger.info("✓ Conservation laws satisfied")
                self.test_results["components"]["diffusion"] = "operational"
                self.test_results["tests_passed"] += 1
            else:
                logger.warning(f"⚠ Conservation violation: {conservation_error:.2%}")
                self.test_results["components"]["diffusion"] = "violated"
                self.test_results["tests_failed"] += 1

        except Exception as e:
            logger.error(f"✗ Diffusion test failed: {e}")
            self.test_results["components"]["diffusion"] = "failed"
            self.test_results["tests_failed"] += 1

    def test_portal_vortex_mechanics(self):
        """Test portal and vortex implementations"""
        logger.info("\n=== Testing Portal/Vortex Mechanics ===")

        try:
            # Test Portal Manager
            from src.engines.portal_manager import PortalManager

            pm = PortalManager(max_portals=20)

            # Create portal network
            portal_ids = []
            for i in range(5):
                pid = pm.create_portal(i, i + 1, initial_stability=0.95)
                portal_ids.append(pid)

            # Test traversal
            cargo = {"semantic_vector": np.random.rand(100)}
            success, transformed, info = pm.traverse_portal(
                portal_ids[0], cargo, energy_available=5.0
            )

            logger.info(f"✓ Portal traversal: {success}")
            logger.info(f"  Energy consumed: {info.get('energy_consumed', 0):.4f}")

            # Test Vortex Dynamics
            from src.engines.vortex_dynamics import VortexDynamicsEngine

            vde = VortexDynamicsEngine(grid_size=32)

            # Create vortices
            for _ in range(3):
                vde.create_vortex(
                    position=(np.random.rand() * 10, np.random.rand() * 10),
                    circulation=np.random.randn(),
                )

            # Generate field state
            vortex_state = vde.generate_vortex_field_state()

            logger.info(f"✓ Vortex field generated")
            logger.info(f"  Total circulation: {vde.calculate_circulation():.4f}")
            logger.info(f"  Field energy: {vortex_state.energy:.4f}")
            logger.info(f"  Enstrophy: {vortex_state.enstrophy:.4f}")

            self.test_results["scientific_metrics"]["total_circulation"] = float(
                vde.calculate_circulation()
            )
            self.test_results["scientific_metrics"]["vortex_energy"] = float(
                vortex_state.energy
            )
            self.test_results["components"]["portal_vortex"] = "operational"
            self.test_results["tests_passed"] += 1

        except Exception as e:
            logger.error(f"✗ Portal/Vortex test failed: {e}")
            self.test_results["components"]["portal_vortex"] = "failed"
            self.test_results["tests_failed"] += 1

    def test_semantic_coherence(self):
        """Test semantic processing coherence"""
        logger.info("\n=== Testing Semantic Coherence ===")

        try:
            from sklearn.metrics.pairwise import cosine_similarity

            from src.core.embedding_utils import encode_batch, encode_text

            # Test semantic relationships
            test_pairs = [
                ("quantum entanglement", "quantum superposition"),
                ("thermodynamic entropy", "information theory"),
                ("cognitive emergence", "consciousness"),
                ("portal mechanics", "dimensional transitions"),
            ]

            similarities = []
            for text1, text2 in test_pairs:
                emb1 = encode_text(text1)
                emb2 = encode_text(text2)

                # Handle different return types
                if hasattr(emb1, "reshape"):
                    emb1 = emb1.reshape(1, -1)
                    emb2 = emb2.reshape(1, -1)
                else:
                    emb1 = np.array(emb1).reshape(1, -1)
                    emb2 = np.array(emb2).reshape(1, -1)

                sim = cosine_similarity(emb1, emb2)[0, 0]
                similarities.append(sim)
                logger.info(f"  '{text1}' <-> '{text2}': {sim:.4f}")

            avg_similarity = np.mean(similarities)
            logger.info(f"✓ Average semantic similarity: {avg_similarity:.4f}")

            self.test_results["scientific_metrics"]["semantic_coherence"] = float(
                avg_similarity
            )
            self.test_results["components"]["semantics"] = "operational"
            self.test_results["tests_passed"] += 1

        except Exception as e:
            logger.error(f"✗ Semantic test failed: {e}")
            self.test_results["components"]["semantics"] = "failed"
            self.test_results["tests_failed"] += 1

    def test_system_integration(self):
        """Test full system integration"""
        logger.info("\n=== Testing System Integration ===")

        try:
            from src.core.kimera_system import KimeraSystem

            # Initialize system
            kimera = KimeraSystem()
            kimera.initialize()  # Ensure initialization

            # Get proper status
            status = kimera.get_system_status()

            logger.info(f"✓ KimeraSystem initialized")
            logger.info(
                f"  State: {status['state'].name if hasattr(status['state'], 'name') else status['state']}"
            )

            # Check components
            ready_components = [k for k, v in status["components"].items() if v]
            logger.info(
                f"  Ready components: {len(ready_components)}/{len(status['components'])}"
            )

            for comp, ready in status["components"].items():
                if ready:
                    logger.info(f"    ✓ {comp}")
                else:
                    logger.warning(f"    ✗ {comp}")

            self.test_results["components"]["integration"] = (
                "operational" if len(ready_components) > 5 else "partial"
            )

            if len(ready_components) > 5:
                self.test_results["tests_passed"] += 1
            else:
                self.test_results["tests_failed"] += 1

        except Exception as e:
            logger.error(f"✗ Integration test failed: {e}")
            self.test_results["components"]["integration"] = "failed"
            self.test_results["tests_failed"] += 1

    def generate_report(self):
        """Generate final test report"""

        # Calculate success rate
        total_tests = (
            self.test_results["tests_passed"] + self.test_results["tests_failed"]
        )
        success_rate = (
            (self.test_results["tests_passed"] / total_tests * 100)
            if total_tests > 0
            else 0
        )

        self.test_results["success_rate"] = success_rate
        self.test_results["gpu_available"] = torch.cuda.is_available()

        if torch.cuda.is_available():
            self.test_results["gpu_name"] = torch.cuda.get_device_name(0)
            self.test_results["gpu_memory_gb"] = (
                torch.cuda.get_device_properties(0).total_memory / 1024**3
            )

        # Save report
        report_file = f"kimera_scientific_test_{int(time.time())}.json"
        with open(report_file, "w") as f:
            json.dump(self.test_results, f, indent=2, default=str)

        # Print summary
        print("\n" + "=" * 70)
        print("KIMERA SCIENTIFIC TEST SUMMARY")
        print("=" * 70)
        print(
            f"Success Rate: {success_rate:.1f}% ({self.test_results['tests_passed']}/{total_tests})"
        )
        print(f"GPU: {self.test_results.get('gpu_name', 'Not available')}")

        print("\nComponent Status:")
        for comp, status in self.test_results["components"].items():
            symbol = "✓" if status == "operational" else "✗"
            print(f"  {symbol} {comp.upper()}: {status}")

        print("\nScientific Metrics:")
        for metric, value in self.test_results["scientific_metrics"].items():
            if isinstance(value, bool):
                print(f"  - {metric}: {value}")
            elif isinstance(value, (int, float)):
                print(f"  - {metric}: {value:.6f}")
            else:
                print(f"  - {metric}: {value}")

        print("=" * 70)
        print(f"\nDetailed report saved to: {report_file}")

        return self.test_results

    def run_all_tests(self):
        """Run all scientific tests"""
        logger.info("Starting Kimera Scientific Test Suite...")

        # Run tests in order
        self.test_postgresql_connection()
        self.test_thermodynamic_physics()
        self.test_quantum_mechanics()
        self.test_diffusion_dynamics()
        self.test_portal_vortex_mechanics()
        self.test_semantic_coherence()
        self.test_system_integration()

        return self.generate_report()


if __name__ == "__main__":
    tester = KimeraScientificTester()
    report = tester.run_all_tests()
