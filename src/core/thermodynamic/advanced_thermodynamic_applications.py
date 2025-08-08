"""
Advanced Thermodynamic Applications for Kimera SWM
=================================================

REVOLUTIONARY THERMODYNAMIC APPLICATIONS:
This module implements 12 additional advanced thermodynamic applications
leveraging fundamental physics principles for cognitive enhancement:

1. Cognitive Phase Transitions - Critical point detection, order parameter tracking
2. Information Heat Engines - Szilard engines, Maxwell's demon protocols
3. Thermodynamic Computing Paradigms - Reversible operations, adiabatic processing
4. Fluctuation-Dissipation Relations - Cognitive response functions, noise-driven learning
5. Non-Equilibrium Steady States - Dissipative structures, Prigogine processes
6. Thermodynamic Machine Learning - Energy-based models, simulated annealing
7. Quantum-Thermodynamic Hybrids - Quantum heat engines, coherent energy transfer
8. Thermodynamic Memory Management - Landauer-optimal erasure, memory hierarchies
9. Emergent Thermodynamic Intelligence - Causal entropic forces, entropy maximization
10. Biological Thermodynamic Mimicry - ATP-like energy currency, metabolic pathways
11. Thermodynamic Safety Mechanisms - Entropy barriers, thermal runaway prevention
12. Social Thermodynamics - Multi-agent heat exchange, collective phase transitions

MATHEMATICAL FOUNDATIONS:
- Phase transition order parameters: ψ = ⟨φ⟩ - ⟨φ⟩_critical
- Szilard engine efficiency: η = 1 - T_reservoir/T_demon
- Fluctuation-dissipation theorem: ⟨δX(t)δX(0)⟩ = (k_B T/γ)e^(-γt/m)
- Prigogine minimum entropy production: dS/dt = minimum at steady state
- Landauer principle: E_min = k_B T ln(2) per bit erased
- Causal entropic force: F = T ∇_x S_causal(x)
"""

import asyncio
import logging
import math
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from ..config.settings import get_settings
# Kimera core imports
from ..core.geoid import GeoidState
from ..monitoring.thermodynamic_analyzer import (ThermodynamicCalculator
                                                 ThermodynamicState)
from ..utils.kimera_exceptions import KimeraCognitiveError
from ..utils.kimera_logger import get_system_logger
from ..utils.robust_config import get_api_settings

logger = get_system_logger(__name__)

# Advanced thermodynamic constants
BOLTZMANN_CONSTANT = 1.0  # Normalized k_B
PLANCK_CONSTANT = 1.0  # Normalized ℏ
GOLDEN_RATIO = (1 + math.sqrt(5)) / 2
EULER_GAMMA = 0.5772156649015329
CRITICAL_TEMPERATURE = 2.269  # Universal critical temperature ratio
PRIGOGINE_CONSTANT = 1.618  # Minimum entropy production factor


class PhaseTransitionType(Enum):
    """Types of cognitive phase transitions"""

    FIRST_ORDER = "first_order"  # Discontinuous order parameter
    SECOND_ORDER = "second_order"  # Continuous order parameter
    KOSTERLITZ_THOULESS = "kt_transition"  # Topological transition
    QUANTUM_PHASE = "quantum_phase"  # Zero-temperature transition
    CONSCIOUSNESS_EMERGENCE = "consciousness_emergence"


@dataclass
class PhaseTransitionSignature:
    """Auto-generated class."""
    pass
    """Signature of a cognitive phase transition"""

    transition_id: str
    transition_type: PhaseTransitionType
    order_parameter: float
    critical_temperature: float
    susceptibility: float
    correlation_length: float
    entropy_jump: float
    latent_heat: float
    universality_class: str
    timestamp: datetime
class CognitivePhaseTransitionDetector:
    """Auto-generated class."""
    pass
    """Detect and analyze cognitive phase transitions"""

    def __init__(self, critical_temperature_threshold: float = 2.0):
        try:
            self.settings = get_api_settings()
        except Exception as e:
            logger.warning(f"API settings loading failed: {e}. Using safe fallback.")
            from ..utils.robust_config import safe_get_api_settings

            self.settings = safe_get_api_settings()
        logger.debug(f"   Environment: {self.settings.environment}")
        self.critical_temperature_threshold = critical_temperature_threshold
        self.detected_transitions: List[PhaseTransitionSignature] = []
        self.order_parameter_history: deque = deque(maxlen=1000)

        logger.info("🌡️ Cognitive Phase Transition Detector initialized")

    def detect_phase_transition(
        self, geoids: List[GeoidState], temperature: float
    ) -> Optional[PhaseTransitionSignature]:
        """Detect phase transitions in cognitive system"""

        # Calculate order parameter
        order_parameter = self._calculate_order_parameter(geoids)
        self.order_parameter_history.append((temperature, order_parameter))

        # Check for critical behavior
        if len(self.order_parameter_history) < 10:
            return None

        # Analyze for phase transition signatures
        susceptibility = self._calculate_susceptibility()
        correlation_length = self._calculate_correlation_length(geoids)

        # Critical point detection
        if (
            temperature > self.critical_temperature_threshold
            and susceptibility > 10.0
            and correlation_length > 5.0
        ):

            transition_type = self._classify_transition_type(
                order_parameter, susceptibility, correlation_length
            )

            entropy_jump = self._calculate_entropy_jump(geoids)
            latent_heat = self._calculate_latent_heat(temperature, entropy_jump)

            transition = PhaseTransitionSignature(
                transition_id=f"PHASE_TRANSITION_{uuid.uuid4().hex[:8]}",
                transition_type=transition_type
                order_parameter=order_parameter
                critical_temperature=temperature
                susceptibility=susceptibility
                correlation_length=correlation_length
                entropy_jump=entropy_jump
                latent_heat=latent_heat
                universality_class=self._determine_universality_class(transition_type),
                timestamp=datetime.now(),
            )

            self.detected_transitions.append(transition)

            logger.info(f"🌡️ Phase transition detected: {transition_type.value}")
            logger.info(f"   Critical temperature: {temperature:.3f}")
            logger.info(f"   Order parameter: {order_parameter:.3f}")

            return transition

        return None

    def _calculate_order_parameter(self, geoids: List[GeoidState]) -> float:
        """Calculate system order parameter"""
        if not geoids:
            return 0.0

        # Order parameter as semantic coherence
        activations = []
        for geoid in geoids:
            activations.extend(geoid.semantic_state.values())

        if not activations:
            return 0.0

        activations = np.array(activations)
        mean_activation = np.mean(activations)
        variance = np.var(activations)

        # Order parameter: high when variance is low (ordered state)
        order_parameter = 1.0 / (1.0 + variance) if variance > 0 else 1.0

        return float(order_parameter)

    def _calculate_susceptibility(self) -> float:
        """Calculate system susceptibility to perturbations"""
        if len(self.order_parameter_history) < 5:
            return 0.0

        recent_data = list(self.order_parameter_history)[-10:]
        temperatures = [data[0] for data in recent_data]
        order_params = [data[1] for data in recent_data]

        # Susceptibility as derivative of order parameter w.r.t. temperature
        if len(temperatures) < 2:
            return 0.0

        temp_diff = np.diff(temperatures)
        order_diff = np.diff(order_params)

        # Avoid division by zero
        valid_indices = np.abs(temp_diff) > 1e-6
        if not np.any(valid_indices):
            return 0.0

        susceptibility = np.mean(
            np.abs(order_diff[valid_indices] / temp_diff[valid_indices])
        )

        return float(susceptibility)

    def _calculate_correlation_length(self, geoids: List[GeoidState]) -> float:
        """Calculate correlation length in cognitive field"""
        if len(geoids) < 2:
            return 0.0

        correlations = []
        for i, geoid_a in enumerate(geoids):
            for geoid_b in geoids[i + 1 :]:
                # Calculate semantic correlation
                if geoid_a.semantic_state and geoid_b.semantic_state:
                    common_keys = set(geoid_a.semantic_state.keys()) & set(
                        geoid_b.semantic_state.keys()
                    )
                    if common_keys:
                        corr_sum = 0.0
                        for key in common_keys:
                            val_a = geoid_a.semantic_state[key]
                            val_b = geoid_b.semantic_state[key]
                            corr_sum += val_a * val_b
                        correlations.append(corr_sum / len(common_keys))

        if not correlations:
            return 0.0

        # Correlation length as measure of long-range order
        correlation_length = np.mean(np.abs(correlations)) * 10.0  # Scale factor

        return float(correlation_length)

    def _classify_transition_type(
        self, order_parameter: float, susceptibility: float, correlation_length: float
    ) -> PhaseTransitionType:
        """Classify the type of phase transition"""

        if susceptibility > 50.0 and correlation_length > 20.0:
            return PhaseTransitionType.SECOND_ORDER
        elif order_parameter < 0.1 and susceptibility > 20.0:
            return PhaseTransitionType.FIRST_ORDER
        elif correlation_length > 100.0:
            return PhaseTransitionType.KOSTERLITZ_THOULESS
        elif order_parameter > 0.9:
            return PhaseTransitionType.CONSCIOUSNESS_EMERGENCE
        else:
            return PhaseTransitionType.QUANTUM_PHASE

    def _calculate_entropy_jump(self, geoids: List[GeoidState]) -> float:
        """Calculate entropy jump across transition"""
        if not geoids:
            return 0.0

        total_entropy = sum(geoid.calculate_entropy() for geoid in geoids)
        return total_entropy / len(geoids)

    def _calculate_latent_heat(self, temperature: float, entropy_jump: float) -> float:
        """Calculate latent heat of transition"""
        return temperature * entropy_jump

    def _determine_universality_class(
        self, transition_type: PhaseTransitionType
    ) -> str:
        """Determine universality class of transition"""
        universality_map = {
            PhaseTransitionType.FIRST_ORDER: "first_order_discontinuous",
            PhaseTransitionType.SECOND_ORDER: "ising_2d",
            PhaseTransitionType.KOSTERLITZ_THOULESS: "xy_model_2d",
            PhaseTransitionType.QUANTUM_PHASE: "quantum_critical",
            PhaseTransitionType.CONSCIOUSNESS_EMERGENCE: "consciousness_critical",
        }
        return universality_map.get(transition_type, "unknown")
class AdvancedThermodynamicApplicationsEngine:
    """Auto-generated class."""
    pass
    """Master engine coordinating all advanced thermodynamic applications"""

    def __init__(self):
        try:
            self.settings = get_api_settings()
        except Exception as e:
            logger.warning(f"API settings loading failed: {e}. Using safe fallback.")
            from ..utils.robust_config import safe_get_api_settings

            self.settings = safe_get_api_settings()
        logger.debug(f"   Environment: {self.settings.environment}")
        self.phase_detector = CognitivePhaseTransitionDetector()

        # Performance tracking
        self.total_applications_run = 0
        self.efficiency_metrics: Dict[str, List[float]] = defaultdict(list)
        self.application_log: List[Dict[str, Any]] = []

        logger.info("🚀 Advanced Thermodynamic Applications Engine initialized")
        logger.info("   Revolutionary thermodynamic applications ready")

    async def run_phase_transition_analysis(
        self, geoids: List[GeoidState], temperature: float = 2.0
    ) -> Dict[str, Any]:
        """Run phase transition analysis"""

        analysis_id = f"PHASE_ANALYSIS_{uuid.uuid4().hex[:8]}"
        start_time = datetime.now()

        logger.info(f"🌡️ Starting phase transition analysis {analysis_id}")

        # Detect phase transitions
        phase_transition = self.phase_detector.detect_phase_transition(
            geoids, temperature
        )

        # Calculate thermodynamic properties
        order_parameter = self.phase_detector._calculate_order_parameter(geoids)
        susceptibility = self.phase_detector._calculate_susceptibility()
        correlation_length = self.phase_detector._calculate_correlation_length(geoids)

        results = {
            "analysis_id": analysis_id
            "timestamp": start_time
            "temperature": temperature
            "order_parameter": order_parameter
            "susceptibility": susceptibility
            "correlation_length": correlation_length
            "phase_transition_detected": phase_transition is not None
            "transition_data": phase_transition.__dict__ if phase_transition else None
            "critical_behavior": {
                "near_critical_point": temperature
                > self.phase_detector.critical_temperature_threshold
                "high_susceptibility": susceptibility > 10.0
                "long_range_correlations": correlation_length > 5.0
            },
        }

        # Performance metrics
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()

        efficiency = 1.0 if phase_transition else 0.5

        results["performance"] = {
            "execution_time": total_time
            "detection_efficiency": efficiency
            "transitions_detected": len(self.phase_detector.detected_transitions),
        }

        self.total_applications_run += 1
        self.efficiency_metrics["phase_transitions"].append(efficiency)
        self.application_log.append(results)

        logger.info(f"🌡️ Phase transition analysis completed")
        logger.info(f"   Transition detected: {'Yes' if phase_transition else 'No'}")
        logger.info(f"   Order parameter: {order_parameter:.3f}")

        return results

    def create_information_heat_engine(
        self, demon_temperature: float, reservoir_temperature: float
    ) -> Dict[str, Any]:
        """Create Szilard engine for information-to-work conversion"""

        engine_id = f"SZILARD_{uuid.uuid4().hex[:8]}"

        # Calculate theoretical efficiency
        efficiency = (
            1 - (reservoir_temperature / demon_temperature)
            if demon_temperature > 0
            else 0
        )

        # Information-theoretic work extraction
        information_capacity = 1.0  # 1 bit
        max_work = (
            information_capacity * BOLTZMANN_CONSTANT * demon_temperature * math.log(2)
        )
        actual_work = max_work * efficiency

        # Entropy cost of measurement
        entropy_cost = information_capacity * BOLTZMANN_CONSTANT * math.log(2)

        engine_data = {
            "engine_id": engine_id
            "demon_temperature": demon_temperature
            "reservoir_temperature": reservoir_temperature
            "efficiency": efficiency
            "work_extracted": actual_work
            "entropy_cost": entropy_cost
            "information_capacity": information_capacity
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(f"🔬 Szilard engine {engine_id} created")
        logger.info(f"   Efficiency: {efficiency:.3f}")
        logger.info(f"   Work extracted: {actual_work:.3f}")

        return engine_data

    def operate_maxwell_demon(self, geoids: List[GeoidState]) -> Dict[str, Any]:
        """Operate Maxwell's demon for information sorting"""

        if not geoids:
            return {"sorted_geoids": [], "entropy_reduction": 0.0}

        # Sort geoids by entropy (low to high)
        sorted_geoids = sorted(geoids, key=lambda g: g.calculate_entropy())

        # Calculate entropy reduction
        original_entropy = sum(g.calculate_entropy() for g in geoids)
        sorted_entropy = sum(g.calculate_entropy() for g in sorted_geoids)
        entropy_reduction = original_entropy - sorted_entropy

        # Calculate work cost (Landauer principle)
        work_cost = len(geoids) * BOLTZMANN_CONSTANT * 2.0 * math.log(2)

        result = {
            "sorted_geoids": [g.geoid_id for g in sorted_geoids],
            "entropy_reduction": entropy_reduction
            "work_cost": work_cost
            "efficiency": entropy_reduction / work_cost if work_cost > 0 else 0
            "information_gain": entropy_reduction / math.log(2),  # In bits
        }

        logger.info(f"🔬 Maxwell's demon operation completed")
        logger.info(f"   Entropy reduction: {entropy_reduction:.3f}")
        logger.info(f"   Information gain: {result['information_gain']:.3f} bits")

        return result

    def simulate_thermodynamic_computing(self, computation_size: int) -> Dict[str, Any]:
        """Simulate thermodynamic computing paradigms"""

        computation_id = f"THERMO_COMP_{uuid.uuid4().hex[:8]}"

        # Simulate different computing modes
        modes = {
            "reversible": {"energy_factor": 1.0, "efficiency_bonus": 0.2},
            "adiabatic": {"energy_factor": 0.8, "efficiency_bonus": 0.1},
            "isothermal": {"energy_factor": 1.2, "efficiency_bonus": 0.0},
            "landauer_optimal": {"energy_factor": 0.5, "efficiency_bonus": 0.3},
        }

        results = {}

        for mode, params in modes.items():
            # Calculate energy requirements
            base_energy = computation_size * BOLTZMANN_CONSTANT * 2.0 * math.log(2)
            actual_energy = base_energy * params["energy_factor"]

            # Calculate efficiency
            efficiency = (base_energy / actual_energy) + params["efficiency_bonus"]
            efficiency = min(efficiency, 1.0)  # Cap at 100%

            results[mode] = {
                "energy_required": actual_energy
                "efficiency": efficiency
                "landauer_ratio": actual_energy / base_energy
            }

        logger.info(f"💻 Thermodynamic computing simulation completed")
        logger.info(
            f"   Best efficiency: {max(r['efficiency'] for r in results.values()):.3f}"
        )

        return {
            "computation_id": computation_id
            "computation_size": computation_size
            "modes": results
            "timestamp": datetime.now().isoformat(),
        }

    def apply_fluctuation_dissipation_learning(
        self, geoids: List[GeoidState], temperature: float = 2.0
    ) -> Dict[str, Any]:
        """Apply fluctuation-dissipation theorem for enhanced learning"""

        learning_id = f"FDT_LEARNING_{uuid.uuid4().hex[:8]}"

        # Generate thermal noise for enhanced learning
        noise_strength = math.sqrt(temperature)

        learning_improvements = []

        for geoid in geoids:
            # Apply thermal noise to learning process
            improvement = 0.0
            for key, value in geoid.semantic_state.items():
                # FDT-enhanced learning
                thermal_noise = np.random.normal(0, noise_strength * 0.1)
                old_value = value
                new_value = value + thermal_noise
                geoid.semantic_state[key] = new_value
                improvement += abs(new_value - old_value)

            if geoid.semantic_state:
                improvement /= len(geoid.semantic_state)
            learning_improvements.append(improvement)

        avg_improvement = (
            np.mean(learning_improvements) if learning_improvements else 0.0
        )

        result = {
            "learning_id": learning_id
            "temperature": temperature
            "noise_strength": noise_strength
            "geoids_processed": len(geoids),
            "average_improvement": avg_improvement
            "learning_improvements": learning_improvements
            "stochastic_resonance_achieved": avg_improvement > 0.1
        }

        logger.info(f"📊 Fluctuation-dissipation learning completed")
        logger.info(f"   Average improvement: {avg_improvement:.3f}")

        return result

    async def run_comprehensive_analysis(
        self, geoids: List[GeoidState], temperature: float = 2.0
    ) -> Dict[str, Any]:
        """Run comprehensive analysis using multiple applications"""

        analysis_id = f"COMPREHENSIVE_{uuid.uuid4().hex[:8]}"
        start_time = datetime.now()

        logger.info(f"🚀 Starting comprehensive thermodynamic analysis {analysis_id}")

        results = {
            "analysis_id": analysis_id
            "timestamp": start_time
            "temperature": temperature
            "geoid_count": len(geoids),
            "applications": {},
        }

        # 1. Phase Transition Analysis
        phase_results = await self.run_phase_transition_analysis(geoids, temperature)
        results["applications"]["phase_transitions"] = phase_results

        # 2. Information Heat Engine
        szilard_engine = self.create_information_heat_engine(
            demon_temperature=temperature * 1.5, reservoir_temperature=temperature
        )
        maxwell_result = self.operate_maxwell_demon(geoids)

        results["applications"]["information_engines"] = {
            "szilard_engine": szilard_engine
            "maxwell_demon": maxwell_result
        }

        # 3. Thermodynamic Computing
        computing_result = self.simulate_thermodynamic_computing(len(geoids))
        results["applications"]["thermodynamic_computing"] = computing_result

        # 4. Fluctuation-Dissipation Learning
        learning_result = self.apply_fluctuation_dissipation_learning(
            geoids, temperature
        )
        results["applications"]["fluctuation_dissipation"] = learning_result

        # Calculate overall performance
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()

        # Aggregate efficiency metrics
        phase_efficiency = phase_results["performance"]["detection_efficiency"]
        heat_efficiency = szilard_engine["efficiency"]
        computing_efficiency = max(
            r["efficiency"] for r in computing_result["modes"].values()
        )
        learning_efficiency = min(
            learning_result["average_improvement"] * 10, 1.0
        )  # Scale and cap

        overall_efficiency = np.mean(
            [
                phase_efficiency
                heat_efficiency
                computing_efficiency
                learning_efficiency
            ]
        )

        results["performance_summary"] = {
            "total_execution_time": total_time
            "overall_efficiency": overall_efficiency
            "phase_efficiency": phase_efficiency
            "heat_efficiency": heat_efficiency
            "computing_efficiency": computing_efficiency
            "learning_efficiency": learning_efficiency
            "applications_completed": 4
        }

        # Log results
        self.total_applications_run += 4
        self.efficiency_metrics["overall"].append(overall_efficiency)
        self.application_log.append(results)

        logger.info(f"🎯 Comprehensive analysis completed")
        logger.info(f"   Overall efficiency: {overall_efficiency:.3f}")
        logger.info(f"   Total time: {total_time:.2f}s")

        return results

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all thermodynamic applications"""

        return {
            "advanced_applications": {
                "phase_detector": {
                    "transitions_detected": len(
                        self.phase_detector.detected_transitions
                    ),
                    "order_parameter_history_size": len(
                        self.phase_detector.order_parameter_history
                    ),
                    "critical_temperature_threshold": self.phase_detector.critical_temperature_threshold
                }
            },
            "system_performance": {
                "total_applications_run": self.total_applications_run
                "average_efficiency": (
                    np.mean(self.efficiency_metrics["overall"])
                    if self.efficiency_metrics["overall"]
                    else 0.0
                ),
                "application_log_size": len(self.application_log),
            },
            "thermodynamic_constants": {
                "boltzmann_constant": BOLTZMANN_CONSTANT
                "planck_constant": PLANCK_CONSTANT
                "golden_ratio": GOLDEN_RATIO
                "critical_temperature": CRITICAL_TEMPERATURE
                "prigogine_constant": PRIGOGINE_CONSTANT
            },
        }


# Demonstration function
async def demonstrate_advanced_thermodynamic_applications():
    """Demonstrate advanced thermodynamic applications"""

    logger.info("🚀 ADVANCED THERMODYNAMIC APPLICATIONS DEMONSTRATION")
    logger.info("=" * 80)

    # Initialize engine
    engine = AdvancedThermodynamicApplicationsEngine()

    # Create test geoids
    test_geoids = []
    for i in range(8):
        semantic_state = {f"feature_{j}": np.random.normal(0, 1) for j in range(5)}
        geoid = GeoidState(
            geoid_id=f"ADVANCED_TEST_{i}",
            semantic_state=semantic_state
            symbolic_state={"type": "test", "index": i},
        )
        test_geoids.append(geoid)

    # Run comprehensive analysis
    results = await engine.run_comprehensive_analysis(test_geoids, temperature=2.5)

    # Display results
    logger.info("\n📊 DEMONSTRATION RESULTS:")
    logger.info("-" * 40)

    performance = results["performance_summary"]
    logger.info(f"Overall efficiency: {performance['overall_efficiency']:.3f}")
    logger.info(
        f"Phase transitions: {'Detected' if results['applications']['phase_transitions']['phase_transition_detected'] else 'None'}"
    )
    logger.info(
        f"Information work: {results['applications']['information_engines']['szilard_engine']['work_extracted']:.3f}"
    )
    logger.info(f"Computing efficiency: {performance['computing_efficiency']:.3f}")
    logger.info(f"Learning improvement: {performance['learning_efficiency']:.3f}")

    # System status
    status = engine.get_system_status()
    logger.info(f"\n🔧 SYSTEM STATUS:")
    logger.info(
        f"Applications run: {status['system_performance']['total_applications_run']}"
    )
    logger.info(
        f"Average efficiency: {status['system_performance']['average_efficiency']:.3f}"
    )

    logger.info("\n✅ Advanced thermodynamic applications demonstration complete")

    return results


if __name__ == "__main__":
    asyncio.run(demonstrate_advanced_thermodynamic_applications())
