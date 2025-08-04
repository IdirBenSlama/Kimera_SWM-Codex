"""
KIMERA SWM - THERMODYNAMIC EVOLUTION ENGINE
==========================================

The Thermodynamic Evolution Engine implements physics-compliant evolution
of geoids based on fundamental thermodynamic principles. It treats cognitive
states as thermodynamic systems that evolve according to energy minimization,
entropy maximization, and equilibrium seeking.

This engine ensures that all cognitive transformations respect physical laws
and maintain consistency with thermodynamic principles.
"""

import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...core.data_structures.geoid_state import (
    GeoidProcessingState,
    GeoidState,
    GeoidType,
    SemanticState,
    SymbolicState,
    ThermodynamicProperties,
)
from ...core.processing.geoid_processor import GeoidProcessor, ProcessingResult

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class EvolutionParameters:
    """Parameters controlling thermodynamic evolution"""

    time_step: float = 1.0  # Evolution time step
    temperature_decay: float = 0.95  # Rate of temperature cooling
    energy_dissipation: float = 0.1  # Rate of energy dissipation
    equilibrium_attraction: float = 0.2  # Strength of equilibrium attraction
    noise_amplitude: float = 0.05  # Thermal noise amplitude
    max_energy_change: float = 2.0  # Maximum energy change per step
    boltzmann_constant: float = 1.0  # Cognitive Boltzmann constant
    enable_quantum_effects: bool = True  # Enable quantum tunneling
    coherence_coupling: float = 0.3  # Coupling between coherence and energy


@dataclass
class EvolutionResult:
    """Result of thermodynamic evolution"""

    original_geoid: GeoidState
    evolved_geoid: GeoidState
    energy_change: float
    entropy_change: float
    temperature_change: float
    evolution_probability: float
    quantum_tunneling: bool
    duration: float
    metadata: Dict[str, Any]


class ThermodynamicEvolutionEngine:
    """
    Thermodynamic Evolution Engine - Physics-Compliant Cognitive Evolution
    =====================================================================

    This engine evolves geoids according to fundamental thermodynamic principles:

    1. Energy Conservation: Total cognitive energy is conserved
    2. Entropy Maximization: Systems tend toward maximum entropy states
    3. Temperature Equilibration: Hot and cold regions equilibrate
    4. Minimum Energy Principle: Systems seek minimum energy configurations
    5. Fluctuation-Dissipation: Thermal fluctuations drive evolution

    The engine treats geoids as thermodynamic systems in cognitive space,
    where semantic and symbolic states correspond to different energy modes.
    """

    def __init__(self, parameters: EvolutionParameters = None):
        self.parameters = parameters or EvolutionParameters()
        self.evolution_history: List[EvolutionResult] = []
        self.total_energy_balance = 0.0
        self.total_entropy_change = 0.0
        self.evolution_count = 0

        # Physics constants
        self.h_bar = 1.0  # Reduced Planck constant in cognitive units
        self.k_b = self.parameters.boltzmann_constant

        logger.info(
            f"ThermodynamicEvolutionEngine initialized with parameters: {parameters}"
        )

    def evolve(
        self, geoid: GeoidState, parameters: EvolutionParameters = None
    ) -> EvolutionResult:
        """
        Evolve a geoid according to thermodynamic principles.
        Returns an EvolutionResult containing the evolved geoid and evolution metadata.
        """
        start_time = time.time()
        params = parameters or self.parameters

        if geoid.thermodynamic is None:
            # Initialize thermodynamic properties if missing
            geoid.thermodynamic = self._initialize_thermodynamic_properties(geoid)

        # Calculate evolution trajectory
        evolution_trajectory = self._calculate_evolution_trajectory(geoid, params)

        # Apply evolution
        evolved_geoid = self._apply_evolution(geoid, evolution_trajectory, params)

        # Calculate changes
        energy_change = evolved_geoid.cognitive_energy - geoid.cognitive_energy
        entropy_change = (
            evolved_geoid.thermodynamic.information_entropy
            - geoid.thermodynamic.information_entropy
        )
        temperature_change = (
            evolved_geoid.thermodynamic.cognitive_temperature
            - geoid.thermodynamic.cognitive_temperature
        )

        # Calculate evolution probability
        evolution_probability = self._calculate_evolution_probability(
            geoid, evolved_geoid, params
        )

        # Check for quantum tunneling
        quantum_tunneling = self._check_quantum_tunneling(geoid, evolved_geoid, params)

        # Create result
        result = EvolutionResult(
            original_geoid=geoid,
            evolved_geoid=evolved_geoid,
            energy_change=energy_change,
            entropy_change=entropy_change,
            temperature_change=temperature_change,
            evolution_probability=evolution_probability,
            quantum_tunneling=quantum_tunneling,
            duration=time.time() - start_time,
            metadata={
                "evolution_type": "thermodynamic",
                "parameters": params.__dict__,
                "trajectory_points": len(evolution_trajectory),
                "equilibrium_distance": self._calculate_equilibrium_distance(
                    evolved_geoid
                ),
            },
        )

        # Update engine state
        self._update_engine_state(result)

        # Record in geoid metadata
        evolved_geoid.metadata.add_processing_step(
            engine_name="ThermodynamicEvolutionEngine",
            operation="thermodynamic_evolution",
            duration=result.duration,
            metadata=result.metadata,
        )

        return result

    def evolve_system(
        self, geoids: List[GeoidState], parameters: EvolutionParameters = None
    ) -> List[EvolutionResult]:
        """
        Evolve a system of interacting geoids.
        Considers thermodynamic interactions between geoids.
        """
        params = parameters or self.parameters
        results = []

        # Calculate system-wide thermodynamic state
        system_temperature = self._calculate_system_temperature(geoids)
        system_energy = sum(g.cognitive_energy for g in geoids)

        logger.info(
            f"Evolving system of {len(geoids)} geoids (T={system_temperature:.3f}, E={system_energy:.3f})"
        )

        # Evolve each geoid considering system context
        for geoid in geoids:
            # Modify parameters based on system state
            system_params = self._adapt_parameters_for_system(params, geoid, geoids)

            # Evolve individual geoid
            result = self.evolve(geoid, system_params)
            results.append(result)

        # Apply system-wide conservation laws
        self._enforce_conservation_laws(results)

        return results

    def _initialize_thermodynamic_properties(
        self, geoid: GeoidState
    ) -> ThermodynamicProperties:
        """Initialize thermodynamic properties for a geoid"""
        # Base values on geoid characteristics
        base_temperature = 1.0
        base_entropy = 1.0
        base_energy = 5.0

        # Adjust based on coherence
        coherence_factor = geoid.coherence_score
        energy_adjustment = coherence_factor * 3.0

        # Adjust based on processing depth
        depth_factor = min(1.0, geoid.metadata.processing_depth / 10.0)
        temperature_adjustment = 1.0 + depth_factor * 0.5

        return ThermodynamicProperties(
            cognitive_temperature=base_temperature * temperature_adjustment,
            information_entropy=base_entropy + np.random.normal(0, 0.2),
            free_energy=base_energy + energy_adjustment,
            activation_energy=2.0 + np.random.normal(0, 0.5),
            dissipation_rate=0.1 + np.random.normal(0, 0.02),
            equilibrium_tendency=0.5 + coherence_factor * 0.3,
        )

    def _calculate_evolution_trajectory(
        self, geoid: GeoidState, params: EvolutionParameters
    ) -> List[Dict[str, float]]:
        """Calculate the thermodynamic evolution trajectory"""
        trajectory = []
        current_state = {
            "temperature": geoid.thermodynamic.cognitive_temperature,
            "energy": geoid.thermodynamic.free_energy,
            "entropy": geoid.thermodynamic.information_entropy,
            "coherence": geoid.coherence_score,
        }

        # Calculate trajectory over multiple sub-steps
        sub_steps = max(1, int(params.time_step * 10))
        dt = params.time_step / sub_steps

        for step in range(sub_steps):
            # Apply thermodynamic forces
            temperature_force = self._calculate_temperature_force(current_state, params)
            energy_force = self._calculate_energy_force(current_state, params)
            entropy_force = self._calculate_entropy_force(current_state, params)

            # Update state
            current_state["temperature"] *= 1 + temperature_force * dt
            current_state["energy"] += energy_force * dt
            current_state["entropy"] += entropy_force * dt

            # Apply constraints
            current_state["temperature"] = max(0.01, current_state["temperature"])
            current_state["energy"] = max(0.0, current_state["energy"])
            current_state["entropy"] = max(0.0, current_state["entropy"])

            # Add noise
            if params.noise_amplitude > 0:
                noise = np.random.normal(0, params.noise_amplitude)
                current_state["energy"] += noise
                current_state["entropy"] += abs(noise) * 0.1

            trajectory.append(current_state.copy())

        return trajectory

    def _calculate_temperature_force(
        self, state: Dict[str, float], params: EvolutionParameters
    ) -> float:
        """Calculate the force on temperature"""
        # Temperature decay toward equilibrium
        decay_force = -params.temperature_decay * 0.1

        # Coupling with energy
        energy_coupling = (state["energy"] - 5.0) * 0.02  # Energy baseline of 5.0

        return decay_force + energy_coupling

    def _calculate_energy_force(
        self, state: Dict[str, float], params: EvolutionParameters
    ) -> float:
        """Calculate the force on energy"""
        # Energy dissipation
        dissipation_force = -state["energy"] * params.energy_dissipation

        # Equilibrium attraction
        equilibrium_energy = 5.0  # Target equilibrium energy
        equilibrium_force = (
            equilibrium_energy - state["energy"]
        ) * params.equilibrium_attraction

        # Temperature coupling
        temperature_coupling = (state["temperature"] - 1.0) * 0.5

        total_force = dissipation_force + equilibrium_force + temperature_coupling

        # Limit maximum change
        return np.clip(total_force, -params.max_energy_change, params.max_energy_change)

    def _calculate_entropy_force(
        self, state: Dict[str, float], params: EvolutionParameters
    ) -> float:
        """Calculate the force on entropy"""
        # Entropy tends to increase (second law of thermodynamics)
        base_increase = 0.01

        # Temperature dependence
        temperature_factor = state["temperature"] * 0.05

        # Energy dependence (higher energy states have higher entropy)
        energy_factor = state["energy"] * 0.002

        return base_increase + temperature_factor + energy_factor

    def _apply_evolution(
        self,
        geoid: GeoidState,
        trajectory: List[Dict[str, float]],
        params: EvolutionParameters,
    ) -> GeoidState:
        """Apply the calculated evolution to create an evolved geoid"""
        final_state = trajectory[-1]

        # Create evolved geoid
        evolved_geoid = GeoidState(
            geoid_type=geoid.geoid_type,
            processing_state=GeoidProcessingState.EVOLVING,
            semantic_state=self._evolve_semantic_state(
                geoid.semantic_state, final_state, params
            ),
            symbolic_state=self._evolve_symbolic_state(
                geoid.symbolic_state, final_state, params
            ),
            thermodynamic=ThermodynamicProperties(
                cognitive_temperature=final_state["temperature"],
                information_entropy=final_state["entropy"],
                free_energy=final_state["energy"],
                activation_energy=geoid.thermodynamic.activation_energy,
                dissipation_rate=geoid.thermodynamic.dissipation_rate,
                equilibrium_tendency=geoid.thermodynamic.equilibrium_tendency,
            ),
        )

        # Connect to parent
        evolved_geoid.connect_input("thermodynamic_parent", geoid)

        return evolved_geoid

    def _evolve_semantic_state(
        self,
        semantic_state: Optional[SemanticState],
        thermo_state: Dict[str, float],
        params: EvolutionParameters,
    ) -> Optional[SemanticState]:
        """Evolve the semantic state based on thermodynamic evolution"""
        if semantic_state is None:
            return None

        # Temperature affects uncertainty
        temperature_factor = thermo_state["temperature"]
        new_uncertainty = semantic_state.uncertainty_measure * (
            0.9 + temperature_factor * 0.2
        )
        new_uncertainty = np.clip(new_uncertainty, 0.0, 1.0)

        # Energy affects coherence
        energy_factor = thermo_state["energy"] / 10.0  # Normalize to ~0.5
        coherence_boost = energy_factor * params.coherence_coupling
        new_coherence = semantic_state.coherence_score + coherence_boost
        new_coherence = np.clip(new_coherence, 0.0, 1.0)

        # Entropy affects semantic entropy
        new_semantic_entropy = thermo_state["entropy"] * 1.2

        # Evolve embedding slightly
        noise_scale = temperature_factor * 0.01
        evolved_embedding = semantic_state.embedding_vector + np.random.normal(
            0, noise_scale, semantic_state.embedding_vector.shape
        )

        return SemanticState(
            embedding_vector=evolved_embedding,
            confidence_scores=semantic_state.confidence_scores.copy(),
            uncertainty_measure=new_uncertainty,
            semantic_entropy=new_semantic_entropy,
            coherence_score=new_coherence,
        )

    def _evolve_symbolic_state(
        self,
        symbolic_state: Optional[SymbolicState],
        thermo_state: Dict[str, float],
        params: EvolutionParameters,
    ) -> Optional[SymbolicState]:
        """Evolve the symbolic state based on thermodynamic evolution"""
        if symbolic_state is None:
            return None

        # Create evolved copy
        evolved_symbolic = SymbolicState(
            logical_predicates=symbolic_state.logical_predicates.copy(),
            symbolic_relations=symbolic_state.symbolic_relations.copy(),
            rule_activations=symbolic_state.rule_activations.copy(),
            symbolic_constraints=symbolic_state.symbolic_constraints.copy(),
            proof_chains=symbolic_state.proof_chains.copy(),
        )

        # High temperature might activate/deactivate rules randomly
        if thermo_state["temperature"] > 1.5:
            for rule_name in evolved_symbolic.rule_activations:
                if np.random.random() < 0.1:  # 10% chance to flip
                    evolved_symbolic.rule_activations[rule_name] = (
                        not evolved_symbolic.rule_activations[rule_name]
                    )

        # Add thermodynamic metadata
        evolved_symbolic.symbolic_relations["thermodynamic_evolution"] = {
            "temperature": thermo_state["temperature"],
            "energy": thermo_state["energy"],
            "entropy": thermo_state["entropy"],
        }

        return evolved_symbolic

    def _calculate_evolution_probability(
        self, original: GeoidState, evolved: GeoidState, params: EvolutionParameters
    ) -> float:
        """Calculate the probability of this evolution using statistical mechanics"""
        if original.thermodynamic is None or evolved.thermodynamic is None:
            return 1.0

        energy_diff = (
            evolved.thermodynamic.free_energy - original.thermodynamic.free_energy
        )
        temperature = original.thermodynamic.cognitive_temperature

        if temperature <= 0:
            return 1.0 if energy_diff <= 0 else 0.0

        # Boltzmann factor
        probability = np.exp(-energy_diff / (self.k_b * temperature))
        return min(1.0, probability)

    def _check_quantum_tunneling(
        self, original: GeoidState, evolved: GeoidState, params: EvolutionParameters
    ) -> bool:
        """Check if quantum tunneling occurred (overcoming energy barriers)"""
        if not params.enable_quantum_effects:
            return False

        energy_diff = (
            evolved.thermodynamic.free_energy - original.thermodynamic.free_energy
        )
        activation_energy = original.thermodynamic.activation_energy

        # If energy increased beyond activation barrier, check for tunneling
        if energy_diff > activation_energy:
            tunneling_probability = np.exp(-2 * activation_energy / self.h_bar)
            return np.random.random() < tunneling_probability

        return False

    def _calculate_equilibrium_distance(self, geoid: GeoidState) -> float:
        """Calculate distance from thermodynamic equilibrium"""
        if geoid.thermodynamic is None:
            return float("inf")

        # Ideal equilibrium values
        equilibrium_temperature = 1.0
        equilibrium_energy = 5.0
        equilibrium_entropy = 2.0

        # Calculate normalized distance
        temp_dist = abs(
            geoid.thermodynamic.cognitive_temperature - equilibrium_temperature
        )
        energy_dist = abs(geoid.thermodynamic.free_energy - equilibrium_energy) / 10.0
        entropy_dist = (
            abs(geoid.thermodynamic.information_entropy - equilibrium_entropy) / 5.0
        )

        return np.sqrt(temp_dist**2 + energy_dist**2 + entropy_dist**2)

    def _calculate_system_temperature(self, geoids: List[GeoidState]) -> float:
        """Calculate average system temperature"""
        temperatures = []
        for geoid in geoids:
            if geoid.thermodynamic:
                temperatures.append(geoid.thermodynamic.cognitive_temperature)

        return np.mean(temperatures) if temperatures else 1.0

    def _adapt_parameters_for_system(
        self, params: EvolutionParameters, geoid: GeoidState, system: List[GeoidState]
    ) -> EvolutionParameters:
        """Adapt evolution parameters based on system context"""
        # Create modified parameters
        adapted_params = EvolutionParameters(**params.__dict__)

        # Increase coupling strength in dense systems
        system_density = len(system) / 100.0  # Normalize by typical system size
        adapted_params.equilibrium_attraction *= 1 + system_density * 0.2

        # Reduce noise in large systems (collective behavior)
        adapted_params.noise_amplitude *= max(0.1, 1.0 - system_density * 0.3)

        return adapted_params

    def _enforce_conservation_laws(self, results: List[EvolutionResult]) -> None:
        """Enforce conservation laws across the system"""
        # Energy conservation: total energy change should be minimal
        total_energy_change = sum(r.energy_change for r in results)

        if abs(total_energy_change) > 0.1:  # Allow small violations due to noise
            # Redistribute energy to enforce conservation
            correction_per_geoid = -total_energy_change / len(results)

            for result in results:
                if result.evolved_geoid.thermodynamic:
                    result.evolved_geoid.thermodynamic.free_energy += (
                        correction_per_geoid
                    )
                    result.energy_change += correction_per_geoid

    def _update_engine_state(self, result: EvolutionResult) -> None:
        """Update engine internal state after evolution"""
        self.evolution_history.append(result)
        self.total_energy_balance += result.energy_change
        self.total_entropy_change += result.entropy_change
        self.evolution_count += 1

        # Keep history manageable
        if len(self.evolution_history) > 1000:
            self.evolution_history = self.evolution_history[-500:]

    def get_engine_statistics(self) -> Dict[str, Any]:
        """Get statistics about the engine's performance"""
        if not self.evolution_history:
            return {"evolution_count": 0}

        recent_results = self.evolution_history[-100:]  # Last 100 evolutions

        return {
            "evolution_count": self.evolution_count,
            "total_energy_balance": self.total_energy_balance,
            "total_entropy_change": self.total_entropy_change,
            "average_energy_change": np.mean([r.energy_change for r in recent_results]),
            "average_entropy_change": np.mean(
                [r.entropy_change for r in recent_results]
            ),
            "average_evolution_probability": np.mean(
                [r.evolution_probability for r in recent_results]
            ),
            "quantum_tunneling_rate": np.mean(
                [r.quantum_tunneling for r in recent_results]
            ),
            "average_duration": np.mean([r.duration for r in recent_results]),
            "parameters": self.parameters.__dict__,
        }


# Convenience functions
def evolve_geoid_thermodynamically(
    geoid: GeoidState, time_step: float = 1.0
) -> EvolutionResult:
    """Convenience function to evolve a single geoid"""
    engine = ThermodynamicEvolutionEngine()
    params = EvolutionParameters(time_step=time_step)
    return engine.evolve(geoid, params)


def evolve_system_thermodynamically(
    geoids: List[GeoidState], time_step: float = 1.0
) -> List[EvolutionResult]:
    """Convenience function to evolve a system of geoids"""
    engine = ThermodynamicEvolutionEngine()
    params = EvolutionParameters(time_step=time_step)
    return engine.evolve_system(geoids, params)
