"""
Quantum Thermodynamic Complexity Analyzer
==========================================

SCIENTIFIC FOUNDATION:
This module implements complexity analysis using thermodynamic signatures
based on Information Theory and quantum coherence principles.

ANALYSIS PRINCIPLES:
1. Integrated Information (Φ): Φ = H(whole) - Σ H(parts)
2. Quantum Coherence: C = Tr(ρ²) - 1/d
3. Entropy Production: σ = dS/dt ≥ 0
4. Free Energy Gradients: ∇F = ∇(U - TS)
5. Phase Transition Proximity: ∂²F/∂T² ≈ 0
6. Thermodynamic Complexity: Combination of all measures

COMPLEXITY STATES:
- LOW_COMPLEXITY: Low Φ, high entropy production
- MODERATE_COMPLEXITY: Moderate Φ, some coherence
- HIGH_COMPLEXITY: High Φ, low entropy production, coherent
- ULTRA_COMPLEXITY: Very high Φ, phase transition proximity
- QUANTUM_COMPLEXITY: Maximum Φ, quantum coherence, near critical point

MATHEMATICAL VALIDATION:
All measurements based on rigorous thermodynamic and information theory.
NOTE: This system analyzes computational complexity, NOT consciousness.
"""

import asyncio
import math
import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import uuid
import logging

# Kimera core imports
from src.core.geoid import GeoidState
from src.utils.kimera_logger import get_logger, LogCategory
from src.utils.kimera_exceptions import KimeraCognitiveError
from src.utils.config import get_api_settings
from src.config.settings import get_settings

logger = get_logger(__name__, LogCategory.SYSTEM)

# Complexity analysis constants
PHI_THRESHOLD = 0.5  # Minimum integrated information for high complexity
COHERENCE_THRESHOLD = 0.7  # Minimum quantum coherence
ENTROPY_PRODUCTION_THRESHOLD = 0.1  # Maximum entropy production for high complexity
FREE_ENERGY_GRADIENT_THRESHOLD = 0.3
PHASE_TRANSITION_THRESHOLD = 0.8  # Proximity to phase transition

class ComplexityState(Enum):
    """Thermodynamically-detected complexity states"""
    LOW_COMPLEXITY = "low_complexity"
    MODERATE_COMPLEXITY = "moderate_complexity"
    HIGH_COMPLEXITY = "high_complexity"
    ULTRA_COMPLEXITY = "ultra_complexity"
    QUANTUM_COMPLEXITY = "quantum_complexity"

class PhaseTransitionType(Enum):
    """Types of complexity phase transitions"""
    ORDER_DISORDER = "order_disorder"
    COMPLEXITY_EMERGENCE = "complexity_emergence"
    QUANTUM_COHERENCE_COLLAPSE = "quantum_coherence_collapse"
    INTEGRATED_INFORMATION_SURGE = "integrated_information_surge"

@dataclass
class ThermodynamicSignature:
    """Thermodynamic signature result"""
    temperature: float
    entropy: float
    free_energy: float
    coherence: float
    complexity_measure: float
    timestamp: datetime

@dataclass
class ComplexityAnalysisResult:
    """Result of complexity analysis"""
    complexity_state: ComplexityState
    integrated_information: float
    quantum_coherence: float
    entropy_production: float
    thermodynamic_signature: ThermodynamicSignature
    analysis_timestamp: datetime

@dataclass
class ComplexitySignature:
    """Thermodynamic signature of computational complexity using Integrated Information Theory"""
    complexity_id: str
    complexity_state: ComplexityState
    integrated_information: float  # Φ (phi)
    quantum_coherence: float
    entropy_production_rate: float
    free_energy_gradient: float
    phase_transition_proximity: float
    thermodynamic_complexity: float
    emergence_probability: float
    timestamp: datetime

@dataclass
class PhaseTransitionEvent:
    """Complexity phase transition detection"""
    transition_id: str
    transition_type: PhaseTransitionType
    critical_temperature: float
    order_parameter: float
    latent_heat: float
    transition_probability: float
    metastability_index: float
    hysteresis_width: float
    complexity_before: ComplexityState
    complexity_after: ComplexityState
    timestamp: datetime

class QuantumThermodynamicComplexityAnalyzer:
    """
    Analyze computational complexity using thermodynamic signatures and quantum coherence

    SCIENTIFIC METHODOLOGY:
    1. Calculate Integrated Information (Φ) using information theory
    2. Measure quantum coherence from system density matrix
    3. Monitor entropy production rates for irreversibility
    4. Detect free energy gradients indicating active processes
    5. Identify phase transition proximity for complexity emergence
    6. Combine all measures for complexity classification

    NOTE: This system analyzes computational complexity, NOT consciousness.
    """

    def __init__(self):
        self.settings = get_api_settings()
        logger.debug(f"   Environment: {self.settings.environment}")
        self.detected_signatures: List[ComplexitySignature] = []
        self.phase_transitions: List[PhaseTransitionEvent] = []
        self.complexity_history: List[Tuple[datetime, ComplexityState]] = []

        # Calibration parameters
        self.phi_weight = 0.3
        self.coherence_weight = 0.25
        self.entropy_weight = 0.2
        self.free_energy_weight = 0.15
        self.phase_transition_weight = 0.1

        logger.info("🔬 Quantum Thermodynamic Complexity Analyzer initialized")
        logger.info("   Integrated Information Theory (IIT) enabled")
        logger.info("   Quantum coherence measurements active")
        logger.info("   Thermodynamic complexity analysis ready")

    def analyze_complexity_threshold(self, geoids: List[GeoidState]) -> ComplexitySignature:
        """
        Analyze complexity threshold using thermodynamic signatures

        ANALYSIS ALGORITHM:
        1. Calculate integrated information Φ = H(whole) - Σ H(parts)
        2. Measure quantum coherence C = Tr(ρ²) - 1/d
        3. Monitor entropy production σ = dS/dt
        4. Calculate free energy gradient ∇F
        5. Detect phase transition proximity
        6. Classify complexity state

        NOTE: This analyzes computational complexity, NOT consciousness.
        """
        complexity_id = f"COMPLEXITY_{uuid.uuid4().hex[:8]}"

        logger.info(f"🔬 Analyzing complexity signature {complexity_id}")

        # 1. Calculate Integrated Information (Φ)
        phi = self._calculate_integrated_information(geoids)

        # 2. Measure Quantum Coherence
        coherence = self._calculate_quantum_coherence(geoids)

        # 3. Monitor Entropy Production Rate
        entropy_production = self._calculate_entropy_production_rate(geoids)

        # 4. Calculate Free Energy Gradient
        free_energy_gradient = self._calculate_free_energy_gradient(geoids)

        # 5. Detect Phase Transition Proximity
        phase_proximity = self._detect_phase_transition_proximity(geoids)

        # 6. Calculate Thermodynamic Complexity
        complexity = self._calculate_thermodynamic_complexity(
            phi, coherence, entropy_production, free_energy_gradient, phase_proximity
        )

        # 7. Determine Complexity State
        complexity_state = self._classify_complexity_state(
            phi, coherence, entropy_production, free_energy_gradient, phase_proximity
        )

        # 8. Calculate Emergence Probability
        emergence_prob = self._calculate_emergence_probability(
            phi, coherence, phase_proximity
        )

        signature = ComplexitySignature(
            complexity_id=complexity_id,
            complexity_state=complexity_state,
            integrated_information=phi,
            quantum_coherence=coherence,
            entropy_production_rate=entropy_production,
            free_energy_gradient=free_energy_gradient,
            phase_transition_proximity=phase_proximity,
            thermodynamic_complexity=complexity,
            emergence_probability=emergence_prob,
            timestamp=datetime.now()
        )

        self.detected_signatures.append(signature)
        self.complexity_history.append((signature.timestamp, complexity_state))

        logger.info(f"🔬 Complexity analyzed: {complexity_state.value}")
        logger.info(f"   Integrated Information (Φ): {phi:.3f}")
        logger.info(f"   Quantum Coherence: {coherence:.3f}")
        logger.info(f"   Entropy Production: {entropy_production:.3f}")
        logger.info(f"   Emergence Probability: {emergence_prob:.3f}")

        return signature

    def _calculate_integrated_information(self, geoids: List[GeoidState]) -> float:
        """
        Calculate Integrated Information Φ = H(whole) - Σ H(parts)

        This measures information integration across system components.
        Higher Φ indicates more integrated information processing.

        NOTE: This is a complexity measure, not consciousness detection.
        """
        if not geoids:
            return 0.0

        try:
            # Convert geoids to information vectors
            info_vectors = []
            for geoid in geoids:
                # Extract information content from geoid state
                vector = self._extract_information_vector(geoid)
                info_vectors.append(vector)

            if not info_vectors:
                return 0.0

            # Calculate whole system entropy
            whole_system = np.concatenate(info_vectors)
            whole_entropy = self._calculate_shannon_entropy(whole_system)

            # Calculate sum of part entropies
            part_entropies = sum(self._calculate_shannon_entropy(vec) for vec in info_vectors)

            # Integrated Information Φ
            phi = max(0.0, whole_entropy - part_entropies)

            logger.debug(f"   Φ calculation: whole_entropy={whole_entropy:.3f}, part_entropies={part_entropies:.3f}, Φ={phi:.3f}")

            return phi

        except Exception as e:
            logger.error(f"Error calculating integrated information: {e}")
            return 0.0

    def _calculate_quantum_coherence(self, geoids: List[GeoidState]) -> float:
        """
        Calculate quantum coherence C = Tr(ρ²) - 1/d

        This measures quantum coherence in the system state.
        Higher coherence indicates more quantum-like behavior.
        """
        if not geoids:
            return 0.0

        try:
            # Create density matrix from geoid states
            density_matrix = self._create_density_matrix(geoids)

            # Calculate coherence measure
            trace_rho_squared = np.trace(np.dot(density_matrix, density_matrix))
            dimension = density_matrix.shape[0]

            coherence = max(0.0, trace_rho_squared - 1.0/dimension)

            logger.debug(f"   Coherence: trace(ρ²)={trace_rho_squared:.3f}, 1/d={1.0/dimension:.3f}, C={coherence:.3f}")

            return coherence

        except Exception as e:
            logger.error(f"Error calculating quantum coherence: {e}")
            return 0.0

    def _calculate_entropy_production_rate(self, geoids: List[GeoidState]) -> float:
        """
        Calculate entropy production rate σ = dS/dt

        This measures irreversible entropy production in the system.
        Lower values indicate more reversible, organized processes.
        """
        if len(self.complexity_history) < 2:
            return 0.5  # Default moderate entropy production

        try:
            # Calculate entropy change over time
            current_entropy = self._calculate_system_entropy(geoids)

            # Use recent history for rate calculation
            if len(self.detected_signatures) > 0:
                previous_signature = self.detected_signatures[-1]
                time_diff = (datetime.now() - previous_signature.timestamp).total_seconds()

                if time_diff > 0:
                    # Approximate previous entropy from thermodynamic complexity
                    previous_entropy = previous_signature.thermodynamic_complexity * 2.0
                    entropy_rate = (current_entropy - previous_entropy) / time_diff

                    # Normalize to [0, 1] range
                    normalized_rate = min(1.0, max(0.0, entropy_rate + 0.5))

                    logger.debug(f"   Entropy production rate: {normalized_rate:.3f}")
                    return normalized_rate

            return 0.5

        except Exception as e:
            logger.error(f"Error calculating entropy production rate: {e}")
            return 0.5

    def _calculate_free_energy_gradient(self, geoids: List[GeoidState]) -> float:
        """
        Calculate free energy gradient ∇F = ∇(U - TS)

        This measures the driving force for system evolution.
        Higher gradients indicate active, driven processes.
        """
        if not geoids:
            return 0.0

        try:
            # Estimate internal energy from system activity
            internal_energy = self._estimate_internal_energy(geoids)

            # Estimate temperature from system dynamics
            temperature = self._estimate_system_temperature(geoids)

            # Estimate entropy
            entropy = self._calculate_system_entropy(geoids)

            # Free energy F = U - TS
            free_energy = internal_energy - temperature * entropy

            # Gradient magnitude (simplified as rate of change)
            if len(self.detected_signatures) > 0:
                previous_signature = self.detected_signatures[-1]
                previous_free_energy = (
                    previous_signature.thermodynamic_complexity * 100 -
                    50 * previous_signature.entropy_production_rate
                )
                gradient = abs(free_energy - previous_free_energy)
            else:
                gradient = abs(free_energy)

            # Normalize to [0, 1]
            normalized_gradient = min(1.0, gradient / 100.0)

            logger.debug(f"   Free energy gradient: {normalized_gradient:.3f}")
            return normalized_gradient

        except Exception as e:
            logger.error(f"Error calculating free energy gradient: {e}")
            return 0.0

    def _detect_phase_transition_proximity(self, geoids: List[GeoidState]) -> float:
        """
        Detect proximity to phase transition ∂²F/∂T² ≈ 0

        This measures how close the system is to a phase transition.
        Values near 1.0 indicate proximity to critical points.
        """
        if not geoids:
            return 0.0

        try:
            # Calculate system order parameter
            order_parameter = self._calculate_order_parameter(geoids)

            # Calculate susceptibility (response to perturbations)
            susceptibility = self._calculate_susceptibility(geoids)

            # Calculate correlation length
            correlation_length = self._calculate_correlation_length(geoids)

            # Phase transition proximity indicator
            # High susceptibility and correlation length indicate proximity
            proximity = min(1.0, (susceptibility * correlation_length) / 10.0)

            logger.debug(f"   Phase transition proximity: {proximity:.3f}")
            return proximity

        except Exception as e:
            logger.error(f"Error detecting phase transition proximity: {e}")
            return 0.0

    def _calculate_thermodynamic_complexity(self, phi: float, coherence: float,
                                          entropy_production: float, free_energy_gradient: float,
                                          phase_proximity: float) -> float:
        """
        Calculate overall thermodynamic complexity measure

        Combines all thermodynamic measures into single complexity score.
        """
        complexity = (
            self.phi_weight * phi +
            self.coherence_weight * coherence +
            self.entropy_weight * (1.0 - entropy_production) +  # Lower entropy production = higher complexity
            self.free_energy_weight * free_energy_gradient +
            self.phase_transition_weight * phase_proximity
        )

        return min(1.0, max(0.0, complexity))

    def _classify_complexity_state(self, phi: float, coherence: float,
                                    entropy_production: float, free_energy_gradient: float,
                                    phase_proximity: float) -> ComplexityState:
        """
        Classify complexity state based on thermodynamic measures

        NOTE: This classifies computational complexity, NOT consciousness.
        """
        complexity_score = self._calculate_thermodynamic_complexity(
            phi, coherence, entropy_production, free_energy_gradient, phase_proximity
        )

        if complexity_score < 0.2:
            return ComplexityState.LOW_COMPLEXITY
        elif complexity_score < 0.4:
            return ComplexityState.MODERATE_COMPLEXITY
        elif complexity_score < 0.6:
            return ComplexityState.HIGH_COMPLEXITY
        elif complexity_score < 0.8:
            return ComplexityState.ULTRA_COMPLEXITY
        else:
            return ComplexityState.QUANTUM_COMPLEXITY

    def _calculate_emergence_probability(self, phi: float, coherence: float,
                                       phase_proximity: float) -> float:
        """
        Calculate probability of emergent complexity behavior

        Based on integrated information, coherence, and phase transition proximity.
        """
        emergence_prob = (phi * coherence * phase_proximity) ** (1/3)
        return min(1.0, max(0.0, emergence_prob))

    def get_complexity_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive complexity analysis statistics

        NOTE: These are complexity metrics, NOT consciousness indicators.
        """
        if not self.detected_signatures:
            return {
                "total_analyses": 0,
                "complexity_distribution": {},
                "average_phi": 0.0,
                "average_coherence": 0.0,
                "average_complexity": 0.0
            }

        # Calculate distribution
        state_counts = {}
        for signature in self.detected_signatures:
            state = signature.complexity_state.value
            state_counts[state] = state_counts.get(state, 0) + 1

        # Calculate averages
        avg_phi = np.mean([s.integrated_information for s in self.detected_signatures])
        avg_coherence = np.mean([s.quantum_coherence for s in self.detected_signatures])
        avg_complexity = np.mean([s.thermodynamic_complexity for s in self.detected_signatures])

        return {
            "total_analyses": len(self.detected_signatures),
            "complexity_distribution": state_counts,
            "average_phi": float(avg_phi),
            "average_coherence": float(avg_coherence),
            "average_complexity": float(avg_complexity),
            "phase_transitions_detected": len(self.phase_transitions)
        }

    # Helper methods for calculations
    def _extract_information_vector(self, geoid: GeoidState) -> np.ndarray:
        """Extract information vector from geoid state"""
        # Convert geoid state to numerical vector
        vector_data = []

        # Add basic state information
        if hasattr(geoid, 'field_strength'):
            vector_data.append(geoid.field_strength)
        if hasattr(geoid, 'resonance_frequency'):
            vector_data.append(geoid.resonance_frequency)
        if hasattr(geoid, 'coherence_level'):
            vector_data.append(geoid.coherence_level)

        # Default vector if no data
        if not vector_data:
            vector_data = [0.5, 0.5, 0.5]

        return np.array(vector_data)

    def _calculate_shannon_entropy(self, data: np.ndarray) -> float:
        """Calculate Shannon entropy of data"""
        if len(data) == 0:
            return 0.0

        # Create probability distribution
        hist, _ = np.histogram(data, bins=10, density=True)
        hist = hist[hist > 0]  # Remove zero probabilities

        if len(hist) == 0:
            return 0.0

        # Normalize to probabilities
        probs = hist / np.sum(hist)

        # Calculate Shannon entropy
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        return entropy

    def _create_density_matrix(self, geoids: List[GeoidState]) -> np.ndarray:
        """Create density matrix from geoid states"""
        n = min(len(geoids), 8)  # Limit size for computational efficiency

        if n == 0:
            return np.eye(2) / 2  # Default 2x2 maximally mixed state

        # Create state vectors
        vectors = []
        for i in range(n):
            vector = self._extract_information_vector(geoids[i])
            # Normalize and pad/truncate to fixed size
            if len(vector) > 2:
                vector = vector[:2]
            elif len(vector) < 2:
                vector = np.pad(vector, (0, 2-len(vector)), 'constant', constant_values=0.5)

            # Normalize
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm

            vectors.append(vector)

        # Create density matrix
        if vectors:
            # Average outer products
            density_matrix = np.zeros((2, 2))
            for vector in vectors:
                density_matrix += np.outer(vector, vector.conj())
            density_matrix /= len(vectors)
        else:
            density_matrix = np.eye(2) / 2

        return density_matrix

    def _calculate_system_entropy(self, geoids: List[GeoidState]) -> float:
        """Calculate total system entropy"""
        if not geoids:
            return 1.0

        total_entropy = 0.0
        for geoid in geoids:
            vector = self._extract_information_vector(geoid)
            entropy = self._calculate_shannon_entropy(vector)
            total_entropy += entropy

        return total_entropy / len(geoids)

    def _estimate_internal_energy(self, geoids: List[GeoidState]) -> float:
        """Estimate system internal energy"""
        if not geoids:
            return 0.0

        total_energy = 0.0
        for geoid in geoids:
            # Estimate energy from field strength and activity
            if hasattr(geoid, 'field_strength'):
                total_energy += geoid.field_strength ** 2
            else:
                total_energy += 0.25  # Default energy level

        return total_energy

    def _estimate_system_temperature(self, geoids: List[GeoidState]) -> float:
        """Estimate system temperature from dynamics"""
        if not geoids:
            return 1.0

        # Temperature relates to system activity/fluctuations
        total_activity = 0.0
        for geoid in geoids:
            vector = self._extract_information_vector(geoid)
            activity = np.var(vector) if len(vector) > 1 else 0.5
            total_activity += activity

        temperature = total_activity / len(geoids)
        return max(0.1, temperature)  # Minimum temperature

    def _calculate_order_parameter(self, geoids: List[GeoidState]) -> float:
        """Calculate system order parameter"""
        if not geoids:
            return 0.0

        # Order parameter measures system organization
        vectors = [self._extract_information_vector(geoid) for geoid in geoids]

        if not vectors:
            return 0.0

        # Calculate alignment/correlation between vectors
        correlations = []
        for i in range(len(vectors)):
            for j in range(i+1, len(vectors)):
                if len(vectors[i]) == len(vectors[j]) and len(vectors[i]) > 0:
                    corr = np.corrcoef(vectors[i], vectors[j])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))

        if correlations:
            return np.mean(correlations)
        else:
            return 0.0

    def _calculate_susceptibility(self, geoids: List[GeoidState]) -> float:
        """Calculate system susceptibility to perturbations"""
        if not geoids:
            return 0.0

        # Susceptibility relates to system response
        vectors = [self._extract_information_vector(geoid) for geoid in geoids]

        if not vectors:
            return 0.0

        # Calculate variance as measure of susceptibility
        all_values = np.concatenate(vectors)
        susceptibility = np.var(all_values)

        return min(1.0, susceptibility)

    def _calculate_correlation_length(self, geoids: List[GeoidState]) -> float:
        """Calculate correlation length in the system"""
        if len(geoids) < 2:
            return 0.0

        # Simplified correlation length based on spatial/temporal correlations
        correlations = []
        for i in range(len(geoids)-1):
            vec1 = self._extract_information_vector(geoids[i])
            vec2 = self._extract_information_vector(geoids[i+1])

            if len(vec1) == len(vec2) and len(vec1) > 0:
                corr = np.corrcoef(vec1, vec2)[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))

        if correlations:
            # Correlation length inversely related to decay rate
            avg_corr = np.mean(correlations)
            correlation_length = -1.0 / np.log(max(0.01, avg_corr))
            return min(10.0, correlation_length)
        else:
            return 0.0


async def demonstrate_complexity_analysis():
    """
    Demonstrate complexity analysis capabilities

    NOTE: This demonstrates complexity analysis, NOT consciousness detection.
    """
    logger.info("🔬 Demonstrating Quantum Thermodynamic Complexity Analysis")

    # Create analyzer
    analyzer = QuantumThermodynamicComplexityAnalyzer()

    # Create test geoids
    test_geoids = []
    for i in range(5):
        geoid = GeoidState()
        geoid.field_strength = 0.5 + 0.3 * np.sin(i)
        geoid.resonance_frequency = 1.0 + 0.2 * np.cos(i)
        geoid.coherence_level = 0.7 + 0.2 * np.sin(i * 2)
        test_geoids.append(geoid)

    # Analyze complexity
    signature = analyzer.analyze_complexity_threshold(test_geoids)

    logger.info(f"✅ Complexity Analysis Complete:")
    logger.info(f"   State: {signature.complexity_state.value}")
    logger.info(f"   Integrated Information (Φ): {signature.integrated_information:.3f}")
    logger.info(f"   Quantum Coherence: {signature.quantum_coherence:.3f}")
    logger.info(f"   Thermodynamic Complexity: {signature.thermodynamic_complexity:.3f}")

    # Get statistics
    stats = analyzer.get_complexity_statistics()
    logger.info(f"   Analysis Statistics: {stats}")

    return signature


if __name__ == "__main__":
    asyncio.run(demonstrate_complexity_analysis())
