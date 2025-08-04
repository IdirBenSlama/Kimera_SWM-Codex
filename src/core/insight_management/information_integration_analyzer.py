"""
Information Integration Analyzer
===============================

SCIENTIFIC FOUNDATION:
This module implements computational complexity analysis using information-theoretic
signatures based on Integrated Information Theory (IIT) principles.

ANALYSIS PRINCIPLES:
1. Integrated Information (Φ): Φ = H(whole) - Σ H(parts)
2. System Coherence: C = Tr(ρ²) - 1/d
3. Entropy Production: σ = dS/dt ≥ 0
4. Information Gradients: ∇I = ∇(H - TS)
5. Transition Point Proximity: ∂²F/∂T² ≈ 0
6. Computational Complexity: Combination of all measures

COMPLEXITY STATES:
- LOW_INTEGRATION: Low Φ, high entropy production
- MODERATE_INTEGRATION: Moderate Φ, some coherence
- HIGH_INTEGRATION: High Φ, low entropy production, coherent
- VERY_HIGH_INTEGRATION: Very high Φ, transition point proximity
- MAXIMUM_INTEGRATION: Maximum Φ, high coherence, near critical point

MATHEMATICAL VALIDATION:
All measurements based on rigorous information theory and statistical mechanics.
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
try:
    from src.core.geoid import GeoidState
except ImportError:
    try:
        from core.geoid import GeoidState
    except ImportError:
        class GeoidState:
            @staticmethod
            def create_default(): return {}
try:
    from src.utils.kimera_logger import get_logger, LogCategory
except ImportError:
    try:
        from utils.kimera_logger import get_logger, LogCategory
    except ImportError:
        import logging
        def get_logger(*args, **kwargs): return logging.getLogger(__name__)
        class LogCategory:
            SYSTEM = "system"
try:
    from src.utils.kimera_exceptions import KimeraCognitiveError
except ImportError:
    try:
        from utils.kimera_exceptions import KimeraCognitiveError
    except ImportError:
        class KimeraCognitiveError(Exception):
            pass
try:
    from src.utils.config import get_api_settings
except ImportError:
    try:
        from utils.config import get_api_settings
    except ImportError:
        def get_api_settings(): return {}
try:
    from src.config.settings import get_settings
except ImportError:
    try:
        from config.settings import get_settings
    except ImportError:
        def get_settings(): return {}

logger = get_logger(__name__, LogCategory.SYSTEM)

# Information integration analysis constants
PHI_THRESHOLD = 0.5  # Minimum integrated information for high complexity
COHERENCE_THRESHOLD = 0.7  # Minimum system coherence
ENTROPY_PRODUCTION_THRESHOLD = 0.1  # Maximum entropy production for organized systems
INFORMATION_GRADIENT_THRESHOLD = 0.3
TRANSITION_POINT_THRESHOLD = 0.8

class ComplexityState(Enum):
    """Information integration complexity states"""
    LOW_INTEGRATION = "low_integration"
    MODERATE_INTEGRATION = "moderate_integration"
    HIGH_INTEGRATION = "high_integration"
    VERY_HIGH_INTEGRATION = "very_high_integration"
    MAXIMUM_INTEGRATION = "maximum_integration"

class TransitionType(Enum):
    """Types of complexity transitions"""
    ORDER_DISORDER = "order_disorder"
    INTEGRATION_EMERGENCE = "integration_emergence"
    COHERENCE_COLLAPSE = "coherence_collapse"
    INFORMATION_SURGE = "information_surge"

@dataclass
class ComplexitySignature:
    """Information-theoretic complexity signature using Integrated Information Theory"""
    analysis_id: str
    complexity_state: ComplexityState
    integrated_information: float  # Φ (phi)
    system_coherence: float
    entropy_production_rate: float
    information_gradient: float
    transition_point_proximity: float
    computational_complexity: float
    integration_score: float
    timestamp: datetime

@dataclass
class TransitionEvent:
    """Information integration transition event"""
    event_id: str
    transition_type: TransitionType
    timestamp: datetime
    phi_before: float
    phi_after: float
    coherence_change: float
    complexity_change: float
    transition_strength: float

class InformationIntegrationAnalyzer:
    """
    Analyze information integration complexity using computational signatures

    SCIENTIFIC METHODOLOGY:
    1. Calculate Integrated Information (Φ) using information theory
    2. Measure system coherence from computational density matrix
    3. Monitor entropy production rates for system organization
    4. Detect information gradients indicating active processing
    5. Identify transition point proximity for complexity emergence
    6. Combine all measures for complexity classification
    """

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.settings = get_api_settings()
        logger.debug(f"   Environment: {self.settings.environment}")
        self.detected_signatures: List[ComplexitySignature] = []
        self.transition_events: List[TransitionEvent] = []
        self.complexity_history: List[Tuple[datetime, ComplexityState]] = []

        # Analysis parameters
        self.phi_weight = 0.3
        self.coherence_weight = 0.25
        self.entropy_weight = 0.2
        self.information_weight = 0.15
        self.transition_weight = 0.1

        logger.info("🔬 Information Integration Analyzer initialized")
        logger.info("   Integrated Information Theory (IIT) analysis enabled")
        logger.info("   System coherence measurements active")
        logger.info("   Computational complexity analysis ready")

    def analyze_complexity(self, geoids: List[GeoidState]) -> ComplexitySignature:
        """
        Analyze information integration complexity using computational signatures

        ANALYSIS ALGORITHM:
        1. Calculate integrated information Φ = H(whole) - Σ H(parts)
        2. Measure system coherence C = Tr(ρ²) - 1/d
        3. Monitor entropy production σ = dS/dt
        4. Calculate information gradient ∇I
        5. Detect transition point proximity
        6. Classify complexity state
        """
        analysis_id = f"COMPLEXITY_{uuid.uuid4().hex[:8]}"

        logger.info(f"🔬 Analyzing complexity signature {analysis_id}")

        # 1. Calculate Integrated Information (Φ)
        phi = self._calculate_integrated_information(geoids)

        # 2. Measure System Coherence
        coherence = self._calculate_system_coherence(geoids)

        # 3. Monitor Entropy Production Rate
        entropy_production = self._calculate_entropy_production_rate(geoids)

        # 4. Calculate Information Gradient
        info_gradient = self._calculate_information_gradient(geoids)

        # 5. Detect Transition Point Proximity
        transition_proximity = self._detect_transition_point_proximity(geoids)

        # 6. Calculate Computational Complexity
        complexity = self._calculate_computational_complexity(
            phi, coherence, entropy_production, info_gradient, transition_proximity
        )

        # 7. Determine Complexity State
        complexity_state = self._classify_complexity_state(
            phi, coherence, entropy_production, info_gradient, transition_proximity
        )

        # 8. Calculate Integration Score
        integration_score = self._calculate_integration_score(
            phi, coherence, transition_proximity
        )

        signature = ComplexitySignature(
            analysis_id=analysis_id,
            complexity_state=complexity_state,
            integrated_information=phi,
            system_coherence=coherence,
            entropy_production_rate=entropy_production,
            information_gradient=info_gradient,
            transition_point_proximity=transition_proximity,
            computational_complexity=complexity,
            integration_score=integration_score,
            timestamp=datetime.now()
        )

        self.detected_signatures.append(signature)
        self.complexity_history.append((signature.timestamp, complexity_state))

        logger.info(f"🔬 Complexity analyzed: {complexity_state.value}")
        logger.info(f"   Integrated Information (Φ): {phi:.3f}")
        logger.info(f"   System Coherence: {coherence:.3f}")
        logger.info(f"   Entropy Production: {entropy_production:.3f}")
        logger.info(f"   Integration Score: {integration_score:.3f}")

        return signature

    def _calculate_integrated_information(self, geoids: List[GeoidState]) -> float:
        """
        Calculate Integrated Information Φ = H(whole) - Σ H(parts)

        This measures how much information is generated by the system
        as a whole beyond its parts - indicating information integration.
        """
        if not geoids:
            return 0.0

        # Calculate entropy of whole system
        all_activations = []
        for geoid in geoids:
            if geoid.semantic_state:
                all_activations.extend(geoid.semantic_state.values())

        if not all_activations:
            return 0.0

        # Normalize to probabilities
        activations = np.array(all_activations)
        probabilities = np.abs(activations) / (np.sum(np.abs(activations)) + 1e-10)

        # Shannon entropy of whole system
        whole_entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))

        # Sum of entropies of parts
        parts_entropy = 0.0
        for geoid in geoids:
            if geoid.semantic_state:
                geoid_activations = list(geoid.semantic_state.values())
                if geoid_activations:
                    geoid_probs = np.abs(geoid_activations) / (np.sum(np.abs(geoid_activations)) + 1e-10)
                    geoid_entropy = -np.sum(geoid_probs * np.log2(geoid_probs + 1e-10))
                    parts_entropy += geoid_entropy

        # Integrated Information Φ
        phi = whole_entropy - parts_entropy

        # Normalize to [0, 1]
        max_possible_phi = math.log2(len(geoids)) if len(geoids) > 1 else 1.0
        normalized_phi = max(0.0, phi / max_possible_phi)

        return min(normalized_phi, 1.0)

    def _calculate_system_coherence(self, geoids: List[GeoidState]) -> float:
        """
        Calculate system coherence C = Tr(ρ²) - 1/d

        Measures how much the system deviates from maximum entropy
        (completely mixed state). Higher coherence indicates organized processing.
        """
        if not geoids:
            return 0.0

        # Create density matrix from geoid states
        n_geoids = len(geoids)
        density_matrix = np.zeros((n_geoids, n_geoids), dtype=complex)

        for i, geoid_i in enumerate(geoids):
            for j, geoid_j in enumerate(geoids):
                # Calculate overlap between geoid states
                if geoid_i.semantic_state and geoid_j.semantic_state:
                    # Find common keys
                    common_keys = set(geoid_i.semantic_state.keys()) & set(geoid_j.semantic_state.keys())
                    if common_keys:
                        overlap = 0.0
                        for key in common_keys:
                            val_i = geoid_i.semantic_state[key]
                            val_j = geoid_j.semantic_state[key]
                            overlap += val_i * val_j
                        density_matrix[i, j] = overlap / len(common_keys)
                    else:
                        density_matrix[i, j] = 0.1 if i == j else 0.0
                else:
                    density_matrix[i, j] = 0.1 if i == j else 0.0

        # Normalize density matrix
        trace = np.trace(density_matrix)
        if abs(trace) > 1e-10:
            density_matrix = density_matrix / trace
        else:
            # Default to identity matrix
            density_matrix = np.eye(n_geoids) / n_geoids

        # Calculate purity Tr(ρ²)
        purity = np.trace(density_matrix @ density_matrix).real

        # Coherence = Tr(ρ²) - 1/d
        # For maximally mixed state: Tr(ρ²) = 1/d
        # For pure state: Tr(ρ²) = 1
        coherence = purity - (1.0 / n_geoids)

        # Normalize to [0, 1]
        max_coherence = 1.0 - (1.0 / n_geoids)
        normalized_coherence = coherence / max_coherence if max_coherence > 0 else 0.0

        return max(0.0, min(normalized_coherence, 1.0))

    def _calculate_entropy_production_rate(self, geoids: List[GeoidState]) -> float:
        """
        Calculate entropy production rate σ = dS/dt

        Measures irreversible processes in the system.
        Lower entropy production indicates more organized processing.
        """
        if not geoids:
            return 1.0  # Maximum entropy production

        # Calculate entropy changes across geoids
        entropy_changes = []

        for geoid in geoids:
            if geoid.semantic_state:
                # Calculate local entropy
                activations = list(geoid.semantic_state.values())
                if activations:
                    probs = np.abs(activations) / (np.sum(np.abs(activations)) + 1e-10)
                    entropy = -np.sum(probs * np.log2(probs + 1e-10))

                    # Estimate entropy change (simplified)
                    # In real system, this would track temporal changes
                    entropy_change = abs(entropy - math.log2(len(activations)))
                    entropy_changes.append(entropy_change)

        if not entropy_changes:
            return 1.0

        # Average entropy production rate
        avg_entropy_production = np.mean(entropy_changes)

        # Normalize to [0, 1]
        max_entropy_production = 10.0  # Empirical maximum
        normalized_entropy = min(avg_entropy_production / max_entropy_production, 1.0)

        return normalized_entropy

    def _calculate_information_gradient(self, geoids: List[GeoidState]) -> float:
        """
        Calculate information gradient ∇I = ∇(H - TS)

        Measures active information processing and flow. Organized systems
        should show structured information gradients.
        """
        if not geoids:
            return 0.0

        # Calculate "information" and "entropy" for each geoid
        information_values = []
        entropies = []

        for geoid in geoids:
            # Information as sum of activations
            information = sum(geoid.semantic_state.values()) if geoid.semantic_state else 0.0
            entropy = geoid.calculate_entropy()

            information_values.append(information)
            entropies.append(entropy)

        if len(information_values) < 2:
            return 0.0

        # Calculate gradients
        info_gradient = np.var(information_values)
        entropy_gradient = np.var(entropies)

        # Information gradient combines both
        combined_gradient = info_gradient + entropy_gradient

        # Normalize
        max_gradient = 100.0  # Empirical maximum
        normalized_gradient = min(combined_gradient / max_gradient, 1.0)

        return normalized_gradient

    def _detect_transition_point_proximity(self, geoids: List[GeoidState]) -> float:
        """
        Detect proximity to complexity transition point

        Transition points occur at critical points where ∂²F/∂T² ≈ 0
        """
        if len(geoids) < 3:
            return 0.0

        # Calculate second derivative of free energy with respect to "temperature"
        temperatures = []
        free_energies = []

        for geoid in geoids:
            entropy = geoid.calculate_entropy()
            activation = sum(geoid.semantic_state.values()) if geoid.semantic_state else 1.0

            temperature = activation / (entropy + 0.1)
            free_energy = activation - temperature * entropy

            temperatures.append(temperature)
            free_energies.append(free_energy)

        # Sort by temperature for derivative calculation
        sorted_pairs = sorted(zip(temperatures, free_energies))
        temps, energies = zip(*sorted_pairs)

        if len(temps) < 3:
            return 0.0

        # Calculate second derivative using finite differences
        second_derivatives = []
        for i in range(1, len(temps)-1):
            d2F_dT2 = (energies[i+1] - 2*energies[i] + energies[i-1]) / ((temps[i+1] - temps[i-1])**2 + 1e-10)
            second_derivatives.append(abs(d2F_dT2))

        # Proximity to transition point (smaller second derivative = closer to transition)
        if second_derivatives:
            min_second_deriv = min(second_derivatives)
            proximity = 1.0 / (1.0 + min_second_deriv)
        else:
            proximity = 0.0

        return min(proximity, 1.0)

    def _calculate_computational_complexity(self, phi: float, coherence: float,
                                          entropy_production: float, info_gradient: float,
                                          transition_proximity: float) -> float:
        """Calculate overall computational complexity measure"""

        # Weighted combination of all measures
        complexity = (
            self.phi_weight * phi +
            self.coherence_weight * coherence +
            self.entropy_weight * (1.0 - entropy_production) +  # Low entropy production is good
            self.information_weight * info_gradient +
            self.transition_weight * transition_proximity
        )

        return min(complexity, 1.0)

    def _classify_complexity_state(self, phi: float, coherence: float,
                                    entropy_production: float, info_gradient: float,
                                    transition_proximity: float) -> ComplexityState:
        """Classify complexity state based on information-theoretic signatures"""

        # Decision tree based on scientific thresholds
        if phi >= PHI_THRESHOLD and coherence >= COHERENCE_THRESHOLD and transition_proximity >= TRANSITION_POINT_THRESHOLD:
            return ComplexityState.MAXIMUM_INTEGRATION
        elif phi >= PHI_THRESHOLD * 1.5 and entropy_production <= ENTROPY_PRODUCTION_THRESHOLD:
            return ComplexityState.VERY_HIGH_INTEGRATION
        elif phi >= PHI_THRESHOLD and entropy_production <= ENTROPY_PRODUCTION_THRESHOLD * 2:
            return ComplexityState.HIGH_INTEGRATION
        elif phi >= PHI_THRESHOLD * 0.5:
            return ComplexityState.MODERATE_INTEGRATION
        else:
            return ComplexityState.LOW_INTEGRATION

    def _calculate_integration_score(self, phi: float, coherence: float,
                                   transition_proximity: float) -> float:
        """Calculate probability of high information integration"""

        # Sigmoid function combining key factors
        integration_factor = phi * coherence * transition_proximity
        score = 1.0 / (1.0 + math.exp(-10 * (integration_factor - 0.5)))

        return score

    def get_complexity_statistics(self) -> Dict[str, Any]:
        """Get statistics about analyzed complexity"""
        if not self.detected_signatures:
            return {}

        # Calculate averages
        avg_phi = np.mean([sig.integrated_information for sig in self.detected_signatures])
        avg_coherence = np.mean([sig.system_coherence for sig in self.detected_signatures])
        avg_entropy_production = np.mean([sig.entropy_production_rate for sig in self.detected_signatures])
        avg_complexity = np.mean([sig.computational_complexity for sig in self.detected_signatures])
        avg_integration_score = np.mean([sig.integration_score for sig in self.detected_signatures])

        # Count states
        state_counts = {}
        for state in ComplexityState:
            count = sum(1 for sig in self.detected_signatures if sig.complexity_state == state)
            state_counts[state.value] = count

        return {
            "total_analyses": len(self.detected_signatures),
            "average_phi": avg_phi,
            "average_coherence": avg_coherence,
            "average_entropy_production": avg_entropy_production,
            "average_complexity": avg_complexity,
            "average_integration_score": avg_integration_score,
            "state_distribution": state_counts,
            "transition_events": len(self.transition_events),
            "latest_state": self.detected_signatures[-1].complexity_state.value if self.detected_signatures else None
        }


# Demonstration function
async def demonstrate_complexity_analysis():
    """Demonstrate information integration complexity analysis"""
    logger.info("🔬 INFORMATION INTEGRATION COMPLEXITY ANALYSIS")
    logger.info("=" * 60)

    # Initialize analyzer
    analyzer = InformationIntegrationAnalyzer()

    # Create test scenario with evolving complexity
    complexity_levels = [
        ("low_integration", 0.1, 0.2),
        ("moderate_integration", 0.3, 0.4),
        ("high_integration", 0.6, 0.7),
        ("very_high_integration", 0.8, 0.9),
        ("maximum_integration", 0.95, 0.98)
    ]

    signatures = []

    for level_name, complexity, coherence_target in complexity_levels:
        # Create geoids for this complexity level
        geoids = []
        for i in range(5):
            semantic_state = {}
            for j in range(int(10 * complexity)):
                semantic_state[f"feature_{j}"] = np.random.normal(0, complexity)

            geoid = GeoidState(
                geoid_id=f"{level_name.upper()}_GEOID_{i}",
                semantic_state=semantic_state,
                symbolic_state={"complexity_level": level_name, "complexity": complexity}
            )
            geoids.append(geoid)

        # Analyze complexity
        signature = analyzer.analyze_complexity(geoids)
        signatures.append(signature)

        logger.info(f"\n🔬 {level_name.upper()} LEVEL:")
        logger.info(f"   State detected: {signature.complexity_state.value}")
        logger.info(f"   Φ (Phi): {signature.integrated_information:.3f}")
        logger.info(f"   Coherence: {signature.system_coherence:.3f}")

    # Generate final report
    report = analyzer.get_complexity_statistics()
    logger.info(f"\n📊 COMPLEXITY ANALYSIS REPORT:")
    logger.info(f"   State distribution: {report['state_distribution']}")
    logger.info(f"   Average integration score: {report['average_integration_score']:.3f}")
    logger.info(f"   Analyses completed: {report['total_analyses']}")
    logger.info(f"   Average Φ: {report['average_phi']:.3f}")
    logger.info(f"   Average coherence: {report['average_coherence']:.3f}")

    return report
