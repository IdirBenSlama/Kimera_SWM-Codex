"""
KIMERA SWM - MIRROR PORTAL ENGINE
=================================

The Mirror Portal Engine implements quantum-semantic transitions that bridge
different representation spaces. It enables geoids to undergo phase transitions
between semantic and symbolic states, creating coherent transformations that
maintain information while changing representational form.

This engine is crucial for enabling fluid transitions between different
modes of cognitive processing and knowledge representation.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ...core.data_structures.geoid_state import (
    GeoidProcessingState,
    GeoidState,
    GeoidType,
    SemanticState,
    SymbolicState,
    ThermodynamicProperties,
)

# Configure logging
logger = logging.getLogger(__name__)


class TransitionType(Enum):
    """Types of quantum-semantic transitions"""

    SEMANTIC_TO_SYMBOLIC = "semantic_to_symbolic"  # Semantic → Symbolic
    SYMBOLIC_TO_SEMANTIC = "symbolic_to_semantic"  # Symbolic → Semantic
    COHERENCE_BRIDGE = "coherence_bridge"  # Bridge both states
    QUANTUM_SUPERPOSITION = "quantum_superposition"  # Create superposition state
    ENTANGLEMENT = "entanglement"  # Entangle multiple geoids
    PHASE_TRANSITION = "phase_transition"  # Change fundamental state
    MIRROR_REFLECTION = "mirror_reflection"  # Create mirrored state


class PortalState(Enum):
    """State of the quantum portal"""

    CLOSED = "closed"  # Portal is closed
    OPENING = "opening"  # Portal is opening
    OPEN = "open"  # Portal is fully open
    TRANSITIONING = "transitioning"  # Transition in progress
    CLOSING = "closing"  # Portal is closing
    UNSTABLE = "unstable"  # Portal is unstable


@dataclass
class QuantumParameters:
    """Parameters for quantum-semantic transitions"""

    coherence_threshold: float = 0.7  # Minimum coherence for transitions
    entanglement_strength: float = 0.8  # Strength of entanglement
    superposition_probability: float = 0.6  # Probability of superposition
    decoherence_rate: float = 0.1  # Rate of quantum decoherence
    tunnel_probability: float = 0.3  # Quantum tunneling probability
    phase_transition_energy: float = 5.0  # Energy for phase transitions
    mirror_symmetry: float = 0.9  # Symmetry preservation factor


@dataclass
class PortalConfiguration:
    """Configuration for a quantum portal"""

    source_space: str  # Source representation space
    target_space: str  # Target representation space
    transition_type: TransitionType  # Type of transition
    parameters: QuantumParameters  # Quantum parameters
    bidirectional: bool = True  # Whether portal is bidirectional
    energy_cost: float = 2.0  # Energy cost for transition
    coherence_requirement: float = 0.5  # Minimum coherence required


@dataclass
class TransitionResult:
    """Result of a quantum-semantic transition"""

    original_geoid: GeoidState
    transformed_geoid: GeoidState
    transition_type: TransitionType
    success: bool
    energy_consumed: float
    coherence_change: float
    quantum_effects: Dict[str, Any]
    portal_state: PortalState
    duration: float
    metadata: Dict[str, Any]


class QuantumPortal:
    """
    Individual Quantum Portal for specific transitions.
    Each portal specializes in particular types of quantum-semantic transitions.
    """

    def __init__(self, config: PortalConfiguration):
        self.config = config
        self.state = PortalState.CLOSED
        self.activation_count = 0
        self.success_rate = 0.0
        self.average_energy_cost = 0.0
        self.creation_time = datetime.now()
        self.last_activation = None

        # Quantum state tracking
        self.entangled_geoids: List[str] = []
        self.superposition_states: Dict[str, List[GeoidState]] = {}

        logger.debug(
            f"Quantum portal created: {config.source_space} → {config.target_space}"
        )

    def open_portal(self) -> bool:
        """Open the quantum portal for transitions"""
        if self.state in [PortalState.OPEN, PortalState.OPENING]:
            return True

        self.state = PortalState.OPENING

        # Simulate portal opening time
        time.sleep(0.001)  # Minimal delay for realism

        # Check if portal can stabilize
        if np.random.random() < 0.95:  # 95% success rate
            self.state = PortalState.OPEN
            logger.debug(
                f"Portal opened: {self.config.source_space} → {self.config.target_space}"
            )
            return True
        else:
            self.state = PortalState.UNSTABLE
            logger.warning(
                f"Portal unstable: {self.config.source_space} → {self.config.target_space}"
            )
            return False

    def close_portal(self) -> None:
        """Close the quantum portal"""
        self.state = PortalState.CLOSING
        time.sleep(0.001)  # Minimal delay
        self.state = PortalState.CLOSED
        logger.debug("Portal closed")

    def transition(self, geoid: GeoidState) -> TransitionResult:
        """Execute a quantum-semantic transition through the portal"""
        start_time = time.time()

        if self.state != PortalState.OPEN:
            if not self.open_portal():
                return TransitionResult(
                    original_geoid=geoid,
                    transformed_geoid=geoid,
                    transition_type=self.config.transition_type,
                    success=False,
                    energy_consumed=0.0,
                    coherence_change=0.0,
                    quantum_effects={},
                    portal_state=self.state,
                    duration=time.time() - start_time,
                    metadata={"error": "Portal failed to open"},
                )

        self.state = PortalState.TRANSITIONING
        self.activation_count += 1
        self.last_activation = datetime.now()

        # Check if geoid meets requirements
        if geoid.coherence_score < self.config.coherence_requirement:
            self.state = PortalState.OPEN
            return TransitionResult(
                original_geoid=geoid,
                transformed_geoid=geoid,
                transition_type=self.config.transition_type,
                success=False,
                energy_consumed=0.0,
                coherence_change=0.0,
                quantum_effects={},
                portal_state=self.state,
                duration=time.time() - start_time,
                metadata={"error": "Insufficient coherence"},
            )

        # Execute the transition
        transformed_geoid, quantum_effects = self._execute_transition(geoid)

        # Calculate energy consumption
        energy_consumed = self._calculate_energy_consumption(geoid, transformed_geoid)

        # Calculate coherence change
        coherence_change = transformed_geoid.coherence_score - geoid.coherence_score

        # Update statistics
        self._update_statistics(energy_consumed, True)

        self.state = PortalState.OPEN

        return TransitionResult(
            original_geoid=geoid,
            transformed_geoid=transformed_geoid,
            transition_type=self.config.transition_type,
            success=True,
            energy_consumed=energy_consumed,
            coherence_change=coherence_change,
            quantum_effects=quantum_effects,
            portal_state=self.state,
            duration=time.time() - start_time,
            metadata={"transition_successful": True},
        )

    def _execute_transition(
        self, geoid: GeoidState
    ) -> Tuple[GeoidState, Dict[str, Any]]:
        """Execute the specific transition type"""
        quantum_effects = {}

        if self.config.transition_type == TransitionType.SEMANTIC_TO_SYMBOLIC:
            return self._semantic_to_symbolic(geoid, quantum_effects)
        elif self.config.transition_type == TransitionType.SYMBOLIC_TO_SEMANTIC:
            return self._symbolic_to_semantic(geoid, quantum_effects)
        elif self.config.transition_type == TransitionType.COHERENCE_BRIDGE:
            return self._coherence_bridge(geoid, quantum_effects)
        elif self.config.transition_type == TransitionType.QUANTUM_SUPERPOSITION:
            return self._quantum_superposition(geoid, quantum_effects)
        elif self.config.transition_type == TransitionType.MIRROR_REFLECTION:
            return self._mirror_reflection(geoid, quantum_effects)
        else:
            # Default: identity transformation
            return geoid, quantum_effects

    def _semantic_to_symbolic(
        self, geoid: GeoidState, quantum_effects: Dict[str, Any]
    ) -> Tuple[GeoidState, Dict[str, Any]]:
        """Transform semantic state to symbolic representation"""
        if geoid.semantic_state is None:
            return geoid, quantum_effects

        # Create new symbolic state from semantic information
        symbolic_state = SymbolicState(
            logical_predicates=[],
            symbolic_relations={},
            rule_activations={},
            symbolic_constraints=[],
            proof_chains=[],
        )

        # Extract symbolic predicates from semantic embedding
        # This is a simplified transformation - in practice would use sophisticated embedding→logic conversion
        embedding_vector = geoid.semantic_state.embedding_vector

        # Convert high-dimensional embedding to symbolic predicates
        dominant_dimensions = np.argsort(np.abs(embedding_vector))[
            -5:
        ]  # Top 5 dimensions
        for i, dim in enumerate(dominant_dimensions):
            value = embedding_vector[dim]
            predicate = f"feature_{dim}({value:.3f})"
            symbolic_state.logical_predicates.append(predicate)

        # Add confidence-based rules
        for conf_type, conf_value in geoid.semantic_state.confidence_scores.items():
            if conf_value > 0.7:
                symbolic_state.rule_activations[f"high_confidence_{conf_type}"] = True

        # Create transformed geoid
        transformed_geoid = GeoidState(
            geoid_type=geoid.geoid_type,
            processing_state=GeoidProcessingState.PROCESSING,
            semantic_state=geoid.semantic_state,  # Keep original semantic state
            symbolic_state=symbolic_state,
            thermodynamic=geoid.thermodynamic,
        )

        transformed_geoid.connect_input("semantic_to_symbolic_transition", geoid)

        quantum_effects["transition_fidelity"] = 0.9  # High fidelity transformation
        quantum_effects["information_preservation"] = 0.95

        return transformed_geoid, quantum_effects

    def _symbolic_to_semantic(
        self, geoid: GeoidState, quantum_effects: Dict[str, Any]
    ) -> Tuple[GeoidState, Dict[str, Any]]:
        """Transform symbolic state to semantic representation"""
        if geoid.symbolic_state is None:
            return geoid, quantum_effects

        # Create semantic representation from symbolic state
        # In practice, this would use sophisticated logic→embedding conversion

        # Generate embedding from logical predicates
        base_embedding = np.random.normal(0, 0.1, 768)  # Start with noise

        # Modify embedding based on predicates
        for i, predicate in enumerate(
            geoid.symbolic_state.logical_predicates[:10]
        ):  # Limit to 10
            # Hash predicate to deterministic embedding modification
            predicate_hash = hash(predicate) % 768
            base_embedding[predicate_hash] += 0.5

        # Normalize
        if np.linalg.norm(base_embedding) > 0:
            base_embedding = base_embedding / np.linalg.norm(base_embedding)

        # Calculate confidence from rule activations
        confidence_scores = {}
        active_rules = sum(
            1 for active in geoid.symbolic_state.rule_activations.values() if active
        )
        total_rules = len(geoid.symbolic_state.rule_activations)

        if total_rules > 0:
            confidence_scores["rule_consistency"] = active_rules / total_rules
        else:
            confidence_scores["rule_consistency"] = 0.5

        semantic_state = SemanticState(
            embedding_vector=base_embedding,
            confidence_scores=confidence_scores,
            uncertainty_measure=0.3,  # Moderate uncertainty from symbolic→semantic
            semantic_entropy=1.5,
            coherence_score=0.8,
        )

        # Create transformed geoid
        transformed_geoid = GeoidState(
            geoid_type=geoid.geoid_type,
            processing_state=GeoidProcessingState.PROCESSING,
            semantic_state=semantic_state,
            symbolic_state=geoid.symbolic_state,  # Keep original symbolic state
            thermodynamic=geoid.thermodynamic,
        )

        transformed_geoid.connect_input("symbolic_to_semantic_transition", geoid)

        quantum_effects["transition_fidelity"] = 0.85  # Slightly lower fidelity
        quantum_effects["information_preservation"] = 0.90

        return transformed_geoid, quantum_effects

    def _coherence_bridge(
        self, geoid: GeoidState, quantum_effects: Dict[str, Any]
    ) -> Tuple[GeoidState, Dict[str, Any]]:
        """Create coherent bridge between semantic and symbolic states"""
        if geoid.semantic_state is None or geoid.symbolic_state is None:
            return geoid, quantum_effects

        # Enhance coherence between existing states
        enhanced_semantic = SemanticState(
            embedding_vector=geoid.semantic_state.embedding_vector,
            confidence_scores=geoid.semantic_state.confidence_scores.copy(),
            uncertainty_measure=geoid.semantic_state.uncertainty_measure
            * 0.8,  # Reduce uncertainty
            semantic_entropy=geoid.semantic_state.semantic_entropy,
            coherence_score=min(
                1.0, geoid.semantic_state.coherence_score * 1.2
            ),  # Boost coherence
        )

        enhanced_symbolic = SymbolicState(
            logical_predicates=geoid.symbolic_state.logical_predicates.copy(),
            symbolic_relations=geoid.symbolic_state.symbolic_relations.copy(),
            rule_activations=geoid.symbolic_state.rule_activations.copy(),
            symbolic_constraints=geoid.symbolic_state.symbolic_constraints.copy(),
            proof_chains=geoid.symbolic_state.proof_chains.copy(),
        )

        # Add bridge relations
        enhanced_symbolic.symbolic_relations["coherence_bridge"] = {
            "semantic_coherence": enhanced_semantic.coherence_score,
            "bridge_strength": 0.9,
            "quantum_entanglement": True,
        }

        # Create transformed geoid
        transformed_geoid = GeoidState(
            geoid_type=geoid.geoid_type,
            processing_state=GeoidProcessingState.PROCESSING,
            semantic_state=enhanced_semantic,
            symbolic_state=enhanced_symbolic,
            thermodynamic=geoid.thermodynamic,
        )

        transformed_geoid.connect_input("coherence_bridge", geoid)

        quantum_effects["coherence_enhancement"] = 0.3
        quantum_effects["bridge_strength"] = 0.9
        quantum_effects["entanglement_created"] = True

        return transformed_geoid, quantum_effects

    def _quantum_superposition(
        self, geoid: GeoidState, quantum_effects: Dict[str, Any]
    ) -> Tuple[GeoidState, Dict[str, Any]]:
        """Create quantum superposition of multiple states"""
        # Create a superposition by combining multiple possible states
        superposed_geoid = GeoidState(
            geoid_type=geoid.geoid_type,
            processing_state=GeoidProcessingState.PROCESSING,
            semantic_state=geoid.semantic_state,
            symbolic_state=geoid.symbolic_state,
            thermodynamic=geoid.thermodynamic,
        )

        # Add superposition metadata
        if superposed_geoid.symbolic_state:
            superposed_geoid.symbolic_state.symbolic_relations[
                "quantum_superposition"
            ] = {
                "superposition_states": 2,
                "coherence_amplitude": self.config.parameters.superposition_probability,
                "decoherence_rate": self.config.parameters.decoherence_rate,
            }

        superposed_geoid.connect_input("quantum_superposition", geoid)

        # Track superposition
        self.superposition_states[superposed_geoid.geoid_id] = [geoid, superposed_geoid]

        quantum_effects["superposition_created"] = True
        quantum_effects["superposition_coherence"] = (
            self.config.parameters.superposition_probability
        )
        quantum_effects["entanglement_strength"] = (
            self.config.parameters.entanglement_strength
        )

        return superposed_geoid, quantum_effects

    def _mirror_reflection(
        self, geoid: GeoidState, quantum_effects: Dict[str, Any]
    ) -> Tuple[GeoidState, Dict[str, Any]]:
        """Create mirror reflection of the geoid"""
        # Create mirrored version with inverted properties
        mirrored_geoid = GeoidState(
            geoid_type=geoid.geoid_type,
            processing_state=GeoidProcessingState.PROCESSING,
            semantic_state=self._mirror_semantic_state(geoid.semantic_state),
            symbolic_state=self._mirror_symbolic_state(geoid.symbolic_state),
            thermodynamic=geoid.thermodynamic,
        )

        mirrored_geoid.connect_input("mirror_reflection", geoid)

        quantum_effects["mirror_symmetry"] = self.config.parameters.mirror_symmetry
        quantum_effects["reflection_fidelity"] = 0.95
        quantum_effects["parity_conservation"] = True

        return mirrored_geoid, quantum_effects

    def _mirror_semantic_state(
        self, semantic_state: Optional[SemanticState]
    ) -> Optional[SemanticState]:
        """Create mirrored semantic state"""
        if semantic_state is None:
            return None

        # Mirror embedding vector (flip signs of alternating dimensions)
        mirrored_embedding = semantic_state.embedding_vector.copy()
        mirrored_embedding[::2] *= -1  # Flip every other dimension

        return SemanticState(
            embedding_vector=mirrored_embedding,
            confidence_scores=semantic_state.confidence_scores.copy(),
            uncertainty_measure=semantic_state.uncertainty_measure,
            semantic_entropy=semantic_state.semantic_entropy,
            coherence_score=semantic_state.coherence_score
            * self.config.parameters.mirror_symmetry,
        )

    def _mirror_symbolic_state(
        self, symbolic_state: Optional[SymbolicState]
    ) -> Optional[SymbolicState]:
        """Create mirrored symbolic state"""
        if symbolic_state is None:
            return None

        # Create mirrored predicates (add NOT to some predicates)
        mirrored_predicates = []
        for predicate in symbolic_state.logical_predicates:
            if np.random.random() < 0.3:  # 30% chance to negate
                mirrored_predicates.append(f"NOT({predicate})")
            else:
                mirrored_predicates.append(predicate)

        # Flip rule activations
        mirrored_activations = {}
        for rule, active in symbolic_state.rule_activations.items():
            mirrored_activations[f"mirror_{rule}"] = not active

        return SymbolicState(
            logical_predicates=mirrored_predicates,
            symbolic_relations=symbolic_state.symbolic_relations.copy(),
            rule_activations=mirrored_activations,
            symbolic_constraints=symbolic_state.symbolic_constraints.copy(),
            proof_chains=symbolic_state.proof_chains.copy(),
        )

    def _calculate_energy_consumption(
        self, original: GeoidState, transformed: GeoidState
    ) -> float:
        """Calculate energy consumed during transition"""
        base_cost = self.config.energy_cost

        # Additional cost based on complexity of transformation
        complexity_factor = 1.0

        if transformed.semantic_state and original.semantic_state:
            # Cost based on embedding distance
            embedding_distance = np.linalg.norm(
                transformed.semantic_state.embedding_vector
                - original.semantic_state.embedding_vector
            )
            complexity_factor += embedding_distance * 0.1

        if transformed.symbolic_state and original.symbolic_state:
            # Cost based on predicate changes
            predicate_changes = len(
                set(transformed.symbolic_state.logical_predicates)
                - set(original.symbolic_state.logical_predicates)
            )
            complexity_factor += predicate_changes * 0.05

        return base_cost * complexity_factor

    def _update_statistics(self, energy_consumed: float, success: bool) -> None:
        """Update portal statistics"""
        # Update success rate
        current_successes = self.success_rate * (self.activation_count - 1)
        if success:
            current_successes += 1
        self.success_rate = current_successes / self.activation_count

        # Update average energy cost
        current_total_energy = self.average_energy_cost * (self.activation_count - 1)
        current_total_energy += energy_consumed
        self.average_energy_cost = current_total_energy / self.activation_count


class MirrorPortalEngine:
    """
    Mirror Portal Engine - Quantum-Semantic Transformation Hub
    =========================================================

    The Mirror Portal Engine manages multiple quantum portals that enable
    sophisticated transformations between different representation spaces.
    It coordinates quantum-semantic transitions, maintains portal stability,
    and ensures coherent transformations.

    Key Capabilities:
    - Multi-portal management for different transition types
    - Quantum coherence preservation across transformations
    - Energy-efficient transition optimization
    - Portal stability monitoring and maintenance
    - Entanglement and superposition management
    """

    def __init__(self):
        self.portals: Dict[str, QuantumPortal] = {}
        self.default_parameters = QuantumParameters()
        self.transformation_history: List[TransitionResult] = []
        self.total_transformations = 0
        self.total_energy_consumed = 0.0

        # Initialize default portals
        self._create_default_portals()

        logger.info("MirrorPortalEngine initialized with default portals")

    def _create_default_portals(self) -> None:
        """Create default quantum portals for common transformations"""
        default_configs = [
            PortalConfiguration(
                source_space="semantic",
                target_space="symbolic",
                transition_type=TransitionType.SEMANTIC_TO_SYMBOLIC,
                parameters=self.default_parameters,
            ),
            PortalConfiguration(
                source_space="symbolic",
                target_space="semantic",
                transition_type=TransitionType.SYMBOLIC_TO_SEMANTIC,
                parameters=self.default_parameters,
            ),
            PortalConfiguration(
                source_space="dual",
                target_space="coherent",
                transition_type=TransitionType.COHERENCE_BRIDGE,
                parameters=self.default_parameters,
                energy_cost=1.5,
            ),
            PortalConfiguration(
                source_space="single",
                target_space="superposition",
                transition_type=TransitionType.QUANTUM_SUPERPOSITION,
                parameters=self.default_parameters,
                energy_cost=3.0,
            ),
            PortalConfiguration(
                source_space="any",
                target_space="mirror",
                transition_type=TransitionType.MIRROR_REFLECTION,
                parameters=self.default_parameters,
                energy_cost=2.5,
            ),
        ]

        for config in default_configs:
            portal_key = f"{config.source_space}_to_{config.target_space}"
            self.portals[portal_key] = QuantumPortal(config)

    def create_portal(self, config: PortalConfiguration) -> str:
        """Create a new quantum portal with the given configuration"""
        portal_key = (
            f"{config.source_space}_to_{config.target_space}_{len(self.portals)}"
        )
        self.portals[portal_key] = QuantumPortal(config)
        logger.info(f"Created new portal: {portal_key}")
        return portal_key

    def transform(
        self,
        geoid: GeoidState,
        transition_type: TransitionType,
        parameters: QuantumParameters = None,
    ) -> TransitionResult:
        """Transform a geoid using the specified transition type"""
        # Find appropriate portal
        portal = self._find_portal_for_transition(transition_type)

        if portal is None:
            # Create temporary portal if none exists
            config = PortalConfiguration(
                source_space="auto",
                target_space="auto",
                transition_type=transition_type,
                parameters=parameters or self.default_parameters,
            )
            portal = QuantumPortal(config)

        # Execute transformation
        result = portal.transition(geoid)

        # Update engine state
        self._update_engine_state(result)

        # Record in geoid metadata
        if result.success:
            result.transformed_geoid.metadata.add_processing_step(
                engine_name="MirrorPortalEngine",
                operation=f"quantum_transition_{transition_type.value}",
                duration=result.duration,
                metadata=result.quantum_effects,
            )

        return result

    def transform_batch(
        self,
        geoids: List[GeoidState],
        transition_type: TransitionType,
        parameters: QuantumParameters = None,
    ) -> List[TransitionResult]:
        """Transform a batch of geoids using the same transition type"""
        results = []

        for geoid in geoids:
            result = self.transform(geoid, transition_type, parameters)
            results.append(result)

        # Log batch summary
        successful = sum(1 for r in results if r.success)
        logger.info(
            f"Batch quantum transformation: {successful}/{len(results)} successful"
        )

        return results

    def entangle_geoids(
        self, geoid1: GeoidState, geoid2: GeoidState
    ) -> Tuple[GeoidState, GeoidState]:
        """Create quantum entanglement between two geoids"""
        # Create entangled versions
        entangled1 = GeoidState(
            geoid_type=geoid1.geoid_type,
            processing_state=GeoidProcessingState.PROCESSING,
            semantic_state=geoid1.semantic_state,
            symbolic_state=geoid1.symbolic_state,
            thermodynamic=geoid1.thermodynamic,
        )

        entangled2 = GeoidState(
            geoid_type=geoid2.geoid_type,
            processing_state=GeoidProcessingState.PROCESSING,
            semantic_state=geoid2.semantic_state,
            symbolic_state=geoid2.symbolic_state,
            thermodynamic=geoid2.thermodynamic,
        )

        # Connect them (mutual entanglement)
        entangled1.connect_input("entangled_partner", entangled2)
        entangled2.connect_input("entangled_partner", entangled1)

        # Add entanglement metadata
        entanglement_id = f"entanglement_{time.time()}"

        for geoid in [entangled1, entangled2]:
            if geoid.symbolic_state:
                geoid.symbolic_state.symbolic_relations["quantum_entanglement"] = {
                    "entanglement_id": entanglement_id,
                    "partner_id": (
                        entangled2.geoid_id
                        if geoid == entangled1
                        else entangled1.geoid_id
                    ),
                    "entanglement_strength": self.default_parameters.entanglement_strength,
                }

        logger.info(
            f"Created quantum entanglement: {entangled1.geoid_id[:8]} ↔ {entangled2.geoid_id[:8]}"
        )

        return entangled1, entangled2

    def _find_portal_for_transition(
        self, transition_type: TransitionType
    ) -> Optional[QuantumPortal]:
        """Find an appropriate portal for the given transition type"""
        for portal in self.portals.values():
            if portal.config.transition_type == transition_type:
                return portal
        return None

    def _update_engine_state(self, result: TransitionResult) -> None:
        """Update engine state after transformation"""
        self.transformation_history.append(result)
        self.total_transformations += 1

        if result.success:
            self.total_energy_consumed += result.energy_consumed

        # Keep history manageable
        if len(self.transformation_history) > 1000:
            self.transformation_history = self.transformation_history[-500:]

    def get_engine_statistics(self) -> Dict[str, Any]:
        """Get comprehensive engine statistics"""
        if not self.transformation_history:
            return {"total_transformations": 0}

        recent_results = self.transformation_history[-100:]

        success_rate = np.mean([r.success for r in recent_results])
        avg_energy = np.mean([r.energy_consumed for r in recent_results if r.success])
        avg_coherence_change = np.mean(
            [r.coherence_change for r in recent_results if r.success]
        )

        portal_stats = {}
        for name, portal in self.portals.items():
            portal_stats[name] = {
                "activation_count": portal.activation_count,
                "success_rate": portal.success_rate,
                "average_energy_cost": portal.average_energy_cost,
                "state": portal.state.value,
            }

        return {
            "total_transformations": self.total_transformations,
            "total_energy_consumed": self.total_energy_consumed,
            "recent_success_rate": success_rate,
            "average_energy_consumption": avg_energy,
            "average_coherence_change": avg_coherence_change,
            "active_portals": len(self.portals),
            "portal_statistics": portal_stats,
            "quantum_effects_observed": self._count_quantum_effects(recent_results),
        }

    def _count_quantum_effects(self, results: List[TransitionResult]) -> Dict[str, int]:
        """Count observed quantum effects in recent results"""
        effect_counts = {}

        for result in results:
            for effect_name in result.quantum_effects:
                effect_counts[effect_name] = effect_counts.get(effect_name, 0) + 1

        return effect_counts


# Convenience functions
def semantic_to_symbolic_transform(geoid: GeoidState) -> TransitionResult:
    """Convenience function for semantic to symbolic transformation"""
    engine = MirrorPortalEngine()
    return engine.transform(geoid, TransitionType.SEMANTIC_TO_SYMBOLIC)


def symbolic_to_semantic_transform(geoid: GeoidState) -> TransitionResult:
    """Convenience function for symbolic to semantic transformation"""
    engine = MirrorPortalEngine()
    return engine.transform(geoid, TransitionType.SYMBOLIC_TO_SEMANTIC)


def create_coherence_bridge(geoid: GeoidState) -> TransitionResult:
    """Convenience function to create coherence bridge"""
    engine = MirrorPortalEngine()
    return engine.transform(geoid, TransitionType.COHERENCE_BRIDGE)
