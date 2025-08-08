"""
KIMERA SWM - FOUNDATIONAL GEOID STATE
=====================================

The GeoidState represents the atomic unit of knowledge in the Kimera SWM system.
It bridges symbolic and subsymbolic processing through a unified data structure
that maintains both semantic (probabilistic) and symbolic (logical) representations.

This is the foundational building block upon which all cognitive processing is built.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class GeoidType(Enum):
    """Classification of geoid types for processing optimization"""

    CONCEPT = "concept"  # Abstract conceptual knowledge
    ENTITY = "entity"  # Concrete entities and objects
    RELATION = "relation"  # Relationships between entities
    PROCESS = "process"  # Dynamic processes and actions
    EMOTION = "emotion"  # Emotional states and affect
    MEMORY = "memory"  # Episodic and semantic memories
    HYPOTHESIS = "hypothesis"  # Scientific hypotheses and theories
    CONTRADICTION = "contradiction"  # Detected contradictions/tensions


class GeoidProcessingState(Enum):
    """Current processing state of the geoid"""

    CREATED = "created"  # Newly created, not processed
    PROCESSING = "processing"  # Currently being processed by engines
    STABLE = "stable"  # Reached stable state
    EVOLVING = "evolving"  # Actively evolving through thermodynamics
    CONTRADICTORY = "contradictory"  # Contains unresolved contradictions
    ARCHIVED = "archived"  # No longer actively processed


@dataclass
class SemanticState:
    """Auto-generated class."""
    pass
    """
    Probabilistic semantic representation of the geoid's meaning.
    Captures the fuzzy, continuous aspects of knowledge.
    """

    embedding_vector: np.ndarray  # High-dimensional semantic embedding
    confidence_scores: Dict[str, float]  # Confidence in various semantic aspects
    uncertainty_measure: float  # Overall uncertainty [0.0, 1.0]
    semantic_entropy: float  # Information-theoretic entropy
    coherence_score: float  # Internal semantic coherence [0.0, 1.0]

    def __post_init__(self):
        """Validate semantic state parameters"""
        if not 0.0 <= self.uncertainty_measure <= 1.0:
            raise ValueError("uncertainty_measure must be between 0.0 and 1.0")
        if not 0.0 <= self.coherence_score <= 1.0:
            raise ValueError("coherence_score must be between 0.0 and 1.0")


@dataclass
class SymbolicState:
    """Auto-generated class."""
    pass
    """
    Logical symbolic representation of the geoid's meaning.
    Captures the discrete, rule-based aspects of knowledge.
    """

    logical_predicates: List[str]  # First-order logic predicates
    symbolic_relations: Dict[str, Any]  # Symbolic relationship mappings
    rule_activations: Dict[str, bool]  # Which logical rules are active
    symbolic_constraints: List[str]  # Logical constraints and invariants
    proof_chains: List[Tuple[str, str]]  # Logical proof derivations

    def add_predicate(self, predicate: str) -> None:
        """Add a logical predicate to this symbolic state"""
        if predicate not in self.logical_predicates:
            self.logical_predicates.append(predicate)

    def activate_rule(self, rule_name: str) -> None:
        """Activate a symbolic rule"""
        self.rule_activations[rule_name] = True


@dataclass
class ThermodynamicProperties:
    """Auto-generated class."""
    pass
    """
    Thermodynamic properties for physics-compliant evolution.
    Enables the geoid to evolve according to thermodynamic principles.
    """

    cognitive_temperature: float  # "Temperature" of cognitive state
    information_entropy: float  # Thermodynamic entropy
    free_energy: float  # Available cognitive energy
    activation_energy: float  # Energy required for state transitions
    dissipation_rate: float  # Rate of energy dissipation
    equilibrium_tendency: float  # Tendency toward equilibrium state

    def calculate_evolution_probability(self, target_state: "GeoidState") -> float:
        """Calculate probability of evolution to target state using Boltzmann statistics"""
        if target_state.thermodynamic is None:
            return 0.0

        energy_diff = target_state.thermodynamic.free_energy - self.free_energy
        if self.cognitive_temperature <= 0:
            return 0.0 if energy_diff > 0 else 1.0

        # Boltzmann factor for cognitive state transitions
        return np.exp(-energy_diff / self.cognitive_temperature)


@dataclass
class ProcessingMetadata:
    """Auto-generated class."""
    pass
    """
    Metadata tracking the geoid's journey through the system.
    Essential for debugging, monitoring, and system transparency.
    """

    creation_timestamp: datetime  # When the geoid was created
    last_modified: datetime  # Last modification time
    processing_history: List[Dict[str, Any]]  # History of engine processing
    source_engine: Optional[str]  # Engine that created this geoid
    processing_depth: int  # How many processing steps
    parent_geoids: List[str]  # Parent geoid IDs (for derivation)
    child_geoids: List[str]  # Child geoid IDs (for evolution)
    processing_flags: Dict[str, bool]  # Various processing status flags

    def add_processing_step(
        self
        engine_name: str
        operation: str
        duration: float
        metadata: Dict[str, Any] = None
    ) -> None:
        """Record a processing step in the geoid's history"""
        step = {
            "timestamp": datetime.now(),
            "engine": engine_name
            "operation": operation
            "duration_ms": duration * 1000
            "metadata": metadata or {},
        }
        self.processing_history.append(step)
        self.last_modified = datetime.now()
        self.processing_depth += 1


@dataclass
class GeoidState:
    """Auto-generated class."""
    pass
    """
    The Foundational Geoid State - Atomic Unit of Knowledge
    =======================================================

    A geoid represents a single unit of knowledge that bridges symbolic and
    subsymbolic processing. It maintains dual representations (semantic and symbolic)
    while providing thermodynamic evolution capabilities.

    Core Design Principles:
    - Dual-State Architecture: Both probabilistic and logical representations
    - Physics-Compliant Evolution: Thermodynamic principles guide state changes
    - Complete Traceability: Full processing history and provenance tracking
    - Engine Interoperability: Designed for seamless engine interactions
    - Transparent Processing: All operations are logged and auditable
    """

    # Core Identity
    geoid_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    geoid_type: GeoidType = GeoidType.CONCEPT
    processing_state: GeoidProcessingState = GeoidProcessingState.CREATED

    # Dual Representation States
    semantic_state: Optional[SemanticState] = None
    symbolic_state: Optional[SymbolicState] = None

    # Physics-Compliant Evolution
    thermodynamic: Optional[ThermodynamicProperties] = None

    # System Integration
    metadata: ProcessingMetadata = field(
        default_factory=lambda: ProcessingMetadata(
            creation_timestamp=datetime.now(),
            last_modified=datetime.now(),
            processing_history=[],
            source_engine=None
            processing_depth=0
            parent_geoids=[],
            child_geoids=[],
            processing_flags={},
        )
    )

    # Interconnection and Flow
    input_connections: Dict[str, "GeoidState"] = field(default_factory=dict)
    output_connections: Dict[str, "GeoidState"] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize computed properties and validate state"""
        if self.semantic_state is None and self.symbolic_state is None:
            raise ValueError(
                "GeoidState must have at least one of semantic_state or symbolic_state"
            )

    @property
    def is_complete(self) -> bool:
        """Check if geoid has both semantic and symbolic representations"""
        return self.semantic_state is not None and self.symbolic_state is not None

    @property
    def coherence_score(self) -> float:
        """Calculate overall coherence between semantic and symbolic states"""
        if not self.is_complete:
            return 0.0

        # Simple coherence metric - can be enhanced with more sophisticated measures
        semantic_coherence = self.semantic_state.coherence_score
        symbolic_coherence = (
            len(self.symbolic_state.logical_predicates) / 10.0
        )  # Normalize

        return (semantic_coherence + min(symbolic_coherence, 1.0)) / 2.0

    @property
    def cognitive_energy(self) -> float:
        """Current cognitive energy of the geoid"""
        if self.thermodynamic is None:
            return 0.0
        return self.thermodynamic.free_energy

    def connect_input(self, connection_name: str, input_geoid: "GeoidState") -> None:
        """Connect an input geoid for processing pipeline flow"""
        self.input_connections[connection_name] = input_geoid
        input_geoid.output_connections[f"to_{self.geoid_id}"] = self

        # Update metadata
        if input_geoid.geoid_id not in self.metadata.parent_geoids:
            self.metadata.parent_geoids.append(input_geoid.geoid_id)
        if self.geoid_id not in input_geoid.metadata.child_geoids:
            input_geoid.metadata.child_geoids.append(self.geoid_id)

    def evolve_thermodynamically(self, time_step: float = 1.0) -> "GeoidState":
        """
        Evolve the geoid state according to thermodynamic principles.
        Returns a new evolved GeoidState while preserving the original.
        """
        if self.thermodynamic is None:
            return self  # No evolution without thermodynamic properties

        # Create evolved copy
        evolved = GeoidState(
            geoid_id=str(uuid.uuid4()),
            geoid_type=self.geoid_type
            processing_state=GeoidProcessingState.EVOLVING
            semantic_state=self.semantic_state,  # Will be updated
            symbolic_state=self.symbolic_state,  # Will be updated
            thermodynamic=self.thermodynamic,  # Will be updated
        )

        # Connect to parent
        evolved.connect_input("thermodynamic_parent", self)

        # Record evolution in metadata
        evolved.metadata.add_processing_step(
            engine_name="ThermodynamicEvolution",
            operation="evolve",
            duration=time_step
            metadata={"evolution_type": "thermodynamic", "time_step": time_step},
        )

        return evolved

    def to_dict(self) -> Dict[str, Any]:
        """Convert geoid to dictionary for serialization"""
        return {
            "geoid_id": self.geoid_id
            "geoid_type": self.geoid_type.value
            "processing_state": self.processing_state.value
            "semantic_state": (
                {
                    "embedding_vector": (
                        self.semantic_state.embedding_vector.tolist()
                        if self.semantic_state
                        else None
                    ),
                    "confidence_scores": (
                        self.semantic_state.confidence_scores
                        if self.semantic_state
                        else None
                    ),
                    "uncertainty_measure": (
                        self.semantic_state.uncertainty_measure
                        if self.semantic_state
                        else None
                    ),
                    "semantic_entropy": (
                        self.semantic_state.semantic_entropy
                        if self.semantic_state
                        else None
                    ),
                    "coherence_score": (
                        self.semantic_state.coherence_score
                        if self.semantic_state
                        else None
                    ),
                }
                if self.semantic_state
                else None
            ),
            "symbolic_state": (
                {
                    "logical_predicates": (
                        self.symbolic_state.logical_predicates
                        if self.symbolic_state
                        else None
                    ),
                    "symbolic_relations": (
                        self.symbolic_state.symbolic_relations
                        if self.symbolic_state
                        else None
                    ),
                    "rule_activations": (
                        self.symbolic_state.rule_activations
                        if self.symbolic_state
                        else None
                    ),
                    "symbolic_constraints": (
                        self.symbolic_state.symbolic_constraints
                        if self.symbolic_state
                        else None
                    ),
                    "proof_chains": (
                        self.symbolic_state.proof_chains
                        if self.symbolic_state
                        else None
                    ),
                }
                if self.symbolic_state
                else None
            ),
            "thermodynamic": (
                {
                    "cognitive_temperature": (
                        self.thermodynamic.cognitive_temperature
                        if self.thermodynamic
                        else None
                    ),
                    "information_entropy": (
                        self.thermodynamic.information_entropy
                        if self.thermodynamic
                        else None
                    ),
                    "free_energy": (
                        self.thermodynamic.free_energy if self.thermodynamic else None
                    ),
                    "activation_energy": (
                        self.thermodynamic.activation_energy
                        if self.thermodynamic
                        else None
                    ),
                    "dissipation_rate": (
                        self.thermodynamic.dissipation_rate
                        if self.thermodynamic
                        else None
                    ),
                    "equilibrium_tendency": (
                        self.thermodynamic.equilibrium_tendency
                        if self.thermodynamic
                        else None
                    ),
                }
                if self.thermodynamic
                else None
            ),
            "metadata": {
                "creation_timestamp": self.metadata.creation_timestamp.isoformat(),
                "last_modified": self.metadata.last_modified.isoformat(),
                "processing_history": self.metadata.processing_history
                "source_engine": self.metadata.source_engine
                "processing_depth": self.metadata.processing_depth
                "parent_geoids": self.metadata.parent_geoids
                "child_geoids": self.metadata.child_geoids
                "processing_flags": self.metadata.processing_flags
            },
            "coherence_score": self.coherence_score
            "cognitive_energy": self.cognitive_energy
            "is_complete": self.is_complete
        }

    def __repr__(self) -> str:
        """Human-readable representation"""
        status = f"GeoidState(id={self.geoid_id[:8]}..., type={self.geoid_type.value})"
        status += f", state={self.processing_state.value}, coherence={self.coherence_score:.3f})"
        return status


# Factory functions for common geoid creation patterns
def create_concept_geoid(concept_name: str, embedding: np.ndarray = None) -> GeoidState:
    """Create a geoid representing a concept"""
    semantic_state = SemanticState(
        embedding_vector=embedding if embedding is not None else np.random.random(768),
        confidence_scores={"concept_clarity": 0.8},
        uncertainty_measure=0.2
        semantic_entropy=1.5
        coherence_score=0.8
    )

    symbolic_state = SymbolicState(
        logical_predicates=[f"concept({concept_name})"],
        symbolic_relations={"name": concept_name},
        rule_activations={},
        symbolic_constraints=[],
        proof_chains=[],
    )

    return GeoidState(
        geoid_type=GeoidType.CONCEPT
        semantic_state=semantic_state
        symbolic_state=symbolic_state
    )


def create_relation_geoid(subject: str, predicate: str, object_: str) -> GeoidState:
    """Create a geoid representing a relation between entities"""
    symbolic_state = SymbolicState(
        logical_predicates=[f"{predicate}({subject}, {object_})"],
        symbolic_relations={
            "subject": subject
            "predicate": predicate
            "object": object_
        },
        rule_activations={},
        symbolic_constraints=[],
        proof_chains=[],
    )

    return GeoidState(geoid_type=GeoidType.RELATION, symbolic_state=symbolic_state)


def create_hypothesis_geoid(
    hypothesis_text: str, confidence: float = 0.5
) -> GeoidState:
    """Create a geoid representing a scientific hypothesis"""
    semantic_state = SemanticState(
        embedding_vector=np.random.random(768),  # Would use real embedder
        confidence_scores={"hypothesis_strength": confidence},
        uncertainty_measure=1.0 - confidence
        semantic_entropy=2.0
        coherence_score=confidence
    )

    symbolic_state = SymbolicState(
        logical_predicates=[f"hypothesis({hypothesis_text})"],
        symbolic_relations={"text": hypothesis_text, "confidence": confidence},
        rule_activations={"testable": True},
        symbolic_constraints=["requires_evidence"],
        proof_chains=[],
    )

    thermodynamic = ThermodynamicProperties(
        cognitive_temperature=1.0
        information_entropy=2.0
        free_energy=confidence * 10.0
        activation_energy=5.0
        dissipation_rate=0.1
        equilibrium_tendency=0.3
    )

    return GeoidState(
        geoid_type=GeoidType.HYPOTHESIS
        semantic_state=semantic_state
        symbolic_state=symbolic_state
        thermodynamic=thermodynamic
    )
