"""
KIMERA SWM - CORE GEOID PROCESSOR
=================================

The GeoidProcessor serves as the central processing hub for all geoid operations.
It orchestrates transformations, manages processing pipelines, and ensures
proper flow between different processing engines.

This is the critical component that bridges individual geoids with the
engine ecosystem, providing a clean, transparent, and auditable processing interface.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from ..data_structures.geoid_state import (
    GeoidProcessingState,
    GeoidState,
    GeoidType,
    SemanticState,
    SymbolicState,
    ThermodynamicProperties,
)

# Configure logging
logger = logging.getLogger(__name__)


class ProcessingMode(Enum):
    """Different modes of geoid processing"""

    SEQUENTIAL = "sequential"  # Process one geoid at a time
    PARALLEL = "parallel"  # Process multiple geoids simultaneously
    BATCH = "batch"  # Process in batches for efficiency
    STREAMING = "streaming"  # Continuous stream processing
    THERMODYNAMIC = "thermodynamic"  # Physics-based evolution processing


class ProcessingPriority(Enum):
    """Priority levels for geoid processing"""

    CRITICAL = "critical"  # Immediate processing required
    HIGH = "high"  # High priority processing
    NORMAL = "normal"  # Standard priority processing
    LOW = "low"  # Background processing
    DEFERRED = "deferred"  # Process when resources available


@dataclass
class ProcessingRequest:
    """Request for geoid processing"""

    geoid: GeoidState
    operation: str
    parameters: Dict[str, Any]
    priority: ProcessingPriority = ProcessingPriority.NORMAL
    callback: Optional[Callable] = None
    timeout: Optional[float] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ProcessingResult:
    """Result of geoid processing operation"""

    original_geoid: GeoidState
    processed_geoid: Optional[GeoidState]
    operation: str
    success: bool
    duration: float
    error_message: Optional[str] = None
    warnings: List[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.metadata is None:
            self.metadata = {}


class GeoidProcessor:
    """
    Core Geoid Processor - Central Processing Hub
    ============================================

    The GeoidProcessor orchestrates all geoid transformations and provides
    a unified interface for engine interactions. It ensures proper processing
    flow, maintains audit trails, and handles error conditions gracefully.

    Key Responsibilities:
    - Coordinate geoid transformations across engines
    - Maintain processing history and audit trails
    - Handle errors and recovery scenarios
    - Optimize processing performance
    - Ensure data integrity and consistency
    """

    def __init__(self, mode: ProcessingMode = ProcessingMode.SEQUENTIAL):
        self.mode = mode
        self.processing_queue: List[ProcessingRequest] = []
        self.completed_operations: List[ProcessingResult] = []
        self.active_operations: Dict[str, ProcessingRequest] = {}
        self.registered_operations: Dict[str, Callable] = {}
        self.performance_metrics: Dict[str, Any] = {
            "total_processed": 0,
            "average_duration": 0.0,
            "error_rate": 0.0,
            "operations_by_type": {},
        }

        # Register core operations
        self._register_core_operations()

        logger.info(f"GeoidProcessor initialized with mode: {mode.value}")

    def _register_core_operations(self) -> None:
        """Register core geoid processing operations"""
        self.registered_operations.update(
            {
                "semantic_enhancement": self._enhance_semantic_state,
                "symbolic_enrichment": self._enrich_symbolic_state,
                "thermodynamic_evolution": self._evolve_thermodynamically,
                "coherence_analysis": self._analyze_coherence,
                "state_validation": self._validate_state,
                "metadata_update": self._update_metadata,
                "connection_mapping": self._map_connections,
                "energy_calculation": self._calculate_energy,
            }
        )

    def register_operation(self, name: str, operation: Callable) -> None:
        """Register a custom processing operation"""
        if name in self.registered_operations:
            logger.warning(f"Overriding existing operation: {name}")

        self.registered_operations[name] = operation
        logger.info(f"Registered operation: {name}")

    def process_geoid(
        self,
        geoid: GeoidState,
        operation: str,
        parameters: Dict[str, Any] = None,
        priority: ProcessingPriority = ProcessingPriority.NORMAL,
    ) -> ProcessingResult:
        """
        Process a single geoid with the specified operation.
        Returns the processing result immediately (synchronous processing).
        """
        if parameters is None:
            parameters = {}

        start_time = time.time()

        # Create processing request
        request = ProcessingRequest(
            geoid=geoid,
            operation=operation,
            parameters=parameters,
            priority=priority,
            metadata={"request_time": datetime.now()},
        )

        # Validate operation
        if operation not in self.registered_operations:
            error_msg = f"Unknown operation: {operation}"
            logger.error(error_msg)
            return ProcessingResult(
                original_geoid=geoid,
                processed_geoid=None,
                operation=operation,
                success=False,
                duration=time.time() - start_time,
                error_message=error_msg,
            )

        # Execute operation
        try:
            operation_func = self.registered_operations[operation]
            processed_geoid = operation_func(geoid, parameters)

            # Record processing step in geoid metadata
            if processed_geoid and processed_geoid != geoid:
                processed_geoid.metadata.add_processing_step(
                    engine_name="GeoidProcessor",
                    operation=operation,
                    duration=time.time() - start_time,
                    metadata=parameters,
                )

            # Create successful result
            result = ProcessingResult(
                original_geoid=geoid,
                processed_geoid=processed_geoid,
                operation=operation,
                success=True,
                duration=time.time() - start_time,
                metadata={"parameters": parameters},
            )

            # Update metrics
            self._update_metrics(result)

            logger.debug(
                f"Successfully processed geoid {geoid.geoid_id[:8]} with operation {operation}"
            )
            return result

        except Exception as e:
            error_msg = f"Error processing geoid {geoid.geoid_id[:8]} with operation {operation}: {str(e)}"
            logger.error(error_msg, exc_info=True)

            result = ProcessingResult(
                original_geoid=geoid,
                processed_geoid=None,
                operation=operation,
                success=False,
                duration=time.time() - start_time,
                error_message=error_msg,
            )

            self._update_metrics(result)
            return result

    def process_batch(
        self,
        geoids: List[GeoidState],
        operation: str,
        parameters: Dict[str, Any] = None,
    ) -> List[ProcessingResult]:
        """Process a batch of geoids with the same operation"""
        if parameters is None:
            parameters = {}

        results = []
        logger.info(
            f"Processing batch of {len(geoids)} geoids with operation: {operation}"
        )

        for geoid in geoids:
            result = self.process_geoid(geoid, operation, parameters)
            results.append(result)

        # Log batch summary
        successful = sum(1 for r in results if r.success)
        logger.info(
            f"Batch processing complete: {successful}/{len(results)} successful"
        )

        return results

    def _enhance_semantic_state(
        self, geoid: GeoidState, parameters: Dict[str, Any]
    ) -> GeoidState:
        """Enhance the semantic state of a geoid"""
        if geoid.semantic_state is None:
            # Create new semantic state if none exists
            import numpy as np

            geoid.semantic_state = SemanticState(
                embedding_vector=np.random.random(768),  # Would use real embedder
                confidence_scores={"generated": 0.5},
                uncertainty_measure=0.5,
                semantic_entropy=1.0,
                coherence_score=0.5,
            )
        else:
            # Enhance existing semantic state
            enhancement_factor = parameters.get("enhancement_factor", 1.1)
            geoid.semantic_state.coherence_score = min(
                1.0, geoid.semantic_state.coherence_score * enhancement_factor
            )
            geoid.semantic_state.uncertainty_measure = max(
                0.0, geoid.semantic_state.uncertainty_measure * 0.9
            )

        geoid.processing_state = GeoidProcessingState.PROCESSING
        return geoid

    def _enrich_symbolic_state(
        self, geoid: GeoidState, parameters: Dict[str, Any]
    ) -> GeoidState:
        """Enrich the symbolic state of a geoid"""
        if geoid.symbolic_state is None:
            # Create new symbolic state if none exists
            geoid.symbolic_state = SymbolicState(
                logical_predicates=[],
                symbolic_relations={},
                rule_activations={},
                symbolic_constraints=[],
                proof_chains=[],
            )

        # Add new predicates from parameters
        new_predicates = parameters.get("predicates", [])
        for predicate in new_predicates:
            geoid.symbolic_state.add_predicate(predicate)

        # Activate new rules
        new_rules = parameters.get("rules", {})
        for rule_name, active in new_rules.items():
            if active:
                geoid.symbolic_state.activate_rule(rule_name)

        geoid.processing_state = GeoidProcessingState.PROCESSING
        return geoid

    def _evolve_thermodynamically(
        self, geoid: GeoidState, parameters: Dict[str, Any]
    ) -> GeoidState:
        """Evolve geoid according to thermodynamic principles"""
        time_step = parameters.get("time_step", 1.0)

        if geoid.thermodynamic is None:
            # Create thermodynamic properties if none exist
            geoid.thermodynamic = ThermodynamicProperties(
                cognitive_temperature=1.0,
                information_entropy=1.0,
                free_energy=5.0,
                activation_energy=2.0,
                dissipation_rate=0.1,
                equilibrium_tendency=0.5,
            )

        # Perform thermodynamic evolution
        evolved_geoid = geoid.evolve_thermodynamically(time_step)
        evolved_geoid.processing_state = GeoidProcessingState.EVOLVING

        return evolved_geoid

    def _analyze_coherence(
        self, geoid: GeoidState, parameters: Dict[str, Any]
    ) -> GeoidState:
        """Analyze and potentially improve geoid coherence"""
        threshold = parameters.get("coherence_threshold", 0.7)

        current_coherence = geoid.coherence_score

        if current_coherence < threshold:
            # Mark as needing attention
            geoid.metadata.processing_flags["low_coherence"] = True
            geoid.processing_state = GeoidProcessingState.PROCESSING
        else:
            # Mark as coherent
            geoid.metadata.processing_flags["high_coherence"] = True
            geoid.processing_state = GeoidProcessingState.STABLE

        return geoid

    def _validate_state(
        self, geoid: GeoidState, parameters: Dict[str, Any]
    ) -> GeoidState:
        """Validate the integrity of a geoid's state"""
        validation_errors = []

        # Check semantic state validity
        if geoid.semantic_state:
            if not 0.0 <= geoid.semantic_state.uncertainty_measure <= 1.0:
                validation_errors.append("Invalid uncertainty_measure range")
            if not 0.0 <= geoid.semantic_state.coherence_score <= 1.0:
                validation_errors.append("Invalid coherence_score range")

        # Check symbolic state validity
        if geoid.symbolic_state:
            if not isinstance(geoid.symbolic_state.logical_predicates, list):
                validation_errors.append("logical_predicates must be a list")

        # Update validation status
        if validation_errors:
            geoid.metadata.processing_flags["validation_errors"] = validation_errors
            logger.warning(
                f"Validation errors for geoid {geoid.geoid_id[:8]}: {validation_errors}"
            )
        else:
            geoid.metadata.processing_flags["validated"] = True

        return geoid

    def _update_metadata(
        self, geoid: GeoidState, parameters: Dict[str, Any]
    ) -> GeoidState:
        """Update geoid metadata with new information"""
        updates = parameters.get("metadata_updates", {})

        for key, value in updates.items():
            geoid.metadata.processing_flags[key] = value

        geoid.metadata.last_modified = datetime.now()
        return geoid

    def _map_connections(
        self, geoid: GeoidState, parameters: Dict[str, Any]
    ) -> GeoidState:
        """Map connections between geoids"""
        input_connections = parameters.get("input_connections", {})

        for connection_name, input_geoid in input_connections.items():
            if isinstance(input_geoid, GeoidState):
                geoid.connect_input(connection_name, input_geoid)

        return geoid

    def _calculate_energy(
        self, geoid: GeoidState, parameters: Dict[str, Any]
    ) -> GeoidState:
        """Calculate and update cognitive energy"""
        if geoid.thermodynamic is None:
            return geoid

        # Simple energy calculation based on coherence and activity
        base_energy = parameters.get("base_energy", 5.0)
        coherence_bonus = geoid.coherence_score * 2.0
        activity_penalty = geoid.metadata.processing_depth * 0.1

        new_energy = max(0.0, base_energy + coherence_bonus - activity_penalty)
        geoid.thermodynamic.free_energy = new_energy

        return geoid

    def _update_metrics(self, result: ProcessingResult) -> None:
        """Update performance metrics based on processing result"""
        self.performance_metrics["total_processed"] += 1

        # Update average duration
        total_ops = self.performance_metrics["total_processed"]
        avg_duration = self.performance_metrics["average_duration"]
        self.performance_metrics["average_duration"] = (
            avg_duration * (total_ops - 1) + result.duration
        ) / total_ops

        # Update error rate
        if not result.success:
            errors = sum(1 for r in self.completed_operations if not r.success) + 1
            self.performance_metrics["error_rate"] = errors / total_ops

        # Update operations by type
        op_counts = self.performance_metrics["operations_by_type"]
        op_counts[result.operation] = op_counts.get(result.operation, 0) + 1

        # Store completed operation
        self.completed_operations.append(result)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of processing performance"""
        return {
            "total_processed": self.performance_metrics["total_processed"],
            "average_duration_ms": self.performance_metrics["average_duration"] * 1000,
            "error_rate_percent": self.performance_metrics["error_rate"] * 100,
            "operations_by_type": self.performance_metrics["operations_by_type"],
            "mode": self.mode.value,
            "registered_operations": list(self.registered_operations.keys()),
            "queue_size": len(self.processing_queue),
            "active_operations": len(self.active_operations),
        }

    def clear_history(self) -> None:
        """Clear processing history (useful for testing or memory management)"""
        self.completed_operations.clear()
        self.performance_metrics = {
            "total_processed": 0,
            "average_duration": 0.0,
            "error_rate": 0.0,
            "operations_by_type": {},
        }
        logger.info("Processing history cleared")


# Convenience functions for common processing patterns
def enhance_geoid_semantics(
    geoid: GeoidState, enhancement_factor: float = 1.1
) -> ProcessingResult:
    """Convenience function to enhance geoid semantic state"""
    processor = GeoidProcessor()
    return processor.process_geoid(
        geoid, "semantic_enhancement", {"enhancement_factor": enhancement_factor}
    )


def validate_geoid(geoid: GeoidState) -> ProcessingResult:
    """Convenience function to validate a geoid"""
    processor = GeoidProcessor()
    return processor.process_geoid(geoid, "state_validation")


def evolve_geoid(geoid: GeoidState, time_step: float = 1.0) -> ProcessingResult:
    """Convenience function to evolve a geoid thermodynamically"""
    processor = GeoidProcessor()
    return processor.process_geoid(
        geoid, "thermodynamic_evolution", {"time_step": time_step}
    )
