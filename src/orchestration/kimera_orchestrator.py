"""
KIMERA SWM - SYSTEM ORCHESTRATOR
================================

The Kimera Orchestrator serves as the central coordination hub for the entire
Kimera SWM cognitive system. It manages engine interactions, coordinates
processing pipelines, and provides a unified interface for complex cognitive
operations across all domains.

This is the system's "conductor" that orchestrates the symphony of cognitive engines.
"""

import asyncio
import logging
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from src.core.data_structures.geoid_state import (
    GeoidProcessingState,
    GeoidState,
    GeoidType,
)
from src.core.processing.geoid_processor import (
    GeoidProcessor,
    ProcessingMode,
    ProcessingPriority,
    ProcessingResult,
)
from src.core.utilities.geoid_registry import GeoidRegistry, get_global_registry
from src.engines.field_dynamics.cognitive_field_engine import (
    CognitiveFieldEngine,
    FieldParameters,
    FieldResult,
    FieldType,
)
from src.engines.thermodynamic.thermodynamic_evolution_engine import (
    EvolutionParameters,
    EvolutionResult,
    ThermodynamicEvolutionEngine,
)
from src.engines.transformation.mirror_portal_engine import (
    MirrorPortalEngine,
    QuantumParameters,
    TransitionResult,
    TransitionType,
)

# GPU System Integration
try:
    from src.core.gpu.gpu_integration import (
        GPUWorkloadType,
        get_gpu_integration_system,
        submit_gpu_task,
    )
    from src.core.gpu.gpu_manager import get_gpu_manager, is_gpu_available
    from src.engines.gpu.gpu_geoid_processor import get_gpu_geoid_processor
    from src.engines.gpu.gpu_thermodynamic_engine import get_gpu_thermodynamic_engine

    GPU_ORCHESTRATION_AVAILABLE = True
except ImportError as e:
    GPU_ORCHESTRATION_AVAILABLE = False


# Configure logging
logger = logging.getLogger(__name__)

# Log GPU orchestration status after logger is defined
if GPU_ORCHESTRATION_AVAILABLE:
    logger.info("GPU orchestration imports successful")
else:
    logger.warning("GPU orchestration not available")


class OrchestrationMode(Enum):
    """Modes of orchestration"""

    SEQUENTIAL = "sequential"  # Process engines sequentially
    PARALLEL = "parallel"  # Process engines in parallel
    ADAPTIVE = "adaptive"  # Adaptive scheduling based on load
    PIPELINE = "pipeline"  # Pipeline processing
    REACTIVE = "reactive"  # Reactive to system state


class ProcessingStrategy(Enum):
    """High-level processing strategies"""

    EXPLORATION = "exploration"  # Explore knowledge space
    CONSOLIDATION = "consolidation"  # Consolidate existing knowledge
    TRANSFORMATION = "transformation"  # Transform representations
    OPTIMIZATION = "optimization"  # Optimize for efficiency
    EMERGENCE = "emergence"  # Enable emergent behaviors
    SCIENTIFIC = "scientific"  # Scientific hypothesis testing


@dataclass
class OrchestrationParameters:
    """Parameters controlling orchestration behavior"""

    mode: OrchestrationMode = OrchestrationMode.ADAPTIVE
    strategy: ProcessingStrategy = ProcessingStrategy.EXPLORATION
    max_parallel_engines: int = 4
    engine_timeout: float = 30.0
    retry_attempts: int = 3
    enable_caching: bool = True
    energy_conservation: bool = True
    coherence_threshold: float = 0.7
    emergence_detection: bool = True
    adaptive_scheduling: bool = True


@dataclass
class ProcessingPipeline:
    """Definition of a processing pipeline"""

    pipeline_id: str
    name: str
    description: str
    stages: List[Dict[str, Any]]  # List of processing stages
    prerequisites: List[str]  # Required capabilities
    expected_duration: float  # Expected processing time
    priority: ProcessingPriority = ProcessingPriority.NORMAL


@dataclass
class OrchestrationResult:
    """Result of orchestrated processing"""

    session_id: str
    original_geoids: List[GeoidState]
    processed_geoids: List[GeoidState]
    pipeline_used: str
    engines_executed: List[str]
    processing_duration: float
    energy_consumed: float
    emergent_phenomena: List[Dict[str, Any]]
    errors: List[str]
    warnings: List[str]
    performance_metrics: Dict[str, Any]
    metadata: Dict[str, Any]


class EngineCoordinator:
    """
    Coordinates individual engines and manages their interactions.
    """

    def __init__(self):
        # Initialize standard engines
        self.geoid_processor = GeoidProcessor(ProcessingMode.PARALLEL)
        self.thermodynamic_engine = ThermodynamicEvolutionEngine()
        self.mirror_portal_engine = MirrorPortalEngine()
        self.field_engine = CognitiveFieldEngine()

        # Initialize GPU-accelerated engines
        self.gpu_available = False
        self.gpu_manager = None
        self.gpu_integration_system = None
        self.gpu_geoid_processor = None
        self.gpu_thermodynamic_engine = None

        if GPU_ORCHESTRATION_AVAILABLE:
            self._initialize_gpu_engines()

        # Engine registry (includes both standard and GPU engines)
        self.engines = {
            "geoid_processor": self.geoid_processor,
            "thermodynamic_engine": self.thermodynamic_engine,
            "mirror_portal_engine": self.mirror_portal_engine,
            "field_engine": self.field_engine,
        }

        # Add GPU engines to registry if available
        if self.gpu_available:
            self.engines.update(
                {
                    "gpu_geoid_processor": self.gpu_geoid_processor,
                    "gpu_thermodynamic_engine": self.gpu_thermodynamic_engine,
                    "gpu_integration_system": self.gpu_integration_system,
                }
            )

        # Engine capabilities
        self.engine_capabilities = {
            "geoid_processor": [
                "semantic_enhancement",
                "symbolic_enrichment",
                "state_validation",
                "coherence_analysis",
                "metadata_update",
                "energy_calculation",
            ],
            "thermodynamic_engine": [
                "thermodynamic_evolution",
                "energy_conservation",
                "entropy_analysis",
                "system_equilibration",
                "phase_transitions",
            ],
            "mirror_portal_engine": [
                "quantum_transitions",
                "semantic_symbolic_bridge",
                "superposition_creation",
                "entanglement",
                "mirror_reflection",
                "coherence_enhancement",
            ],
            "field_engine": [
                "field_processing",
                "collective_dynamics",
                "emergence_detection",
                "spatial_reasoning",
                "interaction_modeling",
            ],
        }

        # Add GPU engine capabilities if available
        if self.gpu_available:
            self.engine_capabilities.update(
                {
                    "gpu_geoid_processor": [
                        "gpu_semantic_enhancement",
                        "gpu_parallel_processing",
                        "gpu_batch_operations",
                        "gpu_high_throughput",
                        "gpu_memory_optimization",
                        "gpu_acceleration",
                    ],
                    "gpu_thermodynamic_engine": [
                        "gpu_thermodynamic_evolution",
                        "gpu_quantum_field_dynamics",
                        "gpu_ensemble_processing",
                        "gpu_parallel_evolution",
                        "gpu_memory_efficient",
                        "gpu_high_performance",
                    ],
                    "gpu_integration_system": [
                        "gpu_task_orchestration",
                        "gpu_performance_monitoring",
                        "gpu_load_balancing",
                        "gpu_adaptive_scheduling",
                        "gpu_resource_management",
                    ],
                }
            )

        # Performance tracking
        self.engine_performance = {
            engine_name: {
                "total_operations": 0,
                "average_duration": 0.0,
                "success_rate": 1.0,
                "last_used": None,
            }
            for engine_name in self.engines.keys()
        }

        logger.info("EngineCoordinator initialized with all engines")

    def _initialize_gpu_engines(self) -> None:
        """Initialize GPU-accelerated engines if available."""
        try:
            if is_gpu_available():
                # Initialize GPU Manager
                self.gpu_manager = get_gpu_manager()

                # Initialize GPU Integration System
                self.gpu_integration_system = get_gpu_integration_system()

                # Initialize GPU Geoid Processor
                self.gpu_geoid_processor = get_gpu_geoid_processor()

                # Initialize GPU Thermodynamic Engine
                self.gpu_thermodynamic_engine = get_gpu_thermodynamic_engine()

                self.gpu_available = True
                logger.info("🚀 GPU engines initialized successfully")

                # Log GPU capabilities
                gpu_info = self.gpu_manager.get_device_info()
                logger.info(f"⚡ GPU Device: {gpu_info.get('name', 'Unknown')}")
                logger.info(
                    f"🔥 GPU Memory: {gpu_info.get('total_memory_gb', 0):.1f}GB"
                )

            else:
                logger.warning("⚠️ GPU hardware not available - using CPU engines only")
                self.gpu_available = False

        except Exception as exc:
            logger.error(f"❌ Failed to initialize GPU engines: {exc}")
            self.gpu_available = False
            # Reset GPU components to None
            self.gpu_manager = None
            self.gpu_integration_system = None
            self.gpu_geoid_processor = None
            self.gpu_thermodynamic_engine = None

    def get_optimal_engine(
        self, operation: str, geoid_count: int = 1, prefer_gpu: bool = True
    ) -> str:
        """
        Determine the optimal engine for a given operation.

        Args:
            operation: The operation to perform
            geoid_count: Number of geoids to process
            prefer_gpu: Whether to prefer GPU engines if available

        Returns:
            Name of the optimal engine to use
        """
        # GPU engine selection logic
        if self.gpu_available and prefer_gpu:
            # For large batches, prefer GPU engines
            if geoid_count >= 5:  # Threshold for GPU efficiency
                if operation in [
                    "semantic_enhancement",
                    "parallel_processing",
                    "batch_operations",
                ]:
                    return "gpu_geoid_processor"
                elif operation in [
                    "thermodynamic_evolution",
                    "quantum_field_dynamics",
                    "ensemble_processing",
                ]:
                    return "gpu_thermodynamic_engine"

            # For GPU-specific operations
            if operation.startswith("gpu_"):
                if "geoid" in operation or "semantic" in operation:
                    return "gpu_geoid_processor"
                elif "thermodynamic" in operation or "evolution" in operation:
                    return "gpu_thermodynamic_engine"
                elif "integration" in operation or "orchestration" in operation:
                    return "gpu_integration_system"

        # Fallback to standard engines
        if operation in self.engine_capabilities.get("geoid_processor", []):
            return "geoid_processor"
        elif operation in self.engine_capabilities.get("thermodynamic_engine", []):
            return "thermodynamic_engine"
        elif operation in self.engine_capabilities.get("mirror_portal_engine", []):
            return "mirror_portal_engine"
        elif operation in self.engine_capabilities.get("field_engine", []):
            return "field_engine"

        # Default to geoid processor
        return "geoid_processor"

    def execute_operation(
        self,
        engine_name: str,
        operation: str,
        geoids: Union[GeoidState, List[GeoidState]],
        parameters: Dict[str, Any] = None,
    ) -> Any:
        """Execute an operation on a specific engine"""
        if engine_name not in self.engines:
            raise ValueError(f"Unknown engine: {engine_name}")

        if parameters is None:
            parameters = {}

        start_time = time.time()
        engine = self.engines[engine_name]

        try:
            # Execute based on engine type
            if engine_name == "geoid_processor":
                if isinstance(geoids, list):
                    result = engine.process_batch(geoids, operation, parameters)
                else:
                    result = engine.process_geoid(geoids, operation, parameters)

            elif engine_name == "thermodynamic_engine":
                if isinstance(geoids, list):
                    result = engine.evolve_system(
                        geoids, EvolutionParameters(**parameters)
                    )
                else:
                    result = engine.evolve(geoids, EvolutionParameters(**parameters))

            elif engine_name == "mirror_portal_engine":
                transition_type = TransitionType(
                    parameters.get("transition_type", "coherence_bridge")
                )
                quantum_params = QuantumParameters(
                    **{k: v for k, v in parameters.items() if k != "transition_type"}
                )
                if isinstance(geoids, list):
                    result = engine.transform_batch(
                        geoids, transition_type, quantum_params
                    )
                else:
                    result = engine.transform(geoids, transition_type, quantum_params)

            elif engine_name == "field_engine":
                field_types = parameters.get("field_types", [FieldType.SEMANTIC_FIELD])
                if not isinstance(geoids, list):
                    geoids = [geoids]
                result = engine.process_geoids_in_fields(geoids, field_types)

            # GPU Engine Operations
            elif engine_name == "gpu_geoid_processor":
                if isinstance(geoids, list):
                    # Async operation for batch processing
                    import asyncio

                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        result = loop.run_until_complete(
                            engine.process_geoid_batch(geoids, operation, **parameters)
                        )
                    finally:
                        loop.close()
                else:
                    result = engine.process_single_geoid(
                        geoids, operation, **parameters
                    )

            elif engine_name == "gpu_thermodynamic_engine":
                from src.engines.gpu.gpu_thermodynamic_engine import (
                    EvolutionParameters as GPUEvolutionParameters,
                )
                from src.engines.gpu.gpu_thermodynamic_engine import (
                    ThermodynamicEnsemble,
                    ThermodynamicRegime,
                )

                if isinstance(geoids, list):
                    # Create thermodynamic ensemble
                    ensemble = ThermodynamicEnsemble(
                        ensemble_id=f"orchestrator_ensemble_{uuid.uuid4().hex[:8]}",
                        geoids=geoids,
                        temperature=parameters.get("temperature", 1.0),
                        pressure=parameters.get("pressure", 1.0),
                        chemical_potential=parameters.get("chemical_potential", 0.0),
                        regime=ThermodynamicRegime(
                            parameters.get("regime", "equilibrium")
                        ),
                    )

                    gpu_params = GPUEvolutionParameters(
                        time_step=parameters.get("time_step", 0.01),
                        max_iterations=parameters.get("max_iterations", 100),
                        temperature_schedule=parameters.get(
                            "temperature_schedule", "linear"
                        ),
                        quantum_corrections=parameters.get("quantum_corrections", True),
                    )

                    # Async operation for ensemble evolution
                    import asyncio

                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        result = loop.run_until_complete(
                            engine.evolve_ensemble(ensemble, gpu_params)
                        )
                    finally:
                        loop.close()
                else:
                    # Single geoid evolution (create single-element ensemble)
                    ensemble = ThermodynamicEnsemble(
                        ensemble_id=f"single_geoid_{uuid.uuid4().hex[:8]}",
                        geoids=[geoids],
                        temperature=parameters.get("temperature", 1.0),
                        pressure=parameters.get("pressure", 1.0),
                        chemical_potential=parameters.get("chemical_potential", 0.0),
                        regime=ThermodynamicRegime(
                            parameters.get("regime", "equilibrium")
                        ),
                    )

                    gpu_params = GPUEvolutionParameters(
                        time_step=parameters.get("time_step", 0.01),
                        max_iterations=parameters.get("max_iterations", 100),
                        temperature_schedule=parameters.get(
                            "temperature_schedule", "linear"
                        ),
                        quantum_corrections=parameters.get("quantum_corrections", True),
                    )

                    import asyncio

                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        evolved_geoids, evolution_data = loop.run_until_complete(
                            engine.evolve_ensemble(ensemble, gpu_params)
                        )
                        result = (
                            evolved_geoids[0],
                            evolution_data,
                        )  # Return first geoid
                    finally:
                        loop.close()

            elif engine_name == "gpu_integration_system":
                # Direct GPU integration system operations
                if operation == "submit_task":
                    workload_type = GPUWorkloadType(
                        parameters.get("workload_type", "geoid_processing")
                    )
                    priority = parameters.get("priority", 5)

                    import asyncio

                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        task_id = loop.run_until_complete(
                            engine.submit_task(
                                workload_type, {"geoids": geoids}, priority
                            )
                        )
                        result = {"task_id": task_id, "status": "submitted"}
                    finally:
                        loop.close()

                elif operation == "get_performance":
                    result = engine.get_performance_summary()

                elif operation == "optimize_performance":
                    import asyncio

                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        result = loop.run_until_complete(engine.optimize_performance())
                    finally:
                        loop.close()
                else:
                    result = {
                        "error": f"Unknown GPU integration operation: {operation}"
                    }

            else:
                raise ValueError(
                    f"Unknown operation {operation} for engine {engine_name}"
                )

            # Update performance metrics
            duration = time.time() - start_time
            self._update_engine_performance(engine_name, duration, True)

            logger.debug(
                f"Engine {engine_name} executed {operation} in {duration:.3f}s"
            )
            return result

        except Exception as e:
            duration = time.time() - start_time
            self._update_engine_performance(engine_name, duration, False)
            logger.error(f"Engine {engine_name} failed operation {operation}: {str(e)}")
            raise

    def get_engine_status(self) -> Dict[str, Any]:
        """Get status of all engines"""
        status = {}

        for engine_name, engine in self.engines.items():
            try:
                if hasattr(engine, "get_engine_statistics"):
                    engine_stats = engine.get_engine_statistics()
                elif hasattr(engine, "get_performance_summary"):
                    engine_stats = engine.get_performance_summary()
                else:
                    engine_stats = {"status": "available"}

                status[engine_name] = {
                    "stats": engine_stats,
                    "performance": self.engine_performance[engine_name],
                    "capabilities": self.engine_capabilities[engine_name],
                }
            except Exception as e:
                status[engine_name] = {"error": str(e), "status": "error"}

        return status

    def _update_engine_performance(
        self, engine_name: str, duration: float, success: bool
    ) -> None:
        """Update performance metrics for an engine"""
        perf = self.engine_performance[engine_name]

        perf["total_operations"] += 1
        perf["last_used"] = datetime.now()

        # Update average duration
        total_ops = perf["total_operations"]
        avg_duration = perf["average_duration"]
        perf["average_duration"] = (
            avg_duration * (total_ops - 1) + duration
        ) / total_ops

        # Update success rate
        if total_ops == 1:
            perf["success_rate"] = 1.0 if success else 0.0
        else:
            current_successes = perf["success_rate"] * (total_ops - 1)
            if success:
                current_successes += 1
            perf["success_rate"] = current_successes / total_ops


class KimeraOrchestrator:
    """
    Kimera System Orchestrator - Central Cognitive Coordination Hub
    ==============================================================

    The KimeraOrchestrator is the master conductor of the Kimera SWM cognitive
    system. It coordinates complex multi-engine processing pipelines, manages
    system resources, and enables sophisticated cognitive behaviors through
    intelligent orchestration of the engine ecosystem.

    Key Responsibilities:
    - Multi-engine pipeline coordination
    - Adaptive processing strategy selection
    - Resource management and optimization
    - Emergent behavior detection and management
    - System health monitoring and recovery
    - Scientific experiment orchestration
    """

    def __init__(self, parameters: OrchestrationParameters = None):
        self.parameters = parameters or OrchestrationParameters()
        self.coordinator = EngineCoordinator()
        self.registry = get_global_registry()

        # Processing pipelines
        self.pipelines: Dict[str, ProcessingPipeline] = {}
        self.active_sessions: Dict[str, Dict[str, Any]] = {}

        # System state
        self.total_orchestrations = 0
        self.orchestration_history: List[OrchestrationResult] = []
        self.system_health = 1.0
        self.last_health_check = datetime.now()

        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(
            max_workers=self.parameters.max_parallel_engines
        )

        # Initialize standard pipelines
        self._initialize_standard_pipelines()

        logger.info(
            f"KimeraOrchestrator initialized with mode: {self.parameters.mode.value}"
        )

    def _initialize_standard_pipelines(self) -> None:
        """Initialize standard processing pipelines"""

        # Scientific Exploration Pipeline
        self.pipelines["scientific_exploration"] = ProcessingPipeline(
            pipeline_id="scientific_exploration",
            name="Scientific Exploration",
            description="Comprehensive scientific exploration of knowledge space",
            stages=[
                {"engine": "geoid_processor", "operation": "state_validation"},
                {
                    "engine": "thermodynamic_engine",
                    "operation": "thermodynamic_evolution",
                },
                {"engine": "mirror_portal_engine", "operation": "coherence_bridge"},
                {"engine": "field_engine", "operation": "field_processing"},
                {"engine": "geoid_processor", "operation": "coherence_analysis"},
            ],
            prerequisites=["geoid_validation"],
            expected_duration=10.0,
        )

        # Rapid Transformation Pipeline
        self.pipelines["rapid_transformation"] = ProcessingPipeline(
            pipeline_id="rapid_transformation",
            name="Rapid Transformation",
            description="Quick transformation for real-time processing",
            stages=[
                {"engine": "mirror_portal_engine", "operation": "semantic_to_symbolic"},
                {"engine": "geoid_processor", "operation": "coherence_analysis"},
            ],
            prerequisites=["basic_geoid"],
            expected_duration=2.0,
        )

        # Deep Analysis Pipeline
        self.pipelines["deep_analysis"] = ProcessingPipeline(
            pipeline_id="deep_analysis",
            name="Deep Cognitive Analysis",
            description="Comprehensive analysis using all engines",
            stages=[
                {"engine": "geoid_processor", "operation": "semantic_enhancement"},
                {"engine": "geoid_processor", "operation": "symbolic_enrichment"},
                {
                    "engine": "thermodynamic_engine",
                    "operation": "thermodynamic_evolution",
                },
                {
                    "engine": "mirror_portal_engine",
                    "operation": "quantum_superposition",
                },
                {"engine": "field_engine", "operation": "collective_dynamics"},
                {"engine": "geoid_processor", "operation": "energy_calculation"},
            ],
            prerequisites=["complete_geoid"],
            expected_duration=15.0,
        )

        # Emergence Detection Pipeline
        self.pipelines["emergence_detection"] = ProcessingPipeline(
            pipeline_id="emergence_detection",
            name="Emergence Detection",
            description="Detect and analyze emergent cognitive phenomena",
            stages=[
                {"engine": "field_engine", "operation": "field_processing"},
                {"engine": "thermodynamic_engine", "operation": "system_evolution"},
                {"engine": "mirror_portal_engine", "operation": "entanglement"},
                {"engine": "geoid_processor", "operation": "coherence_analysis"},
            ],
            prerequisites=["multiple_geoids"],
            expected_duration=12.0,
        )

    def orchestrate(
        self,
        geoids: Union[GeoidState, List[GeoidState]],
        pipeline: str = None,
        strategy: ProcessingStrategy = None,
    ) -> OrchestrationResult:
        """
        Orchestrate processing of geoids through the cognitive system.
        This is the main entry point for complex cognitive operations.
        """
        session_id = str(uuid.uuid4())
        start_time = time.time()

        # Normalize inputs
        if not isinstance(geoids, list):
            geoids = [geoids]

        strategy = strategy or self.parameters.strategy

        # Register geoids
        for geoid in geoids:
            self.registry.register(geoid)

        # Select pipeline
        if pipeline is None:
            pipeline = self._select_optimal_pipeline(geoids, strategy)

        logger.info(
            f"Starting orchestration session {session_id[:8]} with pipeline '{pipeline}'"
        )

        # Initialize session
        self.active_sessions[session_id] = {
            "start_time": start_time,
            "geoids": geoids,
            "pipeline": pipeline,
            "strategy": strategy,
            "status": "active",
        }

        try:
            # Execute pipeline
            result = self._execute_pipeline(session_id, geoids, pipeline)

            # Update session
            self.active_sessions[session_id]["status"] = "completed"
            self.active_sessions[session_id]["result"] = result

            # Update registry with processed geoids
            for geoid in result.processed_geoids:
                self.registry.update(geoid)

            # Update system state
            self._update_orchestration_state(result)

            logger.info(
                f"Orchestration session {session_id[:8]} completed in {result.processing_duration:.3f}s"
            )
            return result

        except Exception as e:
            logger.error(f"Orchestration session {session_id[:8]} failed: {str(e)}")
            self.active_sessions[session_id]["status"] = "failed"
            self.active_sessions[session_id]["error"] = str(e)

            # Create error result
            return OrchestrationResult(
                session_id=session_id,
                original_geoids=geoids,
                processed_geoids=geoids,  # Return originals on failure
                pipeline_used=pipeline,
                engines_executed=[],
                processing_duration=time.time() - start_time,
                energy_consumed=0.0,
                emergent_phenomena=[],
                errors=[str(e)],
                warnings=[],
                performance_metrics={},
                metadata={"status": "failed"},
            )

        finally:
            # Cleanup session after delay
            if session_id in self.active_sessions:
                # Keep for analysis, but mark as finished
                self.active_sessions[session_id]["end_time"] = time.time()

    def _select_optimal_pipeline(
        self, geoids: List[GeoidState], strategy: ProcessingStrategy
    ) -> str:
        """Select the optimal pipeline based on geoids and strategy"""

        # Analyze geoid characteristics
        has_semantic = any(g.semantic_state is not None for g in geoids)
        has_symbolic = any(g.symbolic_state is not None for g in geoids)
        has_thermodynamic = any(g.thermodynamic is not None for g in geoids)
        avg_coherence = sum(g.coherence_score for g in geoids) / len(geoids)

        # Strategy-based selection
        if strategy == ProcessingStrategy.EXPLORATION:
            return "scientific_exploration"
        elif strategy == ProcessingStrategy.TRANSFORMATION:
            return "rapid_transformation"
        elif strategy == ProcessingStrategy.EMERGENCE:
            return "emergence_detection"
        elif strategy == ProcessingStrategy.SCIENTIFIC:
            return "deep_analysis"

        # Characteristic-based fallback
        if len(geoids) > 5 and self.parameters.emergence_detection:
            return "emergence_detection"
        elif avg_coherence < 0.5:
            return "deep_analysis"
        else:
            return "scientific_exploration"

    def _execute_pipeline(
        self, session_id: str, geoids: List[GeoidState], pipeline_name: str
    ) -> OrchestrationResult:
        """Execute a specific processing pipeline"""
        if pipeline_name not in self.pipelines:
            raise ValueError(f"Unknown pipeline: {pipeline_name}")

        pipeline = self.pipelines[pipeline_name]
        processed_geoids = geoids.copy()
        engines_executed = []
        errors = []
        warnings = []
        energy_consumed = 0.0
        emergent_phenomena = []

        start_time = time.time()

        # Execute each stage
        for stage_idx, stage in enumerate(pipeline.stages):
            engine_name = stage["engine"]
            operation = stage.get("operation", "default")
            stage_params = stage.get("parameters", {})

            try:
                logger.debug(
                    f"Executing stage {stage_idx + 1}: {engine_name}.{operation}"
                )

                # Execute stage
                stage_result = self.coordinator.execute_operation(
                    engine_name, operation, processed_geoids, stage_params
                )

                engines_executed.append(f"{engine_name}.{operation}")

                # Extract processed geoids from result
                if hasattr(stage_result, "processed_geoids"):
                    processed_geoids = stage_result.processed_geoids
                elif hasattr(stage_result, "evolved_geoid"):
                    # Single geoid result
                    processed_geoids = [stage_result.evolved_geoid]
                elif hasattr(stage_result, "transformed_geoid"):
                    # Transformation result
                    processed_geoids = [stage_result.transformed_geoid]
                elif isinstance(stage_result, list):
                    # List of results
                    new_geoids = []
                    for result in stage_result:
                        if (
                            hasattr(result, "processed_geoid")
                            and result.processed_geoid
                        ):
                            new_geoids.append(result.processed_geoid)
                        elif hasattr(result, "evolved_geoid"):
                            new_geoids.append(result.evolved_geoid)
                        elif hasattr(result, "transformed_geoid"):
                            new_geoids.append(result.transformed_geoid)
                    if new_geoids:
                        processed_geoids = new_geoids

                # Track energy consumption
                if hasattr(stage_result, "energy_consumed"):
                    energy_consumed += stage_result.energy_consumed
                elif hasattr(stage_result, "energy_change"):
                    energy_consumed += abs(stage_result.energy_change)

                # Track emergent phenomena
                if hasattr(stage_result, "emergent_behaviors"):
                    emergent_phenomena.extend(stage_result.emergent_behaviors)
                elif hasattr(stage_result, "quantum_effects"):
                    emergent_phenomena.append(
                        {
                            "type": "quantum_effects",
                            "stage": stage_idx,
                            "effects": stage_result.quantum_effects,
                        }
                    )

            except Exception as e:
                error_msg = f"Stage {stage_idx + 1} ({engine_name}.{operation}) failed: {str(e)}"
                errors.append(error_msg)
                logger.warning(error_msg)

                # Continue with remaining stages if possible
                continue

        # Calculate performance metrics
        total_duration = time.time() - start_time
        performance_metrics = {
            "stages_completed": len(engines_executed),
            "total_stages": len(pipeline.stages),
            "success_rate": len(engines_executed) / len(pipeline.stages),
            "average_stage_duration": total_duration / max(1, len(engines_executed)),
            "energy_efficiency": len(processed_geoids) / max(0.1, energy_consumed),
        }

        return OrchestrationResult(
            session_id=session_id,
            original_geoids=geoids,
            processed_geoids=processed_geoids,
            pipeline_used=pipeline_name,
            engines_executed=engines_executed,
            processing_duration=total_duration,
            energy_consumed=energy_consumed,
            emergent_phenomena=emergent_phenomena,
            errors=errors,
            warnings=warnings,
            performance_metrics=performance_metrics,
            metadata={
                "pipeline_description": pipeline.description,
                "strategy_used": self.parameters.strategy.value,
                "orchestration_mode": self.parameters.mode.value,
            },
        )

    def _update_orchestration_state(self, result: OrchestrationResult) -> None:
        """Update orchestrator state after processing"""
        self.total_orchestrations += 1
        self.orchestration_history.append(result)

        # Update system health based on results
        success_rate = result.performance_metrics.get("success_rate", 0.0)
        self.system_health = (self.system_health * 0.9) + (success_rate * 0.1)

        # Keep history manageable
        if len(self.orchestration_history) > 100:
            self.orchestration_history = self.orchestration_history[-50:]

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        engine_status = self.coordinator.get_engine_status()
        registry_metrics = self.registry.get_registry_metrics()

        # Calculate recent performance
        recent_results = self.orchestration_history[-20:]
        recent_performance = {
            "average_duration": 0.0,
            "average_success_rate": 0.0,
            "average_energy_efficiency": 0.0,
        }

        if recent_results:
            recent_performance["average_duration"] = sum(
                r.processing_duration for r in recent_results
            ) / len(recent_results)

            recent_performance["average_success_rate"] = sum(
                r.performance_metrics.get("success_rate", 0.0) for r in recent_results
            ) / len(recent_results)

            recent_performance["average_energy_efficiency"] = sum(
                r.performance_metrics.get("energy_efficiency", 0.0)
                for r in recent_results
            ) / len(recent_results)

        return {
            "system_health": self.system_health,
            "total_orchestrations": self.total_orchestrations,
            "active_sessions": len(
                [
                    s
                    for s in self.active_sessions.values()
                    if s.get("status") == "active"
                ]
            ),
            "available_pipelines": list(self.pipelines.keys()),
            "engine_status": engine_status,
            "registry_status": registry_metrics,
            "recent_performance": recent_performance,
            "parameters": self.parameters.__dict__,
        }


# Convenience functions for common orchestration patterns
def explore_knowledge_space(geoids: List[GeoidState]) -> OrchestrationResult:
    """Convenience function for knowledge space exploration"""
    orchestrator = KimeraOrchestrator()
    return orchestrator.orchestrate(geoids, "scientific_exploration")


def rapid_transform_geoids(geoids: List[GeoidState]) -> OrchestrationResult:
    """Convenience function for rapid transformation"""
    orchestrator = KimeraOrchestrator()
    return orchestrator.orchestrate(geoids, "rapid_transformation")


def deep_analyze_geoids(geoids: List[GeoidState]) -> OrchestrationResult:
    """Convenience function for deep analysis"""
    orchestrator = KimeraOrchestrator()
    return orchestrator.orchestrate(geoids, "deep_analysis")


def detect_emergence(geoids: List[GeoidState]) -> OrchestrationResult:
    """Convenience function for emergence detection"""
    orchestrator = KimeraOrchestrator()
    return orchestrator.orchestrate(geoids, "emergence_detection")
