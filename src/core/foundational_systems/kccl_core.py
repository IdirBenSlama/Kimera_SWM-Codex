"""
KCCL Core - Kimera Cognitive Cycle Logic Core
==============================================

The heartbeat of Kimera's cognitive processing. This module provides the
core cognitive cycle logic integrated into the foundational architecture.

KCCL orchestrates the fundamental cognitive processing loop:
- Semantic Pressure Diffusion coordination
- Contradiction detection and resolution
- Vault management integration
- Scar record creation and management
- System health monitoring

This is the master coordinator that ensures all cognitive systems
work together in unified cognitive cycles.
"""

import asyncio
import logging
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from ...config.settings import get_settings
from ...utils.config import get_api_settings
from ...utils.kimera_logger import get_cognitive_logger
from ..embedding_utils import encode_text
from ..scar import ScarRecord

logger = get_cognitive_logger(__name__)


class CognitiveCycleState(Enum):
    """States of the cognitive cycle execution"""

    IDLE = "idle"
    INITIALIZING = "initializing"
    SPDE_PROCESSING = "spde_processing"
    CONTRADICTION_DETECTION = "contradiction_detection"
    SCAR_GENERATION = "scar_generation"
    VAULT_INTEGRATION = "vault_integration"
    COMPLETING = "completing"
    ERROR = "error"


class CyclePhase(Enum):
    """Phases of cognitive cycle execution"""

    PREPARATION = "preparation"
    SEMANTIC_DIFFUSION = "semantic_diffusion"
    CONTRADICTION_ANALYSIS = "contradiction_analysis"
    SCAR_PROCESSING = "scar_processing"
    INTEGRATION = "integration"
    FINALIZATION = "finalization"


@dataclass
class CognitiveCycleMetrics:
    """Auto-generated class."""
    pass
    """Comprehensive metrics for cognitive cycle execution"""

    cycle_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_duration: float = 0.0

    # Processing metrics
    geoids_processed: int = 0
    contradictions_detected: int = 0
    scars_created: int = 0
    errors_encountered: int = 0

    # Entropy metrics
    entropy_before_diffusion: float = 0.0
    entropy_after_diffusion: float = 0.0
    entropy_delta: float = 0.0

    # Performance metrics
    phase_durations: Dict[str, float] = field(default_factory=dict)
    memory_usage: Dict[str, float] = field(default_factory=dict)
    processing_rate: float = 0.0

    # Quality metrics
    coherence_score: float = 0.0
    integration_success: bool = False
    system_health: float = 0.0


@dataclass
class CognitiveCycleResult:
    """Auto-generated class."""
    pass
    """Result of cognitive cycle execution"""

    success: bool
    metrics: CognitiveCycleMetrics
    generated_scars: List[ScarRecord]
    processed_geoids: List[Any]
    system_updates: Dict[str, Any]
    error_log: List[str] = field(default_factory=list)
class KCCLCore:
    """Auto-generated class."""
    pass
    """
    Kimera Cognitive Cycle Logic Core

    The foundational cognitive cycle orchestrator that coordinates
    all cognitive processing systems in unified, coherent cycles.
    """

    def __init__(
        self
        max_geoids_per_cycle: int = 500
        max_tensions_per_cycle: int = 20
        safety_mode: bool = True
    ):
        """
        Initialize KCCL Core

        Args:
            max_geoids_per_cycle: Maximum geoids to process in one cycle
            max_tensions_per_cycle: Maximum tensions to process per cycle
            safety_mode: Enable safety limits and error handling
        """
        self.settings = get_api_settings()

        # Configuration
        self.max_geoids_per_cycle = max_geoids_per_cycle
        self.max_tensions_per_cycle = max_tensions_per_cycle
        self.safety_mode = safety_mode

        # State management
        self.current_state = CognitiveCycleState.IDLE
        self.cycle_count = 0
        self.total_cycles_executed = 0
        self.total_processing_time = 0.0

        # System components (will be injected by orchestrator)
        self.spde_engine = None
        self.contradiction_engine = None
        self.vault_manager = None
        self.transparency_monitor = None

        # Performance tracking
        self.cycle_history = deque(maxlen=1000)
        self.performance_metrics = {
            "average_cycle_time": 0.0
            "cycles_per_second": 0.0
            "success_rate": 0.0
            "error_rate": 0.0
            "geoids_processed_total": 0
            "scars_generated_total": 0
        }

        # Event callbacks
        self.cycle_callbacks = {
            "on_cycle_start": [],
            "on_cycle_complete": [],
            "on_cycle_error": [],
            "on_phase_start": [],
            "on_phase_complete": [],
        }

        logger.info("🔄 KCCL Core initialized")
        logger.info(f"   Max geoids per cycle: {max_geoids_per_cycle}")
        logger.info(f"   Max tensions per cycle: {max_tensions_per_cycle}")
        logger.info(f"   Safety mode: {safety_mode}")

    def register_components(
        self
        spde_engine: Any
        contradiction_engine: Any
        vault_manager: Any
        transparency_monitor: Any = None
    ):
        """Register required cognitive processing components"""
        self.spde_engine = spde_engine
        self.contradiction_engine = contradiction_engine
        self.vault_manager = vault_manager
        self.transparency_monitor = transparency_monitor

        logger.info("✅ KCCL Core components registered")

    def register_callback(self, event_type: str, callback: Callable):
        """Register callback for cognitive cycle events"""
        if event_type in self.cycle_callbacks:
            self.cycle_callbacks[event_type].append(callback)
            logger.debug(f"Registered callback for {event_type}")

    async def execute_cognitive_cycle(
        self, system: Dict[str, Any], cycle_context: Optional[Dict[str, Any]] = None
    ) -> CognitiveCycleResult:
        """
        Execute a complete cognitive cycle

        Args:
            system: System state with active_geoids and other components
            cycle_context: Optional context for cycle execution

        Returns:
            Complete cognitive cycle result with metrics and outputs
        """
        cycle_start_time = time.time()
        cycle_id = f"KCCL_{self.cycle_count:06d}_{int(cycle_start_time)}"
        self.cycle_count += 1

        # Initialize metrics
        metrics = CognitiveCycleMetrics(
            cycle_id=cycle_id, start_time=datetime.now(timezone.utc)
        )

        # Initialize result
        result = CognitiveCycleResult(
            success=False
            metrics=metrics
            generated_scars=[],
            processed_geoids=[],
            system_updates={},
        )

        try:
            self.current_state = CognitiveCycleState.INITIALIZING

            # Trigger cycle start callbacks
            await self._trigger_callbacks(
                "on_cycle_start",
                {"cycle_id": cycle_id, "system": system, "context": cycle_context},
            )

            # Phase 1: Preparation
            await self._execute_preparation_phase(system, metrics, result)

            # Phase 2: Semantic Diffusion
            await self._execute_semantic_diffusion_phase(system, metrics, result)

            # Phase 3: Contradiction Analysis
            await self._execute_contradiction_analysis_phase(system, metrics, result)

            # Phase 4: Scar Processing
            await self._execute_scar_processing_phase(system, metrics, result)

            # Phase 5: Integration
            await self._execute_integration_phase(system, metrics, result)

            # Phase 6: Finalization
            await self._execute_finalization_phase(system, metrics, result)

            # Mark as successful
            result.success = True
            self.current_state = CognitiveCycleState.IDLE

            # Update performance metrics
            self._update_performance_metrics(metrics)

            # Trigger completion callbacks
            await self._trigger_callbacks(
                "on_cycle_complete", {"cycle_id": cycle_id, "result": result}
            )

            logger.debug(f"✅ Cognitive cycle {cycle_id} completed successfully")

        except Exception as e:
            self.current_state = CognitiveCycleState.ERROR
            error_msg = f"Cognitive cycle {cycle_id} failed: {e}"
            logger.error(error_msg)
            result.error_log.append(error_msg)
            metrics.errors_encountered += 1

            # Trigger error callbacks
            await self._trigger_callbacks(
                "on_cycle_error", {"cycle_id": cycle_id, "error": e, "result": result}
            )

        finally:
            # Finalize metrics
            cycle_end_time = time.time()
            metrics.end_time = datetime.now(timezone.utc)
            metrics.total_duration = cycle_end_time - cycle_start_time

            # Store in history
            self.cycle_history.append(result)
            self.total_cycles_executed += 1
            self.total_processing_time += metrics.total_duration

        return result

    async def _execute_preparation_phase(
        self
        system: Dict[str, Any],
        metrics: CognitiveCycleMetrics
        result: CognitiveCycleResult
    ):
        """Execute preparation phase of cognitive cycle"""
        phase_start = time.time()

        await self._trigger_callbacks(
            "on_phase_start",
            {"phase": CyclePhase.PREPARATION, "cycle_id": metrics.cycle_id},
        )

        try:
            # Safety check: limit processing if too many geoids
            active_geoids = system.get("active_geoids", {})
            geoids_to_process = list(active_geoids.values())

            if self.safety_mode and len(geoids_to_process) > self.max_geoids_per_cycle:
                logger.warning(
                    f"Large geoid count ({len(geoids_to_process)}), "
                    f"limiting to {self.max_geoids_per_cycle} for cycle"
                )
                geoids_to_process = geoids_to_process[: self.max_geoids_per_cycle]

            metrics.geoids_processed = len(geoids_to_process)
            result.processed_geoids = geoids_to_process

            logger.debug(
                f"Preparation complete: {len(geoids_to_process)} geoids ready for processing"
            )

        except Exception as e:
            logger.error(f"Preparation phase failed: {e}")
            result.error_log.append(f"Preparation failed: {e}")
            raise

        finally:
            phase_duration = time.time() - phase_start
            metrics.phase_durations[CyclePhase.PREPARATION.value] = phase_duration

            await self._trigger_callbacks(
                "on_phase_complete",
                {
                    "phase": CyclePhase.PREPARATION
                    "duration": phase_duration
                    "cycle_id": metrics.cycle_id
                },
            )

    async def _execute_semantic_diffusion_phase(
        self
        system: Dict[str, Any],
        metrics: CognitiveCycleMetrics
        result: CognitiveCycleResult
    ):
        """Execute semantic pressure diffusion phase"""
        phase_start = time.time()
        self.current_state = CognitiveCycleState.SPDE_PROCESSING

        await self._trigger_callbacks(
            "on_phase_start",
            {"phase": CyclePhase.SEMANTIC_DIFFUSION, "cycle_id": metrics.cycle_id},
        )

        try:
            if not self.spde_engine:
                raise ValueError("SPDE engine not registered")

            geoids_to_process = result.processed_geoids

            # Calculate entropy before diffusion
            entropy_before = sum(
                g.calculate_entropy()
                for g in geoids_to_process
                if hasattr(g, "calculate_entropy")
            )
            metrics.entropy_before_diffusion = entropy_before

            # Apply semantic pressure diffusion
            diffusion_errors = 0
            for geoid in geoids_to_process:
                try:
                    if hasattr(geoid, "semantic_state") and geoid.semantic_state:
                        # Apply SPDE diffusion
                        if hasattr(self.spde_engine, "diffuse"):
                            geoid.semantic_state = self.spde_engine.diffuse(
                                geoid.semantic_state
                            )
                        elif hasattr(self.spde_engine, "process_semantic_diffusion"):
                            geoid.semantic_state = (
                                await self.spde_engine.process_semantic_diffusion(
                                    geoid.semantic_state
                                )
                            )
                except Exception as e:
                    diffusion_errors += 1
                    logger.warning(
                        f"SPDE diffusion failed for geoid {getattr(geoid, 'geoid_id', 'unknown')}: {e}"
                    )
                    continue

            # Calculate entropy after diffusion
            entropy_after = sum(
                g.calculate_entropy()
                for g in geoids_to_process
                if hasattr(g, "calculate_entropy")
            )
            metrics.entropy_after_diffusion = entropy_after
            metrics.entropy_delta = entropy_after - entropy_before
            metrics.errors_encountered += diffusion_errors

            logger.debug(
                f"Semantic diffusion complete: entropy delta = {metrics.entropy_delta:.4f}"
            )

        except Exception as e:
            logger.error(f"Semantic diffusion phase failed: {e}")
            result.error_log.append(f"Semantic diffusion failed: {e}")
            raise

        finally:
            phase_duration = time.time() - phase_start
            metrics.phase_durations[CyclePhase.SEMANTIC_DIFFUSION.value] = (
                phase_duration
            )

            await self._trigger_callbacks(
                "on_phase_complete",
                {
                    "phase": CyclePhase.SEMANTIC_DIFFUSION
                    "duration": phase_duration
                    "cycle_id": metrics.cycle_id
                },
            )

    async def _execute_contradiction_analysis_phase(
        self
        system: Dict[str, Any],
        metrics: CognitiveCycleMetrics
        result: CognitiveCycleResult
    ):
        """Execute contradiction detection and analysis phase"""
        phase_start = time.time()
        self.current_state = CognitiveCycleState.CONTRADICTION_DETECTION

        await self._trigger_callbacks(
            "on_phase_start",
            {"phase": CyclePhase.CONTRADICTION_ANALYSIS, "cycle_id": metrics.cycle_id},
        )

        try:
            if not self.contradiction_engine:
                raise ValueError("Contradiction engine not registered")

            geoids_to_process = result.processed_geoids

            # Detect tension gradients
            tensions = await self._detect_tensions(geoids_to_process)
            metrics.contradictions_detected = len(tensions)

            # Limit tension processing to prevent overload
            if self.safety_mode:
                tensions_to_process = tensions[: self.max_tensions_per_cycle]
                if len(tensions) > self.max_tensions_per_cycle:
                    logger.warning(
                        f"Limited tension processing to {self.max_tensions_per_cycle} "
                        f"out of {len(tensions)} detected"
                    )
            else:
                tensions_to_process = tensions

            # Store tensions for scar processing
            result.system_updates["detected_tensions"] = tensions_to_process

            logger.debug(
                f"Contradiction analysis complete: {len(tensions_to_process)} tensions to process"
            )

        except Exception as e:
            logger.error(f"Contradiction analysis phase failed: {e}")
            result.error_log.append(f"Contradiction analysis failed: {e}")
            raise

        finally:
            phase_duration = time.time() - phase_start
            metrics.phase_durations[CyclePhase.CONTRADICTION_ANALYSIS.value] = (
                phase_duration
            )

            await self._trigger_callbacks(
                "on_phase_complete",
                {
                    "phase": CyclePhase.CONTRADICTION_ANALYSIS
                    "duration": phase_duration
                    "cycle_id": metrics.cycle_id
                },
            )

    async def _execute_scar_processing_phase(
        self
        system: Dict[str, Any],
        metrics: CognitiveCycleMetrics
        result: CognitiveCycleResult
    ):
        """Execute scar generation and processing phase"""
        phase_start = time.time()
        self.current_state = CognitiveCycleState.SCAR_GENERATION

        await self._trigger_callbacks(
            "on_phase_start",
            {"phase": CyclePhase.SCAR_PROCESSING, "cycle_id": metrics.cycle_id},
        )

        try:
            tensions_to_process = result.system_updates.get("detected_tensions", [])
            generated_scars = []

            for tension in tensions_to_process:
                try:
                    # Generate scar record
                    scar = await self._generate_scar_from_tension(
                        tension, metrics.cycle_id
                    )
                    generated_scars.append(scar)

                    # Store scar in vault if available
                    if self.vault_manager and hasattr(self.vault_manager, "store_scar"):
                        await self.vault_manager.store_scar(scar)

                except Exception as e:
                    logger.warning(f"Failed to generate scar from tension: {e}")
                    metrics.errors_encountered += 1
                    continue

            result.generated_scars = generated_scars
            metrics.scars_created = len(generated_scars)

            logger.debug(
                f"Scar processing complete: {len(generated_scars)} scars generated"
            )

        except Exception as e:
            logger.error(f"Scar processing phase failed: {e}")
            result.error_log.append(f"Scar processing failed: {e}")
            raise

        finally:
            phase_duration = time.time() - phase_start
            metrics.phase_durations[CyclePhase.SCAR_PROCESSING.value] = phase_duration

            await self._trigger_callbacks(
                "on_phase_complete",
                {
                    "phase": CyclePhase.SCAR_PROCESSING
                    "duration": phase_duration
                    "cycle_id": metrics.cycle_id
                },
            )

    async def _execute_integration_phase(
        self
        system: Dict[str, Any],
        metrics: CognitiveCycleMetrics
        result: CognitiveCycleResult
    ):
        """Execute integration and coherence phase"""
        phase_start = time.time()
        self.current_state = CognitiveCycleState.VAULT_INTEGRATION

        await self._trigger_callbacks(
            "on_phase_start",
            {"phase": CyclePhase.INTEGRATION, "cycle_id": metrics.cycle_id},
        )

        try:
            # Calculate system coherence
            coherence_score = await self._calculate_system_coherence(
                result.processed_geoids, result.generated_scars
            )
            metrics.coherence_score = coherence_score

            # Integration success if coherence is above threshold
            metrics.integration_success = coherence_score > 0.7

            # Update system state
            result.system_updates.update(
                {
                    "cycle_coherence": coherence_score
                    "integration_success": metrics.integration_success
                    "processed_count": metrics.geoids_processed
                    "scars_count": metrics.scars_created
                }
            )

            logger.debug(f"Integration complete: coherence = {coherence_score:.4f}")

        except Exception as e:
            logger.error(f"Integration phase failed: {e}")
            result.error_log.append(f"Integration failed: {e}")
            raise

        finally:
            phase_duration = time.time() - phase_start
            metrics.phase_durations[CyclePhase.INTEGRATION.value] = phase_duration

            await self._trigger_callbacks(
                "on_phase_complete",
                {
                    "phase": CyclePhase.INTEGRATION
                    "duration": phase_duration
                    "cycle_id": metrics.cycle_id
                },
            )

    async def _execute_finalization_phase(
        self
        system: Dict[str, Any],
        metrics: CognitiveCycleMetrics
        result: CognitiveCycleResult
    ):
        """Execute finalization and cleanup phase"""
        phase_start = time.time()
        self.current_state = CognitiveCycleState.COMPLETING

        await self._trigger_callbacks(
            "on_phase_start",
            {"phase": CyclePhase.FINALIZATION, "cycle_id": metrics.cycle_id},
        )

        try:
            # Calculate system health
            system_health = await self._calculate_system_health(metrics, result)
            metrics.system_health = system_health

            # Calculate processing rate
            if metrics.total_duration > 0:
                metrics.processing_rate = (
                    metrics.geoids_processed / metrics.total_duration
                )

            # Log cycle summary
            logger.info(f"Cycle {metrics.cycle_id} summary:")
            logger.info(f"  Geoids processed: {metrics.geoids_processed}")
            logger.info(f"  Contradictions detected: {metrics.contradictions_detected}")
            logger.info(f"  Scars created: {metrics.scars_created}")
            logger.info(f"  Entropy delta: {metrics.entropy_delta:.4f}")
            logger.info(f"  Coherence score: {metrics.coherence_score:.4f}")
            rate_display = (
                f"{metrics.processing_rate:.2f}"
                if metrics.processing_rate < float("inf")
                else "∞"
            )
            logger.info(f"  Processing rate: {rate_display} geoids/sec")
            logger.info(f"  System health: {metrics.system_health:.4f}")

        except Exception as e:
            logger.error(f"Finalization phase failed: {e}")
            result.error_log.append(f"Finalization failed: {e}")
            raise

        finally:
            phase_duration = time.time() - phase_start
            metrics.phase_durations[CyclePhase.FINALIZATION.value] = phase_duration

            await self._trigger_callbacks(
                "on_phase_complete",
                {
                    "phase": CyclePhase.FINALIZATION
                    "duration": phase_duration
                    "cycle_id": metrics.cycle_id
                },
            )

    async def _detect_tensions(self, geoids: List[Any]) -> List[Any]:
        """Detect tension gradients between geoids"""
        if hasattr(self.contradiction_engine, "detect_tension_gradients"):
            return self.contradiction_engine.detect_tension_gradients(geoids)
        elif hasattr(self.contradiction_engine, "analyze_contradictions"):
            return await self.contradiction_engine.analyze_contradictions(geoids)
        else:
            logger.warning("Contradiction engine has no known tension detection method")
            return []

    async def _generate_scar_from_tension(
        self, tension: Any, cycle_id: str
    ) -> ScarRecord:
        """Generate a scar record from a detected tension"""
        geoid_a = getattr(tension, "geoid_a", "unknown")
        geoid_b = getattr(tension, "geoid_b", "unknown")
        tension_score = getattr(tension, "tension_score", 0.0)

        summary = f"Tension {geoid_a}-{geoid_b}"
        vector = encode_text(summary)

        scar = ScarRecord(
            scar_id=f"SCAR_{uuid.uuid4().hex[:8]}",
            geoids=[geoid_a, geoid_b],
            reason=f"auto-cycle-{cycle_id}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            resolved_by="KCCLCore",
            pre_entropy=0.0
            post_entropy=0.0
            delta_entropy=0.0
            cls_angle=tension_score * 180
            semantic_polarity=0.5,  # Add required semantic_polarity
            mutation_frequency=0.1,  # Add required mutation_frequency
        )

        return scar

    async def _calculate_system_coherence(
        self, geoids: List[Any], scars: List[ScarRecord]
    ) -> float:
        """Calculate overall system coherence"""
        try:
            # Basic coherence calculation based on entropy and scar generation
            if not geoids:
                return 0.0

            # Factor 1: Entropy stability (lower is better)
            entropy_factor = (
                max(0.0, 1.0 - abs(self.cycle_history[-1].metrics.entropy_delta) / 10.0)
                if self.cycle_history
                else 0.5
            )

            # Factor 2: Scar generation rate (moderate is better)
            scar_rate = len(scars) / len(geoids) if geoids else 0.0
            scar_factor = max(
                0.0, 1.0 - abs(scar_rate - 0.1) * 5.0
            )  # Optimal around 10% scar rate

            # Factor 3: Error rate (lower is better)
            error_rate = (
                self.cycle_history[-1].metrics.errors_encountered / len(geoids)
                if self.cycle_history and geoids
                else 0.0
            )
            error_factor = max(0.0, 1.0 - error_rate)

            # Weighted average
            coherence = entropy_factor * 0.4 + scar_factor * 0.3 + error_factor * 0.3

            return min(1.0, max(0.0, coherence))

        except Exception as e:
            logger.warning(f"Failed to calculate system coherence: {e}")
            return 0.0

    async def _calculate_system_health(
        self, metrics: CognitiveCycleMetrics, result: CognitiveCycleResult
    ) -> float:
        """Calculate overall system health"""
        try:
            health_factors = []

            # Processing success rate
            if metrics.geoids_processed > 0:
                success_rate = (
                    metrics.geoids_processed - metrics.errors_encountered
                ) / metrics.geoids_processed
                health_factors.append(success_rate)

            # Integration success
            health_factors.append(1.0 if metrics.integration_success else 0.5)

            # Coherence score
            health_factors.append(metrics.coherence_score)

            # Performance factor (based on processing rate)
            if metrics.processing_rate > 0:
                performance_factor = min(
                    1.0, metrics.processing_rate / 100.0
                )  # Normalize to expected rate
                health_factors.append(performance_factor)

            return sum(health_factors) / len(health_factors) if health_factors else 0.0

        except Exception as e:
            logger.warning(f"Failed to calculate system health: {e}")
            return 0.0

    async def _trigger_callbacks(self, event_type: str, event_data: Dict[str, Any]):
        """Trigger registered callbacks for cognitive cycle events"""
        if event_type in self.cycle_callbacks:
            for callback in self.cycle_callbacks[event_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event_data)
                    else:
                        callback(event_data)
                except Exception as e:
                    logger.warning(
                        f"Callback {callback.__name__} failed for {event_type}: {e}"
                    )

    def _update_performance_metrics(self, metrics: CognitiveCycleMetrics):
        """Update running performance metrics"""
        # Update averages
        self.performance_metrics["average_cycle_time"] = (
            self.performance_metrics["average_cycle_time"]
            * (self.total_cycles_executed - 1)
            + metrics.total_duration
        ) / self.total_cycles_executed

        if metrics.total_duration > 0:
            self.performance_metrics["cycles_per_second"] = 1.0 / metrics.total_duration

        # Update success/error rates
        total_geoids = (
            self.performance_metrics["geoids_processed_total"]
            + metrics.geoids_processed
        )
        if total_geoids > 0:
            success_rate = (total_geoids - metrics.errors_encountered) / total_geoids
            self.performance_metrics["success_rate"] = success_rate
            self.performance_metrics["error_rate"] = 1.0 - success_rate

        # Update totals
        self.performance_metrics["geoids_processed_total"] += metrics.geoids_processed
        self.performance_metrics["scars_generated_total"] += metrics.scars_created

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and metrics"""
        return {
            "current_state": self.current_state.value
            "cycle_count": self.cycle_count
            "total_cycles_executed": self.total_cycles_executed
            "total_processing_time": self.total_processing_time
            "performance_metrics": self.performance_metrics.copy(),
            "components_registered": {
                "spde_engine": self.spde_engine is not None
                "contradiction_engine": self.contradiction_engine is not None
                "vault_manager": self.vault_manager is not None
                "transparency_monitor": self.transparency_monitor is not None
            },
            "configuration": {
                "max_geoids_per_cycle": self.max_geoids_per_cycle
                "max_tensions_per_cycle": self.max_tensions_per_cycle
                "safety_mode": self.safety_mode
            },
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        health_status = {
            "overall_health": "unknown",
            "component_health": {},
            "performance_health": {},
            "system_issues": [],
        }

        try:
            # Check component health
            health_status["component_health"] = {
                "spde_engine": "healthy" if self.spde_engine else "missing",
                "contradiction_engine": (
                    "healthy" if self.contradiction_engine else "missing"
                ),
                "vault_manager": "healthy" if self.vault_manager else "missing",
            }

            # Check performance health
            if self.performance_metrics["error_rate"] > 0.1:
                health_status["system_issues"].append("High error rate detected")

            if self.performance_metrics["average_cycle_time"] > 5.0:
                health_status["system_issues"].append("Slow cycle performance")

            # Overall health assessment
            if not health_status["system_issues"] and all(
                status in ["healthy", "missing"]
                for status in health_status["component_health"].values()
            ):
                health_status["overall_health"] = "healthy"
            elif health_status["system_issues"]:
                health_status["overall_health"] = "degraded"
            else:
                health_status["overall_health"] = "warning"

        except Exception as e:
            health_status["overall_health"] = "error"
            health_status["system_issues"].append(f"Health check failed: {e}")

        return health_status
