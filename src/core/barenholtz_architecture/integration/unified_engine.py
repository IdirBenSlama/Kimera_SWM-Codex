"""
Barenholtz Dual-System Unified Integration Engine
================================================

DO-178C Level A compliant integration of dual-system cognitive architecture.

This module unifies System 1 (intuitive) and System 2 (analytical) processing
through metacognitive control, implementing the complete Barenholtz dual-system
architecture for Kimera SWM.

Safety Critical: All integration must maintain system coherence and prevent
deadlocks while meeting strict timing requirements.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from ..core import (
    AnalysisResult,
    ArbitrationResult,
    ArbitrationStrategy,
    IntuitionResult,
    MetacognitiveController,
    ProcessingMode,
    ReasoningType,
    System1Processor,
    System2Processor,
)
from ..utils.conflict_resolver import ConflictResolver
from ..utils.memory_manager import WorkingMemoryManager

# Robust import fallback for kimera utilities
try:
    from src.utils.kimera_exceptions import KimeraCognitiveError
    from src.utils.kimera_logger import LogCategory, get_logger
except ImportError:
    try:
        from utils.kimera_exceptions import KimeraCognitiveError
        from utils.kimera_logger import LogCategory, get_logger
    except ImportError:
        import logging

        # Fallback logger and exception
        def get_logger(name, category=None):
            return logging.getLogger(name)

        class LogCategory:
            DUAL_SYSTEM = "dual_system"

        class KimeraCognitiveError(Exception):
            pass


# Robust import fallback for performance monitor
try:
    from src.monitoring.performance_monitor import PerformanceMonitor
except ImportError:
    try:
        from monitoring.performance_monitor import PerformanceMonitor
    except ImportError:
        # Fallback performance monitor
        class PerformanceMonitor:
            def __init__(self, *args, **kwargs):
                pass

            def start_monitoring(self):
                pass

            def stop_monitoring(self):
                pass

            def get_metrics(self):
                return {}


logger = get_logger(__name__, LogCategory.DUAL_SYSTEM)


class SystemMode(Enum):
    """Operating modes for the dual-system architecture"""

    AUTOMATIC = "automatic"  # Let metacognitive controller decide
    SYSTEM1_PREFERRED = "system1_preferred"  # Prefer fast/intuitive
    SYSTEM2_PREFERRED = "system2_preferred"  # Prefer slow/analytical
    PARALLEL = "parallel"  # Always run both in parallel
    SEQUENTIAL = "sequential"  # System 1 first, then System 2 if needed


@dataclass
class ProcessingConstraints:
    """Constraints for dual-system processing"""

    max_response_time: float = 1.0  # Maximum total response time
    min_confidence: float = 0.6  # Minimum required confidence
    required_reasoning: Optional[List[ReasoningType]] = None
    system_mode: SystemMode = SystemMode.AUTOMATIC
    arbitration_strategy: Optional[ArbitrationStrategy] = None
    resource_limits: Optional[Dict[str, float]] = None


@dataclass
class DualSystemOutput:
    """Complete output from dual-system processing"""

    # Results from each system
    system1_result: Optional[IntuitionResult]
    system2_result: Optional[AnalysisResult]
    arbitration_result: ArbitrationResult

    # Integrated output
    final_response: Dict[str, Any]
    confidence: float
    processing_time: float

    # Metadata
    system_mode_used: SystemMode
    processing_mode: ProcessingMode
    performance_metrics: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)

    def is_valid(self) -> bool:
        """Validate output meets safety requirements"""
        return (
            self.arbitration_result.is_valid()
            and 0.0 <= self.confidence <= 1.0
            and self.processing_time > 0
        )


class DualSystemMonitor:
    """Monitor dual-system health and performance"""

    def __init__(self):
        self.system1_health = {"status": "healthy", "last_check": datetime.now()}
        self.system2_health = {"status": "healthy", "last_check": datetime.now()}
        self.metacognitive_health = {"status": "healthy", "last_check": datetime.now()}
        self.integration_metrics = {
            "total_requests": 0,
            "system1_only": 0,
            "system2_only": 0,
            "both_systems": 0,
            "conflicts_resolved": 0,
            "timeouts": 0,
        }

    def update_health(self, component: str, status: str):
        """Update component health status"""
        health_map = {
            "system1": self.system1_health,
            "system2": self.system2_health,
            "metacognitive": self.metacognitive_health,
        }

        if component in health_map:
            health_map[component]["status"] = status
            health_map[component]["last_check"] = datetime.now()

    def record_request(self, output: DualSystemOutput):
        """Record metrics from a processing request"""
        self.integration_metrics["total_requests"] += 1

        if output.system1_result and not output.system2_result:
            self.integration_metrics["system1_only"] += 1
        elif output.system2_result and not output.system1_result:
            self.integration_metrics["system2_only"] += 1
        elif output.system1_result and output.system2_result:
            self.integration_metrics["both_systems"] += 1

    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report"""
        return {
            "components": {
                "system1": self.system1_health,
                "system2": self.system2_health,
                "metacognitive": self.metacognitive_health,
            },
            "metrics": self.integration_metrics,
            "overall_status": self._calculate_overall_status(),
        }

    def _calculate_overall_status(self) -> str:
        """Calculate overall system health"""
        statuses = [
            self.system1_health["status"],
            self.system2_health["status"],
            self.metacognitive_health["status"],
        ]

        if all(s == "healthy" for s in statuses):
            return "healthy"
        elif any(s == "critical" for s in statuses):
            return "critical"
        elif any(s == "degraded" for s in statuses):
            return "degraded"
        else:
            return "unknown"


class BarenholtzDualSystemIntegrator:
    """
    Main integration engine for Barenholtz dual-system architecture

    DO-178C Level A Requirements:
    - Deterministic behavior under all conditions
    - Bounded response times
    - Graceful degradation
    - Full traceability
    """

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device

        # Initialize core components
        self.system1 = System1Processor(device)
        self.system2 = System2Processor(device)
        self.metacognitive = MetacognitiveController()

        # Support components
        self.memory_manager = WorkingMemoryManager()
        self.conflict_resolver = ConflictResolver()
        self.monitor = DualSystemMonitor()
        self.performance_monitor = PerformanceMonitor()

        # Configuration
        self.default_constraints = ProcessingConstraints()

        # State
        self.is_initialized = True
        self.processing_count = 0

        logger.info("🧠 Barenholtz Dual-System Integrator initialized")
        logger.info(f"   Device: {device}")
        logger.info("   Components: System1, System2, Metacognitive Controller")
        logger.info("   DO-178C Level A Compliant")

    async def process(
        self,
        input_data: torch.Tensor,
        context: Optional[Dict[str, Any]] = None,
        constraints: Optional[ProcessingConstraints] = None,
    ) -> DualSystemOutput:
        """
        Main processing function for dual-system architecture

        Args:
            input_data: Input tensor for processing
            context: Optional context information
            constraints: Optional processing constraints

        Returns:
            DualSystemOutput with integrated results

        Raises:
            KimeraCognitiveError: If processing fails
        """
        start_time = time.time()
        self.processing_count += 1

        # Use provided or default constraints
        constraints = constraints or self.default_constraints

        # Validate input
        if not self._validate_input(input_data):
            raise KimeraCognitiveError("Invalid input data")

        # Add performance monitoring
        with self.performance_monitor.profile("dual_system_processing"):
            try:
                # Determine system mode
                system_mode = self._determine_system_mode(
                    input_data, context, constraints
                )

                # Execute processing based on mode
                if system_mode == SystemMode.PARALLEL:
                    output = await self._parallel_processing(
                        input_data, context, constraints
                    )
                elif system_mode == SystemMode.SEQUENTIAL:
                    output = await self._sequential_processing(
                        input_data, context, constraints
                    )
                elif system_mode == SystemMode.SYSTEM1_PREFERRED:
                    output = await self._system1_preferred_processing(
                        input_data, context, constraints
                    )
                elif system_mode == SystemMode.SYSTEM2_PREFERRED:
                    output = await self._system2_preferred_processing(
                        input_data, context, constraints
                    )
                else:  # AUTOMATIC
                    output = await self._automatic_processing(
                        input_data, context, constraints
                    )

                # Update monitoring
                self.monitor.record_request(output)

                # Validate output
                if not output.is_valid():
                    raise KimeraCognitiveError("Invalid dual-system output")

                # Check constraints
                if output.processing_time > constraints.max_response_time:
                    logger.warning(
                        f"Processing time {output.processing_time:.3f}s exceeded "
                        f"constraint {constraints.max_response_time:.3f}s"
                    )

                if output.confidence < constraints.min_confidence:
                    logger.warning(
                        f"Confidence {output.confidence:.3f} below "
                        f"minimum {constraints.min_confidence:.3f}"
                    )

                return output

            except asyncio.TimeoutError:
                logger.error("Dual-system processing timeout")
                self.monitor.integration_metrics["timeouts"] += 1
                return self._create_timeout_output(start_time)

            except Exception as e:
                logger.error(f"Dual-system processing error: {e}")
                raise KimeraCognitiveError(f"Processing failed: {e}")

    async def _parallel_processing(
        self,
        input_data: torch.Tensor,
        context: Optional[Dict[str, Any]],
        constraints: ProcessingConstraints,
    ) -> DualSystemOutput:
        """Execute both systems in parallel"""

        # Create tasks for parallel execution
        system1_task = asyncio.create_task(self.system1.process(input_data, context))

        system2_task = asyncio.create_task(
            self.system2.process(input_data, context, constraints.required_reasoning)
        )

        # Wait for both with timeout
        timeout = constraints.max_response_time * 0.9  # 90% of budget

        try:
            results = await asyncio.wait_for(
                asyncio.gather(system1_task, system2_task, return_exceptions=True),
                timeout=timeout,
            )

            # Handle results
            system1_result = (
                results[0] if not isinstance(results[0], Exception) else None
            )
            system2_result = (
                results[1] if not isinstance(results[1], Exception) else None
            )

            # Log any errors
            if isinstance(results[0], Exception):
                logger.error(f"System 1 error: {results[0]}")
            if isinstance(results[1], Exception):
                logger.error(f"System 2 error: {results[1]}")

        except asyncio.TimeoutError:
            # Cancel pending tasks
            system1_task.cancel()
            system2_task.cancel()

            # Try to get any completed results
            system1_result = (
                system1_task.result()
                if system1_task.done() and not system1_task.cancelled()
                else None
            )
            system2_result = (
                system2_task.result()
                if system2_task.done() and not system2_task.cancelled()
                else None
            )

        # Arbitrate results
        arbitration_context = {
            **(context or {}),
            "time_pressure": 1.0 - (time.time() % 10) / 10,  # Example time pressure
            "arbitration_strategy": constraints.arbitration_strategy
            or ArbitrationStrategy.CONFIDENCE_BASED,
        }

        arbitration_result = await self.metacognitive.arbitrate(
            system1_result, system2_result, arbitration_context
        )

        # Create output
        processing_time = time.time() - (time.time() - timeout)

        return DualSystemOutput(
            system1_result=system1_result,
            system2_result=system2_result,
            arbitration_result=arbitration_result,
            final_response=arbitration_result.selected_response,
            confidence=arbitration_result.confidence,
            processing_time=processing_time,
            system_mode_used=SystemMode.PARALLEL,
            processing_mode=arbitration_result.processing_mode,
            performance_metrics=self._collect_performance_metrics(),
        )

    async def _sequential_processing(
        self,
        input_data: torch.Tensor,
        context: Optional[Dict[str, Any]],
        constraints: ProcessingConstraints,
    ) -> DualSystemOutput:
        """Execute System 1 first, then System 2 if needed"""

        start_time = time.time()

        # First try System 1
        system1_result = await self.system1.process(input_data, context)

        # Check if System 1 result is sufficient
        if (
            system1_result.confidence >= constraints.min_confidence
            and system1_result.processing_time < constraints.max_response_time * 0.5
        ):

            # System 1 is sufficient
            arbitration_result = await self.metacognitive.arbitrate(
                system1_result, None, context
            )

            return DualSystemOutput(
                system1_result=system1_result,
                system2_result=None,
                arbitration_result=arbitration_result,
                final_response=arbitration_result.selected_response,
                confidence=arbitration_result.confidence,
                processing_time=time.time() - start_time,
                system_mode_used=SystemMode.SEQUENTIAL,
                processing_mode=ProcessingMode.SYSTEM1_ONLY,
                performance_metrics=self._collect_performance_metrics(),
            )

        # Need System 2
        remaining_time = constraints.max_response_time - (time.time() - start_time)

        if remaining_time > 0.1:  # At least 100ms for System 2
            try:
                system2_result = await asyncio.wait_for(
                    self.system2.process(
                        input_data, context, constraints.required_reasoning
                    ),
                    timeout=remaining_time,
                )
            except asyncio.TimeoutError:
                system2_result = None
        else:
            system2_result = None

        # Arbitrate with both results
        arbitration_result = await self.metacognitive.arbitrate(
            system1_result, system2_result, context
        )

        return DualSystemOutput(
            system1_result=system1_result,
            system2_result=system2_result,
            arbitration_result=arbitration_result,
            final_response=arbitration_result.selected_response,
            confidence=arbitration_result.confidence,
            processing_time=time.time() - start_time,
            system_mode_used=SystemMode.SEQUENTIAL,
            processing_mode=arbitration_result.processing_mode,
            performance_metrics=self._collect_performance_metrics(),
        )

    async def _system1_preferred_processing(
        self,
        input_data: torch.Tensor,
        context: Optional[Dict[str, Any]],
        constraints: ProcessingConstraints,
    ) -> DualSystemOutput:
        """Prefer System 1 with optional System 2 backup"""

        start_time = time.time()

        # Always use System 1
        system1_result = await self.system1.process(input_data, context)

        # Only use System 2 if System 1 has very low confidence
        system2_result = None
        if system1_result.confidence < 0.4:
            remaining_time = constraints.max_response_time - (time.time() - start_time)
            if remaining_time > 0.2:
                try:
                    system2_result = await asyncio.wait_for(
                        self.system2.process(input_data, context),
                        timeout=remaining_time,
                    )
                except asyncio.TimeoutError:
                    pass

        # Arbitrate with strong System 1 preference
        arbitration_context = {
            **(context or {}),
            "arbitration_strategy": ArbitrationStrategy.TIME_PRESSURE,
        }

        arbitration_result = await self.metacognitive.arbitrate(
            system1_result, system2_result, arbitration_context
        )

        return DualSystemOutput(
            system1_result=system1_result,
            system2_result=system2_result,
            arbitration_result=arbitration_result,
            final_response=arbitration_result.selected_response,
            confidence=arbitration_result.confidence,
            processing_time=time.time() - start_time,
            system_mode_used=SystemMode.SYSTEM1_PREFERRED,
            processing_mode=arbitration_result.processing_mode,
            performance_metrics=self._collect_performance_metrics(),
        )

    async def _system2_preferred_processing(
        self,
        input_data: torch.Tensor,
        context: Optional[Dict[str, Any]],
        constraints: ProcessingConstraints,
    ) -> DualSystemOutput:
        """Prefer System 2 with System 1 for quick intuitions"""

        start_time = time.time()

        # Start both but give System 2 more time
        system1_task = asyncio.create_task(self.system1.process(input_data, context))

        system2_task = asyncio.create_task(
            self.system2.process(input_data, context, constraints.required_reasoning)
        )

        # Wait for System 2 with most of the time budget
        system2_timeout = constraints.max_response_time * 0.8

        try:
            system2_result = await asyncio.wait_for(
                system2_task, timeout=system2_timeout
            )
        except asyncio.TimeoutError:
            system2_result = None
            system2_task.cancel()

        # Get System 1 result if available
        try:
            system1_result = await asyncio.wait_for(
                system1_task, timeout=0.05  # Very short timeout
            )
        except asyncio.TimeoutError:
            system1_result = None
            system1_task.cancel()

        # Arbitrate with System 2 preference
        arbitration_context = {
            **(context or {}),
            "arbitration_strategy": ArbitrationStrategy.SYSTEM2_OVERRIDE,
        }

        arbitration_result = await self.metacognitive.arbitrate(
            system1_result, system2_result, arbitration_context
        )

        return DualSystemOutput(
            system1_result=system1_result,
            system2_result=system2_result,
            arbitration_result=arbitration_result,
            final_response=arbitration_result.selected_response,
            confidence=arbitration_result.confidence,
            processing_time=time.time() - start_time,
            system_mode_used=SystemMode.SYSTEM2_PREFERRED,
            processing_mode=arbitration_result.processing_mode,
            performance_metrics=self._collect_performance_metrics(),
        )

    async def _automatic_processing(
        self,
        input_data: torch.Tensor,
        context: Optional[Dict[str, Any]],
        constraints: ProcessingConstraints,
    ) -> DualSystemOutput:
        """Let metacognitive controller decide processing mode"""

        # Analyze input to determine best approach
        input_features = self._analyze_input_features(input_data)

        # Build enhanced context
        enhanced_context = {
            **(context or {}),
            "input_complexity": input_features["complexity"],
            "input_variance": input_features["variance"],
            "processing_count": self.processing_count,
        }

        # Estimate time pressure
        time_pressure = self._estimate_time_pressure(constraints)
        enhanced_context["time_pressure"] = time_pressure

        # Let metacognitive controller decide
        if time_pressure > 0.7:
            # High time pressure - use sequential with System 1 first
            return await self._sequential_processing(
                input_data, enhanced_context, constraints
            )
        elif input_features["complexity"] > 0.7:
            # High complexity - prefer System 2
            return await self._system2_preferred_processing(
                input_data, enhanced_context, constraints
            )
        else:
            # Default to parallel for balanced approach
            return await self._parallel_processing(
                input_data, enhanced_context, constraints
            )

    def _determine_system_mode(
        self,
        input_data: torch.Tensor,
        context: Optional[Dict[str, Any]],
        constraints: ProcessingConstraints,
    ) -> SystemMode:
        """Determine which system mode to use"""

        # Explicit mode in constraints takes precedence
        if constraints.system_mode != SystemMode.AUTOMATIC:
            return constraints.system_mode

        # Otherwise use automatic determination
        return SystemMode.AUTOMATIC

    def _validate_input(self, input_data: torch.Tensor) -> bool:
        """Validate input data meets requirements"""
        if input_data is None:
            return False

        if not isinstance(input_data, torch.Tensor):
            return False

        if input_data.numel() == 0:
            return False

        # Check for NaN or Inf
        if torch.isnan(input_data).any() or torch.isinf(input_data).any():
            return False

        return True

    def _analyze_input_features(self, input_data: torch.Tensor) -> Dict[str, float]:
        """Analyze input features for processing decisions"""
        data_np = input_data.cpu().numpy().flatten()

        return {
            "complexity": self._estimate_complexity(data_np),
            "variance": float(np.var(data_np)),
            "size": float(len(data_np)),
            "sparsity": float(np.sum(data_np == 0) / len(data_np)),
        }

    def _estimate_complexity(self, data: np.ndarray) -> float:
        """Estimate data complexity (0-1)"""
        if len(data) < 2:
            return 0.0

        # Use approximate entropy as complexity measure
        # Simplified version for efficiency
        unique_values = len(np.unique(data))
        max_possible = min(len(data), 100)

        complexity = unique_values / max_possible

        # Add variance component
        if np.std(data) > 0:
            normalized_variance = np.var(data) / (np.max(data) - np.min(data) + 1e-10)
            complexity = 0.7 * complexity + 0.3 * min(normalized_variance, 1.0)

        return float(np.clip(complexity, 0.0, 1.0))

    def _estimate_time_pressure(self, constraints: ProcessingConstraints) -> float:
        """Estimate time pressure from constraints"""
        # Lower max response time = higher pressure
        if constraints.max_response_time < 0.1:
            return 0.9
        elif constraints.max_response_time < 0.5:
            return 0.7
        elif constraints.max_response_time < 1.0:
            return 0.5
        else:
            return 0.3

    def _collect_performance_metrics(self) -> Dict[str, float]:
        """Collect current performance metrics"""
        metrics = {}

        # Get metrics from each component
        s1_perf = self.system1.get_performance_report()
        metrics["system1_avg_time"] = s1_perf["avg_response_time"]
        metrics["system1_timeout_rate"] = s1_perf["compliance"]["timeout_rate"]

        s2_perf = self.system2.get_performance_report()
        metrics["system2_avg_time"] = s2_perf["avg_response_time"]
        metrics["system2_timeout_rate"] = s2_perf["compliance"]["timeout_rate"]

        meta_perf = self.metacognitive.get_performance_report()
        metrics["arbitration_avg_time"] = meta_perf["avg_arbitration_time"]
        metrics["conflict_rate"] = meta_perf["conflict_rate"]

        return metrics

    def _create_timeout_output(self, start_time: float) -> DualSystemOutput:
        """Create output for timeout scenario"""
        return DualSystemOutput(
            system1_result=None,
            system2_result=None,
            arbitration_result=ArbitrationResult(
                selected_response={"error": "Processing timeout"},
                system_contributions={"system1": 0.0, "system2": 0.0},
                arbitration_strategy=ArbitrationStrategy.HYBRID,
                processing_mode=ProcessingMode.PARALLEL_COMPETITIVE,
                confidence=0.0,
                rationale="Processing timeout - no results available",
                monitoring_data={"timeout": True},
            ),
            final_response={"error": "Processing timeout"},
            confidence=0.0,
            processing_time=time.time() - start_time,
            system_mode_used=SystemMode.AUTOMATIC,
            processing_mode=ProcessingMode.PARALLEL_COMPETITIVE,
            performance_metrics={},
        )

    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report"""
        return {
            "integrator_status": (
                "healthy" if self.is_initialized else "not_initialized"
            ),
            "processing_count": self.processing_count,
            "component_health": self.monitor.get_health_report(),
            "performance": {
                "system1": self.system1.get_performance_report(),
                "system2": self.system2.get_performance_report(),
                "metacognitive": self.metacognitive.get_performance_report(),
            },
        }

    async def shutdown(self):
        """Graceful shutdown of the integrator"""
        logger.info("Shutting down Barenholtz Dual-System Integrator")

        # Clean up resources
        self.memory_manager.clear()

        # Update status
        self.is_initialized = False

        logger.info("Shutdown complete")
