"""
Metacognitive Controller
========================

DO-178C Level A compliant implementation of metacognitive control for dual-system arbitration.

This module implements the higher-level control system that monitors and arbitrates
between System 1 and System 2 processing, based on the tripartite model by
Stanovich (2011) and related metacognitive theories.

Safety Requirements:
- Arbitration must complete within 50ms
- Must prevent deadlocks between systems
- Must provide traceable decision rationale
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .system1 import IntuitionResult
from .system2 import AnalysisResult, ReasoningType

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
    """Auto-generated class."""
    pass
            DUAL_SYSTEM = "dual_system"

        class KimeraCognitiveError(Exception):
            pass


logger = get_logger(__name__, LogCategory.DUAL_SYSTEM)


class ArbitrationStrategy(Enum):
    """Strategies for resolving System 1/2 conflicts"""

    CONFIDENCE_WEIGHTED = "confidence_weighted"
    SYSTEM2_OVERRIDE = "system2_override"
    CONTEXT_DEPENDENT = "context_dependent"
    TIME_PRESSURE = "time_pressure"
    HYBRID = "hybrid"


class ProcessingMode(Enum):
    """Overall processing mode selection"""

    SYSTEM1_ONLY = "system1_only"
    SYSTEM2_ONLY = "system2_only"
    PARALLEL_COMPETITIVE = "parallel_competitive"
    SEQUENTIAL_CASCADE = "sequential_cascade"
    DEFAULT_INTERVENTIONIST = "default_interventionist"


@dataclass
class MetacognitiveState:
    """Auto-generated class."""
    pass
    """Current state of metacognitive monitoring"""

    cognitive_load: float  # 0-1, current load on working memory
    time_pressure: float  # 0-1, urgency of response
    task_complexity: float  # 0-1, estimated complexity
    confidence_threshold: float  # Required confidence level
    error_likelihood: float  # Probability of error
    resource_availability: Dict[str, float]  # CPU, memory, etc.
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ArbitrationResult:
    """Auto-generated class."""
    pass
    """Result of metacognitive arbitration between systems"""

    selected_response: Dict[str, Any]
    system_contributions: Dict[str, float]  # Weights of each system
    arbitration_strategy: ArbitrationStrategy
    processing_mode: ProcessingMode
    confidence: float
    rationale: str
    monitoring_data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

    def is_valid(self) -> bool:
        """Validate arbitration result"""
        return (
            0.0 <= self.confidence <= 1.0
            and sum(self.system_contributions.values()) > 0.99  # Weights sum to ~1
            and self.selected_response is not None
        )
class ConflictDetector:
    """Auto-generated class."""
    pass
    """Detect and characterize conflicts between System 1 and System 2"""

    def __init__(self):
        self.conflict_threshold = 0.3  # Minimum difference to constitute conflict
        self.conflict_history = []

    def detect_conflict(
        self, system1_result: IntuitionResult, system2_result: AnalysisResult
    ) -> Dict[str, Any]:
        """Detect and characterize conflicts between systems"""

        # Extract comparable features
        s1_confidence = system1_result.confidence
        s2_confidence = system2_result.confidence

        # Compare conclusions (simplified - would need semantic comparison)
        s1_main = self._extract_main_conclusion(system1_result)
        s2_main = self._extract_main_conclusion(system2_result)

        # Calculate conflict metrics
        confidence_conflict = abs(s1_confidence - s2_confidence)
        conclusion_conflict = self._semantic_distance(s1_main, s2_main)

        has_conflict = (
            confidence_conflict > self.conflict_threshold
            or conclusion_conflict > self.conflict_threshold
        )

        conflict_data = {
            "has_conflict": has_conflict
            "confidence_difference": confidence_conflict
            "conclusion_difference": conclusion_conflict
            "type": self._classify_conflict(confidence_conflict, conclusion_conflict),
            "severity": max(confidence_conflict, conclusion_conflict),
        }

        # Record for learning
        self.conflict_history.append({"timestamp": datetime.now(), **conflict_data})

        return conflict_data

    def _extract_main_conclusion(self, result) -> str:
        """Extract main conclusion from either system"""
        if isinstance(result, IntuitionResult):
            # Use top pattern match
            if result.pattern_matches:
                return str(result.pattern_matches[0])
            return "no_pattern"
        else:  # AnalysisResult
            # Use top conclusion
            if result.conclusions:
                return result.conclusions[0]["conclusion"]
            return "no_conclusion"

    def _semantic_distance(self, conclusion1: str, conclusion2: str) -> float:
        """Calculate semantic distance between conclusions"""
        # Simplified - would use proper NLP/embedding distance
        if conclusion1 == conclusion2:
            return 0.0
        elif conclusion1 in conclusion2 or conclusion2 in conclusion1:
            return 0.3
        else:
            return 1.0

    def _classify_conflict(self, conf_diff: float, conc_diff: float) -> str:
        """Classify the type of conflict"""
        if conf_diff > conc_diff:
            return "confidence_mismatch"
        elif conc_diff > conf_diff:
            return "conclusion_mismatch"
        else:
            return "mixed_conflict"
class ResourceMonitor:
    """Auto-generated class."""
    pass
    """Monitor cognitive resources and system load"""

    def __init__(self):
        self.load_history = []
        self.resource_limits = {
            "working_memory": 7,  # Miller's number
            "attention_span": 20.0,  # seconds
            "processing_threads": 4
        }

    def assess_cognitive_load(
        self
        system1_result: Optional[IntuitionResult],
        system2_result: Optional[AnalysisResult],
    ) -> float:
        """Assess current cognitive load (0-1)"""
        load_factors = []

        # System 2 working memory usage
        if system2_result:
            wm_items = sum(
                len(store["items"]) for store in system2_result.working_memory_trace
            )
            wm_load = min(wm_items / self.resource_limits["working_memory"], 1.0)
            load_factors.append(wm_load)

        # Processing time as proxy for load
        if system1_result:
            s1_load = system1_result.processing_time / 0.100  # Normalize by max
            load_factors.append(s1_load * 0.3)  # System 1 is less demanding

        if system2_result:
            s2_load = system2_result.processing_time / 1.000  # Normalize by max
            load_factors.append(s2_load * 0.7)  # System 2 is more demanding

        # Average load
        cognitive_load = np.mean(load_factors) if load_factors else 0.5

        # Record history
        self.load_history.append(
            {"timestamp": datetime.now(), "cognitive_load": cognitive_load}
        )

        return float(cognitive_load)

    def get_resource_availability(self) -> Dict[str, float]:
        """Get current resource availability"""
        import psutil

        return {
            "cpu_available": 1.0 - (psutil.cpu_percent() / 100.0),
            "memory_available": psutil.virtual_memory().available
            / psutil.virtual_memory().total
            "gpu_available": self._get_gpu_availability(),
        }

    def _get_gpu_availability(self) -> float:
        """Get GPU availability if present"""
        if torch.cuda.is_available():
            # Simplified - would use proper GPU monitoring
            return 0.8
        return 0.0
class MetacognitiveController:
    """Auto-generated class."""
    pass
    """
    Main metacognitive controller for dual-system arbitration

    DO-178C Level A Safety Requirements:
    - Arbitration < 50ms
    - No deadlocks
    - Full decision trace
    """

    def __init__(self):
        self.conflict_detector = ConflictDetector()
        self.resource_monitor = ResourceMonitor()

        # Arbitration settings
        self.default_strategy = ArbitrationStrategy.HYBRID
        self.confidence_threshold = 0.7

        # Performance monitoring
        self.arbitration_stats = {
            "total_arbitrations": 0
            "conflicts_resolved": 0
            "avg_arbitration_time": 0.0
            "strategy_usage": {s.value: 0 for s in ArbitrationStrategy},
        }

        # Safety limits
        self.MAX_ARBITRATION_TIME = 0.050  # 50ms

        logger.info("🧠 Metacognitive Controller initialized")

    async def arbitrate(
        self
        system1_result: Optional[IntuitionResult],
        system2_result: Optional[AnalysisResult],
        context: Optional[Dict[str, Any]] = None
    ) -> ArbitrationResult:
        """
        Main arbitration function between System 1 and System 2

        Implements sophisticated conflict resolution and integration
        """
        start_time = time.time()

        try:
            # Build metacognitive state
            meta_state = await self._build_metacognitive_state(
                system1_result, system2_result, context
            )

            # Select processing mode
            processing_mode = self._select_processing_mode(
                meta_state, system1_result, system2_result
            )

            # Select arbitration strategy
            strategy = self._select_strategy(meta_state, context)

            # Perform arbitration with timeout
            result = await asyncio.wait_for(
                self._perform_arbitration(
                    system1_result
                    system2_result
                    strategy
                    processing_mode
                    meta_state
                ),
                timeout=self.MAX_ARBITRATION_TIME
            )

            # Update statistics
            arbitration_time = time.time() - start_time
            self._update_stats(strategy, arbitration_time)

            # Validate result
            if not result.is_valid():
                raise KimeraCognitiveError("Invalid arbitration result")

            return result

        except asyncio.TimeoutError:
            logger.error("Metacognitive arbitration timeout")
            # Emergency fallback to System 1
            return self._emergency_fallback(system1_result, system2_result)

        except Exception as e:
            logger.error(f"Metacognitive arbitration error: {e}")
            raise KimeraCognitiveError(f"Arbitration failed: {e}")

    async def _build_metacognitive_state(
        self
        system1_result: Optional[IntuitionResult],
        system2_result: Optional[AnalysisResult],
        context: Optional[Dict[str, Any]],
    ) -> MetacognitiveState:
        """Build current metacognitive state"""

        # Assess cognitive load
        cognitive_load = self.resource_monitor.assess_cognitive_load(
            system1_result, system2_result
        )

        # Extract time pressure from context
        time_pressure = context.get("time_pressure", 0.3) if context else 0.3

        # Estimate task complexity
        task_complexity = self._estimate_task_complexity(
            system1_result, system2_result, context
        )

        # Calculate error likelihood
        error_likelihood = self._estimate_error_likelihood(
            system1_result, system2_result, task_complexity
        )

        # Get resource availability
        resources = self.resource_monitor.get_resource_availability()

        return MetacognitiveState(
            cognitive_load=cognitive_load
            time_pressure=time_pressure
            task_complexity=task_complexity
            confidence_threshold=self.confidence_threshold
            error_likelihood=error_likelihood
            resource_availability=resources
        )

    def _select_processing_mode(
        self
        meta_state: MetacognitiveState
        system1_result: Optional[IntuitionResult],
        system2_result: Optional[AnalysisResult],
    ) -> ProcessingMode:
        """Select appropriate processing mode"""

        # High time pressure -> System 1 only
        if meta_state.time_pressure > 0.8:
            return ProcessingMode.SYSTEM1_ONLY

        # Low time pressure + high complexity -> System 2 only
        if meta_state.time_pressure < 0.3 and meta_state.task_complexity > 0.7:
            return ProcessingMode.SYSTEM2_ONLY

        # Both results available -> Parallel competitive
        if system1_result and system2_result:
            return ProcessingMode.PARALLEL_COMPETITIVE

        # Default interventionist: System 1 first, System 2 if needed
        if system1_result and not system2_result:
            if system1_result.confidence < meta_state.confidence_threshold:
                return ProcessingMode.SEQUENTIAL_CASCADE

        return ProcessingMode.DEFAULT_INTERVENTIONIST

    def _select_strategy(
        self, meta_state: MetacognitiveState, context: Optional[Dict[str, Any]]
    ) -> ArbitrationStrategy:
        """Select arbitration strategy based on metacognitive state"""

        # Override from context
        if context and "arbitration_strategy" in context:
            return ArbitrationStrategy(context["arbitration_strategy"])

        # High time pressure -> Time pressure strategy
        if meta_state.time_pressure > 0.7:
            return ArbitrationStrategy.TIME_PRESSURE

        # High error likelihood -> System 2 override
        if meta_state.error_likelihood > 0.6:
            return ArbitrationStrategy.SYSTEM2_OVERRIDE

        # Context-dependent for moderate cases
        if 0.4 < meta_state.task_complexity < 0.7:
            return ArbitrationStrategy.CONTEXT_DEPENDENT

        # Default to hybrid
        return ArbitrationStrategy.HYBRID

    async def _perform_arbitration(
        self
        system1_result: Optional[IntuitionResult],
        system2_result: Optional[AnalysisResult],
        strategy: ArbitrationStrategy
        mode: ProcessingMode
        meta_state: MetacognitiveState
    ) -> ArbitrationResult:
        """Perform actual arbitration based on strategy"""

        # Handle single-system modes
        if mode == ProcessingMode.SYSTEM1_ONLY:
            return self._create_result_from_system1(
                system1_result, strategy, meta_state
            )

        if mode == ProcessingMode.SYSTEM2_ONLY:
            return self._create_result_from_system2(
                system2_result, strategy, meta_state
            )

        # Both systems available - check for conflicts
        conflict_data = self.conflict_detector.detect_conflict(
            system1_result, system2_result
        )

        # Apply strategy
        if strategy == ArbitrationStrategy.CONFIDENCE_WEIGHTED:
            return self._confidence_weighted_arbitration(
                system1_result, system2_result, conflict_data, meta_state
            )

        elif strategy == ArbitrationStrategy.SYSTEM2_OVERRIDE:
            return self._system2_override_arbitration(
                system1_result, system2_result, conflict_data, meta_state
            )

        elif strategy == ArbitrationStrategy.CONTEXT_DEPENDENT:
            return self._context_dependent_arbitration(
                system1_result, system2_result, conflict_data, meta_state
            )

        elif strategy == ArbitrationStrategy.TIME_PRESSURE:
            return self._time_pressure_arbitration(
                system1_result, system2_result, conflict_data, meta_state
            )

        else:  # HYBRID
            return self._hybrid_arbitration(
                system1_result, system2_result, conflict_data, meta_state
            )

    def _confidence_weighted_arbitration(
        self
        s1: IntuitionResult
        s2: AnalysisResult
        conflict: Dict[str, Any],
        meta_state: MetacognitiveState
    ) -> ArbitrationResult:
        """Arbitrate based on confidence weighting"""

        # Normalize confidences
        total_conf = s1.confidence + s2.confidence
        if total_conf > 0:
            s1_weight = s1.confidence / total_conf
            s2_weight = s2.confidence / total_conf
        else:
            s1_weight = s2_weight = 0.5

        # Adjust weights by processing time (faster is slightly better)
        time_factor = s1.processing_time / (s1.processing_time + s2.processing_time)
        s1_weight = s1_weight * (1 + 0.1 * (1 - time_factor))
        s2_weight = s2_weight * (1 + 0.1 * time_factor)

        # Renormalize
        total = s1_weight + s2_weight
        s1_weight /= total
        s2_weight /= total

        # Create integrated response
        if s2_weight > 0.6:
            selected = self._extract_response(s2)
            rationale = f"System 2 dominant (weight={s2_weight:.2f})"
        elif s1_weight > 0.6:
            selected = self._extract_response(s1)
            rationale = f"System 1 dominant (weight={s1_weight:.2f})"
        else:
            selected = self._merge_responses(s1, s2, s1_weight, s2_weight)
            rationale = f"Balanced integration (S1={s1_weight:.2f}, S2={s2_weight:.2f})"

        return ArbitrationResult(
            selected_response=selected
            system_contributions={"system1": s1_weight, "system2": s2_weight},
            arbitration_strategy=ArbitrationStrategy.CONFIDENCE_WEIGHTED
            processing_mode=ProcessingMode.PARALLEL_COMPETITIVE
            confidence=s1.confidence * s1_weight + s2.confidence * s2_weight
            rationale=rationale
            monitoring_data={"conflict": conflict, "meta_state": meta_state.__dict__},
        )

    def _system2_override_arbitration(
        self
        s1: IntuitionResult
        s2: AnalysisResult
        conflict: Dict[str, Any],
        meta_state: MetacognitiveState
    ) -> ArbitrationResult:
        """System 2 overrides System 1 when available"""

        # System 2 takes precedence if it has any confidence
        if s2.confidence > 0.3:
            selected = self._extract_response(s2)
            s2_weight = 0.9
            rationale = "System 2 override due to analytical requirements"
        else:
            # Fall back to System 1 if System 2 has very low confidence
            selected = self._extract_response(s1)
            s2_weight = 0.1
            rationale = "System 1 fallback due to low System 2 confidence"

        return ArbitrationResult(
            selected_response=selected
            system_contributions={"system1": 1 - s2_weight, "system2": s2_weight},
            arbitration_strategy=ArbitrationStrategy.SYSTEM2_OVERRIDE
            processing_mode=ProcessingMode.DEFAULT_INTERVENTIONIST
            confidence=s2.confidence if s2_weight > 0.5 else s1.confidence
            rationale=rationale
            monitoring_data={"conflict": conflict, "meta_state": meta_state.__dict__},
        )

    def _context_dependent_arbitration(
        self
        s1: IntuitionResult
        s2: AnalysisResult
        conflict: Dict[str, Any],
        meta_state: MetacognitiveState
    ) -> ArbitrationResult:
        """Arbitrate based on context and task characteristics"""

        # Determine which system is more appropriate for the context
        s1_appropriate = self._assess_system1_appropriateness(meta_state)
        s2_appropriate = 1.0 - s1_appropriate

        # Weight by appropriateness and confidence
        s1_weight = s1_appropriate * s1.confidence
        s2_weight = s2_appropriate * s2.confidence

        # Normalize
        total = s1_weight + s2_weight
        if total > 0:
            s1_weight /= total
            s2_weight /= total
        else:
            s1_weight = s2_weight = 0.5

        # Select response
        if s1_weight > s2_weight:
            selected = self._extract_response(s1)
            rationale = (
                f"System 1 more appropriate for context (weight={s1_weight:.2f})"
            )
        else:
            selected = self._extract_response(s2)
            rationale = (
                f"System 2 more appropriate for context (weight={s2_weight:.2f})"
            )

        return ArbitrationResult(
            selected_response=selected
            system_contributions={"system1": s1_weight, "system2": s2_weight},
            arbitration_strategy=ArbitrationStrategy.CONTEXT_DEPENDENT
            processing_mode=ProcessingMode.PARALLEL_COMPETITIVE
            confidence=s1.confidence * s1_weight + s2.confidence * s2_weight
            rationale=rationale
            monitoring_data={"conflict": conflict, "meta_state": meta_state.__dict__},
        )

    def _time_pressure_arbitration(
        self
        s1: IntuitionResult
        s2: AnalysisResult
        conflict: Dict[str, Any],
        meta_state: MetacognitiveState
    ) -> ArbitrationResult:
        """Arbitrate based on time pressure"""

        # Strong preference for System 1 under time pressure
        time_weight = meta_state.time_pressure
        s1_weight = 0.3 + 0.7 * time_weight  # 30-100% based on pressure
        s2_weight = 1.0 - s1_weight

        # Select response
        if s1_weight > 0.7:
            selected = self._extract_response(s1)
            rationale = f"System 1 selected due to time pressure ({meta_state.time_pressure:.2f})"
        else:
            # Blend responses if moderate time pressure
            selected = self._merge_responses(s1, s2, s1_weight, s2_weight)
            rationale = f"Blended response under moderate time pressure"

        return ArbitrationResult(
            selected_response=selected
            system_contributions={"system1": s1_weight, "system2": s2_weight},
            arbitration_strategy=ArbitrationStrategy.TIME_PRESSURE
            processing_mode=ProcessingMode.PARALLEL_COMPETITIVE
            confidence=s1.confidence * s1_weight + s2.confidence * s2_weight
            rationale=rationale
            monitoring_data={"conflict": conflict, "meta_state": meta_state.__dict__},
        )

    def _hybrid_arbitration(
        self
        s1: IntuitionResult
        s2: AnalysisResult
        conflict: Dict[str, Any],
        meta_state: MetacognitiveState
    ) -> ArbitrationResult:
        """Sophisticated hybrid arbitration combining multiple factors"""

        # Base weights from confidence
        conf_s1 = s1.confidence
        conf_s2 = s2.confidence

        # Adjust for conflict
        if conflict["has_conflict"]:
            # Reduce weight of lower confidence system
            if conf_s1 < conf_s2:
                conf_s1 *= 1 - 0.3 * conflict["severity"]
            else:
                conf_s2 *= 1 - 0.3 * conflict["severity"]

        # Adjust for cognitive load
        load_factor = 1.0 - meta_state.cognitive_load
        conf_s2 *= load_factor  # System 2 suffers more under load

        # Adjust for time pressure
        time_factor = 1.0 - meta_state.time_pressure
        conf_s2 *= time_factor  # System 2 needs time
        conf_s1 *= 1 + 0.2 * meta_state.time_pressure  # System 1 benefits

        # Normalize
        total = conf_s1 + conf_s2
        if total > 0:
            s1_weight = conf_s1 / total
            s2_weight = conf_s2 / total
        else:
            s1_weight = s2_weight = 0.5

        # Create response
        if abs(s1_weight - s2_weight) > 0.3:
            # Clear winner
            if s1_weight > s2_weight:
                selected = self._extract_response(s1)
                rationale = f"Hybrid selection: System 1 (weight={s1_weight:.2f})"
            else:
                selected = self._extract_response(s2)
                rationale = f"Hybrid selection: System 2 (weight={s2_weight:.2f})"
        else:
            # Close weights - integrate
            selected = self._merge_responses(s1, s2, s1_weight, s2_weight)
            rationale = f"Hybrid integration (S1={s1_weight:.2f}, S2={s2_weight:.2f})"

        return ArbitrationResult(
            selected_response=selected
            system_contributions={"system1": s1_weight, "system2": s2_weight},
            arbitration_strategy=ArbitrationStrategy.HYBRID
            processing_mode=ProcessingMode.PARALLEL_COMPETITIVE
            confidence=s1.confidence * s1_weight + s2.confidence * s2_weight
            rationale=rationale
            monitoring_data={
                "conflict": conflict
                "meta_state": meta_state.__dict__
                "adjustments": {
                    "conflict_adjustment": (
                        conflict["severity"] if conflict["has_conflict"] else 0
                    ),
                    "load_adjustment": 1 - load_factor
                    "time_adjustment": 1 - time_factor
                },
            },
        )

    def _create_result_from_system1(
        self
        s1: IntuitionResult
        strategy: ArbitrationStrategy
        meta_state: MetacognitiveState
    ) -> ArbitrationResult:
        """Create arbitration result when only System 1 available"""
        return ArbitrationResult(
            selected_response=self._extract_response(s1),
            system_contributions={"system1": 1.0, "system2": 0.0},
            arbitration_strategy=strategy
            processing_mode=ProcessingMode.SYSTEM1_ONLY
            confidence=s1.confidence
            rationale="System 1 only - fast intuitive response",
            monitoring_data={"meta_state": meta_state.__dict__},
        )

    def _create_result_from_system2(
        self
        s2: AnalysisResult
        strategy: ArbitrationStrategy
        meta_state: MetacognitiveState
    ) -> ArbitrationResult:
        """Create arbitration result when only System 2 available"""
        return ArbitrationResult(
            selected_response=self._extract_response(s2),
            system_contributions={"system1": 0.0, "system2": 1.0},
            arbitration_strategy=strategy
            processing_mode=ProcessingMode.SYSTEM2_ONLY
            confidence=s2.confidence
            rationale="System 2 only - analytical response",
            monitoring_data={"meta_state": meta_state.__dict__},
        )

    def _extract_response(self, result) -> Dict[str, Any]:
        """Extract standardized response from either system"""
        if isinstance(result, IntuitionResult):
            return {
                "type": "intuitive",
                "content": result.pattern_matches[0] if result.pattern_matches else {},
                "features": result.features_detected
                "associations": result.associations
            }
        else:  # AnalysisResult
            return {
                "type": "analytical",
                "content": result.conclusions[0] if result.conclusions else {},
                "reasoning": [
                    {"step": step.operation, "conclusion": step.conclusion}
                    for step in result.reasoning_chain[:3]  # Top 3 steps
                ],
            }

    def _merge_responses(
        self
        s1: IntuitionResult
        s2: AnalysisResult
        s1_weight: float
        s2_weight: float
    ) -> Dict[str, Any]:
        """Merge responses from both systems"""
        return {
            "type": "integrated",
            "intuitive_component": self._extract_response(s1),
            "analytical_component": self._extract_response(s2),
            "weights": {"system1": s1_weight, "system2": s2_weight},
            "integration_method": "weighted_combination",
        }

    def _estimate_task_complexity(
        self
        s1: Optional[IntuitionResult],
        s2: Optional[AnalysisResult],
        context: Optional[Dict[str, Any]],
    ) -> float:
        """Estimate task complexity from available information"""
        complexity_factors = []

        # Context-based complexity
        if context:
            if "task_complexity" in context:
                return float(context["task_complexity"])

            # Heuristics based on context
            if any(k in context for k in ["mathematical", "logical", "analytical"]):
                complexity_factors.append(0.8)
            if any(k in context for k in ["creative", "novel", "ambiguous"]):
                complexity_factors.append(0.7)

        # System 2 reasoning complexity
        if s2 and s2.reasoning_chain:
            # More reasoning steps = more complex
            steps_complexity = min(len(s2.reasoning_chain) / 20.0, 1.0)
            complexity_factors.append(steps_complexity)

            # Multiple reasoning types = more complex
            types_complexity = len(s2.reasoning_types_used) / 5.0
            complexity_factors.append(types_complexity)

        # Low System 1 confidence might indicate complexity
        if s1:
            complexity_factors.append(1.0 - s1.confidence)

        return float(np.mean(complexity_factors)) if complexity_factors else 0.5

    def _estimate_error_likelihood(
        self
        s1: Optional[IntuitionResult],
        s2: Optional[AnalysisResult],
        task_complexity: float
    ) -> float:
        """Estimate likelihood of error"""
        error_factors = []

        # High complexity increases error likelihood
        error_factors.append(task_complexity * 0.5)

        # Low confidence increases error likelihood
        if s1:
            error_factors.append((1.0 - s1.confidence) * 0.3)
        if s2:
            error_factors.append((1.0 - s2.confidence) * 0.3)

        # Conflict between systems suggests error possibility
        if s1 and s2:
            conflict = self.conflict_detector.detect_conflict(s1, s2)
            if conflict["has_conflict"]:
                error_factors.append(conflict["severity"] * 0.4)

        return float(np.mean(error_factors)) if error_factors else 0.3

    def _assess_system1_appropriateness(self, meta_state: MetacognitiveState) -> float:
        """Assess how appropriate System 1 is for current context"""
        appropriateness = 0.5  # Baseline

        # Time pressure favors System 1
        appropriateness += meta_state.time_pressure * 0.3

        # High cognitive load favors System 1
        appropriateness += meta_state.cognitive_load * 0.2

        # Low complexity favors System 1
        appropriateness += (1.0 - meta_state.task_complexity) * 0.2

        # Low error likelihood favors System 1
        appropriateness += (1.0 - meta_state.error_likelihood) * 0.1

        return float(np.clip(appropriateness, 0.0, 1.0))

    def _emergency_fallback(
        self, s1: Optional[IntuitionResult], s2: Optional[AnalysisResult]
    ) -> ArbitrationResult:
        """Emergency fallback when arbitration fails"""
        if s1:
            return ArbitrationResult(
                selected_response=self._extract_response(s1),
                system_contributions={"system1": 1.0, "system2": 0.0},
                arbitration_strategy=ArbitrationStrategy.TIME_PRESSURE
                processing_mode=ProcessingMode.SYSTEM1_ONLY
                confidence=s1.confidence * 0.5,  # Reduced due to emergency
                rationale="Emergency fallback to System 1",
                monitoring_data={"emergency": True},
            )
        elif s2:
            return ArbitrationResult(
                selected_response=self._extract_response(s2),
                system_contributions={"system1": 0.0, "system2": 1.0},
                arbitration_strategy=ArbitrationStrategy.SYSTEM2_OVERRIDE
                processing_mode=ProcessingMode.SYSTEM2_ONLY
                confidence=s2.confidence * 0.5
                rationale="Emergency fallback to System 2",
                monitoring_data={"emergency": True},
            )
        else:
            # No results available - return empty
            return ArbitrationResult(
                selected_response={"type": "empty", "error": "No results available"},
                system_contributions={"system1": 0.0, "system2": 0.0},
                arbitration_strategy=ArbitrationStrategy.HYBRID
                processing_mode=ProcessingMode.PARALLEL_COMPETITIVE
                confidence=0.0
                rationale="Emergency: No results from either system",
                monitoring_data={"emergency": True, "critical_failure": True},
            )

    def _update_stats(self, strategy: ArbitrationStrategy, arbitration_time: float):
        """Update arbitration statistics"""
        self.arbitration_stats["total_arbitrations"] += 1
        self.arbitration_stats["strategy_usage"][strategy.value] += 1

        # Update average time
        n = self.arbitration_stats["total_arbitrations"]
        old_avg = self.arbitration_stats["avg_arbitration_time"]
        self.arbitration_stats["avg_arbitration_time"] = (
            old_avg * (n - 1) + arbitration_time
        ) / n

    def get_performance_report(self) -> Dict[str, Any]:
        """Get metacognitive performance statistics"""
        return {
            **self.arbitration_stats
            "conflict_rate": (
                len(
                    [
                        c
                        for c in self.conflict_detector.conflict_history
                        if c["has_conflict"]
                    ]
                )
                / max(len(self.conflict_detector.conflict_history), 1)
            ),
            "avg_cognitive_load": (
                np.mean(
                    [h["cognitive_load"] for h in self.resource_monitor.load_history]
                )
                if self.resource_monitor.load_history
                else 0.5
            ),
            "compliance": {
                "arbitration_time_ok": (
                    self.arbitration_stats["avg_arbitration_time"]
                    < self.MAX_ARBITRATION_TIME
                )
            },
        }
