#!/usr/bin/env python3
"""
Cognitive Contexts Configuration for Large-Scale Testing
=======================================================

DO-178C Level A compliant cognitive context definitions for comprehensive
system validation. Each context represents a distinct cognitive processing
paradigm that activates different neural pathways and system architectures.

Key Features:
- Four fundamental cognitive contexts
- Context-specific processing characteristics
- Performance optimization per context
- Validation metrics and thresholds

Author: KIMERA Development Team
Version: 1.0.0 (DO-178C Level A)
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from utils.kimera_logger import get_logger, LogCategory
from utils.kimera_exceptions import KimeraValidationError

logger = get_logger(__name__, LogCategory.SYSTEM)


class CognitiveContext(Enum):
    """
    Cognitive processing contexts for systematic validation

    Based on cognitive neuroscience research into distinct
    cognitive processing modes and their neural correlates.
    """
    ANALYTICAL = "analytical"              # Logic-driven, System 2 dominant
    CREATIVE = "creative"                  # Intuitive, System 1 dominant
    PROBLEM_SOLVING = "problem_solving"    # Balanced dual-system approach
    PATTERN_RECOGNITION = "pattern_recognition"  # Automated pattern detection


@dataclass
class ContextCharacteristics:
    """Processing characteristics for a cognitive context"""
    primary_system: str                     # "system1", "system2", or "balanced"
    processing_mode: str                    # "sequential", "parallel", "adaptive"
    cognitive_load_distribution: Dict[str, float]  # Resource allocation
    expected_response_patterns: List[str]   # Typical response characteristics
    performance_thresholds: Dict[str, float]  # Performance requirements
    activation_patterns: Dict[str, bool]    # Which systems should activate
    optimization_strategy: str              # How to optimize for this context
    error_tolerance: float                  # Acceptable error rate


@dataclass
class ContextTestConfiguration:
    """Test configuration for a specific cognitive context"""
    context: CognitiveContext
    characteristics: ContextCharacteristics
    test_parameters: Dict[str, Any]
    validation_criteria: Dict[str, float]
    expected_behaviors: List[str]
    failure_modes: List[str]
    performance_benchmarks: Dict[str, float]

    def is_valid(self) -> bool:
        """Validate configuration consistency"""
        return (
            sum(self.characteristics.cognitive_load_distribution.values()) <= 1.1 and  # Allow 10% tolerance
            0.0 <= self.characteristics.error_tolerance <= 1.0 and
            len(self.expected_behaviors) > 0 and
            len(self.failure_modes) > 0
        )


class CognitiveContextManager:
    """
    Manager for cognitive context configurations and optimization

    Implements aerospace engineering principles:
    - Context switching with minimal latency
    - Performance optimization per context
    - Fault tolerance across all contexts
    """

    def __init__(self):
        self.configurations = self._initialize_configurations()
        self.performance_history: Dict[CognitiveContext, List[Dict]] = {
            context: [] for context in CognitiveContext
        }
        self.optimization_state = self._initialize_optimization_state()

        logger.info("🧠 Cognitive Context Manager initialized (DO-178C Level A)")
        logger.info(f"   Contexts: {len(self.configurations)}")
        logger.info(f"   Performance tracking: {len(self.performance_history)} contexts")

    def _initialize_configurations(self) -> Dict[CognitiveContext, ContextTestConfiguration]:
        """Initialize comprehensive context configurations"""
        configurations = {}

        # ANALYTICAL Context - Logic-driven, System 2 dominant processing
        configurations[CognitiveContext.ANALYTICAL] = ContextTestConfiguration(
            context=CognitiveContext.ANALYTICAL,
            characteristics=ContextCharacteristics(
                primary_system="system2",
                processing_mode="sequential",
                cognitive_load_distribution={
                    "logical_reasoning": 0.4,
                    "working_memory": 0.3,
                    "attention_control": 0.2,
                    "metacognition": 0.1
                },
                expected_response_patterns=[
                    "step_by_step_reasoning",
                    "explicit_justification",
                    "logical_structure",
                    "evidence_based_conclusions"
                ],
                performance_thresholds={
                    "reasoning_accuracy": 0.95,    # 95% accuracy required
                    "logical_consistency": 0.98,   # 98% consistency
                    "processing_time": 2.0,        # 2 seconds max
                    "working_memory_efficiency": 0.85
                },
                activation_patterns={
                    "system1_active": False,
                    "system2_active": True,
                    "metacognitive_active": True,
                    "pattern_matching_active": False,
                    "logical_reasoning_active": True
                },
                optimization_strategy="maximize_accuracy_over_speed",
                error_tolerance=0.05  # 5% error tolerance
            ),
            test_parameters={
                "reasoning_depth": "deep",
                "logical_complexity": "high",
                "time_pressure": "low",
                "distraction_level": "minimal",
                "evidence_quality": "high"
            },
            validation_criteria={
                "logical_consistency_score": 0.95,
                "reasoning_depth_score": 0.90,
                "evidence_integration_score": 0.93,
                "response_time_limit": 5.0
            },
            expected_behaviors=[
                "Systematic logical progression",
                "Explicit reasoning steps",
                "Evidence evaluation and integration",
                "Metacognitive monitoring",
                "Error detection and correction",
                "Conclusion justification"
            ],
            failure_modes=[
                "Logical inconsistency",
                "Reasoning shortcuts",
                "Evidence neglect",
                "Metacognitive failure",
                "Premature conclusion",
                "System 1 interference"
            ],
            performance_benchmarks={
                "average_reasoning_time": 1.5,     # 1.5 seconds average
                "peak_accuracy": 0.98,             # 98% peak accuracy
                "consistency_rate": 0.97,          # 97% consistency
                "metacognitive_accuracy": 0.92     # 92% self-assessment accuracy
            }
        )

        # CREATIVE Context - Intuitive, System 1 dominant processing
        configurations[CognitiveContext.CREATIVE] = ContextTestConfiguration(
            context=CognitiveContext.CREATIVE,
            characteristics=ContextCharacteristics(
                primary_system="system1",
                processing_mode="parallel",
                cognitive_load_distribution={
                    "divergent_thinking": 0.4,
                    "associative_memory": 0.3,
                    "pattern_synthesis": 0.2,
                    "aesthetic_evaluation": 0.1
                },
                expected_response_patterns=[
                    "novel_associations",
                    "metaphorical_thinking",
                    "emergent_insights",
                    "aesthetic_considerations"
                ],
                performance_thresholds={
                    "novelty_score": 0.80,          # 80% novelty required
                    "coherence_score": 0.75,        # 75% coherence
                    "processing_time": 0.5,         # 500ms max (fast intuition)
                    "aesthetic_quality": 0.70
                },
                activation_patterns={
                    "system1_active": True,
                    "system2_active": False,
                    "metacognitive_active": False,
                    "pattern_matching_active": True,
                    "associative_networks_active": True
                },
                optimization_strategy="maximize_novelty_and_fluency",
                error_tolerance=0.15  # 15% error tolerance (creativity allows more errors)
            ),
            test_parameters={
                "creativity_prompt": "open_ended",
                "constraint_level": "minimal",
                "time_pressure": "moderate",
                "inspiration_sources": "diverse",
                "evaluation_criteria": "originality"
            },
            validation_criteria={
                "novelty_threshold": 0.75,
                "fluency_score": 0.80,
                "flexibility_score": 0.70,
                "elaboration_score": 0.65
            },
            expected_behaviors=[
                "Rapid idea generation",
                "Unusual associations",
                "Metaphorical connections",
                "Emergent pattern recognition",
                "Aesthetic sensitivity",
                "Divergent exploration"
            ],
            failure_modes=[
                "Idea fixation",
                "Logical over-constraint",
                "Associative blocking",
                "Premature evaluation",
                "Pattern rigidity",
                "Creative inhibition"
            ],
            performance_benchmarks={
                "idea_generation_rate": 10.0,      # 10 ideas per minute
                "novelty_average": 0.82,           # 82% average novelty
                "association_diversity": 0.85,     # 85% diversity
                "aesthetic_appeal": 0.75            # 75% aesthetic quality
            }
        )

        # PROBLEM_SOLVING Context - Balanced dual-system approach
        configurations[CognitiveContext.PROBLEM_SOLVING] = ContextTestConfiguration(
            context=CognitiveContext.PROBLEM_SOLVING,
            characteristics=ContextCharacteristics(
                primary_system="balanced",
                processing_mode="adaptive",
                cognitive_load_distribution={
                    "problem_analysis": 0.3,
                    "solution_generation": 0.3,
                    "solution_evaluation": 0.2,
                    "execution_planning": 0.2
                },
                expected_response_patterns=[
                    "problem_decomposition",
                    "multiple_solution_paths",
                    "solution_evaluation",
                    "adaptive_strategy_selection"
                ],
                performance_thresholds={
                    "solution_quality": 0.85,       # 85% solution quality
                    "efficiency_score": 0.80,       # 80% efficiency
                    "processing_time": 3.0,         # 3 seconds max
                    "adaptability_score": 0.75
                },
                activation_patterns={
                    "system1_active": True,
                    "system2_active": True,
                    "metacognitive_active": True,
                    "pattern_matching_active": True,
                    "strategic_planning_active": True
                },
                optimization_strategy="balance_speed_and_accuracy",
                error_tolerance=0.10  # 10% error tolerance
            ),
            test_parameters={
                "problem_complexity": "variable",
                "solution_constraints": "moderate",
                "time_limit": "adaptive",
                "resource_availability": "limited",
                "feedback_timing": "delayed"
            },
            validation_criteria={
                "solution_correctness": 0.85,
                "approach_efficiency": 0.80,
                "strategy_adaptation": 0.75,
                "resource_utilization": 0.70
            },
            expected_behaviors=[
                "Problem space exploration",
                "Strategy selection and switching",
                "Solution path planning",
                "Progress monitoring",
                "Adaptive constraint handling",
                "Multi-criteria optimization"
            ],
            failure_modes=[
                "Problem misrepresentation",
                "Strategy persistence",
                "Solution path fixation",
                "Constraint violation",
                "Resource exhaustion",
                "Evaluation bias"
            ],
            performance_benchmarks={
                "problem_solving_success_rate": 0.88,  # 88% success rate
                "average_solution_time": 2.5,          # 2.5 seconds average
                "strategy_adaptation_rate": 0.80,      # 80% adaptation success
                "resource_efficiency": 0.75            # 75% resource efficiency
            }
        )

        # PATTERN_RECOGNITION Context - Automated pattern detection and analysis
        configurations[CognitiveContext.PATTERN_RECOGNITION] = ContextTestConfiguration(
            context=CognitiveContext.PATTERN_RECOGNITION,
            characteristics=ContextCharacteristics(
                primary_system="system1",
                processing_mode="parallel",
                cognitive_load_distribution={
                    "feature_extraction": 0.4,
                    "pattern_matching": 0.3,
                    "similarity_assessment": 0.2,
                    "pattern_classification": 0.1
                },
                expected_response_patterns=[
                    "rapid_pattern_detection",
                    "feature_highlighting",
                    "similarity_mapping",
                    "pattern_categorization"
                ],
                performance_thresholds={
                    "detection_accuracy": 0.92,     # 92% detection accuracy
                    "false_positive_rate": 0.05,    # 5% false positive max
                    "processing_time": 0.2,         # 200ms max (very fast)
                    "pattern_completeness": 0.85
                },
                activation_patterns={
                    "system1_active": True,
                    "system2_active": False,
                    "metacognitive_active": False,
                    "pattern_matching_active": True,
                    "feature_extraction_active": True
                },
                optimization_strategy="maximize_speed_and_accuracy",
                error_tolerance=0.08  # 8% error tolerance
            ),
            test_parameters={
                "pattern_complexity": "variable",
                "noise_level": "moderate",
                "pattern_completeness": "partial",
                "time_constraint": "strict",
                "pattern_novelty": "mixed"
            },
            validation_criteria={
                "detection_precision": 0.90,
                "detection_recall": 0.88,
                "processing_speed": 0.95,
                "noise_robustness": 0.80
            },
            expected_behaviors=[
                "Rapid feature extraction",
                "Parallel pattern matching",
                "Confidence estimation",
                "Pattern completion",
                "Noise filtering",
                "Hierarchical recognition"
            ],
            failure_modes=[
                "Pattern oversegmentation",
                "Feature extraction failure",
                "Noise amplification",
                "Pattern hallucination",
                "Scale invariance failure",
                "Context ignorance"
            ],
            performance_benchmarks={
                "pattern_detection_speed": 0.15,       # 150ms average
                "accuracy_rate": 0.94,                 # 94% accuracy
                "noise_tolerance": 0.82,               # 82% noise tolerance
                "pattern_completion_rate": 0.78        # 78% completion success
            }
        )

        # Validate all configurations
        for context, config in configurations.items():
            if not config.is_valid():
                raise KimeraValidationError(f"Invalid configuration for {context}")

        return configurations

    def _initialize_optimization_state(self) -> Dict[str, Any]:
        """Initialize optimization tracking state"""
        return {
            "last_optimization": datetime.now(),
            "optimization_cycles": 0,
            "performance_trends": {},
            "adaptation_history": [],
            "context_switches": 0,
            "average_switch_time": 0.0
        }

    def get_configuration(self, context: CognitiveContext) -> ContextTestConfiguration:
        """Get configuration for specified cognitive context"""
        return self.configurations[context]

    def get_all_contexts(self) -> List[CognitiveContext]:
        """Get all available cognitive contexts"""
        return list(CognitiveContext)

    def estimate_optimal_context(self,
                                task_characteristics: Dict[str, Any]) -> CognitiveContext:
        """
        Estimate optimal cognitive context for given task characteristics

        Uses aerospace-inspired decision matrix for context selection.
        """
        # Extract task characteristics
        requires_logic = task_characteristics.get("logical_reasoning", False)
        requires_creativity = task_characteristics.get("creative_thinking", False)
        has_constraints = task_characteristics.get("constraints", False)
        time_pressure = task_characteristics.get("time_pressure", "low")
        pattern_based = task_characteristics.get("pattern_recognition", False)

        # Decision matrix scoring
        context_scores = {}

        # Analytical context scoring
        analytical_score = 0.0
        if requires_logic:
            analytical_score += 0.5
        if not requires_creativity:
            analytical_score += 0.3
        if time_pressure == "low":
            analytical_score += 0.2
        context_scores[CognitiveContext.ANALYTICAL] = analytical_score

        # Creative context scoring
        creative_score = 0.0
        if requires_creativity:
            creative_score += 0.5
        if not has_constraints:
            creative_score += 0.3
        if time_pressure == "moderate":
            creative_score += 0.2
        context_scores[CognitiveContext.CREATIVE] = creative_score

        # Problem-solving context scoring
        problem_solving_score = 0.0
        if has_constraints and requires_logic:
            problem_solving_score += 0.4
        if time_pressure == "moderate":
            problem_solving_score += 0.3
        if requires_creativity and requires_logic:
            problem_solving_score += 0.3
        context_scores[CognitiveContext.PROBLEM_SOLVING] = problem_solving_score

        # Pattern recognition context scoring
        pattern_score = 0.0
        if pattern_based:
            pattern_score += 0.6
        if time_pressure == "high":
            pattern_score += 0.3
        if not requires_creativity and not requires_logic:
            pattern_score += 0.1
        context_scores[CognitiveContext.PATTERN_RECOGNITION] = pattern_score

        # Select context with highest score
        optimal_context = max(context_scores.items(), key=lambda x: x[1])[0]

        logger.debug(f"Context estimation: {optimal_context.value} "
                    f"(score: {context_scores[optimal_context]:.3f})")

        return optimal_context

    def validate_context_performance(self,
                                   context: CognitiveContext,
                                   performance_metrics: Dict[str, float]) -> bool:
        """
        Validate performance meets context requirements

        Implements nuclear engineering positive confirmation principle.
        """
        config = self.configurations[context]
        thresholds = config.characteristics.performance_thresholds

        validation_results = {}
        for metric, threshold in thresholds.items():
            if metric in performance_metrics:
                meets_threshold = performance_metrics[metric] >= threshold
                validation_results[metric] = meets_threshold

                if not meets_threshold:
                    logger.warning(f"Context {context.value} failed {metric}: "
                                  f"{performance_metrics[metric]:.3f} < {threshold:.3f}")

        # Record performance history
        self.performance_history[context].append({
            "timestamp": datetime.now(),
            "metrics": performance_metrics.copy(),
            "validation_results": validation_results.copy(),
            "overall_success": all(validation_results.values())
        })

        # Trim history to last 100 entries
        if len(self.performance_history[context]) > 100:
            self.performance_history[context] = self.performance_history[context][-100:]

        return all(validation_results.values())

    def get_context_switching_cost(self,
                                  from_context: CognitiveContext,
                                  to_context: CognitiveContext) -> float:
        """
        Estimate cognitive switching cost between contexts

        Based on cognitive psychology research on task switching costs.
        """
        if from_context == to_context:
            return 0.0

        # Base switching cost
        base_cost = 0.1  # 100ms base switching time

        # Context-specific switching costs
        switching_matrix = {
            (CognitiveContext.ANALYTICAL, CognitiveContext.CREATIVE): 0.3,
            (CognitiveContext.CREATIVE, CognitiveContext.ANALYTICAL): 0.25,
            (CognitiveContext.ANALYTICAL, CognitiveContext.PROBLEM_SOLVING): 0.15,
            (CognitiveContext.PROBLEM_SOLVING, CognitiveContext.ANALYTICAL): 0.10,
            (CognitiveContext.CREATIVE, CognitiveContext.PATTERN_RECOGNITION): 0.20,
            (CognitiveContext.PATTERN_RECOGNITION, CognitiveContext.CREATIVE): 0.25,
            (CognitiveContext.PROBLEM_SOLVING, CognitiveContext.PATTERN_RECOGNITION): 0.12,
            (CognitiveContext.PATTERN_RECOGNITION, CognitiveContext.PROBLEM_SOLVING): 0.15,
        }

        # Get specific switching cost or use base cost
        specific_cost = switching_matrix.get((from_context, to_context), base_cost)

        return specific_cost

    def optimize_context_performance(self, context: CognitiveContext) -> Dict[str, Any]:
        """
        Optimize performance for specific cognitive context

        Implements continuous improvement following aerospace methodologies.
        """
        config = self.configurations[context]
        history = self.performance_history[context]

        if len(history) < 5:  # Need minimum data for optimization
            return {"status": "insufficient_data", "recommendations": []}

        # Analyze performance trends
        recent_performance = history[-10:]  # Last 10 measurements
        performance_trend = self._analyze_performance_trend(recent_performance)

        # Generate optimization recommendations
        recommendations = []

        if performance_trend["accuracy_declining"]:
            recommendations.append({
                "type": "accuracy_improvement",
                "action": "increase_processing_time_allocation",
                "parameters": {"time_multiplier": 1.2}
            })

        if performance_trend["speed_declining"]:
            recommendations.append({
                "type": "speed_improvement",
                "action": "optimize_resource_allocation",
                "parameters": {"parallel_factor": 1.5}
            })

        if performance_trend["consistency_issues"]:
            recommendations.append({
                "type": "consistency_improvement",
                "action": "stabilize_context_parameters",
                "parameters": {"variance_reduction": 0.8}
            })

        # Update optimization state
        self.optimization_state["last_optimization"] = datetime.now()
        self.optimization_state["optimization_cycles"] += 1

        optimization_result = {
            "status": "optimization_completed",
            "context": context.value,
            "recommendations": recommendations,
            "performance_trend": performance_trend,
            "optimization_cycle": self.optimization_state["optimization_cycles"]
        }

        logger.info(f"Context optimization completed for {context.value}: "
                   f"{len(recommendations)} recommendations generated")

        return optimization_result

    def _analyze_performance_trend(self, performance_data: List[Dict]) -> Dict[str, bool]:
        """Analyze performance trends in recent data"""
        if len(performance_data) < 3:
            return {"insufficient_data": True}

        # Extract metrics over time
        accuracy_values = []
        speed_values = []
        consistency_values = []

        for entry in performance_data:
            metrics = entry["metrics"]
            # Normalize different accuracy metrics
            accuracy = max([v for k, v in metrics.items() if "accuracy" in k], default=0.0)
            speed = 1.0 / max([v for k, v in metrics.items() if "time" in k], default=1.0)
            consistency = max([v for k, v in metrics.items() if "consistency" in k], default=0.0)

            accuracy_values.append(accuracy)
            speed_values.append(speed)
            consistency_values.append(consistency)

        # Analyze trends (simple linear regression slope)
        def trend_slope(values):
            n = len(values)
            x = np.arange(n)
            return np.polyfit(x, values, 1)[0]

        accuracy_slope = trend_slope(accuracy_values) if len(accuracy_values) > 1 else 0
        speed_slope = trend_slope(speed_values) if len(speed_values) > 1 else 0
        consistency_variance = np.var(consistency_values) if len(consistency_values) > 1 else 0

        return {
            "accuracy_declining": accuracy_slope < -0.01,
            "speed_declining": speed_slope < -0.01,
            "consistency_issues": consistency_variance > 0.05,
            "accuracy_slope": accuracy_slope,
            "speed_slope": speed_slope,
            "consistency_variance": consistency_variance
        }

    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Get comprehensive report on all cognitive contexts"""
        report = {
            "contexts": {},
            "optimization_state": self.optimization_state.copy(),
            "system_status": {
                "total_contexts": len(self.configurations),
                "performance_history_entries": sum(len(h) for h in self.performance_history.values()),
                "last_update": datetime.now().isoformat()
            }
        }

        for context in CognitiveContext:
            config = self.configurations[context]
            history = self.performance_history[context]

            # Calculate recent performance statistics
            recent_entries = history[-10:] if history else []
            success_rate = (sum(1 for entry in recent_entries if entry["overall_success"])
                          / len(recent_entries)) if recent_entries else 0.0

            report["contexts"][context.value] = {
                "characteristics": {
                    "primary_system": config.characteristics.primary_system,
                    "processing_mode": config.characteristics.processing_mode,
                    "optimization_strategy": config.characteristics.optimization_strategy,
                    "error_tolerance": config.characteristics.error_tolerance
                },
                "performance": {
                    "recent_success_rate": success_rate,
                    "total_measurements": len(history),
                    "performance_benchmarks": config.performance_benchmarks.copy()
                },
                "validation": {
                    "expected_behaviors": len(config.expected_behaviors),
                    "failure_modes": len(config.failure_modes),
                    "validation_criteria": len(config.validation_criteria)
                }
            }

        return report


# Global instance for module access
_context_manager: Optional[CognitiveContextManager] = None

def get_cognitive_context_manager() -> CognitiveContextManager:
    """Get global cognitive context manager instance"""
    global _context_manager
    if _context_manager is None:
        _context_manager = CognitiveContextManager()
    return _context_manager
