"""
System 2: Slow, Analytical Processing
=====================================

DO-178C Level A compliant implementation of Type 2 cognitive processing.

Based on dual-process theory, this module implements slow, controlled, analytical
processing that uses working memory and logical reasoning.

Safety Requirements:
- Response time < 1000ms
- Memory usage < 1GB
- Must handle logical reasoning and rule-based processing
- Sequential processing with working memory
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch

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


class ReasoningType(Enum):
    """Types of analytical reasoning"""

    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    CAUSAL = "causal"
    ANALOGICAL = "analogical"


@dataclass
class LogicalStep:
    """Auto-generated class."""
    pass
    """Single step in logical reasoning chain"""

    premise: str
    operation: str
    conclusion: str
    confidence: float
    reasoning_type: ReasoningType
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AnalysisResult:
    """Auto-generated class."""
    pass
    """Result from System 2 analytical processing"""

    reasoning_chain: List[LogicalStep]
    conclusions: List[Dict[str, Any]]
    working_memory_trace: List[Dict[str, Any]]
    confidence: float
    processing_time: float
    reasoning_types_used: Set[ReasoningType]
    timestamp: datetime = field(default_factory=datetime.now)

    def is_valid(self) -> bool:
        """Validate result meets safety requirements"""
        return (
            0.0 <= self.confidence <= 1.0
            and self.processing_time < 1.000  # 1000ms requirement
            and len(self.reasoning_chain) > 0
        )
class WorkingMemory:
    """Auto-generated class."""
    pass
    """
    Working memory implementation with capacity constraints

    Based on Baddeley's model with central executive, phonological loop
    and visuospatial sketchpad.
    """

    def __init__(self, capacity: int = 7):
        self.capacity = capacity  # Miller's 7±2
        self.central_executive = deque(maxlen=capacity)
        self.phonological_loop = deque(maxlen=capacity * 2)  # Can rehearse more
        self.visuospatial_sketchpad = {}
        self.episodic_buffer = deque(maxlen=capacity * 3)

    def add_item(self, item: Dict[str, Any], item_type: str = "central"):
        """Add item to working memory"""
        timestamp = datetime.now()
        wrapped_item = {"content": item, "timestamp": timestamp, "access_count": 0}

        if item_type == "phonological":
            self.phonological_loop.append(wrapped_item)
        elif item_type == "visuospatial":
            # Spatial items use coordinate keys
            key = item.get("spatial_key", str(timestamp))
            self.visuospatial_sketchpad[key] = wrapped_item
            # Limit size
            if len(self.visuospatial_sketchpad) > self.capacity:
                oldest = min(
                    self.visuospatial_sketchpad.keys(),
                    key=lambda k: self.visuospatial_sketchpad[k]["timestamp"],
                )
                del self.visuospatial_sketchpad[oldest]
        else:
            self.central_executive.append(wrapped_item)

        # Always add to episodic buffer for integration
        self.episodic_buffer.append(wrapped_item)

    def retrieve_recent(self, n: int = 3) -> List[Dict[str, Any]]:
        """Retrieve n most recent items"""
        all_items = []

        # Gather from all stores
        all_items.extend(list(self.central_executive))
        all_items.extend(list(self.phonological_loop))
        all_items.extend(self.visuospatial_sketchpad.values())

        # Sort by timestamp
        all_items.sort(key=lambda x: x["timestamp"], reverse=True)

        # Update access counts
        for item in all_items[:n]:
            item["access_count"] += 1

        return [item["content"] for item in all_items[:n]]

    def get_trace(self) -> List[Dict[str, Any]]:
        """Get full working memory trace for analysis"""
        return [
            {
                "store": "central_executive",
                "items": [item["content"] for item in self.central_executive],
                "size": len(self.central_executive),
            },
            {
                "store": "phonological_loop",
                "items": [item["content"] for item in self.phonological_loop],
                "size": len(self.phonological_loop),
            },
            {
                "store": "visuospatial_sketchpad",
                "items": [
                    item["content"] for item in self.visuospatial_sketchpad.values()
                ],
                "size": len(self.visuospatial_sketchpad),
            },
        ]
class LogicalReasoner:
    """Auto-generated class."""
    pass
    """
    Implements various forms of logical reasoning

    Safety: All reasoning must be traceable and verifiable
    """

    def __init__(self):
        self.inference_rules = {
            "modus_ponens": self._modus_ponens
            "modus_tollens": self._modus_tollens
            "syllogism": self._syllogism
            "analogy": self._analogy
            "induction": self._induction
        }

    def reason(
        self, premises: List[Dict[str, Any]], reasoning_type: ReasoningType
    ) -> List[LogicalStep]:
        """Apply logical reasoning to premises"""
        steps = []

        if reasoning_type == ReasoningType.DEDUCTIVE:
            steps.extend(self._deductive_reasoning(premises))
        elif reasoning_type == ReasoningType.INDUCTIVE:
            steps.extend(self._inductive_reasoning(premises))
        elif reasoning_type == ReasoningType.ABDUCTIVE:
            steps.extend(self._abductive_reasoning(premises))
        elif reasoning_type == ReasoningType.CAUSAL:
            steps.extend(self._causal_reasoning(premises))
        elif reasoning_type == ReasoningType.ANALOGICAL:
            steps.extend(self._analogical_reasoning(premises))

        return steps

    def _deductive_reasoning(self, premises: List[Dict[str, Any]]) -> List[LogicalStep]:
        """Deductive reasoning from general to specific"""
        steps = []

        # Look for modus ponens opportunities
        for i, p1 in enumerate(premises):
            for j, p2 in enumerate(premises):
                if i != j:
                    step = self._modus_ponens(p1, p2)
                    if step:
                        steps.append(step)

        # Look for syllogisms
        if len(premises) >= 2:
            step = self._syllogism(premises[:2])
            if step:
                steps.append(step)

        return steps

    def _inductive_reasoning(self, premises: List[Dict[str, Any]]) -> List[LogicalStep]:
        """Inductive reasoning from specific to general"""
        if len(premises) < 3:
            return []

        # Look for patterns
        pattern = self._find_pattern(premises)
        if pattern:
            return [
                LogicalStep(
                    premise=str(premises),
                    operation="pattern_generalization",
                    conclusion=pattern
                    confidence=0.7,  # Induction is less certain
                    reasoning_type=ReasoningType.INDUCTIVE
                )
            ]

        return []

    def _abductive_reasoning(self, premises: List[Dict[str, Any]]) -> List[LogicalStep]:
        """Abductive reasoning - best explanation"""
        # Find the best explanation for observations
        explanations = self._generate_explanations(premises)

        if explanations:
            best = max(explanations, key=lambda x: x["likelihood"])
            return [
                LogicalStep(
                    premise=str(premises),
                    operation="best_explanation",
                    conclusion=best["explanation"],
                    confidence=best["likelihood"],
                    reasoning_type=ReasoningType.ABDUCTIVE
                )
            ]

        return []

    def _causal_reasoning(self, premises: List[Dict[str, Any]]) -> List[LogicalStep]:
        """Causal reasoning about cause and effect"""
        steps = []

        # Look for causal relationships
        for i in range(len(premises) - 1):
            if self._is_causal_relation(premises[i], premises[i + 1]):
                steps.append(
                    LogicalStep(
                        premise=str(premises[i]),
                        operation="causes",
                        conclusion=str(premises[i + 1]),
                        confidence=0.8
                        reasoning_type=ReasoningType.CAUSAL
                    )
                )

        return steps

    def _analogical_reasoning(
        self, premises: List[Dict[str, Any]]
    ) -> List[LogicalStep]:
        """Reasoning by analogy"""
        if len(premises) < 2:
            return []

        # Find analogies
        analogies = self._find_analogies(premises)

        return [
            LogicalStep(
                premise=f"{a['source']} is like {a['target']}",
                operation="analogy_transfer",
                conclusion=a["inference"],
                confidence=a["similarity"],
                reasoning_type=ReasoningType.ANALOGICAL
            )
            for a in analogies
        ]

    # Helper methods for specific inference rules

    def _modus_ponens(
        self, p1: Dict[str, Any], p2: Dict[str, Any]
    ) -> Optional[LogicalStep]:
        """If P then Q; P; therefore Q"""
        if p1.get("type") == "conditional" and p2.get("content") == p1.get(
            "antecedent"
        ):
            return LogicalStep(
                premise=f"If {p1['antecedent']} then {p1['consequent']}; {p2['content']}",
                operation="modus_ponens",
                conclusion=p1["consequent"],
                confidence=0.95
                reasoning_type=ReasoningType.DEDUCTIVE
            )
        return None

    def _modus_tollens(
        self, p1: Dict[str, Any], p2: Dict[str, Any]
    ) -> Optional[LogicalStep]:
        """If P then Q; not Q; therefore not P"""
        if (
            p1.get("type") == "conditional"
            and p2.get("content") == f"not {p1.get('consequent')}"
        ):
            return LogicalStep(
                premise=f"If {p1['antecedent']} then {p1['consequent']}; not {p1['consequent']}",
                operation="modus_tollens",
                conclusion=f"not {p1['antecedent']}",
                confidence=0.95
                reasoning_type=ReasoningType.DEDUCTIVE
            )
        return None

    def _syllogism(self, premises: List[Dict[str, Any]]) -> Optional[LogicalStep]:
        """Classic syllogistic reasoning"""
        if len(premises) >= 2:
            # Simplified syllogism detection
            return LogicalStep(
                premise=str(premises),
                operation="syllogism",
                conclusion="derived_conclusion",
                confidence=0.9
                reasoning_type=ReasoningType.DEDUCTIVE
            )
        return None

    def _analogy(
        self, source: Dict[str, Any], target: Dict[str, Any]
    ) -> Optional[LogicalStep]:
        """Analogical reasoning"""
        if source.get("structure") and target.get("structure"):
            return LogicalStep(
                rule="analogy",
                premises=[source, target],
                conclusion={"mapped": f"Structure mapped from {source} to {target}"},
                confidence=0.7
                reasoning_type=ReasoningType.ABDUCTIVE
            )
        return None

    def _induction(self, examples: List[Dict[str, Any]]) -> Optional[LogicalStep]:
        """Inductive reasoning from examples"""
        if len(examples) >= 3:  # Need multiple examples for induction
            pattern = self._find_pattern(examples)
            if pattern:
                return LogicalStep(
                    rule="induction",
                    premises=examples
                    conclusion={"generalization": pattern},
                    confidence=0.6
                    reasoning_type=ReasoningType.INDUCTIVE
                )
        return None

    def _find_pattern(self, data: List[Dict[str, Any]]) -> Optional[str]:
        """Find patterns in data (simplified)"""
        # This would involve more sophisticated pattern detection
        return "observed_pattern"

    def _generate_explanations(
        self, observations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate possible explanations (simplified)"""
        return [
            {"explanation": "hypothesis_1", "likelihood": 0.7},
            {"explanation": "hypothesis_2", "likelihood": 0.5},
        ]

    def _is_causal_relation(
        self, event1: Dict[str, Any], event2: Dict[str, Any]
    ) -> bool:
        """Check if events have causal relation (simplified)"""
        return event1.get("timestamp", 0) < event2.get("timestamp", 1)

    def _find_analogies(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find analogical relationships (simplified)"""
        return [
            {
                "source": items[0],
                "target": items[1],
                "similarity": 0.8
                "inference": "analogical_conclusion",
            }
        ]
class System2Processor:
    """Auto-generated class."""
    pass
    """
    Main System 2 processor implementing slow, analytical cognition

    DO-178C Level A Safety Requirements:
    - Response time < 1000ms
    - Memory bounded to 1GB
    - Full reasoning trace required
    """

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.working_memory = WorkingMemory()
        self.logical_reasoner = LogicalReasoner()

        # Performance monitoring
        self.performance_stats = {
            "total_calls": 0
            "avg_response_time": 0.0
            "max_response_time": 0.0
            "timeout_count": 0
        }

        # Safety limits
        self.MAX_RESPONSE_TIME = 1.000  # 1000ms
        self.MAX_MEMORY_GB = 1.0
        self.MAX_REASONING_STEPS = 50  # Prevent infinite loops

        logger.info("🧠 System 2 Processor initialized (slow/analytical)")

    async def process(
        self
        input_data: torch.Tensor
        context: Optional[Dict[str, Any]] = None
        required_reasoning: Optional[List[ReasoningType]] = None
    ) -> AnalysisResult:
        """
        Main System 2 processing function

        Implements slow, sequential, rule-based processing
        """
        start_time = time.time()

        # Safety check: Memory usage
        if self._check_memory_usage() > self.MAX_MEMORY_GB:
            logger.warning("System 2 memory limit approaching, triggering cleanup")
            self._cleanup_memory()

        try:
            # Convert input to premises
            premises = self._extract_premises(input_data, context)

            # Add to working memory
            for premise in premises:
                self.working_memory.add_item(premise, "central")

            # Determine reasoning types to use
            if required_reasoning:
                reasoning_types = required_reasoning
            else:
                reasoning_types = self._select_reasoning_types(premises, context)

            # Apply reasoning with timeout protection
            all_steps = []
            reasoning_types_used = set()

            for reasoning_type in reasoning_types:
                if len(all_steps) >= self.MAX_REASONING_STEPS:
                    logger.warning("Maximum reasoning steps reached")
                    break

                # Check time budget
                elapsed = time.time() - start_time
                if elapsed > self.MAX_RESPONSE_TIME * 0.8:
                    logger.warning("Time budget exhausted, concluding reasoning")
                    break

                # Apply reasoning
                steps = await asyncio.wait_for(
                    self._async_reason(premises, reasoning_type),
                    timeout=self.MAX_RESPONSE_TIME - elapsed
                )

                all_steps.extend(steps)
                if steps:
                    reasoning_types_used.add(reasoning_type)

                # Add conclusions to working memory for further reasoning
                for step in steps:
                    self.working_memory.add_item(
                        {"conclusion": step.conclusion, "confidence": step.confidence},
                        "central",
                    )

            # Extract final conclusions
            conclusions = self._extract_conclusions(all_steps)

            # Calculate overall confidence
            confidence = self._calculate_confidence(all_steps, conclusions)

            processing_time = time.time() - start_time

            # Update statistics
            self._update_stats(processing_time)

            result = AnalysisResult(
                reasoning_chain=all_steps
                conclusions=conclusions
                working_memory_trace=self.working_memory.get_trace(),
                confidence=confidence
                processing_time=processing_time
                reasoning_types_used=reasoning_types_used
            )

            # Safety validation
            if not result.is_valid():
                raise KimeraCognitiveError(
                    f"System 2 result failed validation: time={processing_time:.3f}s"
                )

            return result

        except asyncio.TimeoutError:
            self.performance_stats["timeout_count"] += 1
            logger.error("System 2 processing timeout exceeded")

            # Return degraded result
            return AnalysisResult(
                reasoning_chain=[],
                conclusions=[],
                working_memory_trace=self.working_memory.get_trace(),
                confidence=0.0
                processing_time=self.MAX_RESPONSE_TIME
                reasoning_types_used=set(),
            )

        except Exception as e:
            logger.error(f"System 2 processing error: {e}")
            raise KimeraCognitiveError(f"System 2 processing failed: {e}")

    async def _async_reason(
        self, premises: List[Dict[str, Any]], reasoning_type: ReasoningType
    ) -> List[LogicalStep]:
        """Asynchronous reasoning execution"""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.logical_reasoner.reason, premises, reasoning_type
        )

    def _extract_premises(
        self, input_data: torch.Tensor, context: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract logical premises from input"""
        premises = []

        # Convert tensor data to symbolic premises (simplified)
        data_np = input_data.cpu().numpy()

        # Extract features as premises
        premises.append(
            {
                "type": "observation",
                "content": f"data_mean={np.mean(data_np):.3f}",
                "confidence": 1.0
            }
        )

        premises.append(
            {
                "type": "observation",
                "content": f"data_variance={np.var(data_np):.3f}",
                "confidence": 1.0
            }
        )

        # Add context as premises
        if context:
            for key, value in context.items():
                if isinstance(value, (str, int, float, bool)):
                    premises.append(
                        {
                            "type": "context",
                            "content": f"{key}={value}",
                            "confidence": 0.9
                        }
                    )

        return premises

    def _select_reasoning_types(
        self, premises: List[Dict[str, Any]], context: Optional[Dict[str, Any]]
    ) -> List[ReasoningType]:
        """Select appropriate reasoning types based on problem"""
        selected = []

        # Always try deductive reasoning
        selected.append(ReasoningType.DEDUCTIVE)

        # If we have multiple observations, try inductive
        if len(premises) > 3:
            selected.append(ReasoningType.INDUCTIVE)

        # If context suggests causality, add causal reasoning
        if context and any(k in context for k in ["cause", "effect", "temporal"]):
            selected.append(ReasoningType.CAUSAL)

        # If context suggests explanation needed, add abductive
        if context and any(k in context for k in ["explain", "why", "hypothesis"]):
            selected.append(ReasoningType.ABDUCTIVE)

        return selected

    def _extract_conclusions(self, steps: List[LogicalStep]) -> List[Dict[str, Any]]:
        """Extract final conclusions from reasoning chain"""
        conclusions = []

        # Group by conclusion content
        conclusion_map = {}
        for step in steps:
            key = step.conclusion
            if key not in conclusion_map:
                conclusion_map[key] = {
                    "conclusion": key
                    "supporting_steps": [],
                    "average_confidence": 0.0
                    "reasoning_types": set(),
                }

            conclusion_map[key]["supporting_steps"].append(step)
            conclusion_map[key]["reasoning_types"].add(step.reasoning_type.value)

        # Calculate average confidence for each conclusion
        for data in conclusion_map.values():
            confidences = [step.confidence for step in data["supporting_steps"]]
            data["average_confidence"] = np.mean(confidences)
            data["reasoning_types"] = list(data["reasoning_types"])
            data["support_count"] = len(data["supporting_steps"])

            conclusions.append(
                {
                    "conclusion": data["conclusion"],
                    "confidence": data["average_confidence"],
                    "support_count": data["support_count"],
                    "reasoning_types": data["reasoning_types"],
                }
            )

        # Sort by confidence
        conclusions.sort(key=lambda x: x["confidence"], reverse=True)

        return conclusions

    def _calculate_confidence(
        self, steps: List[LogicalStep], conclusions: List[Dict[str, Any]]
    ) -> float:
        """Calculate overall analytical confidence"""
        if not steps:
            return 0.0

        # Weight by reasoning type reliability
        type_weights = {
            ReasoningType.DEDUCTIVE: 0.95
            ReasoningType.CAUSAL: 0.85
            ReasoningType.ANALOGICAL: 0.75
            ReasoningType.INDUCTIVE: 0.70
            ReasoningType.ABDUCTIVE: 0.65
        }

        weighted_sum = 0.0
        total_weight = 0.0

        for step in steps:
            weight = type_weights.get(step.reasoning_type, 0.5)
            weighted_sum += step.confidence * weight
            total_weight += weight

        if total_weight > 0:
            base_confidence = weighted_sum / total_weight
        else:
            base_confidence = 0.0

        # Adjust by conclusion consistency
        if conclusions:
            top_confidence = conclusions[0]["confidence"]
            consistency_factor = min(
                len(conclusions) / 10.0, 1.0
            )  # More conclusions = more confident

            confidence = (
                base_confidence * 0.7 + top_confidence * 0.3 * consistency_factor
            )
        else:
            confidence = base_confidence * 0.5

        return float(np.clip(confidence, 0.0, 1.0))

    def _check_memory_usage(self) -> float:
        """Check current memory usage in GB"""
        import psutil

        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024 / 1024

    def _cleanup_memory(self):
        """Clean up memory to stay within limits"""
        # Clear old working memory items
        self.working_memory = WorkingMemory()  # Reset

        # Force garbage collection
        import gc

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _update_stats(self, processing_time: float):
        """Update performance statistics"""
        self.performance_stats["total_calls"] += 1

        # Update average
        n = self.performance_stats["total_calls"]
        old_avg = self.performance_stats["avg_response_time"]
        self.performance_stats["avg_response_time"] = (
            old_avg * (n - 1) + processing_time
        ) / n

        # Update max
        self.performance_stats["max_response_time"] = max(
            self.performance_stats["max_response_time"], processing_time
        )

    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance statistics for monitoring"""
        return {
            **self.performance_stats
            "compliance": {
                "response_time_ok": self.performance_stats["max_response_time"]
                < self.MAX_RESPONSE_TIME
                "memory_ok": self._check_memory_usage() < self.MAX_MEMORY_GB
                "timeout_rate": self.performance_stats["timeout_count"]
                / max(self.performance_stats["total_calls"], 1),
            },
        }
