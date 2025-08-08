"""
Core Components of Barenholtz Dual-System Architecture
=====================================================

DO-178C Level A compliant cognitive processing systems.
"""

from .metacognitive import (ArbitrationResult, ArbitrationStrategy,
                            MetacognitiveController, MetacognitiveState, ProcessingMode)
from .system1 import IntuitionResult, System1Processor
from .system2 import AnalysisResult, ReasoningType, System2Processor

__all__ = [
    # System 1
    "System1Processor",
    "IntuitionResult",
    # System 2
    "System2Processor",
    "AnalysisResult",
    "ReasoningType",
    # Metacognitive
    "MetacognitiveController",
    "ArbitrationResult",
    "ArbitrationStrategy",
    "ProcessingMode",
    "MetacognitiveState",
]
