"""
Core Components of Barenholtz Dual-System Architecture
=====================================================

DO-178C Level A compliant cognitive processing systems.
"""

from .system1 import System1Processor, IntuitionResult
from .system2 import System2Processor, AnalysisResult, ReasoningType
from .metacognitive import (
    MetacognitiveController,
    ArbitrationResult,
    ArbitrationStrategy,
    ProcessingMode,
    MetacognitiveState
)

__all__ = [
    # System 1
    'System1Processor',
    'IntuitionResult',

    # System 2
    'System2Processor',
    'AnalysisResult',
    'ReasoningType',

    # Metacognitive
    'MetacognitiveController',
    'ArbitrationResult',
    'ArbitrationStrategy',
    'ProcessingMode',
    'MetacognitiveState'
]
