"""
Barenholtz Dual-System Architecture
===================================

DO-178C Level A compliant implementation of dual-system cognitive processing.

This module implements a sophisticated cognitive architecture based on dual-process
theory, integrating fast intuitive processing (System 1) with slow analytical
processing (System 2) through a metacognitive control layer.

Safety Critical: All components must meet DO-178C Level A requirements.
"""

from .core.metacognitive import ArbitrationResult, MetacognitiveController
from .core.system1 import IntuitionResult, System1Processor
from .core.system2 import AnalysisResult, System2Processor
from .integration.unified_engine import (
    BarenholtzDualSystemIntegrator,
    DualSystemOutput,
    ProcessingConstraints,
    SystemMode,
)
from .utils.conflict_resolver import ConflictResolver
from .utils.memory_manager import WorkingMemoryManager

__all__ = [
    # Main integration
    "BarenholtzDualSystemIntegrator",
    "DualSystemOutput",
    "ProcessingConstraints",
    "SystemMode",
    # Core systems
    "System1Processor",
    "System2Processor",
    "MetacognitiveController",
    # Results
    "IntuitionResult",
    "AnalysisResult",
    "ArbitrationResult",
    # Utilities
    "WorkingMemoryManager",
    "ConflictResolver",
]

# Version and compliance information
__version__ = "1.0.0"
__standard__ = "DO-178C Level A"
__certification__ = "Pending"
