"""
Barenholtz Dual-System Architecture
===================================

DO-178C Level A compliant implementation of dual-system cognitive processing.

This module implements a sophisticated cognitive architecture based on dual-process
theory, integrating fast intuitive processing (System 1) with slow analytical
processing (System 2) through a metacognitive control layer.

Safety Critical: All components must meet DO-178C Level A requirements.
"""

from .integration.unified_engine import (
    BarenholtzDualSystemIntegrator,
    DualSystemOutput,
    ProcessingConstraints,
    SystemMode
)

from .core.system1 import System1Processor, IntuitionResult
from .core.system2 import System2Processor, AnalysisResult
from .core.metacognitive import MetacognitiveController, ArbitrationResult

from .utils.memory_manager import WorkingMemoryManager
from .utils.conflict_resolver import ConflictResolver

__all__ = [
    # Main integration
    'BarenholtzDualSystemIntegrator',
    'DualSystemOutput',
    'ProcessingConstraints',
    'SystemMode',

    # Core systems
    'System1Processor',
    'System2Processor',
    'MetacognitiveController',

    # Results
    'IntuitionResult',
    'AnalysisResult',
    'ArbitrationResult',

    # Utilities
    'WorkingMemoryManager',
    'ConflictResolver'
]

# Version and compliance information
__version__ = '1.0.0'
__standard__ = 'DO-178C Level A'
__certification__ = 'Pending'
