"""
Kimera SWM - Foundational Systems Core
=====================================

TIER 0 - Critical Foundational Components
This module contains the core architectural foundations that everything else depends on:

- KCCL Core: Kimera Cognitive Cycle Logic - the heartbeat of cognitive processing
- SPDE Core: Semantic Pressure Diffusion Engine - mathematical physics foundation  
- Barenholtz Core: Dual-system cognitive architecture foundation
- Cognitive Cycle Core: Core cycle management and orchestration

These are the architectural pillars that define what Kimera IS.
All other cognitive capabilities build upon these foundational systems.
"""

from .kccl_core import KCCLCore, CognitiveCycleState
from .spde_core import SPDECore, SemanticDiffusionEngine, AdvancedSPDEEngine
from .barenholtz_core import BarenholtzCore, DualSystemProcessor, AlignmentEngine
from .cognitive_cycle_core import CognitiveCycleCore, CycleOrchestrator

__all__ = [
    'KCCLCore',
    'SPDECore', 
    'BarenholtzCore',
    'CognitiveCycleCore',
    'CognitiveCycleState',
    'SemanticDiffusionEngine',
    'AdvancedSPDEEngine',
    'DualSystemProcessor',
    'AlignmentEngine',
    'CycleOrchestrator'
]

# Version information
__version__ = "1.0.0"
__status__ = "Production"
__architecture_tier__ = "TIER_0_FOUNDATIONAL"