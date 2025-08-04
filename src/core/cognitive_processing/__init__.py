"""
Kimera SWM - Cognitive Processing Core
=====================================

TIER 1 - Core Cognitive Capabilities
This module contains the primary cognitive processing capabilities:

- Understanding Core: Genuine understanding and comprehension engine
- Consciousness Core: Consciousness detection and awareness systems
- Memory Attention Core: Working memory and attention mechanisms

These systems provide the fundamental cognitive capabilities that operate
on the foundational architecture to enable genuine intelligence.
"""

from .consciousness_core import (
    ConsciousnessCore,
    ConsciousnessDetector,
    QuantumConsciousness,
)
from .memory_attention_core import (
    AttentionMechanism,
    MemoryAttentionCore,
    WorkingMemory,
)
from .understanding_core import (
    GenuineUnderstanding,
    SemanticProcessor,
    UnderstandingCore,
)

__all__ = [
    "UnderstandingCore",
    "ConsciousnessCore",
    "MemoryAttentionCore",
    "GenuineUnderstanding",
    "SemanticProcessor",
    "ConsciousnessDetector",
    "QuantumConsciousness",
    "WorkingMemory",
    "AttentionMechanism",
]

# Version information
__version__ = "1.0.0"
__status__ = "Production"
__architecture_tier__ = "TIER_1_COGNITIVE"
