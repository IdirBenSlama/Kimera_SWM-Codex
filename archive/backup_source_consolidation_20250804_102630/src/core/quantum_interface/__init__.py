"""
Quantum Interface Package - DO-178C Level A Implementation
=========================================================

This package provides quantum-classical interface capabilities for KIMERA SWM
with full DO-178C Level A safety compliance for safety-critical aerospace
applications.

Package Components:
- QuantumClassicalBridge: Hybrid quantum-classical processing
- QuantumEnhancedUniversalTranslator: Multi-modal semantic translation
- QuantumInterfaceIntegrator: Unified interface orchestration

Author: KIMERA Development Team
Version: 1.0.0 - DO-178C Level A Compliant
Safety Level: Catastrophic (Level A)
"""

# Import main components
from .integration import (
    QuantumInterfaceIntegrator,
    QuantumInterfaceMode,
    create_quantum_interface_integrator
)

from .classical_interface.quantum_classical_bridge import (
    QuantumClassicalBridge,
    HybridProcessingMode,
    HybridProcessingResult,
    InterfaceMetrics,
    create_quantum_classical_bridge
)

from .translation_systems.quantum_enhanced_translator import (
    QuantumEnhancedUniversalTranslator,
    SemanticModality,
    ConsciousnessState,
    QuantumCoherenceState,
    TemporalDynamics,
    UncertaintyPrinciple,
    TranslationResult,
    create_quantum_enhanced_translator
)

# Package metadata
__version__ = "1.0.0"
__safety_level__ = "DO-178C Level A"
__components__ = ["quantum_classical_bridge", "quantum_enhanced_translator", "integrator"]

# Export main interface
__all__ = [
    # Main integrator
    "QuantumInterfaceIntegrator",
    "QuantumInterfaceMode",
    "create_quantum_interface_integrator",

    # Quantum-Classical Bridge
    "QuantumClassicalBridge",
    "HybridProcessingMode",
    "HybridProcessingResult",
    "InterfaceMetrics",
    "create_quantum_classical_bridge",

    # Quantum-Enhanced Translator
    "QuantumEnhancedUniversalTranslator",
    "SemanticModality",
    "ConsciousnessState",
    "QuantumCoherenceState",
    "TemporalDynamics",
    "UncertaintyPrinciple",
    "TranslationResult",
    "create_quantum_enhanced_translator",

    # Package metadata
    "__version__",
    "__safety_level__",
    "__components__"
]

# Package initialization logging
import logging
logger = logging.getLogger(__name__)
logger.info("🌌 DO-178C Level A Quantum Interface Package v1.0.0 loaded")
logger.info(f"   Components available: {__components__}")
logger.info(f"   Safety Level: {__safety_level__}")
