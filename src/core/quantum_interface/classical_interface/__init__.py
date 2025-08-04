"""
Quantum-Classical Interface Module - DO-178C Level A
===================================================

This module contains the quantum-classical bridge implementation for hybrid
processing with aerospace-grade safety compliance.

Author: KIMERA Development Team
Version: 1.0.0 - DO-178C Level A Compliant
"""

from .quantum_classical_bridge import (
    HybridProcessingMode,
    HybridProcessingResult,
    InterfaceMetrics,
    QuantumClassicalBridge,
    create_quantum_classical_bridge,
)

__all__ = [
    "QuantumClassicalBridge",
    "HybridProcessingMode",
    "HybridProcessingResult",
    "InterfaceMetrics",
    "create_quantum_classical_bridge",
]
