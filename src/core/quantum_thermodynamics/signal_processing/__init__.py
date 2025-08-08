"""
Quantum Thermodynamic Signal Processing - DO-178C Level A
=========================================================

This module provides quantum thermodynamic signal processing capabilities
bridging TCSE framework and quantum cognitive processing.

Author: KIMERA Development Team
Version: 1.0.0 - DO-178C Level A Compliant
"""

from .quantum_thermodynamic_signal_processor import (
    CorrectionResult, QuantumSignalSuperposition, QuantumThermodynamicSignalProcessor
    SignalDecoherenceController)

__all__ = [
    "QuantumThermodynamicSignalProcessor",
    "QuantumSignalSuperposition",
    "SignalDecoherenceController",
    "CorrectionResult",
]
