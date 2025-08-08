"""
Quantum-Resistant Cryptography Systems - DO-178C Level A
========================================================

This module provides post-quantum cryptographic protection against
future quantum computing threats to cognitive data.

Author: KIMERA Development Team
Version: 1.0.0 - DO-178C Level A Compliant
"""

from .quantum_resistant_crypto import (CryptographicResult, DilithiumParams
                                       LatticeParams, QuantumResistantCrypto)

__all__ = [
    "QuantumResistantCrypto",
    "LatticeParams",
    "DilithiumParams",
    "CryptographicResult",
]
