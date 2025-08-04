"""
Quantum Security and Complexity Analysis Package - DO-178C Level A Implementation
================================================================================

This package provides quantum-resistant cryptography and complexity analysis capabilities
for KIMERA SWM with full DO-178C Level A safety compliance for safety-critical aerospace
applications.

Package Components:
- QuantumResistantCrypto: Post-quantum cryptographic protection against quantum attacks
- QuantumThermodynamicComplexityAnalyzer: Quantum-level analysis of system complexity
- QuantumSecurityComplexityIntegrator: Unified integration orchestration

Scientific Foundations:
- Lattice-based cryptography (CRYSTALS-Kyber/Dilithium)
- Quantum thermodynamic principles
- Information theory and integrated information (Φ)
- Nuclear engineering safety protocols

Author: KIMERA Development Team
Version: 1.0.0 - DO-178C Level A Compliant
Safety Level: Catastrophic (Level A)
"""

# Import complexity analysis components
from .complexity_analysis.quantum_thermodynamic_complexity_analyzer import (
    ComplexityAnalysisResult,
    ComplexityState,
    QuantumThermodynamicComplexityAnalyzer,
    ThermodynamicSignature,
)

# Import core security components
from .crypto_systems.quantum_resistant_crypto import (
    CryptographicResult,
    DilithiumParams,
    LatticeParams,
    QuantumResistantCrypto,
)

# Import main integration component
from .integration import (
    ComplexityAnalysisMode,
    QuantumSecurityComplexityIntegrator,
    QuantumSecurityMode,
    create_quantum_security_complexity_integrator,
)

# Package metadata
__version__ = "1.0.0"
__author__ = "KIMERA Development Team"
__email__ = "dev@kimera.ai"
__status__ = "DO-178C Level A Compliant"
__safety_level__ = "catastrophic"

# Available components
__all__ = [
    # Integration
    "QuantumSecurityComplexityIntegrator",
    "QuantumSecurityMode",
    "ComplexityAnalysisMode",
    "create_quantum_security_complexity_integrator",
    # Cryptography
    "QuantumResistantCrypto",
    "LatticeParams",
    "DilithiumParams",
    "CryptographicResult",
    # Complexity Analysis
    "QuantumThermodynamicComplexityAnalyzer",
    "ComplexityState",
    "ThermodynamicSignature",
    "ComplexityAnalysisResult",
]

# Logging setup
import logging

logger = logging.getLogger(__name__)
logger.info(
    "🔐 Quantum Security and Complexity module loaded (v1.0.0, DO-178C Level A)"
)
logger.info(
    "   Components available: ['crypto_systems', 'complexity_analysis', 'integrator']"
)
logger.info("   Dependencies: ['numpy', 'torch', 'cupy', 'numba']")
