"""
Quantum Thermodynamics Package - DO-178C Level A Implementation
==============================================================

This package provides quantum thermodynamic signal processing and truth monitoring
capabilities for KIMERA SWM with full DO-178C Level A safety compliance for
safety-critical aerospace applications.

Package Components:
- QuantumThermodynamicSignalProcessor: Quantum thermodynamic signal processing engine
- QuantumTruthMonitor: Real-time truth state monitoring in quantum superposition
- QuantumThermodynamicsIntegrator: Unified integration orchestration

Scientific Foundations:
- Quantum thermodynamics and signal processing
- Quantum measurement theory and coherence dynamics
- Epistemic validation and truth evolution tracking
- Nuclear engineering safety protocols

Author: KIMERA Development Team
Version: 1.0.0 - DO-178C Level A Compliant
Safety Level: Catastrophic (Level A)
"""

# Import main integration component
from .integration import (
    QuantumThermodynamicsIntegrator,
    SignalProcessingMode,
    TruthMonitoringMode,
    create_quantum_thermodynamics_integrator
)

# Import signal processing components
from .signal_processing.quantum_thermodynamic_signal_processor import (
    QuantumThermodynamicSignalProcessor,
    QuantumSignalSuperposition,
    SignalDecoherenceController,
    CorrectionResult
)

# Import truth monitoring components
from .truth_monitoring.quantum_truth_monitor import (
    QuantumTruthMonitor,
    QuantumTruthState,
    QuantumMeasurement,
    ClaimTruthEvolution,
    TruthMonitoringResult
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
    "QuantumThermodynamicsIntegrator",
    "SignalProcessingMode",
    "TruthMonitoringMode",
    "create_quantum_thermodynamics_integrator",

    # Signal Processing
    "QuantumThermodynamicSignalProcessor",
    "QuantumSignalSuperposition",
    "SignalDecoherenceController",
    "CorrectionResult",

    # Truth Monitoring
    "QuantumTruthMonitor",
    "QuantumTruthState",
    "QuantumMeasurement",
    "ClaimTruthEvolution",
    "TruthMonitoringResult",
]

# Logging setup
import logging
logger = logging.getLogger(__name__)
logger.info("🌡️ Quantum Thermodynamics module loaded (v1.0.0, DO-178C Level A)")
logger.info("   Components available: ['signal_processing', 'truth_monitoring', 'integrator']")
logger.info("   Dependencies: ['numpy', 'torch', 'quantum_cognitive_engine']")
