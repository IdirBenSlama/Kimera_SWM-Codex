"""
Real-Time Signal Evolution - DO-178C Level A
============================================

This module provides real-time cognitive signal evolution capabilities
with thermal management and GPU batch optimization.

Author: KIMERA Development Team
Version: 1.0.0 - DO-178C Level A Compliant
"""

from .real_time_signal_evolution import (GeoidStreamProcessor
                                         RealTimeSignalEvolutionEngine
                                         SignalEvolutionResult
                                         ThermalBudgetSignalController)

__all__ = [
    "RealTimeSignalEvolutionEngine",
    "ThermalBudgetSignalController",
    "SignalEvolutionResult",
    "GeoidStreamProcessor",
]
