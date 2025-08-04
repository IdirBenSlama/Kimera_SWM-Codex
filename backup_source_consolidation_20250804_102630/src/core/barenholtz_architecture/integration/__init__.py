"""
Integration Components for Barenholtz Architecture
==================================================

DO-178C Level A compliant integration modules.
"""

from .unified_engine import (
    BarenholtzDualSystemIntegrator,
    DualSystemOutput,
    ProcessingConstraints,
    SystemMode
)

__all__ = [
    'BarenholtzDualSystemIntegrator',
    'DualSystemOutput',
    'ProcessingConstraints',
    'SystemMode'
]
