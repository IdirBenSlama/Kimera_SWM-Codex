"""
Kimera Pharmaceutical Testing Framework

This module provides comprehensive computational and laboratory protocols for pharmaceutical
development and testing, with initial focus on KCl extended-release capsule development.

Follows USP standards and integrates with Kimera's cognitive fidelity principles.
"""

from .analysis.dissolution_analyzer import DissolutionAnalyzer
from .core.kcl_testing_engine import KClTestingEngine
from .protocols.usp_protocols import USPProtocolEngine
from .validation.pharmaceutical_validator import PharmaceuticalValidator

__all__ = [
    "KClTestingEngine",
    "USPProtocolEngine",
    "DissolutionAnalyzer",
    "PharmaceuticalValidator",
]

__version__ = "1.0.0"
__author__ = "Kimera Pharmaceutical Testing Team"
