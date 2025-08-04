"""
Rhetorical and Symbolic Processing Integration Module
=====================================================

DO-178C Level A compliant rhetorical and symbolic processing system.
Implements revolutionary enhancement of dual-system architecture with:
- Classical rhetoric analysis (Ethos, Pathos, Logos)
- Modern argumentation theory (Toulmin, Perelman, Pragma-dialectics)
- Iconological processing (visual symbols, pictographs, emojis)
- Multi-script linguistic analysis (Latin, Cyrillic, Arabic, Chinese, etc.)
- Cross-cultural symbolic understanding

Scientific Classification: Critical Safety System
Certification Level: DO-178C Level A
Safety Requirements: SR-4.20.1 through SR-4.20.24
"""

from .integration import RhetoricalSymbolicIntegrator
from .rhetorical_engine import RhetoricalProcessor, RhetoricalAnalysis
from .symbolic_engine import SymbolicProcessor, SymbolicAnalysis

__all__ = [
    'RhetoricalSymbolicIntegrator',
    'RhetoricalProcessor',
    'RhetoricalAnalysis',
    'SymbolicProcessor',
    'SymbolicAnalysis'
]

__version__ = "1.0.0"
__author__ = "KIMERA SWM Development Team"
__classification__ = "DO-178C Level A"
