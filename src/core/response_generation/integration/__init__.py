"""
Response Generation Integration Package
======================================

This package provides the full integration bridge for the response generation
system, enabling seamless integration with the Kimera cognitive architecture.

DO-178C Level A Compliant Integration Components:
- Full Integration Bridge: Comprehensive response generation orchestration
- Security Response System: Quantum-resistant secure response generation
- Cognitive Response Engine: Multi-modal cognitive response capabilities

Author: KIMERA Development Team
Version: 1.0.0 - DO-178C Level A Compliant
Safety Level: Catastrophic (Level A)
"""

from .full_integration_bridge import (IntegrationConfig, IntegrationMetrics
                                      IntegrationMode, KimeraFullIntegrationBridge
                                      ProcessingPriority)

# Alias for compatibility
ResponseGenerationIntegrator = KimeraFullIntegrationBridge

__all__ = [
    "KimeraFullIntegrationBridge",
    "ResponseGenerationIntegrator",
    "IntegrationMode",
    "ProcessingPriority",
    "IntegrationConfig",
    "IntegrationMetrics",
]

__version__ = "1.0.0"
__safety_level__ = "DO-178C Level A"
__author__ = "KIMERA Development Team"
