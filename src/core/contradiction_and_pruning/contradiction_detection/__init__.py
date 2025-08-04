"""
Proactive Contradiction Detection Module
=======================================

This module implements proactive scanning for contradictions across all geoids
to increase SCAR utilization and improve semantic memory formation.

Key Components:
- ProactiveContradictionDetector: Main detection engine
- TensionGradient: Formal representation of detected contradictions
- GeoidState: Immutable geoid representation for analysis
- Multiple detection strategies with formal verification

Safety Compliance: DO-178C Level A
"""

from .proactive_contradiction_detector import (
    DetectionStrategy,
    GeoidState,
    HealthStatus,
    ProactiveContradictionDetector,
    ProactiveDetectionConfig,
    TensionGradient,
    create_proactive_contradiction_detector,
    sanitize_for_json,
)

__all__ = [
    "ProactiveContradictionDetector",
    "ProactiveDetectionConfig",
    "TensionGradient",
    "GeoidState",
    "DetectionStrategy",
    "HealthStatus",
    "create_proactive_contradiction_detector",
    "sanitize_for_json",
]
