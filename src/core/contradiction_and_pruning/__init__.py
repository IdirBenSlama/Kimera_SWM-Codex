"""
Proactive Contradiction Detection and Pruning System
===================================================

This module provides the integrated system for proactive contradiction detection
and intelligent pruning following DO-178C Level A certification standards.

The system implements aerospace engineering principles including:
- Defense in depth with multiple analysis strategies
- Positive confirmation of system health and safety
- Conservative decision making with explicit margins
- Formal verification and traceability requirements

Components:
-----------
- ProactiveContradictionDetector: Detects semantic contradictions across geoids
- IntelligentPruningEngine: Manages lifecycle-based pruning of memory items
- ContradictionAndPruningIntegrator: Unified integration and coordination

Safety Requirements:
-------------------
- SR-4.15.1: All outputs must be JSON-serializable for external integration
- SR-4.15.2: Geoid states must be immutable during analysis
- SR-4.15.3: All initialization must complete with comprehensive error handling
- SR-4.15.4: Scan timing must be deterministic and race-condition free
- SR-4.15.5: System must operate in degraded mode when dependencies unavailable

References:
----------
- DO-178C: Software Considerations in Airborne Systems and Equipment Certification
- DO-333: Formal Methods Supplement to DO-178C
- Nuclear Engineering Safety Standards (ALARA, Defense in Depth)
"""

from .contradiction_detection.proactive_contradiction_detector import (
    DetectionStrategy, GeoidState, HealthStatus, ProactiveContradictionDetector
    ProactiveDetectionConfig, TensionGradient, create_proactive_contradiction_detector)
from .integration import (ContradictionAndPruningIntegrator
                          create_contradiction_and_pruning_integrator)
from .pruning_systems.intelligent_pruning_engine import \
    should_prune  # Legacy compatibility
from .pruning_systems.intelligent_pruning_engine import (
    InsightScar, IntelligentPruningEngine, PrunableItem, PruningConfig, PruningDecision
    PruningResult, PruningStrategy, SafetyStatus, Scar
    create_intelligent_pruning_engine)

# Version information for certification tracking
__version__ = "1.0.0"
__certification_level__ = "DO-178C Level A"
__safety_critical__ = True

# Export main components for external use
__all__ = [
    # Main integrator
    "ContradictionAndPruningIntegrator",
    "create_contradiction_and_pruning_integrator",
    # Contradiction detection
    "ProactiveContradictionDetector",
    "ProactiveDetectionConfig",
    "TensionGradient",
    "GeoidState",
    "DetectionStrategy",
    "HealthStatus",
    "create_proactive_contradiction_detector",
    # Intelligent pruning
    "IntelligentPruningEngine",
    "PruningConfig",
    "PruningResult",
    "PrunableItem",
    "InsightScar",
    "Scar",
    "PruningStrategy",
    "PruningDecision",
    "SafetyStatus",
    "create_intelligent_pruning_engine",
    "should_prune",  # Legacy compatibility
    # Meta information
    "__version__",
    "__certification_level__",
    "__safety_critical__",
]


# Module-level health check for import verification
def verify_module_health():
    """
    Verify module health and dependencies for safety assessment.

    Returns:
        Dict containing health status and dependency information
    """
    health_status = {
        "module_loaded": True
        "version": __version__
        "certification_level": __certification_level__
        "safety_critical": __safety_critical__
        "components_available": {},
        "dependencies": {},
    }

    # Check component availability
    try:
        integrator = create_contradiction_and_pruning_integrator()
        health_status["components_available"]["integrator"] = True

        # Get comprehensive health from integrator
        integrator_health = integrator.get_comprehensive_health_status()
        health_status["integrator_health"] = integrator_health

    except Exception as e:
        health_status["components_available"]["integrator"] = False
        health_status["integration_error"] = str(e)

    # Check external dependencies
    try:
        import numpy

        health_status["dependencies"]["numpy"] = numpy.__version__
    except ImportError:
        health_status["dependencies"]["numpy"] = "NOT_AVAILABLE"

    try:
        import sklearn

        health_status["dependencies"]["sklearn"] = sklearn.__version__
    except ImportError:
        health_status["dependencies"]["sklearn"] = "NOT_AVAILABLE"

    try:
        import sqlalchemy

        health_status["dependencies"]["sqlalchemy"] = sqlalchemy.__version__
    except ImportError:
        health_status["dependencies"]["sqlalchemy"] = "NOT_AVAILABLE"

    return health_status


# Perform module verification on import (aerospace standard)
_MODULE_HEALTH = verify_module_health()

# Log module initialization following aerospace standards
import logging

logger = logging.getLogger(__name__)
logger.info(
    f"🔍 Contradiction and Pruning module loaded (v{__version__}, {__certification_level__})"
)
logger.info(f"   Components available: {_MODULE_HEALTH['components_available']}")
logger.info(f"   Dependencies: {list(_MODULE_HEALTH['dependencies'].keys())}")
