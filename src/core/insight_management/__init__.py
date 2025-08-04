"""
Insight Management Module
========================

DO-178C Level A compliant module for insight generation, validation, and lifecycle management.

This module integrates:
- Information Integration Analyzer
- Insight Entropy Validator
- Insight Feedback Engine
- Insight Lifecycle Manager

Safety Critical: All insights must be validated before propagation.
"""

from .information_integration_analyzer import (
    ComplexitySignature,
    ComplexityState,
    InformationIntegrationAnalyzer,
    TransitionEvent,
)
from .insight_entropy import (
    calculate_adaptive_entropy_threshold,
    validate_insight_entropy_reduction,
)
from .insight_feedback import EngagementRecord, EngagementType, InsightFeedbackEngine
from .insight_lifecycle import (
    FeedbackEvent,
    manage_insight_lifecycle,
    update_utility_score,
)
from .integration import InsightManagementIntegrator

__all__ = [
    "InsightManagementIntegrator",
    "InformationIntegrationAnalyzer",
    "ComplexityState",
    "ComplexitySignature",
    "TransitionEvent",
    "validate_insight_entropy_reduction",
    "calculate_adaptive_entropy_threshold",
    "InsightFeedbackEngine",
    "EngagementType",
    "EngagementRecord",
    "update_utility_score",
    "manage_insight_lifecycle",
    "FeedbackEvent",
]

# Version tracking for DO-178C compliance
__version__ = "4.10.0"
__safety_level__ = "DO-178C Level A"
