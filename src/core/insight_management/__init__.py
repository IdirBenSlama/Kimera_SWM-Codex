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

from .integration import InsightManagementIntegrator
from .information_integration_analyzer import (
    InformationIntegrationAnalyzer,
    ComplexityState,
    ComplexitySignature,
    TransitionEvent
)
from .insight_entropy import (
    validate_insight_entropy_reduction,
    calculate_adaptive_entropy_threshold
)
from .insight_feedback import (
    InsightFeedbackEngine,
    EngagementType,
    EngagementRecord
)
from .insight_lifecycle import (
    update_utility_score,
    manage_insight_lifecycle,
    FeedbackEvent
)

__all__ = [
    'InsightManagementIntegrator',
    'InformationIntegrationAnalyzer',
    'ComplexityState',
    'ComplexitySignature',
    'TransitionEvent',
    'validate_insight_entropy_reduction',
    'calculate_adaptive_entropy_threshold',
    'InsightFeedbackEngine',
    'EngagementType',
    'EngagementRecord',
    'update_utility_score',
    'manage_insight_lifecycle',
    'FeedbackEvent'
]

# Version tracking for DO-178C compliance
__version__ = "4.10.0"
__safety_level__ = "DO-178C Level A"
