"""
Intelligent Pruning Systems Module
=================================

This module implements intelligent pruning of entities from memory,
such as SCARs and Insights, based on thermodynamic and lifecycle criteria.

Key Components:
- IntelligentPruningEngine: Main pruning engine with multiple strategies
- PruningResult: Formal representation of pruning decisions
- PrunableItem: Base class for items that can be pruned
- Multiple pruning strategies with safety assessment

Safety Compliance: DO-178C Level A
"""

from .intelligent_pruning_engine import should_prune  # Legacy compatibility
from .intelligent_pruning_engine import (
    InsightScar,
    IntelligentPruningEngine,
    PrunableItem,
    PruningConfig,
    PruningDecision,
    PruningResult,
    PruningStrategy,
    SafetyStatus,
    Scar,
    create_intelligent_pruning_engine,
)

__all__ = [
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
    "should_prune",
]
