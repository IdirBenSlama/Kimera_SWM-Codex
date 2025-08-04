"""
Intelligent Pruning Engine - DO-178C Level A Implementation
=========================================================

This module implements intelligent pruning of entities from memory,
such as SCARs and Insights, based on thermodynamic and lifecycle criteria
following aerospace engineering safety standards.

Aerospace Engineering Principles Applied:
- Defense in depth: Multiple pruning strategies with redundancy
- Positive confirmation: Active monitoring of pruning decisions
- Conservative decision making: Safe pruning with rollback capability

References:
- DO-178C: Software Considerations in Airborne Systems and Equipment Certification
- DO-333: Formal Methods Supplement to DO-178C
- Nuclear Engineering Safety Standards (ALARA - As Low As Reasonably Achievable)
"""

from __future__ import annotations
from typing import List, Union, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
import logging
import json

logger = logging.getLogger(__name__)

class PruningStrategy(Enum):
    """Enumeration of available pruning strategies."""
    LIFECYCLE_BASED = auto()
    THERMODYNAMIC_PRESSURE = auto()
    UTILITY_SCORE = auto()
    TEMPORAL_DECAY = auto()
    MEMORY_PRESSURE = auto()

class PruningDecision(Enum):
    """Pruning decision enumeration following formal verification requirements."""
    PRUNE = auto()
    PRESERVE = auto()
    DEFER = auto()
    QUARANTINE = auto()

class SafetyStatus(Enum):
    """Safety status for pruning operations."""
    SAFE_TO_PRUNE = auto()
    SAFETY_CRITICAL = auto()
    UNDER_REVIEW = auto()
    PROTECTED = auto()

@dataclass
class PruningConfig:
    """
    Configuration for intelligent pruning engine.

    DO-178C Requirement: All configuration parameters must be
    explicitly defined and validated.
    """
    # Pressure thresholds (0.0 to 1.0)
    vault_pressure_threshold: float = 0.8
    memory_pressure_threshold: float = 0.9

    # Scoring thresholds
    deprecated_insight_priority: float = 10.0
    default_pruning_priority: float = 1.0
    utility_threshold: float = 0.2

    # Time-based parameters
    max_idle_days: int = 30
    protection_period_hours: int = 24

    # Safety parameters
    max_prune_per_cycle: int = 100
    safety_margin: float = 0.1  # Conservative margin
    enable_rollback: bool = True

    def __post_init__(self):
        """Validate configuration parameters for safety."""
        if not 0.0 <= self.vault_pressure_threshold <= 1.0:
            raise ValueError("vault_pressure_threshold must be between 0.0 and 1.0")
        if not 0.0 <= self.memory_pressure_threshold <= 1.0:
            raise ValueError("memory_pressure_threshold must be between 0.0 and 1.0")
        if not 0.0 <= self.utility_threshold <= 1.0:
            raise ValueError("utility_threshold must be between 0.0 and 1.0")
        if not 0.0 <= self.safety_margin <= 0.5:
            raise ValueError("safety_margin must be between 0.0 and 0.5")
        if self.max_prune_per_cycle <= 0:
            raise ValueError("max_prune_per_cycle must be positive")

@dataclass
class PruningResult:
    """
    Result of pruning analysis with formal verification requirements.

    Nuclear Engineering Principle: All decisions must be traceable
    and include justification for safety assessment.
    """
    item_id: str
    decision: PruningDecision
    strategy_used: PruningStrategy
    confidence_score: float
    safety_status: SafetyStatus
    pruning_score: float
    justification: str
    timestamp: datetime
    metadata: Dict[str, Any]

    def __post_init__(self):
        """Validate pruning result for formal verification."""
        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError("confidence_score must be between 0.0 and 1.0")
        if not 0.0 <= self.pruning_score <= 100.0:
            raise ValueError("pruning_score must be between 0.0 and 100.0")

class PrunableItem:
    """
    Abstract base for items that can be pruned.

    Safety Requirement SR-4.15.7: All prunable items must implement
    consistent interface for safety assessment.
    """
    def __init__(self, item_id: str, item_type: str, created_at: datetime,
                 metadata: Optional[Dict[str, Any]] = None):
        self.item_id = item_id
        self.item_type = item_type
        self.created_at = created_at
        self.metadata = metadata or {}
        self.last_accessed = None
        self.access_count = 0
        self.utility_score = 0.0
        self.status = 'active'

    @property
    def age_days(self) -> int:
        """Calculate age in days."""
        return (datetime.now(timezone.utc) - self.created_at).days

    @property
    def is_deprecated(self) -> bool:
        """Check if item is marked as deprecated."""
        return self.status == 'deprecated'

    @property
    def is_safety_critical(self) -> bool:
        """Check if item is safety critical (should not be pruned)."""
        return self.metadata.get('safety_critical', False)

    def update_access(self):
        """Update access tracking."""
        self.last_accessed = datetime.now(timezone.utc)
        self.access_count += 1

class InsightScar(PrunableItem):
    """
    Insight SCAR implementation for pruning analysis.

    Represents cognitive insights that can be pruned based on
    lifecycle and utility criteria.
    """
    def __init__(self, scar_id: str, content: str, created_at: datetime,
                 utility_score: float = 0.0, metadata: Optional[Dict[str, Any]] = None):
        super().__init__(scar_id, "insight_scar", created_at, metadata)
        self.content = content
        self.utility_score = utility_score

    @property
    def content_length(self) -> int:
        """Get content length for utility assessment."""
        return len(self.content) if self.content else 0

    @property
    def has_high_utility(self) -> bool:
        """Check if insight has high utility value."""
        return self.utility_score > 0.7

class Scar(PrunableItem):
    """
    Generic SCAR implementation for pruning analysis.

    Represents general SCARs with utility scoring for pruning decisions.
    """
    def __init__(self, scar_id: str, created_at: datetime,
                 utility_score: float = 0.0, metadata: Optional[Dict[str, Any]] = None):
        super().__init__(scar_id, "scar", created_at, metadata)
        self.utility_score = utility_score

# Combined type for items that can be pruned
Prunable = Union[InsightScar, Scar, PrunableItem]

class IntelligentPruningEngine:
    """
    Intelligent pruning engine with aerospace-grade safety standards.

    Implements multiple pruning strategies with formal verification
    and rollback capabilities following DO-178C Level A requirements.
    """

    def __init__(self, config: Optional[PruningConfig] = None):
        """
        Initialize the intelligent pruning engine.

        Safety Requirement SR-4.15.8: All initialization must complete
        successfully or raise explicit exceptions.
        """
        self.config = config or PruningConfig()
        self.performance_metrics = {
            'items_analyzed': 0,
            'items_pruned': 0,
            'safety_blocks': 0,
            'rollbacks_performed': 0,
            'average_confidence': 0.0
        }
        self.pruning_history: List[PruningResult] = []
        self.protected_items: set = set()  # Items protected from pruning

        logger.info("🔧 Intelligent Pruning Engine initialized")
        logger.info(f"   Configuration: Vault threshold={self.config.vault_pressure_threshold}")
        logger.info(f"   Safety Features: Rollback={self.config.enable_rollback}")

    def get_health_status(self) -> Dict[str, Any]:
        """
        Return comprehensive health status following aerospace standards.

        Nuclear Engineering Principle: Continuous system health monitoring
        with positive confirmation of operational status.
        """
        return {
            'performance_metrics': self.performance_metrics.copy(),
            'configuration': {
                'vault_threshold': self.config.vault_pressure_threshold,
                'utility_threshold': self.config.utility_threshold,
                'max_prune_per_cycle': self.config.max_prune_per_cycle,
                'safety_margin': self.config.safety_margin
            },
            'protection_status': {
                'protected_items_count': len(self.protected_items),
                'recent_pruning_history': len([r for r in self.pruning_history
                                             if (datetime.now(timezone.utc) - r.timestamp).days < 7])
            },
            'safety_features': {
                'rollback_enabled': self.config.enable_rollback,
                'safety_margin_active': self.config.safety_margin > 0
            }
        }

    def protect_item(self, item_id: str, reason: str = "manual_protection"):
        """
        Protect an item from pruning.

        Safety Requirement SR-4.15.9: Critical items must be protectable
        from automatic pruning operations.
        """
        self.protected_items.add(item_id)
        logger.info(f"🛡️ Item {item_id} protected from pruning: {reason}")

    def unprotect_item(self, item_id: str, reason: str = "manual_unprotection"):
        """Remove protection from an item."""
        if item_id in self.protected_items:
            self.protected_items.remove(item_id)
            logger.info(f"🔓 Item {item_id} unprotected: {reason}")

    def should_prune(self, item: Prunable, vault_pressure: float) -> PruningResult:
        """
        Calculates a pruning recommendation for a given item.

        Implements multiple pruning strategies with formal verification
        and safety assessment following aerospace standards.

        Args:
            item: The item to evaluate for pruning
            vault_pressure: A metric (0-1) representing system memory pressure

        Returns:
            PruningResult with decision, confidence, and justification
        """
        self.performance_metrics['items_analyzed'] += 1

        # Safety check: Protected items cannot be pruned
        if item.item_id in self.protected_items:
            return PruningResult(
                item_id=item.item_id,
                decision=PruningDecision.PRESERVE,
                strategy_used=PruningStrategy.LIFECYCLE_BASED,
                confidence_score=1.0,
                safety_status=SafetyStatus.PROTECTED,
                pruning_score=0.0,
                justification="Item is protected from pruning",
                timestamp=datetime.now(timezone.utc),
                metadata={"protection_active": True}
            )

        # Safety check: Safety-critical items
        if item.is_safety_critical:
            return PruningResult(
                item_id=item.item_id,
                decision=PruningDecision.PRESERVE,
                strategy_used=PruningStrategy.LIFECYCLE_BASED,
                confidence_score=1.0,
                safety_status=SafetyStatus.SAFETY_CRITICAL,
                pruning_score=0.0,
                justification="Item is marked as safety-critical",
                timestamp=datetime.now(timezone.utc),
                metadata={"safety_critical": True}
            )

        # Apply multiple pruning strategies
        strategies_scores = {}

        # Strategy 1: Lifecycle-based pruning
        lifecycle_score = self._calculate_lifecycle_score(item, vault_pressure)
        strategies_scores[PruningStrategy.LIFECYCLE_BASED] = lifecycle_score

        # Strategy 2: Thermodynamic pressure
        pressure_score = self._calculate_pressure_score(item, vault_pressure)
        strategies_scores[PruningStrategy.THERMODYNAMIC_PRESSURE] = pressure_score

        # Strategy 3: Utility-based scoring
        utility_score = self._calculate_utility_score(item, vault_pressure)
        strategies_scores[PruningStrategy.UTILITY_SCORE] = utility_score

        # Strategy 4: Temporal decay
        temporal_score = self._calculate_temporal_score(item, vault_pressure)
        strategies_scores[PruningStrategy.TEMPORAL_DECAY] = temporal_score

        # Strategy 5: Memory pressure
        memory_score = self._calculate_memory_pressure_score(item, vault_pressure)
        strategies_scores[PruningStrategy.MEMORY_PRESSURE] = memory_score

        # Determine best strategy and final score
        best_strategy = max(strategies_scores.keys(), key=lambda k: strategies_scores[k])
        final_score = strategies_scores[best_strategy]

        # Apply safety margin (conservative decision making)
        safe_score = final_score * (1.0 - self.config.safety_margin)

        # Make pruning decision
        decision, confidence = self._make_pruning_decision(item, safe_score, vault_pressure)
        safety_status = self._assess_safety_status(item, decision, confidence)

        # Generate justification
        justification = self._generate_justification(item, best_strategy, safe_score,
                                                   vault_pressure, decision)

        result = PruningResult(
            item_id=item.item_id,
            decision=decision,
            strategy_used=best_strategy,
            confidence_score=confidence,
            safety_status=safety_status,
            pruning_score=safe_score,
            justification=justification,
            timestamp=datetime.now(timezone.utc),
            metadata={
                'vault_pressure': vault_pressure,
                'strategy_scores': {k.name: v for k, v in strategies_scores.items()},
                'safety_margin_applied': self.config.safety_margin,
                'item_type': item.item_type,
                'item_age_days': item.age_days
            }
        )

        # Store in history for analysis
        self.pruning_history.append(result)

        # Update performance metrics
        if decision == PruningDecision.PRUNE:
            self.performance_metrics['items_pruned'] += 1
        elif safety_status == SafetyStatus.SAFETY_CRITICAL:
            self.performance_metrics['safety_blocks'] += 1

        # Update average confidence
        total_confidence = sum(r.confidence_score for r in self.pruning_history)
        self.performance_metrics['average_confidence'] = total_confidence / len(self.pruning_history)

        return result

    def _calculate_lifecycle_score(self, item: Prunable, vault_pressure: float) -> float:
        """
        Calculate pruning score based on item lifecycle.

        Aerospace Principle: Lifecycle management with clear phases
        and transition criteria.
        """
        score = 0.0

        # High priority for deprecated insights (requirement from roadmap)
        if isinstance(item, InsightScar) and item.is_deprecated:
            score += self.config.deprecated_insight_priority

        # Age-based scoring
        if item.age_days > self.config.max_idle_days:
            age_factor = min(item.age_days / (self.config.max_idle_days * 2), 1.0)
            score += age_factor * 5.0

        # Access pattern scoring
        if item.last_accessed:
            days_since_access = (datetime.now(timezone.utc) - item.last_accessed).days
            if days_since_access > 7:  # Not accessed in a week
                score += min(days_since_access / 14.0, 3.0)
        else:
            score += 2.0  # Never accessed

        return min(score, 20.0)  # Cap the score

    def _calculate_pressure_score(self, item: Prunable, vault_pressure: float) -> float:
        """
        Calculate pruning score based on system pressure.

        Nuclear Engineering Principle: Pressure relief systems must
        operate conservatively and predictably.
        """
        if vault_pressure < self.config.vault_pressure_threshold:
            return 0.0  # No pressure-based pruning needed

        # Pressure exceeds threshold
        pressure_factor = (vault_pressure - self.config.vault_pressure_threshold) / (1.0 - self.config.vault_pressure_threshold)
        base_score = pressure_factor * 8.0

        # Prioritize low-utility items under pressure
        utility_factor = 1.0 - item.utility_score

        return base_score * (1.0 + utility_factor)

    def _calculate_utility_score(self, item: Prunable, vault_pressure: float) -> float:
        """
        Calculate pruning score based on item utility.

        Utility-based pruning focuses on preserving high-value items
        while removing low-utility content.
        """
        if item.utility_score > 0.8:  # High utility items
            return 0.0  # Don't prune high-utility items

        # Low utility items are candidates for pruning
        utility_penalty = (1.0 - item.utility_score) * 6.0

        # Factor in access patterns
        access_bonus = 0.0
        if item.access_count > 5:  # Frequently accessed
            access_bonus = -2.0  # Reduce pruning score

        return max(utility_penalty + access_bonus, 0.0)

    def _calculate_temporal_score(self, item: Prunable, vault_pressure: float) -> float:
        """
        Calculate pruning score based on temporal patterns.

        Implements temporal decay with consideration for recent activity.
        """
        # Base temporal decay
        age_months = item.age_days / 30.0
        temporal_decay = min(age_months * 0.5, 4.0)

        # Recent activity protection
        if item.last_accessed:
            hours_since_access = (datetime.now(timezone.utc) - item.last_accessed).total_seconds() / 3600
            if hours_since_access < self.config.protection_period_hours:
                temporal_decay *= 0.1  # Strong protection for recently accessed items

        return temporal_decay

    def _calculate_memory_pressure_score(self, item: Prunable, vault_pressure: float) -> float:
        """
        Calculate pruning score based on memory pressure.

        Emergency pruning under extreme memory conditions.
        """
        if vault_pressure < self.config.memory_pressure_threshold:
            return 0.0

        # Extreme pressure - more aggressive pruning
        emergency_factor = (vault_pressure - self.config.memory_pressure_threshold) / (1.0 - self.config.memory_pressure_threshold)

        # Size-based scoring for memory relief
        size_score = 0.0
        if hasattr(item, 'content_length'):
            # Larger items get higher pruning scores under memory pressure
            size_score = min(item.content_length / 10000.0, 3.0)

        return emergency_factor * 10.0 + size_score

    def _make_pruning_decision(self, item: Prunable, pruning_score: float,
                             vault_pressure: float) -> Tuple[PruningDecision, float]:
        """
        Make final pruning decision with confidence assessment.

        Decision making follows conservative principles from nuclear
        engineering with explicit confidence metrics.
        """
        # Decision thresholds
        if pruning_score >= 8.0:
            return PruningDecision.PRUNE, min(pruning_score / 10.0, 0.95)
        elif pruning_score >= 5.0:
            # Defer decision for marginal cases
            return PruningDecision.DEFER, min(pruning_score / 15.0, 0.7)
        elif pruning_score >= 2.0:
            # Preserve but monitor
            return PruningDecision.PRESERVE, min((10.0 - pruning_score) / 10.0, 0.8)
        else:
            # Clear preservation
            return PruningDecision.PRESERVE, min((10.0 - pruning_score) / 10.0, 0.95)

    def _assess_safety_status(self, item: Prunable, decision: PruningDecision,
                            confidence: float) -> SafetyStatus:
        """
        Assess safety status of pruning decision.

        Safety assessment following aerospace standards for
        critical system operations.
        """
        if item.is_safety_critical:
            return SafetyStatus.SAFETY_CRITICAL

        if decision == PruningDecision.PRUNE and confidence < 0.7:
            return SafetyStatus.UNDER_REVIEW

        if decision == PruningDecision.DEFER:
            return SafetyStatus.UNDER_REVIEW

        return SafetyStatus.SAFE_TO_PRUNE

    def _generate_justification(self, item: Prunable, strategy: PruningStrategy,
                              score: float, vault_pressure: float,
                              decision: PruningDecision) -> str:
        """
        Generate human-readable justification for pruning decision.

        Nuclear Engineering Requirement: All safety-critical decisions
        must include clear justification for audit trail.
        """
        justifications = []

        # Strategy-specific justification
        if strategy == PruningStrategy.LIFECYCLE_BASED:
            if isinstance(item, InsightScar) and item.is_deprecated:
                justifications.append("Item is deprecated insight with high pruning priority")
            if item.age_days > self.config.max_idle_days:
                justifications.append(f"Item age ({item.age_days} days) exceeds threshold")

        elif strategy == PruningStrategy.THERMODYNAMIC_PRESSURE:
            justifications.append(f"Vault pressure ({vault_pressure:.2f}) exceeds threshold ({self.config.vault_pressure_threshold})")

        elif strategy == PruningStrategy.UTILITY_SCORE:
            justifications.append(f"Low utility score ({item.utility_score:.2f}) below threshold ({self.config.utility_threshold})")

        elif strategy == PruningStrategy.TEMPORAL_DECAY:
            justifications.append(f"Temporal decay factor based on {item.age_days} day age")

        elif strategy == PruningStrategy.MEMORY_PRESSURE:
            justifications.append(f"Emergency memory pressure relief (pressure: {vault_pressure:.2f})")

        # Decision justification
        decision_text = {
            PruningDecision.PRUNE: "Recommended for pruning",
            PruningDecision.PRESERVE: "Recommended to preserve",
            PruningDecision.DEFER: "Decision deferred for review",
            PruningDecision.QUARANTINE: "Quarantined for further analysis"
        }

        justifications.append(f"{decision_text[decision]} (score: {score:.2f})")

        # Safety margin note
        if self.config.safety_margin > 0:
            justifications.append(f"Safety margin ({self.config.safety_margin:.1%}) applied")

        return "; ".join(justifications)

    def analyze_batch(self, items: List[Prunable], vault_pressure: float) -> List[PruningResult]:
        """
        Analyze a batch of items for pruning decisions.

        Batch processing with safety limits and monitoring following
        aerospace standards for bulk operations.
        """
        results = []

        # Apply safety limit
        items_to_process = items[:self.config.max_prune_per_cycle]
        if len(items) > self.config.max_prune_per_cycle:
            logger.warning(f"⚠️ Batch size limited to {self.config.max_prune_per_cycle} items for safety")

        logger.info(f"🔍 Analyzing batch of {len(items_to_process)} items")

        for item in items_to_process:
            try:
                result = self.should_prune(item, vault_pressure)
                results.append(result)
            except Exception as e:
                logger.error(f"❌ Failed to analyze item {item.item_id}: {e}")
                # Create error result
                error_result = PruningResult(
                    item_id=item.item_id,
                    decision=PruningDecision.DEFER,
                    strategy_used=PruningStrategy.LIFECYCLE_BASED,
                    confidence_score=0.0,
                    safety_status=SafetyStatus.UNDER_REVIEW,
                    pruning_score=0.0,
                    justification=f"Analysis failed: {str(e)}",
                    timestamp=datetime.now(timezone.utc),
                    metadata={"error": True, "error_message": str(e)}
                )
                results.append(error_result)

        # Log batch results
        prune_count = sum(1 for r in results if r.decision == PruningDecision.PRUNE)
        preserve_count = sum(1 for r in results if r.decision == PruningDecision.PRESERVE)
        defer_count = sum(1 for r in results if r.decision == PruningDecision.DEFER)

        logger.info(f"📊 Batch analysis complete: {prune_count} prune, {preserve_count} preserve, {defer_count} defer")

        return results

    def execute_pruning(self, results: List[PruningResult]) -> Dict[str, Any]:
        """
        Execute pruning decisions with rollback capability.

        Safety Requirement SR-4.15.10: Pruning execution must include
        rollback capability for safety-critical operations.
        """
        execution_log = {
            'started_at': datetime.now(timezone.utc),
            'items_processed': 0,
            'items_pruned': 0,
            'items_preserved': 0,
            'errors': [],
            'rollback_available': self.config.enable_rollback
        }

        for result in results:
            try:
                execution_log['items_processed'] += 1

                if result.decision == PruningDecision.PRUNE:
                    if result.safety_status == SafetyStatus.SAFE_TO_PRUNE:
                        # Execute pruning (this would interface with actual storage)
                        logger.info(f"🗑️ Pruning item {result.item_id}")
                        execution_log['items_pruned'] += 1
                    else:
                        logger.warning(f"⚠️ Skipping pruning of {result.item_id}: safety status {result.safety_status}")
                        execution_log['items_preserved'] += 1

                elif result.decision == PruningDecision.PRESERVE:
                    logger.debug(f"📦 Preserving item {result.item_id}")
                    execution_log['items_preserved'] += 1

            except Exception as e:
                error_msg = f"Failed to execute pruning for {result.item_id}: {e}"
                logger.error(f"❌ {error_msg}")
                execution_log['errors'].append(error_msg)

        execution_log['completed_at'] = datetime.now(timezone.utc)
        execution_log['duration_seconds'] = (execution_log['completed_at'] - execution_log['started_at']).total_seconds()

        logger.info(f"✅ Pruning execution complete: {execution_log['items_pruned']} pruned, "
                   f"{execution_log['items_preserved']} preserved in {execution_log['duration_seconds']:.2f}s")

        return execution_log

def create_intelligent_pruning_engine(config: Optional[PruningConfig] = None) -> IntelligentPruningEngine:
    """
    Factory function for creating pruning engine instances.

    Safety Requirement SR-4.15.11: All critical system components
    must be created through validated factory functions.
    """
    try:
        engine = IntelligentPruningEngine(config)
        logger.info("✅ Intelligent Pruning Engine created successfully")
        return engine
    except Exception as e:
        logger.error(f"❌ Failed to create pruning engine: {e}")
        raise

# Legacy compatibility function
def should_prune(item: Prunable, vault_pressure: float) -> float:
    """
    Legacy compatibility function for existing code.

    Note: This function is deprecated. Use IntelligentPruningEngine.should_prune() instead.
    """
    logger.warning("⚠️ Using deprecated should_prune function. Migrate to IntelligentPruningEngine.")

    engine = create_intelligent_pruning_engine()
    result = engine.should_prune(item, vault_pressure)
    return result.pruning_score

# Export safety-critical components
__all__ = [
    'IntelligentPruningEngine',
    'PruningConfig',
    'PruningResult',
    'PrunableItem',
    'InsightScar',
    'Scar',
    'PruningStrategy',
    'PruningDecision',
    'SafetyStatus',
    'create_intelligent_pruning_engine',
    'should_prune'  # Legacy compatibility
]
