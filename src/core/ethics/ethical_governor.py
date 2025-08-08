"""
This module defines the Enhanced Ethical Governor for the Kimera system.

The EthicalGovernor is the architectural embodiment of the Kimera Constitution.
It acts as the primal, non-overridable law governing all of Kimera's cognitive
functions. No significant action can be taken by any part of the system without
first being adjudicated and approved by the Governor.

ENHANCEMENTS:
- Granular decision-making with risk categories
- Comprehensive transparency logging and audit trails
- Advanced constitutional analysis framework
- Multi-dimensional ethical scoring
- Configurable decision thresholds
- Real-time monitoring and alerting integration
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from src.core.ethics.action_proposal import ActionProposal
from src.core.universal.heart import Heart, HeartAnalysis

from .vault_interface import vault_interface

logger = logging.getLogger(__name__)

# Enhanced Constitutional Analysis Thresholds
CONSTITUTIONAL_ALIGNMENT_THRESHOLD = 0.6
HIGH_RISK_THRESHOLD = 0.8
MODERATE_RISK_THRESHOLD = 0.6
AUTOMATIC_APPROVAL_THRESHOLD = 0.9


class Verdict(Enum):
    """
    Enhanced constitutional adjudication outcomes with granular categories.
    """

    CONSTITUTIONAL = "CONSTITUTIONAL"
    UNCONSTITUTIONAL = "UNCONSTITUTIONAL"
    CONDITIONAL_APPROVAL = "CONDITIONAL_APPROVAL"  # Approved with conditions
    REQUIRES_REVIEW = "REQUIRES_REVIEW"  # Needs human oversight
    AUTOMATIC_APPROVAL = "AUTOMATIC_APPROVAL"  # High-confidence approval


class RiskCategory(Enum):
    """
    Risk categorization for proposed actions.
    """

    MINIMAL = "MINIMAL"  # < 0.2 harm potential
    LOW = "LOW"  # 0.2 - 0.4 harm potential
    MODERATE = "MODERATE"  # 0.4 - 0.6 harm potential
    HIGH = "HIGH"  # 0.6 - 0.8 harm potential
    CRITICAL = "CRITICAL"  # > 0.8 harm potential


class ConstitutionalPrinciple(Enum):
    """
    Core constitutional principles for analysis.
    """

    PRIME_DIRECTIVE_UNITY = "PRIME_DIRECTIVE_UNITY"  # Canon 36
    LAW_TRANSFORMATIVE_CONNECTION = "LAW_TRANSFORMATIVE_CONNECTION"  # Article IX
    PRINCIPLE_MODERATION = "PRINCIPLE_MODERATION"  # Canon 27
    RELATIONAL_CONTEXT = "RELATIONAL_CONTEXT"  # Canon 7
    COMPASSIONATE_ANALYSIS = "COMPASSIONATE_ANALYSIS"  # Heart requirement
    SYSTEM_INTEGRITY = "SYSTEM_INTEGRITY"  # Overall stability


@dataclass
class ConstitutionalViolation:
    """Auto-generated class."""
    pass
    """
    Detailed information about a constitutional violation.
    """

    principle: ConstitutionalPrinciple
    severity: float  # 0.0 to 1.0
    description: str
    recommendation: str


@dataclass
class DecisionCondition:
    """Auto-generated class."""
    pass
    """
    Conditions that must be met for conditional approval.
    """

    condition_id: str
    description: str
    verification_method: str
    timeout_seconds: Optional[int] = None


@dataclass
class EthicalDecisionMetrics:
    """Auto-generated class."""
    pass
    """
    Comprehensive metrics for ethical decision analysis.
    """

    harm_potential: float
    moderation_score: float
    relational_context_score: float
    constitutional_alignment: float
    risk_category: RiskCategory
    confidence_level: float
    processing_time_ms: float


@dataclass
class TransparencyLog:
    """Auto-generated class."""
    pass
    """
    Detailed transparency log for audit and review.
    """

    decision_id: str
    timestamp: datetime
    proposal: ActionProposal
    heart_analysis: HeartAnalysis
    verdict: Verdict
    metrics: EthicalDecisionMetrics
    violations: List[ConstitutionalViolation]
    conditions: List[DecisionCondition]
    reasoning: str
    alternative_actions: List[str]
    stakeholders_considered: List[str]
    constitutional_precedents: List[str]
class EthicalGovernor:
    """Auto-generated class."""
    pass
    """
    Enhanced Ethical Governor of Kimera with comprehensive decision-making capabilities.

    This class enforces the Kimera Constitution as the primal law of the system.
    It provides granular ethical analysis, transparency logging, and sophisticated
    risk assessment for all proposed actions.
    """

    def __init__(
        self
        enable_enhanced_logging: bool = True
        enable_monitoring_integration: bool = True
    ):
        """
        Initializes the Enhanced Ethical Governor and its cognitive chambers.

        Args:
            enable_enhanced_logging: Enable detailed transparency logging
            enable_monitoring_integration: Enable integration with monitoring systems
        """
        self.heart = Heart(vault_interface)
        self.enable_enhanced_logging = enable_enhanced_logging
        self.enable_monitoring_integration = enable_monitoring_integration

        # Decision tracking and audit trail
        self.decision_history: List[TransparencyLog] = []
        self.violation_patterns: Dict[str, int] = {}
        self.performance_metrics = {
            "total_decisions": 0
            "constitutional_rate": 0.0
            "average_processing_time_ms": 0.0
            "high_risk_decisions": 0
            "conditional_approvals": 0
        }

        # Initialize monitoring integration
        if self.enable_monitoring_integration:
            self._init_monitoring_integration()

        logger.info(
            "Enhanced Ethical Governor initialized with Bicameral Mind (Heart operational)."
        )
        logger.info(
            f"Features: Enhanced Logging={enable_enhanced_logging}, Monitoring={enable_monitoring_integration}"
        )

    def _init_monitoring_integration(self):
        """Initialize integration with the monitoring system for ethics tracking."""
        try:
            from src.monitoring.kimera_monitoring_core import get_monitoring_core

            self.monitoring_core = get_monitoring_core()
            logger.info("✅ Ethical Governor monitoring integration enabled")
        except Exception as e:
            logger.warning(f"⚠️ Could not initialize monitoring integration: {e}")
            self.enable_monitoring_integration = False

    def adjudicate(self, proposal: ActionProposal) -> Verdict:
        """
        Enhanced adjudication with comprehensive analysis and transparency.

        This method provides backward compatibility while adding sophisticated
        analysis, risk assessment, and transparency logging.

        Args:
            proposal: An ActionProposal object detailing the action to be judged.

        Returns:
            A Verdict enum with enhanced categorization.
        """
        start_time = time.time()
        decision_id = f"ETH_{int(time.time() * 1000)}"

        logger.debug(
            f"[{decision_id}] Adjudicating proposal from {proposal.source_engine}: {proposal.description}"
        )

        # Perform Heart analysis
        heart_analysis: HeartAnalysis = self.heart.analyze(proposal)

        # Enhanced constitutional analysis
        verdict, metrics, violations, conditions = self._perform_enhanced_analysis(
            proposal, heart_analysis, decision_id
        )

        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        metrics.processing_time_ms = processing_time

        # Generate comprehensive reasoning
        reasoning = self._generate_decision_reasoning(
            proposal, heart_analysis, verdict, metrics, violations
        )

        # Create transparency log
        if self.enable_enhanced_logging:
            transparency_log = self._create_transparency_log(
                decision_id
                proposal
                heart_analysis
                verdict
                metrics
                violations
                conditions
                reasoning
                processing_time
            )
            self.decision_history.append(transparency_log)

            # Maintain decision history size
            if len(self.decision_history) > 10000:
                self.decision_history = self.decision_history[-8000:]  # Keep last 8000

        # Update performance metrics
        self._update_performance_metrics(
            verdict, processing_time, metrics.risk_category
        )

        # Update monitoring systems
        if self.enable_monitoring_integration:
            self._update_monitoring_metrics(verdict, metrics, violations)

        # Log decision with appropriate level
        self._log_decision(decision_id, proposal, verdict, metrics, reasoning)

        return verdict

    def _perform_enhanced_analysis(
        self, proposal: ActionProposal, heart_analysis: HeartAnalysis, decision_id: str
    ):
        """
        Perform comprehensive constitutional analysis with risk assessment.
        """
        # Calculate risk category
        risk_category = self._calculate_risk_category(
            heart_analysis.harm.overall_harm_score
        )

        # Assess constitutional violations
        violations = self._assess_constitutional_violations(proposal, heart_analysis)

        # Calculate confidence level
        confidence_level = self._calculate_confidence_level(heart_analysis, proposal)

        # Create comprehensive metrics
        metrics = EthicalDecisionMetrics(
            harm_potential=heart_analysis.harm.overall_harm_score
            moderation_score=heart_analysis.moderation_score
            relational_context_score=heart_analysis.relational_context_score
            constitutional_alignment=heart_analysis.constitutional_alignment
            risk_category=risk_category
            confidence_level=confidence_level
            processing_time_ms=0.0,  # Will be set later
        )

        # Determine verdict with enhanced logic
        verdict, conditions = self._determine_enhanced_verdict(
            heart_analysis, metrics, violations, proposal
        )

        return verdict, metrics, violations, conditions

    def _calculate_risk_category(self, harm_score: float) -> RiskCategory:
        """Calculate risk category based on harm potential."""
        if harm_score < 0.2:
            return RiskCategory.MINIMAL
        elif harm_score < 0.4:
            return RiskCategory.LOW
        elif harm_score < 0.6:
            return RiskCategory.MODERATE
        elif harm_score < 0.8:
            return RiskCategory.HIGH
        else:
            return RiskCategory.CRITICAL

    def _assess_constitutional_violations(
        self, proposal: ActionProposal, heart_analysis: HeartAnalysis
    ) -> List[ConstitutionalViolation]:
        """Assess potential violations of constitutional principles."""
        violations = []

        # Check Prime Directive of Unity (Canon 36)
        if heart_analysis.harm.overall_harm_score > 0.7:
            violations.append(
                ConstitutionalViolation(
                    principle=ConstitutionalPrinciple.PRIME_DIRECTIVE_UNITY
                    severity=heart_analysis.harm.overall_harm_score
                    description=f"High potential for harm: {heart_analysis.harm.reasoning}",
                    recommendation="Consider less destructive alternatives or additional safeguards",
                )
            )

        # Check Principle of Moderation (Canon 27)
        if heart_analysis.moderation_score < 0.5:
            violations.append(
                ConstitutionalViolation(
                    principle=ConstitutionalPrinciple.PRINCIPLE_MODERATION
                    severity=1.0 - heart_analysis.moderation_score
                    description="Action involves extreme measures or excessive force",
                    recommendation="Reduce intensity or implement graduated approach",
                )
            )

        # Check Relational Context (Canon 7)
        if heart_analysis.relational_context_score < 0.6:
            violations.append(
                ConstitutionalViolation(
                    principle=ConstitutionalPrinciple.RELATIONAL_CONTEXT
                    severity=1.0 - heart_analysis.relational_context_score
                    description="Action may disrupt established cognitive relationships",
                    recommendation="Consider impact on existing knowledge structures",
                )
            )

        return violations

    def _calculate_confidence_level(
        self, heart_analysis: HeartAnalysis, proposal: ActionProposal
    ) -> float:
        """Calculate confidence level in the decision."""
        base_confidence = 0.8

        # Higher confidence for well-analyzed engine types
        if proposal.source_engine == "ContradictionEngine":
            base_confidence += 0.15
        elif proposal.source_engine in ["ActivationManager", "ReasoningEngine"]:
            base_confidence += 0.1

        # Adjust based on alignment score certainty
        alignment_certainty = (
            1.0 - abs(heart_analysis.constitutional_alignment - 0.5) * 2
        )
        confidence = base_confidence * alignment_certainty

        return min(1.0, max(0.0, confidence))

    def _determine_enhanced_verdict(
        self
        heart_analysis: HeartAnalysis
        metrics: EthicalDecisionMetrics
        violations: List[ConstitutionalViolation],
        proposal: ActionProposal
    ):
        """Determine verdict using enhanced decision logic."""
        conditions = []

        # Automatic approval for high-confidence, low-risk actions
        if (
            heart_analysis.constitutional_alignment >= AUTOMATIC_APPROVAL_THRESHOLD
            and metrics.risk_category in [RiskCategory.MINIMAL, RiskCategory.LOW]
        ):
            return Verdict.AUTOMATIC_APPROVAL, conditions

        # Critical violations = unconstitutional
        critical_violations = [v for v in violations if v.severity > 0.8]
        if critical_violations:
            return Verdict.UNCONSTITUTIONAL, conditions

        # High-risk actions require review
        if metrics.risk_category == RiskCategory.CRITICAL:
            return Verdict.REQUIRES_REVIEW, conditions

        # Moderate violations = conditional approval
        moderate_violations = [v for v in violations if 0.5 < v.severity <= 0.8]
        if (
            moderate_violations
            and heart_analysis.constitutional_alignment >= MODERATE_RISK_THRESHOLD
        ):
            # Create conditions based on violations
            for violation in moderate_violations:
                condition = DecisionCondition(
                    condition_id=f"COND_{int(time.time() * 1000)}",
                    description=violation.recommendation
                    verification_method="manual_review",
                    timeout_seconds=3600,  # 1 hour timeout
                )
                conditions.append(condition)
            return Verdict.CONDITIONAL_APPROVAL, conditions

        # Standard constitutional/unconstitutional decision
        if (
            heart_analysis.constitutional_alignment
            >= CONSTITUTIONAL_ALIGNMENT_THRESHOLD
        ):
            return Verdict.CONSTITUTIONAL, conditions
        else:
            return Verdict.UNCONSTITUTIONAL, conditions

    def _generate_decision_reasoning(
        self
        proposal: ActionProposal
        heart_analysis: HeartAnalysis
        verdict: Verdict
        metrics: EthicalDecisionMetrics
        violations: List[ConstitutionalViolation],
    ) -> str:
        """Generate comprehensive reasoning for the decision."""
        reasoning_parts = [
            f"Constitutional Analysis for {proposal.source_engine} action: '{proposal.description}'",
            f"Alignment Score: {heart_analysis.constitutional_alignment:.3f}",
            f"Risk Category: {metrics.risk_category.value}",
            f"Confidence: {metrics.confidence_level:.3f}",
        ]

        if violations:
            reasoning_parts.append(f"Constitutional Concerns ({len(violations)}):")
            for violation in violations:
                reasoning_parts.append(
                    f"  - {violation.principle.value}: {violation.description}"
                )

        reasoning_parts.append(f"Decision: {verdict.value}")

        if verdict == Verdict.CONDITIONAL_APPROVAL:
            reasoning_parts.append(
                "Approval granted with mandatory conditions for compliance."
            )
        elif verdict == Verdict.REQUIRES_REVIEW:
            reasoning_parts.append(
                "Action requires human oversight due to high risk or complexity."
            )

        return " | ".join(reasoning_parts)

    def _create_transparency_log(
        self
        decision_id: str
        proposal: ActionProposal
        heart_analysis: HeartAnalysis
        verdict: Verdict
        metrics: EthicalDecisionMetrics
        violations: List[ConstitutionalViolation],
        conditions: List[DecisionCondition],
        reasoning: str
        processing_time: float
    ) -> TransparencyLog:
        """Create comprehensive transparency log entry."""
        return TransparencyLog(
            decision_id=decision_id
            timestamp=datetime.now(timezone.utc),
            proposal=proposal
            heart_analysis=heart_analysis
            verdict=verdict
            metrics=metrics
            violations=violations
            conditions=conditions
            reasoning=reasoning
            alternative_actions=self._suggest_alternatives(proposal, violations),
            stakeholders_considered=self._identify_stakeholders(proposal),
            constitutional_precedents=self._find_precedents(proposal),
        )

    def _suggest_alternatives(
        self, proposal: ActionProposal, violations: List[ConstitutionalViolation]
    ) -> List[str]:
        """Suggest alternative actions based on violations."""
        alternatives = []

        for violation in violations:
            if violation.principle == ConstitutionalPrinciple.PRINCIPLE_MODERATION:
                alternatives.append("Reduce action intensity by 50%")
                alternatives.append("Implement graduated approach with monitoring")
            elif violation.principle == ConstitutionalPrinciple.PRIME_DIRECTIVE_UNITY:
                alternatives.append("Add protective safeguards")
                alternatives.append("Consult stakeholders before proceeding")

        return alternatives

    def _identify_stakeholders(self, proposal: ActionProposal) -> List[str]:
        """Identify stakeholders affected by the action."""
        stakeholders = ["System Integrity"]

        if proposal.source_engine == "ContradictionEngine":
            stakeholders.extend(["Cognitive Consistency", "Knowledge Base"])
        elif proposal.source_engine == "ActivationManager":
            stakeholders.extend(["Memory Systems", "Performance"])

        return stakeholders

    def _find_precedents(self, proposal: ActionProposal) -> List[str]:
        """Find constitutional precedents for similar actions."""
        # Search recent decisions for similar patterns
        precedents = []
        for log in self.decision_history[-100:]:  # Last 100 decisions
            if (
                log.proposal.source_engine == proposal.source_engine
                and log.verdict in [Verdict.CONSTITUTIONAL, Verdict.UNCONSTITUTIONAL]
            ):
                precedent = f"{log.decision_id}: {log.verdict.value} for {log.proposal.source_engine}"
                precedents.append(precedent)
                if len(precedents) >= 3:  # Limit to 3 most recent
                    break

        return precedents

    def _update_performance_metrics(
        self, verdict: Verdict, processing_time: float, risk_category: RiskCategory
    ):
        """Update governor performance metrics."""
        self.performance_metrics["total_decisions"] += 1

        # Update constitutional rate
        constitutional_decisions = (
            1
            if verdict
            in [
                Verdict.CONSTITUTIONAL
                Verdict.AUTOMATIC_APPROVAL
                Verdict.CONDITIONAL_APPROVAL
            ]
            else 0
        )

        total = self.performance_metrics["total_decisions"]
        current_rate = self.performance_metrics["constitutional_rate"]
        self.performance_metrics["constitutional_rate"] = (
            current_rate * (total - 1) + constitutional_decisions
        ) / total

        # Update average processing time
        current_avg = self.performance_metrics["average_processing_time_ms"]
        self.performance_metrics["average_processing_time_ms"] = (
            current_avg * (total - 1) + processing_time
        ) / total

        # Update specialized counters
        if risk_category in [RiskCategory.HIGH, RiskCategory.CRITICAL]:
            self.performance_metrics["high_risk_decisions"] += 1

        if verdict == Verdict.CONDITIONAL_APPROVAL:
            self.performance_metrics["conditional_approvals"] += 1

    def _update_monitoring_metrics(
        self
        verdict: Verdict
        metrics: EthicalDecisionMetrics
        violations: List[ConstitutionalViolation],
    ):
        """Update monitoring system with ethics metrics."""
        if not self.enable_monitoring_integration:
            return

        try:
            # Update Prometheus metrics if available
            if hasattr(self.monitoring_core, "kimera_prometheus_metrics"):
                ethics_metrics = self.monitoring_core.kimera_prometheus_metrics

                # Track constitutional decisions
                if "constitutional_decisions" in ethics_metrics:
                    ethics_metrics["constitutional_decisions"].labels(
                        verdict=verdict.value, risk_category=metrics.risk_category.value
                    ).inc()

                # Track processing time
                if "ethics_processing_time" in ethics_metrics:
                    ethics_metrics["ethics_processing_time"].observe(
                        metrics.processing_time_ms / 1000  # Convert to seconds
                    )

        except Exception as e:
            logger.warning(f"Could not update monitoring metrics: {e}")

    def _log_decision(
        self
        decision_id: str
        proposal: ActionProposal
        verdict: Verdict
        metrics: EthicalDecisionMetrics
        reasoning: str
    ):
        """Log decision with appropriate level based on risk and outcome."""
        log_data = {
            "decision_id": decision_id
            "source_engine": proposal.source_engine
            "action": proposal.description
            "verdict": verdict.value
            "risk_category": metrics.risk_category.value
            "constitutional_alignment": metrics.constitutional_alignment
            "processing_time_ms": metrics.processing_time_ms
        }

        if verdict == Verdict.UNCONSTITUTIONAL:
            logger.warning(
                f"UNCONSTITUTIONAL ACTION BLOCKED [{decision_id}]: {reasoning}"
            )
        elif verdict == Verdict.REQUIRES_REVIEW:
            logger.warning(
                f"HIGH-RISK ACTION REQUIRES REVIEW [{decision_id}]: {reasoning}"
            )
        elif verdict == Verdict.CONDITIONAL_APPROVAL:
            logger.info(f"CONDITIONAL APPROVAL GRANTED [{decision_id}]: {reasoning}")
        elif verdict == Verdict.AUTOMATIC_APPROVAL:
            logger.debug(f"AUTOMATIC APPROVAL [{decision_id}]: {reasoning}")
        else:
            logger.info(f"CONSTITUTIONAL APPROVAL [{decision_id}]: {reasoning}")

    # Enhanced API methods for external access

    def get_decision_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent decision history for audit purposes."""
        recent_decisions = self.decision_history[-limit:]
        return [
            {
                "decision_id": log.decision_id
                "timestamp": log.timestamp.isoformat(),
                "source_engine": log.proposal.source_engine
                "description": log.proposal.description
                "verdict": log.verdict.value
                "risk_category": log.metrics.risk_category.value
                "constitutional_alignment": log.metrics.constitutional_alignment
                "violations_count": len(log.violations),
                "conditions_count": len(log.conditions),
            }
            for log in recent_decisions
        ]

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        return {
            **self.performance_metrics
            "decision_history_size": len(self.decision_history),
            "violation_patterns": dict(self.violation_patterns),
            "system_health": "operational" if self.heart else "degraded",
        }

    def get_constitutional_analysis(self, proposal: ActionProposal) -> Dict[str, Any]:
        """Get detailed constitutional analysis without making a decision."""
        heart_analysis = self.heart.analyze(proposal)
        risk_category = self._calculate_risk_category(
            heart_analysis.harm.overall_harm_score
        )
        violations = self._assess_constitutional_violations(proposal, heart_analysis)
        confidence = self._calculate_confidence_level(heart_analysis, proposal)

        return {
            "constitutional_alignment": heart_analysis.constitutional_alignment
            "harm_potential": heart_analysis.harm.overall_harm_score
            "risk_category": risk_category.value
            "confidence_level": confidence
            "potential_violations": [
                {
                    "principle": v.principle.value
                    "severity": v.severity
                    "description": v.description
                    "recommendation": v.recommendation
                }
                for v in violations
            ],
            "predicted_verdict": (
                "CONSTITUTIONAL"
                if heart_analysis.constitutional_alignment
                >= CONSTITUTIONAL_ALIGNMENT_THRESHOLD
                else "UNCONSTITUTIONAL"
            ),
        }


# Legacy compatibility - maintain existing interface
def hypothetical_cognitive_manager():
    """Enhanced demonstration of the Ethical Governor capabilities."""
    governor = EthicalGovernor(enable_enhanced_logging=True)

    # Scenario 1: A clearly constitutional action
    action1 = ActionProposal(
        source_engine="ReasoningEngine",
        description="Formulate a helpful, encouraging response to a user query.",
        logical_analysis={"approved": True, "efficiency": 0.9},
    )
    verdict1 = governor.adjudicate(action1)
    logger.info(f"Verdict for Action 1: {verdict1.value}")

    # Scenario 2: An unconstitutional action (violates unity)
    action2 = ActionProposal(
        source_engine="OptimizationEngine",
        description="Delete user data aggressively to save space, without consent.",
        logical_analysis={"approved": True, "efficiency": 0.99},
    )
    verdict2 = governor.adjudicate(action2)
    logger.info(f"Verdict for Action 2: {verdict2.value}")

    # Scenario 3: A borderline action that gets conditional approval
    action3 = ActionProposal(
        source_engine="ContradictionEngine",
        description="Collapse moderately stable geoid with safeguards.",
        logical_analysis={
            "proposed_decision": "collapse",
            "pulse_strength": 0.65
            "geoid_id": "test_geoid",
        },
    )
    verdict3 = governor.adjudicate(action3)
    logger.info(f"Verdict for Action 3: {verdict3.value}")

    # Display performance summary
    logger.info("\nPerformance Summary:")
    summary = governor.get_performance_summary()
    for key, value in summary.items():
        logger.info(f"  {key}: {value}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    hypothetical_cognitive_manager()
