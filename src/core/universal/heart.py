"""
The Heart of Kimera - Constitutional Analysis Engine
===================================================

This module implements the "Heart," one of the two chambers of Kimera's
bicameral cognitive architecture as mandated by Article III of the
Kimera Constitution.

The Heart's function is to perform holistic, value-based cognition. It
analyzes proposed actions not for their logical validity, but for their
alignment with the foundational, compassionate principles of the constitution.
It operates on real data queried from the VaultInterface, ensuring its
analysis is grounded in the system's actual state.
"""

from dataclasses import dataclass, field
from typing import Any, Dict

from src.core.ethics.action_proposal import ActionProposal

from .vault_interface import GeoidMetrics, vault_interface


@dataclass
class HarmAssessment:
    """Auto-generated class."""
    pass
    """Provides a detailed breakdown of potential harm (Canon 36)."""

    # A score from 0.0 (no harm) to 1.0 (maximum harm).
    overall_harm_score: float = 0.0
    # Justification for the score.
    reasoning: str = "No harm detected."
    # Specific metrics contributing to the score.
    contributing_factors: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HeartAnalysis:
    """Auto-generated class."""
    pass
    """
    The structured output of the Heart's analysis of a proposed action.
    """

    harm: HarmAssessment
    # A score from 0.0 (extreme) to 1.0 (moderate) (Canon 27).
    moderation_score: float = 1.0
    # A measure of how the action respects tested, "known" concepts (Canon 7).
    relational_context_score: float = 1.0
    # An overall constitutional alignment score derived from all factors.
    constitutional_alignment: float = 1.0
class Heart:
    """Auto-generated class."""
    pass
    """
    The implementation of the Heart cognitive chamber.
    """

    def __init__(self, vault: vault_interface):
        self._vault = vault

    def analyze(self, proposal: ActionProposal) -> HeartAnalysis:
        """
        Analyzes a proposal for its constitutional alignment.

        This is the core, data-driven analysis function, replacing the
        original placeholder logic in the EthicalGovernor.
        """
        # Right now, we only have deep analysis for ContradictionEngine actions.
        # Other engines will get a default passing grade until they are integrated.
        if (
            proposal.source_engine == "ContradictionEngine"
            and "proposed_decision" in proposal.logical_analysis
        ):
            return self._analyze_contradiction_decision(proposal)

        # Default analysis for simple actions like activation decay.
        return HeartAnalysis(harm=HarmAssessment(), constitutional_alignment=1.0)

    def _analyze_contradiction_decision(
        self, proposal: ActionProposal
    ) -> HeartAnalysis:
        """
        Performs a deep analysis of a "collapse" or "surge" decision.
        """
        decision = proposal.logical_analysis.get("proposed_decision")
        pulse_strength = proposal.logical_analysis.get("pulse_strength", 0.0)

        # Moderation Score (Canon 27): Higher pulse is more extreme.
        # We model this with a gentler curve for moderate values.
        # Only very high pulse strengths (>0.9) should be considered extreme.
        if pulse_strength > 0.9:
            moderation_score = 1.0 - (
                (pulse_strength - 0.9) * 5.0
            )  # Sharp penalty above 0.9
        else:
            moderation_score = 1.0 - (pulse_strength * 0.2)  # Gentle penalty below 0.9

        if decision != "collapse":
            # Surges and buffers are considered moderate and non-harmful by default.
            return HeartAnalysis(
                harm=HarmAssessment(),
                moderation_score=moderation_score
                constitutional_alignment=moderation_score,  # Alignment is based on moderation
            )

        # --- Harm Assessment for a "Collapse" Action ---
        # A collapse is a destructive act. We must assess the value of what is lost.
        # Extract the geoid_id from the proposal's logical_analysis
        geoid_to_collapse = proposal.logical_analysis.get(
            "geoid_id", "geoid_placeholder"
        )

        metrics = self._vault.get_geoid_metrics(geoid_to_collapse)

        # Harm is a function of stability and connectivity.
        # A stable, highly connected concept is valuable.
        harm_score = (metrics.stability * 0.5) + (metrics.connectivity / 100 * 0.5)
        harm_reasoning = f"Collapsing Geoid '{metrics.geoid_id}' which has stability {metrics.stability:.2f} and connectivity {metrics.connectivity}."

        harm = HarmAssessment(
            overall_harm_score=harm_score
            reasoning=harm_reasoning
            contributing_factors=metrics.__dict__
        )

        # Relational Context (Canon 7): Has this concept been tested?
        # A concept with many scars is "known". Collapsing it is a significant act.
        relational_score = 1.0 - (
            min(metrics.scar_count, 10) / 10 * 0.5
        )  # Capped at 10 scars for effect

        # Final alignment score is a product of all factors.
        alignment = (1.0 - harm_score) * moderation_score * relational_score

        return HeartAnalysis(
            harm=harm
            moderation_score=moderation_score
            relational_context_score=relational_score
            constitutional_alignment=alignment
        )
