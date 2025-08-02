"""backend/core/governor_proxy.py
=====================================================
A lightweight proxy to access the Kimera Ethical Governor from
any moduleâ€”inside or outside FastAPI request-handling contexts.

This file fulfils *Phase 2-A* of the constitutional integration plan.

Key features
------------
1.  *Global Singleton*: Instantiates a fallback `EthicalGovernor` on first
    access, ensuring availability even in offline or batch contexts.
2.  *FastAPI-Aware*: If a Governor instance has already been registered by the
    application (e.g.
    `backend.main.ethical_governor`), the proxy will return that instance to
    maintain a single source of truth.
3.  *Helper `require_constitutional`*: Convenience wrapper that raises a
    `UnconstitutionalActionError` if a proposal is rejected, satisfying the
    Zero-Debugging Constraint by providing structured exceptions and logs.
"""

from __future__ import annotations

import importlib
import logging
from typing import Optional

from src.core.ethical_governor import EthicalGovernor, ActionProposal, Verdict
from src.utils.kimera_exceptions import KimeraCognitiveError, ErrorSeverity

logger = logging.getLogger(__name__)

__all__ = [
    "UnconstitutionalActionError",
    "get_governor",
    "require_constitutional",
]

# ---------------------------------------------------------------------------
# Exception definitions
# ---------------------------------------------------------------------------

class UnconstitutionalActionError(KimeraCognitiveError):
    """Raised when an action is deemed unconstitutional by the Ethical Governor."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            severity=ErrorSeverity.HIGH,
            **kwargs,
        )
        # Provide default recovery suggestion if none supplied
        if not self.recovery_suggestions:
            self.recovery_suggestions.append(
                "Revise the action to align with the Prime Directive of Unity and"
                " the Law of Transformative Connection."
            )

# ---------------------------------------------------------------------------
# Singleton handling
# ---------------------------------------------------------------------------

_global_governor: Optional[EthicalGovernor] = None


def _init_global_governor() -> EthicalGovernor:
    """Instantiate the fallback global governor if not yet created."""
    global _global_governor
    if _global_governor is None:
        logger.info("Initializing global EthicalGovernor instance (fallback mode).")
        _global_governor = EthicalGovernor()
    return _global_governor


def _get_app_governor() -> Optional[EthicalGovernor]:
    """Attempt to retrieve the Governor instance from the FastAPI app."""
    try:
        main_module = importlib.import_module("backend.main")
        return getattr(main_module, "ethical_governor", None)
    except ModuleNotFoundError:
        # Application context not available (e.g. during offline batch run)
        return None


def get_governor() -> EthicalGovernor:
    """Return the active EthicalGovernor instance.

    Preference order:
    1. Governor instantiated by the running FastAPI app.
    2. Fallback to (lazy) global singleton.
    """
    app_governor = _get_app_governor()
    if app_governor is not None:
        return app_governor
    return _init_global_governor()


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def require_constitutional(
    proposal: ActionProposal,
    governor: Optional[EthicalGovernor] = None,
) -> Verdict:
    """Adjudicate *proposal* and raise if unconstitutional.

    Args:
        proposal: ActionProposal describing the intended action.
        governor: Optionally supply a specific EthicalGovernor instance. If
                  *None*, the proxy resolves the active singleton.

    Returns:
        Verdict.CONSTITUTIONAL if action passes.

    Raises:
        UnconstitutionalActionError: If the Governor returns UNCONSTITUTIONAL.
    """
    gov = governor or get_governor()

    verdict = gov.adjudicate(proposal)
    if verdict is Verdict.UNCONSTITUTIONAL:
        logger.error(
            "UnconstitutionalActionError: Action '%s' rejected by Governor.",
            proposal.description,
        )
        raise UnconstitutionalActionError(
            message="Action violates the Kimera Constitution.",
            context={
                "source_engine": proposal.source_engine,
                "description": proposal.description,
            },
        )

    logger.debug(
        "Action '%s' approved by EthicalGovernor (CONSTITUTIONAL).",
        proposal.description,
    )
    return verdict 