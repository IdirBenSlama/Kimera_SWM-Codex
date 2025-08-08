"""
Defines the ActionProposal class, a structured representation of an
action proposed by a system component, to be adjudicated by the
EthicalGovernor.

This class is in its own file to prevent circular dependencies between
the Governor, Heart, and other cognitive engines.
"""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class ActionProposal:
    """Auto-generated class."""
    pass
    """
    A structured representation of an action proposed by a system component.
    This object is passed to the EthicalGovernor for adjudication.
    """

    source_engine: str
    description: str
    logical_analysis: Dict[str, Any]
    associated_data: Dict[str, Any] = field(default_factory=dict)
