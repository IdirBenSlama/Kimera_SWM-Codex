"""
KIMERA Governance Module
========================

Implements aerospace-grade governance patterns for system reliability.
Based on DO-178C (Software Considerations in Airborne Systems) and
ISO 26262 (Functional Safety) standards.

Key Principles:
1. Redundancy and Voting Systems (Triple Modular Redundancy)
2. Fail-Safe Defaults
3. Continuous Health Monitoring
4. Deterministic Behavior
5. Formal Verification Support
"""

from .audit_trail import AuditEvent, AuditTrail
from .decision_voter import DecisionVoter, VotingStrategy
from .erl import EthicalReflexLayer as ERL
from .erl import EthicalViolationType as ContentCategory
from .governance_engine import (
    GovernanceDecision,
    GovernanceEngine,
    GovernancePolicy,
    create_default_policies,
)
from .safety_monitor import SafetyLevel, SafetyMonitor

__all__ = [
    "GovernanceEngine",
    "GovernancePolicy",
    "GovernanceDecision",
    "SafetyMonitor",
    "SafetyLevel",
    "DecisionVoter",
    "VotingStrategy",
    "AuditTrail",
    "AuditEvent",
    "ERL",
    "ContentCategory",
    "create_default_policies",
]
