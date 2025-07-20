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

from .governance_engine import GovernanceEngine, GovernancePolicy, GovernanceDecision, create_default_policies
from .safety_monitor import SafetyMonitor, SafetyLevel
from .decision_voter import DecisionVoter, VotingStrategy
from .audit_trail import AuditTrail, AuditEvent
from .erl import EthicalReflexLayer as ERL, EthicalViolationType as ContentCategory

__all__ = [
    'GovernanceEngine',
    'GovernancePolicy',
    'GovernanceDecision',
    'SafetyMonitor',
    'SafetyLevel',
        'DecisionVoter',
    'VotingStrategy',
    'AuditTrail',
    'AuditEvent',
    'ERL',
    'ContentCategory',
    'create_default_policies'
]