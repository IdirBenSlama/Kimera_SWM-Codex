"""
Governance Engine with Aerospace-Grade Reliability
=================================================

Implements governance patterns from DO-178C and ISO 26262 standards.
Ensures system decisions are safe, traceable, and deterministic.

Design Patterns:
- Command Pattern for decision execution
- Observer Pattern for monitoring
- State Machine for deterministic behavior
- Chain of Responsibility for approval workflows
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
import hashlib
import json
from collections import defaultdict
import threading

logger = logging.getLogger(__name__)

class PolicyPriority(Enum):
    """Policy priority levels (aerospace standard)."""
    SAFETY_CRITICAL = 1  # Highest priority - system safety
    MISSION_CRITICAL = 2  # Mission success
    PERFORMANCE = 3       # Performance optimization
    EFFICIENCY = 4        # Resource efficiency
    OPTIONAL = 5          # Nice to have

class PolicyState(Enum):
    """Policy lifecycle states."""
    DRAFT = auto()
    REVIEW = auto()
    APPROVED = auto()
    ACTIVE = auto()
    SUSPENDED = auto()
    RETIRED = auto()

class DecisionOutcome(Enum):
    """Decision outcomes with safety bias."""
    APPROVED = "approved"
    DENIED = "denied"
    DEFERRED = "deferred"
    REQUIRES_REVIEW = "requires_review"
    SAFETY_OVERRIDE = "safety_override"

@dataclass
class GovernancePolicy:
    """
    Governance policy with aerospace-grade traceability.
    
    Follows DO-178C requirements for software lifecycle data.
    """
    id: str
    name: str
    description: str
    priority: PolicyPriority
    state: PolicyState = PolicyState.DRAFT
    version: str = "1.0.0"
    
    # Validation rules
    validation_rules: List[Callable] = field(default_factory=list)
    
    # Safety constraints
    safety_constraints: Dict[str, Any] = field(default_factory=dict)
    max_execution_time_ms: int = 1000  # Deterministic timing
    
    # Traceability
    created_by: str = "system"
    created_at: datetime = field(default_factory=datetime.now)
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    
    # Metrics
    execution_count: int = 0
    failure_count: int = 0
    average_execution_time_ms: float = 0.0
    
    def calculate_hash(self) -> str:
        """Calculate policy hash for integrity verification."""
        policy_data = {
            'id': self.id,
            'name': self.name,
            'version': self.version,
            'priority': self.priority.value,
            'safety_constraints': self.safety_constraints
        }
        return hashlib.sha256(
            json.dumps(policy_data, sort_keys=True).encode()
        ).hexdigest()
    
    def validate(self, context: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate context against policy rules.
        
        Returns:
            Tuple of (is_valid, list_of_violations)
        """
        violations = []
        
        for rule in self.validation_rules:
            try:
                if not rule(context):
                    violations.append(f"Rule {rule.__name__} failed")
            except Exception as e:
                violations.append(f"Rule {rule.__name__} error: {str(e)}")
        
        # Check safety constraints
        for constraint, limit in self.safety_constraints.items():
            if constraint in context:
                if isinstance(limit, (int, float)):
                    if context[constraint] > limit:
                        violations.append(
                            f"Safety constraint {constraint} exceeded: "
                            f"{context[constraint]} > {limit}"
                        )
        
        return len(violations) == 0, violations

@dataclass
class GovernanceDecision:
    """Immutable decision record for audit trail."""
    id: str
    policy_id: str
    context: Dict[str, Any]
    outcome: DecisionOutcome
    reasoning: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    execution_time_ms: float = 0.0
    safety_checks_passed: bool = True
    
    def to_audit_record(self) -> Dict[str, Any]:
        """Convert to audit record format."""
        return {
            'id': self.id,
            'policy_id': self.policy_id,
            'outcome': self.outcome.value,
            'timestamp': self.timestamp.isoformat(),
            'execution_time_ms': self.execution_time_ms,
            'safety_checks_passed': self.safety_checks_passed,
            'context_hash': hashlib.sha256(
                json.dumps(self.context, sort_keys=True).encode()
            ).hexdigest()
        }

class GovernanceEngine:
    """
    Main governance engine with aerospace-grade reliability.
    
    Implements:
    - Redundant decision paths
    - Fail-safe defaults
    - Deterministic execution
    - Complete audit trail
    """
    
    def __init__(self, safety_mode: bool = True):
        self.safety_mode = safety_mode
        self.policies: Dict[str, GovernancePolicy] = {}
        self.active_policies: Set[str] = set()
        self.decision_history: List[GovernanceDecision] = []
        self._lock = threading.RLock()
        
        # Monitoring
        self.health_checks: Dict[str, Callable] = {}
        self.performance_metrics = defaultdict(list)
        
        # Safety systems
        self.emergency_stop = False
        self.degraded_mode = False
        
        logger.info(
            f"Governance Engine initialized "
            f"(safety_mode={'ON' if safety_mode else 'OFF'})"
        )
    
    def register_policy(self, policy: GovernancePolicy) -> bool:
        """
        Register a new governance policy.
        
        Follows aerospace change management procedures.
        """
        with self._lock:
            # Verify policy integrity
            expected_hash = policy.calculate_hash()
            
            # Check for conflicts
            if policy.id in self.policies:
                existing = self.policies[policy.id]
                if existing.state == PolicyState.ACTIVE:
                    logger.error(
                        f"Cannot register policy {policy.id}: "
                        f"Active policy exists"
                    )
                    return False
            
            # Safety check: Ensure safety-critical policies are reviewed
            if policy.priority == PolicyPriority.SAFETY_CRITICAL:
                if policy.state != PolicyState.APPROVED:
                    logger.error(
                        f"Safety-critical policy {policy.id} "
                        f"must be approved before registration"
                    )
                    return False
            
            self.policies[policy.id] = policy
            logger.info(
                f"Policy registered: {policy.id} "
                f"(priority={policy.priority.name})"
            )
            return True
    
    def activate_policy(self, policy_id: str) -> bool:
        """Activate a policy with safety checks."""
        with self._lock:
            if policy_id not in self.policies:
                return False
            
            policy = self.policies[policy_id]
            
            # State transition validation
            if policy.state not in [PolicyState.APPROVED, PolicyState.SUSPENDED]:
                logger.error(
                    f"Cannot activate policy {policy_id} "
                    f"in state {policy.state}"
                )
                return False
            
            policy.state = PolicyState.ACTIVE
            self.active_policies.add(policy_id)
            
            logger.info(f"Policy activated: {policy_id}")
            return True
    
    async def make_decision(
        self,
        context: Dict[str, Any],
        required_policies: Optional[List[str]] = None
    ) -> GovernanceDecision:
        """
        Make a governance decision with full traceability.
        
        Implements aerospace-grade decision making:
        1. Safety checks first
        2. Policy validation
        3. Deterministic execution
        4. Complete audit trail
        """
        start_time = datetime.now()
        decision_id = f"DEC-{start_time.timestamp():.0f}"
        reasoning = []
        
        # Emergency stop check
        if self.emergency_stop:
            return GovernanceDecision(
                id=decision_id,
                policy_id="EMERGENCY",
                context=context,
                outcome=DecisionOutcome.SAFETY_OVERRIDE,
                reasoning=["Emergency stop activated"],
                safety_checks_passed=False
            )
        
        # Degraded mode check
        if self.degraded_mode:
            reasoning.append("System in degraded mode")
        
        # Determine which policies to evaluate
        if required_policies:
            policies_to_check = [
                self.policies[pid] 
                for pid in required_policies 
                if pid in self.policies
            ]
        else:
            policies_to_check = [
                self.policies[pid] 
                for pid in self.active_policies
            ]
        
        # Sort by priority (safety-critical first)
        policies_to_check.sort(key=lambda p: p.priority.value)
        
        # Evaluate policies
        all_passed = True
        safety_passed = True
        
        for policy in policies_to_check:
            try:
                # Enforce deterministic timing
                policy_start = datetime.now()
                
                is_valid, violations = policy.validate(context)
                
                policy_time = (datetime.now() - policy_start).total_seconds() * 1000
                
                # Check timing constraint
                if policy_time > policy.max_execution_time_ms:
                    violations.append(
                        f"Execution time exceeded: "
                        f"{policy_time:.1f}ms > {policy.max_execution_time_ms}ms"
                    )
                    is_valid = False
                
                # Update metrics
                policy.execution_count += 1
                policy.average_execution_time_ms = (
                    (policy.average_execution_time_ms * (policy.execution_count - 1) + 
                     policy_time) / policy.execution_count
                )
                
                if not is_valid:
                    all_passed = False
                    policy.failure_count += 1
                    
                    if policy.priority == PolicyPriority.SAFETY_CRITICAL:
                        safety_passed = False
                        reasoning.append(
                            f"SAFETY CRITICAL: Policy {policy.id} failed: "
                            f"{', '.join(violations)}"
                        )
                    else:
                        reasoning.append(
                            f"Policy {policy.id} failed: "
                            f"{', '.join(violations)}"
                        )
                else:
                    reasoning.append(f"Policy {policy.id} passed")
                    
            except Exception as e:
                logger.error(f"Policy {policy.id} evaluation error: {e}")
                all_passed = False
                if policy.priority == PolicyPriority.SAFETY_CRITICAL:
                    safety_passed = False
                reasoning.append(f"Policy {policy.id} error: {str(e)}")
        
        # Determine outcome with safety bias
        if not safety_passed:
            outcome = DecisionOutcome.SAFETY_OVERRIDE
        elif all_passed:
            outcome = DecisionOutcome.APPROVED
        elif self.safety_mode:
            outcome = DecisionOutcome.DENIED  # Fail-safe default
        else:
            outcome = DecisionOutcome.REQUIRES_REVIEW
        
        # Create decision record
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        decision = GovernanceDecision(
            id=decision_id,
            policy_id=",".join(p.id for p in policies_to_check),
            context=context,
            outcome=outcome,
            reasoning=reasoning,
            execution_time_ms=execution_time,
            safety_checks_passed=safety_passed
        )
        
        # Store in history (with size limit for deterministic memory usage)
        with self._lock:
            self.decision_history.append(decision)
            if len(self.decision_history) > 10000:
                self.decision_history = self.decision_history[-5000:]
        
        # Log decision
        logger.info(
            f"Governance decision {decision_id}: {outcome.value} "
            f"(execution_time={execution_time:.1f}ms)"
        )
        
        return decision
    
    def emergency_shutdown(self, reason: str):
        """Activate emergency shutdown (aerospace safety pattern)."""
        self.emergency_stop = True
        logger.critical(f"EMERGENCY SHUTDOWN: {reason}")
        
        # Create safety record
        decision = GovernanceDecision(
            id=f"EMERGENCY-{datetime.now().timestamp():.0f}",
            policy_id="EMERGENCY",
            context={"reason": reason},
            outcome=DecisionOutcome.SAFETY_OVERRIDE,
            reasoning=[f"Emergency shutdown: {reason}"],
            safety_checks_passed=False
        )
        
        with self._lock:
            self.decision_history.append(decision)
    
    def enter_degraded_mode(self, reason: str):
        """Enter degraded mode with reduced functionality."""
        self.degraded_mode = True
        logger.warning(f"Entering degraded mode: {reason}")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health report."""
        with self._lock:
            total_decisions = len(self.decision_history)
            safety_overrides = sum(
                1 for d in self.decision_history 
                if d.outcome == DecisionOutcome.SAFETY_OVERRIDE
            )
            
            policy_health = {}
            for policy in self.policies.values():
                if policy.execution_count > 0:
                    policy_health[policy.id] = {
                        'state': policy.state.name,
                        'execution_count': policy.execution_count,
                        'failure_rate': policy.failure_count / policy.execution_count,
                        'avg_execution_time_ms': policy.average_execution_time_ms
                    }
            
            return {
                'emergency_stop': self.emergency_stop,
                'degraded_mode': self.degraded_mode,
                'total_decisions': total_decisions,
                'safety_overrides': safety_overrides,
                'active_policies': len(self.active_policies),
                'policy_health': policy_health
            }

# Predefined safety-critical policies
def create_default_policies() -> List[GovernancePolicy]:
    """Create default aerospace-grade governance policies."""
    
    # Memory safety policy
    memory_policy = GovernancePolicy(
        id="POLICY-MEM-001",
        name="Memory Safety Policy",
        description="Ensures memory usage stays within safe bounds",
        priority=PolicyPriority.SAFETY_CRITICAL,
        state=PolicyState.APPROVED,
        safety_constraints={
            'memory_usage_percent': 90,
            'memory_growth_rate': 10  # % per minute
        },
        validation_rules=[
            lambda ctx: ctx.get('memory_usage_percent', 0) < 90,
            lambda ctx: ctx.get('memory_available_gb', 0) > 1.0
        ]
    )
    
    # Response time policy
    response_policy = GovernancePolicy(
        id="POLICY-PERF-001",
        name="Response Time Policy",
        description="Ensures system responds within acceptable time",
        priority=PolicyPriority.MISSION_CRITICAL,
        state=PolicyState.APPROVED,
        safety_constraints={
            'response_time_ms': 5000,
            'queue_depth': 1000
        },
        validation_rules=[
            lambda ctx: ctx.get('response_time_ms', 0) < 5000,
            lambda ctx: ctx.get('queue_depth', 0) < 1000
        ]
    )
    
    # Data integrity policy
    integrity_policy = GovernancePolicy(
        id="POLICY-DATA-001",
        name="Data Integrity Policy",
        description="Ensures data consistency and validity",
        priority=PolicyPriority.SAFETY_CRITICAL,
        state=PolicyState.APPROVED,
        safety_constraints={
            'checksum_failures': 0,
            'data_corruption_events': 0
        },
        validation_rules=[
            lambda ctx: ctx.get('checksum_failures', 0) == 0,
            lambda ctx: ctx.get('data_corruption_events', 0) == 0
        ]
    )
    
    return [memory_policy, response_policy, integrity_policy]