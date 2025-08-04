"""
Decision Voter - Triple Modular Redundancy Pattern
==================================================

Implements voting mechanisms from aerospace and nuclear engineering:
- Triple Modular Redundancy (TMR)
- Byzantine Fault Tolerance
- N-Version Programming

Used in:
- Flight control systems
- Nuclear reactor safety systems
- Medical device controllers
"""

import asyncio
import hashlib
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar

import numpy as np

logger = logging.getLogger(__name__)

T = TypeVar("T")


class VotingStrategy(Enum):
    """Voting strategies for redundant systems."""

    MAJORITY = "majority"  # Simple majority (2 out of 3)
    UNANIMOUS = "unanimous"  # All must agree
    WEIGHTED = "weighted"  # Weighted by confidence
    BYZANTINE = "byzantine"  # Byzantine fault tolerant
    MEDIAN = "median"  # Median value (for numeric)
    AVERAGE = "average"  # Average value (for numeric)


class VoterStatus(Enum):
    """Status of individual voters."""

    HEALTHY = auto()
    DEGRADED = auto()
    FAILED = auto()
    BYZANTINE = auto()  # Producing incorrect results


@dataclass
class VoteResult(Generic[T]):
    """Result of a voting process."""

    winner: Optional[T]
    confidence: float
    consensus_level: float
    votes: List[T]
    voter_statuses: List[VoterStatus]
    strategy_used: VotingStrategy
    execution_time_ms: float
    disagreements: List[str] = field(default_factory=list)


@dataclass
class Voter:
    """Individual voter in the redundant system."""

    id: str
    name: str
    compute_func: Callable
    weight: float = 1.0
    status: VoterStatus = VoterStatus.HEALTHY

    # Performance tracking
    total_votes: int = 0
    agreement_rate: float = 1.0
    average_execution_time_ms: float = 0.0
    consecutive_failures: int = 0


class DecisionVoter(Generic[T]):
    """
    Implements redundant decision making with voting.

    Based on TMR (Triple Modular Redundancy) used in:
    - Boeing 777 flight control
    - Space Shuttle computers
    - Nuclear safety systems
    """

    def __init__(
        self,
        min_voters: int = 3,
        max_disagreement_rate: float = 0.1,
        byzantine_threshold: int = 5,
    ):
        self.min_voters = min_voters
        self.max_disagreement_rate = max_disagreement_rate
        self.byzantine_threshold = byzantine_threshold

        self.voters: List[Voter] = []
        self.voting_history: List[VoteResult] = []

        logger.info(
            f"Decision Voter initialized "
            f"(min_voters={min_voters}, "
            f"max_disagreement={max_disagreement_rate:.1%})"
        )

    def register_voter(self, voter: Voter):
        """Register a voter in the redundant system."""
        if len(self.voters) >= 7:  # Practical limit
            raise ValueError("Maximum 7 voters supported")

        self.voters.append(voter)
        logger.info(f"Registered voter: {voter.name} (total: {len(self.voters)})")

    async def vote(
        self,
        input_data: Any,
        strategy: VotingStrategy = VotingStrategy.MAJORITY,
        timeout_ms: int = 1000,
    ) -> VoteResult[T]:
        """
        Execute redundant computation and vote on result.

        Implements aerospace-grade voting with:
        - Timeout protection
        - Byzantine fault detection
        - Automatic voter health tracking
        """
        if len(self.voters) < self.min_voters:
            raise ValueError(
                f"Insufficient voters: {len(self.voters)} < {self.min_voters}"
            )

        start_time = time.time()

        # Execute all voters in parallel
        tasks = []
        for voter in self.voters:
            if voter.status != VoterStatus.FAILED:
                task = asyncio.create_task(
                    self._execute_voter(voter, input_data, timeout_ms)
                )
                tasks.append((voter, task))

        # Collect results
        votes = []
        voter_statuses = []

        for voter, task in tasks:
            try:
                result = await task
                if result is not None:
                    votes.append(result)
                    voter_statuses.append(voter.status)
                else:
                    voter_statuses.append(VoterStatus.FAILED)
            except Exception as e:
                logger.error(f"Voter {voter.name} failed: {e}")
                voter_statuses.append(VoterStatus.FAILED)
                self._update_voter_health(voter, success=False)

        # Apply voting strategy
        if not votes:
            return VoteResult(
                winner=None,
                confidence=0.0,
                consensus_level=0.0,
                votes=[],
                voter_statuses=voter_statuses,
                strategy_used=strategy,
                execution_time_ms=(time.time() - start_time) * 1000,
                disagreements=["All voters failed"],
            )

        winner, confidence, disagreements = self._apply_voting_strategy(
            votes, strategy, voter_statuses
        )

        # Calculate consensus level
        consensus_level = self._calculate_consensus(votes)

        # Update voter health based on agreement
        self._update_voter_agreement(votes, winner)

        # Check for Byzantine behavior
        self._check_byzantine_behavior()

        execution_time = (time.time() - start_time) * 1000

        result = VoteResult(
            winner=winner,
            confidence=confidence,
            consensus_level=consensus_level,
            votes=votes,
            voter_statuses=voter_statuses,
            strategy_used=strategy,
            execution_time_ms=execution_time,
            disagreements=disagreements,
        )

        # Store in history
        self.voting_history.append(result)
        if len(self.voting_history) > 1000:
            self.voting_history = self.voting_history[-500:]

        # Log result
        logger.info(
            f"Vote complete: winner={winner}, "
            f"confidence={confidence:.2f}, "
            f"consensus={consensus_level:.2f}, "
            f"time={execution_time:.1f}ms"
        )

        return result

    async def _execute_voter(
        self, voter: Voter, input_data: Any, timeout_ms: int
    ) -> Optional[T]:
        """Execute a single voter with timeout protection."""
        try:
            start_time = time.time()

            # Create timeout
            timeout = timeout_ms / 1000.0

            if asyncio.iscoroutinefunction(voter.compute_func):
                result = await asyncio.wait_for(
                    voter.compute_func(input_data), timeout=timeout
                )
            else:
                # Run sync function in executor
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, voter.compute_func, input_data),
                    timeout=timeout,
                )

            # Update performance metrics
            execution_time = (time.time() - start_time) * 1000
            voter.total_votes += 1
            voter.average_execution_time_ms = (
                voter.average_execution_time_ms * (voter.total_votes - 1)
                + execution_time
            ) / voter.total_votes
            voter.consecutive_failures = 0

            return result

        except asyncio.TimeoutError:
            logger.warning(f"Voter {voter.name} timed out")
            voter.consecutive_failures += 1
            return None
        except Exception as e:
            logger.error(f"Voter {voter.name} error: {e}")
            voter.consecutive_failures += 1
            return None

    def _apply_voting_strategy(
        self,
        votes: List[T],
        strategy: VotingStrategy,
        voter_statuses: List[VoterStatus],
    ) -> Tuple[Optional[T], float, List[str]]:
        """Apply the specified voting strategy."""
        disagreements = []

        if strategy == VotingStrategy.MAJORITY:
            # Simple majority voting
            vote_counts = Counter(self._hashable_vote(v) for v in votes)
            most_common = vote_counts.most_common(1)[0]
            winner_hash, count = most_common

            # Find original vote
            winner = None
            for vote in votes:
                if self._hashable_vote(vote) == winner_hash:
                    winner = vote
                    break

            confidence = count / len(votes)

            if confidence < 0.5:
                disagreements.append("No majority consensus")

        elif strategy == VotingStrategy.UNANIMOUS:
            # All must agree
            unique_votes = set(self._hashable_vote(v) for v in votes)
            if len(unique_votes) == 1:
                winner = votes[0]
                confidence = 1.0
            else:
                winner = None
                confidence = 0.0
                disagreements.append("Unanimous vote required but not achieved")

        elif strategy == VotingStrategy.MEDIAN:
            # For numeric values
            try:
                numeric_votes = [float(v) for v in votes]
                winner = np.median(numeric_votes)
                # Confidence based on spread
                std_dev = np.std(numeric_votes)
                mean_val = np.mean(numeric_votes)
                confidence = 1.0 - min(1.0, std_dev / (abs(mean_val) + 1))
            except (ValueError, TypeError):
                winner = votes[0]  # Fallback
                confidence = 1.0 / len(votes)
                disagreements.append("Non-numeric values for median voting")

        elif strategy == VotingStrategy.BYZANTINE:
            # Byzantine fault tolerant (requires 2f+1 voters for f faults)
            vote_counts = Counter(self._hashable_vote(v) for v in votes)
            most_common = vote_counts.most_common(1)[0]
            winner_hash, count = most_common

            # Need more than 2/3 agreement for Byzantine
            if count > len(votes) * 2 / 3:
                for vote in votes:
                    if self._hashable_vote(vote) == winner_hash:
                        winner = vote
                        break
                confidence = count / len(votes)
            else:
                winner = None
                confidence = 0.0
                disagreements.append("Insufficient Byzantine consensus")

        else:
            # Default to majority
            return self._apply_voting_strategy(
                votes, VotingStrategy.MAJORITY, voter_statuses
            )

        return winner, confidence, disagreements

    def _hashable_vote(self, vote: Any) -> str:
        """Convert vote to hashable string for comparison."""
        if isinstance(vote, (str, int, float, bool)):
            return str(vote)
        elif isinstance(vote, (list, tuple)):
            return str(tuple(vote))
        elif isinstance(vote, dict):
            return hashlib.sha256(str(sorted(vote.items())).encode()).hexdigest()
        else:
            return hashlib.sha256(str(vote).encode()).hexdigest()

    def _calculate_consensus(self, votes: List[T]) -> float:
        """Calculate consensus level among votes."""
        if not votes:
            return 0.0

        vote_hashes = [self._hashable_vote(v) for v in votes]
        vote_counts = Counter(vote_hashes)

        # Shannon entropy for consensus measurement
        total = len(votes)
        entropy = 0.0
        for count in vote_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)

        # Normalize (0 = perfect consensus, 1 = maximum disagreement)
        max_entropy = np.log2(len(votes))
        consensus = 1.0 - (entropy / max_entropy if max_entropy > 0 else 0)

        return consensus

    def _update_voter_health(self, voter: Voter, success: bool):
        """Update voter health status."""
        if success:
            voter.consecutive_failures = 0
            if voter.status == VoterStatus.DEGRADED:
                voter.status = VoterStatus.HEALTHY
        else:
            voter.consecutive_failures += 1

            if voter.consecutive_failures >= self.byzantine_threshold:
                voter.status = VoterStatus.BYZANTINE
            elif voter.consecutive_failures >= 3:
                voter.status = VoterStatus.FAILED
            elif voter.consecutive_failures >= 1:
                voter.status = VoterStatus.DEGRADED

    def _update_voter_agreement(self, votes: List[T], winner: Optional[T]):
        """Update voter agreement rates."""
        if winner is None:
            return

        winner_hash = self._hashable_vote(winner)

        for i, voter in enumerate(self.voters):
            if i < len(votes):
                vote_hash = self._hashable_vote(votes[i])
                if vote_hash == winner_hash:
                    # Voter agreed with consensus
                    voter.agreement_rate = (
                        voter.agreement_rate * 0.95 + 0.05
                    )  # Exponential moving average
                else:
                    # Voter disagreed
                    voter.agreement_rate = voter.agreement_rate * 0.95

    def _check_byzantine_behavior(self):
        """Check for Byzantine (malicious/faulty) voters."""
        for voter in self.voters:
            if voter.agreement_rate < (1 - self.max_disagreement_rate):
                if voter.status != VoterStatus.BYZANTINE:
                    logger.warning(
                        f"Voter {voter.name} showing Byzantine behavior "
                        f"(agreement rate: {voter.agreement_rate:.2%})"
                    )
                    voter.status = VoterStatus.BYZANTINE

    def get_voter_health_report(self) -> Dict[str, Any]:
        """Get comprehensive voter health report."""
        healthy_count = sum(1 for v in self.voters if v.status == VoterStatus.HEALTHY)

        report = {
            "total_voters": len(self.voters),
            "healthy_voters": healthy_count,
            "health_percentage": (
                healthy_count / len(self.voters) * 100 if self.voters else 0
            ),
            "voters": [],
        }

        for voter in self.voters:
            report["voters"].append(
                {
                    "id": voter.id,
                    "name": voter.name,
                    "status": voter.status.name,
                    "agreement_rate": voter.agreement_rate,
                    "total_votes": voter.total_votes,
                    "avg_execution_time_ms": voter.average_execution_time_ms,
                    "consecutive_failures": voter.consecutive_failures,
                }
            )

        # Recent voting statistics
        if self.voting_history:
            recent_votes = self.voting_history[-10:]
            report["recent_consensus_avg"] = np.mean(
                [v.consensus_level for v in recent_votes]
            )
            report["recent_confidence_avg"] = np.mean(
                [v.confidence for v in recent_votes]
            )

        return report


def create_redundant_voters(compute_func: Callable, count: int = 3) -> List[Voter]:
    """Create multiple redundant voters with the same compute function."""
    voters = []
    for i in range(count):
        voter = Voter(
            id=f"voter-{i+1}",
            name=f"Redundant Voter {i+1}",
            compute_func=compute_func,
            weight=1.0,
        )
        voters.append(voter)
    return voters
