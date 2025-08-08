"""
PORTAL MAXWELL DEMON ENGINE
============================

Revolutionary information sorting engine that uses quantum portals to intelligently
sort information while respecting Landauer's principle. This demon performs work
to decrease entropy locally while paying the fundamental thermodynamic cost.

Key Features:
- Intelligent information sorting through quantum portals
- Landauer cost calculation and enforcement
- Information work extraction: W = k_B T ln(2) per bit
- Quantum tunnel-based information processing
- Entropy management and optimization
"""

import logging
import math
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


class SortingStrategy(Enum):
    """Information sorting strategies"""

    ENTROPY_MINIMIZATION = "entropy_min"
    COHERENCE_MAXIMIZATION = "coherence_max"
    SEMANTIC_CLUSTERING = "semantic_cluster"
    QUANTUM_TUNNELING = "quantum_tunnel"
    HYBRID_OPTIMIZATION = "hybrid_opt"


@dataclass
class InformationPacket:
    """Auto-generated class."""
    pass
    """Represents a packet of information to be sorted"""

    packet_id: str
    semantic_vector: np.ndarray
    entropy_content: float
    coherence_score: float
    quantum_state: Optional[np.ndarray] = None
    portal_address: Optional[str] = None
    sort_priority: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class QuantumPortal:
    """Auto-generated class."""
    pass
    """Represents a quantum portal for information transport"""

    portal_id: str
    portal_type: str  # 'high_entropy', 'low_entropy', 'coherent', 'mixed'
    capacity: int
    current_load: int
    energy_cost_per_bit: float
    portal_efficiency: float
    quantum_coherence: float
    tunnel_probability: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SortingOperation:
    """Auto-generated class."""
    pass
    """Represents a completed sorting operation"""

    operation_id: str
    packets_sorted: int
    bits_processed: float
    landauer_cost: float
    work_performed: float
    entropy_reduction: float
    information_gain: float
    sorting_efficiency: float
    energy_conservation_error: float
    operation_duration: float
    strategy_used: SortingStrategy
    timestamp: datetime = field(default_factory=datetime.now)
class PortalMaxwellDemon:
    """Auto-generated class."""
    pass
    """
    Portal Maxwell Demon Engine

    Performs intelligent information sorting through quantum portals while
    respecting fundamental thermodynamic limits and Landauer's principle.
    """

    def __init__(
        self
        temperature: float = 1.0
        landauer_efficiency: float = 0.9
        quantum_coherence_threshold: float = 0.7
    ):
        """
        Initialize the Portal Maxwell Demon

        Args:
            temperature: Operating temperature for Landauer cost calculations
            landauer_efficiency: Efficiency factor for Landauer limit adherence
            quantum_coherence_threshold: Minimum coherence for quantum operations
        """
        self.temperature = temperature
        self.landauer_efficiency = landauer_efficiency
        self.quantum_coherence_threshold = quantum_coherence_threshold

        # Physical constants (normalized for cognitive fields)
        self.boltzmann_constant = 1.0
        self.planck_constant = 1.0
        self.cognitive_scaling_factor = 0.1

        # Portal management
        self.portals: Dict[str, QuantumPortal] = {}
        self.sorted_information: Dict[str, List[InformationPacket]] = defaultdict(list)

        # Operation tracking
        self.operations_completed = 0
        self.total_bits_processed = 0.0
        self.total_landauer_cost = 0.0
        self.total_entropy_reduction = 0.0
        self.operation_history = []

        # Initialize default portals
        self._initialize_default_portals()

        logger.info(
            f"👹 Portal Maxwell Demon initialized (T={temperature}, efficiency={landauer_efficiency})"
        )

    def _initialize_default_portals(self):
        """Initialize the default set of quantum portals"""
        portal_configs = [
            {
                "portal_id": "high_entropy_portal",
                "portal_type": "high_entropy",
                "capacity": 1000
                "energy_cost_per_bit": 0.1
                "portal_efficiency": 0.8
                "quantum_coherence": 0.3
                "tunnel_probability": 0.6
            },
            {
                "portal_id": "low_entropy_portal",
                "portal_type": "low_entropy",
                "capacity": 500
                "energy_cost_per_bit": 0.05
                "portal_efficiency": 0.95
                "quantum_coherence": 0.9
                "tunnel_probability": 0.9
            },
            {
                "portal_id": "coherent_portal",
                "portal_type": "coherent",
                "capacity": 750
                "energy_cost_per_bit": 0.07
                "portal_efficiency": 0.9
                "quantum_coherence": 0.85
                "tunnel_probability": 0.8
            },
            {
                "portal_id": "mixed_portal",
                "portal_type": "mixed",
                "capacity": 1200
                "energy_cost_per_bit": 0.08
                "portal_efficiency": 0.75
                "quantum_coherence": 0.5
                "tunnel_probability": 0.7
            },
        ]

        for config in portal_configs:
            portal = QuantumPortal(
                portal_id=config["portal_id"],
                portal_type=config["portal_type"],
                capacity=config["capacity"],
                current_load=0
                energy_cost_per_bit=config["energy_cost_per_bit"],
                portal_efficiency=config["portal_efficiency"],
                quantum_coherence=config["quantum_coherence"],
                tunnel_probability=config["tunnel_probability"],
            )
            self.portals[portal.portal_id] = portal

    def calculate_landauer_cost(self, bits_erased: float) -> float:
        """
        Calculate the fundamental Landauer cost for information erasure

        Args:
            bits_erased: Number of bits to be erased

        Returns:
            Minimum energy cost according to Landauer's principle
        """
        landauer_cost = (
            bits_erased
            * self.boltzmann_constant
            * self.temperature
            * math.log(2)
            * self.cognitive_scaling_factor
        )

        return landauer_cost / self.landauer_efficiency

    def analyze_information_packets(
        self, packets: List[InformationPacket]
    ) -> Dict[str, Any]:
        """
        Analyze information packets to determine optimal sorting strategy

        Args:
            packets: List of InformationPacket objects to analyze

        Returns:
            Analysis results including sorting recommendations
        """
        if not packets:
            return {"error": "No packets provided for analysis"}

        # Calculate entropy statistics
        entropies = [p.entropy_content for p in packets]
        coherences = [p.coherence_score for p in packets]

        total_entropy = sum(entropies)
        mean_entropy = np.mean(entropies)
        entropy_variance = np.var(entropies)
        mean_coherence = np.mean(coherences)

        # Calculate semantic clustering potential
        if len(packets) > 1 and packets[0].semantic_vector is not None:
            vectors = np.array(
                [p.semantic_vector for p in packets if p.semantic_vector is not None]
            )
            if len(vectors) > 1:
                semantic_variance = np.var(vectors, axis=0).mean()
                clustering_potential = 1.0 / (1.0 + semantic_variance)
            else:
                clustering_potential = 0.5
        else:
            clustering_potential = 0.1

        # Determine optimal sorting strategy
        if entropy_variance > 0.5 and mean_entropy > 2.0:
            recommended_strategy = SortingStrategy.ENTROPY_MINIMIZATION
        elif mean_coherence > 0.8:
            recommended_strategy = SortingStrategy.COHERENCE_MAXIMIZATION
        elif clustering_potential > 0.7:
            recommended_strategy = SortingStrategy.SEMANTIC_CLUSTERING
        elif mean_coherence > self.quantum_coherence_threshold:
            recommended_strategy = SortingStrategy.QUANTUM_TUNNELING
        else:
            recommended_strategy = SortingStrategy.HYBRID_OPTIMIZATION

        # Calculate expected sorting efficiency
        efficiency_factors = {
            SortingStrategy.ENTROPY_MINIMIZATION: min(entropy_variance, 1.0),
            SortingStrategy.COHERENCE_MAXIMIZATION: mean_coherence
            SortingStrategy.SEMANTIC_CLUSTERING: clustering_potential
            SortingStrategy.QUANTUM_TUNNELING: mean_coherence
            SortingStrategy.HYBRID_OPTIMIZATION: (
                entropy_variance + mean_coherence + clustering_potential
            )
            / 3.0
        }

        expected_efficiency = efficiency_factors[recommended_strategy]

        return {
            "packet_count": len(packets),
            "total_entropy": total_entropy
            "mean_entropy": mean_entropy
            "entropy_variance": entropy_variance
            "mean_coherence": mean_coherence
            "clustering_potential": clustering_potential
            "recommended_strategy": recommended_strategy
            "expected_efficiency": expected_efficiency
            "sorting_complexity": total_entropy * len(packets),
            "quantum_suitability": mean_coherence > self.quantum_coherence_threshold
        }

    def select_optimal_portal(
        self, packet: InformationPacket, strategy: SortingStrategy
    ) -> Optional[QuantumPortal]:
        """
        Select the optimal portal for routing an information packet

        Args:
            packet: InformationPacket to route
            strategy: SortingStrategy being used

        Returns:
            Optimal QuantumPortal or None if no suitable portal available
        """
        available_portals = [
            p for p in self.portals.values() if p.current_load < p.capacity
        ]

        if not available_portals:
            return None

        # Score portals based on strategy and packet characteristics
        portal_scores = {}

        for portal in available_portals:
            score = 0.0

            # Base efficiency score
            score += portal.portal_efficiency * 0.3

            # Strategy-specific scoring
            if strategy == SortingStrategy.ENTROPY_MINIMIZATION:
                if (
                    packet.entropy_content > 1.5
                    and portal.portal_type == "high_entropy"
                ):
                    score += 0.4
                elif (
                    packet.entropy_content <= 1.5
                    and portal.portal_type == "low_entropy"
                ):
                    score += 0.4

            elif strategy == SortingStrategy.COHERENCE_MAXIMIZATION:
                score += portal.quantum_coherence * 0.4
                if portal.portal_type == "coherent":
                    score += 0.2

            elif strategy == SortingStrategy.QUANTUM_TUNNELING:
                score += portal.tunnel_probability * 0.4
                score += portal.quantum_coherence * 0.3

            elif strategy == SortingStrategy.SEMANTIC_CLUSTERING:
                if portal.portal_type == "mixed":
                    score += 0.3
                score += (1.0 - portal.energy_cost_per_bit) * 0.2

            else:  # HYBRID_OPTIMIZATION
                score += (
                    (
                        portal.portal_efficiency
                        + portal.quantum_coherence
                        + portal.tunnel_probability
                    )
                    / 3.0
                    * 0.4
                )

            # Capacity utilization penalty
            utilization = portal.current_load / portal.capacity
            score *= 1.0 - utilization * 0.5

            # Energy cost consideration
            score *= 1.0 - portal.energy_cost_per_bit * 0.2

            portal_scores[portal.portal_id] = score

        # Select portal with highest score
        best_portal_id = max(portal_scores.keys(), key=lambda x: portal_scores[x])
        return self.portals[best_portal_id]

    def perform_sorting_operation(
        self
        packets: List[InformationPacket],
        strategy: Optional[SortingStrategy] = None
    ) -> SortingOperation:
        """
        Perform a complete information sorting operation

        Args:
            packets: List of InformationPacket objects to sort
            strategy: Optional SortingStrategy (auto-determined if None)

        Returns:
            SortingOperation results
        """
        start_time = time.time()
        operation_id = str(uuid.uuid4())

        # Analyze packets if strategy not provided
        if strategy is None:
            analysis = self.analyze_information_packets(packets)
            strategy = analysis["recommended_strategy"]

        # Calculate initial entropy
        initial_entropy = sum(p.entropy_content for p in packets)

        # Process each packet
        sorted_count = 0
        total_bits = 0.0
        total_landauer_cost = 0.0
        total_work = 0.0

        for packet in packets:
            # Select optimal portal
            portal = self.select_optimal_portal(packet, strategy)

            if portal is None:
                continue  # Skip if no portal available

            # Calculate information bits for this packet
            packet_bits = (
                packet.entropy_content * len(packet.semantic_vector)
                if packet.semantic_vector is not None
                else packet.entropy_content
            )

            # Calculate work required for sorting
            sorting_work = packet_bits * portal.energy_cost_per_bit

            # Calculate Landauer cost for entropy reduction
            entropy_reduction = max(
                0
                packet.entropy_content
                - (packet.entropy_content * portal.portal_efficiency),
            )
            landauer_cost = self.calculate_landauer_cost(entropy_reduction)

            # Total work including Landauer cost
            total_work_packet = sorting_work + landauer_cost

            # Apply quantum tunneling if applicable
            if (
                packet.coherence_score > self.quantum_coherence_threshold
                and portal.quantum_coherence > self.quantum_coherence_threshold
            ):
                # Quantum tunneling reduces work requirement
                tunnel_efficiency = portal.tunnel_probability
                total_work_packet *= 1.0 - tunnel_efficiency * 0.3

            # Route packet through portal
            packet.portal_address = portal.portal_id
            packet.sort_priority = sorted_count

            # Update portal load
            portal.current_load += 1

            # Add to sorted information
            self.sorted_information[portal.portal_type].append(packet)

            # Update totals
            sorted_count += 1
            total_bits += packet_bits
            total_landauer_cost += landauer_cost
            total_work += total_work_packet

        # Calculate final entropy after sorting
        final_entropy = sum(
            p.entropy_content * portal.portal_efficiency
            for p in packets
            if (portal := self.portals.get(p.portal_address)) is not None
        )

        entropy_reduction = initial_entropy - final_entropy
        information_gain = (
            entropy_reduction * self.boltzmann_constant * self.temperature
        )

        # Calculate sorting efficiency
        theoretical_min_work = (
            total_bits * self.boltzmann_constant * self.temperature * math.log(2)
        )
        sorting_efficiency = theoretical_min_work / max(total_work, 0.001)

        # Energy conservation check
        energy_in = total_work
        energy_out = information_gain + total_landauer_cost
        conservation_error = abs(energy_in - energy_out) / max(energy_in, 0.001)

        # Create operation record
        operation = SortingOperation(
            operation_id=operation_id
            packets_sorted=sorted_count
            bits_processed=total_bits
            landauer_cost=total_landauer_cost
            work_performed=total_work
            entropy_reduction=entropy_reduction
            information_gain=information_gain
            sorting_efficiency=sorting_efficiency
            energy_conservation_error=conservation_error
            operation_duration=time.time() - start_time
            strategy_used=strategy
        )

        # Update demon statistics
        self.operations_completed += 1
        self.total_bits_processed += total_bits
        self.total_landauer_cost += total_landauer_cost
        self.total_entropy_reduction += entropy_reduction
        self.operation_history.append(operation)

        logger.info(
            f"👹 Sorting complete: {sorted_count} packets, efficiency={sorting_efficiency:.3f}"
        )

        return operation

    def get_portal_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all quantum portals"""
        portal_status = {}

        for portal_id, portal in self.portals.items():
            utilization = portal.current_load / portal.capacity
            throughput = len(self.sorted_information.get(portal.portal_type, []))

            portal_status[portal_id] = {
                "portal_type": portal.portal_type
                "capacity_utilization": utilization
                "current_load": portal.current_load
                "total_capacity": portal.capacity
                "throughput": throughput
                "energy_cost_per_bit": portal.energy_cost_per_bit
                "portal_efficiency": portal.portal_efficiency
                "quantum_coherence": portal.quantum_coherence
                "tunnel_probability": portal.tunnel_probability
                "operational_status": (
                    "active" if utilization < 0.9 else "near_capacity"
                ),
            }

        return portal_status

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics for the Maxwell Demon"""
        if self.operations_completed == 0:
            return {"error": "No operations completed yet"}

        avg_efficiency = np.mean(
            [op.sorting_efficiency for op in self.operation_history]
        )
        avg_landauer_cost = self.total_landauer_cost / self.operations_completed
        avg_entropy_reduction = self.total_entropy_reduction / self.operations_completed

        # Calculate theoretical efficiency limit
        theoretical_limit = self.landauer_efficiency
        efficiency_ratio = avg_efficiency / theoretical_limit

        return {
            "operations_completed": self.operations_completed
            "total_bits_processed": self.total_bits_processed
            "total_landauer_cost": self.total_landauer_cost
            "total_entropy_reduction": self.total_entropy_reduction
            "average_efficiency": avg_efficiency
            "average_landauer_cost": avg_landauer_cost
            "average_entropy_reduction": avg_entropy_reduction
            "theoretical_efficiency_limit": theoretical_limit
            "efficiency_ratio": efficiency_ratio
            "performance_rating": min(efficiency_ratio, 1.0),
            "information_packets_sorted": sum(
                len(packets) for packets in self.sorted_information.values()
            ),
            "portal_utilization": {
                pid: p.current_load / p.capacity for pid, p in self.portals.items()
            },
        }

    def reset_portals(self):
        """Reset all portal loads and sorted information"""
        for portal in self.portals.values():
            portal.current_load = 0

        self.sorted_information.clear()
        logger.info("👹 Portal loads reset")

    async def shutdown(self):
        """Shutdown the Maxwell Demon gracefully"""
        try:
            logger.info("👹 Portal Maxwell Demon shutting down...")

            # Reset all portals
            self.reset_portals()

            # Clear operation history
            self.operation_history.clear()

            # Reset statistics
            self.operations_completed = 0
            self.total_bits_processed = 0.0
            self.total_landauer_cost = 0.0
            self.total_entropy_reduction = 0.0

            logger.info("✅ Portal Maxwell Demon shutdown complete")

        except Exception as e:
            logger.error(f"❌ Error during Maxwell Demon shutdown: {e}")
