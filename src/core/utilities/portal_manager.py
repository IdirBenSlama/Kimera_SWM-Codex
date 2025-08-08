"""
Portal Manager
==============
Manages interdimensional portals for cognitive state transitions.

This module implements the portal mechanics that allow Kimera to:
- Create portals between different cognitive dimensions
- Enable state transitions across semantic spaces
- Maintain portal stability and coherence
- Track dimensional crossings and transformations
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import torch

from ..config.settings import get_settings
from ..utils.robust_config import get_api_settings

logger = logging.getLogger(__name__)


class PortalType(Enum):
    """Types of interdimensional portals"""

    SEMANTIC = "semantic"  # Between semantic spaces
    TEMPORAL = "temporal"  # Between time states
    QUANTUM = "quantum"  # Between quantum states
    THERMODYNAMIC = "thermodynamic"  # Between energy states
    COGNITIVE = "cognitive"  # Between cognitive states


class PortalStability(Enum):
    """Portal stability classifications"""

    STABLE = "stable"
    UNSTABLE = "unstable"
    CRITICAL = "critical"
    COLLAPSING = "collapsing"
    CLOSED = "closed"


@dataclass
class Portal:
    """Auto-generated class."""
    pass
    """Represents an interdimensional portal"""

    portal_id: str
    portal_type: PortalType
    source_dimension: int
    target_dimension: int
    stability: float  # 0-1, where 1 is perfectly stable
    energy_cost: float
    throughput: float  # Information flow rate
    creation_time: datetime = field(default_factory=datetime.now)
    last_transit: Optional[datetime] = None
    transit_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def stability_status(self) -> PortalStability:
        """Get stability classification"""
        if self.stability >= 0.8:
            return PortalStability.STABLE
        elif self.stability >= 0.5:
            return PortalStability.UNSTABLE
        elif self.stability >= 0.2:
            return PortalStability.CRITICAL
        elif self.stability > 0:
            return PortalStability.COLLAPSING
        else:
            return PortalStability.CLOSED

    @property
    def is_traversable(self) -> bool:
        """Check if portal can be traversed"""
        return self.stability > 0.2 and self.stability_status != PortalStability.CLOSED


@dataclass
class PortalNetwork:
    """Auto-generated class."""
    pass
    """Network of interconnected portals"""

    graph: nx.DiGraph
    portals: Dict[str, Portal]
    dimension_map: Dict[int, List[str]]  # dimension -> portal_ids

    def add_portal(self, portal: Portal):
        """Add portal to network"""
        self.portals[portal.portal_id] = portal
        self.graph.add_edge(
            portal.source_dimension
            portal.target_dimension
            portal_id=portal.portal_id
            weight=1.0 / (portal.energy_cost + 1e-6),
        )

        # Update dimension map
        if portal.source_dimension not in self.dimension_map:
            self.dimension_map[portal.source_dimension] = []
        self.dimension_map[portal.source_dimension].append(portal.portal_id)

    def find_path(self, source_dim: int, target_dim: int) -> Optional[List[str]]:
        """Find optimal path through portals"""
        try:
            path = nx.shortest_path(self.graph, source_dim, target_dim, weight="weight")

            # Convert dimension path to portal path
            portal_path = []
            for i in range(len(path) - 1):
                edge_data = self.graph.get_edge_data(path[i], path[i + 1])
                if edge_data:
                    portal_path.append(edge_data["portal_id"])

            return portal_path
        except nx.NetworkXNoPath:
            return None
class PortalManager:
    """Auto-generated class."""
    pass
    """
    Manages creation, maintenance, and traversal of interdimensional portals
    """

    def __init__(self, max_portals: int = 100, stability_threshold: float = 0.2):
        self.settings = get_api_settings()
        logger.debug(f"   Environment: {self.settings.environment}")
        self.max_portals = max_portals
        self.stability_threshold = stability_threshold
        self.portals: Dict[str, Portal] = {}
        self.network = PortalNetwork(graph=nx.DiGraph(), portals={}, dimension_map={})

        # Portal dynamics parameters
        self.decay_rate = 0.01  # Stability decay per time unit
        self.reinforcement_rate = 0.1  # Stability gain per transit
        self.energy_scale = 1.0

        # Metrics
        self.total_portals_created = 0
        self.total_transits = 0
        self.total_collapses = 0

        logger.info(f"Portal Manager initialized with max_portals={max_portals}")

    def create_portal(
        self
        source_dim: int
        target_dim: int
        portal_type: PortalType = PortalType.COGNITIVE
        initial_stability: float = 0.8
        energy_cost: Optional[float] = None
    ) -> str:
        """
        Create a new interdimensional portal

        Args:
            source_dim: Source dimension index
            target_dim: Target dimension index
            portal_type: Type of portal
            initial_stability: Initial stability (0-1)
            energy_cost: Energy required for transit

        Returns:
            Portal ID
        """
        # Check portal limit
        if len(self.portals) >= self.max_portals:
            self._cleanup_unstable_portals()

        # Calculate energy cost if not provided
        if energy_cost is None:
            dimension_distance = abs(target_dim - source_dim)
            energy_cost = self.energy_scale * np.log1p(dimension_distance)

        # Create portal
        portal_id = f"portal_{uuid.uuid4().hex[:8]}"
        portal = Portal(
            portal_id=portal_id
            portal_type=portal_type
            source_dimension=source_dim
            target_dimension=target_dim
            stability=initial_stability
            energy_cost=energy_cost
            throughput=self._calculate_throughput(initial_stability, energy_cost),
        )

        # Add to registry and network
        self.portals[portal_id] = portal
        self.network.add_portal(portal)
        self.total_portals_created += 1

        logger.info(
            f"Created {portal_type.value} portal {portal_id}: "
            f"dim {source_dim} -> {target_dim}, stability={initial_stability:.2f}"
        )

        return portal_id

    def traverse_portal(
        self, portal_id: str, cargo: Any, energy_available: float
    ) -> Tuple[bool, Any, Dict[str, Any]]:
        """
        Traverse a portal with cargo

        Args:
            portal_id: Portal to traverse
            cargo: Data/state to transport
            energy_available: Available energy for transit

        Returns:
            Tuple of (success, transformed_cargo, transit_info)
        """
        if portal_id not in self.portals:
            return False, cargo, {"error": "Portal not found"}

        portal = self.portals[portal_id]

        # Check traversability
        if not portal.is_traversable:
            return (
                False
                cargo
                {"error": f"Portal not traversable (stability={portal.stability:.2f})"},
            )

        # Check energy
        if energy_available < portal.energy_cost:
            return (
                False
                cargo
                {
                    "error": f"Insufficient energy (need {portal.energy_cost:.2f}, "
                    f"have {energy_available:.2f})"
                },
            )

        # Perform transit
        transformed_cargo = self._transform_cargo(cargo, portal)

        # Update portal state
        portal.last_transit = datetime.now()
        portal.transit_count += 1
        portal.stability = min(1.0, portal.stability + self.reinforcement_rate)

        # Update metrics
        self.total_transits += 1

        transit_info = {
            "portal_id": portal_id
            "source_dim": portal.source_dimension
            "target_dim": portal.target_dimension
            "energy_consumed": portal.energy_cost
            "stability_after": portal.stability
            "throughput": portal.throughput
            "transform_applied": True
        }

        logger.info(f"Successful transit through {portal_id}")

        return True, transformed_cargo, transit_info

    def _transform_cargo(self, cargo: Any, portal: Portal) -> Any:
        """
        Transform cargo during dimensional transit

        This applies dimensional transformation based on portal type
        """
        if portal.portal_type == PortalType.SEMANTIC:
            # Semantic transformation
            if hasattr(cargo, "embedding"):
                # Rotate embedding in semantic space
                angle = (
                    (portal.target_dimension - portal.source_dimension) * np.pi / 180
                )
                rotation = self._create_rotation_matrix(len(cargo.embedding), angle)
                transformed_embedding = rotation @ cargo.embedding
                cargo.embedding = transformed_embedding

        elif portal.portal_type == PortalType.QUANTUM:
            # Quantum state transformation
            if hasattr(cargo, "state_vector"):
                # Apply phase shift
                phase = (portal.target_dimension - portal.source_dimension) * np.pi / 4
                cargo.state_vector *= np.exp(1j * phase)

        elif portal.portal_type == PortalType.THERMODYNAMIC:
            # Energy state transformation
            if hasattr(cargo, "energy"):
                # Scale energy based on dimension ratio
                energy_scale = (portal.target_dimension + 1) / (
                    portal.source_dimension + 1
                )
                cargo.energy *= energy_scale

        elif portal.portal_type == PortalType.TEMPORAL:
            # Temporal transformation
            if hasattr(cargo, "timestamp"):
                # Shift timestamp (simplified)
                time_shift = (
                    portal.target_dimension - portal.source_dimension
                ) * 3600  # hours
                cargo.timestamp += time_shift

        return cargo

    def _create_rotation_matrix(self, dim: int, angle: float) -> np.ndarray:
        """Create rotation matrix for dimensional transformation"""
        # Simple 2D rotation extended to higher dimensions
        rotation = np.eye(dim)
        if dim >= 2:
            rotation[0, 0] = np.cos(angle)
            rotation[0, 1] = -np.sin(angle)
            rotation[1, 0] = np.sin(angle)
            rotation[1, 1] = np.cos(angle)
        return rotation

    def _calculate_throughput(self, stability: float, energy_cost: float) -> float:
        """Calculate information throughput of portal"""
        # Throughput inversely proportional to energy cost, proportional to stability
        return stability / (energy_cost + 1.0)

    def update_portal_dynamics(self, dt: float = 1.0):
        """
        Update portal stability over time

        Args:
            dt: Time step
        """
        collapsed_portals = []

        for portal_id, portal in self.portals.items():
            # Natural decay
            portal.stability -= self.decay_rate * dt

            # Check for collapse
            if portal.stability <= 0:
                portal.stability = 0
                collapsed_portals.append(portal_id)

        # Remove collapsed portals
        for portal_id in collapsed_portals:
            self._collapse_portal(portal_id)

    def _collapse_portal(self, portal_id: str):
        """Handle portal collapse"""
        if portal_id in self.portals:
            portal = self.portals[portal_id]

            # Remove from network
            self.network.graph.remove_edge(
                portal.source_dimension, portal.target_dimension
            )

            # Remove from registries
            del self.portals[portal_id]
            del self.network.portals[portal_id]

            # Update dimension map
            if portal.source_dimension in self.network.dimension_map:
                self.network.dimension_map[portal.source_dimension].remove(portal_id)

            self.total_collapses += 1
            logger.info(f"Portal {portal_id} collapsed")

    def _cleanup_unstable_portals(self):
        """Remove most unstable portals to make room"""
        # Sort by stability
        sorted_portals = sorted(self.portals.items(), key=lambda x: x[1].stability)

        # Remove bottom 10%
        num_to_remove = max(1, len(sorted_portals) // 10)
        for portal_id, _ in sorted_portals[:num_to_remove]:
            self._collapse_portal(portal_id)

    def find_portal_path(self, source_dim: int, target_dim: int) -> Optional[List[str]]:
        """
        Find optimal path through portal network

        Args:
            source_dim: Starting dimension
            target_dim: Target dimension

        Returns:
            List of portal IDs to traverse, or None if no path exists
        """
        return self.network.find_path(source_dim, target_dim)

    def get_portal_status(self, portal_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed portal status"""
        if portal_id not in self.portals:
            return None

        portal = self.portals[portal_id]
        return {
            "portal_id": portal_id
            "type": portal.portal_type.value
            "source_dimension": portal.source_dimension
            "target_dimension": portal.target_dimension
            "stability": portal.stability
            "stability_status": portal.stability_status.value
            "energy_cost": portal.energy_cost
            "throughput": portal.throughput
            "transit_count": portal.transit_count
            "is_traversable": portal.is_traversable
            "creation_time": portal.creation_time.isoformat(),
            "last_transit": (
                portal.last_transit.isoformat() if portal.last_transit else None
            ),
        }

    def get_manager_metrics(self) -> Dict[str, Any]:
        """Get portal manager metrics"""
        active_portals = [p for p in self.portals.values() if p.is_traversable]

        return {
            "total_portals": len(self.portals),
            "active_portals": len(active_portals),
            "total_created": self.total_portals_created
            "total_transits": self.total_transits
            "total_collapses": self.total_collapses
            "dimensions_connected": len(self.network.graph.nodes()),
            "portal_types": {
                ptype.value: sum(
                    1 for p in self.portals.values() if p.portal_type == ptype
                )
                for ptype in PortalType
            },
            "average_stability": (
                np.mean([p.stability for p in self.portals.values()])
                if self.portals
                else 0.0
            ),
        }


def create_portal_manager(max_portals: int = 100) -> PortalManager:
    """Factory function to create portal manager"""
    return PortalManager(max_portals=max_portals)
