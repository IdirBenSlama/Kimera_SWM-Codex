"""
KIMERA SWM - GEOID REGISTRY
===========================

The GeoidRegistry serves as the central repository for all geoids in the system.
It provides indexing, search, relationship management, and lifecycle tracking
for geoids across all processing domains.

This is the system's memory and knowledge management component.
"""

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Callable, Any, Tuple
import logging

import uuid

from ..data_structures.geoid_state import (
    GeoidState, GeoidType, GeoidProcessingState
)


# Configure logging
logger = logging.getLogger(__name__)


class IndexType(Enum):
    """Types of indexes maintained by the registry"""
    TYPE = "type"                    # Index by geoid type
    STATE = "state"                  # Index by processing state
    ENERGY = "energy"                # Index by energy level
    COHERENCE = "coherence"          # Index by coherence score
    TIMESTAMP = "timestamp"          # Index by creation time
    ENGINE = "engine"                # Index by source engine
    RELATIONSHIPS = "relationships"   # Index by connected geoids


@dataclass
class RegistryStatistics:
    """Statistics about the geoid registry"""
    total_geoids: int
    geoids_by_type: Dict[GeoidType, int]
    geoids_by_state: Dict[GeoidProcessingState, int]
    average_coherence: float
    average_energy: float
    total_relationships: int
    most_connected_geoid: Optional[str]
    oldest_geoid: Optional[str]
    newest_geoid: Optional[str]
    processing_depth_distribution: Dict[int, int]


@dataclass
class SearchQuery:
    """Query specification for geoid search"""
    geoid_types: Optional[List[GeoidType]] = None
    processing_states: Optional[List[GeoidProcessingState]] = None
    min_coherence: Optional[float] = None
    max_coherence: Optional[float] = None
    min_energy: Optional[float] = None
    max_energy: Optional[float] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    source_engines: Optional[List[str]] = None
    has_semantic_state: Optional[bool] = None
    has_symbolic_state: Optional[bool] = None
    has_thermodynamic_state: Optional[bool] = None
    limit: Optional[int] = None


class GeoidRegistry:
    """
    Central Geoid Registry - System Knowledge Repository
    ===================================================
    
    The GeoidRegistry manages all geoids in the Kimera SWM system, providing:
    - Centralized storage and indexing
    - Advanced search and query capabilities
    - Relationship mapping and analysis
    - Lifecycle management and cleanup
    - Performance monitoring and optimization
    
    This serves as the system's collective memory and enables sophisticated
    knowledge management across all domains and engines.
    """
    
    def __init__(self, max_geoids: int = 1000000, enable_auto_cleanup: bool = True):
        self.max_geoids = max_geoids
        self.enable_auto_cleanup = enable_auto_cleanup
        
        # Core storage
        self._geoids: Dict[str, GeoidState] = {}
        
        # Indexes for fast lookup
        self._indexes: Dict[IndexType, Dict[Any, Set[str]]] = {
            IndexType.TYPE: defaultdict(set),
            IndexType.STATE: defaultdict(set),
            IndexType.ENERGY: defaultdict(set),
            IndexType.COHERENCE: defaultdict(set),
            IndexType.TIMESTAMP: defaultdict(set),
            IndexType.ENGINE: defaultdict(set),
            IndexType.RELATIONSHIPS: defaultdict(set)
        }
        
        # Relationship tracking
        self._relationships: Dict[str, Set[str]] = defaultdict(set)
        self._reverse_relationships: Dict[str, Set[str]] = defaultdict(set)
        
        # Performance tracking
        self._operation_count = 0
        self._last_cleanup = datetime.now()
        self._registry_metrics = {
            'registrations': 0,
            'removals': 0,
            'searches': 0,
            'updates': 0,
            'cleanup_runs': 0
        }
        
        logger.info(f"GeoidRegistry initialized with max_geoids={max_geoids}, auto_cleanup={enable_auto_cleanup}")
    
    def register(self, geoid: GeoidState) -> bool:
        """
        Register a new geoid in the registry.
        Returns True if successful, False if already exists or capacity exceeded.
        """
        if geoid.geoid_id in self._geoids:
            logger.warning(f"Geoid {geoid.geoid_id[:8]} already registered")
            return False
        
        # Check capacity
        if len(self._geoids) >= self.max_geoids:
            if self.enable_auto_cleanup:
                self._auto_cleanup()
            else:
                logger.error(f"Registry at capacity ({self.max_geoids}), cannot register new geoid")
                return False
        
        # Register the geoid
        self._geoids[geoid.geoid_id] = geoid
        self._update_indexes(geoid, is_new=True)
        self._update_relationships(geoid)
        
        self._registry_metrics['registrations'] += 1
        self._operation_count += 1
        
        logger.debug(f"Registered geoid {geoid.geoid_id[:8]} (type: {geoid.geoid_type.value})")
        return True
    
    def get(self, geoid_id: str) -> Optional[GeoidState]:
        """Get a geoid by ID"""
        return self._geoids.get(geoid_id)
    
    def exists(self, geoid_id: str) -> bool:
        """Check if a geoid exists in the registry"""
        return geoid_id in self._geoids
    
    def remove(self, geoid_id: str) -> bool:
        """
        Remove a geoid from the registry.
        Returns True if successful, False if not found.
        """
        if geoid_id not in self._geoids:
            logger.warning(f"Cannot remove non-existent geoid {geoid_id[:8]}")
            return False
        
        geoid = self._geoids[geoid_id]
        
        # Remove from indexes
        self._remove_from_indexes(geoid)
        
        # Remove relationships
        self._remove_relationships(geoid_id)
        
        # Remove from storage
        del self._geoids[geoid_id]
        
        self._registry_metrics['removals'] += 1
        self._operation_count += 1
        
        logger.debug(f"Removed geoid {geoid_id[:8]}")
        return True
    
    def update(self, geoid: GeoidState) -> bool:
        """
        Update an existing geoid in the registry.
        Returns True if successful, False if not found.
        """
        if geoid.geoid_id not in self._geoids:
            logger.warning(f"Cannot update non-existent geoid {geoid.geoid_id[:8]}")
            return False
        
        old_geoid = self._geoids[geoid.geoid_id]
        
        # Remove old indexes
        self._remove_from_indexes(old_geoid)
        
        # Update storage
        self._geoids[geoid.geoid_id] = geoid
        
        # Update indexes
        self._update_indexes(geoid, is_new=False)
        self._update_relationships(geoid)
        
        self._registry_metrics['updates'] += 1
        self._operation_count += 1
        
        logger.debug(f"Updated geoid {geoid.geoid_id[:8]}")
        return True
    
    def search(self, query: SearchQuery) -> List[GeoidState]:
        """
        Search for geoids matching the given query.
        Returns a list of matching geoids, limited by query.limit.
        """
        self._registry_metrics['searches'] += 1
        
        # Start with all geoids
        candidates = set(self._geoids.keys())
        
        # Apply filters
        if query.geoid_types:
            type_candidates = set()
            for geoid_type in query.geoid_types:
                type_candidates.update(self._indexes[IndexType.TYPE][geoid_type])
            candidates &= type_candidates
        
        if query.processing_states:
            state_candidates = set()
            for state in query.processing_states:
                state_candidates.update(self._indexes[IndexType.STATE][state])
            candidates &= state_candidates
        
        if query.source_engines:
            engine_candidates = set()
            for engine in query.source_engines:
                engine_candidates.update(self._indexes[IndexType.ENGINE][engine])
            candidates &= engine_candidates
        
        # Apply range filters
        filtered_candidates = []
        for geoid_id in candidates:
            geoid = self._geoids[geoid_id]
            
            # Coherence filter
            if query.min_coherence is not None and geoid.coherence_score < query.min_coherence:
                continue
            if query.max_coherence is not None and geoid.coherence_score > query.max_coherence:
                continue
            
            # Energy filter
            if query.min_energy is not None and geoid.cognitive_energy < query.min_energy:
                continue
            if query.max_energy is not None and geoid.cognitive_energy > query.max_energy:
                continue
            
            # Timestamp filter
            if query.created_after is not None and geoid.metadata.creation_timestamp < query.created_after:
                continue
            if query.created_before is not None and geoid.metadata.creation_timestamp > query.created_before:
                continue
            
            # State presence filters
            if query.has_semantic_state is not None:
                has_semantic = geoid.semantic_state is not None
                if has_semantic != query.has_semantic_state:
                    continue
            
            if query.has_symbolic_state is not None:
                has_symbolic = geoid.symbolic_state is not None
                if has_symbolic != query.has_symbolic_state:
                    continue
            
            if query.has_thermodynamic_state is not None:
                has_thermo = geoid.thermodynamic is not None
                if has_thermo != query.has_thermodynamic_state:
                    continue
            
            filtered_candidates.append(geoid)
        
        # Apply limit
        if query.limit is not None:
            filtered_candidates = filtered_candidates[:query.limit]
        
        logger.debug(f"Search returned {len(filtered_candidates)} geoids")
        return filtered_candidates
    
    def get_related_geoids(self, geoid_id: str, max_distance: int = 1) -> Dict[int, Set[str]]:
        """
        Get geoids related to the given geoid up to max_distance.
        Returns a dictionary mapping distance to sets of geoid IDs.
        """
        if geoid_id not in self._geoids:
            return {}
        
        result = {0: {geoid_id}}
        current_level = {geoid_id}
        
        for distance in range(1, max_distance + 1):
            next_level = set()
            for current_id in current_level:
                # Add direct relationships
                next_level.update(self._relationships[current_id])
                next_level.update(self._reverse_relationships[current_id])
            
            # Remove already seen geoids
            for prev_distance in range(distance):
                next_level -= result[prev_distance]
            
            if not next_level:
                break
            
            result[distance] = next_level
            current_level = next_level
        
        return result
    
    def get_statistics(self) -> RegistryStatistics:
        """Get comprehensive statistics about the registry"""
        if not self._geoids:
            return RegistryStatistics(
                total_geoids=0,
                geoids_by_type={},
                geoids_by_state={},
                average_coherence=0.0,
                average_energy=0.0,
                total_relationships=0,
                most_connected_geoid=None,
                oldest_geoid=None,
                newest_geoid=None,
                processing_depth_distribution={}
            )
        
        # Calculate basic counts
        total_geoids = len(self._geoids)
        
        # Count by type and state
        geoids_by_type = defaultdict(int)
        geoids_by_state = defaultdict(int)
        coherence_sum = 0.0
        energy_sum = 0.0
        processing_depths = defaultdict(int)
        
        oldest_geoid = None
        newest_geoid = None
        oldest_time = None
        newest_time = None
        
        connection_counts = defaultdict(int)
        
        for geoid_id, geoid in self._geoids.items():
            geoids_by_type[geoid.geoid_type] += 1
            geoids_by_state[geoid.processing_state] += 1
            coherence_sum += geoid.coherence_score
            energy_sum += geoid.cognitive_energy
            processing_depths[geoid.metadata.processing_depth] += 1
            
            # Track oldest and newest
            creation_time = geoid.metadata.creation_timestamp
            if oldest_time is None or creation_time < oldest_time:
                oldest_time = creation_time
                oldest_geoid = geoid_id
            if newest_time is None or creation_time > newest_time:
                newest_time = creation_time
                newest_geoid = geoid_id
            
            # Count connections
            connection_count = (
                len(self._relationships[geoid_id]) + 
                len(self._reverse_relationships[geoid_id])
            )
            connection_counts[geoid_id] = connection_count
        
        # Find most connected geoid
        most_connected_geoid = None
        if connection_counts:
            most_connected_geoid = max(connection_counts.items(), key=lambda x: x[1])[0]
        
        # Calculate total relationships
        total_relationships = sum(len(rels) for rels in self._relationships.values())
        
        return RegistryStatistics(
            total_geoids=total_geoids,
            geoids_by_type=dict(geoids_by_type),
            geoids_by_state=dict(geoids_by_state),
            average_coherence=coherence_sum / total_geoids,
            average_energy=energy_sum / total_geoids,
            total_relationships=total_relationships,
            most_connected_geoid=most_connected_geoid,
            oldest_geoid=oldest_geoid,
            newest_geoid=newest_geoid,
            processing_depth_distribution=dict(processing_depths)
        )
    
    def _update_indexes(self, geoid: GeoidState, is_new: bool) -> None:
        """Update all indexes for a geoid"""
        geoid_id = geoid.geoid_id
        
        # Type index
        self._indexes[IndexType.TYPE][geoid.geoid_type].add(geoid_id)
        
        # State index
        self._indexes[IndexType.STATE][geoid.processing_state].add(geoid_id)
        
        # Energy index (bucketed)
        energy_bucket = int(geoid.cognitive_energy // 5) * 5  # 5-unit buckets
        self._indexes[IndexType.ENERGY][energy_bucket].add(geoid_id)
        
        # Coherence index (bucketed)
        coherence_bucket = int(geoid.coherence_score * 10) / 10  # 0.1 buckets
        self._indexes[IndexType.COHERENCE][coherence_bucket].add(geoid_id)
        
        # Timestamp index (daily buckets)
        timestamp_bucket = geoid.metadata.creation_timestamp.date()
        self._indexes[IndexType.TIMESTAMP][timestamp_bucket].add(geoid_id)
        
        # Engine index
        if geoid.metadata.source_engine:
            self._indexes[IndexType.ENGINE][geoid.metadata.source_engine].add(geoid_id)
    
    def _remove_from_indexes(self, geoid: GeoidState) -> None:
        """Remove a geoid from all indexes"""
        geoid_id = geoid.geoid_id
        
        # Remove from all index types
        for index_type, index_dict in self._indexes.items():
            for key, geoid_set in index_dict.items():
                geoid_set.discard(geoid_id)
    
    def _update_relationships(self, geoid: GeoidState) -> None:
        """Update relationship indexes for a geoid"""
        geoid_id = geoid.geoid_id
        
        # Clear existing relationships for this geoid
        old_relationships = self._relationships[geoid_id].copy()
        for related_id in old_relationships:
            self._reverse_relationships[related_id].discard(geoid_id)
        self._relationships[geoid_id].clear()
        
        # Add new relationships from parent geoids
        for parent_id in geoid.metadata.parent_geoids:
            if parent_id in self._geoids:
                self._relationships[geoid_id].add(parent_id)
                self._reverse_relationships[parent_id].add(geoid_id)
        
        # Add new relationships from input connections
        for connection_name, input_geoid in geoid.input_connections.items():
            if input_geoid.geoid_id in self._geoids:
                self._relationships[geoid_id].add(input_geoid.geoid_id)
                self._reverse_relationships[input_geoid.geoid_id].add(geoid_id)
    
    def _remove_relationships(self, geoid_id: str) -> None:
        """Remove all relationships for a geoid"""
        # Remove outgoing relationships
        for related_id in self._relationships[geoid_id]:
            self._reverse_relationships[related_id].discard(geoid_id)
        del self._relationships[geoid_id]
        
        # Remove incoming relationships
        for related_id in self._reverse_relationships[geoid_id]:
            self._relationships[related_id].discard(geoid_id)
        del self._reverse_relationships[geoid_id]
    
    def _auto_cleanup(self) -> None:
        """Automatically clean up old or inactive geoids"""
        if datetime.now() - self._last_cleanup < timedelta(minutes=5):
            return  # Don't cleanup too frequently
        
        # Find candidates for removal
        cleanup_candidates = []
        cutoff_time = datetime.now() - timedelta(hours=24)  # Older than 24 hours
        
        for geoid_id, geoid in self._geoids.items():
            # Remove archived geoids older than 24 hours
            if (geoid.processing_state == GeoidProcessingState.ARCHIVED and 
                geoid.metadata.last_modified < cutoff_time):
                cleanup_candidates.append(geoid_id)
            
            # Remove very low energy geoids with low coherence
            elif (geoid.cognitive_energy < 0.1 and 
                  geoid.coherence_score < 0.1 and
                  geoid.metadata.processing_depth < 2):
                cleanup_candidates.append(geoid_id)
        
        # Remove up to 10% of capacity
        max_to_remove = max(1, self.max_geoids // 10)
        candidates_to_remove = cleanup_candidates[:max_to_remove]
        
        for geoid_id in candidates_to_remove:
            self.remove(geoid_id)
        
        self._registry_metrics['cleanup_runs'] += 1
        self._last_cleanup = datetime.now()
        
        if candidates_to_remove:
            logger.info(f"Auto-cleanup removed {len(candidates_to_remove)} geoids")
    
    def get_registry_metrics(self) -> Dict[str, Any]:
        """Get registry performance metrics"""
        stats = self.get_statistics()
        return {
            **self._registry_metrics,
            'current_size': len(self._geoids),
            'capacity_utilization': len(self._geoids) / self.max_geoids,
            'total_operations': self._operation_count,
            'average_coherence': stats.average_coherence,
            'average_energy': stats.average_energy,
            'index_sizes': {
                index_type.value: sum(len(s) for s in index_dict.values())
                for index_type, index_dict in self._indexes.items()
            }
        }


# Global registry instance
_global_registry: Optional[GeoidRegistry] = None


def get_global_registry() -> GeoidRegistry:
    """Get the global geoid registry instance"""
    global _global_registry
    if _global_registry is None:
        _global_registry = GeoidRegistry()
    return _global_registry


def initialize_registry(max_geoids: int = 1000000, enable_auto_cleanup: bool = True) -> GeoidRegistry:
    """Initialize the global registry with custom parameters"""
    global _global_registry
    _global_registry = GeoidRegistry(max_geoids, enable_auto_cleanup)
    return _global_registry 