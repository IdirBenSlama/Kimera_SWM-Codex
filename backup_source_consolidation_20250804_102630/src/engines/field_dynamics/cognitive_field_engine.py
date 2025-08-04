"""
KIMERA SWM - COGNITIVE FIELD ENGINE
===================================

The Cognitive Field Engine implements field-based processing of geoids,
treating cognitive space as a dynamic field where geoids interact through
field potentials, forces, and emergent behaviors. This enables sophisticated
collective intelligence and emergent cognitive phenomena.

This engine bridges individual geoid processing with collective field dynamics.
"""

import numpy as np
import time
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
import logging
from enum import Enum

from ...core.data_structures.geoid_state import (
    GeoidState, GeoidType, GeoidProcessingState,
    SemanticState, SymbolicState, ThermodynamicProperties
)


# Configure logging
logger = logging.getLogger(__name__)


class FieldType(Enum):
    """Types of cognitive fields"""
    SEMANTIC_FIELD = "semantic_field"           # Semantic similarity field
    SYMBOLIC_FIELD = "symbolic_field"           # Logical relationship field
    ENERGY_FIELD = "energy_field"               # Thermodynamic energy field
    COHERENCE_FIELD = "coherence_field"         # Coherence interaction field
    ATTENTION_FIELD = "attention_field"         # Attention/salience field
    MEMORY_FIELD = "memory_field"               # Memory consolidation field
    CREATIVITY_FIELD = "creativity_field"       # Creative emergence field


class InteractionType(Enum):
    """Types of field interactions"""
    ATTRACTION = "attraction"                   # Attractive interaction
    REPULSION = "repulsion"                     # Repulsive interaction
    RESONANCE = "resonance"                     # Resonant coupling
    INTERFERENCE = "interference"               # Wave interference
    ENTANGLEMENT = "entanglement"               # Quantum entanglement
    EMERGENCE = "emergence"                     # Emergent behavior
    SYNCHRONIZATION = "synchronization"        # Phase synchronization


@dataclass
class FieldParameters:
    """Parameters controlling field dynamics"""
    field_strength: float = 1.0                # Overall field strength
    interaction_range: float = 10.0            # Range of field interactions
    decay_rate: float = 0.1                    # Field decay rate
    coupling_strength: float = 0.5             # Inter-field coupling
    noise_level: float = 0.05                  # Field noise level
    resonance_frequency: float = 1.0           # Natural resonance frequency
    damping_coefficient: float = 0.1           # Damping for stability
    emergence_threshold: float = 0.8           # Threshold for emergent behavior


@dataclass
class FieldState:
    """State of a cognitive field"""
    field_type: FieldType
    geoid_positions: Dict[str, np.ndarray]     # Geoid positions in field space
    field_values: np.ndarray                   # Field values on grid
    field_gradients: np.ndarray                # Field gradients
    energy_density: np.ndarray                 # Energy density distribution
    interaction_matrix: np.ndarray             # Geoid interaction strengths
    temporal_evolution: List[np.ndarray]       # Field evolution history
    emergent_structures: List[Dict[str, Any]]  # Detected emergent structures


@dataclass
class FieldResult:
    """Result of field processing"""
    original_geoids: List[GeoidState]
    processed_geoids: List[GeoidState]
    field_state: FieldState
    emergent_behaviors: List[Dict[str, Any]]
    energy_changes: Dict[str, float]
    interaction_events: List[Dict[str, Any]]
    processing_duration: float
    metadata: Dict[str, Any]


class CognitiveField:
    """
    Individual Cognitive Field - Specific Field Type Handler
    =======================================================
    
    Each CognitiveField manages a specific type of field (semantic, symbolic, etc.)
    and implements the physics and dynamics specific to that field type.
    """
    
    def __init__(self, field_type: FieldType, parameters: FieldParameters,
                 grid_size: Tuple[int, int, int] = (50, 50, 50)):
        self.field_type = field_type
        self.parameters = parameters
        self.grid_size = grid_size
        
        # Initialize field arrays
        self.field_values = np.zeros(grid_size)
        self.field_gradients = np.zeros((*grid_size, 3))  # 3D gradients
        self.energy_density = np.zeros(grid_size)
        self.potential_field = np.zeros(grid_size)
        
        # Geoid tracking
        self.geoids_in_field: Dict[str, GeoidState] = {}
        self.geoid_positions: Dict[str, np.ndarray] = {}
        self.geoid_velocities: Dict[str, np.ndarray] = {}
        
        # Field dynamics
        self.evolution_step = 0
        self.last_update = datetime.now()
        
        # Create coordinate grids
        self.x_grid, self.y_grid, self.z_grid = np.meshgrid(
            np.linspace(-1, 1, grid_size[0]),
            np.linspace(-1, 1, grid_size[1]),
            np.linspace(-1, 1, grid_size[2]),
            indexing='ij'
        )
        
        logger.debug(f"Cognitive field initialized: {field_type.value}")
    
    def add_geoid(self, geoid: GeoidState, position: np.ndarray = None) -> bool:
        """Add a geoid to the field at the specified position"""
        if position is None:
            # Auto-assign position based on geoid properties
            position = self._calculate_natural_position(geoid)
        
        # Validate position
        if not self._is_valid_position(position):
            logger.warning(f"Invalid position for geoid {geoid.geoid_id[:8]}")
            return False
        
        self.geoids_in_field[geoid.geoid_id] = geoid
        self.geoid_positions[geoid.geoid_id] = position
        self.geoid_velocities[geoid.geoid_id] = np.zeros(3)
        
        # Update field
        self._update_field_from_geoid(geoid, position)
        
        logger.debug(f"Added geoid {geoid.geoid_id[:8]} to {self.field_type.value} field")
        return True
    
    def remove_geoid(self, geoid_id: str) -> bool:
        """Remove a geoid from the field"""
        if geoid_id not in self.geoids_in_field:
            return False
        
        del self.geoids_in_field[geoid_id]
        del self.geoid_positions[geoid_id]
        del self.geoid_velocities[geoid_id]
        
        # Recalculate field
        self._recalculate_field()
        
        logger.debug(f"Removed geoid {geoid_id[:8]} from {self.field_type.value} field")
        return True
    
    def evolve_field(self, time_step: float = 1.0) -> FieldState:
        """Evolve the field dynamics for one time step"""
        start_time = time.time()
        
        # Calculate forces on each geoid
        forces = self._calculate_forces()
        
        # Update geoid positions and velocities
        self._update_geoid_dynamics(forces, time_step)
        
        # Update field values
        self._update_field_dynamics(time_step)
        
        # Detect emergent structures
        emergent_structures = self._detect_emergent_structures()
        
        # Create field state snapshot
        field_state = FieldState(
            field_type=self.field_type,
            geoid_positions=self.geoid_positions.copy(),
            field_values=self.field_values.copy(),
            field_gradients=self.field_gradients.copy(),
            energy_density=self.energy_density.copy(),
            interaction_matrix=self._calculate_interaction_matrix(),
            temporal_evolution=[],  # Would store recent history
            emergent_structures=emergent_structures
        )
        
        self.evolution_step += 1
        self.last_update = datetime.now()
        
        logger.debug(f"Field evolution step {self.evolution_step} completed in {time.time() - start_time:.3f}s")
        return field_state
    
    def _calculate_natural_position(self, geoid: GeoidState) -> np.ndarray:
        """Calculate natural position for a geoid based on its properties"""
        if self.field_type == FieldType.SEMANTIC_FIELD and geoid.semantic_state:
            # Use PCA of embedding vector to determine 3D position
            embedding = geoid.semantic_state.embedding_vector
            # Simple projection to 3D (in practice would use proper dimensionality reduction)
            position = np.array([
                np.mean(embedding[:256]),      # X dimension
                np.mean(embedding[256:512]),   # Y dimension  
                np.mean(embedding[512:768]) if len(embedding) > 512 else 0.0  # Z dimension
            ])
            # Normalize to field bounds [-1, 1]
            return np.clip(position, -1, 1)
            
        elif self.field_type == FieldType.ENERGY_FIELD:
            # Position based on energy level
            energy = geoid.cognitive_energy
            # Create 3D position from energy and coherence
            return np.array([
                (energy - 5.0) / 10.0,  # Normalized energy
                (geoid.coherence_score - 0.5) * 2,  # Coherence
                np.random.uniform(-0.5, 0.5)  # Random Z
            ])
            
        elif self.field_type == FieldType.COHERENCE_FIELD:
            # Position based on coherence score
            coherence = geoid.coherence_score
            return np.array([
                (coherence - 0.5) * 2,  # Primary axis
                np.random.uniform(-0.3, 0.3),  # Small variation
                np.random.uniform(-0.3, 0.3)   # Small variation
            ])
            
        else:
            # Default: random position
            return np.random.uniform(-0.8, 0.8, 3)
    
    def _is_valid_position(self, position: np.ndarray) -> bool:
        """Check if position is valid within field bounds"""
        return np.all(np.abs(position) <= 1.0) and len(position) == 3
    
    def _update_field_from_geoid(self, geoid: GeoidState, position: np.ndarray) -> None:
        """Update field values based on a geoid's presence"""
        # Convert position to grid indices
        grid_pos = self._position_to_grid_indices(position)
        
        # Calculate influence based on geoid properties
        influence = self._calculate_geoid_influence(geoid)
        
        # Apply Gaussian influence around position
        for i in range(max(0, grid_pos[0] - 5), min(self.grid_size[0], grid_pos[0] + 6)):
            for j in range(max(0, grid_pos[1] - 5), min(self.grid_size[1], grid_pos[1] + 6)):
                for k in range(max(0, grid_pos[2] - 5), min(self.grid_size[2], grid_pos[2] + 6)):
                    distance = np.sqrt((i - grid_pos[0])**2 + (j - grid_pos[1])**2 + (k - grid_pos[2])**2)
                    weight = np.exp(-distance**2 / (2 * 2.0**2))  # Gaussian with sigma=2
                    self.field_values[i, j, k] += influence * weight
    
    def _calculate_geoid_influence(self, geoid: GeoidState) -> float:
        """Calculate how strongly a geoid influences the field"""
        base_influence = 1.0
        
        if self.field_type == FieldType.SEMANTIC_FIELD and geoid.semantic_state:
            # Influence based on semantic coherence
            return base_influence * geoid.semantic_state.coherence_score
            
        elif self.field_type == FieldType.ENERGY_FIELD:
            # Influence based on cognitive energy
            return base_influence * (geoid.cognitive_energy / 10.0)
            
        elif self.field_type == FieldType.COHERENCE_FIELD:
            # Influence based on overall coherence
            return base_influence * geoid.coherence_score
            
        else:
            return base_influence
    
    def _position_to_grid_indices(self, position: np.ndarray) -> Tuple[int, int, int]:
        """Convert continuous position to grid indices"""
        # Map [-1, 1] to [0, grid_size-1]
        indices = ((position + 1) / 2 * (np.array(self.grid_size) - 1)).astype(int)
        return tuple(np.clip(indices, 0, np.array(self.grid_size) - 1))
    
    def _calculate_forces(self) -> Dict[str, np.ndarray]:
        """Calculate forces acting on each geoid in the field"""
        forces = {}
        
        for geoid_id, position in self.geoid_positions.items():
            # Calculate gradient-based force (field acts to move geoids along gradients)
            gradient_force = self._interpolate_gradient_at_position(position)
            
            # Calculate inter-geoid forces
            interaction_force = self._calculate_interaction_forces(geoid_id)
            
            # Combine forces
            total_force = gradient_force + interaction_force
            
            # Apply damping
            velocity = self.geoid_velocities[geoid_id]
            damping_force = -self.parameters.damping_coefficient * velocity
            
            forces[geoid_id] = total_force + damping_force
        
        return forces
    
    def _interpolate_gradient_at_position(self, position: np.ndarray) -> np.ndarray:
        """Interpolate field gradient at a specific position"""
        # Convert position to grid coordinates
        grid_coords = (position + 1) / 2 * (np.array(self.grid_size) - 1)
        
        # Simple trilinear interpolation (simplified for clarity)
        i, j, k = grid_coords.astype(int)
        i = np.clip(i, 0, self.grid_size[0] - 2)
        j = np.clip(j, 0, self.grid_size[1] - 2)
        k = np.clip(k, 0, self.grid_size[2] - 2)
        
        # Return gradient at nearest grid point (simplified)
        return self.field_gradients[i, j, k] * self.parameters.field_strength
    
    def _calculate_interaction_forces(self, geoid_id: str) -> np.ndarray:
        """Calculate forces from interactions with other geoids"""
        current_geoid = self.geoids_in_field[geoid_id]
        current_position = self.geoid_positions[geoid_id]
        total_force = np.zeros(3)
        
        for other_id, other_position in self.geoid_positions.items():
            if other_id == geoid_id:
                continue
            
            other_geoid = self.geoids_in_field[other_id]
            
            # Calculate distance and direction
            displacement = other_position - current_position
            distance = np.linalg.norm(displacement)
            
            if distance == 0:
                continue
            
            direction = displacement / distance
            
            # Calculate interaction strength
            interaction_strength = self._calculate_interaction_strength(
                current_geoid, other_geoid, distance
            )
            
            # Apply force (positive = attraction, negative = repulsion)
            force_magnitude = interaction_strength / (distance**2 + 0.1)  # Avoid singularity
            total_force += force_magnitude * direction
        
        return total_force * self.parameters.coupling_strength
    
    def _calculate_interaction_strength(self, geoid1: GeoidState, geoid2: GeoidState, distance: float) -> float:
        """Calculate interaction strength between two geoids"""
        if distance > self.parameters.interaction_range:
            return 0.0
        
        base_strength = 1.0
        
        if self.field_type == FieldType.SEMANTIC_FIELD:
            # Semantic similarity creates attraction
            if geoid1.semantic_state and geoid2.semantic_state:
                similarity = np.dot(
                    geoid1.semantic_state.embedding_vector,
                    geoid2.semantic_state.embedding_vector
                ) / (np.linalg.norm(geoid1.semantic_state.embedding_vector) *
                     np.linalg.norm(geoid2.semantic_state.embedding_vector))
                return base_strength * similarity
                
        elif self.field_type == FieldType.ENERGY_FIELD:
            # Energy difference creates flow
            energy_diff = abs(geoid1.cognitive_energy - geoid2.cognitive_energy)
            return base_strength * np.exp(-energy_diff / 5.0)  # Exponential decay
            
        elif self.field_type == FieldType.COHERENCE_FIELD:
            # Coherence similarity creates resonance
            coherence_similarity = 1.0 - abs(geoid1.coherence_score - geoid2.coherence_score)
            return base_strength * coherence_similarity
        
        return base_strength * 0.1  # Default weak interaction
    
    def _update_geoid_dynamics(self, forces: Dict[str, np.ndarray], time_step: float) -> None:
        """Update geoid positions and velocities based on forces"""
        for geoid_id, force in forces.items():
            geoid = self.geoids_in_field[geoid_id]
            
            # Simple mass model (could be based on geoid properties)
            mass = 1.0 + geoid.cognitive_energy / 10.0
            
            # Update velocity: v = v + (F/m) * dt
            acceleration = force / mass
            self.geoid_velocities[geoid_id] += acceleration * time_step
            
            # Update position: x = x + v * dt
            self.geoid_positions[geoid_id] += self.geoid_velocities[geoid_id] * time_step
            
            # Keep within bounds
            self.geoid_positions[geoid_id] = np.clip(self.geoid_positions[geoid_id], -1, 1)
    
    def _update_field_dynamics(self, time_step: float) -> None:
        """Update field values based on field dynamics"""
        # Calculate field gradients
        self.field_gradients = np.gradient(self.field_values)
        
        # Update energy density
        self.energy_density = 0.5 * np.sum(self.field_gradients**2, axis=0)
        
        # Apply field decay
        self.field_values *= (1 - self.parameters.decay_rate * time_step)
        
        # Add noise
        if self.parameters.noise_level > 0:
            noise = np.random.normal(0, self.parameters.noise_level, self.field_values.shape)
            self.field_values += noise * time_step
        
        # Recalculate field from current geoid positions
        self._recalculate_field()
    
    def _recalculate_field(self) -> None:
        """Recalculate field values from all geoids"""
        self.field_values.fill(0)
        
        for geoid_id, position in self.geoid_positions.items():
            geoid = self.geoids_in_field[geoid_id]
            self._update_field_from_geoid(geoid, position)
    
    def _calculate_interaction_matrix(self) -> np.ndarray:
        """Calculate interaction matrix between all geoids"""
        n_geoids = len(self.geoids_in_field)
        matrix = np.zeros((n_geoids, n_geoids))
        
        geoid_list = list(self.geoids_in_field.items())
        
        for i, (id1, geoid1) in enumerate(geoid_list):
            for j, (id2, geoid2) in enumerate(geoid_list):
                if i != j:
                    distance = np.linalg.norm(
                        self.geoid_positions[id1] - self.geoid_positions[id2]
                    )
                    matrix[i, j] = self._calculate_interaction_strength(geoid1, geoid2, distance)
        
        return matrix
    
    def _detect_emergent_structures(self) -> List[Dict[str, Any]]:
        """Detect emergent structures in the field"""
        structures = []
        
        # Detect clusters of geoids
        clusters = self._detect_geoid_clusters()
        for cluster in clusters:
            structures.append({
                'type': 'geoid_cluster',
                'geoids': cluster,
                'strength': len(cluster) / len(self.geoids_in_field),
                'center': np.mean([self.geoid_positions[gid] for gid in cluster], axis=0)
            })
        
        # Detect field maxima
        maxima = self._detect_field_maxima()
        for maximum in maxima:
            structures.append({
                'type': 'field_maximum',
                'position': maximum['position'],
                'strength': maximum['value'],
                'radius': maximum['radius']
            })
        
        return structures
    
    def _detect_geoid_clusters(self) -> List[List[str]]:
        """Detect clusters of nearby geoids"""
        clusters = []
        visited = set()
        
        for geoid_id in self.geoid_positions:
            if geoid_id in visited:
                continue
            
            cluster = self._find_cluster_from_geoid(geoid_id, visited)
            if len(cluster) >= 2:  # Minimum cluster size
                clusters.append(cluster)
        
        return clusters
    
    def _find_cluster_from_geoid(self, start_geoid: str, visited: set) -> List[str]:
        """Find cluster starting from a specific geoid using DFS"""
        cluster = []
        stack = [start_geoid]
        cluster_threshold = 0.3  # Maximum distance for cluster membership
        
        while stack:
            current_id = stack.pop()
            if current_id in visited:
                continue
            
            visited.add(current_id)
            cluster.append(current_id)
            
            # Find nearby geoids
            current_pos = self.geoid_positions[current_id]
            for other_id, other_pos in self.geoid_positions.items():
                if other_id not in visited:
                    distance = np.linalg.norm(current_pos - other_pos)
                    if distance < cluster_threshold:
                        stack.append(other_id)
        
        return cluster
    
    def _detect_field_maxima(self) -> List[Dict[str, Any]]:
        """Detect local maxima in the field"""
        maxima = []
        
        # Simple peak detection (3x3x3 local maxima)
        for i in range(1, self.grid_size[0] - 1):
            for j in range(1, self.grid_size[1] - 1):
                for k in range(1, self.grid_size[2] - 1):
                    center_value = self.field_values[i, j, k]
                    
                    # Check if this is a local maximum
                    is_maximum = True
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            for dk in [-1, 0, 1]:
                                if di == 0 and dj == 0 and dk == 0:
                                    continue
                                if self.field_values[i+di, j+dj, k+dk] >= center_value:
                                    is_maximum = False
                                    break
                            if not is_maximum:
                                break
                        if not is_maximum:
                            break
                    
                    if is_maximum and center_value > 0.1:  # Threshold for significance
                        # Convert grid indices back to position
                        position = np.array([i, j, k]) / (np.array(self.grid_size) - 1) * 2 - 1
                        maxima.append({
                            'position': position,
                            'value': center_value,
                            'radius': 0.1  # Simplified radius
                        })
        
        return maxima


class CognitiveFieldEngine:
    """
    Cognitive Field Engine - Multi-Field Dynamics Coordinator
    =========================================================
    
    The CognitiveFieldEngine manages multiple cognitive fields and coordinates
    their interactions to create sophisticated collective intelligence behaviors.
    It enables emergent phenomena through field coupling and resonance.
    """
    
    def __init__(self, field_parameters: FieldParameters = None):
        self.field_parameters = field_parameters or FieldParameters()
        self.fields: Dict[FieldType, CognitiveField] = {}
        self.field_coupling_matrix = np.eye(len(FieldType))  # Identity matrix (no coupling initially)
        self.processing_history: List[FieldResult] = []
        self.total_field_operations = 0
        
        # Initialize default fields
        self._initialize_default_fields()
        
        logger.info("CognitiveFieldEngine initialized with default fields")
    
    def _initialize_default_fields(self) -> None:
        """Initialize default cognitive fields"""
        default_field_types = [
            FieldType.SEMANTIC_FIELD,
            FieldType.ENERGY_FIELD,
            FieldType.COHERENCE_FIELD,
            FieldType.ATTENTION_FIELD
        ]
        
        for field_type in default_field_types:
            self.fields[field_type] = CognitiveField(field_type, self.field_parameters)
    
    def add_field(self, field_type: FieldType, parameters: FieldParameters = None) -> None:
        """Add a new cognitive field"""
        params = parameters or self.field_parameters
        self.fields[field_type] = CognitiveField(field_type, params)
        logger.info(f"Added cognitive field: {field_type.value}")
    
    def process_geoids_in_fields(self, geoids: List[GeoidState], 
                                field_types: List[FieldType] = None,
                                evolution_steps: int = 10) -> FieldResult:
        """Process geoids through multiple cognitive fields"""
        start_time = time.time()
        
        if field_types is None:
            field_types = list(self.fields.keys())
        
        # Add geoids to specified fields
        for field_type in field_types:
            if field_type in self.fields:
                field = self.fields[field_type]
                for geoid in geoids:
                    field.add_geoid(geoid)
        
        # Evolve fields
        field_states = {}
        emergent_behaviors = []
        interaction_events = []
        
        for step in range(evolution_steps):
            step_events = []
            
            for field_type in field_types:
                if field_type in self.fields:
                    field = self.fields[field_type]
                    field_state = field.evolve_field(time_step=1.0)
                    field_states[field_type] = field_state
                    
                    # Record emergent behaviors
                    for structure in field_state.emergent_structures:
                        emergent_behaviors.append({
                            'step': step,
                            'field': field_type,
                            'structure': structure
                        })
            
            # Apply field coupling
            coupling_events = self._apply_field_coupling(field_types)
            step_events.extend(coupling_events)
            
            interaction_events.extend(step_events)
        
        # Extract processed geoids
        processed_geoids = []
        energy_changes = {}
        
        for geoid in geoids:
            # Create evolved version of geoid based on field processing
            evolved_geoid = self._create_field_evolved_geoid(geoid, field_states)
            processed_geoids.append(evolved_geoid)
            
            # Track energy changes
            energy_changes[geoid.geoid_id] = (
                evolved_geoid.cognitive_energy - geoid.cognitive_energy
            )
        
        # Create result
        result = FieldResult(
            original_geoids=geoids,
            processed_geoids=processed_geoids,
            field_state=field_states.get(field_types[0]) if field_states else None,
            emergent_behaviors=emergent_behaviors,
            energy_changes=energy_changes,
            interaction_events=interaction_events,
            processing_duration=time.time() - start_time,
            metadata={
                'field_types_used': [ft.value for ft in field_types],
                'evolution_steps': evolution_steps,
                'total_emergent_structures': len(emergent_behaviors)
            }
        )
        
        # Update engine state
        self._update_engine_state(result)
        
        return result
    
    def _apply_field_coupling(self, field_types: List[FieldType]) -> List[Dict[str, Any]]:
        """Apply coupling between different field types"""
        coupling_events = []
        
        # Simple coupling: transfer energy between fields
        for i, field_type1 in enumerate(field_types):
            for j, field_type2 in enumerate(field_types):
                if i < j and field_type1 in self.fields and field_type2 in self.fields:
                    coupling_strength = self.field_parameters.coupling_strength
                    
                    if coupling_strength > 0.01:
                        coupling_events.append({
                            'type': 'field_coupling',
                            'field1': field_type1.value,
                            'field2': field_type2.value,
                            'strength': coupling_strength
                        })
        
        return coupling_events
    
    def _create_field_evolved_geoid(self, original_geoid: GeoidState, 
                                   field_states: Dict[FieldType, FieldState]) -> GeoidState:
        """Create an evolved geoid based on field processing"""
        # Start with original geoid
        evolved_geoid = GeoidState(
            geoid_type=original_geoid.geoid_type,
            processing_state=GeoidProcessingState.PROCESSING,
            semantic_state=original_geoid.semantic_state,
            symbolic_state=original_geoid.symbolic_state,
            thermodynamic=original_geoid.thermodynamic
        )
        
        # Apply field effects
        for field_type, field_state in field_states.items():
            if original_geoid.geoid_id in field_state.geoid_positions:
                # Get final position in field
                final_position = field_state.geoid_positions[original_geoid.geoid_id]
                
                # Modify geoid based on field type and position
                if field_type == FieldType.COHERENCE_FIELD:
                    # Coherence field affects coherence score
                    if evolved_geoid.semantic_state:
                        # Position in coherence field affects coherence
                        coherence_boost = (final_position[0] + 1) / 2 * 0.2  # 0-0.2 boost
                        evolved_geoid.semantic_state.coherence_score = min(
                            1.0, evolved_geoid.semantic_state.coherence_score + coherence_boost
                        )
                
                elif field_type == FieldType.ENERGY_FIELD:
                    # Energy field affects thermodynamic properties
                    if evolved_geoid.thermodynamic:
                        energy_change = final_position[0] * 2.0  # -2 to +2 energy change
                        evolved_geoid.thermodynamic.free_energy = max(
                            0.0, evolved_geoid.thermodynamic.free_energy + energy_change
                        )
        
        # Connect to original
        evolved_geoid.connect_input("field_evolution", original_geoid)
        
        # Add processing metadata
        evolved_geoid.metadata.add_processing_step(
            engine_name="CognitiveFieldEngine",
            operation="field_processing",
            duration=1.0,  # Simplified
            metadata={'fields_processed': list(field_states.keys())}
        )
        
        return evolved_geoid
    
    def _update_engine_state(self, result: FieldResult) -> None:
        """Update engine state after field processing"""
        self.processing_history.append(result)
        self.total_field_operations += 1
        
        # Keep history manageable
        if len(self.processing_history) > 100:
            self.processing_history = self.processing_history[-50:]
    
    def get_engine_statistics(self) -> Dict[str, Any]:
        """Get comprehensive engine statistics"""
        if not self.processing_history:
            return {'total_operations': 0}
        
        recent_results = self.processing_history[-20:]
        
        avg_duration = np.mean([r.processing_duration for r in recent_results])
        avg_emergent_behaviors = np.mean([len(r.emergent_behaviors) for r in recent_results])
        
        field_stats = {}
        for field_type, field in self.fields.items():
            field_stats[field_type.value] = {
                'geoids_in_field': len(field.geoids_in_field),
                'evolution_steps': field.evolution_step,
                'last_update': field.last_update.isoformat()
            }
        
        return {
            'total_operations': self.total_field_operations,
            'average_processing_duration': avg_duration,
            'average_emergent_behaviors': avg_emergent_behaviors,
            'active_fields': len(self.fields),
            'field_statistics': field_stats,
            'field_parameters': self.field_parameters.__dict__
        }


# Convenience functions
def process_geoids_in_semantic_field(geoids: List[GeoidState]) -> FieldResult:
    """Convenience function to process geoids in semantic field"""
    engine = CognitiveFieldEngine()
    return engine.process_geoids_in_fields(geoids, [FieldType.SEMANTIC_FIELD])


def process_geoids_in_energy_field(geoids: List[GeoidState]) -> FieldResult:
    """Convenience function to process geoids in energy field"""
    engine = CognitiveFieldEngine()
    return engine.process_geoids_in_fields(geoids, [FieldType.ENERGY_FIELD]) 