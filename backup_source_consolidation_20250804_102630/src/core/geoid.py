from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, TYPE_CHECKING

import numpy as np
if TYPE_CHECKING:
    from ..engines.enhanced_vortex_system import EnhancedVortexBattery

@dataclass
class GeoidState:
    """Core Geoid implementation following DOC-201 specification"""

    geoid_id: str
    semantic_state: Dict[str, float] = field(default_factory=dict)
    symbolic_state: Dict[str, Any] = field(default_factory=dict)
    embedding_vector: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def calculate_entropy(self) -> float:
        """Calculate Shannon entropy of the semantic state."""
        if not self.semantic_state:
            return 0.0

        # Normalize the semantic state values into a probability distribution
        # before calculating entropy. This is done on-the-fly.
        values = np.array(list(self.semantic_state.values()))
        total = np.sum(values)
        if total <= 0:
            return 0.0
            
        probabilities = values / total
        probabilities = probabilities[probabilities > 0]
        
        if probabilities.size == 0:
            return 0.0
        return float(-np.sum(probabilities * np.log2(probabilities)))

    def calculate_entropic_signal_properties(self) -> Dict[str, float]:
        """Calculate TCSE signal properties from existing semantic_state."""
        if not self.semantic_state:
            return {'signal_temperature': 1.0, 'cognitive_potential': 0.0, 'signal_coherence': 0.0, 'entropic_flow_capacity': 0}
        
        return {
            'signal_temperature': self.get_signal_temperature(),
            'cognitive_potential': self.get_cognitive_potential(),
            'signal_coherence': self.get_signal_coherence(),
            'entropic_flow_capacity': len(self.semantic_state)
        }
    
    def get_signal_temperature(self) -> float:
        """
        Extract information temperature from semantic variance.
        This metric treats the variance of semantic values as a proxy for thermal agitation in an information-theoretic context.
        A higher variance implies a 'hotter', more disordered state.
        """
        if not self.semantic_state:
            return 1.0
        values = np.array(list(self.semantic_state.values()))
        return float(np.var(values)) if len(values) > 1 else 1.0
    
    def get_cognitive_potential(self) -> float:
        """
        Calculate cognitive energy potential for signal evolution.
        Defined as the product of entropy and information temperature, this represents the total energy that can be
        extracted or utilized during cognitive operations, analogous to thermodynamic potential.
        """
        entropy = self.calculate_entropy()
        return entropy * self.get_signal_temperature()
    
    def get_signal_coherence(self) -> float:
        """
        Measure signal coherence based on semantic consistency.
        Inversely proportional to entropy. A low-entropy, highly-ordered state has high coherence (approaching 1.0),
        while a high-entropy, disordered state has low coherence.
        """
        return 1.0 / (1.0 + self.calculate_entropy())

    def establish_vortex_signal_coherence(self, vortex_battery: 'EnhancedVortexBattery') -> str:
        """Establish quantum coherence between geoid signal and vortex energy."""
        signal_entropy = self.calculate_entropy()
        signal_temp = self.get_signal_temperature()
        
        # Position based on signal properties
        position = (signal_entropy * 10, signal_temp * 10)
        initial_energy = self.get_cognitive_potential()
        
        # Create coherent vortex
        signal_vortex = vortex_battery.create_energy_vortex(position, initial_energy)
        
        # Store vortex reference in geoid metadata
        if signal_vortex:
            self.metadata['signal_vortex_id'] = signal_vortex.vortex_id
            self.metadata['vortex_coherence_established'] = datetime.now().isoformat()
            return signal_vortex.vortex_id
        return ""

    def evolve_via_vortex_coherence(self, vortex_battery: 'EnhancedVortexBattery') -> Dict[str, float]:
        """Evolve signal state using quantum vortex coherence."""
        if 'signal_vortex_id' not in self.metadata:
            # If no vortex is linked, the state cannot evolve via this mechanism.
            return self.semantic_state
        
        # Get current signal properties to determine energy required for evolution
        signal_properties = self.calculate_entropic_signal_properties()
        
        # Power evolution using vortex energy
        evolution_result = vortex_battery.power_signal_evolution(
            self.semantic_state, 
            signal_properties['cognitive_potential']
        )
        
        if evolution_result.get("success"):
            # Update semantic state with evolved signal
            self.semantic_state = evolution_result["evolved_signal"]
            
            # Record evolution in metadata for traceability and analysis
            self.metadata['last_vortex_evolution'] = datetime.now().isoformat()
            self.metadata['fibonacci_enhancement'] = evolution_result["fibonacci_enhancement"]
        
        return self.semantic_state

    def update_semantic_state(self, new_features: Dict[str, float]):
        self.semantic_state.update(new_features)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'geoid_id': self.geoid_id,
            'semantic_state': self.semantic_state,
            'symbolic_state': self.symbolic_state,
            'embedding_vector': self.embedding_vector,
            'metadata': self.metadata
        }

