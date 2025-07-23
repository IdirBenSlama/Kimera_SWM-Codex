"""
Cognitive Field Metrics - Quantum-Inspired Monitoring
=====================================================

Implements cognitive field theory metrics based on:
- Quantum field theory principles
- Neural field dynamics
- Information geometry
- Thermodynamic computing

This module tracks the emergent properties of KIMERA's cognitive field.
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
from collections import deque
import math

logger = logging.getLogger(__name__)

@dataclass
class CognitiveFieldState:
    """Represents the state of the cognitive field at a moment in time."""
    timestamp: datetime
    
    # Field properties
    coherence: float  # 0-1, quantum-like coherence
    entropy: float    # Information entropy
    energy: float     # Cognitive energy level
    temperature: float  # Conceptual temperature
    
    # Topology metrics
    curvature: float  # Semantic space curvature
    dimensionality: int  # Effective dimensions
    connectivity: float  # Network connectivity measure
    
    # Dynamics
    flux: float  # Rate of information flow
    oscillation_freq: float  # Dominant frequency
    phase: float  # Field phase (0-2π)
    
    # Emergence indicators
    complexity: float  # Kolmogorov complexity estimate
    self_organization: float  # Order parameter
    criticality: float  # Distance from critical point

@dataclass
class CognitiveWavePacket:
    """Represents a wave packet in the cognitive field."""
    id: str
    position: np.ndarray  # Position in semantic space
    momentum: np.ndarray  # Direction of propagation
    amplitude: float
    frequency: float
    wavelength: float
    group_velocity: float
    dispersion: float
    created_at: datetime = field(default_factory=datetime.now)

class CognitiveFieldMetrics:
    """
    Monitors the quantum-like properties of KIMERA's cognitive field.
    
    Based on:
    - Quantum field theory formalism
    - Neural field equations
    - Information geometry
    - Thermodynamic computing principles
    """
    
    def __init__(self, dimension: int = 768, history_size: int = 1000):
        self.dimension = dimension
        self.history_size = history_size
        
        # Field state history
        self.field_history: deque = deque(maxlen=history_size)
        self.wave_packets: Dict[str, CognitiveWavePacket] = {}
        
        # Field operators (quantum-inspired)
        self.hamiltonian = self._initialize_hamiltonian()
        self.momentum_operator = self._initialize_momentum_operator()
        
        # Monitoring state
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
        
        logger.info(f"CognitiveFieldMetrics initialized (dim={dimension})")
    
    def _initialize_hamiltonian(self) -> np.ndarray:
        """Initialize the Hamiltonian operator for the cognitive field."""
        # Simplified Hamiltonian for cognitive dynamics
        H = np.random.randn(self.dimension, self.dimension)
        H = (H + H.T) / 2  # Make Hermitian
        return H
    
    def _initialize_momentum_operator(self) -> np.ndarray:
        """Initialize momentum operator."""
        # Discrete derivative operator
        P = np.zeros((self.dimension, self.dimension))
        for i in range(self.dimension - 1):
            P[i, i+1] = 1
            P[i+1, i] = -1
        return P * 1j  # Imaginary unit for quantum formalism
    
    async def start_monitoring(self):
        """Start cognitive field monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Cognitive field monitoring started")
    
    async def stop_monitoring(self):
        """Stop monitoring."""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Cognitive field monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._monitoring:
            try:
                # Measure field state
                state = await self.measure_field_state()
                self.field_history.append(state)
                
                # Detect anomalies
                anomalies = self.detect_field_anomalies(state)
                if anomalies:
                    logger.warning(f"Field anomalies detected: {anomalies}")
                
                # Update wave packets
                self.evolve_wave_packets()
                
                await asyncio.sleep(0.1)  # 10Hz monitoring
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(1)
    
    async def measure_field_state(self) -> CognitiveFieldState:
        """Measure the current state of the cognitive field."""
        # This would integrate with actual KIMERA components
        # For now, we simulate measurements
        
        # Coherence: measure from geoid correlations
        coherence = self._measure_coherence()
        
        # Entropy: from information content
        entropy = self._measure_entropy()
        
        # Energy: from cognitive activity
        energy = self._measure_energy()
        
        # Temperature: from conceptual dynamics
        temperature = self._measure_temperature()
        
        # Topology metrics
        curvature = self._measure_curvature()
        dimensionality = self._estimate_dimensionality()
        connectivity = self._measure_connectivity()
        
        # Dynamics
        flux = self._measure_flux()
        oscillation_freq = self._measure_oscillation()
        phase = self._measure_phase()
        
        # Emergence
        complexity = self._estimate_complexity()
        self_organization = self._measure_self_organization()
        criticality = self._measure_criticality()
        
        return CognitiveFieldState(
            timestamp=datetime.now(),
            coherence=coherence,
            entropy=entropy,
            energy=energy,
            temperature=temperature,
            curvature=curvature,
            dimensionality=dimensionality,
            connectivity=connectivity,
            flux=flux,
            oscillation_freq=oscillation_freq,
            phase=phase,
            complexity=complexity,
            self_organization=self_organization,
            criticality=criticality
        )
    
    def _measure_coherence(self) -> float:
        """Measure quantum-like coherence in the cognitive field."""
        # Simplified: based on correlation patterns
        if len(self.field_history) < 2:
            return 1.0
        
        # Compare recent states
        recent_states = list(self.field_history)[-10:]
        if len(recent_states) < 2:
            return 1.0
        
        # Calculate phase coherence
        phases = [s.phase for s in recent_states]
        phase_diffs = np.diff(phases)
        coherence = np.exp(-np.std(phase_diffs))
        
        return float(np.clip(coherence, 0, 1))
    
    def _measure_entropy(self) -> float:
        """Measure information entropy of the field."""
        # Shannon entropy of field distribution
        if not self.wave_packets:
            return 0.0
        
        # Amplitude distribution
        amplitudes = [wp.amplitude for wp in self.wave_packets.values()]
        if not amplitudes:
            return 0.0
        
        # Normalize to probability distribution
        total = sum(amplitudes)
        if total == 0:
            return 0.0
        
        probs = [a/total for a in amplitudes]
        
        # Shannon entropy
        entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in probs)
        
        return float(entropy)
    
    def _measure_energy(self) -> float:
        """Measure total cognitive energy."""
        # E = Σ ħω|ψ|²
        if not self.wave_packets:
            return 0.0
        
        energy = sum(
            wp.frequency * wp.amplitude**2 
            for wp in self.wave_packets.values()
        )
        
        return float(energy)
    
    def _measure_temperature(self) -> float:
        """Measure conceptual temperature (activity level)."""
        # Based on energy fluctuations
        if len(self.field_history) < 10:
            return 1.0
        
        recent_energies = [s.energy for s in list(self.field_history)[-10:]]
        temperature = np.std(recent_energies) + 0.1  # Avoid zero
        
        return float(temperature)
    
    def _measure_curvature(self) -> float:
        """Measure semantic space curvature."""
        # Ricci scalar approximation
        if len(self.wave_packets) < 3:
            return 0.0
        
        # Sample positions
        positions = [wp.position for wp in list(self.wave_packets.values())[:10]]
        if len(positions) < 3:
            return 0.0
        
        # Approximate curvature from triangle defects
        curvature = 0.0
        for i in range(len(positions)-2):
            # Angle sum in triangle
            v1 = positions[i+1] - positions[i]
            v2 = positions[i+2] - positions[i+1]
            v3 = positions[i] - positions[i+2]
            
            # Angles
            angles = []
            for va, vb in [(v1, -v3), (v2, -v1), (v3, -v2)]:
                cos_angle = np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-10)
                angles.append(np.arccos(np.clip(cos_angle, -1, 1)))
            
            # Defect from π
            defect = sum(angles) - np.pi
            curvature += defect
        
        return float(curvature / max(1, len(positions)-2))
    
    def _estimate_dimensionality(self) -> int:
        """Estimate effective dimensionality of cognitive activity."""
        if not self.wave_packets:
            return 1
        
        # PCA-like analysis on positions
        positions = [wp.position for wp in self.wave_packets.values()]
        if len(positions) < 2:
            return 1
        
        # Covariance matrix
        positions_array = np.array(positions)
        cov = np.cov(positions_array.T)
        
        # Eigenvalues
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        
        # Effective dimension from participation ratio
        if len(eigenvalues) == 0:
            return 1
        
        participation = (sum(eigenvalues)**2) / sum(eigenvalues**2)
        
        return max(1, int(participation))
    
    def _measure_connectivity(self) -> float:
        """Measure field connectivity."""
        if len(self.wave_packets) < 2:
            return 0.0
        
        # Graph density approximation
        n = len(self.wave_packets)
        
        # Count "connections" (nearby wave packets)
        connections = 0
        positions = list(self.wave_packets.values())
        
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(positions[i].position - positions[j].position)
                if dist < 1.0:  # Threshold
                    connections += 1
        
        # Normalize by maximum possible connections
        max_connections = n * (n - 1) / 2
        connectivity = connections / max_connections if max_connections > 0 else 0
        
        return float(connectivity)
    
    def _measure_flux(self) -> float:
        """Measure information flux through the field."""
        if len(self.field_history) < 2:
            return 0.0
        
        # Rate of change of energy
        recent = list(self.field_history)[-2:]
        dt = (recent[1].timestamp - recent[0].timestamp).total_seconds()
        
        if dt == 0:
            return 0.0
        
        flux = abs(recent[1].energy - recent[0].energy) / dt
        
        return float(flux)
    
    def _measure_oscillation(self) -> float:
        """Measure dominant oscillation frequency."""
        if len(self.field_history) < 10:
            return 0.0
        
        # FFT on energy time series
        energies = [s.energy for s in list(self.field_history)[-64:]]
        if len(energies) < 4:
            return 0.0
        
        # Compute FFT
        fft = np.fft.fft(energies)
        freqs = np.fft.fftfreq(len(energies))
        
        # Find dominant frequency
        power = np.abs(fft)**2
        dominant_idx = np.argmax(power[1:len(power)//2]) + 1
        dominant_freq = abs(freqs[dominant_idx])
        
        return float(dominant_freq * 10)  # Scale to Hz
    
    def _measure_phase(self) -> float:
        """Measure global field phase."""
        if not self.wave_packets:
            return 0.0
        
        # Average phase of wave packets
        phases = []
        for wp in self.wave_packets.values():
            # Phase from position and momentum
            phase = np.angle(np.dot(wp.position, wp.momentum))
            phases.append(phase)
        
        # Circular mean
        mean_phase = np.angle(np.mean(np.exp(1j * np.array(phases))))
        
        return float(mean_phase + np.pi)  # Shift to [0, 2π]
    
    def _estimate_complexity(self) -> float:
        """Estimate Kolmogorov complexity of field state."""
        # Approximation based on compression ratio
        if len(self.field_history) < 10:
            return 0.0
        
        # Sample recent states
        recent = list(self.field_history)[-10:]
        
        # Create state vector
        state_data = []
        for s in recent:
            state_data.extend([
                s.coherence, s.entropy, s.energy,
                s.temperature, s.curvature
            ])
        
        # Estimate complexity from variance
        complexity = np.std(state_data) * len(state_data)
        
        return float(np.clip(complexity, 0, 100))
    
    def _measure_self_organization(self) -> float:
        """Measure self-organization level."""
        # Order parameter: 1 - entropy/max_entropy
        current_entropy = self._measure_entropy()
        max_entropy = np.log2(max(1, len(self.wave_packets)))
        
        if max_entropy == 0:
            return 1.0
        
        self_org = 1 - (current_entropy / max_entropy)
        
        return float(np.clip(self_org, 0, 1))
    
    def _measure_criticality(self) -> float:
        """Measure distance from critical point."""
        # Based on power-law detection in fluctuations
        if len(self.field_history) < 50:
            return 0.5
        
        # Energy fluctuations
        energies = [s.energy for s in list(self.field_history)[-50:]]
        fluctuations = np.abs(np.diff(energies))
        
        if len(fluctuations) < 2:
            return 0.5
        
        # Check for power-law distribution (sign of criticality)
        # Simplified: use variance/mean ratio
        if np.mean(fluctuations) == 0:
            return 0.5
        
        criticality_index = np.var(fluctuations) / np.mean(fluctuations)
        
        # Near 1 suggests criticality
        distance = abs(criticality_index - 1)
        criticality = np.exp(-distance)
        
        return float(criticality)
    
    def evolve_wave_packets(self, dt: float = 0.1):
        """Evolve wave packets according to field dynamics."""
        for wp_id, wp in list(self.wave_packets.items()):
            # Schrödinger-like evolution
            # iħ ∂ψ/∂t = Ĥψ
            
            # Update position (group velocity)
            wp.position += wp.group_velocity * wp.momentum * dt
            
            # Dispersion
            wp.wavelength *= (1 + wp.dispersion * dt)
            
            # Amplitude decay
            wp.amplitude *= np.exp(-0.01 * dt)  # Small decay
            
            # Remove if amplitude too small
            if wp.amplitude < 0.01:
                del self.wave_packets[wp_id]
    
    def create_wave_packet(
        self,
        position: np.ndarray,
        momentum: np.ndarray,
        frequency: float = 1.0
    ) -> str:
        """Create a new wave packet in the field."""
        import uuid
        
        wp_id = str(uuid.uuid4())
        
        # Wave properties
        wavelength = 2 * np.pi / (np.linalg.norm(momentum) + 1e-10)
        group_velocity = frequency / (2 * np.pi) * wavelength
        
        wp = CognitiveWavePacket(
            id=wp_id,
            position=position,
            momentum=momentum,
            amplitude=1.0,
            frequency=frequency,
            wavelength=wavelength,
            group_velocity=group_velocity,
            dispersion=0.1  # Small dispersion
        )
        
        self.wave_packets[wp_id] = wp
        
        return wp_id
    
    def detect_field_anomalies(self, state: CognitiveFieldState) -> List[str]:
        """Detect anomalies in the cognitive field."""
        anomalies = []
        
        # Check coherence collapse
        if state.coherence < 0.1:
            anomalies.append("Coherence collapse detected")
        
        # Check entropy spike
        if state.entropy > 10:
            anomalies.append("Entropy spike - possible information overload")
        
        # Check energy anomaly
        if state.energy > 1000:
            anomalies.append("Energy surge detected")
        elif state.energy < 0.01:
            anomalies.append("Energy depletion warning")
        
        # Check criticality
        if state.criticality > 0.9:
            anomalies.append("System near critical point")
        
        # Check dimensionality collapse
        if state.dimensionality == 1 and len(self.wave_packets) > 10:
            anomalies.append("Dimensional collapse - cognitive narrowing")
        
        return anomalies
    
    def get_field_summary(self) -> Dict[str, Any]:
        """Get summary of current field state."""
        if not self.field_history:
            return {
                "status": "No data",
                "measurements": 0
            }
        
        current = self.field_history[-1]
        
        # Calculate trends
        trends = {}
        if len(self.field_history) > 10:
            recent = list(self.field_history)[-10:]
            for attr in ['coherence', 'entropy', 'energy', 'complexity']:
                values = [getattr(s, attr) for s in recent]
                trend = np.polyfit(range(len(values)), values, 1)[0]
                trends[f"{attr}_trend"] = "increasing" if trend > 0.01 else "decreasing" if trend < -0.01 else "stable"
        
        return {
            "timestamp": current.timestamp.isoformat(),
            "coherence": current.coherence,
            "entropy": current.entropy,
            "energy": current.energy,
            "temperature": current.temperature,
            "dimensionality": current.dimensionality,
            "complexity": current.complexity,
            "criticality": current.criticality,
            "active_wave_packets": len(self.wave_packets),
            "trends": trends,
            "health": "good" if current.coherence > 0.5 and current.entropy < 5 else "degraded"
        }

# Global instance
_cognitive_field_metrics: Optional[CognitiveFieldMetrics] = None

def get_cognitive_field_metrics() -> CognitiveFieldMetrics:
    """Get global cognitive field metrics instance."""
    global _cognitive_field_metrics
    if _cognitive_field_metrics is None:
        _cognitive_field_metrics = CognitiveFieldMetrics()
    return _cognitive_field_metrics