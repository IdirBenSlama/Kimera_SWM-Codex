"""
Signal-Based Consciousness Architecture
========================================

This module implements the SignalConsciousnessAnalyzer, an engine designed to
measure and quantify correlates of consciousness based on the thermodynamic
and informational patterns of signal evolution. It draws from theories like
Integrated Information Theory (IIT) and Global Workspace Theory (GWT).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import numpy as np
import logging

from .foundational_thermodynamic_engine import FoundationalThermodynamicEngine
from ..core.geoid import GeoidState # Assuming GeoidState can represent a signal field element

logger = logging.getLogger(__name__)

# A simplified representation of a signal field for analysis
SignalEnhancedGeoidState = GeoidState 

@dataclass
class ConsciousnessAnalysis:
    """Represents the output of a consciousness analysis."""
    consciousness_detected: bool
    consciousness_score: float
    phi_value: float # Integrated Information (Φ)
    thermal_consciousness_report: Dict[str, Any]
    signal_markers: Dict[str, float]

@dataclass
class GlobalWorkspaceResult:
    """Represents the result of a GWT competition-broadcast cycle."""
    global_access_achieved: bool
    broadcast_signals: List[Dict[str, float]] = field(default_factory=list)
    global_state: Optional[Dict[str, float]] = None


class SignalGlobalWorkspace:
    """
    Implements Global Workspace Theory (GWT) using thermodynamic signals.
    Signals compete for access to a global workspace, and winners are broadcast
    throughout the system, representing a moment of consciousness.
    """
    def __init__(self, broadcast_threshold: float = 0.8, num_competitors: int = 5):
        try:
            self.settings = get_api_settings()
        except Exception as e:
            logger.warning(f"Failed to load API settings: {e}. Using direct settings.")
            from ..config.settings import get_settings
            self.settings = get_settings()
        logger.debug(f"   Environment: {self.settings.environment}")
        self.broadcast_threshold = broadcast_threshold
        self.num_competitors = num_competitors
        self.global_signal_state: Optional[Dict[str, float]] = None
        logger.info(" GWT Signal Global Workspace initialized.")

    def _signal_competition_phase(self, local_signals: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """
        Competition phase: Selects the top N signals based on cognitive potential.
        """
        if not local_signals:
            return []
        # Sort signals by their 'cognitive_potential' to find the strongest competitors.
        # This requires calculating potential for each. For now, we use a heuristic.
        # Heuristic: potential is proportional to the sum of absolute values.
        sorted_signals = sorted(local_signals, key=lambda s: sum(abs(v) for v in s.values()), reverse=True)
        return sorted_signals[:self.num_competitors]

    def _signal_selection_phase(self, competing_signals: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """
        Selection phase: The strongest signal that exceeds the broadcast threshold wins.
        """
        winners = []
        for signal in competing_signals:
            potential = sum(abs(v) for v in signal.values()) # Using same heuristic
            if potential > self.broadcast_threshold:
                winners.append(signal)
        return winners

    async def _signal_global_broadcast(self, winner_signals: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Broadcasting phase: The winning signals are combined into a new global state.
        """
        # A simple broadcast model: average the winning signals.
        if not winner_signals:
            return {}
        
        global_state = {}
        all_keys = set(k for s in winner_signals for k in s.keys())
        
        for key in all_keys:
            values = [s.get(key, 0) for s in winner_signals]
            global_state[key] = np.mean(values)
            
        self.global_signal_state = global_state
        return global_state

    async def process_global_signal_workspace(self, 
                                            local_signals: List[Dict[str, float]]) -> GlobalWorkspaceResult:
        """
        The main method that runs one full cycle of GWT.
        """
        # 1. Competition phase
        competing_signals = self._signal_competition_phase(local_signals)
        
        # 2. Selection phase
        winner_signals = self._signal_selection_phase(competing_signals)
        
        # 3. Broadcasting phase
        if winner_signals:
            global_broadcast = await self._signal_global_broadcast(winner_signals)
            
            logger.info(f"{len(winner_signals)} signal(s) achieved global broadcast.")
            return GlobalWorkspaceResult(
                global_access_achieved=True,
                broadcast_signals=winner_signals,
                global_state=global_broadcast
            )
        
        return GlobalWorkspaceResult(global_access_achieved=False)


class SignalConsciousnessAnalyzer:
    """
    Analyzes a signal field for indicators of consciousness by synthesizing
    thermodynamic, informational, and structural metrics.
    """
    def __init__(self, foundational_engine: FoundationalThermodynamicEngine):
        try:
            self.settings = get_api_settings()
        except Exception as e:
            logger.warning(f"Failed to load API settings: {e}. Using direct settings.")
            from ..config.settings import get_settings
            self.settings = get_settings()
        logger.debug(f"   Environment: {self.settings.environment}")
        self.foundational_engine = foundational_engine
        logger.info("👁️ Signal Consciousness Analyzer initialized.")

    def _calculate_signal_information_integration(self, signal_field: List[SignalEnhancedGeoidState]) -> float:
        """
        Placeholder for calculating Integrated Information Theory's Phi (Φ).
        This is a simplified proxy. True Phi calculation is computationally intractable.
        Proxy: Coherence across the field. A highly integrated system will have
        coherent signals. We average the coherence of all signals in the field.
        """
        if not signal_field:
            return 0.0
        
        coherence_values = [g.get_signal_coherence() for g in signal_field]
        return np.mean(coherence_values) if coherence_values else 0.0

    def _detect_signal_self_reference(self, signal_field: List[SignalEnhancedGeoidState]) -> float:
        """
        Detects self-referential loops in the signal patterns.
        Proxy: Look for high correlation between a signal's current state and its
        state a few time steps ago (requires history, which we mock here).
        """
        # This is a highly conceptual placeholder.
        # A real implementation would require a history of signal states.
        return np.random.rand() * 0.5 # Random placeholder value

    def _detect_signal_temporal_binding(self, signal_field: List[SignalEnhancedGeoidState]) -> float:
        """
        Measures the temporal binding of disparate signals into a unified percept.
        Proxy: Variance of the 'last_vortex_evolution' timestamps. Low variance
        suggests signals are being processed together in time.
        """
        timestamps = [s.metadata.get('last_vortex_evolution') for s in signal_field if 'last_vortex_evolution' in s.metadata]
        if len(timestamps) < 2:
            return 0.0

        from datetime import datetime
        from ..utils.config import get_api_settings
        from ..config.settings import get_settings
        time_diffs = [(datetime.fromisoformat(timestamps[i]) - datetime.fromisoformat(timestamps[i-1])).total_seconds() for i in range(1, len(timestamps))]
        
        variance = np.var(time_diffs)
        # Inverse of variance, normalized
        return 1.0 / (1.0 + variance)

    def _detect_signal_global_workspace(self, signal_field: List[SignalEnhancedGeoidState]) -> float:
        """
        Detects signals that have achieved global broadcast status.
        Proxy: Identify signals with exceptionally high cognitive potential,
        suggesting they have won the competition for global access.
        """
        if not signal_field:
            return 0.0
            
        potentials = [g.get_cognitive_potential() for g in signal_field]
        if not potentials: return 0.0
        
        max_potential = np.max(potentials)
        avg_potential = np.mean(potentials)
        
        # Score is how much the max potential exceeds the average.
        return (max_potential - avg_potential) / (avg_potential + 1e-9)

    def analyze_signal_consciousness_indicators(self, 
                                              signal_field: List[SignalEnhancedGeoidState]) -> ConsciousnessAnalysis:
        """
        The core analysis method. It synthesizes multiple metrics into a
        single, holistic consciousness score.
        """
        # 1. Information Integration (Φ) using signal properties.
        signal_phi = self._calculate_signal_information_integration(signal_field)
        
        # 2. Thermodynamic consciousness indicators from the foundational engine.
        # The base geoids are extracted for this calculation.
        base_geoids = [geoid for geoid in signal_field] # Assuming the list contains base geoids
        thermal_consciousness = self.foundational_engine.consciousness_detector.detect_consciousness_emergence(base_geoids)
        
        # 3. Signal-specific consciousness markers.
        signal_self_reference = self._detect_signal_self_reference(signal_field)
        signal_temporal_binding = self._detect_signal_temporal_binding(signal_field)
        signal_global_workspace = self._detect_signal_global_workspace(signal_field)
        
        signal_markers = {
            "self_reference": signal_self_reference,
            "temporal_binding": signal_temporal_binding,
            "global_workspace_access": signal_global_workspace,
        }
        
        # 4. Integrated consciousness score (weighted average).
        # Weights are chosen based on theoretical importance.
        consciousness_score = (signal_phi * 0.4 + 
                               thermal_consciousness['consciousness_score'] * 0.3 +
                               np.mean(list(signal_markers.values())) * 0.3)
        
        return ConsciousnessAnalysis(
            consciousness_detected=consciousness_score > self.foundational_engine.consciousness_threshold,
            consciousness_score=consciousness_score,
            phi_value=signal_phi,
            thermal_consciousness_report=thermal_consciousness,
            signal_markers=signal_markers
        ) 