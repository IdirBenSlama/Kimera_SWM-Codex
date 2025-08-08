"""
Thermodynamic Analyzer for Kimera SWM

Implements thermodynamic analysis and monitoring following the computational tools
specification. Focuses on energy-like quantities, entropy production, and
thermodynamic constraints in the semantic working memory system.
"""

import logging
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.geoid import GeoidState
from ..utils.thermodynamic_utils import (PHYSICS_CONSTANTS,
                                         calculate_physical_temperature,
                                         calculate_theoretical_carnot_efficiency,
                                         calculate_total_energy)


@dataclass
class ThermodynamicState:
    """Auto-generated class."""
    pass
    """Container for thermodynamic state measurements"""

    timestamp: datetime
    total_energy: float
    free_energy: float
    entropy_production: float
    temperature: float
    pressure: float
    chemical_potential: float
    work_done: float
    heat_dissipated: float
    efficiency: float
    reversibility_index: float
    landauer_cost: float
    metadata: Dict[str, Any] = field(default_factory=dict)
class ThermodynamicAnalyzer:
    """Auto-generated class."""
    pass
    """
    Comprehensive thermodynamic analysis for Kimera SWM

    Monitors energy flows, entropy production, and thermodynamic constraints
    in the semantic working memory system.
    """

    def __init__(self, history_size: int = 1000, vault_capacity: int = 10000):
        self.history_size = history_size
        self.vault_capacity = vault_capacity
        self.states: deque = deque(maxlen=history_size)
        self.logger = logging.getLogger(__name__)

        # Previous state for calculating changes
        self.previous_state: Optional[ThermodynamicState] = None

        # System parameters
        self.coupling_strength = 0.1  # Interaction coupling
        self.thermal_reservoir_temp = 1.0  # Background temperature

        # Tracking variables
        self.total_work_done = 0.0
        self.total_heat_dissipated = 0.0
        self.total_entropy_produced = 0.0
        self.bit_erasure_count = 0

    def analyze_thermodynamic_state(
        self,
        geoids: List[GeoidState],
        vault_info: Dict[str, Any],
        system_entropy: float,
    ) -> ThermodynamicState:
        """
        Perform comprehensive thermodynamic analysis of current system state
        """
        timestamp = datetime.now()

        # Calculate basic thermodynamic quantities using CANONICAL functions
        # This requires a list of energies, not geoids. We need a helper.
        energies = [
            sum(g.semantic_state.values()) for g in geoids if g.semantic_state
        ]  # Simplified energy for now

        total_energy = calculate_total_energy(geoids)
        temperature = calculate_physical_temperature(energies)
        pressure = self._calculate_semantic_pressure(geoids, self.vault_capacity)

        # Calculate free energy
        free_energy = self._calculate_free_energy(
            total_energy, system_entropy, temperature
        )

        # Calculate chemical potential (energy per geoid)
        chemical_potential = total_energy / len(geoids) if geoids else 0.0

        # Calculate changes from previous state
        entropy_production = 0.0
        work_done = 0.0
        heat_dissipated = 0.0
        efficiency = 0.0
        reversibility_index = 1.0

        if self.previous_state is not None:
            time_delta = (timestamp - self.previous_state.timestamp).total_seconds()

            # Entropy production rate
            entropy_production = self._calculate_entropy_production_rate(
                self.previous_state.metadata.get("system_entropy", 0.0),
                system_entropy,
                time_delta,
            )

            # Work and heat calculations
            work_done, heat_dissipated = self._calculate_work_and_heat(
                self.previous_state.total_energy,
                total_energy,
                self.previous_state.metadata.get("system_entropy", 0.0),
                system_entropy,
                temperature,
            )

            # Thermodynamic efficiency
            if heat_dissipated > 0:
                efficiency = work_done / (work_done + heat_dissipated)

            # Reversibility index (0 = irreversible, 1 = reversible)
            if entropy_production >= 0:
                reversibility_index = 1.0 / (1.0 + entropy_production)

            # Update totals
            self.total_work_done += work_done
            self.total_heat_dissipated += heat_dissipated
            self.total_entropy_produced += entropy_production * time_delta

        # Calculate Landauer cost for information processing
        # Estimate bit erasures from geoid state changes
        bit_erasures = self._estimate_bit_erasures(geoids)
        landauer_cost = self._calculate_landauer_cost(bit_erasures, temperature)

        # Store and return the new state
        new_state = ThermodynamicState(
            timestamp=timestamp,
            total_energy=total_energy,
            free_energy=free_energy,
            entropy_production=entropy_production,
            temperature=temperature,
            pressure=pressure,
            chemical_potential=chemical_potential,
            work_done=work_done,
            heat_dissipated=heat_dissipated,
            efficiency=efficiency,
            reversibility_index=reversibility_index,
            landauer_cost=landauer_cost,
            metadata={
                "system_entropy": system_entropy,
                "geoid_count": len(geoids),
                "vault_info": vault_info,
                "bit_erasures": bit_erasures,
                "total_work_done": self.total_work_done,
                "total_heat_dissipated": self.total_heat_dissipated,
                "total_entropy_produced": self.total_entropy_produced,
                "vault_capacity": self.vault_capacity,
            },
        )

        self.states.append(new_state)
        self.previous_state = new_state

        self.bit_erasure_count += bit_erasures

        return new_state

    def _calculate_free_energy(
        self, total_energy: float, entropy: float, temperature: float
    ) -> float:
        return total_energy - temperature * entropy

    def _calculate_semantic_pressure(
        self, geoids: List[GeoidState], vault_capacity: int
    ) -> float:
        if vault_capacity <= 0:
            return 0.0
        total_information = sum(
            len(g.semantic_state) + sum(g.semantic_state.values()) for g in geoids
        )
        return total_information / vault_capacity

    def _calculate_landauer_cost(self, bit_erasures: int, temperature: float) -> float:
        return (
            bit_erasures
            * PHYSICS_CONSTANTS["normalized_kb"]
            * temperature
            * math.log(2)
        )

    def _calculate_entropy_production_rate(
        self, entropy_before: float, entropy_after: float, time_delta: float
    ) -> float:
        if time_delta <= 0:
            return 0.0
        return (entropy_after - entropy_before) / time_delta

    def _calculate_work_and_heat(
        self,
        energy_before: float,
        energy_after: float,
        entropy_before: float,
        entropy_after: float,
        temperature: float,
    ) -> Tuple[float, float]:
        energy_change = energy_after - energy_before
        entropy_change = entropy_after - entropy_before
        heat = temperature * entropy_change
        work = heat - energy_change
        return work, heat

    def _estimate_bit_erasures(self, geoids: List[GeoidState]) -> int:
        """
        Estimate number of bits erased in semantic transformations
        Based on changes in semantic state complexity
        """
        if self.previous_state is None:
            return 0

        current_complexity = sum(len(geoid.semantic_state) for geoid in geoids)
        previous_complexity = self.previous_state.metadata.get(
            "total_complexity", current_complexity
        )

        # Estimate bit erasures from complexity reduction
        complexity_reduction = max(0, previous_complexity - current_complexity)

        # Each reduced feature represents approximately 8 bits (rough estimate)
        estimated_erasures = int(complexity_reduction * 8)

        self.bit_erasure_count += estimated_erasures
        return estimated_erasures

    def check_thermodynamic_constraints(
        self, state: ThermodynamicState
    ) -> List[Dict[str, Any]]:
        """
        Check for violations of thermodynamic constraints
        """
        violations = []

        # Second Law: Entropy should not decrease in isolated system
        if state.entropy_production < -1e-6:  # Small tolerance for numerical errors
            violations.append(
                {
                    "type": "second_law_violation",
                    "description": "Entropy production is negative",
                    "value": state.entropy_production,
                    "severity": "high",
                }
            )

        # Energy conservation check
        if self.previous_state is not None:
            energy_change = state.total_energy - self.previous_state.total_energy
            expected_change = state.work_done - state.heat_dissipated

            if abs(energy_change - expected_change) > 0.1:  # Tolerance
                violations.append(
                    {
                        "type": "energy_conservation_violation",
                        "description": "Energy change inconsistent with work-heat balance",
                        "energy_change": energy_change,
                        "expected_change": expected_change,
                        "severity": "medium",
                    }
                )

        # Temperature bounds
        if state.temperature < 0.01 or state.temperature > 100:
            violations.append(
                {
                    "type": "temperature_anomaly",
                    "description": "Temperature outside expected range",
                    "value": state.temperature,
                    "severity": "medium",
                }
            )

        # Pressure bounds
        if state.pressure > 1.0:  # System approaching capacity
            violations.append(
                {
                    "type": "high_pressure_warning",
                    "description": "System pressure indicates near-capacity operation",
                    "value": state.pressure,
                    "severity": "low",
                }
            )

        return violations

    def calculate_thermodynamic_efficiency(
        self, window_size: int = 100
    ) -> Dict[str, float]:
        """
        Calculate various thermodynamic efficiency measures
        """
        if len(self.states) < 2:
            return {}

        recent_states = list(self.states)[-window_size:]

        # Carnot efficiency (theoretical maximum) using CANONICAL function
        avg_temp = np.mean([s.temperature for s in recent_states])
        carnot_efficiency = calculate_theoretical_carnot_efficiency(
            avg_temp, self.thermal_reservoir_temp
        )

        # Actual efficiency
        total_work = sum(s.work_done for s in recent_states)
        total_heat = sum(s.heat_dissipated for s in recent_states)
        actual_efficiency = (
            total_work / (total_work + total_heat)
            if (total_work + total_heat) > 0
            else 0
        )

        # Reversibility measure
        avg_reversibility = np.mean([s.reversibility_index for s in recent_states])

        # Information processing efficiency
        total_landauer_cost = sum(s.landauer_cost for s in recent_states)
        total_information_processed = sum(
            s.metadata.get("bit_erasures", 0) for s in recent_states
        )
        info_efficiency = (
            total_information_processed / total_landauer_cost
            if total_landauer_cost > 0
            else 0
        )

        return {
            "carnot_efficiency": carnot_efficiency,
            "actual_efficiency": actual_efficiency,
            "efficiency_ratio": (
                actual_efficiency / carnot_efficiency if carnot_efficiency > 0 else 0
            ),
            "reversibility_index": avg_reversibility,
            "information_efficiency": info_efficiency,
            "total_entropy_produced": self.total_entropy_produced,
        }

    def get_thermodynamic_trends(
        self, window_size: int = 100
    ) -> Dict[str, List[float]]:
        """Get thermodynamic trends over recent measurements"""
        if len(self.states) < 2:
            return {}

        recent_states = list(self.states)[-window_size:]

        return {
            "total_energy": [s.total_energy for s in recent_states],
            "free_energy": [s.free_energy for s in recent_states],
            "temperature": [s.temperature for s in recent_states],
            "pressure": [s.pressure for s in recent_states],
            "entropy_production": [s.entropy_production for s in recent_states],
            "efficiency": [s.efficiency for s in recent_states],
            "reversibility_index": [s.reversibility_index for s in recent_states],
            "landauer_cost": [s.landauer_cost for s in recent_states],
            "timestamps": [s.timestamp.isoformat() for s in recent_states],
        }

    def detect_thermodynamic_anomalies(
        self, threshold_std: float = 2.0
    ) -> List[Dict[str, Any]]:
        """Detect anomalous thermodynamic behavior"""
        if len(self.states) < 10:
            return []

        recent_states = list(self.states)[-50:]
        anomalies = []

        # Energy anomalies
        energies = [s.total_energy for s in recent_states]
        energy_mean = np.mean(energies)
        energy_std = np.std(energies)

        for state in recent_states[-5:]:
            if abs(state.total_energy - energy_mean) > threshold_std * energy_std:
                anomalies.append(
                    {
                        "timestamp": state.timestamp.isoformat(),
                        "type": "energy_anomaly",
                        "value": state.total_energy,
                        "deviation": abs(state.total_energy - energy_mean) / energy_std,
                        "severity": (
                            "high" if state.total_energy > energy_mean else "low"
                        ),
                    }
                )

        # Temperature anomalies
        temperatures = [s.temperature for s in recent_states]
        temp_mean = np.mean(temperatures)
        temp_std = np.std(temperatures)

        for state in recent_states[-5:]:
            if abs(state.temperature - temp_mean) > threshold_std * temp_std:
                anomalies.append(
                    {
                        "timestamp": state.timestamp.isoformat(),
                        "type": "temperature_anomaly",
                        "value": state.temperature,
                        "deviation": abs(state.temperature - temp_mean) / temp_std,
                        "severity": "medium",
                    }
                )

        # Efficiency anomalies
        efficiencies = [s.efficiency for s in recent_states if s.efficiency > 0]
        if efficiencies:
            eff_mean = np.mean(efficiencies)
            eff_std = np.std(efficiencies)

            for state in recent_states[-5:]:
                if (
                    state.efficiency > 0
                    and abs(state.efficiency - eff_mean) > threshold_std * eff_std
                ):
                    anomalies.append(
                        {
                            "timestamp": state.timestamp.isoformat(),
                            "type": "efficiency_anomaly",
                            "value": state.efficiency,
                            "deviation": abs(state.efficiency - eff_mean) / eff_std,
                            "severity": (
                                "low" if state.efficiency < eff_mean else "high"
                            ),
                        }
                    )

        return anomalies

    def export_thermodynamic_data(self) -> List[Dict[str, Any]]:
        """Export thermodynamic measurements for analysis"""
        return [
            {
                "timestamp": s.timestamp.isoformat(),
                "total_energy": s.total_energy,
                "free_energy": s.free_energy,
                "entropy_production": s.entropy_production,
                "temperature": s.temperature,
                "pressure": s.pressure,
                "chemical_potential": s.chemical_potential,
                "work_done": s.work_done,
                "heat_dissipated": s.heat_dissipated,
                "efficiency": s.efficiency,
                "reversibility_index": s.reversibility_index,
                "landauer_cost": s.landauer_cost,
                "metadata": s.metadata,
            }
            for s in self.states
        ]
class ThermodynamicCalculator:
    """Auto-generated class."""
    pass
    """
    Static utility methods for thermodynamic calculations.
    Used by tests and other modules that need individual calculations.
    """

    @staticmethod
    def calculate_landauer_cost(bits_erased: int, temperature: float) -> float:
        """
        Calculate the Landauer cost for bit erasure.

        Args:
            bits_erased: Number of bits erased
            temperature: System temperature

        Returns:
            Landauer cost (energy units)
        """
        return (
            bits_erased * PHYSICS_CONSTANTS["normalized_kb"] * temperature * math.log(2)
        )

    @staticmethod
    def calculate_entropy_production_rate(
        entropy_before: float, entropy_after: float, time_delta: float
    ) -> float:
        """
        Calculate the entropy production rate.

        Args:
            entropy_before: Initial entropy
            entropy_after: Final entropy
            time_delta: Time interval

        Returns:
            Entropy production rate
        """
        if time_delta <= 0:
            return 0.0
        return (entropy_after - entropy_before) / time_delta

    @staticmethod
    def calculate_work_and_heat(
        energy_before: float,
        energy_after: float,
        entropy_before: float,
        entropy_after: float,
        temperature: float,
    ) -> Tuple[float, float]:
        """
        Calculate work and heat based on the first law analogue.

        Args:
            energy_before: Initial energy
            energy_after: Final energy
            entropy_before: Initial entropy
            entropy_after: Final entropy
            temperature: System temperature

        Returns:
            Tuple of (work, heat)
        """
        energy_change = energy_after - energy_before
        entropy_change = entropy_after - entropy_before
        heat = temperature * entropy_change
        work = heat - energy_change
        return work, heat
