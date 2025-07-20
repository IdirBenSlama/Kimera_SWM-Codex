"""
Thermodynamic Signal Validation Suite
======================================

This module implements the ThermodynamicSignalValidationSuite, a dedicated
system for rigorously testing signal evolution sequences against the fundamental
laws of thermodynamics.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List
import numpy as np
import logging

from .foundational_thermodynamic_engine import FoundationalThermodynamicEngine
from ..core.geoid import GeoidState
from ..utils.config import get_api_settings
from ..config.settings import get_settings

logger = logging.getLogger(__name__)

@dataclass
class FirstLawResult:
    """Represents the result of a First Law of Thermodynamics compliance test (Energy Conservation)."""
    compliant: bool
    conservation_error_percent: float
    energy_initial: float
    energy_final: float
    message: str

@dataclass
class SecondLawResult:
    """Represents the result of a Second Law of Thermodynamics compliance test (Entropy Increase)."""
    compliant: bool
    violation_count: int
    entropy_sequence: List[float]
    message: str

class ThermodynamicSignalValidationSuite:
    """
    Runs comprehensive validation of signal evolution sequences against
    the laws of thermodynamics.
    """
    def __init__(self, foundational_engine: FoundationalThermodynamicEngine, energy_conservation_tolerance: float = 0.01):
        self.settings = get_api_settings()
        logger.debug(f"   Environment: {self.settings.environment}")
self.foundational_engine = foundational_engine
        self.energy_conservation_tolerance = energy_conservation_tolerance # 1% tolerance
        logger.info(f"🔬 Thermodynamic Signal Validation Suite initialized with {energy_conservation_tolerance*100}% energy tolerance.")
        
    async def validate_first_law_compliance(self, 
                                          signal_evolution_sequence: List[GeoidState]) -> FirstLawResult:
        """
        Validates energy conservation across a signal evolution sequence.
        For a closed system, the total cognitive potential should remain constant.
        """
        if len(signal_evolution_sequence) < 2:
            return FirstLawResult(True, 0, 0, 0, "Sequence too short to validate.")

        initial_energy = signal_evolution_sequence[0].get_cognitive_potential()
        final_energy = signal_evolution_sequence[-1].get_cognitive_potential()
        
        if initial_energy == 0 and final_energy == 0:
             return FirstLawResult(True, 0, 0, 0, "No energy in sequence.")
        if initial_energy == 0:
            return FirstLawResult(False, np.inf, 0, final_energy, "Energy appeared from a zero-energy state.")

        conservation_error = abs(final_energy - initial_energy) / initial_energy
        compliant = conservation_error < self.energy_conservation_tolerance
        
        return FirstLawResult(
            compliant=compliant,
            conservation_error_percent=conservation_error * 100,
            energy_initial=initial_energy,
            energy_final=final_energy,
            message="Energy conservation compliant." if compliant else "First Law violation: Energy not conserved."
        )
    
    async def validate_second_law_compliance(self, 
                                           signal_evolution_sequence: List[GeoidState]) -> SecondLawResult:
        """
        Validates that entropy does not decrease over a signal evolution sequence.
        """
        if len(signal_evolution_sequence) < 2:
            return SecondLawResult(True, 0, [], "Sequence too short to validate.")

        entropy_sequence = [geoid.calculate_entropy() for geoid in signal_evolution_sequence]
        
        violations = 0
        for i in range(1, len(entropy_sequence)):
            # Allow for minor floating point inaccuracies
            if entropy_sequence[i] < (entropy_sequence[i-1] - 1e-9):
                violations += 1
        
        compliant = violations == 0
        
        return SecondLawResult(
            compliant=compliant,
            violation_count=violations,
            entropy_sequence=entropy_sequence,
            message="Entropy is non-decreasing." if compliant else "Second Law violation: Entropy decreased."
        ) 