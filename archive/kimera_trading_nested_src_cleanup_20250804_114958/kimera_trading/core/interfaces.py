"""
Cognitive-Thermodynamic interface specifications for KIMERA trading.

Design Principles:
1. Consciousness-First: All decisions flow from consciousness states
2. Thermodynamic Constraints: Energy conservation in all operations
3. Quantum Superposition: Multiple states until observation
4. Cognitive Coherence: Maintain system-wide cognitive harmony
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Protocol

import numpy as np


class ConsciousnessProtocol(Protocol):
    """Protocol for consciousness-aware components"""

    @property
    def consciousness_level(self) -> float:
        """Current consciousness level (0-1)"""
        ...

    @abstractmethod
    async def synchronize_consciousness(self, market_consciousness: float) -> None:
        """Synchronize with market consciousness"""
        ...


class ThermodynamicProtocol(Protocol):
    """Protocol for thermodynamic components"""

    @abstractmethod
    def calculate_entropy(self, state: Dict[str, Any]) -> float:
        """Calculate system entropy"""
        ...

    @abstractmethod
    def energy_gradient(self, from_state: Any, to_state: Any) -> float:
        """Calculate energy gradient between states"""
        ...


class QuantumProtocol(Protocol):
    """Protocol for quantum-inspired components"""

    @abstractmethod
    def superposition_state(self) -> np.ndarray:
        """Return current superposition state vector"""
        ...

    @abstractmethod
    def collapse_wavefunction(self, observation: Any) -> Any:
        """Collapse superposition to definite state"""
        ...
