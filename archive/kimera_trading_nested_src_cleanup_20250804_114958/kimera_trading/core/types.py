from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class ConsciousnessState:
    """Represents consciousness state of system/market"""

    level: float  # 0-1 scale
    coherence: float  # Cognitive coherence
    awareness_vector: np.ndarray  # Multi-dimensional awareness
    synchronization: float  # Market sync level


@dataclass
class ThermodynamicState:
    """Represents thermodynamic state"""

    entropy: float  # System entropy
    temperature: float  # Market "temperature"
    energy: float  # Available energy
    phase: str  # Market phase (solid/liquid/gas/plasma)


@dataclass
class QuantumOrder:
    """Order existing in superposition"""

    state_vector: np.ndarray  # Quantum state
    probabilities: Dict[str, float]  # Execution probabilities
    entanglement: Optional[List["QuantumOrder"]]  # Entangled orders


class MarketPhase(Enum):
    """Market phases from thermodynamic perspective"""

    SOLID = "solid"  # Low volatility, structured
    LIQUID = "liquid"  # Normal trading
    GAS = "gas"  # High volatility
    PLASMA = "plasma"  # Extreme conditions
    BOSE_EINSTEIN = "bose_einstein"  # Condensed, correlated
