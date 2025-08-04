"""
Quantum Field Engine
====================
Implements quantum field theory concepts for Kimera's cognitive dynamics.

This engine models cognitive states as quantum fields, enabling:
- Superposition of cognitive states
- Entanglement between semantic concepts
- Wave function collapse during observation/measurement
- Quantum tunneling between cognitive basins
"""

import cmath
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from scipy.linalg import expm

from ..config.settings import get_settings
from ..utils.config import get_api_settings

logger = logging.getLogger(__name__)


class QuantumOperator(Enum):
    """Standard quantum operators"""

    HAMILTONIAN = "hamiltonian"
    POSITION = "position"
    MOMENTUM = "momentum"
    SPIN = "spin"
    ANNIHILATION = "annihilation"
    CREATION = "creation"


@dataclass
class QuantumState:
    """Represents a quantum state in Hilbert space"""

    state_vector: np.ndarray  # Complex state vector
    basis_labels: List[str]  # Basis state labels
    entanglement_measure: float
    coherence: float
    purity: float

    @property
    def dimension(self) -> int:
        return len(self.state_vector)

    def normalize(self):
        """Normalize the state vector"""
        norm = np.linalg.norm(self.state_vector)
        if norm > 0:
            self.state_vector /= norm


class QuantumFieldEngine:
    """
    Quantum field theory engine for cognitive dynamics
    """

    def __init__(self, dimension: int = 10, device: str = "cpu"):
        self.settings = get_api_settings()
        logger.debug(f"   Environment: {self.settings.environment}")
        self.dimension = dimension
        self.device = device
        self.hbar = 1.0  # Natural units

        # Quantum operators
        self.operators = self._initialize_operators()

        # Quantum states registry
        self.states = {}

        logger.info(f"Quantum Field Engine initialized with dimension {dimension}")

    def _initialize_operators(self) -> Dict[str, np.ndarray]:
        """Initialize standard quantum operators"""
        d = self.dimension
        operators = {}

        # Position operator (diagonal in position basis)
        operators["position"] = np.diag(np.linspace(-1, 1, d))

        # Momentum operator (derivative in position basis)
        operators["momentum"] = self._create_momentum_operator(d)

        # Hamiltonian (kinetic + potential)
        operators["hamiltonian"] = self._create_hamiltonian(d)

        # Ladder operators
        operators["annihilation"] = self._create_annihilation_operator(d)
        operators["creation"] = operators["annihilation"].conj().T

        # Pauli matrices for spin-1/2
        operators["sigma_x"] = np.array([[0, 1], [1, 0]], dtype=complex)
        operators["sigma_y"] = np.array([[0, -1j], [1j, 0]], dtype=complex)
        operators["sigma_z"] = np.array([[1, 0], [0, -1]], dtype=complex)

        return operators

    def _create_momentum_operator(self, d: int) -> np.ndarray:
        """Create momentum operator using finite differences"""
        p = np.zeros((d, d), dtype=complex)
        dx = 2.0 / (d - 1)

        for i in range(d):
            if i > 0:
                p[i, i - 1] = 1j / (2 * dx)
            if i < d - 1:
                p[i, i + 1] = -1j / (2 * dx)

        return p

    def _create_hamiltonian(self, d: int) -> np.ndarray:
        """Create Hamiltonian operator"""
        # Create position and momentum operators first
        position = np.diag(np.linspace(-1, 1, d))
        momentum = self._create_momentum_operator(d)

        # Kinetic energy: -ℏ²/2m ∇²
        kinetic = -0.5 * momentum @ momentum

        # Potential energy: harmonic oscillator V = 0.5 * k * x²
        potential = 0.5 * position @ position

        return kinetic + potential

    def _create_annihilation_operator(self, d: int) -> np.ndarray:
        """Create annihilation operator for harmonic oscillator"""
        a = np.zeros((d, d), dtype=complex)
        for n in range(d - 1):
            a[n, n + 1] = np.sqrt(n + 1)
        return a

    def create_superposition(
        self, states: List[np.ndarray], coefficients: List[complex]
    ) -> QuantumState:
        """
        Create quantum superposition of states

        |ψ⟩ = Σ cᵢ|φᵢ⟩
        """
        if len(states) != len(coefficients):
            raise ValueError("Number of states must match number of coefficients")

        # Normalize coefficients
        norm = np.sqrt(sum(abs(c) ** 2 for c in coefficients))
        coefficients = [c / norm for c in coefficients]

        # Create superposition
        superposed = np.zeros(states[0].shape, dtype=complex)
        for state, coeff in zip(states, coefficients):
            superposed += coeff * state

        # Calculate quantum properties
        entanglement = self.calculate_entanglement(superposed)
        coherence = self.calculate_coherence(superposed)
        purity = self.calculate_purity(superposed)

        return QuantumState(
            state_vector=superposed,
            basis_labels=[f"state_{i}" for i in range(len(superposed))],
            entanglement_measure=entanglement,
            coherence=coherence,
            purity=purity,
        )

    def create_entangled_state(self, subsystem_dims: List[int]) -> QuantumState:
        """
        Create maximally entangled state (e.g., Bell state)

        |Φ⁺⟩ = 1/√2 (|00⟩ + |11⟩)
        """
        total_dim = np.prod(subsystem_dims)
        state = np.zeros(total_dim, dtype=complex)

        # Create Bell-like state
        if len(subsystem_dims) == 2 and all(d == 2 for d in subsystem_dims):
            # Standard Bell state
            state[0] = 1 / np.sqrt(2)  # |00⟩
            state[3] = 1 / np.sqrt(2)  # |11⟩
        else:
            # Generalized GHZ state
            state[0] = 1 / np.sqrt(2)
            state[-1] = 1 / np.sqrt(2)

        entanglement = self.calculate_entanglement(state)
        coherence = self.calculate_coherence(state)
        purity = self.calculate_purity(state)

        return QuantumState(
            state_vector=state,
            basis_labels=[f"basis_{i}" for i in range(total_dim)],
            entanglement_measure=entanglement,
            coherence=coherence,
            purity=purity,
        )

    def apply_measurement(
        self, state: QuantumState, observable: np.ndarray
    ) -> Tuple[float, QuantumState]:
        """
        Perform quantum measurement and collapse wave function

        Returns measurement outcome and collapsed state
        """
        # Ensure observable is Hermitian
        if not np.allclose(observable, observable.conj().T):
            raise ValueError("Observable must be Hermitian")

        # Get eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(observable)

        # Calculate probabilities
        probabilities = []
        for i in range(len(eigenvalues)):
            projection = np.abs(np.vdot(eigenvectors[:, i], state.state_vector)) ** 2
            probabilities.append(projection)

        # Choose outcome based on probabilities
        outcome_idx = np.random.choice(len(eigenvalues), p=probabilities)
        outcome = eigenvalues[outcome_idx]

        # Collapse state
        collapsed_state = eigenvectors[:, outcome_idx].copy()
        collapsed_state /= np.linalg.norm(collapsed_state)

        # Create new quantum state
        new_state = QuantumState(
            state_vector=collapsed_state,
            basis_labels=state.basis_labels,
            entanglement_measure=0.0,  # Measurement destroys entanglement
            coherence=self.calculate_coherence(collapsed_state),
            purity=1.0,  # Pure state after measurement
        )

        return outcome, new_state

    def time_evolution(
        self, state: QuantumState, hamiltonian: np.ndarray, time: float
    ) -> QuantumState:
        """
        Evolve quantum state under Hamiltonian

        |ψ(t)⟩ = exp(-iHt/ℏ)|ψ(0)⟩
        """
        # Unitary evolution operator
        U = expm(-1j * hamiltonian * time / self.hbar)

        # Evolve state
        evolved_state = U @ state.state_vector

        # Calculate new quantum properties
        entanglement = self.calculate_entanglement(evolved_state)
        coherence = self.calculate_coherence(evolved_state)
        purity = self.calculate_purity(evolved_state)

        return QuantumState(
            state_vector=evolved_state,
            basis_labels=state.basis_labels,
            entanglement_measure=entanglement,
            coherence=coherence,
            purity=purity,
        )

    def calculate_entanglement(self, state: np.ndarray) -> float:
        """
        Calculate entanglement entropy (von Neumann entropy)

        S = -Tr(ρ log ρ)
        """
        # Create density matrix
        rho = np.outer(state, state.conj())

        # Calculate eigenvalues
        eigenvalues = np.linalg.eigvalsh(rho)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Remove zeros

        # Von Neumann entropy
        entropy = -np.sum(eigenvalues * np.log(eigenvalues))

        return float(entropy)

    def calculate_coherence(self, state: np.ndarray) -> float:
        """
        Calculate quantum coherence (l1-norm of off-diagonal elements)
        """
        rho = np.outer(state, state.conj())

        # Sum of absolute values of off-diagonal elements
        coherence = 0.0
        for i in range(len(state)):
            for j in range(len(state)):
                if i != j:
                    coherence += abs(rho[i, j])

        return float(coherence)

    def calculate_purity(self, state: np.ndarray) -> float:
        """
        Calculate state purity Tr(ρ²)

        Purity = 1 for pure states, < 1 for mixed states
        """
        rho = np.outer(state, state.conj())
        purity = np.real(np.trace(rho @ rho))
        return float(purity)

    def quantum_tunneling_probability(
        self, barrier_height: float, particle_energy: float, barrier_width: float
    ) -> float:
        """
        Calculate quantum tunneling probability through potential barrier

        Uses WKB approximation
        """
        if particle_energy >= barrier_height:
            return 1.0  # Classical passage

        # Effective mass (normalized)
        m = 1.0

        # Tunneling coefficient
        kappa = np.sqrt(2 * m * (barrier_height - particle_energy)) / self.hbar

        # Transmission probability (WKB approximation)
        T = np.exp(-2 * kappa * barrier_width)

        return float(T)

    def decoherence_time(
        self, system_size: int, coupling_strength: float, temperature: float
    ) -> float:
        """
        Estimate decoherence time for quantum system

        τ_d ∝ ℏ / (k_B T × coupling × system_size)
        """
        k_B = 1.0  # Natural units

        if temperature <= 0 or coupling_strength <= 0:
            return float("inf")  # No decoherence

        decoherence_time = self.hbar / (
            k_B * temperature * coupling_strength * system_size
        )

        return float(decoherence_time)

    def apply_quantum_gate(
        self, state: QuantumState, gate: np.ndarray, qubits: List[int]
    ) -> QuantumState:
        """
        Apply quantum gate to specified qubits
        """
        # For simplicity, assume gate acts on consecutive qubits
        new_state = state.state_vector.copy()

        # Apply gate (simplified for small systems)
        if len(qubits) == 1:
            # Single qubit gate
            new_state = gate @ new_state
        else:
            # Multi-qubit gate (tensor product structure)
            # This is a simplified implementation
            new_state = gate @ new_state

        # Normalize
        new_state /= np.linalg.norm(new_state)

        return QuantumState(
            state_vector=new_state,
            basis_labels=state.basis_labels,
            entanglement_measure=self.calculate_entanglement(new_state),
            coherence=self.calculate_coherence(new_state),
            purity=self.calculate_purity(new_state),
        )

    def get_quantum_metrics(self) -> Dict[str, Any]:
        """Get quantum system metrics"""
        return {
            "dimension": self.dimension,
            "num_states": len(self.states),
            "operators": list(self.operators.keys()),
            "hbar": self.hbar,
        }


def create_quantum_field_engine(
    dimension: int = 10, device: str = "cpu"
) -> QuantumFieldEngine:
    """Factory function to create quantum field engine"""
    return QuantumFieldEngine(dimension=dimension, device=device)
