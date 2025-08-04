import pytest

from kimera_trading.quantum.superposition import QuantumStateManager


def test_quantum_state_manager():
    manager = QuantumStateManager()
    manager.initialize_superposition()
    assert manager.state_vector is not None
