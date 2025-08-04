import numpy as np
import pytest

from kimera_trading.quantum.measurement import QuantumMeasurement


def test_quantum_measurement():
    measurement = QuantumMeasurement()
    state_vector = np.array([1, 0])
    result = measurement.measure(state_vector)
    assert isinstance(result, int)
