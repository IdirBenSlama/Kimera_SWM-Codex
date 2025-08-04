import pytest

from kimera_trading.thermodynamic.carnot_risk import CarnotCycleRiskModel


def test_carnot_cycle_risk_model():
    model = CarnotCycleRiskModel()
    efficiency = model.calculate_efficiency()
    assert isinstance(efficiency, float)
