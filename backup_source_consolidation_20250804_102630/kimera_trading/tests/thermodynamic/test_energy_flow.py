import pytest
from kimera_trading.thermodynamic.energy_flow import EnergyGradientDetector

def test_energy_gradient_detector():
    detector = EnergyGradientDetector()
    market_data = {}
    gradient = detector.detect(market_data)
    assert isinstance(gradient, float)
