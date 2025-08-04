import pytest

from kimera_trading.thermodynamic.entropy_engine import MarketEntropyCalculator


def test_market_entropy_calculator():
    calculator = MarketEntropyCalculator()
    market_data = {"price_history": [100, 101, 102, 101, 103]}
    entropy = calculator.calculate(market_data)
    assert isinstance(entropy, float)
