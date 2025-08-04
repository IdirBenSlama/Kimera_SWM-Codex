import pytest
from kimera_trading.quantum.projected_branes import ProjectedBranes

def test_projected_branes():
    branes = ProjectedBranes()
    market_data = {}
    reduced_data = branes.reduce(market_data)
    assert reduced_data is not None
