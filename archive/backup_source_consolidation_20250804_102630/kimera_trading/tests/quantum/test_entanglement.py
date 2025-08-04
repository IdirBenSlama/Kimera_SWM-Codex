import pytest
from kimera_trading.quantum.entanglement import MarketEntanglementDetector

def test_market_entanglement_detector():
    detector = MarketEntanglementDetector()
    market_data = {}
    entanglement = detector.detect(market_data)
    assert isinstance(entanglement, list)
