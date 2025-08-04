import pytest

from kimera_trading.core.types import MarketPhase
from kimera_trading.thermodynamic.phase_transitions import MarketPhaseDetector


def test_market_phase_detector():
    detector = MarketPhaseDetector()
    market_data = {}
    phase = detector.detect(market_data)
    assert isinstance(phase, MarketPhase)
