import numpy as np
import pytest

from kimera_trading.cognitive.consciousness_detector import MarketConsciousnessDetector
from kimera_trading.core.types import ConsciousnessState


@pytest.mark.asyncio
async def test_market_consciousness_detector_logic():
    detector = MarketConsciousnessDetector()
    # Create a mock market data that suggests a certain level of consciousness
    market_data = {
        "prices": {
            "BTC": np.array([100, 102, 101, 103, 105]),
            "ETH": np.array([10, 11, 10.5, 11.5, 12]),
        },
        "sentiment": 0.7,  # Positive sentiment
    }
    consciousness_state = await detector.detect_consciousness(market_data)
    assert isinstance(consciousness_state, ConsciousnessState)
    assert 0 <= consciousness_state.level <= 1
    assert 0 <= consciousness_state.coherence <= 1
