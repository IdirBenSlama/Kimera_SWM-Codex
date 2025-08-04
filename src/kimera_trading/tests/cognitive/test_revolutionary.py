import pytest

from kimera_trading.cognitive.revolutionary import RevolutionaryMarketIntelligence


@pytest.mark.asyncio
async def test_revolutionary_market_intelligence():
    intelligence = RevolutionaryMarketIntelligence()
    intelligence.revolution_threshold = 0.5
    market_state = {}
    historical_context = {}
    signal = await intelligence.detect_revolutionary_moment(
        market_state, historical_context
    )
    assert signal.detected is False
