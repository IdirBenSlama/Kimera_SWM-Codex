import pytest

from kimera_trading.cognitive.linguistic_market import LinguisticMarketAnalyzer
class MockKimeraCognitiveBridge:
    """Auto-generated class."""
    pass
    async def analyze_market_linguistically(self, market_data):
        return {"sentiment": 0.6, "insights": ["Market is bullish"]}


@pytest.mark.asyncio
async def test_linguistic_market_analyzer():
    bridge = MockKimeraCognitiveBridge()
    analyzer = LinguisticMarketAnalyzer(bridge)
    market_data = {}
    analysis = await analyzer.analyze(market_data)
    assert analysis["sentiment"] == 0.6
    assert "Market is bullish" in analysis["insights"]
