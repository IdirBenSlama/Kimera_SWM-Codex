"""Tests for KimeraSemanticMarketAnalyzer.

This module contains pytest tests for the KimeraSemanticMarketAnalyzer class which handles:
- Market geoid creation from market data
- Semantic contradiction detection between market states
- Integration with other Kimera trading system components

Test cases verify:
- Proper geoid creation with correct semantic state mapping
- Accurate detection of semantic contradictions
- Correct interaction with mocked system components
"""
import pytest
import asyncio
import dataclasses
from datetime import datetime
from unittest.mock import MagicMock
from backend.trading.kimera_integrated_trading_system import (
    KimeraSemanticMarketAnalyzer,
    KimeraMarketData,
    GeoidState
)
from backend.utils.kimera_logger import get_trading_logger
logger = get_trading_logger("test_market_analyzer")

@pytest.fixture
def mock_kimera_system():
    """Fixture providing a mocked Kimera trading system with all required components.
    
    Returns:
        MagicMock: A mock Kimera system with:
            - get_contradiction_engine() returning a mock engine
            - get_thermodynamic_engine() returning a mock engine
            - get_vault_manager() returning a mock manager
            
    Example:
        >>> system = mock_kimera_system()
        >>> assert system.get_contradiction_engine() is not None
    """
    logger.debug("Creating mock Kimera system")
    system = MagicMock()
    system.get_contradiction_engine.return_value = MagicMock()
    system.get_thermodynamic_engine.return_value = MagicMock()
    system.get_vault_manager.return_value = MagicMock()
    logger.debug("Mock Kimera system created with mocked components")
    return system

@pytest.fixture
def sample_market_data():
    """Fixture providing sample KimeraMarketData for testing.
    
    Returns:
        KimeraMarketData: A populated market data object with:
            - Symbol: BTCUSDT
            - Price: 50000.0
            - Volume: 1000.0
            - Standard 24h metrics
            - Current timestamp
            
    Example:
        >>> data = sample_market_data()
        >>> assert data.symbol == "BTCUSDT"
        >>> assert data.price == 50000.0
    """
    return KimeraMarketData(
        symbol="BTCUSDT",
        price=50000.0,
        volume=1000.0,
        high_24h=51000.0,
        low_24h=49000.0,
        change_24h=1000.0,
        change_pct_24h=2.0,
        bid=49999.0,
        ask=50001.0,
        spread=2.0,
        timestamp=datetime.now()
    )

@pytest.mark.asyncio
async def test_create_market_geoid(mock_kimera_system, sample_market_data):
    """Test market geoid creation from market data.
    
    Verifies that:
    1. Analyzer creates a valid GeoidState from market data
    2. Geoid contains correct semantic state mapping
    3. Key market metrics are properly transferred
    
    Args:
        mock_kimera_system: Mock trading system fixture
        sample_market_data: Sample market data fixture
        
    Raises:
        AssertionError: If any verification fails
    """
    logger.debug("Starting test_create_market_geoid")
    analyzer = KimeraSemanticMarketAnalyzer(mock_kimera_system)
    logger.debug("Created analyzer instance")
    
    geoid = await analyzer.create_market_geoid(sample_market_data)
    logger.debug(f"Created market geoid: {geoid.geoid_id}")
    
    assert geoid is not None
    assert isinstance(geoid, GeoidState)
    assert "BTCUSDT" in geoid.geoid_id
    assert geoid.semantic_state['price'] == 50000.0
    assert geoid.semantic_state['volume'] == 1000.0
    
    logger.debug("test_create_market_geoid assertions passed")

@pytest.mark.asyncio
async def test_detect_contradictions(mock_kimera_system, sample_market_data):
    logger.debug("Starting test_detect_contradictions")
    analyzer = KimeraSemanticMarketAnalyzer(mock_kimera_system)
    
    # Create first geoid with bullish sentiment
    data1 = dataclasses.replace(sample_market_data)
    data1.price = 50000.0
    data1.timestamp = datetime(2025, 1, 1, 12, 0, 0)  # Unique timestamp
    data1.semantic_state = {
        "sentiment": "bullish",
        "trend": "upward",
        "volatility": "low"
    }
    geoid1 = await analyzer.create_market_geoid(data1)
    logger.debug(f"Created bullish geoid: {geoid1.geoid_id}")
    
    # Create second geoid with bearish sentiment
    data2 = dataclasses.replace(sample_market_data)
    data2.price = 55000.0
    data2.timestamp = datetime(2025, 1, 1, 12, 1, 0)  # Unique timestamp 1 minute later
    data2.semantic_state = {
        "sentiment": "bearish",
        "trend": "downward",
        "volatility": "high"
    }
    geoid2 = await analyzer.create_market_geoid(data2)
    logger.debug(f"Created bearish geoid: {geoid2.geoid_id}")
    
    contradictions = await analyzer.detect_semantic_contradictions(
        [geoid1, geoid2], {}, []
    )
    logger.debug(f"Found {len(contradictions)} contradictions")
    
    assert len(contradictions) > 0, "Should detect contradictions between bullish/bearish states"
    assert contradictions[0].confidence > 0.5, "Contradiction confidence should be significant"
    logger.debug("test_detect_contradictions assertions passed")
