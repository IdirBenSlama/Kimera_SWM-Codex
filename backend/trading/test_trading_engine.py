"""Tests for KimeraIntegratedTradingEngine"""
import pytest
import asyncio
from datetime import datetime
from unittest.mock import MagicMock, patch
from backend.trading.kimera_integrated_trading_system import (
    KimeraIntegratedTradingEngine,
    KimeraMarketData,
    KimeraCognitiveSignal,
    TradingStrategy,
    MarketRegime
)
from backend.utils.kimera_logger import get_trading_logger
logger = get_trading_logger(__name__)

@pytest.fixture
def mock_config():
    return {
        'starting_capital': 1000.0,
        'max_position_size': 0.2,
        'trading_symbols': ['BTCUSDT'],
        'market_data_interval': 1,
        'semantic_analysis_interval': 1,
        'signal_generation_interval': 1,
        'position_management_interval': 1,
        'enable_vault_protection': False
    }

@pytest.fixture
def mock_kimera_system():
    system = MagicMock()
    system.get_contradiction_engine.return_value = MagicMock()
    system.get_thermodynamics_engine.return_value = MagicMock()
    system.get_vault_manager.return_value = None
    system.get_gpu_foundation.return_value = MagicMock()
    system._initialization_complete = True
    return system

@pytest.fixture
def sample_signal():
    return KimeraCognitiveSignal(
        signal_id="test_signal",
        symbol="BTCUSDT",
        action="buy",
        confidence=0.8,
        conviction=0.7,
        reasoning=["Strong bullish momentum"],
        strategy=TradingStrategy.SEMANTIC_CONTRADICTION,
        market_regime=MarketRegime.BULL_STRONG,
        suggested_allocation_pct=0.1,
        entry_price=50000.0,
        stop_loss=49000.0,
        profit_targets=[51000.0, 52000.0]
    )

@pytest.mark.asyncio
async def test_engine_initialization(mock_config, mock_kimera_system):
    logger.debug("Starting test_engine_initialization")
    with patch('backend.trading.kimera_integrated_trading_system.get_kimera_system',
              return_value=mock_kimera_system):
        engine = KimeraIntegratedTradingEngine(mock_config)
        logger.debug("Engine initialized")
        
        assert engine.is_running == False
        assert len(engine.positions) == 0
        assert len(engine.active_signals) == 0
        
        logger.debug("test_engine_initialization assertions passed")

@pytest.mark.asyncio
async def test_market_data_processing(mock_config, mock_kimera_system):
    with patch('backend.trading.kimera_integrated_trading_system.get_kimera_system',
              return_value=mock_kimera_system):
        engine = KimeraIntegratedTradingEngine(mock_config)
        engine.is_running = True
        
        # Simulate market data update
        await engine._market_data_loop()
        
        assert 'BTCUSDT' in engine.market_data_cache
        assert len(engine.market_data_cache['BTCUSDT']) > 0
        assert 'BTCUSDT' in engine.market_geoids

@pytest.mark.asyncio
async def test_signal_generation(mock_config, mock_kimera_system, sample_signal):
    with patch('backend.trading.kimera_integrated_trading_system.get_kimera_system',
              return_value=mock_kimera_system):
        engine = KimeraIntegratedTradingEngine(mock_config)
        engine.is_running = True
        
        # Mock market data
        engine.market_data_cache['BTCUSDT'] = [MagicMock(price=50000.0)]
        
        # Mock signal generation
        with patch.object(engine, '_generate_kimera_signal', 
                         return_value=sample_signal):
            await engine._signal_generation_loop()
            
            assert 'BTCUSDT' in engine.active_signals
            assert engine.active_signals['BTCUSDT'].action == "buy"

@pytest.mark.asyncio
async def test_position_management(mock_config, mock_kimera_system, sample_signal):
    logger.debug("Starting test_position_management")
    with patch('backend.trading.kimera_integrated_trading_system.get_kimera_system',
              return_value=mock_kimera_system):
        engine = KimeraIntegratedTradingEngine(mock_config)
        engine.is_running = True
        logger.debug("Engine started")
        
        # Add active signal
        engine.active_signals['BTCUSDT'] = sample_signal
        logger.debug(f"Added signal for {sample_signal.symbol}")
        
        # Mock position creation
        await engine._position_management_loop()
        logger.debug("Position management loop completed")
        
        # Should create a position from the signal
        assert len(engine.positions) == 1
        position = next(iter(engine.positions.values()))
        logger.debug(f"Created position: {position.symbol}")
        
        assert position.symbol == "BTCUSDT"
        assert position.is_active == True
        logger.debug("test_position_management assertions passed")
