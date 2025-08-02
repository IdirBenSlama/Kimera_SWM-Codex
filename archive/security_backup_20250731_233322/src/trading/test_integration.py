"""End-to-end tests for Kimera trading system integration"""
import pytest
import asyncio
from datetime import datetime
from unittest.mock import MagicMock, patch
from src.trading.kimera_integrated_trading_system import (
    KimeraIntegratedTradingEngine,
    KimeraSemanticMarketAnalyzer,
    KimeraMarketData,
    KimeraCognitiveSignal,
    TradingStrategy,
    MarketRegime
)
from src.utils.kimera_logger import get_trading_logger
logger = get_trading_logger(__name__)

@pytest.fixture
def mock_config():
    return {
        'starting_capital': 10000.0,
        'max_position_size': 0.3,
        'trading_symbols': ['BTCUSDT', 'ETHUSDT'],
        'market_data_interval': 0.5,
        'semantic_analysis_interval': 0.5,
        'signal_generation_interval': 0.5,
        'position_management_interval': 0.5,
        'enable_vault_protection': True
    }

@pytest.fixture
def mock_kimera_system():
    system = MagicMock()
    system.get_contradiction_engine.return_value = MagicMock()
    system.get_thermodynamics_engine.return_value = MagicMock()
    system.get_vault_manager.return_value = MagicMock()
    system.get_gpu_foundation.return_value = MagicMock()
    system._initialization_complete = True
    return system

@pytest.fixture
def sample_market_data():
    return [
        KimeraMarketData(
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
        ),
        KimeraMarketData(
            symbol="ETHUSDT",
            price=3000.0,
            volume=5000.0,
            high_24h=3100.0,
            low_24h=2900.0,
            change_24h=100.0,
            change_pct_24h=3.33,
            bid=2999.0,
            ask=3001.0,
            spread=2.0,
            timestamp=datetime.now()
        )
    ]

@pytest.mark.asyncio
async def test_full_trading_cycle(mock_config, mock_kimera_system, sample_market_data):
    """Test complete trading cycle from market data to position creation"""
    logger.debug("Starting test_full_trading_cycle")
    with patch('backend.trading.kimera_integrated_trading_system.get_kimera_system',
              return_value=mock_kimera_system):
        # Initialize components
        engine = KimeraIntegratedTradingEngine(mock_config)
        analyzer = KimeraSemanticMarketAnalyzer(mock_kimera_system)
        logger.debug("Initialized engine and analyzer")
        
        # Start the engine
        engine.is_running = True
        logger.debug("Engine started")
        
        # Simulate market data updates
        for data in sample_market_data:
            geoid = await analyzer.create_market_geoid(data)
            engine.market_data_cache[data.symbol] = [data]
            engine.market_geoids[data.symbol] = geoid
        
        # Verify market data processing
        assert len(engine.market_data_cache) == 2
        assert len(engine.market_geoids) == 2
        
        # Mock signal generation
        test_signal = KimeraCognitiveSignal(
            signal_id="test_signal",
            symbol="BTCUSDT",
            action="buy",
            confidence=0.85,
            conviction=0.75,
            reasoning=["Strong bullish divergence"],
            strategy=TradingStrategy.SEMANTIC_CONTRADICTION,
            market_regime=MarketRegime.BULL_STRONG,
            suggested_allocation_pct=0.2,
            entry_price=50000.0,
            stop_loss=49000.0,
            profit_targets=[51000.0, 52000.0]
        )
        
        with patch.object(engine, '_generate_kimera_signal', 
                         return_value=test_signal):
            await engine._signal_generation_loop()
            
            # Verify signal was generated
            assert 'BTCUSDT' in engine.active_signals
            assert engine.active_signals['BTCUSDT'].action == "buy"
            
            # Process position management
            await engine._position_management_loop()
            logger.debug("Position management loop completed")
            
            # Verify position was created
            assert len(engine.positions) == 1
            position = next(iter(engine.positions.values()))
            logger.debug(f"Created position: {position.symbol}")
            
            assert position.symbol == "BTCUSDT"
            assert position.entry_price == 50000.0
            assert position.size_pct == 0.2
            logger.debug("test_full_trading_cycle assertions passed")

@pytest.mark.asyncio
async def test_vault_protection(mock_config, mock_kimera_system, sample_market_data):
    """Test vault protection mechanisms during trading"""
    with patch('backend.trading.kimera_integrated_trading_system.get_kimera_system',
              return_value=mock_kimera_system):
        engine = KimeraIntegratedTradingEngine(mock_config)
        analyzer = KimeraSemanticMarketAnalyzer(mock_kimera_system)
        engine.is_running = True
        
        # Create test data with high volatility
        volatile_data = sample_market_data[0]
        volatile_data.price = 40000.0  # 20% drop from previous price
        
        geoid = await analyzer.create_market_geoid(volatile_data)
        engine.market_data_cache['BTCUSDT'] = [volatile_data]
        engine.market_geoids['BTCUSDT'] = geoid
        
        # Mock vault protection triggering
        mock_kimera_system.get_vault_manager.return_value.check_protection_conditions.return_value = True
        
        # This should trigger vault protection
        await engine._market_data_loop()
        
        # Verify vault protection was activated
        mock_kimera_system.get_vault_manager.return_value.activate_protection.assert_called_once()
        assert engine.protection_active == True
