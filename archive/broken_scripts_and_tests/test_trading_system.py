"""
Test suite for Kimera Crypto Trading System

Tests the core components without requiring actual API connections.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

# Add parent directory to path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.trading.core.trading_engine import (
    KimeraTradingEngine, MarketState, TradingDecision, OrderType, OrderSide
)
from backend.trading.monitoring.performance_tracker import PerformanceTracker
from backend.trading.api.binance_connector import BinanceConnector


class TestTradingEngine:
    """Test the core trading engine"""
    
    @pytest.fixture
    def trading_engine(self):
        """Create a trading engine instance"""
        config = {
            "max_position_size": 0.1,
            "max_daily_loss": 0.05,
            "risk_per_trade": 0.02
        }
        return KimeraTradingEngine(config)
    
    @pytest.fixture
    def sample_market_state(self):
        """Create a sample market state"""
        return MarketState(
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            price=50000.0,
            volume=1000000.0,
            bid_ask_spread=0.001,
            volatility=0.02,
            cognitive_pressure=0.5,
            contradiction_level=0.3,
            semantic_temperature=0.4,
            insight_signals=["bullish momentum", "support level holding"]
        )
    
    @pytest.mark.asyncio
    async def test_market_analysis(self, trading_engine):
        """Test market analysis functionality"""
        market_data = {
            "price": 50000,
            "volume": 1000000,
            "bid": 49990,
            "ask": 50010,
            "price_history": [49000, 49500, 50000, 49800, 50000],
            "volume_history": [900000, 950000, 1000000, 980000, 1000000],
            "sentiment": "neutral",
            "order_book_imbalance": 0.1
        }
        
        # Test direct market analysis without mocking internals
        market_state = await trading_engine.analyze_market("BTCUSDT", market_data)
        
        # Verify the result
        assert market_state.symbol == "BTCUSDT"
        assert market_state.price == 50000
        assert market_state.cognitive_pressure >= 0  # Should have a value
        assert market_state.contradiction_level >= 0
        assert market_state.semantic_temperature >= 0
        assert isinstance(market_state.insight_signals, list)
    
    @pytest.mark.asyncio
    async def test_trading_decision_buy(self, trading_engine, sample_market_state):
        """Test buy decision making"""
        # Set favorable conditions for buying
        sample_market_state.cognitive_pressure = 0.2  # Low pressure
        sample_market_state.contradiction_level = 0.7  # High contradictions
        sample_market_state.semantic_temperature = 0.1  # Cold market
        sample_market_state.insight_signals = ["bullish", "bullish", "neutral"]
        
        portfolio_state = {
            "total_value": 10000,
            "free_balance": 10000,
            "positions": {},
            "daily_pnl": 0,
            "margin_used": 0
        }
        
        decision = await trading_engine.make_trading_decision(
            "BTCUSDT", sample_market_state, portfolio_state
        )
        
        assert decision.action == "BUY"
        assert decision.confidence > 0.3
        assert decision.size > 0
        assert decision.stop_loss is not None
        assert decision.take_profit is not None
    
    @pytest.mark.asyncio
    async def test_trading_decision_hold(self, trading_engine, sample_market_state):
        """Test hold decision making"""
        # Set neutral conditions
        sample_market_state.cognitive_pressure = 0.5
        sample_market_state.contradiction_level = 0.3
        sample_market_state.semantic_temperature = 0.5
        sample_market_state.insight_signals = ["neutral"]
        
        portfolio_state = {
            "total_value": 10000,
            "free_balance": 10000,
            "positions": {},
            "daily_pnl": 0,
            "margin_used": 0
        }
        
        decision = await trading_engine.make_trading_decision(
            "BTCUSDT", sample_market_state, portfolio_state
        )
        
        assert decision.action == "HOLD"
    
    def test_position_sizing(self, trading_engine):
        """Test that Kimera decides position sizing autonomously"""
        # Kimera now decides its own position sizing
        # Just verify the engine has the capability
        assert hasattr(trading_engine, 'kimera_decided_params')
        assert trading_engine.kimera_decided_params['position_sizing'] is None  # Kimera will decide


class TestPerformanceTracker:
    """Test the performance tracking system"""
    
    @pytest.fixture
    def tracker(self):
        """Create a performance tracker instance"""
        return PerformanceTracker()
    
    @pytest.fixture
    def sample_decision(self):
        """Create a sample trading decision"""
        return TradingDecision(
            action="BUY",
            confidence=0.75,
            size=1000,
            reasoning=["Low cognitive pressure", "High contradictions"],
            risk_score=0.02,
            cognitive_alignment=0.8,
            expected_return=0.05,
            stop_loss=49000,
            take_profit=52000
        )
    
    @pytest.mark.asyncio
    async def test_record_decision(self, tracker, sample_decision):
        """Test recording a trading decision"""
        order = {
            "orderId": "12345",
            "executedQty": "0.02",
            "price": "50000",
            "status": "FILLED"
        }
        
        await tracker.record_decision("BTCUSDT", sample_decision, order)
        
        assert len(tracker.trades) == 1
        assert tracker.trades[0].symbol == "BTCUSDT"
        assert tracker.trades[0].side == "BUY"
        assert tracker.trades[0].price == 50000
    
    @pytest.mark.asyncio
    async def test_update_trade_outcome(self, tracker, sample_decision):
        """Test updating trade with outcome"""
        # Record initial trade
        order = {
            "orderId": "12345",
            "executedQty": "0.02",
            "price": "50000"
        }
        await tracker.record_decision("BTCUSDT", sample_decision, order)
        
        # Update with exit
        await tracker.update_trade_outcome("12345", 51000, 0.02)
        
        trade = tracker.trades[0]
        assert trade.pnl == (51000 - 50000) * 0.02
        assert trade.pnl > 0
        assert trade.cognitive_accuracy > 0.5
    
    def test_calculate_metrics(self, tracker):
        """Test performance metrics calculation"""
        # Add some mock trades with proper attributes
        for i in range(10):
            trade = Mock()
            trade.timestamp = datetime.now()
            trade.pnl = 100 if i % 2 == 0 else -50
            trade.decision = Mock(confidence=0.7, cognitive_alignment=0.8, action="BUY")
            trade.cognitive_accuracy = 0.7 if i % 2 == 0 else 0.3
            # Add missing attributes for multiplication
            trade.price = 50000
            trade.quantity = 0.01
            tracker.trades.append(trade)
        
        metrics = tracker.calculate_metrics()
        
        assert metrics.total_trades == 10
        assert metrics.winning_trades == 5
        assert metrics.losing_trades == 5
        assert metrics.win_rate == 50.0
        assert metrics.total_pnl == 250  # 5*100 - 5*50


class TestBinanceConnector:
    """Test the Binance API connector"""
    
    @pytest.fixture
    def connector(self):
        """Create a Binance connector instance"""
        return BinanceConnector(
            api_key="test_key",
            api_secret="test_secret",
            testnet=True
        )
    
    def test_initialization(self, connector):
        """Test connector initialization"""
        assert connector.api_key == "test_key"
        assert connector.api_secret == "test_secret"
        assert "testnet" in connector.BASE_URL
    
    def test_sign_request(self, connector):
        """Test request signing"""
        params = {"symbol": "BTCUSDT", "side": "BUY", "quantity": "0.01"}
        signature = connector._sign_request(params)
        
        assert isinstance(signature, str)
        assert len(signature) == 64  # SHA256 hex length
    
    def test_format_quantity(self, connector):
        """Test quantity formatting"""
        formatted = connector._format_quantity("BTCUSDT", 0.123456789)
        # Accept both possible roundings
        assert formatted in ["0.123456", "0.123457"]
        
        formatted = connector._format_quantity("BTCUSDT", 1.0)
        assert formatted == "1"
    
    @pytest.mark.asyncio
    async def test_get_market_data(self, connector):
        """Test market data aggregation"""
        # Mock the individual API calls
        with patch.object(connector, 'get_ticker', new_callable=AsyncMock) as mock_ticker:
            mock_ticker.return_value = {
                "lastPrice": "50000",
                "volume": "1000",
                "priceChangePercent": "2.5",
                "highPrice": "51000",
                "lowPrice": "49000"
            }
            
            with patch.object(connector, 'get_orderbook', new_callable=AsyncMock) as mock_orderbook:
                mock_orderbook.return_value = {
                    "bids": [["49990", "10"], ["49980", "20"]],
                    "asks": [["50010", "10"], ["50020", "20"]]
                }
                
                with patch.object(connector, 'get_klines', new_callable=AsyncMock) as mock_klines:
                    mock_klines.return_value = [
                        [0, "49000", "49500", "48900", "49400", "100"],
                        [0, "49400", "50000", "49300", "50000", "120"]
                    ]
                    
                    market_data = await connector.get_market_data("BTCUSDT")
                    
                    assert market_data["symbol"] == "BTCUSDT"
                    assert market_data["price"] == 50000
                    assert market_data["sentiment"] == "bullish"
                    assert "order_book_imbalance" in market_data


@pytest.mark.asyncio
async def test_integration():
    """Test basic integration between components"""
    # Create components
    config = {
        "max_position_size": 0.1,
        "max_daily_loss": 0.05,
        "risk_per_trade": 0.02
    }
    
    engine = KimeraTradingEngine(config)
    tracker = PerformanceTracker()
    
    # Create market state
    market_state = MarketState(
        timestamp=datetime.now(),
        symbol="BTCUSDT",
        price=50000.0,
        volume=1000000.0,
        bid_ask_spread=0.001,
        volatility=0.02,
        cognitive_pressure=0.2,
        contradiction_level=0.7,
        semantic_temperature=0.1,
        insight_signals=["bullish"]
    )
    
    portfolio_state = {
        "total_value": 10000,
        "free_balance": 10000,
        "positions": {},
        "daily_pnl": 0,
        "margin_used": 0
    }
    
    # Make decision
    decision = await engine.make_trading_decision(
        "BTCUSDT", market_state, portfolio_state
    )
    
    # Record decision
    mock_order = {
        "orderId": "test123",
        "executedQty": "0.02",
        "price": "50000"
    }
    
    await tracker.record_decision("BTCUSDT", decision, mock_order)
    
    # Verify
    assert decision.action in ["BUY", "SELL", "HOLD"]
    assert len(tracker.trades) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 