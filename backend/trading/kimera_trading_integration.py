"""
KIMERA Trading Integration Module
=================================

Main integration point that connects all trading components and provides
a unified interface for Kimera's semantic thermodynamic reactor.

This module orchestrates:
- Semantic Trading Reactor for contradiction detection
- Execution Bridge for order management
- Monitoring Dashboard for real-time visibility
- Data infrastructure for market feeds
- Risk and compliance management
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import json

# Import all trading components
from backend.trading.core.semantic_trading_reactor import (
    SemanticTradingReactor, 
    TradingRequest, 
    TradingResult,
    create_semantic_trading_reactor
)
from backend.trading.execution.semantic_execution_bridge import (
    SemanticExecutionBridge,
    ExecutionRequest,
    ExecutionResult,
    OrderType,
    create_semantic_execution_bridge
)
from backend.trading.monitoring.semantic_trading_dashboard import (
    SemanticTradingDashboard,
    TradingMetrics,
    SystemHealth,
    create_semantic_trading_dashboard
)
from backend.trading.data.database import QuestDBManager
from backend.trading.data.stream import KafkaManager
from backend.trading.connectors.data_providers import YahooFinanceConnector

# Import sentiment analysis components
from backend.trading.intelligence.market_sentiment_analyzer import MarketSentimentAnalyzer
from backend.trading.intelligence.news_feed_processor import NewsFeedProcessor

logger = logging.getLogger(__name__)


@dataclass
class KimeraTradingConfig:
    """Configuration for the integrated trading system"""
    # Core settings
    tension_threshold: float = 0.4
    max_position_size: float = 10000
    risk_per_trade: float = 0.02
    
    # Infrastructure
    questdb_host: str = "localhost"
    questdb_port: int = 9009
    kafka_servers: str = "localhost:9092"
    
    # Exchange configuration
    exchanges: Dict[str, Dict[str, Any]] = None
    
    # Monitoring
    dashboard_port: int = 8050
    prometheus_port: int = 9090
    
    # Features
    enable_paper_trading: bool = True
    enable_sentiment_analysis: bool = True
    enable_news_processing: bool = True
    
    # API Keys
    taapi_api_key: Optional[str] = None


class KimeraTradingIntegration:
    """
    Main integration class that orchestrates all trading components
    and provides a unified interface for Kimera's reactor.
    """
    
    def __init__(self, config: KimeraTradingConfig):
        """
        Initialize the integrated trading system
        
        Args:
            config: System configuration
        """
        self.config = config
        self.is_running = False
        
        logger.info("🚀 Initializing KIMERA Trading Integration")
        
        # Initialize core components
        self._initialize_components()
        
        # System state
        self.active_strategies = {}
        self.performance_history = []
        
        logger.info("✅ KIMERA Trading Integration ready")
    
    def _initialize_components(self):
        """Initialize all trading components"""
        # Semantic Trading Reactor
        reactor_config = {
            'tension_threshold': self.config.tension_threshold,
            'questdb_host': self.config.questdb_host,
            'questdb_port': self.config.questdb_port,
            'kafka_servers': self.config.kafka_servers
        }
        self.trading_reactor = create_semantic_trading_reactor(reactor_config)
        
        # Execution Bridge
        execution_config = {
            'exchanges': self.config.exchanges or {},
            'max_order_size': self.config.max_position_size,
            'max_daily_volume': self.config.max_position_size * 100
        }
        self.execution_bridge = create_semantic_execution_bridge(execution_config)
        
        # Monitoring Dashboard
        self.dashboard = create_semantic_trading_dashboard({})
        
        # Data Infrastructure
        self.db_manager = QuestDBManager(
            host=self.config.questdb_host,
            port=self.config.questdb_port
        )
        self.kafka_manager = KafkaManager(
            bootstrap_servers=self.config.kafka_servers
        )
        
        # Market Intelligence
        if self.config.enable_sentiment_analysis:
            self.sentiment_analyzer = MarketSentimentAnalyzer()
        else:
            self.sentiment_analyzer = None
            
        if self.config.enable_news_processing:
            # Initialize news processor with CryptoPanic support
            news_config = {
                'cryptopanic_api_key': self.config.exchanges.get('cryptopanic_api_key') if self.config.exchanges else None,
                'cryptopanic_testnet': True
            }
            self.news_processor = NewsFeedProcessor(news_config)
        else:
            self.news_processor = None
        
        logger.info("   ✓ Trading Reactor initialized")
        logger.info("   ✓ Execution Bridge initialized")
        logger.info("   ✓ Monitoring Dashboard initialized")
        logger.info(f"   ✓ Sentiment Analysis: {'enabled' if self.sentiment_analyzer else 'disabled'}")
        logger.info(f"   ✓ News Processing: {'enabled' if self.news_processor else 'disabled'}")
    
    async def process_market_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a market event through the entire trading pipeline
        
        This is the main entry point for Kimera's reactor to send market data
        
        Args:
            event: Market event containing price data, news, etc.
            
        Returns:
            Processing result with actions taken
        """
        try:
            # 1. Enrich event with semantic context
            semantic_context = await self._build_semantic_context(event)
            
            # 2. Create trading request
            trading_request = TradingRequest(
                action_type='analyze',
                market_data=event.get('market_data', {}),
                semantic_context=semantic_context,
                risk_parameters={
                    'max_position_size': self.config.max_position_size,
                    'risk_per_trade': self.config.risk_per_trade
                }
            )
            
            # 3. Process through trading reactor
            analysis_result = await self.trading_reactor.process_request(trading_request)
            
            # 4. Log contradictions to dashboard
            for contradiction in analysis_result.contradiction_map:
                self.dashboard.log_contradiction(contradiction)
            
            # 5. Execute trades if confidence is high
            if analysis_result.confidence > 0.6 and analysis_result.action_taken in ['buy', 'sell', 'short']:
                execution_result = await self._execute_trading_decision(analysis_result, event)
                
                # 6. Update dashboard
                await self._update_dashboard(analysis_result, execution_result)
                
                return {
                    'status': 'executed',
                    'analysis': analysis_result,
                    'execution': execution_result,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                # Just monitoring
                await self._update_dashboard(analysis_result, None)
                
                return {
                    'status': 'monitored',
                    'analysis': analysis_result,
                    'reason': 'insufficient_confidence' if analysis_result.confidence <= 0.6 else 'no_action_needed',
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error processing market event: {e}", exc_info=True)
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _build_semantic_context(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Build semantic context from various data sources"""
        context = {}
        
        # Get sentiment analysis if available
        if self.sentiment_analyzer and 'symbol' in event.get('market_data', {}):
            symbol = event['market_data']['symbol']
            
            # News sentiment
            if self.news_processor:
                news_sentiment = await self.news_processor.get_sentiment(symbol)
                context['news_sentiment'] = news_sentiment.get('score', 0)
                context['news_volume'] = news_sentiment.get('volume', 0)
            
            # Social sentiment
            social_sentiment = await self.sentiment_analyzer.analyze_social_media(symbol)
            context['social_sentiment'] = social_sentiment.get('score', 0)
            context['social_volume'] = social_sentiment.get('volume', 0)
            context['viral_coefficient'] = social_sentiment.get('viral_coefficient', 0)
        
        # Add any additional context from the event
        if 'context' in event:
            context.update(event['context'])
        
        return context
    
    async def _execute_trading_decision(self, 
                                      analysis: TradingResult, 
                                      event: Dict[str, Any]) -> ExecutionResult:
        """Execute the trading decision through the execution bridge"""
        market_data = event.get('market_data', {})
        
        # Determine order type based on market conditions
        order_type = OrderType.MARKET  # Default to market order
        price = None
        
        if analysis.semantic_analysis.get('volatility', 0) > 0.02:
            # High volatility - use limit order
            order_type = OrderType.LIMIT
            current_price = market_data.get('price', 0)
            
            if analysis.action_taken == 'buy':
                price = current_price * 0.999  # Slightly below market
            else:
                price = current_price * 1.001  # Slightly above market
        
        # Create execution request
        execution_request = ExecutionRequest(
            order_id=f"kimera_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            symbol=market_data.get('symbol', 'UNKNOWN'),
            side=analysis.action_taken if analysis.action_taken != 'short' else 'sell',
            quantity=self._calculate_position_size(analysis),
            order_type=order_type,
            price=price,
            metadata={
                'semantic_score': analysis.confidence,
                'contradiction_ids': [c['contradiction_id'] for c in analysis.contradiction_map]
            }
        )
        
        # Execute through bridge
        execution_result = await self.execution_bridge.execute_order(execution_request)
        
        # Log trade
        self.dashboard.log_trade({
            'order_id': execution_request.order_id,
            'symbol': execution_request.symbol,
            'side': execution_request.side,
            'size': execution_request.quantity,
            'price': execution_result.average_price,
            'execution_time': execution_result.execution_time,
            'semantic_score': analysis.confidence
        })
        
        return execution_result
    
    def _calculate_position_size(self, analysis: TradingResult) -> float:
        """Calculate position size based on analysis confidence and risk parameters"""
        base_size = self.config.max_position_size * self.config.risk_per_trade
        
        # Scale by confidence
        position_size = base_size * analysis.confidence
        
        # Apply Kelly criterion adjustment based on win rate
        if self.performance_history:
            recent_trades = self.performance_history[-20:]
            win_rate = sum(1 for t in recent_trades if t.get('pnl', 0) > 0) / len(recent_trades)
            
            # Kelly fraction
            kelly_fraction = (win_rate - (1 - win_rate)) / 1  # Assuming 1:1 payoff
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
            
            position_size *= kelly_fraction
        
        return max(10, min(position_size, self.config.max_position_size))  # Apply bounds
    
    async def _update_dashboard(self, analysis: TradingResult, execution: Optional[ExecutionResult]):
        """Update the monitoring dashboard with latest metrics"""
        # Calculate current metrics
        total_pnl = sum(t.get('pnl', 0) for t in self.performance_history)
        win_rate = 0.0
        if self.performance_history:
            wins = sum(1 for t in self.performance_history if t.get('pnl', 0) > 0)
            win_rate = wins / len(self.performance_history)
        
        # Create metrics update
        metrics = TradingMetrics(
            timestamp=datetime.now(),
            total_pnl=total_pnl,
            daily_pnl=sum(t.get('pnl', 0) for t in self.performance_history[-50:]),  # Approximate daily
            win_rate=win_rate,
            sharpe_ratio=self._calculate_sharpe_ratio(),
            max_drawdown=self._calculate_max_drawdown(),
            active_positions=len(self.trading_reactor.active_positions),
            total_volume=sum(t.get('size', 0) for t in self.performance_history),
            contradiction_count=len(analysis.contradiction_map),
            semantic_pressure=analysis.semantic_analysis.get('thermodynamic_pressure', 0)
        )
        
        self.dashboard.update_metrics(metrics)
        
        # Update system health
        health = SystemHealth(
            cpu_usage=30.0,  # Placeholder - would get from system
            memory_usage=45.0,  # Placeholder
            latency_ms=execution.execution_time * 1000 if execution else 0,
            error_rate=0.01,  # Placeholder
            uptime_hours=24.0,  # Placeholder
            connected_exchanges=len(self.execution_bridge.exchanges),
            data_feed_status={'market_data': True, 'news': True}  # Placeholder
        )
        
        self.dashboard.update_system_health(health)
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio from performance history"""
        if len(self.performance_history) < 2:
            return 0.0
        
        returns = [t.get('pnl', 0) / self.config.max_position_size for t in self.performance_history]
        
        if not returns:
            return 0.0
        
        import numpy as np
        return np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)  # Annualized
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from performance history"""
        if not self.performance_history:
            return 0.0
        
        cumulative_pnl = []
        running_total = 0
        
        for trade in self.performance_history:
            running_total += trade.get('pnl', 0)
            cumulative_pnl.append(running_total)
        
        if not cumulative_pnl:
            return 0.0
        
        peak = cumulative_pnl[0]
        max_dd = 0
        
        for value in cumulative_pnl:
            if value > peak:
                peak = value
            drawdown = (peak - value) / (peak + 1e-6)
            max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    async def start(self):
        """Start the integrated trading system"""
        if self.is_running:
            logger.warning("Trading system is already running")
            return
        
        logger.info("Starting KIMERA Trading System...")
        
        # Connect to data sources
        self.db_manager.connect()
        self.kafka_manager.connect_producer()
        self.kafka_manager.connect_consumer(['market_data', 'news', 'sentiment'])
        
        # Start monitoring
        asyncio.create_task(self.execution_bridge.monitor_active_orders())
        
        # Start dashboard in background
        asyncio.create_task(self._run_dashboard())
        
        self.is_running = True
        logger.info("✅ KIMERA Trading System started successfully")
    
    async def stop(self):
        """Stop the integrated trading system"""
        if not self.is_running:
            logger.warning("Trading system is not running")
            return
        
        logger.info("Stopping KIMERA Trading System...")
        
        # Close all positions
        await self._close_all_positions()
        
        # Disconnect from data sources
        self.db_manager.close()
        self.kafka_manager.close()
        
        self.is_running = False
        logger.info("✅ KIMERA Trading System stopped")
    
    async def _close_all_positions(self):
        """Close all active positions"""
        for position_id, position in self.trading_reactor.active_positions.items():
            logger.info(f"Closing position {position_id}")
            # Implementation would depend on position structure
    
    async def _run_dashboard(self):
        """Run the monitoring dashboard"""
        await asyncio.to_thread(
            self.dashboard.run_dashboard,
            port=self.config.dashboard_port
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            'is_running': self.is_running,
            'active_positions': len(self.trading_reactor.active_positions),
            'total_trades': len(self.performance_history),
            'performance_summary': self.dashboard.get_performance_summary(),
            'execution_analytics': self.execution_bridge.get_execution_analytics(),
            'connected_exchanges': list(self.execution_bridge.exchanges.keys()),
            'timestamp': datetime.now().isoformat()
        }


def create_kimera_trading_system(config: Optional[Dict[str, Any]] = None) -> KimeraTradingIntegration:
    """
    Factory function to create the integrated KIMERA trading system
    
    Args:
        config: System configuration dictionary
        
    Returns:
        KimeraTradingIntegration instance
    """
    if config is None:
        config = {}
    
    trading_config = KimeraTradingConfig(**config)
    
    return KimeraTradingIntegration(trading_config)


# Convenience function for Kimera's reactor to use
async def process_trading_opportunity(market_event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simple interface for Kimera's reactor to process trading opportunities
    
    Args:
        market_event: Market event data
        
    Returns:
        Processing result
    """
    # Get or create singleton instance
    if not hasattr(process_trading_opportunity, '_instance'):
        process_trading_opportunity._instance = create_kimera_trading_system()
        await process_trading_opportunity._instance.start()
    
    return await process_trading_opportunity._instance.process_market_event(market_event) 