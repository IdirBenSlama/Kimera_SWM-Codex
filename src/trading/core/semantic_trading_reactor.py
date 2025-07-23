"""
KIMERA Semantic Trading Reactor
==============================

A plug-and-play trading module that interfaces with KIMERA's semantic thermodynamic reactor
to identify market contradictions and execute autonomous trading strategies.

This module leverages KIMERA's unique ability to detect semantic breaches and contradictions
across multiple data streams to find trading opportunities that traditional systems miss.

Architecture:
- Interfaces with KIMERA's core reactor for contradiction detection
- Manages real-time market data ingestion and analysis
- Executes trades based on semantic thermodynamic principles
- Provides enterprise-grade monitoring and compliance
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import os

# Import KIMERA core components
try:
    from src.engines.contradiction_engine import ContradictionEngine, TensionGradient
    from src.engines.thermodynamics import SemanticThermodynamicsEngine
    from src.core.geoid import GeoidState
    from src.core.insight import InsightScar
    from src.core.native_math import NativeMath
    KIMERA_CORE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"KIMERA core components not available: {e}")
    KIMERA_CORE_AVAILABLE = False

# Import trading infrastructure
try:
    from src.trading.data.database import QuestDBManager
    from src.trading.data.stream import KafkaManager
    from src.trading.connectors.data_providers import YahooFinanceConnector
    TRADING_INFRA_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Trading infrastructure not available: {e}")
    TRADING_INFRA_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class TradingRequest:
    """Request structure for the trading reactor"""
    action_type: str  # 'analyze', 'execute', 'monitor'
    market_data: Dict[str, Any]
    semantic_context: Dict[str, Any]
    risk_parameters: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TradingResult:
    """Result structure from the trading reactor"""
    action_taken: str
    position: Optional[Dict[str, Any]]
    semantic_analysis: Dict[str, float]
    contradiction_map: List[Dict[str, Any]]
    execution_time: float
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketContradiction:
    """Represents a detected market contradiction"""
    contradiction_id: str
    source_a: str  # e.g., "price_action"
    source_b: str  # e.g., "news_sentiment"
    tension_score: float
    semantic_distance: float
    opportunity_type: str  # 'long', 'short', 'arbitrage'
    timestamp: datetime


class SemanticTradingReactor:
    """
    The core trading reactor that interfaces with KIMERA's semantic engine
    to detect and exploit market contradictions autonomously.
    """
    
    def __init__(self, config: Dict[str, Any], reactor_interface: Optional[Any] = None):
        """
        Initialize the Semantic Trading Reactor
        
        Args:
            config: Configuration parameters
            reactor_interface: Interface to KIMERA's main reactor
        """
        self.config = config
        self.reactor_interface = reactor_interface
        
        # Initialize KIMERA components
        if KIMERA_CORE_AVAILABLE:
            self.contradiction_engine = ContradictionEngine(
                tension_threshold=config.get('tension_threshold', 0.4)
            )
            self.thermodynamics_engine = SemanticThermodynamicsEngine()
        else:
            logger.warning("Running without KIMERA core - limited functionality")
            self.contradiction_engine = None
            self.thermodynamics_engine = None
        
        # Initialize trading infrastructure
        if TRADING_INFRA_AVAILABLE:
            self.db_manager = QuestDBManager(
                host=config.get('questdb_host', 'localhost'),
                port=config.get('questdb_port', 9009)
            )
            self.kafka_manager = KafkaManager(
                bootstrap_servers=config.get('kafka_servers', 'localhost:9092')
            )
            self.data_provider = YahooFinanceConnector(kafka_manager=self.kafka_manager)
        else:
            logger.warning("Running without trading infrastructure")
            self.db_manager = None
            self.kafka_manager = None
            self.data_provider = None
        
        # Trading state
        self.active_positions = {}
        self.contradiction_history = []
        self.performance_metrics = {
            'total_trades': 0,
            'successful_trades': 0,
            'total_pnl': 0.0,
            'sharpe_ratio': 0.0
        }
        
        # Initialize technical analysis connector
        self.taapi_connector = None
        taapi_key = config.get('taapi_api_key') or os.getenv('TAAPI_API_KEY')
        if taapi_key:
            try:
                from src.trading.connectors.taapi_connector import create_taapi_connector
                self.taapi_connector = create_taapi_connector(taapi_key)
                logger.info("📊 TAAPI technical analysis integration enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize TAAPI: {e}")
        
        logger.info("🚀 Semantic Trading Reactor initialized")
        logger.info(f"   Contradiction detection: {'✓' if self.contradiction_engine else '✗'}")
        logger.info(f"   Market data streaming: {'✓' if self.kafka_manager else '✗'}")
        logger.info(f"   Time-series database: {'✓' if self.db_manager else '✗'}")
        logger.info(f"   Technical analysis: {'✓' if self.taapi_connector else '✗'}")
    
    async def process_request(self, request: TradingRequest) -> TradingResult:
        """
        Process a trading request through the semantic reactor
        
        Args:
            request: The trading request to process
            
        Returns:
            TradingResult with analysis and actions taken
        """
        start_time = time.time()
        
        try:
            if request.action_type == 'analyze':
                result = await self._analyze_market(request)
            elif request.action_type == 'execute':
                result = await self._execute_trade(request)
            elif request.action_type == 'monitor':
                result = await self._monitor_positions(request)
            else:
                raise ValueError(f"Unknown action type: {request.action_type}")
            
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing trading request: {e}", exc_info=True)
            return TradingResult(
                action_taken='error',
                position=None,
                semantic_analysis={},
                contradiction_map=[],
                execution_time=time.time() - start_time,
                confidence=0.0,
                metadata={'error': str(e)}
            )
    
    async def _analyze_market(self, request: TradingRequest) -> TradingResult:
        """
        Analyze market data for contradictions and opportunities
        """
        market_data = request.market_data
        semantic_context = request.semantic_context
        
        # Create geoid states from different data sources
        geoids = self._create_market_geoids(market_data, semantic_context)
        
        # Detect contradictions
        contradictions = []
        if self.contradiction_engine and len(geoids) >= 2:
            tension_gradients = self.contradiction_engine.detect_tension_gradients(geoids)
            
            for gradient in tension_gradients:
                contradiction = MarketContradiction(
                    contradiction_id=str(uuid.uuid4()),
                    source_a=gradient.geoid_a,
                    source_b=gradient.geoid_b,
                    tension_score=gradient.tension_score,
                    semantic_distance=gradient.tension_score,  # Simplified
                    opportunity_type=self._determine_opportunity_type(gradient),
                    timestamp=datetime.now()
                )
                contradictions.append(contradiction)
                self.contradiction_history.append(contradiction)
        
        # Calculate semantic metrics
        semantic_analysis = self._calculate_semantic_metrics(geoids, contradictions)
        
        # Determine trading action
        action, confidence = self._determine_trading_action(contradictions, semantic_analysis)
        
        return TradingResult(
            action_taken=action,
            position=None,
            semantic_analysis=semantic_analysis,
            contradiction_map=[self._contradiction_to_dict(c) for c in contradictions],
            execution_time=0.0,  # Will be set by caller
            confidence=confidence,
            metadata={'geoid_count': len(geoids)}
        )
    
    def _create_market_geoids(self, 
                             market_data: Dict[str, Any], 
                             semantic_context: Dict[str, Any]) -> List[GeoidState]:
        """
        Create geoid states from market data and semantic context
        """
        geoids = []
        
        # Price action geoid
        price_geoid = GeoidState(
            geoid_id=f"price_{market_data.get('symbol', 'unknown')}",
            semantic_state={
                'price': market_data.get('price', 0),
                'volume': market_data.get('volume', 0),
                'momentum': market_data.get('momentum', 0),
                'volatility': market_data.get('volatility', 0)
            },
            symbolic_state={
                'trend': market_data.get('trend', 'neutral'),
                'support': market_data.get('support', 0),
                'resistance': market_data.get('resistance', 0)
            }
        )
        geoids.append(price_geoid)
        
        # News sentiment geoid
        if 'news_sentiment' in semantic_context:
            news_geoid = GeoidState(
                geoid_id=f"news_{market_data.get('symbol', 'unknown')}",
                semantic_state={
                    'sentiment_score': semantic_context['news_sentiment'],
                    'sentiment_volume': semantic_context.get('news_volume', 0),
                    'sentiment_momentum': semantic_context.get('sentiment_momentum', 0)
                },
                symbolic_state={
                    'sentiment_direction': 'positive' if semantic_context['news_sentiment'] > 0 else 'negative',
                    'sentiment_strength': abs(semantic_context['news_sentiment'])
                }
            )
            geoids.append(news_geoid)
        
        # Social sentiment geoid
        if 'social_sentiment' in semantic_context:
            social_geoid = GeoidState(
                geoid_id=f"social_{market_data.get('symbol', 'unknown')}",
                semantic_state={
                    'social_score': semantic_context['social_sentiment'],
                    'social_volume': semantic_context.get('social_volume', 0),
                    'viral_coefficient': semantic_context.get('viral_coefficient', 0)
                },
                symbolic_state={
                    'social_trend': semantic_context.get('social_trend', 'neutral'),
                    'influencer_sentiment': semantic_context.get('influencer_sentiment', 0)
                }
            )
            geoids.append(social_geoid)
        
        # Technical indicators geoid
        if 'technical_indicators' in market_data:
            tech_geoid = GeoidState(
                geoid_id=f"technical_{market_data.get('symbol', 'unknown')}",
                semantic_state=market_data['technical_indicators'],
                symbolic_state={
                    'signal': market_data.get('technical_signal', 'neutral'),
                    'strength': market_data.get('signal_strength', 0)
                }
            )
            geoids.append(tech_geoid)
        elif self.taapi_connector and market_data.get('symbol'):
            # Fetch real-time technical indicators from TAAPI
            # Note: This is a simplified sync approach - in production use proper async
            try:
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                tech_data = loop.run_until_complete(self._fetch_technical_indicators(market_data['symbol']))
                loop.close()
            except Exception as e:
                logger.warning(f"Failed to fetch technical indicators: {e}")
                tech_data = None
            
            if tech_data:
                tech_geoid = GeoidState(
                    geoid_id=f"technical_{market_data.get('symbol', 'unknown')}",
                    semantic_state=tech_data['indicators'],
                    symbolic_state={
                        'signal': tech_data['signal'],
                        'strength': tech_data['strength']
                    }
                )
                geoids.append(tech_geoid)
        
        return geoids
    
    async def _fetch_technical_indicators(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch technical indicators from TAAPI
        """
        if not self.taapi_connector:
            return None
        
        try:
            from src.trading.connectors.taapi_connector import Indicator, Timeframe
            
            # Convert symbol format (e.g., BTC-USD to BTC/USDT)
            exchange_symbol = symbol.replace('-USD', '/USDT')
            
            # Fetch key indicators
            indicators = await self.taapi_connector.get_bulk_indicators(
                indicators=[
                    Indicator.RSI,
                    Indicator.MACD,
                    Indicator.BBands,
                    Indicator.ADX
                ],
                symbol=exchange_symbol,
                exchange='binance',
                timeframe=Timeframe.ONE_HOUR
            )
            
            # Process indicators into semantic state
            semantic_state = {}
            signal = 'neutral'
            strength = 0.5
            
            if 'rsi' in indicators:
                rsi_value = indicators['rsi'].value
                if isinstance(rsi_value, (int, float)):
                    semantic_state['rsi'] = rsi_value / 100  # Normalize to 0-1
                    if rsi_value < 30:
                        signal = 'bullish'
                        strength = 0.7
                    elif rsi_value > 70:
                        signal = 'bearish'
                        strength = 0.7
            
            if 'macd' in indicators and isinstance(indicators['macd'].value, dict):
                macd_data = indicators['macd'].value
                semantic_state['macd_signal'] = 1.0 if macd_data.get('macd', 0) > macd_data.get('signal', 0) else 0.0
            
            if 'adx' in indicators and isinstance(indicators['adx'].value, (int, float)):
                semantic_state['trend_strength'] = indicators['adx'].value / 100
                strength *= (1 + semantic_state['trend_strength'])
            
            return {
                'indicators': semantic_state,
                'signal': signal,
                'strength': min(strength, 1.0)
            }
            
        except Exception as e:
            logger.error(f"Failed to fetch technical indicators: {e}")
            return None
    
    def _determine_opportunity_type(self, gradient: TensionGradient) -> str:
        """
        Determine the type of trading opportunity from a tension gradient
        """
        # Simplified logic - in production this would be much more sophisticated
        if 'price' in gradient.geoid_a and 'news' in gradient.geoid_b:
            return 'sentiment_arbitrage'
        elif 'price' in gradient.geoid_a and 'technical' in gradient.geoid_b:
            return 'technical_divergence'
        elif gradient.tension_score > 0.7:
            return 'high_tension_reversal'
        else:
            return 'standard_contradiction'
    
    def _calculate_semantic_metrics(self, 
                                  geoids: List[GeoidState], 
                                  contradictions: List[MarketContradiction]) -> Dict[str, float]:
        """
        Calculate semantic metrics from geoids and contradictions
        """
        metrics = {
            'total_entropy': 0.0,
            'average_tension': 0.0,
            'semantic_coherence': 1.0,
            'contradiction_intensity': 0.0,
            'thermodynamic_pressure': 0.0
        }
        
        # Calculate total entropy
        if geoids:
            entropies = [g.calculate_entropy() for g in geoids]
            metrics['total_entropy'] = sum(entropies)
            
            # Thermodynamic pressure (simplified)
            metrics['thermodynamic_pressure'] = np.std(entropies) if len(entropies) > 1 else 0.0
        
        # Calculate average tension and contradiction intensity
        if contradictions:
            tensions = [c.tension_score for c in contradictions]
            metrics['average_tension'] = np.mean(tensions)
            metrics['contradiction_intensity'] = max(tensions)
            
            # Semantic coherence decreases with contradictions
            metrics['semantic_coherence'] = 1.0 - (metrics['average_tension'] * 0.5)
        
        return metrics
    
    def _determine_trading_action(self, 
                                contradictions: List[MarketContradiction], 
                                semantic_analysis: Dict[str, float]) -> Tuple[str, float]:
        """
        Determine trading action based on contradictions and semantic analysis
        """
        if not contradictions:
            return 'hold', 0.5
        
        # Find the strongest contradiction
        strongest = max(contradictions, key=lambda c: c.tension_score)
        
        # Base confidence on contradiction strength and semantic metrics
        confidence = strongest.tension_score * semantic_analysis.get('thermodynamic_pressure', 0.5)
        confidence = min(max(confidence, 0.0), 1.0)
        
        # Determine action based on opportunity type and metrics
        if strongest.opportunity_type == 'sentiment_arbitrage':
            if semantic_analysis['average_tension'] > 0.6:
                return 'buy', confidence
            else:
                return 'sell', confidence
        elif strongest.opportunity_type == 'high_tension_reversal':
            return 'short', confidence
        elif semantic_analysis['contradiction_intensity'] > 0.8:
            return 'hedge', confidence
        else:
            return 'monitor', confidence * 0.5
    
    async def _execute_trade(self, request: TradingRequest) -> TradingResult:
        """
        Execute a trade based on the analysis
        """
        # First analyze the market
        analysis_result = await self._analyze_market(request)
        
        if analysis_result.confidence < 0.6:
            return TradingResult(
                action_taken='no_trade',
                position=None,
                semantic_analysis=analysis_result.semantic_analysis,
                contradiction_map=analysis_result.contradiction_map,
                execution_time=0.0,
                confidence=analysis_result.confidence,
                metadata={'reason': 'insufficient_confidence'}
            )
        
        # Calculate position size based on risk parameters
        position_size = self._calculate_position_size(
            analysis_result.confidence,
            request.risk_parameters
        )
        
        # Create position
        position = {
            'symbol': request.market_data.get('symbol', 'UNKNOWN'),
            'side': analysis_result.action_taken,
            'size': position_size,
            'entry_price': request.market_data.get('price', 0),
            'entry_time': datetime.now(),
            'semantic_score': analysis_result.confidence,
            'contradiction_ids': [c['contradiction_id'] for c in analysis_result.contradiction_map]
        }
        
        # Store position
        position_id = str(uuid.uuid4())
        self.active_positions[position_id] = position
        
        # Update metrics
        self.performance_metrics['total_trades'] += 1
        
        # Store in database if available
        if self.db_manager:
            self.db_manager.write_market_data(
                symbol=position['symbol'],
                price=position['entry_price'],
                volume=position['size'],
                timestamp=position['entry_time']
            )
        
        return TradingResult(
            action_taken='trade_executed',
            position=position,
            semantic_analysis=analysis_result.semantic_analysis,
            contradiction_map=analysis_result.contradiction_map,
            execution_time=0.0,
            confidence=analysis_result.confidence,
            metadata={'position_id': position_id}
        )
    
    def _calculate_position_size(self, confidence: float, risk_params: Dict[str, float]) -> float:
        """
        Calculate position size based on confidence and risk parameters
        """
        max_position = risk_params.get('max_position_size', 1000.0)
        risk_per_trade = risk_params.get('risk_per_trade', 0.02)
        
        # Kelly criterion inspired sizing
        position_size = max_position * confidence * risk_per_trade
        
        # Apply limits
        min_size = risk_params.get('min_position_size', 10.0)
        position_size = max(min(position_size, max_position), min_size)
        
        return position_size
    
    async def _monitor_positions(self, request: TradingRequest) -> TradingResult:
        """
        Monitor existing positions and update based on new data
        """
        monitoring_results = []
        
        for position_id, position in self.active_positions.items():
            # Check if position should be closed
            current_price = request.market_data.get('price', position['entry_price'])
            pnl = (current_price - position['entry_price']) * position['size']
            
            if position['side'] == 'sell' or position['side'] == 'short':
                pnl = -pnl
            
            # Simple exit logic - in production this would be much more sophisticated
            should_close = False
            if pnl > position['entry_price'] * 0.02:  # 2% profit
                should_close = True
                reason = 'take_profit'
            elif pnl < -position['entry_price'] * 0.01:  # 1% loss
                should_close = True
                reason = 'stop_loss'
            
            monitoring_results.append({
                'position_id': position_id,
                'pnl': pnl,
                'should_close': should_close,
                'reason': reason if should_close else 'monitoring'
            })
        
        return TradingResult(
            action_taken='positions_monitored',
            position=None,
            semantic_analysis={},
            contradiction_map=[],
            execution_time=0.0,
            confidence=1.0,
            metadata={'monitoring_results': monitoring_results}
        )
    
    def _contradiction_to_dict(self, contradiction: MarketContradiction) -> Dict[str, Any]:
        """
        Convert a MarketContradiction to a dictionary
        """
        return {
            'contradiction_id': contradiction.contradiction_id,
            'source_a': contradiction.source_a,
            'source_b': contradiction.source_b,
            'tension_score': contradiction.tension_score,
            'semantic_distance': contradiction.semantic_distance,
            'opportunity_type': contradiction.opportunity_type,
            'timestamp': contradiction.timestamp.isoformat()
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get a summary of trading performance
        """
        return {
            'performance_metrics': self.performance_metrics,
            'active_positions': len(self.active_positions),
            'total_contradictions_detected': len(self.contradiction_history),
            'recent_contradictions': [
                self._contradiction_to_dict(c) 
                for c in self.contradiction_history[-10:]
            ]
        }


def create_semantic_trading_reactor(config: Optional[Dict[str, Any]] = None) -> Optional[SemanticTradingReactor]:
    """
    Factory function to create a Semantic Trading Reactor
    
    Args:
        config: Configuration dictionary
        
    Returns:
        SemanticTradingReactor instance or None if creation fails
    """
    if config is None:
        config = {
            'tension_threshold': 0.4,
            'questdb_host': 'localhost',
            'questdb_port': 9009,
            'kafka_servers': 'localhost:9092'
        }
    
    try:
        return SemanticTradingReactor(config)
    except Exception as e:
        logger.error(f"Failed to create Semantic Trading Reactor: {e}", exc_info=True)
        return None 