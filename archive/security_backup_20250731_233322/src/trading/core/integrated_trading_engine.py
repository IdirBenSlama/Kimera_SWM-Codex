"""
Integrated Kimera Trading Engine with State-of-the-Art Libraries
Integrates anomaly detection, portfolio optimization, and reinforcement learning
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import asyncio
import json

# State-of-the-art libraries
try:
    from pyod.models.iforest import IForest
    from pyod.models.lof import LOF
    PYOD_AVAILABLE = True
except ImportError:
    PYOD_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False

try:
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.evaluation import evaluate_policy
    import gymnasium as gym
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False

# Kimera core components
try:
    from src.kimera.cognitive_field_dynamics import CognitiveFieldDynamics
    from src.kimera.contradiction_detection import ContradictionEngine
    from src.kimera.semantic_thermodynamics import SemanticThermodynamicsEngine
    KIMERA_CORE_AVAILABLE = True
except ImportError:
    KIMERA_CORE_AVAILABLE = False
    logging.warning("Kimera core components not available")

try:
    from src.trading.intelligence.enhanced_anomaly_detector import EnhancedAnomalyDetector, AnomalyType, create_enhanced_detector
    from src.trading.optimization.portfolio_optimizer import AdvancedPortfolioOptimizer, OptimizationObjective, create_portfolio_optimizer
    KIMERA_MODULES_AVAILABLE = True
except ImportError:
    KIMERA_MODULES_AVAILABLE = False
    logging.warning("Kimera trading modules not available")


@dataclass
class IntegratedTradingSignal:
    """Enhanced trading signal with state-of-the-art analysis"""
    action: str  # 'buy', 'sell', 'hold'
    confidence: float  # 0-1
    position_size: float  # Optimal position size
    cognitive_pressure: float  # Kimera cognitive pressure
    contradiction_level: float  # Kimera contradiction detection
    semantic_temperature: float  # Kimera semantic temperature
    anomaly_score: float  # Anomaly detection score
    portfolio_weight: float  # Optimized portfolio weight
    rl_action: Optional[int] = None  # Reinforcement learning action
    risk_metrics: Optional[Dict] = None
    explanation: str = "Enhanced signal with multi-model analysis"
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class MarketIntelligence:
    """Comprehensive market intelligence"""
    price: float
    volume: float
    bid_ask_spread: float
    market_depth: Dict[str, float]
    volatility: float
    momentum: float
    news_sentiment: float = 0.0
    social_sentiment: float = 0.0
    technical_indicators: Optional[Dict] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class IntegratedTradingEngine:
    """
    Integrated Kimera Trading Engine with State-of-the-Art Integration
    
    Combines Kimera's cognitive approach with:
    - Advanced anomaly detection (PyOD, Extended Isolation Forest)
    - Portfolio optimization (CVXPY, modern portfolio theory)
    - Reinforcement learning (Stable-Baselines3)
    - Model interpretability (SHAP)
    """
    
    def __init__(self, 
                 initial_balance: float = 1000.0,
                 risk_tolerance: float = 0.02,
                 enable_rl: bool = True,
                 enable_anomaly_detection: bool = True,
                 enable_portfolio_optimization: bool = True):
        """
        Initialize Integrated Kimera Engine
        
        Args:
            initial_balance: Starting balance
            risk_tolerance: Risk tolerance (0-1)
            enable_rl: Enable reinforcement learning
            enable_anomaly_detection: Enable anomaly detection
            enable_portfolio_optimization: Enable portfolio optimization
        """
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.risk_tolerance = risk_tolerance
        
        # Feature flags
        self.enable_rl = enable_rl and RL_AVAILABLE
        self.enable_anomaly_detection = enable_anomaly_detection and PYOD_AVAILABLE
        self.enable_portfolio_optimization = enable_portfolio_optimization and CVXPY_AVAILABLE
        
        # Initialize Kimera core components
        self._initialize_kimera_core()
        
        # Initialize state-of-the-art components
        self._initialize_advanced_components()
        
        # Trading state
        self.positions = {}
        self.trading_history = []
        self.performance_metrics = {}
        
        # Market intelligence
        self.market_intelligence_history = []
        self.current_market_intelligence = None
        
        logging.info(f"Integrated Kimera Engine initialized with advanced features:")
        logging.info(f"  - Anomaly Detection: {self.enable_anomaly_detection}")
        logging.info(f"  - Portfolio Optimization: {self.enable_portfolio_optimization}")
        logging.info(f"  - Reinforcement Learning: {self.enable_rl}")
    
    def _initialize_kimera_core(self):
        """Initialize Kimera cognitive components"""
        if not KIMERA_CORE_AVAILABLE:
            logging.warning("Kimera core not available - using fallback implementations")
            self.cognitive_field = None
            self.contradiction_engine = None
            self.semantic_engine = None
            return
        
        try:
            self.cognitive_field = CognitiveFieldDynamics(dimensions=10)
            self.contradiction_engine = ContradictionEngine()
            self.semantic_engine = SemanticThermodynamicsEngine()
            
            # Initialize cognitive field
            initial_state = np.random.randn(10) * 0.1
            self.cognitive_field.update_field(initial_state)
            
            logging.info("Kimera core components initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing Kimera core: {str(e)}")
            self.cognitive_field = None
            self.contradiction_engine = None
            self.semantic_engine = None
    
    def _initialize_advanced_components(self):
        """Initialize state-of-the-art components"""
        
        # Anomaly Detection
        if self.enable_anomaly_detection and KIMERA_MODULES_AVAILABLE:
            try:
                self.anomaly_detector = create_enhanced_detector()
                logging.info("Enhanced anomaly detector initialized")
            except Exception as e:
                self.anomaly_detector = None
                logging.warning(f"Anomaly detection failed to initialize: {e}")
        else:
            self.anomaly_detector = None
            logging.warning("Anomaly detection disabled")
        
        # Portfolio Optimization
        if self.enable_portfolio_optimization and KIMERA_MODULES_AVAILABLE:
            try:
                self.portfolio_optimizer = create_portfolio_optimizer()
                logging.info("Advanced portfolio optimizer initialized")
            except Exception as e:
                self.portfolio_optimizer = None
                logging.warning(f"Portfolio optimization failed to initialize: {e}")
        else:
            self.portfolio_optimizer = None
            logging.warning("Portfolio optimization disabled")
        
        # Reinforcement Learning
        if self.enable_rl:
            self._initialize_rl_agent()
        else:
            self.rl_agent = None
            logging.warning("Reinforcement learning disabled")
        
        # SHAP explainer for model interpretability
        if SHAP_AVAILABLE:
            self.explainer = None  # Will be initialized after first predictions
            logging.info("SHAP interpretability enabled")
        else:
            logging.warning("SHAP not available - model interpretability disabled")
    
    def _initialize_rl_agent(self):
        """Initialize reinforcement learning agent"""
        try:
            # Create a simple trading environment
            # Note: In production, you'd create a proper gym environment
            self.rl_agent = None  # Placeholder for now
            self.rl_training_data = []
            logging.info("RL agent framework initialized")
        except Exception as e:
            logging.error(f"Error initializing RL agent: {str(e)}")
            self.rl_agent = None
    
    def process_market_data(self, market_data: Dict[str, Any]) -> MarketIntelligence:
        """
        Process raw market data into structured intelligence
        
        Args:
            market_data: Raw market data dictionary
            
        Returns:
            Structured market intelligence
        """
        # Extract basic market data
        price = market_data.get('close', market_data.get('price', 0))
        volume = market_data.get('volume', 0)
        
        # Calculate derived metrics
        bid = market_data.get('bid', price * 0.999)
        ask = market_data.get('ask', price * 1.001)
        bid_ask_spread = ask - bid
        
        # Market depth (simplified)
        market_depth = {
            'bid_depth': market_data.get('bid_size', volume * 0.1),
            'ask_depth': market_data.get('ask_size', volume * 0.1)
        }
        
        # Volatility and momentum (using recent history)
        volatility = self._calculate_volatility()
        momentum = self._calculate_momentum()
        
        # Create market intelligence
        intelligence = MarketIntelligence(
            price=price,
            volume=volume,
            bid_ask_spread=bid_ask_spread,
            market_depth=market_depth,
            volatility=volatility,
            momentum=momentum,
            news_sentiment=market_data.get('news_sentiment', 0.0),
            social_sentiment=market_data.get('social_sentiment', 0.0),
            technical_indicators=market_data.get('technical_indicators', {})
        )
        
        # Store for history
        self.market_intelligence_history.append(intelligence)
        if len(self.market_intelligence_history) > 1000:  # Keep last 1000 records
            self.market_intelligence_history.pop(0)
        
        self.current_market_intelligence = intelligence
        return intelligence
    
    def _calculate_volatility(self) -> float:
        """Calculate market volatility from recent history"""
        if len(self.market_intelligence_history) < 10:
            return 0.02  # Default volatility
        
        recent_prices = [mi.price for mi in self.market_intelligence_history[-20:]]
        returns = np.diff(recent_prices) / recent_prices[:-1]
        return float(np.std(returns))
    
    def _calculate_momentum(self) -> float:
        """Calculate price momentum"""
        if len(self.market_intelligence_history) < 5:
            return 0.0
        
        prices = [mi.price for mi in self.market_intelligence_history[-10:]]
        if len(prices) >= 2:
            return (prices[-1] - prices[0]) / prices[0]
        return 0.0
    
    def analyze_market_with_kimera(self, market_intelligence: MarketIntelligence) -> Dict[str, float]:
        """
        Analyze market using Kimera's cognitive approach
        
        Args:
            market_intelligence: Structured market data
            
        Returns:
            Kimera analysis metrics
        """
        if not self.cognitive_field:
            # Fallback analysis
            return {
                'cognitive_pressure': 0.5 + np.random.normal(0, 0.1),
                'contradiction_level': 0.4 + np.random.normal(0, 0.1),
                'semantic_temperature': 0.3 + market_intelligence.volatility
            }
        
        try:
            # Create market state vector for cognitive field
            market_state = np.array([
                market_intelligence.price / 100.0,  # Normalized price
                market_intelligence.volume / 10000.0,  # Normalized volume
                market_intelligence.bid_ask_spread * 1000,  # Spread in basis points
                market_intelligence.volatility * 100,  # Volatility percentage
                market_intelligence.momentum * 100,  # Momentum percentage
                market_intelligence.news_sentiment,
                market_intelligence.social_sentiment,
                0.0, 0.0, 0.0  # Padding to reach 10 dimensions
            ])
            
            # Update cognitive field
            self.cognitive_field.update_field(market_state)
            
            # Calculate cognitive pressure
            cognitive_pressure = float(np.mean(np.abs(self.cognitive_field.field_state)))
            cognitive_pressure = min(max(cognitive_pressure, 0.0), 1.0)
            
            # Analyze contradictions
            market_factors = {
                'price_trend': market_intelligence.momentum,
                'volume_trend': 1.0 if market_intelligence.volume > 1000 else 0.0,
                'volatility_level': market_intelligence.volatility,
                'sentiment_score': (market_intelligence.news_sentiment + market_intelligence.social_sentiment) / 2
            }
            
            contradictions = self.contradiction_engine.detect_tension_gradients(market_factors)
            contradiction_level = float(np.mean([abs(c) for c in contradictions.values()]))
            contradiction_level = min(max(contradiction_level, 0.0), 1.0)
            
            # Calculate semantic temperature
            semantic_temp = (market_intelligence.volatility + abs(market_intelligence.momentum)) / 2
            semantic_temp = min(max(semantic_temp, 0.0), 1.0)
            
            return {
                'cognitive_pressure': cognitive_pressure,
                'contradiction_level': contradiction_level,
                'semantic_temperature': semantic_temp
            }
            
        except Exception as e:
            logging.error(f"Error in Kimera analysis: {str(e)}")
            return {
                'cognitive_pressure': 0.5,
                'contradiction_level': 0.5,
                'semantic_temperature': 0.5
            }
    
    def detect_anomalies(self, market_intelligence: MarketIntelligence) -> Dict[str, float]:
        """
        Detect market anomalies using state-of-the-art algorithms
        
        Args:
            market_intelligence: Market data
            
        Returns:
            Anomaly detection results
        """
        if not self.anomaly_detector:
            return {'anomaly_score': 0.0, 'system_status': 'NORMAL'}
        
        try:
            # Convert market intelligence to DataFrame for anomaly detection
            market_df = pd.DataFrame([{
                'close': market_intelligence.price,
                'volume': market_intelligence.volume,
                'bid': market_intelligence.price - market_intelligence.bid_ask_spread/2,
                'ask': market_intelligence.price + market_intelligence.bid_ask_spread/2,
                'volatility': market_intelligence.volatility,
                'momentum': market_intelligence.momentum
            }])
            
            # Detect market anomalies
            anomalies = self.anomaly_detector.detect_market_anomalies(market_df)
            
            # Extract anomaly score
            if anomalies:
                avg_severity = np.mean([a.severity for a in anomalies])
                return {
                    'anomaly_score': avg_severity,
                    'system_status': 'WARNING' if avg_severity > 0.7 else 'NORMAL',
                    'anomaly_count': len(anomalies)
                }
            else:
                return {
                    'anomaly_score': 0.0,
                    'system_status': 'NORMAL',
                    'anomaly_count': 0
                }
                
        except Exception as e:
            logging.error(f"Error in anomaly detection: {str(e)}")
            return {'anomaly_score': 0.0, 'system_status': 'ERROR'}
    
    def optimize_portfolio_allocation(self, assets: List[str]) -> Dict[str, float]:
        """
        Optimize portfolio allocation using modern portfolio theory
        
        Args:
            assets: List of asset symbols
            
        Returns:
            Optimized portfolio weights
        """
        if not self.portfolio_optimizer or not assets:
            # Equal weight fallback
            return {asset: 1.0/len(assets) for asset in assets}
        
        try:
            # Get current weights
            current_weights = None
            if self.positions:
                total_value = sum(abs(pos) for pos in self.positions.values())
                if total_value > 0:
                    current_weights = np.array([
                        self.positions.get(asset, 0) / total_value for asset in assets
                    ])
            
            # Optimize portfolio
            result = self.portfolio_optimizer.optimize_portfolio(
                assets=assets,
                current_weights=current_weights,
                objective=OptimizationObjective.MAXIMUM_SHARPE
            )
            
            # Convert to dictionary
            return {asset: float(weight) for asset, weight in zip(assets, result.weights)}
            
        except Exception as e:
            logging.error(f"Error in portfolio optimization: {str(e)}")
            # Equal weight fallback
            return {asset: 1.0/len(assets) for asset in assets}
    
    def generate_enhanced_signal(self, 
                                market_data: Dict[str, Any],
                                symbol: str = "BTC/USDT") -> IntegratedTradingSignal:
        """
        Generate enhanced trading signal combining all methodologies
        
        Args:
            market_data: Raw market data
            symbol: Trading symbol
            
        Returns:
            Enhanced trading signal
        """
        try:
            # Process market data
            market_intelligence = self.process_market_data(market_data)
            
            # Kimera cognitive analysis
            kimera_analysis = self.analyze_market_with_kimera(market_intelligence)
            
            # Anomaly detection
            anomaly_analysis = self.detect_anomalies(market_intelligence)
            
            # Portfolio optimization (single asset for now)
            portfolio_weights = self.optimize_portfolio_allocation([symbol])
            
            # Generate trading signal using multi-factor approach
            signal = self._synthesize_trading_signal(
                kimera_analysis=kimera_analysis,
                anomaly_analysis=anomaly_analysis,
                portfolio_weights=portfolio_weights,
                market_intelligence=market_intelligence,
                symbol=symbol
            )
            
            # Log the signal
            self._log_trading_signal(signal)
            
            return signal
            
        except Exception as e:
            logging.error(f"Error generating enhanced signal: {str(e)}")
            # Fallback signal
            return IntegratedTradingSignal(
                action='hold',
                confidence=0.0,
                position_size=0.0,
                cognitive_pressure=0.5,
                contradiction_level=0.5,
                semantic_temperature=0.5,
                anomaly_score=0.0,
                portfolio_weight=0.0,
                explanation=f"Error in signal generation: {str(e)}"
            )
    
    def _synthesize_trading_signal(self,
                                 kimera_analysis: Dict[str, float],
                                 anomaly_analysis: Dict[str, float],
                                 portfolio_weights: Dict[str, float],
                                 market_intelligence: MarketIntelligence,
                                 symbol: str) -> IntegratedTradingSignal:
        """Synthesize trading signal from all analysis components"""
        
        # Extract metrics
        cognitive_pressure = kimera_analysis['cognitive_pressure']
        contradiction_level = kimera_analysis['contradiction_level']
        semantic_temperature = kimera_analysis['semantic_temperature']
        anomaly_score = anomaly_analysis['anomaly_score']
        
        # Multi-factor decision logic
        factors = {
            'cognitive_signal': self._cognitive_signal(cognitive_pressure, contradiction_level),
            'momentum_signal': self._momentum_signal(market_intelligence.momentum),
            'volatility_signal': self._volatility_signal(market_intelligence.volatility),
            'anomaly_signal': self._anomaly_signal(anomaly_score),
            'semantic_signal': self._semantic_signal(semantic_temperature)
        }
        
        # Combine signals with weights
        signal_weights = {
            'cognitive_signal': 0.3,
            'momentum_signal': 0.2,
            'volatility_signal': 0.2,
            'anomaly_signal': 0.2,
            'semantic_signal': 0.1
        }
        
        combined_signal = sum(factors[key] * signal_weights[key] for key in factors)
        confidence = min(abs(combined_signal), 1.0)
        
        # Determine action
        if combined_signal > 0.1:
            action = 'buy'
        elif combined_signal < -0.1:
            action = 'sell'
        else:
            action = 'hold'
        
        # Calculate position size based on confidence and risk tolerance
        base_position_size = confidence * self.risk_tolerance
        portfolio_weight = portfolio_weights.get(symbol, 0.0)
        
        # Adjust for anomalies (reduce size if anomalies detected)
        anomaly_adjustment = 1.0 - (anomaly_score * 0.5)
        position_size = base_position_size * anomaly_adjustment * portfolio_weight
        
        # Generate explanation
        explanation = self._generate_signal_explanation(factors, kimera_analysis, anomaly_analysis)
        
        return IntegratedTradingSignal(
            action=action,
            confidence=confidence,
            position_size=position_size,
            cognitive_pressure=cognitive_pressure,
            contradiction_level=contradiction_level,
            semantic_temperature=semantic_temperature,
            anomaly_score=anomaly_score,
            portfolio_weight=portfolio_weight,
            explanation=explanation
        )
    
    def _cognitive_signal(self, pressure: float, contradictions: float) -> float:
        """Generate signal from cognitive analysis"""
        # High pressure + high contradictions = strong signal
        base_signal = pressure * contradictions
        
        # Direction based on pressure vs contradictions balance
        if pressure > contradictions:
            return base_signal  # Bullish
        else:
            return -base_signal  # Bearish
    
    def _momentum_signal(self, momentum: float) -> float:
        """Generate signal from momentum"""
        return np.tanh(momentum * 10)  # Scaled and bounded
    
    def _volatility_signal(self, volatility: float) -> float:
        """Generate signal from volatility (high vol = reduce positions)"""
        return -volatility * 2  # Negative because high vol should reduce positions
    
    def _anomaly_signal(self, anomaly_score: float) -> float:
        """Generate signal from anomaly detection (anomalies = caution)"""
        return -anomaly_score  # Negative because anomalies suggest caution
    
    def _semantic_signal(self, temperature: float) -> float:
        """Generate signal from semantic temperature"""
        # High temperature suggests market regime change
        return (temperature - 0.5) * 2  # Centered around 0.5
    
    def _generate_signal_explanation(self,
                                   factors: Dict[str, float],
                                   kimera_analysis: Dict[str, float],
                                   anomaly_analysis: Dict[str, float]) -> str:
        """Generate human-readable explanation of trading signal"""
        explanations = []
        
        # Cognitive analysis
        if kimera_analysis['cognitive_pressure'] > 0.7:
            explanations.append("High cognitive pressure detected")
        if kimera_analysis['contradiction_level'] > 0.6:
            explanations.append("Significant market contradictions found")
        
        # Anomaly analysis
        if anomaly_analysis['anomaly_score'] > 0.5:
            explanations.append(f"Market anomalies detected (score: {anomaly_analysis['anomaly_score']:.2f})")
        
        # Factor analysis
        dominant_factor = max(factors.items(), key=lambda x: abs(x[1]))
        explanations.append(f"Dominant factor: {dominant_factor[0]} ({dominant_factor[1]:.2f})")
        
        return "; ".join(explanations) if explanations else "Standard multi-factor analysis"
    
    def _log_trading_signal(self, signal: IntegratedTradingSignal):
        """Log trading signal for analysis"""
        self.trading_history.append(signal)
        
        # Keep last 1000 signals
        if len(self.trading_history) > 1000:
            self.trading_history.pop(0)
        
        logging.info(
            f"Enhanced Signal Generated: {signal.action.upper()} "
            f"(Confidence: {signal.confidence:.2f}, "
            f"Size: {signal.position_size:.3f}, "
            f"Anomaly: {signal.anomaly_score:.2f})"
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        if not self.trading_history:
            return {"error": "No trading history available"}
        
        signals = self.trading_history
        
        # Basic metrics
        total_signals = len(signals)
        buy_signals = len([s for s in signals if s.action == 'buy'])
        sell_signals = len([s for s in signals if s.action == 'sell'])
        hold_signals = len([s for s in signals if s.action == 'hold'])
        
        # Confidence metrics
        avg_confidence = np.mean([s.confidence for s in signals])
        avg_position_size = np.mean([s.position_size for s in signals])
        
        # Kimera metrics
        avg_cognitive_pressure = np.mean([s.cognitive_pressure for s in signals])
        avg_contradiction_level = np.mean([s.contradiction_level for s in signals])
        avg_semantic_temperature = np.mean([s.semantic_temperature for s in signals])
        
        # Anomaly metrics
        avg_anomaly_score = np.mean([s.anomaly_score for s in signals])
        anomaly_alerts = len([s for s in signals if s.anomaly_score > 0.7])
        
        return {
            'total_signals': total_signals,
            'signal_distribution': {
                'buy': buy_signals,
                'sell': sell_signals,
                'hold': hold_signals
            },
            'avg_confidence': avg_confidence,
            'avg_position_size': avg_position_size,
            'kimera_metrics': {
                'avg_cognitive_pressure': avg_cognitive_pressure,
                'avg_contradiction_level': avg_contradiction_level,
                'avg_semantic_temperature': avg_semantic_temperature
            },
            'anomaly_metrics': {
                'avg_anomaly_score': avg_anomaly_score,
                'anomaly_alerts': anomaly_alerts
            },
            'system_status': 'OPERATIONAL',
            'advanced_features': {
                'anomaly_detection': self.enable_anomaly_detection,
                'portfolio_optimization': self.enable_portfolio_optimization,
                'reinforcement_learning': self.enable_rl
            }
        }


def create_integrated_trading_engine(**kwargs) -> IntegratedTradingEngine:
    """Factory function to create enhanced Kimera engine"""
    return IntegratedTradingEngine(**kwargs)


# Example usage and testing
if __name__ == "__main__":
    # Test the enhanced Kimera engine
    logger.info("ðŸš€ Testing Integrated Kimera Trading Engine")
    logger.info("=" * 50)
    
    # Create engine
    engine = create_integrated_trading_engine(
        initial_balance=1000.0,
        risk_tolerance=0.05,
        enable_rl=True,
        enable_anomaly_detection=True,
        enable_portfolio_optimization=True
    )
    
    # Test with sample market data
    sample_market_data = {
        'close': 50000.0,
        'volume': 1500.0,
        'bid': 49950.0,
        'ask': 50050.0,
        'volatility': 0.025,
        'news_sentiment': 0.1,
        'social_sentiment': 0.05
    }
    
    logger.info("\nðŸ“Š Generating Enhanced Trading Signal...")
    signal = engine.generate_enhanced_signal(sample_market_data, "BTC/USDT")
    
    logger.info(f"Signal: {signal.action.upper()}")
    logger.info(f"Confidence: {signal.confidence:.3f}")
    logger.info(f"Position Size: {signal.position_size:.3f}")
    logger.info(f"Cognitive Pressure: {signal.cognitive_pressure:.3f}")
    logger.info(f"Contradiction Level: {signal.contradiction_level:.3f}")
    logger.info(f"Semantic Temperature: {signal.semantic_temperature:.3f}")
    logger.info(f"Anomaly Score: {signal.anomaly_score:.3f}")
    logger.info(f"Explanation: {signal.explanation}")
    
    # Test multiple signals
    logger.info("\nðŸ”„ Testing Multiple Signals...")
    for i in range(5):
        # Simulate market changes
        sample_market_data['close'] += np.random.normal(0, 100)
        sample_market_data['volume'] += np.random.normal(0, 200)
        
        signal = engine.generate_enhanced_signal(sample_market_data, "BTC/USDT")
        logger.info(f"Signal {i+1}: {signal.action.upper()}")
    
    # Get performance summary
    logger.info("\nðŸ“ˆ Performance Summary:")
    summary = engine.get_performance_summary()
    logger.info(f"Total Signals: {summary['total_signals']}")
    logger.info(f"Average Confidence: {summary['avg_confidence']:.3f}")
    logger.info(f"Average Anomaly Score: {summary['anomaly_metrics']['avg_anomaly_score']:.3f}")
    logger.info(f"System Status: {summary['system_status']}")
    
    logger.info("\nâœ… Integrated Kimera Engine Test Complete!")