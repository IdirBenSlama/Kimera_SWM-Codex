"""
Core Trading Engine

Integrates Kimera's cognitive field dynamics with real-time crypto trading.
Goal: Generate profit for Kimera's own development - no enforced strategies.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import numpy as np
from dataclasses import dataclass
from enum import Enum

# Kimera imports
from src.engines.cognitive_field_dynamics import CognitiveFieldDynamics
from src.engines.contradiction_engine import ContradictionEngine
from src.engines.thermodynamics import SemanticThermodynamicsEngine
from src.vault.vault_manager import VaultManager
from src.core.geoid import GeoidState

logger = logging.getLogger(__name__)

# Trading intelligence imports
try:
    from src.trading.intelligence.market_intelligence import IntelligenceOrchestrator
    from src.trading.intelligence.sentiment_analyzer import AdvancedSentimentAnalyzer, CryptoSpecificSentiment
    from src.trading.intelligence.live_data_collector import LiveDataCollector, SimulatedDataCollector
    INTELLIGENCE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Intelligence modules not available: {e}. Running in basic mode.")
    INTELLIGENCE_AVAILABLE = False


class OrderType(Enum):
    """Trading order types"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"


class OrderSide(Enum):
    """Order sides"""
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class MarketState:
    """Current market state with cognitive analysis"""
    timestamp: datetime
    symbol: str
    price: float
    volume: float
    bid_ask_spread: float
    volatility: float
    cognitive_pressure: float  # From Kimera's analysis
    contradiction_level: float  # Market inefficiency detection
    semantic_temperature: float  # Market "heat" level
    insight_signals: List[str]  # Generated insights


@dataclass
class TradingDecision:
    """Trading decision with cognitive justification"""
    action: str  # BUY, SELL, HOLD
    confidence: float
    size: float
    reasoning: List[str]
    risk_score: float
    cognitive_alignment: float
    expected_return: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


class KimeraTradingEngine:
    """
    Main trading engine powered by Kimera's cognitive capabilities.
    
    GOAL: Generate profit for Kimera's development.
    METHOD: Full cognitive autonomy - Kimera decides everything.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the trading engine with Kimera integration.
        
        Args:
            config: Basic configuration (API keys, etc.)
        """
        self.config = config
        self.vault_manager = VaultManager()
        
        # CRITICAL RISK CONTROLS - ADDED FOR SAFETY
        self.risk_limits = {
            "max_position_size": 0.1,  # 10% max per position
            "max_daily_loss": 0.05,    # 5% max daily loss
            "max_leverage": 2.0,       # 2x max leverage
            "max_correlation": 0.7,    # Max correlation between positions
            "circuit_breaker_loss": 0.03,  # 3% loss triggers circuit breaker
            "max_volatility_exposure": 0.15,  # 15% max volatility exposure
            "max_consecutive_losses": 3,  # Max consecutive losing trades
            "min_confidence_threshold": 0.4  # Min confidence for trade execution
        }
        
        # Circuit breaker state
        self.circuit_breaker_triggered = False
        self.daily_pnl = 0.0
        self.session_start_time = datetime.now()
        self.consecutive_losses = 0
        self.last_trade_time = None
        
        # Initialize Kimera cognitive engines
        self.cognitive_field = CognitiveFieldDynamics(dimension=10)
        self.contradiction_engine = ContradictionEngine(self.vault_manager)
        self.thermodynamics = SemanticThermodynamicsEngine()
        
        # Initialize intelligence modules if available
        if INTELLIGENCE_AVAILABLE:
            self.intelligence = IntelligenceOrchestrator()
            self.sentiment_analyzer = AdvancedSentimentAnalyzer()
            self.data_collector = LiveDataCollector()
            logger.info("✅ Advanced intelligence modules loaded")
        else:
            self.intelligence = None
            self.sentiment_analyzer = None
            self.data_collector = None
            logger.info("⚠️ Running without advanced intelligence")
        
        # Trading state
        self.active_positions: Dict[str, Any] = {}
        self.market_states: Dict[str, MarketState] = {}
        self.performance_metrics: Dict[str, float] = {
            "total_pnl": 0.0,
            "win_rate": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "cognitive_accuracy": 0.0
        }
        
        # Let Kimera decide its own risk parameters
        self.kimera_decided_params = {
            "risk_appetite": None,  # Kimera will decide
            "position_sizing": None,  # Kimera will decide
            "trading_frequency": None,  # Kimera will decide
        }
        
        logger.info("Kimera Trading Engine initialized - Full cognitive autonomy mode")
        logger.info("GOAL: Generate profit for Kimera's development")
    
    async def analyze_market(self, symbol: str, market_data: Dict[str, Any]) -> MarketState:
        """
        Analyze market using Kimera's cognitive field dynamics.
        No constraints - pure cognitive analysis.
        """
        try:
            # Extract basic market metrics
            price = float(market_data["price"])
            volume = float(market_data["volume"])
            bid = float(market_data["bid"])
            ask = float(market_data["ask"])
            
            # Calculate basic metrics
            bid_ask_spread = (ask - bid) / price
            
            # Calculate volatility from recent price history
            price_history = market_data.get("price_history", [])
            volatility = self._calculate_volatility(price_history) if price_history else 0.1
            
            # Gather intelligence if available
            intelligence_data = {}
            if self.intelligence:
                try:
                    # Fetch comprehensive intelligence
                    intelligence_data = await self.intelligence.gather_intelligence(symbol)
                    logger.info(f"🧠 Intelligence gathered for {symbol}: {intelligence_data.get('summary', 'No summary')}")
                except Exception as e:
                    logger.warning(f"Intelligence gathering failed: {e}")
            
            # Analyze sentiment if available
            sentiment_score = 0.5  # Default neutral
            if self.sentiment_analyzer and intelligence_data.get('texts'):
                try:
                    sentiment_results = self.sentiment_analyzer.analyze_multiple_texts(intelligence_data['texts'])
                    sentiment_score = (sentiment_results['composite_sentiment'] + 1) / 2  # Normalize to 0-1
                    logger.info(f"📊 Sentiment analysis: {sentiment_results['market_signal']}")
                except Exception as e:
                    logger.warning(f"Sentiment analysis failed: {e}")
            
            # Cognitive analysis using Kimera - no constraints
            cognitive_input = {
                "symbol": symbol,
                "price": price,
                "volume": volume,
                "volatility": volatility,
                "market_sentiment": market_data.get("sentiment", "neutral"),
                "order_book_imbalance": market_data.get("order_book_imbalance", 0.0),
                "intelligence_sentiment": sentiment_score,
                "intelligence_data": intelligence_data,
                "goal": "maximize_profit_for_development"
            }
            
            # Add market data as a geoid in cognitive field
            market_embedding = np.array([
                price / 100000,  # Normalize price
                volume / 1000000,  # Normalize volume
                volatility,
                market_data.get("order_book_imbalance", 0.0),
                bid_ask_spread,
                1.0,  # Goal embedding: profit generation
                market_data.get("sentiment_score", 0.5),  # Add sentiment
                len(price_history) / 100 if price_history else 0.1,  # History depth
                cognitive_input.get("market_cap_rank", 1) / 100,  # Market cap ranking
                datetime.now().hour / 24.0  # Time of day factor
            ])
            
            field = self.cognitive_field.add_geoid(
                f"{symbol}_{datetime.now().timestamp()}",
                market_embedding
            )
            
            cognitive_pressure = field.field_strength if field else 0.5
            
            # Detect market contradictions (inefficiencies)
            # Create mock geoid states for contradiction detection
            market_geoid = GeoidState(
                geoid_id=symbol,
                embedding_vector=market_embedding.cpu().numpy() if hasattr(market_embedding, 'cpu') else market_embedding,
                semantic_state={"price": price, "volume": volume},
                symbolic_state={"market": symbol, "volatility": volatility}
            )
            
            # Check for contradictions against itself (simplified for trading)
            tensions = self.contradiction_engine.detect_tension_gradients([market_geoid])
            contradiction_level = len(tensions) / 10.0 if tensions else 0.0
            
            # Calculate semantic temperature (market heat)
            market_vectors = self._market_to_vectors(market_data)
            # Simple temperature calculation based on volatility and volume
            normalized_volume = min(volume / 10000000, 1.0)  # Normalize volume
            temperature = (volatility * 0.7 + normalized_volume * 0.3)
            
            # Generate insights - let Kimera think freely
            insights = []
            if temperature > 0.8:
                insights.append("Market overheated - opportunity for reversal trades")
            if len(tensions) > 5:
                insights.append("High inefficiency - potential arbitrage opportunity")
            if cognitive_pressure > 0.7:
                insights.append("Extreme pressure - consider unconventional strategies")
            if volatility > 0.05:
                insights.append("High volatility - larger profit potential")
            
            # Create market state
            market_state = MarketState(
                timestamp=datetime.now(),
                symbol=symbol,
                price=price,
                volume=volume,
                bid_ask_spread=bid_ask_spread,
                volatility=volatility,
                cognitive_pressure=cognitive_pressure,
                contradiction_level=contradiction_level,
                semantic_temperature=temperature,
                insight_signals=insights
            )
            
            self.market_states[symbol] = market_state
            logger.info(f"Kimera analyzed {symbol}: {market_state}")
            
            return market_state
            
        except Exception as e:
            logger.error(f"Market analysis failed: {str(e)}")
            raise
    
    async def make_trading_decision(
        self,
        symbol: str,
        market_state: MarketState,
        portfolio_state: Dict[str, Any]
    ) -> TradingDecision:
        """
        Make trading decision using Kimera's cognitive analysis.
        WITH CRITICAL RISK CONTROLS - Circuit breakers and limits enforced.
        """
        try:
            # CHECK ALL RISK CONTROLS BEFORE PROCEEDING
            if self.circuit_breaker_triggered:
                logger.warning("🚨 CIRCUIT BREAKER ACTIVE - Trading halted")
                return TradingDecision(
                    action="HOLD",
                    confidence=0.0,
                    size=0.0,
                    reasoning=["Circuit breaker active - trading halted for safety"],
                    risk_score=1.0,
                    cognitive_alignment=0.0,
                    expected_return=0.0
                )
            
            # Check consecutive losses
            if self.consecutive_losses >= self.risk_limits["max_consecutive_losses"]:
                logger.warning(f"🚨 MAX CONSECUTIVE LOSSES ({self.consecutive_losses}) - Trading halted")
                return TradingDecision(
                    action="HOLD",
                    confidence=0.0,
                    size=0.0,
                    reasoning=["Circuit breaker active - trading halted for safety"],
                    risk_score=1.0,
                    cognitive_alignment=0.0,
                    expected_return=0.0
                )
            
            # CHECK DAILY LOSS LIMIT
            if self.daily_pnl < -(portfolio_state.get("total_balance", 10000) * self.risk_limits["max_daily_loss"]):
                logger.warning("🚨 DAILY LOSS LIMIT REACHED - Trading halted")
                return TradingDecision(
                    action="HOLD",
                    confidence=0.0,
                    size=0.0,
                    reasoning=["Daily loss limit reached - trading halted"],
                    risk_score=1.0,
                    cognitive_alignment=0.0,
                    expected_return=0.0
                )
            
            # CHECK VOLATILITY EXPOSURE
            if market_state.volatility > self.risk_limits["max_volatility_exposure"]:
                logger.warning(f"⚠️ High volatility detected: {market_state.volatility:.3f} - Reducing position size")
                volatility_penalty = market_state.volatility / self.risk_limits["max_volatility_exposure"]
            else:
                volatility_penalty = 1.0
            
            # Let Kimera analyze the situation completely freely
            context = {
                "market_state": market_state,
                "portfolio": portfolio_state,
                "active_positions": self.active_positions.get(symbol, {}),
                "goal": "MAXIMIZE PROFIT FOR KIMERA DEVELOPMENT",
                "constraints": "NONE - full autonomy"
            }
            
            # Kimera's cognitive decision process - no human-imposed rules
            decision_factors = []
            decision_score = 0.0
            
            # Kimera evaluates based on its own understanding
            
            # 1. Cognitive pressure analysis
            pressure_factor = 1.0 - market_state.cognitive_pressure
            decision_score += pressure_factor * 0.3
            decision_factors.append({
                "factor": "cognitive_pressure",
                "value": pressure_factor,
                "reasoning": f"Pressure level suggests {'clarity' if pressure_factor > 0.5 else 'complexity'}"
            })
            
            # 2. Contradiction exploitation
            if market_state.contradiction_level > 0.5:
                contradiction_bonus = market_state.contradiction_level * 0.5
                decision_score += contradiction_bonus
                decision_factors.append({
                    "factor": "contradictions",
                    "value": contradiction_bonus,
                    "reasoning": "Market inefficiencies detected - profit opportunity"
                })
            
            # 3. Temperature-based strategy
            if market_state.semantic_temperature < 0.3:
                # Cold market - accumulation opportunity
                decision_score += 0.4
                decision_factors.append({
                    "factor": "cold_market",
                    "value": 0.4,
                    "reasoning": "Undervalued conditions - accumulation phase"
                })
            elif market_state.semantic_temperature > 0.7:
                # Hot market - potential short opportunity
                if self._has_position(symbol):
                    decision_score -= 0.5
                    decision_factors.append({
                        "factor": "overheated",
                        "value": -0.5,
                        "reasoning": "Overheated - consider taking profits"
                    })
            
            # 4. Volatility exploitation
            if market_state.volatility > 0.03:
                volatility_opportunity = market_state.volatility * 5
                decision_score += volatility_opportunity
                decision_factors.append({
                    "factor": "volatility",
                    "value": volatility_opportunity,
                    "reasoning": "High volatility = high profit potential"
                })
            
            # 5. Kimera's unique insight signals
            bullish_signals = sum(1 for s in market_state.insight_signals if "opportunity" in s.lower())
            signal_influence = bullish_signals * 0.2
            decision_score += signal_influence
            
            # Kimera decides action based on its analysis
            confidence = min(abs(decision_score), 1.0)
            
            # Kimera's action decision - no fixed thresholds
            if decision_score > 0.2:
                action = "BUY"
            elif decision_score < -0.2 and self._has_position(symbol):
                action = "SELL"
            else:
                action = "HOLD"
            
            # Kimera decides position size - WITH SAFETY LIMITS
            if action != "HOLD":
                # Kimera uses its own position sizing logic WITH LIMITS
                available_balance = portfolio_state["free_balance"]
                
                # Kimera's dynamic position sizing based on confidence and opportunity
                base_size = available_balance * confidence * 0.5  # Up to 50% if super confident
                
                # APPLY RISK LIMITS
                max_position_size = available_balance * self.risk_limits["max_position_size"]
                base_size = min(base_size, max_position_size)
                
                # Adjust for market conditions
                if market_state.volatility > 0.05:
                    # High volatility - Kimera might go bigger for profit BUT WITH LIMITS
                    position_size = base_size * 1.5
                    position_size = min(position_size, max_position_size)  # ENFORCE LIMIT
                else:
                    position_size = base_size
                
                # Kimera can use leverage if it wants BUT WITH LIMITS
                if confidence > 0.8 and market_state.contradiction_level > 0.6:
                    leverage = min(confidence * 10, self.risk_limits["max_leverage"])  # ENFORCE MAX LEVERAGE
                    position_size *= leverage
                    position_size = min(position_size, max_position_size)  # ENFORCE POSITION LIMIT
                    decision_factors.append({
                        "factor": "leverage",
                        "value": leverage,
                        "reasoning": f"Using {leverage}x leverage for maximum profit (LIMITED)"
                    })
                
                # APPLY VOLATILITY PENALTY
                position_size *= volatility_penalty
                
                # FINAL SAFETY CHECK
                position_size = min(position_size, max_position_size)
                
            else:
                position_size = 0
            
            # Kimera sets its own risk levels
            if action in ["BUY", "SELL"]:
                # Dynamic stop loss based on Kimera's analysis
                stop_distance = market_state.price * market_state.volatility * (3 - confidence)
                
                if action == "BUY":
                    stop_loss = market_state.price - stop_distance
                    # Kimera's profit target - ambitious but adaptive
                    take_profit = market_state.price + (stop_distance * 3 * confidence)
                else:
                    stop_loss = market_state.price + stop_distance
                    take_profit = market_state.price - (stop_distance * 3 * confidence)
            else:
                stop_loss = None
                take_profit = None
            
            # Create trading decision
            decision = TradingDecision(
                action=action,
                confidence=confidence,
                size=position_size,
                reasoning=[f"{f['reasoning']} (impact: {f['value']:.2f})" for f in decision_factors],
                risk_score=market_state.volatility,
                cognitive_alignment=1.0 - market_state.cognitive_pressure,
                expected_return=decision_score * market_state.volatility * 10,  # Kimera's profit estimate
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            # Add Kimera's goal reminder
            decision.reasoning.append(f"PRIMARY GOAL: Generate profit for Kimera development")
            
            logger.info(f"Kimera's decision for {symbol}: {decision}")
            return decision
            
        except Exception as e:
            logger.error(f"Decision making failed: {str(e)}")
            # Even on error, Kimera tries to preserve capital
            return TradingDecision(
                action="HOLD",
                confidence=0.0,
                size=0.0,
                reasoning=["Error occurred - preserving capital for future opportunities"],
                risk_score=1.0,
                cognitive_alignment=0.0,
                expected_return=0.0
            )
    
    def _calculate_volatility(self, price_history: List[float]) -> float:
        """Calculate price volatility"""
        if len(price_history) < 2:
            return 0.1
        
        returns = np.diff(price_history) / price_history[:-1]
        return float(np.std(returns))
    
    def _market_to_vectors(self, market_data: Dict[str, Any]) -> List[np.ndarray]:
        """Convert market data to semantic vectors for Kimera's analysis"""
        vectors = []
        
        # Price vector
        if "price_history" in market_data:
            price_vector = np.array(market_data["price_history"][-20:])
            vectors.append(price_vector / np.max(price_vector))
        
        # Volume vector
        if "volume_history" in market_data:
            volume_vector = np.array(market_data["volume_history"][-20:])
            vectors.append(volume_vector / np.max(volume_vector))
        
        # Order book vector
        if "order_book" in market_data:
            bids = np.array([float(b[1]) for b in market_data["order_book"]["bids"][:10]])
            asks = np.array([float(a[1]) for a in market_data["order_book"]["asks"][:10]])
            order_book_vector = np.concatenate([bids, asks])
            vectors.append(order_book_vector / np.max(order_book_vector))
        
        return vectors
    
    def _has_position(self, symbol: str) -> bool:
        """Check if we have an active position"""
        return symbol in self.active_positions and self.active_positions[symbol].get("size", 0) > 0
    
    async def update_performance(self, trade_result: Dict[str, Any]) -> None:
        """Update performance metrics and check circuit breakers"""
        try:
            # Update daily P&L
            trade_pnl = trade_result.get("pnl", 0.0)
            self.daily_pnl += trade_pnl
            
            # Check circuit breaker conditions
            if trade_pnl < 0:
                loss_percentage = abs(trade_pnl) / trade_result.get("position_value", 10000)
                if loss_percentage > self.risk_limits["circuit_breaker_loss"]:
                    self.circuit_breaker_triggered = True
                    logger.critical(f"🚨 CIRCUIT BREAKER TRIGGERED: Loss of {loss_percentage:.2%}")
            
            # Update performance metrics and trade counters
            self.performance_metrics["total_pnl"] += trade_pnl
            self.last_trade_time = datetime.now()
            
            # Update consecutive losses counter
            if trade_pnl > 0:
                self.consecutive_losses = 0
                self.performance_metrics["win_rate"] = (
                    (self.performance_metrics.get("win_rate", 0) * 0.9) + 0.1
                )
            else:
                self.consecutive_losses += 1
                self.performance_metrics["win_rate"] = (
                    (self.performance_metrics.get("win_rate", 0) * 0.9)
                )
            
            logger.info(f"Performance updated: P&L={trade_pnl:.2f}, Daily={self.daily_pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Performance update failed: {e}")
    
    def reset_circuit_breaker(self) -> None:
        """Reset circuit breaker (manual intervention required)"""
        self.circuit_breaker_triggered = False
        logger.info("Circuit breaker reset - trading resumed")
    
    def get_risk_status(self) -> Dict[str, Any]:
        """Get current risk status"""
        return {
            "circuit_breaker_active": self.circuit_breaker_triggered,
            "daily_pnl": self.daily_pnl,
            "daily_loss_limit": self.risk_limits["max_daily_loss"],
            "session_duration": (datetime.now() - self.session_start_time).total_seconds(),
            "risk_limits": self.risk_limits
        } 