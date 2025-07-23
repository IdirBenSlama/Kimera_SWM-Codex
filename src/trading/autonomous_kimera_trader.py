"""
KIMERA AUTONOMOUS TRADER
========================

Fully autonomous trading system with advanced AI decision-making.
NO SAFETY LIMITS - PURE COGNITIVE TRADING INTELLIGENCE.

MISSION: Grow €5 to €100 using autonomous cognitive algorithms.

KIMERA PHILOSOPHY:
- Maximum cognitive adaptability
- Dynamic strategy evolution
- Multi-dimensional market analysis
- Quantum-inspired decision trees
- Risk is opportunity when properly calculated
- Autonomous learning and adaptation
"""

import os
import json
import time
import logging
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN
from enum import Enum
import requests
import warnings
warnings.filterwarnings('ignore')

# Advanced imports for AI decision making
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.optimize import minimize
# import talib  # Optional - using custom technical indicators

# Configure aggressive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - KIMERA - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/autonomous_kimera.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('KIMERA_AUTONOMOUS')

class MarketRegime(Enum):
    """Market regime classification"""
    BULL_STRONG = "bull_strong"
    BULL_WEAK = "bull_weak" 
    BEAR_STRONG = "bear_strong"
    BEAR_WEAK = "bear_weak"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    BREAKOUT = "breakout"

class TradingStrategy(Enum):
    """Dynamic trading strategies"""
    MOMENTUM_SURFING = "momentum_surfing"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT_HUNTER = "breakout_hunter"
    VOLATILITY_HARVESTER = "volatility_harvester"
    ARBITRAGE_SEEKER = "arbitrage_seeker"
    TREND_RIDER = "trend_rider"
    CHAOS_EXPLOITER = "chaos_exploiter"

@dataclass
class CognitiveSignal:
    """Advanced cognitive trading signal"""
    symbol: str
    action: str  # 'buy', 'sell', 'hold', 'close'
    confidence: float  # 0.0 to 1.0
    conviction: float  # How strongly Kimera believes in this trade
    reasoning: List[str]
    strategy: TradingStrategy
    market_regime: MarketRegime
    
    # Position sizing (dynamic)
    suggested_allocation_pct: float  # % of portfolio
    max_risk_pct: float  # Max risk Kimera is willing to take
    
    # Price targets
    entry_price: float
    stop_loss: Optional[float]  # None = no stop loss
    profit_targets: List[float]  # Multiple profit levels
    
    # Time horizon
    holding_period_hours: float
    
    # Market analysis
    technical_score: float
    fundamental_score: float
    sentiment_score: float
    momentum_score: float
    
    timestamp: datetime

@dataclass
class AutonomousPosition:
    """Advanced position with dynamic management"""
    symbol: str
    side: str
    amount_eur: float
    amount_crypto: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    
    # Dynamic management
    stop_loss: Optional[float]
    profit_targets: List[float]
    targets_hit: List[bool]
    
    # Cognitive aspects
    strategy: TradingStrategy
    conviction: float
    entry_reasoning: List[str]
    
    # Time management
    entry_time: datetime
    max_holding_hours: float
    
    is_active: bool = True

class KimeraAutonomousTrader:
    """
    Fully autonomous trading system with advanced AI decision-making
    """
    
    def __init__(self, api_key: str, target_eur: float = 100.0):
        """
        Initialize autonomous Kimera trader
        
        Args:
            api_key: CDP API key  
            target_eur: Target portfolio value (€100)
        """
        self.api_key = api_key
        self.target_eur = target_eur
        self.start_capital = 5.0
        
        # Cognitive state
        self.current_strategy = TradingStrategy.MOMENTUM_SURFING
        self.market_regime = MarketRegime.SIDEWAYS
        self.learning_rate = 0.1
        self.adaptation_threshold = 0.3
        
        # Portfolio state
        self.portfolio_value = 5.0
        self.positions: Dict[str, AutonomousPosition] = {}
        self.trade_history: List[Dict] = []
        
        # AI models
        self.price_predictor = None
        self.volatility_predictor = None
        self.regime_classifier = None
        self.scaler = StandardScaler()
        
        # Market data
        self.price_history: Dict[str, pd.DataFrame] = {}
        self.market_features: Dict[str, Any] = {}
        
        # Performance tracking
        self.wins = 0
        self.losses = 0
        self.total_trades = 0
        self.max_drawdown = 0.0
        self.sharpe_ratio = 0.0
        
        # Cognitive evolution
        self.strategy_performance: Dict[TradingStrategy, float] = {
            strategy: 0.0 for strategy in TradingStrategy
        }
        
        logger.info("KIMERA AUTONOMOUS TRADER INITIALIZED")
        logger.info(f"   Target: EUR {target_eur}")
        logger.info(f"   Starting Capital: EUR {self.start_capital}")
        logger.info(f"   Growth Required: {(target_eur/self.start_capital)*100:.0f}%")
        logger.info("   NO SAFETY LIMITS - PURE AUTONOMOUS INTELLIGENCE")
        
        self._initialize_ai_models()
        self._load_autonomous_state()
    
    def _initialize_ai_models(self):
        """Initialize AI models for decision making"""
        try:
            # Price prediction model
            self.price_predictor = GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
            
            # Volatility prediction
            self.volatility_predictor = RandomForestRegressor(
                n_estimators=100,
                max_depth=8,
                random_state=42
            )
            
            logger.info("🤖 AI Models initialized for autonomous decision making")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize AI models: {e}")
    
    def _load_autonomous_state(self):
        """Load autonomous trading state"""
        try:
            if os.path.exists('data/autonomous_state.json'):
                with open('data/autonomous_state.json', 'r') as f:
                    state = json.load(f)
                    self.portfolio_value = state.get('portfolio_value', 5.0)
                    self.current_strategy = TradingStrategy(state.get('current_strategy', 'momentum_surfing'))
                    self.wins = state.get('wins', 0)
                    self.losses = state.get('losses', 0)
                    
                logger.info(f"Loaded autonomous state: Portfolio EUR {self.portfolio_value:.2f}")
        except Exception as e:
            logger.warning(f"Could not load autonomous state: {e}")
    
    def _save_autonomous_state(self):
        """Save autonomous trading state"""
        try:
            os.makedirs('data', exist_ok=True)
            state = {
                'portfolio_value': self.portfolio_value,
                'current_strategy': self.current_strategy.value,
                'market_regime': self.market_regime.value,
                'wins': self.wins,
                'losses': self.losses,
                'total_trades': self.total_trades,
                'strategy_performance': {k.value: v for k, v in self.strategy_performance.items()},
                'timestamp': datetime.now().isoformat()
            }
            with open('data/autonomous_state.json', 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save autonomous state: {e}")
    
    async def get_portfolio_status(self) -> Dict[str, Any]:
        """Get comprehensive portfolio status"""
        try:
            # Calculate total portfolio value
            total_value = self.portfolio_value
            
            # Progress to target
            progress_pct = (total_value / self.target_eur) * 100
            growth_from_start = ((total_value / self.start_capital) - 1) * 100
            
            # Win rate
            win_rate = (self.wins / max(self.total_trades, 1)) * 100
            
            status = {
                'portfolio_value_eur': total_value,
                'target_eur': self.target_eur,
                'progress_pct': progress_pct,
                'growth_from_start_pct': growth_from_start,
                'active_positions': len(self.positions),
                'current_strategy': self.current_strategy.value,
                'market_regime': self.market_regime.value,
                'total_trades': self.total_trades,
                'win_rate_pct': win_rate,
                'wins': self.wins,
                'losses': self.losses
            }
            
            return status
            
        except Exception as e:
            logger.error(f"❌ Failed to get portfolio status: {e}")
            return {}
    
    async def fetch_market_data(self, symbol: str, timeframe: str = '1h', limit: int = 200) -> pd.DataFrame:
        """Fetch comprehensive market data for analysis"""
        try:
            # Using CoinGecko API for free market data
            url = f"https://api.coingecko.com/api/v3/coins/{symbol.lower()}/market_chart"
            params = {
                'vs_currency': 'eur',
                'days': '30'
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if 'prices' in data:
                df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # Add technical indicators
                df['returns'] = df['price'].pct_change()
                df['volatility'] = df['returns'].rolling(24).std()
                df['sma_20'] = df['price'].rolling(20).mean()
                df['sma_50'] = df['price'].rolling(50).mean()
                df['rsi'] = self._calculate_rsi(df['price'])
                df['bollinger_upper'], df['bollinger_lower'] = self._calculate_bollinger_bands(df['price'])
                
                self.price_history[symbol] = df
                return df
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch market data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, lower
    
    def analyze_market_regime(self, symbol: str) -> MarketRegime:
        """Advanced market regime analysis"""
        try:
            if symbol not in self.price_history:
                return MarketRegime.SIDEWAYS
            
            df = self.price_history[symbol].copy()
            
            # Calculate regime indicators
            recent_returns = df['returns'].tail(24).mean()
            volatility = df['volatility'].tail(1).iloc[0]
            trend_strength = abs(df['sma_20'].tail(1).iloc[0] - df['sma_50'].tail(1).iloc[0]) / df['price'].tail(1).iloc[0]
            
            # Advanced regime classification
            if recent_returns > 0.02 and trend_strength > 0.05:
                regime = MarketRegime.BULL_STRONG
            elif recent_returns > 0.005:
                regime = MarketRegime.BULL_WEAK
            elif recent_returns < -0.02 and trend_strength > 0.05:
                regime = MarketRegime.BEAR_STRONG
            elif recent_returns < -0.005:
                regime = MarketRegime.BEAR_WEAK
            elif volatility > 0.05:
                regime = MarketRegime.VOLATILE
            else:
                regime = MarketRegime.SIDEWAYS
            
            self.market_regime = regime
            logger.info(f"📊 Market regime for {symbol}: {regime.value}")
            return regime
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze market regime: {e}")
            return MarketRegime.SIDEWAYS
    
    def select_optimal_strategy(self, regime: MarketRegime) -> TradingStrategy:
        """Select optimal strategy based on market regime"""
        strategy_map = {
            MarketRegime.BULL_STRONG: TradingStrategy.MOMENTUM_SURFING,
            MarketRegime.BULL_WEAK: TradingStrategy.TREND_RIDER,
            MarketRegime.BEAR_STRONG: TradingStrategy.MEAN_REVERSION,
            MarketRegime.BEAR_WEAK: TradingStrategy.VOLATILITY_HARVESTER,
            MarketRegime.SIDEWAYS: TradingStrategy.MEAN_REVERSION,
            MarketRegime.VOLATILE: TradingStrategy.VOLATILITY_HARVESTER,
            MarketRegime.BREAKOUT: TradingStrategy.BREAKOUT_HUNTER
        }
        
        # Consider strategy performance history
        base_strategy = strategy_map.get(regime, TradingStrategy.MOMENTUM_SURFING)
        
        # Adaptive strategy selection based on performance
        if self.total_trades > 10:
            best_strategy = max(self.strategy_performance.items(), key=lambda x: x[1])
            if best_strategy[1] > self.strategy_performance[base_strategy] + 0.1:
                logger.info(f"🧠 Adapting to best performing strategy: {best_strategy[0].value}")
                return best_strategy[0]
        
        return base_strategy
    
    def generate_cognitive_signal(self, symbol: str) -> Optional[CognitiveSignal]:
        """Generate advanced cognitive trading signal"""
        try:
            if symbol not in self.price_history:
                return None
            
            df = self.price_history[symbol].copy()
            current_price = df['price'].tail(1).iloc[0]
            
            # Analyze market regime
            regime = self.analyze_market_regime(symbol)
            
            # Select optimal strategy
            strategy = self.select_optimal_strategy(regime)
            self.current_strategy = strategy
            
            # Multi-dimensional analysis
            technical_score = self._analyze_technical_indicators(df)
            momentum_score = self._analyze_momentum(df)
            volatility_score = self._analyze_volatility(df)
            sentiment_score = self._analyze_market_sentiment()
            
            # Combine scores with cognitive weighting
            confidence = (technical_score * 0.3 + momentum_score * 0.3 + 
                         volatility_score * 0.2 + sentiment_score * 0.2)
            
            # Determine action based on strategy and confidence
            action = self._determine_action(strategy, confidence, df)
            
            if action == 'hold':
                return None
            
            # Calculate position sizing based on conviction
            conviction = min(confidence * 1.2, 1.0)  # Amplify confidence for conviction
            allocation_pct = self._calculate_position_size(conviction, regime, strategy)
            
            # Risk management (but not safety limits)
            max_risk_pct = min(conviction * 0.5, 0.3)  # Max 30% risk on single trade
            
            # Price targets
            stop_loss = self._calculate_dynamic_stop_loss(current_price, action, volatility_score)
            profit_targets = self._calculate_profit_targets(current_price, action, conviction)
            
            # Holding period based on strategy
            holding_hours = self._calculate_holding_period(strategy, volatility_score)
            
            signal = CognitiveSignal(
                symbol=symbol,
                action=action,
                confidence=confidence,
                conviction=conviction,
                reasoning=self._generate_reasoning(strategy, technical_score, momentum_score),
                strategy=strategy,
                market_regime=regime,
                suggested_allocation_pct=allocation_pct,
                max_risk_pct=max_risk_pct,
                entry_price=current_price,
                stop_loss=stop_loss,
                profit_targets=profit_targets,
                holding_period_hours=holding_hours,
                technical_score=technical_score,
                fundamental_score=0.5,  # Placeholder
                sentiment_score=sentiment_score,
                momentum_score=momentum_score,
                timestamp=datetime.now()
            )
            
            logger.info(f"🧠 Generated cognitive signal for {symbol}:")
            logger.info(f"   Action: {action} | Confidence: {confidence:.2f} | Conviction: {conviction:.2f}")
            logger.info(f"   Strategy: {strategy.value} | Regime: {regime.value}")
            logger.info(f"   Allocation: {allocation_pct:.1f}% | Max Risk: {max_risk_pct:.1f}%")
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Failed to generate cognitive signal: {e}")
            return None
    
    def _analyze_technical_indicators(self, df: pd.DataFrame) -> float:
        """Analyze technical indicators for scoring"""
        try:
            current_price = df['price'].tail(1).iloc[0]
            sma_20 = df['sma_20'].tail(1).iloc[0]
            sma_50 = df['sma_50'].tail(1).iloc[0]
            rsi = df['rsi'].tail(1).iloc[0]
            
            score = 0.5  # Neutral base
            
            # Price vs moving averages
            if current_price > sma_20 > sma_50:
                score += 0.3  # Strong uptrend
            elif current_price > sma_20:
                score += 0.1  # Mild uptrend
            elif current_price < sma_20 < sma_50:
                score -= 0.3  # Strong downtrend
            elif current_price < sma_20:
                score -= 0.1  # Mild downtrend
            
            # RSI analysis
            if 30 < rsi < 70:
                score += 0.1  # Neutral RSI
            elif rsi > 80:
                score -= 0.2  # Overbought
            elif rsi < 20:
                score += 0.2  # Oversold
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"❌ Technical analysis failed: {e}")
            return 0.5
    
    def _analyze_momentum(self, df: pd.DataFrame) -> float:
        """Analyze momentum indicators"""
        try:
            returns_1h = df['returns'].tail(1).iloc[0]
            returns_24h = df['returns'].tail(24).mean()
            volatility = df['volatility'].tail(1).iloc[0]
            
            momentum_score = 0.5
            
            # Recent momentum
            if returns_1h > 0.01:
                momentum_score += 0.3
            elif returns_1h > 0.005:
                momentum_score += 0.1
            elif returns_1h < -0.01:
                momentum_score -= 0.3
            elif returns_1h < -0.005:
                momentum_score -= 0.1
            
            # Medium-term momentum  
            if returns_24h > 0.02:
                momentum_score += 0.2
            elif returns_24h < -0.02:
                momentum_score -= 0.2
            
            return max(0.0, min(1.0, momentum_score))
            
        except Exception as e:
            logger.error(f"❌ Momentum analysis failed: {e}")
            return 0.5
    
    def _analyze_volatility(self, df: pd.DataFrame) -> float:
        """Analyze volatility for opportunity scoring"""
        try:
            current_vol = df['volatility'].tail(1).iloc[0]
            avg_vol = df['volatility'].tail(24).mean()
            
            vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
            
            # Higher volatility = higher opportunity (for Kimera)
            if vol_ratio > 1.5:
                return 0.8  # High volatility opportunity
            elif vol_ratio > 1.2:
                return 0.7
            elif vol_ratio < 0.8:
                return 0.3  # Low volatility
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"❌ Volatility analysis failed: {e}")
            return 0.5
    
    def _analyze_market_sentiment(self) -> float:
        """Analyze market sentiment (simplified)"""
        # Placeholder for sentiment analysis
        # Could integrate with social media, news, etc.
        return 0.5
    
    def _determine_action(self, strategy: TradingStrategy, confidence: float, df: pd.DataFrame) -> str:
        """Determine trading action based on strategy and analysis"""
        current_price = df['price'].tail(1).iloc[0]
        sma_20 = df['sma_20'].tail(1).iloc[0]
        rsi = df['rsi'].tail(1).iloc[0]
        
        # Strategy-specific logic
        if strategy == TradingStrategy.MOMENTUM_SURFING:
            if confidence > 0.6 and current_price > sma_20 and rsi < 80:
                return 'buy'
            elif confidence < 0.4 and current_price < sma_20:
                return 'sell'
        
        elif strategy == TradingStrategy.MEAN_REVERSION:
            if confidence > 0.6 and rsi < 30:
                return 'buy'
            elif confidence < 0.4 and rsi > 70:
                return 'sell'
        
        elif strategy == TradingStrategy.VOLATILITY_HARVESTER:
            if confidence > 0.5:
                # Buy or sell based on momentum direction
                recent_return = df['returns'].tail(1).iloc[0]
                return 'buy' if recent_return > 0 else 'sell'
        
        # Default to hold
        return 'hold'
    
    def _calculate_position_size(self, conviction: float, regime: MarketRegime, strategy: TradingStrategy) -> float:
        """Calculate position size based on conviction and conditions"""
        base_allocation = 0.2  # 20% base allocation
        
        # Conviction multiplier
        conviction_multiplier = 0.5 + (conviction * 1.5)  # 0.5x to 2.0x
        
        # Regime multiplier
        regime_multipliers = {
            MarketRegime.BULL_STRONG: 1.5,
            MarketRegime.BULL_WEAK: 1.2,
            MarketRegime.BEAR_STRONG: 0.8,
            MarketRegime.BEAR_WEAK: 1.0,
            MarketRegime.SIDEWAYS: 1.0,
            MarketRegime.VOLATILE: 1.3,
            MarketRegime.BREAKOUT: 1.8
        }
        
        regime_mult = regime_multipliers.get(regime, 1.0)
        
        # Strategy multiplier
        strategy_multipliers = {
            TradingStrategy.MOMENTUM_SURFING: 1.4,
            TradingStrategy.VOLATILITY_HARVESTER: 1.6,
            TradingStrategy.BREAKOUT_HUNTER: 1.8,
            TradingStrategy.MEAN_REVERSION: 1.2,
            TradingStrategy.TREND_RIDER: 1.3
        }
        
        strategy_mult = strategy_multipliers.get(strategy, 1.0)
        
        # Calculate final allocation
        allocation = base_allocation * conviction_multiplier * regime_mult * strategy_mult
        
        # Cap at 80% of portfolio for single position
        return min(allocation, 0.8)
    
    def _calculate_dynamic_stop_loss(self, current_price: float, action: str, volatility_score: float) -> Optional[float]:
        """Calculate dynamic stop loss based on market conditions"""
        if action == 'buy':
            # Adaptive stop loss based on volatility
            stop_pct = 0.02 + (volatility_score * 0.08)  # 2% to 10% stop loss
            return current_price * (1 - stop_pct)
        elif action == 'sell':
            stop_pct = 0.02 + (volatility_score * 0.08)
            return current_price * (1 + stop_pct)
        
        return None
    
    def _calculate_profit_targets(self, current_price: float, action: str, conviction: float) -> List[float]:
        """Calculate multiple profit targets"""
        targets = []
        
        if action == 'buy':
            # Conviction-based profit targets
            base_target = 0.03 + (conviction * 0.07)  # 3% to 10% base target
            targets.append(current_price * (1 + base_target))
            targets.append(current_price * (1 + base_target * 2))
            targets.append(current_price * (1 + base_target * 3))
        
        elif action == 'sell':
            base_target = 0.03 + (conviction * 0.07)
            targets.append(current_price * (1 - base_target))
            targets.append(current_price * (1 - base_target * 2))
            targets.append(current_price * (1 - base_target * 3))
        
        return targets
    
    def _calculate_holding_period(self, strategy: TradingStrategy, volatility_score: float) -> float:
        """Calculate optimal holding period"""
        base_hours = {
            TradingStrategy.MOMENTUM_SURFING: 2.0,
            TradingStrategy.MEAN_REVERSION: 6.0,
            TradingStrategy.BREAKOUT_HUNTER: 1.0,
            TradingStrategy.VOLATILITY_HARVESTER: 0.5,
            TradingStrategy.TREND_RIDER: 12.0
        }
        
        base = base_hours.get(strategy, 4.0)
        
        # Adjust for volatility
        volatility_adjustment = 1.0 - (volatility_score * 0.5)
        
        return base * volatility_adjustment
    
    def _generate_reasoning(self, strategy: TradingStrategy, technical_score: float, momentum_score: float) -> List[str]:
        """Generate human-readable reasoning for the trade"""
        reasoning = []
        
        reasoning.append(f"Strategy: {strategy.value}")
        reasoning.append(f"Technical Score: {technical_score:.2f}")
        reasoning.append(f"Momentum Score: {momentum_score:.2f}")
        
        if technical_score > 0.6:
            reasoning.append("Strong technical indicators support the trade")
        elif technical_score < 0.4:
            reasoning.append("Technical indicators suggest caution")
        
        if momentum_score > 0.6:
            reasoning.append("Positive momentum detected")
        elif momentum_score < 0.4:
            reasoning.append("Negative momentum present")
        
        return reasoning
    
    async def execute_autonomous_trade(self, signal: CognitiveSignal) -> bool:
        """Execute trade based on cognitive signal"""
        try:
            # Calculate position size in EUR
            position_eur = self.portfolio_value * signal.suggested_allocation_pct
            
            logger.info(f"🚀 EXECUTING AUTONOMOUS TRADE:")
            logger.info(f"   Symbol: {signal.symbol}")
            logger.info(f"   Action: {signal.action}")
            logger.info(f"   Position Size: €{position_eur:.2f}")
            logger.info(f"   Confidence: {signal.confidence:.2f}")
            logger.info(f"   Conviction: {signal.conviction:.2f}")
            logger.info(f"   Strategy: {signal.strategy.value}")
            
            # Create position
            position = AutonomousPosition(
                symbol=signal.symbol,
                side=signal.action,
                amount_eur=position_eur,
                amount_crypto=position_eur / signal.entry_price,
                entry_price=signal.entry_price,
                current_price=signal.entry_price,
                unrealized_pnl=0.0,
                stop_loss=signal.stop_loss,
                profit_targets=signal.profit_targets,
                targets_hit=[False] * len(signal.profit_targets),
                strategy=signal.strategy,
                conviction=signal.conviction,
                entry_reasoning=signal.reasoning,
                entry_time=datetime.now(),
                max_holding_hours=signal.holding_period_hours
            )
            
            # Add to positions
            self.positions[signal.symbol] = position
            
            # Update statistics
            self.total_trades += 1
            
            # Save state
            self._save_autonomous_state()
            
            logger.info(f"✅ Autonomous trade executed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to execute autonomous trade: {e}")
            return False
    
    async def manage_positions(self):
        """Autonomous position management"""
        try:
            for symbol, position in list(self.positions.items()):
                if not position.is_active:
                    continue
                
                # Get current price
                current_df = await self.fetch_market_data(symbol)
                if current_df.empty:
                    continue
                
                current_price = current_df['price'].tail(1).iloc[0]
                position.current_price = current_price
                
                # Calculate P&L
                if position.side == 'buy':
                    position.unrealized_pnl = (current_price - position.entry_price) * position.amount_crypto
                else:  # sell
                    position.unrealized_pnl = (position.entry_price - current_price) * position.amount_crypto
                
                # Check stop loss
                should_close = False
                close_reason = ""
                
                if position.stop_loss:
                    if position.side == 'buy' and current_price <= position.stop_loss:
                        should_close = True
                        close_reason = "Stop loss hit"
                    elif position.side == 'sell' and current_price >= position.stop_loss:
                        should_close = True
                        close_reason = "Stop loss hit"
                
                # Check profit targets
                for i, target in enumerate(position.profit_targets):
                    if not position.targets_hit[i]:
                        if position.side == 'buy' and current_price >= target:
                            position.targets_hit[i] = True
                            logger.info(f"🎯 Profit target {i+1} hit for {symbol}: €{target:.2f}")
                            
                            # Close partial position or full position based on target
                            if i == len(position.profit_targets) - 1:  # Final target
                                should_close = True
                                close_reason = "Final profit target reached"
                        
                        elif position.side == 'sell' and current_price <= target:
                            position.targets_hit[i] = True
                            logger.info(f"🎯 Profit target {i+1} hit for {symbol}: €{target:.2f}")
                            
                            if i == len(position.profit_targets) - 1:
                                should_close = True
                                close_reason = "Final profit target reached"
                
                # Check holding period
                holding_hours = (datetime.now() - position.entry_time).total_seconds() / 3600
                if holding_hours >= position.max_holding_hours:
                    should_close = True
                    close_reason = "Maximum holding period reached"
                
                # Close position if needed
                if should_close:
                    await self._close_position(symbol, close_reason)
        
        except Exception as e:
            logger.error(f"❌ Failed to manage positions: {e}")
    
    async def _close_position(self, symbol: str, reason: str):
        """Close a position"""
        try:
            position = self.positions[symbol]
            
            # Calculate final P&L
            final_pnl = position.unrealized_pnl
            
            # Update portfolio value
            self.portfolio_value += final_pnl
            
            # Update statistics
            if final_pnl > 0:
                self.wins += 1
            else:
                self.losses += 1
            
            # Update strategy performance
            self.strategy_performance[position.strategy] += final_pnl / position.amount_eur
            
            # Log closure
            logger.info(f"🔄 POSITION CLOSED:")
            logger.info(f"   Symbol: {symbol}")
            logger.info(f"   Reason: {reason}")
            logger.info(f"   P&L: €{final_pnl:.2f}")
            logger.info(f"   Portfolio Value: €{self.portfolio_value:.2f}")
            
            # Remove from active positions
            position.is_active = False
            del self.positions[symbol]
            
            # Save state
            self._save_autonomous_state()
            
        except Exception as e:
            logger.error(f"❌ Failed to close position: {e}")
    
    async def autonomous_trading_cycle(self):
        """Main autonomous trading cycle"""
        logger.info("🧠 Starting autonomous trading cycle...")
        
        # Symbols to trade
        symbols = ['bitcoin', 'ethereum', 'solana', 'cardano', 'polkadot']
        
        try:
            # Fetch market data for all symbols
            for symbol in symbols:
                await self.fetch_market_data(symbol)
            
            # Manage existing positions
            await self.manage_positions()
            
            # Generate new signals if not at position limit
            if len(self.positions) < 3:  # Max 3 concurrent positions
                for symbol in symbols:
                    if symbol not in self.positions:
                        signal = self.generate_cognitive_signal(symbol)
                        
                        if signal and signal.confidence > 0.6:
                            await self.execute_autonomous_trade(signal)
                            break  # Only one new position per cycle
            
            # Log current status
            status = await self.get_portfolio_status()
            logger.info(f"📊 Portfolio Status:")
            logger.info(f"   Value: €{status['portfolio_value_eur']:.2f}")
            logger.info(f"   Progress: {status['progress_pct']:.1f}%")
            logger.info(f"   Active Positions: {status['active_positions']}")
            logger.info(f"   Win Rate: {status['win_rate_pct']:.1f}%")
            
            # Check if target reached
            if self.portfolio_value >= self.target_eur:
                logger.info(f"🎉 TARGET REACHED! Portfolio: €{self.portfolio_value:.2f}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Autonomous trading cycle failed: {e}")
            return False
    
    async def run_autonomous_trader(self, cycle_interval_minutes: int = 15):
        """Run the autonomous trader continuously"""
        logger.info(f"🚀 KIMERA AUTONOMOUS TRADER STARTED")
        logger.info(f"   Target: €{self.target_eur}")
        logger.info(f"   Cycle Interval: {cycle_interval_minutes} minutes")
        logger.info("   NO SAFETY LIMITS - PURE AUTONOMOUS INTELLIGENCE")
        
        try:
            while True:
                target_reached = await self.autonomous_trading_cycle()
                
                if target_reached:
                    logger.info("🎯 Mission accomplished! Target reached.")
                    break
                
                # Wait for next cycle
                await asyncio.sleep(cycle_interval_minutes * 60)
                
        except KeyboardInterrupt:
            logger.info("🛑 Autonomous trader stopped by user")
        except Exception as e:
            logger.error(f"❌ Autonomous trader crashed: {e}")

# Factory function
def create_autonomous_kimera(api_key: str, target_eur: float = 100.0) -> KimeraAutonomousTrader:
    """Create autonomous Kimera trader instance"""
    return KimeraAutonomousTrader(api_key, target_eur)

if __name__ == "__main__":
    # Example usage
    API_KEY = "9268de76-b5f4-4683-b593-327fb2c19503"
    
    trader = create_autonomous_kimera(API_KEY, target_eur=100.0)
    
    # Run autonomous trader
    asyncio.run(trader.run_autonomous_trader(cycle_interval_minutes=15)) 