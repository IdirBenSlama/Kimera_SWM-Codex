"""
Kimera Trading System Core Types
===============================

Contains all shared types and data structures used across the trading system.
This helps prevent circular dependencies between modules.
"""

from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional
from src.core.geoid import GeoidState
from src.engines.contradiction_engine import TensionGradient

class MarketRegime(Enum):
    """Market regime classification enhanced with Kimera semantic understanding"""
    BULL_STRONG = "bull_strong"
    BULL_WEAK = "bull_weak" 
    BEAR_STRONG = "bear_strong"
    BEAR_WEAK = "bear_weak"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    BREAKOUT = "breakout"
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"
    CAPITULATION = "capitulation"
    SEMANTIC_ANOMALY = "semantic_anomaly"
    THERMODYNAMIC_TRANSITION = "thermodynamic_transition"

class TradingStrategy(Enum):
    """Trading strategies leveraging Kimera's semantic capabilities"""
    SEMANTIC_CONTRADICTION = "semantic_contradiction"
    THERMODYNAMIC_EQUILIBRIUM = "thermodynamic_equilibrium"
    COGNITIVE_FIELD_DYNAMICS = "cognitive_field_dynamics"
    GEOID_TENSION_ARBITRAGE = "geoid_tension_arbitrage"
    MOMENTUM_SURFING = "momentum_surfing"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT_HUNTER = "breakout_hunter"
    VOLATILITY_HARVESTER = "volatility_harvester"
    TREND_RIDER = "trend_rider"
    CHAOS_EXPLOITER = "chaos_exploiter"
    KIMERA_VAULT_PROTECTED = "kimera_vault_protected"
    MULTI_AGENT_SYNTHESIS = "multi_agent_synthesis"

class SemanticSignalType(Enum):
    """Types of semantic signals detected by Kimera engines"""
    PRICE_SENTIMENT_CONTRADICTION = "price_sentiment_contradiction"
    VOLUME_MOMENTUM_DIVERGENCE = "volume_momentum_divergence"
    NEWS_MARKET_DISSONANCE = "news_market_dissonance"
    TECHNICAL_FUNDAMENTAL_CONFLICT = "technical_fundamental_conflict"
    THERMODYNAMIC_PRESSURE_BUILDUP = "thermodynamic_pressure_buildup"
    COGNITIVE_FIELD_DISTORTION = "cognitive_field_distortion"
    GEOID_TENSION_GRADIENT = "geoid_tension_gradient"

@dataclass
class KimeraSemanticContradiction:
    """Enhanced semantic contradiction leveraging Kimera's engines"""
    contradiction_id: str
    geoid_a: GeoidState
    geoid_b: GeoidState
    tension_gradient: TensionGradient
    thermodynamic_pressure: float
    semantic_distance: float
    signal_type: SemanticSignalType
    opportunity_type: str
    confidence: float
    timestamp: datetime
    kimera_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass 
class KimeraCognitiveSignal:
    """Enhanced cognitive signal with full Kimera integration"""
    signal_id: str
    symbol: str
    action: str
    confidence: float
    conviction: float
    reasoning: List[str]
    strategy: TradingStrategy
    market_regime: MarketRegime
    source_geoid: Optional[GeoidState] = None
    semantic_contradictions: List[KimeraSemanticContradiction] = field(default_factory=list)
    thermodynamic_state: Dict[str, float] = field(default_factory=dict)
    cognitive_field_analysis: Dict[str, Any] = field(default_factory=dict)
    suggested_allocation_pct: float = 0.0
    max_risk_pct: float = 0.02
    entry_price: float = 0.0
    stop_loss: Optional[float] = None
    profit_targets: List[float] = field(default_factory=list)
    holding_period_hours: float = 4.0
    technical_score: float = 0.0
    fundamental_score: float = 0.0
    sentiment_score: float = 0.0
    momentum_score: float = 0.0
    contradiction_score: float = 0.0
    thermodynamic_score: float = 0.0
    cognitive_field_score: float = 0.0
    vault_security_score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class KimeraTradingPosition:
    """Trading position with Kimera semantic tracking"""
    position_id: str
    symbol: str
    side: str
    amount_base: float
    amount_quote: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    source_geoid: GeoidState
    semantic_context: Dict[str, Any]
    thermodynamic_validation: bool
    vault_protected: bool
    stop_loss: Optional[float]
    profit_targets: List[float]
    targets_hit: List[bool]
    trailing_stop: Optional[float]
    strategy: TradingStrategy
    conviction: float
    entry_reasoning: List[str]
    entry_time: datetime
    max_holding_hours: float
    last_update: datetime
    exchange: str
    order_ids: List[str]
    is_active: bool = True

@dataclass
class KimeraMarketData:
    """Market data structure enhanced with Kimera semantic analysis
    
    Attributes:
        symbol: Trading pair symbol (e.g. 'BTCUSDT')
        price: Current market price
        volume: 24h trading volume
        high_24h: 24h high price
        low_24h: 24h low price
        change_24h: Absolute price change over 24h
        change_pct_24h: Percentage price change over 24h
        bid: Current best bid price
        ask: Current best ask price
        spread: Current bid-ask spread
        timestamp: Data timestamp
        market_geoid: Geospatial state representation
        semantic_temperature: Semantic volatility measure (0-1)
        thermodynamic_pressure: Market pressure indicator
        cognitive_field_strength: Cognitive influence measure
        contradiction_count: Number of detected contradictions
        rsi: Relative Strength Index (14 period)
        macd: Moving Average Convergence Divergence
        bollinger_upper: Upper Bollinger Band
        bollinger_lower: Lower Bollinger Band
        moving_avg_20: 20-period moving average
        moving_avg_50: 50-period moving average
        volatility: Annualized volatility
        momentum: Price momentum indicator
        sentiment_score: Market sentiment (-1 to 1)
        semantic_state: Key semantic attributes
        symbolic_state: Numeric semantic attributes
    """
    symbol: str
    price: float
    volume: float
    high_24h: float
    low_24h: float
    change_24h: float
    change_pct_24h: float
    bid: float
    ask: float
    spread: float
    timestamp: datetime
    market_geoid: Optional[GeoidState] = None
    semantic_temperature: Optional[float] = None
    thermodynamic_pressure: Optional[float] = None
    cognitive_field_strength: Optional[float] = None
    contradiction_count: int = 0
    rsi: Optional[float] = None
    macd: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_lower: Optional[float] = None
    moving_avg_20: Optional[float] = None
    moving_avg_50: Optional[float] = None
    volatility: Optional[float] = None
    momentum: Optional[float] = None
    sentiment_score: Optional[float] = None
    semantic_state: Dict[str, str] = field(default_factory=dict)
    symbolic_state: Dict[str, float] = field(default_factory=dict)