"""
Strategic Profiler System for KIMERA Trading
===========================================

Creates adaptive profiling systems that identify, analyze and respond to different 
market participants and opportunities through behavioral analysis, context selection,
and dependency selection with scientific rigor.

Integrates:
- Anthropomorphic profiling for behavioral analysis
- Context field selection for market targeting
- Manipulation detection for participant identification
- Rules engine for strategic responses
- Warfare tactics for competitive advantage
"""

import logging
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json

# KIMERA Core Imports
from ..core.anthropomorphic_profiler import (
    AnthropomorphicProfiler, PersonalityProfile, PersonalityTrait, 
    InteractionAnalysis, create_default_profiler
)
from ..core.context_field_selector import (
    ContextFieldSelector, ContextFieldConfig, FieldCategory,
    ProcessingLevel, create_domain_selector
)
from ..core.selective_feedback_interpreter import SelectiveFeedbackInterpreter
from .market_manipulation_detector import AdvancedManipulationDetector, ManipulationSignal
from .advanced_rules_engine import AdvancedRulesEngine, TradingRule, RuleType
from .advanced_financial_processor import AdvancedFinancialProcessor, TechnicalSignal

logger = logging.getLogger(__name__)

class TraderArchetype(Enum):
    """Strategic trader archetypes based on behavioral patterns"""
    INSTITUTIONAL_WHALE = "institutional_whale"
    ALGORITHMIC_HFT = "algorithmic_hft"
    RETAIL_MOMENTUM = "retail_momentum"
    SMART_MONEY = "smart_money"
    MARKET_MAKER = "market_maker"
    ARBITRAGEUR = "arbitrageur"
    MANIPULATOR = "manipulator"
    PANIC_SELLER = "panic_seller"
    DIAMOND_HANDS = "diamond_hands"
    SWING_TRADER = "swing_trader"

class MarketRegime(Enum):
    """Market regime classifications"""
    BULL_TRENDING = "bull_trending"
    BEAR_TRENDING = "bear_trending"
    SIDEWAYS_RANGING = "sideways_ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRISIS_MODE = "crisis_mode"
    EUPHORIA_MODE = "euphoria_mode"

class StrategicIntent(Enum):
    """Strategic intentions of market participants"""
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"
    MANIPULATION = "manipulation"
    ARBITRAGE = "arbitrage"
    HEDGING = "hedging"
    SPECULATION = "speculation"
    LIQUIDATION = "liquidation"

@dataclass
class TraderProfile:
    """Comprehensive trader behavioral profile"""
    archetype: TraderArchetype
    confidence: float
    behavioral_traits: Dict[str, float]
    trading_patterns: Dict[str, Any]
    risk_profile: Dict[str, float]
    time_horizons: List[str]
    preferred_instruments: List[str]
    market_impact: float
    detection_signals: List[str]
    counter_strategies: List[str]
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class MarketContext:
    """Market context for strategic analysis"""
    regime: MarketRegime
    volatility_level: float
    liquidity_level: float
    sentiment_score: float
    manipulation_risk: float
    opportunity_score: float
    active_participants: List[TraderArchetype]
    dominant_intent: StrategicIntent
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class StrategicResponse:
    """Strategic response to identified market participants"""
    target_archetype: TraderArchetype
    response_strategy: str
    tactical_actions: List[str]
    risk_adjustments: Dict[str, float]
    timing_considerations: Dict[str, Any]
    success_probability: float
    expected_profit: float
    max_drawdown: float

class StrategicProfilerSystem:
    """
    Master strategic profiler system that combines all KIMERA intelligence
    capabilities to create adaptive trader behavioral profiles
    """
    
    def __init__(self):
        # Initialize core components
        self.anthropomorphic_profiler = create_default_profiler()
        self.context_selector = create_domain_selector('financial')
        self.selective_interpreter = SelectiveFeedbackInterpreter(self.anthropomorphic_profiler)
        self.manipulation_detector = AdvancedManipulationDetector()
        self.rules_engine = AdvancedRulesEngine()
        self.financial_processor = AdvancedFinancialProcessor()
        
        # Profiling state
        self.active_profiles: Dict[str, TraderProfile] = {}
        self.market_context_history: List[MarketContext] = []
        self.strategic_responses: Dict[str, List[StrategicResponse]] = {}
        
        # Learning systems
        self.behavioral_patterns: Dict[TraderArchetype, Dict[str, Any]] = {}
        self.success_rates: Dict[str, float] = {}
        self.adaptation_history: List[Dict[str, Any]] = []
        
        # Initialize archetype definitions
        self._initialize_trader_archetypes()
        
        logger.info("🎯 Strategic Profiler System initialized - Ready for market warfare")
    
    def _initialize_trader_archetypes(self):
        """Initialize behavioral patterns for different trader archetypes"""
        
        self.behavioral_patterns = {
            TraderArchetype.INSTITUTIONAL_WHALE: {
                'volume_patterns': {'large_blocks': 0.9, 'iceberg_orders': 0.8, 'time_weighted': 0.7},
                'timing_patterns': {'off_hours': 0.6, 'gradual_execution': 0.9, 'news_avoidance': 0.8},
                'price_impact': {'minimal_slippage': 0.9, 'stealth_execution': 0.8},
                'behavioral_traits': {'patience': 0.9, 'discipline': 0.9, 'information_advantage': 0.8},
                'detection_signals': ['large_volume_low_impact', 'iceberg_patterns', 'gradual_accumulation'],
                'counter_strategies': ['front_run_detection', 'momentum_piggyback', 'liquidity_provision']
            },
            
            TraderArchetype.ALGORITHMIC_HFT: {
                'volume_patterns': {'high_frequency': 0.95, 'small_sizes': 0.8, 'rapid_cancellation': 0.9},
                'timing_patterns': {'microsecond_precision': 0.95, 'market_open_close': 0.8},
                'price_impact': {'spread_capture': 0.9, 'latency_arbitrage': 0.8},
                'behavioral_traits': {'speed': 0.95, 'precision': 0.9, 'emotionless': 0.95},
                'detection_signals': ['sub_second_patterns', 'spread_scalping', 'order_flow_prediction'],
                'counter_strategies': ['latency_competition', 'adverse_selection', 'toxic_flow_avoidance']
            },
            
            TraderArchetype.SMART_MONEY: {
                'volume_patterns': {'strategic_timing': 0.9, 'contrarian_moves': 0.8},
                'timing_patterns': {'information_edge': 0.9, 'early_positioning': 0.8},
                'price_impact': {'trend_initiation': 0.8, 'reversal_timing': 0.7},
                'behavioral_traits': {'insight': 0.9, 'contrarian': 0.7, 'patient': 0.8},
                'detection_signals': ['early_trend_entry', 'contrarian_positioning', 'information_advantage'],
                'counter_strategies': ['follow_smart_money', 'early_trend_detection', 'insider_tracking']
            },
            
            TraderArchetype.RETAIL_MOMENTUM: {
                'volume_patterns': {'trend_following': 0.8, 'social_driven': 0.7, 'emotional_sizing': 0.9},
                'timing_patterns': {'news_reactive': 0.9, 'fomo_driven': 0.8, 'weekend_effect': 0.6},
                'price_impact': {'momentum_amplification': 0.7, 'volatility_creation': 0.8},
                'behavioral_traits': {'emotional': 0.9, 'impulsive': 0.8, 'social_influenced': 0.9},
                'detection_signals': ['social_sentiment_correlation', 'news_reaction_patterns', 'momentum_chasing'],
                'counter_strategies': ['contrarian_positioning', 'momentum_fade', 'sentiment_exploitation']
            },
            
            TraderArchetype.MANIPULATOR: {
                'volume_patterns': {'spoofing': 0.8, 'wash_trading': 0.7, 'layering': 0.9},
                'timing_patterns': {'low_liquidity_targeting': 0.9, 'news_manipulation': 0.7},
                'price_impact': {'artificial_movements': 0.9, 'false_signals': 0.8},
                'behavioral_traits': {'deceptive': 0.95, 'opportunistic': 0.9, 'risk_taking': 0.8},
                'detection_signals': ['manipulation_patterns', 'artificial_volume', 'false_breakouts'],
                'counter_strategies': ['manipulation_detection', 'regulatory_reporting', 'avoidance_protocols']
            }
        }
    
    async def analyze_market_participants(self, market_data: Dict[str, Any]) -> List[TraderProfile]:
        """
        Analyze market data to identify active trader archetypes and their behavioral patterns
        """
        try:
            logger.info("🔍 Analyzing market participants...")
            
            # Prepare market data for analysis
            processed_data = await self._prepare_market_data(market_data)
            
            # Run parallel analysis
            analysis_tasks = [
                self._detect_institutional_activity(processed_data),
                self._detect_algorithmic_activity(processed_data),
                self._detect_retail_patterns(processed_data),
                self._detect_smart_money_flows(processed_data),
                self._detect_manipulation_patterns(processed_data)
            ]
            
            detection_results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            
            # Consolidate detected profiles
            detected_profiles = []
            for result in detection_results:
                if isinstance(result, list):
                    detected_profiles.extend(result)
                elif isinstance(result, Exception):
                    logger.error(f"Detection error: {result}")
            
            # Update active profiles
            self._update_active_profiles(detected_profiles)
            
            logger.info(f"✅ Detected {len(detected_profiles)} active trader archetypes")
            return detected_profiles
            
        except Exception as e:
            logger.error(f"Error analyzing market participants: {e}")
            return []
    
    async def _prepare_market_data(self, market_data: Dict[str, Any]) -> pd.DataFrame:
        """Prepare and enrich market data for participant analysis"""
        
        # Convert to DataFrame if needed
        if isinstance(market_data, dict):
            df = pd.DataFrame([market_data])
        else:
            df = market_data.copy()
        
        # Add technical indicators
        if len(df) >= 20:  # Need minimum data for indicators
            signals = await self.financial_processor.process_market_data(df)
            
            # Merge technical signals
            for signal in signals:
                df[f'signal_{signal.indicator}'] = signal.value
                df[f'signal_{signal.indicator}_confidence'] = signal.confidence
        
        # Add manipulation detection features
        manipulation_signals = await self.manipulation_detector.analyze_manipulation(df)
        df['manipulation_risk'] = len(manipulation_signals) / max(len(df), 1)
        
        return df
    
    async def _detect_institutional_activity(self, data: pd.DataFrame) -> List[TraderProfile]:
        """Detect institutional whale activity patterns"""
        profiles = []
        
        try:
            # Large volume, low price impact patterns
            if 'volume' in data.columns and 'close' in data.columns:
                volume_ma = data['volume'].rolling(20).mean()
                price_change = data['close'].pct_change()
                
                # Detect iceberg patterns (large volume, minimal price impact)
                iceberg_conditions = (
                    (data['volume'] > volume_ma * 2) &
                    (abs(price_change) < price_change.rolling(20).std())
                )
                
                if iceberg_conditions.any():
                    confidence = min(0.85, iceberg_conditions.sum() / len(data))
                    
                    profile = TraderProfile(
                        archetype=TraderArchetype.INSTITUTIONAL_WHALE,
                        confidence=confidence,
                        behavioral_traits={
                            'patience': 0.9,
                            'discipline': 0.9,
                            'stealth_execution': 0.8
                        },
                        trading_patterns={
                            'iceberg_orders': True,
                            'time_weighted_execution': True,
                            'minimal_market_impact': True
                        },
                        risk_profile={
                            'position_size': 0.9,
                            'time_horizon': 0.8,
                            'risk_tolerance': 0.6
                        },
                        time_horizons=['medium_term', 'long_term'],
                        preferred_instruments=['large_cap_stocks', 'etfs', 'bonds'],
                        market_impact=0.8,
                        detection_signals=['large_volume_low_impact', 'gradual_accumulation'],
                        counter_strategies=['momentum_piggyback', 'liquidity_provision']
                    )
                    profiles.append(profile)
                    
        except Exception as e:
            logger.error(f"Error detecting institutional activity: {e}")
        
        return profiles
    
    async def _detect_algorithmic_activity(self, data: pd.DataFrame) -> List[TraderProfile]:
        """Detect high-frequency algorithmic trading patterns"""
        profiles = []
        
        try:
            # Look for sub-second patterns and rapid order cancellations
            if 'timestamp' in data.columns:
                # Simulate HFT detection (would need tick data for real detection)
                time_diffs = pd.to_datetime(data['timestamp']).diff().dt.total_seconds()
                
                # Detect rapid-fire trading
                rapid_trading = (time_diffs < 1.0).sum() / len(data) if len(data) > 1 else 0
                
                if rapid_trading > 0.3:  # 30% of trades within 1 second
                    confidence = min(0.95, rapid_trading * 2)
                    
                    profile = TraderProfile(
                        archetype=TraderArchetype.ALGORITHMIC_HFT,
                        confidence=confidence,
                        behavioral_traits={
                            'speed': 0.95,
                            'precision': 0.9,
                            'emotionless': 0.95
                        },
                        trading_patterns={
                            'high_frequency': True,
                            'spread_scalping': True,
                            'latency_arbitrage': True
                        },
                        risk_profile={
                            'position_size': 0.3,
                            'time_horizon': 0.1,
                            'risk_tolerance': 0.4
                        },
                        time_horizons=['seconds', 'minutes'],
                        preferred_instruments=['liquid_stocks', 'futures', 'forex'],
                        market_impact=0.3,
                        detection_signals=['sub_second_patterns', 'spread_scalping'],
                        counter_strategies=['latency_competition', 'adverse_selection']
                    )
                    profiles.append(profile)
                    
        except Exception as e:
            logger.error(f"Error detecting algorithmic activity: {e}")
        
        return profiles
    
    async def _detect_retail_patterns(self, data: pd.DataFrame) -> List[TraderProfile]:
        """Detect retail momentum trading patterns"""
        profiles = []
        
        try:
            # Look for momentum chasing and emotional trading patterns
            if 'close' in data.columns and 'volume' in data.columns:
                price_momentum = data['close'].pct_change(5)  # 5-period momentum
                volume_spike = data['volume'] / data['volume'].rolling(10).mean()
                
                # Detect momentum chasing (high volume on price moves)
                momentum_chasing = (
                    (abs(price_momentum) > price_momentum.std() * 2) &
                    (volume_spike > 2.0)
                )
                
                if momentum_chasing.any():
                    confidence = min(0.8, momentum_chasing.sum() / len(data) * 3)
                    
                    profile = TraderProfile(
                        archetype=TraderArchetype.RETAIL_MOMENTUM,
                        confidence=confidence,
                        behavioral_traits={
                            'emotional': 0.9,
                            'impulsive': 0.8,
                            'social_influenced': 0.9
                        },
                        trading_patterns={
                            'momentum_chasing': True,
                            'news_reactive': True,
                            'fomo_driven': True
                        },
                        risk_profile={
                            'position_size': 0.6,
                            'time_horizon': 0.4,
                            'risk_tolerance': 0.8
                        },
                        time_horizons=['minutes', 'hours', 'days'],
                        preferred_instruments=['meme_stocks', 'crypto', 'options'],
                        market_impact=0.5,
                        detection_signals=['momentum_chasing', 'social_sentiment_correlation'],
                        counter_strategies=['contrarian_positioning', 'momentum_fade']
                    )
                    profiles.append(profile)
                    
        except Exception as e:
            logger.error(f"Error detecting retail patterns: {e}")
        
        return profiles
    
    async def _detect_smart_money_flows(self, data: pd.DataFrame) -> List[TraderProfile]:
        """Detect smart money and early trend detection"""
        profiles = []
        
        try:
            # Look for early trend entry and contrarian positioning
            if 'close' in data.columns and len(data) >= 50:
                # Calculate trend strength
                ma_short = data['close'].rolling(10).mean()
                ma_long = data['close'].rolling(50).mean()
                trend_strength = (ma_short - ma_long) / ma_long
                
                # Detect early trend entries (before major moves)
                future_returns = data['close'].shift(-10) / data['close'] - 1
                early_entries = (
                    (abs(trend_strength) < 0.02) &  # Flat trend
                    (abs(future_returns) > 0.05)    # But big move coming
                )
                
                if early_entries.any():
                    confidence = min(0.85, early_entries.sum() / len(data) * 5)
                    
                    profile = TraderProfile(
                        archetype=TraderArchetype.SMART_MONEY,
                        confidence=confidence,
                        behavioral_traits={
                            'insight': 0.9,
                            'contrarian': 0.7,
                            'patient': 0.8
                        },
                        trading_patterns={
                            'early_positioning': True,
                            'contrarian_moves': True,
                            'information_advantage': True
                        },
                        risk_profile={
                            'position_size': 0.7,
                            'time_horizon': 0.8,
                            'risk_tolerance': 0.6
                        },
                        time_horizons=['days', 'weeks', 'months'],
                        preferred_instruments=['undervalued_stocks', 'sectors', 'commodities'],
                        market_impact=0.7,
                        detection_signals=['early_trend_entry', 'contrarian_positioning'],
                        counter_strategies=['follow_smart_money', 'early_trend_detection']
                    )
                    profiles.append(profile)
                    
        except Exception as e:
            logger.error(f"Error detecting smart money flows: {e}")
        
        return profiles
    
    async def _detect_manipulation_patterns(self, data: pd.DataFrame) -> List[TraderProfile]:
        """Detect market manipulation patterns"""
        profiles = []
        
        try:
            # Use manipulation detector
            manipulation_signals = await self.manipulation_detector.analyze_manipulation(data)
            
            if manipulation_signals:
                # Group by manipulation type
                manipulation_types = {}
                for signal in manipulation_signals:
                    if signal.manipulation_type not in manipulation_types:
                        manipulation_types[signal.manipulation_type] = []
                    manipulation_types[signal.manipulation_type].append(signal)
                
                # Create profile for each type detected
                for manip_type, signals in manipulation_types.items():
                    avg_confidence = sum(s.confidence for s in signals) / len(signals)
                    
                    if avg_confidence > 0.6:
                        profile = TraderProfile(
                            archetype=TraderArchetype.MANIPULATOR,
                            confidence=avg_confidence,
                            behavioral_traits={
                                'deceptive': 0.95,
                                'opportunistic': 0.9,
                                'risk_taking': 0.8
                            },
                            trading_patterns={
                                'manipulation_type': manip_type,
                                'artificial_movements': True,
                                'false_signals': True
                            },
                            risk_profile={
                                'position_size': 0.5,
                                'time_horizon': 0.3,
                                'risk_tolerance': 0.9
                            },
                            time_horizons=['minutes', 'hours'],
                            preferred_instruments=['low_liquidity', 'small_cap', 'crypto'],
                            market_impact=0.8,
                            detection_signals=[f'{manip_type}_patterns', 'artificial_volume'],
                            counter_strategies=['manipulation_detection', 'avoidance_protocols']
                        )
                        profiles.append(profile)
                        
        except Exception as e:
            logger.error(f"Error detecting manipulation patterns: {e}")
        
        return profiles
    
    def _update_active_profiles(self, new_profiles: List[TraderProfile]):
        """Update active trader profiles with new detections"""
        
        for profile in new_profiles:
            profile_key = f"{profile.archetype.value}_{profile.confidence:.2f}"
            
            if profile_key in self.active_profiles:
                # Update existing profile
                existing = self.active_profiles[profile_key]
                existing.confidence = (existing.confidence + profile.confidence) / 2
                existing.last_updated = datetime.now()
            else:
                # Add new profile
                self.active_profiles[profile_key] = profile
        
        # Remove old profiles (older than 1 hour)
        cutoff_time = datetime.now() - timedelta(hours=1)
        expired_keys = [
            key for key, profile in self.active_profiles.items()
            if profile.last_updated < cutoff_time
        ]
        
        for key in expired_keys:
            del self.active_profiles[key]
    
    async def generate_strategic_responses(self, market_context: MarketContext) -> Dict[TraderArchetype, StrategicResponse]:
        """
        Generate strategic responses to identified market participants
        """
        try:
            logger.info("⚔️ Generating strategic responses...")
            
            responses = {}
            
            for archetype in market_context.active_participants:
                response = await self._create_strategic_response(archetype, market_context)
                if response:
                    responses[archetype] = response
            
            # Store responses for learning
            timestamp = datetime.now().isoformat()
            self.strategic_responses[timestamp] = list(responses.values())
            
            logger.info(f"✅ Generated {len(responses)} strategic responses")
            return responses
            
        except Exception as e:
            logger.error(f"Error generating strategic responses: {e}")
            return {}
    
    async def _create_strategic_response(self, target_archetype: TraderArchetype, 
                                       market_context: MarketContext) -> Optional[StrategicResponse]:
        """Create strategic response for specific trader archetype"""
        
        try:
            if target_archetype not in self.behavioral_patterns:
                return None
            
            archetype_data = self.behavioral_patterns[target_archetype]
            counter_strategies = archetype_data.get('counter_strategies', [])
            
            # Determine response strategy based on market context and archetype
            if target_archetype == TraderArchetype.INSTITUTIONAL_WHALE:
                strategy = "piggyback_momentum"
                actions = ["monitor_large_orders", "follow_direction", "provide_liquidity"]
                success_prob = 0.75
                expected_profit = 0.03
                max_drawdown = 0.02
                
            elif target_archetype == TraderArchetype.ALGORITHMIC_HFT:
                strategy = "avoid_competition"
                actions = ["avoid_spread_scalping", "focus_longer_timeframes", "use_hidden_orders"]
                success_prob = 0.60
                expected_profit = 0.02
                max_drawdown = 0.01
                
            elif target_archetype == TraderArchetype.RETAIL_MOMENTUM:
                strategy = "contrarian_positioning"
                actions = ["fade_momentum", "provide_liquidity", "exploit_overreactions"]
                success_prob = 0.70
                expected_profit = 0.04
                max_drawdown = 0.03
                
            elif target_archetype == TraderArchetype.SMART_MONEY:
                strategy = "follow_smart_money"
                actions = ["copy_positions", "early_trend_detection", "monitor_flows"]
                success_prob = 0.80
                expected_profit = 0.05
                max_drawdown = 0.02
                
            elif target_archetype == TraderArchetype.MANIPULATOR:
                strategy = "avoidance_protocol"
                actions = ["detect_manipulation", "avoid_trading", "report_suspicious_activity"]
                success_prob = 0.90
                expected_profit = 0.0
                max_drawdown = 0.0
                
            else:
                # Default response
                strategy = "adaptive_monitoring"
                actions = ["monitor_patterns", "gradual_positioning", "risk_management"]
                success_prob = 0.65
                expected_profit = 0.025
                max_drawdown = 0.015
            
            # Adjust for market context
            if market_context.volatility_level > 0.05:
                max_drawdown *= 1.5
                success_prob *= 0.9
            
            if market_context.manipulation_risk > 0.3:
                actions.append("enhanced_monitoring")
                success_prob *= 0.8
            
            response = StrategicResponse(
                target_archetype=target_archetype,
                response_strategy=strategy,
                tactical_actions=actions,
                risk_adjustments={
                    'position_size_multiplier': 1.0 - market_context.manipulation_risk * 0.5,
                    'stop_loss_tightening': market_context.volatility_level,
                    'profit_target_adjustment': 1.0 + market_context.opportunity_score * 0.2
                },
                timing_considerations={
                    'entry_delay': market_context.manipulation_risk * 10,  # seconds
                    'exit_urgency': market_context.volatility_level,
                    'monitoring_frequency': 'high' if market_context.manipulation_risk > 0.2 else 'normal'
                },
                success_probability=success_prob,
                expected_profit=expected_profit,
                max_drawdown=max_drawdown
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error creating strategic response for {target_archetype}: {e}")
            return None
    
    async def assess_market_context(self, market_data: Dict[str, Any]) -> MarketContext:
        """
        Assess current market context for strategic decision making
        """
        try:
            # Determine market regime
            regime = await self._determine_market_regime(market_data)
            
            # Calculate key metrics
            volatility = market_data.get('volatility', 0.02)
            liquidity = market_data.get('liquidity', 0.5)
            sentiment = market_data.get('sentiment', 0.5)
            
            # Assess manipulation risk
            manipulation_signals = await self.manipulation_detector.analyze_manipulation(
                pd.DataFrame([market_data]) if isinstance(market_data, dict) else market_data
            )
            manipulation_risk = len(manipulation_signals) / 10  # Normalize
            
            # Calculate opportunity score
            opportunity_score = self._calculate_opportunity_score(market_data)
            
            # Identify active participants
            active_profiles = await self.analyze_market_participants(market_data)
            active_participants = [p.archetype for p in active_profiles if p.confidence > 0.6]
            
            # Determine dominant intent
            dominant_intent = self._determine_dominant_intent(active_participants, market_data)
            
            context = MarketContext(
                regime=regime,
                volatility_level=volatility,
                liquidity_level=liquidity,
                sentiment_score=sentiment,
                manipulation_risk=min(1.0, manipulation_risk),
                opportunity_score=opportunity_score,
                active_participants=active_participants,
                dominant_intent=dominant_intent
            )
            
            # Store in history
            self.market_context_history.append(context)
            if len(self.market_context_history) > 1000:
                self.market_context_history.pop(0)
            
            return context
            
        except Exception as e:
            logger.error(f"Error assessing market context: {e}")
            return MarketContext(
                regime=MarketRegime.SIDEWAYS_RANGING,
                volatility_level=0.02,
                liquidity_level=0.5,
                sentiment_score=0.5,
                manipulation_risk=0.0,
                opportunity_score=0.5,
                active_participants=[],
                dominant_intent=StrategicIntent.SPECULATION
            )
    
    async def _determine_market_regime(self, market_data: Dict[str, Any]) -> MarketRegime:
        """Determine current market regime"""
        
        volatility = market_data.get('volatility', 0.02)
        trend_strength = market_data.get('trend_strength', 0.0)
        sentiment = market_data.get('sentiment', 0.5)
        
        # Crisis detection
        if volatility > 0.08 and sentiment < 0.2:
            return MarketRegime.CRISIS_MODE
        
        # Euphoria detection
        if volatility > 0.06 and sentiment > 0.8:
            return MarketRegime.EUPHORIA_MODE
        
        # Volatility-based classification
        if volatility > 0.05:
            return MarketRegime.HIGH_VOLATILITY
        elif volatility < 0.01:
            return MarketRegime.LOW_VOLATILITY
        
        # Trend-based classification
        if trend_strength > 0.03:
            return MarketRegime.BULL_TRENDING
        elif trend_strength < -0.03:
            return MarketRegime.BEAR_TRENDING
        else:
            return MarketRegime.SIDEWAYS_RANGING
    
    def _calculate_opportunity_score(self, market_data: Dict[str, Any]) -> float:
        """Calculate market opportunity score (0-1)"""
        
        factors = []
        
        # Volatility opportunity (moderate volatility is best)
        volatility = market_data.get('volatility', 0.02)
        vol_score = 1.0 - abs(volatility - 0.03) / 0.03  # Optimal around 3%
        factors.append(max(0, vol_score))
        
        # Liquidity opportunity
        liquidity = market_data.get('liquidity', 0.5)
        factors.append(liquidity)
        
        # Trend clarity
        trend_strength = abs(market_data.get('trend_strength', 0.0))
        factors.append(min(1.0, trend_strength * 10))
        
        # Sentiment extremes (contrarian opportunities)
        sentiment = market_data.get('sentiment', 0.5)
        sentiment_extreme = abs(sentiment - 0.5) * 2  # 0.5 is neutral
        factors.append(sentiment_extreme)
        
        return sum(factors) / len(factors)
    
    def _determine_dominant_intent(self, active_participants: List[TraderArchetype], 
                                 market_data: Dict[str, Any]) -> StrategicIntent:
        """Determine dominant market intent based on participants"""
        
        intent_votes = {intent: 0 for intent in StrategicIntent}
        
        # Map archetypes to likely intents
        archetype_intents = {
            TraderArchetype.INSTITUTIONAL_WHALE: [StrategicIntent.ACCUMULATION, StrategicIntent.DISTRIBUTION],
            TraderArchetype.ALGORITHMIC_HFT: [StrategicIntent.ARBITRAGE],
            TraderArchetype.RETAIL_MOMENTUM: [StrategicIntent.SPECULATION],
            TraderArchetype.SMART_MONEY: [StrategicIntent.ACCUMULATION],
            TraderArchetype.MANIPULATOR: [StrategicIntent.MANIPULATION],
            TraderArchetype.ARBITRAGEUR: [StrategicIntent.ARBITRAGE]
        }
        
        # Vote based on active participants
        for archetype in active_participants:
            if archetype in archetype_intents:
                for intent in archetype_intents[archetype]:
                    intent_votes[intent] += 1
        
        # Return most voted intent, or speculation as default
        if any(intent_votes.values()):
            return max(intent_votes, key=intent_votes.get)
        else:
            return StrategicIntent.SPECULATION
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        return {
            'active_profiles': len(self.active_profiles),
            'archetypes_detected': list(set(p.archetype for p in self.active_profiles.values())),
            'market_contexts_stored': len(self.market_context_history),
            'strategic_responses_generated': len(self.strategic_responses),
            'learning_adaptations': len(self.adaptation_history),
            'success_rates': self.success_rates,
            'last_analysis': max([p.last_updated for p in self.active_profiles.values()]) if self.active_profiles else None,
            'system_health': 'operational'
        }


def create_strategic_profiler_system() -> StrategicProfilerSystem:
    """Create and initialize strategic profiler system"""
    return StrategicProfilerSystem()


# Convenience functions for specific trader types
async def create_whale_hunter_profile() -> TraderProfile:
    """Create specialized profile for hunting institutional whales"""
    return TraderProfile(
        archetype=TraderArchetype.INSTITUTIONAL_WHALE,
        confidence=0.9,
        behavioral_traits={'patience': 0.95, 'discipline': 0.9, 'stealth': 0.85},
        trading_patterns={'iceberg_detection': True, 'flow_analysis': True},
        risk_profile={'position_size': 0.8, 'time_horizon': 0.9, 'risk_tolerance': 0.6},
        time_horizons=['hours', 'days', 'weeks'],
        preferred_instruments=['large_cap', 'etf', 'futures'],
        market_impact=0.9,
        detection_signals=['volume_anomalies', 'stealth_execution'],
        counter_strategies=['momentum_piggyback', 'liquidity_provision']
    )

async def create_hft_detector_profile() -> TraderProfile:
    """Create specialized profile for detecting HFT algorithms"""
    return TraderProfile(
        archetype=TraderArchetype.ALGORITHMIC_HFT,
        confidence=0.95,
        behavioral_traits={'speed': 0.99, 'precision': 0.95, 'predictability': 0.8},
        trading_patterns={'microsecond_timing': True, 'spread_scalping': True},
        risk_profile={'position_size': 0.2, 'time_horizon': 0.1, 'risk_tolerance': 0.3},
        time_horizons=['microseconds', 'milliseconds', 'seconds'],
        preferred_instruments=['liquid_stocks', 'futures', 'forex'],
        market_impact=0.4,
        detection_signals=['sub_second_patterns', 'order_cancellations'],
        counter_strategies=['latency_avoidance', 'hidden_orders']
    )

async def create_retail_sentiment_profile() -> TraderProfile:
    """Create specialized profile for retail sentiment analysis"""
    return TraderProfile(
        archetype=TraderArchetype.RETAIL_MOMENTUM,
        confidence=0.8,
        behavioral_traits={'emotional': 0.95, 'social_driven': 0.9, 'impulsive': 0.85},
        trading_patterns={'news_reactive': True, 'social_following': True, 'fomo_driven': True},
        risk_profile={'position_size': 0.6, 'time_horizon': 0.3, 'risk_tolerance': 0.9},
        time_horizons=['minutes', 'hours', 'days'],
        preferred_instruments=['meme_stocks', 'crypto', 'penny_stocks'],
        market_impact=0.6,
        detection_signals=['social_correlation', 'momentum_chasing'],
        counter_strategies=['contrarian_positioning', 'sentiment_fade']
    ) 