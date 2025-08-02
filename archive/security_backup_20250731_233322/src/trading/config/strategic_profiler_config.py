"""
Strategic Profiler Configuration
==============================

Configuration templates and presets for different strategic profiler scenarios.
Provides behavioral templates, detection thresholds, and response strategies
for various market conditions and trader archetypes.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import json

class ProfilerMode(Enum):
    """Strategic profiler operating modes"""
    DEFENSIVE = "defensive"          # Conservative, risk-averse
    BALANCED = "balanced"           # Standard operation
    AGGRESSIVE = "aggressive"       # High-risk, high-reward
    STEALTH = "stealth"            # Minimal market impact
    WARFARE = "warfare"            # Maximum competitive advantage

class MarketCondition(Enum):
    """Market condition classifications"""
    NORMAL = "normal"
    VOLATILE = "volatile"
    CRISIS = "crisis"
    EUPHORIA = "euphoria"
    ILLIQUID = "illiquid"

@dataclass
class ProfilerConfig:
    """Strategic profiler configuration"""
    mode: ProfilerMode = ProfilerMode.BALANCED
    market_condition: MarketCondition = MarketCondition.NORMAL
    
    # Detection thresholds
    confidence_threshold: float = 0.6
    manipulation_threshold: float = 0.3
    volatility_threshold: float = 0.05
    
    # Response parameters
    max_position_size: float = 0.1
    risk_multiplier: float = 1.0
    response_speed: float = 1.0
    
    # Learning parameters
    adaptation_rate: float = 0.1
    memory_length: int = 1000
    feedback_weight: float = 0.5
    
    # Behavioral templates
    archetype_weights: Dict[str, float] = field(default_factory=dict)
    strategy_preferences: Dict[str, float] = field(default_factory=dict)
    risk_tolerances: Dict[str, float] = field(default_factory=dict)

# Predefined configurations for different scenarios
PROFILER_CONFIGS = {
    
    # DEFENSIVE MODE - Conservative risk management
    "defensive_conservative": ProfilerConfig(
        mode=ProfilerMode.DEFENSIVE,
        market_condition=MarketCondition.NORMAL,
        confidence_threshold=0.8,
        manipulation_threshold=0.2,
        volatility_threshold=0.03,
        max_position_size=0.05,
        risk_multiplier=0.5,
        response_speed=0.7,
        archetype_weights={
            "institutional_whale": 0.9,
            "smart_money": 0.8,
            "manipulator": 0.1,
            "retail_momentum": 0.3
        },
        strategy_preferences={
            "follow_smart_money": 0.9,
            "momentum_piggyback": 0.7,
            "contrarian_positioning": 0.3,
            "avoidance_protocol": 0.9
        },
        risk_tolerances={
            "max_drawdown": 0.02,
            "position_concentration": 0.1,
            "leverage_limit": 1.0
        }
    ),
    
    # BALANCED MODE - Standard operation
    "balanced_standard": ProfilerConfig(
        mode=ProfilerMode.BALANCED,
        market_condition=MarketCondition.NORMAL,
        confidence_threshold=0.6,
        manipulation_threshold=0.3,
        volatility_threshold=0.05,
        max_position_size=0.1,
        risk_multiplier=1.0,
        response_speed=1.0,
        archetype_weights={
            "institutional_whale": 0.8,
            "smart_money": 0.8,
            "algorithmic_hft": 0.6,
            "retail_momentum": 0.5,
            "manipulator": 0.2
        },
        strategy_preferences={
            "follow_smart_money": 0.7,
            "momentum_piggyback": 0.6,
            "contrarian_positioning": 0.5,
            "latency_avoidance": 0.6,
            "sentiment_exploitation": 0.5
        },
        risk_tolerances={
            "max_drawdown": 0.05,
            "position_concentration": 0.2,
            "leverage_limit": 2.0
        }
    ),
    
    # AGGRESSIVE MODE - High-risk, high-reward
    "aggressive_warfare": ProfilerConfig(
        mode=ProfilerMode.AGGRESSIVE,
        market_condition=MarketCondition.VOLATILE,
        confidence_threshold=0.5,
        manipulation_threshold=0.4,
        volatility_threshold=0.08,
        max_position_size=0.2,
        risk_multiplier=2.0,
        response_speed=1.5,
        archetype_weights={
            "institutional_whale": 0.9,
            "smart_money": 0.9,
            "retail_momentum": 0.8,
            "manipulator": 0.3,
            "algorithmic_hft": 0.7
        },
        strategy_preferences={
            "momentum_piggyback": 0.9,
            "contrarian_positioning": 0.8,
            "sentiment_exploitation": 0.9,
            "volatility_trading": 0.8,
            "arbitrage_hunting": 0.7
        },
        risk_tolerances={
            "max_drawdown": 0.1,
            "position_concentration": 0.3,
            "leverage_limit": 3.0
        }
    ),
    
    # STEALTH MODE - Minimal market impact
    "stealth_iceberg": ProfilerConfig(
        mode=ProfilerMode.STEALTH,
        market_condition=MarketCondition.ILLIQUID,
        confidence_threshold=0.7,
        manipulation_threshold=0.25,
        volatility_threshold=0.04,
        max_position_size=0.05,
        risk_multiplier=0.8,
        response_speed=0.5,
        archetype_weights={
            "institutional_whale": 0.95,
            "smart_money": 0.9,
            "algorithmic_hft": 0.4,
            "retail_momentum": 0.2,
            "manipulator": 0.1
        },
        strategy_preferences={
            "stealth_execution": 0.95,
            "iceberg_following": 0.9,
            "gradual_accumulation": 0.8,
            "hidden_orders": 0.9,
            "time_weighted_entry": 0.8
        },
        risk_tolerances={
            "max_drawdown": 0.03,
            "position_concentration": 0.1,
            "leverage_limit": 1.5
        }
    ),
    
    # CRISIS MODE - Crisis market conditions
    "crisis_survival": ProfilerConfig(
        mode=ProfilerMode.DEFENSIVE,
        market_condition=MarketCondition.CRISIS,
        confidence_threshold=0.9,
        manipulation_threshold=0.15,
        volatility_threshold=0.15,
        max_position_size=0.03,
        risk_multiplier=0.3,
        response_speed=0.8,
        archetype_weights={
            "institutional_whale": 0.95,
            "smart_money": 0.9,
            "manipulator": 0.05,
            "retail_momentum": 0.1,
            "panic_seller": 0.8
        },
        strategy_preferences={
            "capital_preservation": 0.95,
            "contrarian_positioning": 0.8,
            "safe_haven_assets": 0.9,
            "liquidity_provision": 0.6,
            "volatility_hedging": 0.8
        },
        risk_tolerances={
            "max_drawdown": 0.015,
            "position_concentration": 0.05,
            "leverage_limit": 1.0
        }
    ),
    
    # EUPHORIA MODE - Bubble market conditions
    "euphoria_contrarian": ProfilerConfig(
        mode=ProfilerMode.AGGRESSIVE,
        market_condition=MarketCondition.EUPHORIA,
        confidence_threshold=0.6,
        manipulation_threshold=0.4,
        volatility_threshold=0.1,
        max_position_size=0.15,
        risk_multiplier=1.5,
        response_speed=1.2,
        archetype_weights={
            "retail_momentum": 0.9,
            "smart_money": 0.8,
            "manipulator": 0.4,
            "institutional_whale": 0.6,
            "diamond_hands": 0.3
        },
        strategy_preferences={
            "contrarian_positioning": 0.9,
            "momentum_fade": 0.8,
            "sentiment_exploitation": 0.9,
            "bubble_detection": 0.8,
            "profit_taking": 0.9
        },
        risk_tolerances={
            "max_drawdown": 0.08,
            "position_concentration": 0.25,
            "leverage_limit": 2.5
        }
    )
}

# Behavioral templates for different trader archetypes
ARCHETYPE_TEMPLATES = {
    
    "institutional_whale": {
        "behavioral_traits": {
            "patience": 0.9,
            "discipline": 0.9,
            "stealth_execution": 0.8,
            "information_advantage": 0.8,
            "risk_management": 0.9
        },
        "trading_patterns": {
            "iceberg_orders": True,
            "time_weighted_execution": True,
            "minimal_market_impact": True,
            "gradual_accumulation": True,
            "off_hours_trading": True
        },
        "detection_signals": [
            "large_volume_low_impact",
            "gradual_price_movement",
            "consistent_direction",
            "iceberg_patterns",
            "stealth_execution_indicators"
        ],
        "counter_strategies": [
            "momentum_piggyback",
            "liquidity_provision",
            "trend_following",
            "position_mirroring"
        ]
    },
    
    "algorithmic_hft": {
        "behavioral_traits": {
            "speed": 0.95,
            "precision": 0.9,
            "emotionless": 0.95,
            "pattern_recognition": 0.9,
            "latency_sensitivity": 0.95
        },
        "trading_patterns": {
            "high_frequency": True,
            "spread_scalping": True,
            "latency_arbitrage": True,
            "order_flow_prediction": True,
            "microsecond_timing": True
        },
        "detection_signals": [
            "sub_second_patterns",
            "spread_scalping_activity",
            "order_cancellation_patterns",
            "latency_arbitrage_signals",
            "volume_clustering"
        ],
        "counter_strategies": [
            "latency_avoidance",
            "hidden_orders",
            "time_randomization",
            "adverse_selection_avoidance"
        ]
    },
    
    "retail_momentum": {
        "behavioral_traits": {
            "emotional": 0.9,
            "impulsive": 0.8,
            "social_influenced": 0.9,
            "trend_following": 0.8,
            "fomo_driven": 0.9
        },
        "trading_patterns": {
            "momentum_chasing": True,
            "news_reactive": True,
            "social_sentiment_following": True,
            "weekend_effect": True,
            "meme_stock_preference": True
        },
        "detection_signals": [
            "social_sentiment_correlation",
            "news_reaction_patterns",
            "momentum_chasing_behavior",
            "retail_volume_spikes",
            "meme_stock_activity"
        ],
        "counter_strategies": [
            "contrarian_positioning",
            "momentum_fade",
            "sentiment_exploitation",
            "retail_flow_analysis"
        ]
    },
    
    "smart_money": {
        "behavioral_traits": {
            "insight": 0.9,
            "contrarian": 0.7,
            "patient": 0.8,
            "information_edge": 0.9,
            "strategic_thinking": 0.9
        },
        "trading_patterns": {
            "early_positioning": True,
            "contrarian_moves": True,
            "information_advantage": True,
            "sector_rotation": True,
            "value_investing": True
        },
        "detection_signals": [
            "early_trend_entry",
            "contrarian_positioning",
            "sector_rotation_patterns",
            "value_accumulation",
            "information_edge_indicators"
        ],
        "counter_strategies": [
            "follow_smart_money",
            "early_trend_detection",
            "insider_flow_tracking",
            "value_momentum_combination"
        ]
    },
    
    "manipulator": {
        "behavioral_traits": {
            "deceptive": 0.95,
            "opportunistic": 0.9,
            "risk_taking": 0.8,
            "timing_focused": 0.9,
            "illiquid_targeting": 0.8
        },
        "trading_patterns": {
            "spoofing": True,
            "wash_trading": True,
            "layering": True,
            "pump_dump": True,
            "low_liquidity_targeting": True
        },
        "detection_signals": [
            "spoofing_patterns",
            "wash_trading_indicators",
            "artificial_volume",
            "pump_dump_signatures",
            "layering_activity"
        ],
        "counter_strategies": [
            "manipulation_detection",
            "avoidance_protocols",
            "regulatory_reporting",
            "liquidity_monitoring"
        ]
    }
}

# Strategy response templates
STRATEGY_TEMPLATES = {
    
    "momentum_piggyback": {
        "description": "Follow institutional momentum with reduced risk",
        "entry_conditions": [
            "institutional_whale_detected",
            "consistent_direction_confirmed",
            "volume_confirmation"
        ],
        "exit_conditions": [
            "momentum_weakening",
            "volume_declining",
            "profit_target_reached"
        ],
        "risk_parameters": {
            "max_position_size": 0.1,
            "stop_loss": 0.02,
            "profit_target": 0.04,
            "time_limit": 3600  # 1 hour
        }
    },
    
    "contrarian_positioning": {
        "description": "Take contrarian positions against retail sentiment",
        "entry_conditions": [
            "extreme_sentiment_detected",
            "retail_momentum_exhaustion",
            "technical_divergence"
        ],
        "exit_conditions": [
            "sentiment_normalization",
            "technical_confirmation",
            "profit_target_reached"
        ],
        "risk_parameters": {
            "max_position_size": 0.08,
            "stop_loss": 0.03,
            "profit_target": 0.06,
            "time_limit": 7200  # 2 hours
        }
    },
    
    "latency_avoidance": {
        "description": "Avoid competing with HFT algorithms",
        "entry_conditions": [
            "hft_activity_low",
            "longer_timeframe_signal",
            "hidden_order_availability"
        ],
        "exit_conditions": [
            "hft_activity_increasing",
            "spread_tightening",
            "profit_target_reached"
        ],
        "risk_parameters": {
            "max_position_size": 0.05,
            "stop_loss": 0.015,
            "profit_target": 0.025,
            "time_limit": 1800  # 30 minutes
        }
    },
    
    "manipulation_avoidance": {
        "description": "Avoid trading during detected manipulation",
        "entry_conditions": [
            "manipulation_risk_low",
            "normal_market_conditions",
            "legitimate_volume_patterns"
        ],
        "exit_conditions": [
            "manipulation_detected",
            "suspicious_activity",
            "regulatory_concerns"
        ],
        "risk_parameters": {
            "max_position_size": 0.03,
            "stop_loss": 0.01,
            "profit_target": 0.02,
            "time_limit": 900  # 15 minutes
        }
    }
}

def get_profiler_config(config_name: str) -> Optional[ProfilerConfig]:
    """Get a predefined profiler configuration"""
    return PROFILER_CONFIGS.get(config_name)

def get_archetype_template(archetype_name: str) -> Optional[Dict[str, Any]]:
    """Get behavioral template for a trader archetype"""
    return ARCHETYPE_TEMPLATES.get(archetype_name)

def get_strategy_template(strategy_name: str) -> Optional[Dict[str, Any]]:
    """Get strategy response template"""
    return STRATEGY_TEMPLATES.get(strategy_name)

def create_custom_config(
    mode: ProfilerMode = ProfilerMode.BALANCED,
    market_condition: MarketCondition = MarketCondition.NORMAL,
    **kwargs
) -> ProfilerConfig:
    """Create a custom profiler configuration"""
    
    config = ProfilerConfig(mode=mode, market_condition=market_condition)
    
    # Update with custom parameters
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config

def save_config_to_file(config: ProfilerConfig, filename: str):
    """Save configuration to JSON file"""
    
    config_dict = {
        'mode': config.mode.value,
        'market_condition': config.market_condition.value,
        'confidence_threshold': config.confidence_threshold,
        'manipulation_threshold': config.manipulation_threshold,
        'volatility_threshold': config.volatility_threshold,
        'max_position_size': config.max_position_size,
        'risk_multiplier': config.risk_multiplier,
        'response_speed': config.response_speed,
        'adaptation_rate': config.adaptation_rate,
        'memory_length': config.memory_length,
        'feedback_weight': config.feedback_weight,
        'archetype_weights': config.archetype_weights,
        'strategy_preferences': config.strategy_preferences,
        'risk_tolerances': config.risk_tolerances
    }
    
    with open(filename, 'w') as f:
        json.dump(config_dict, f, indent=2)

def load_config_from_file(filename: str) -> ProfilerConfig:
    """Load configuration from JSON file"""
    
    with open(filename, 'r') as f:
        config_dict = json.load(f)
    
    return ProfilerConfig(
        mode=ProfilerMode(config_dict['mode']),
        market_condition=MarketCondition(config_dict['market_condition']),
        confidence_threshold=config_dict['confidence_threshold'],
        manipulation_threshold=config_dict['manipulation_threshold'],
        volatility_threshold=config_dict['volatility_threshold'],
        max_position_size=config_dict['max_position_size'],
        risk_multiplier=config_dict['risk_multiplier'],
        response_speed=config_dict['response_speed'],
        adaptation_rate=config_dict['adaptation_rate'],
        memory_length=config_dict['memory_length'],
        feedback_weight=config_dict['feedback_weight'],
        archetype_weights=config_dict['archetype_weights'],
        strategy_preferences=config_dict['strategy_preferences'],
        risk_tolerances=config_dict['risk_tolerances']
    )

# Export all available configurations
AVAILABLE_CONFIGS = list(PROFILER_CONFIGS.keys())
AVAILABLE_ARCHETYPES = list(ARCHETYPE_TEMPLATES.keys())
AVAILABLE_STRATEGIES = list(STRATEGY_TEMPLATES.keys()) 