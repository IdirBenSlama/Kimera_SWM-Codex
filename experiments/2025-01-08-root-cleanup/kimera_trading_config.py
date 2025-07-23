"""
KIMERA SEMANTIC WEALTH MANAGEMENT - TRADING CONFIGURATION
=========================================================

Comprehensive configuration management for the Kimera Trading System.
Supports multiple environments, exchange configurations, and advanced parameters.

Usage:
    from kimera_trading_config import get_trading_config
    config = get_trading_config('production')  # or 'development', 'testing'
"""

import os
from typing import Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Environment(Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"

class ExchangeType(Enum):
    BINANCE = "binance"
    PHEMEX = "phemex"
    COINBASE = "coinbase"
    COINBASE_PRO = "coinbase_pro"

@dataclass
class APIKeyConfig:
    """API key configuration for exchanges"""
    api_key: str
    api_secret: str
    passphrase: str = ""  # For Coinbase Pro
    testnet: bool = True

@dataclass
class ExchangeConfig:
    """Exchange-specific configuration"""
    name: ExchangeType
    api_config: APIKeyConfig
    enabled: bool = True
    rate_limit: int = 1200  # requests per minute
    order_types: List[str] = field(default_factory=lambda: ['market', 'limit'])
    min_order_size: float = 0.001
    max_order_size: float = 1000000.0

@dataclass
class RiskManagementConfig:
    """Risk management parameters"""
    max_position_size: float = 0.25  # 25% of portfolio
    max_total_risk: float = 0.10     # 10% total portfolio risk
    default_stop_loss: float = 0.02  # 2% stop loss
    max_daily_trades: int = 50
    max_concurrent_positions: int = 10
    enable_stop_losses: bool = True
    enable_position_sizing: bool = True
    enable_risk_limits: bool = True
    drawdown_limit: float = 0.20     # 20% max drawdown
    daily_loss_limit: float = 0.05   # 5% daily loss limit

@dataclass
class SemanticConfig:
    """Semantic analysis configuration"""
    contradiction_threshold: float = 0.4
    thermodynamic_sensitivity: float = 0.6
    semantic_confidence_threshold: float = 0.7
    enable_contradiction_detection: bool = True
    enable_thermodynamic_analysis: bool = True
    contradiction_history_limit: int = 1000
    semantic_cache_ttl: int = 300  # seconds

@dataclass
class IntelligenceConfig:
    """Market intelligence configuration"""
    enable_sentiment_analysis: bool = True
    enable_news_processing: bool = True
    enable_technical_analysis: bool = True
    enable_anomaly_detection: bool = True
    sentiment_sources: List[str] = field(default_factory=lambda: ['twitter', 'reddit', 'news'])
    news_sources: List[str] = field(default_factory=lambda: ['cryptopanic', 'coindesk', 'reuters'])
    update_intervals: Dict[str, int] = field(default_factory=lambda: {
        'sentiment': 300,    # 5 minutes
        'news': 600,        # 10 minutes
        'technical': 60,    # 1 minute
        'anomaly': 900      # 15 minutes
    })

@dataclass
class TradingConfig:
    """Main trading configuration"""
    # Environment
    environment: Environment = Environment.DEVELOPMENT
    
    # Capital management
    starting_capital: float = 1000.0
    currency: str = "USD"
    
    # Exchange configuration
    primary_exchange: ExchangeType = ExchangeType.BINANCE
    backup_exchanges: List[ExchangeType] = field(default_factory=lambda: [ExchangeType.PHEMEX])
    exchanges: Dict[ExchangeType, ExchangeConfig] = field(default_factory=dict)
    
    # Risk management
    risk_management: RiskManagementConfig = field(default_factory=RiskManagementConfig)
    
    # Semantic analysis
    semantic: SemanticConfig = field(default_factory=SemanticConfig)
    
    # Market intelligence
    intelligence: IntelligenceConfig = field(default_factory=IntelligenceConfig)
    
    # Trading behavior
    enable_paper_trading: bool = True
    enable_backtesting: bool = True
    enable_real_time_monitoring: bool = True
    enable_automated_trading: bool = False
    
    # Performance settings
    market_data_interval: int = 1      # seconds
    signal_generation_interval: int = 5
    position_management_interval: int = 10
    risk_check_interval: int = 30
    
    # Symbols to trade
    trading_symbols: List[str] = field(default_factory=lambda: [
        'BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'SOLUSDT', 'XRPUSDT'
    ])
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/kimera_trading.log"
    enable_detailed_logging: bool = True

# ===================== CONFIGURATION PRESETS =====================

def get_development_config() -> TradingConfig:
    """Development environment configuration"""
    config = TradingConfig(
        environment=Environment.DEVELOPMENT,
        starting_capital=1000.0,
        enable_paper_trading=True,
        enable_automated_trading=False,
        risk_management=RiskManagementConfig(
            max_position_size=0.10,  # Conservative for development
            max_daily_trades=10,
            enable_risk_limits=True
        ),
        semantic=SemanticConfig(
            contradiction_threshold=0.3,  # More sensitive
            semantic_confidence_threshold=0.6
        ),
        trading_symbols=['BTCUSDT', 'ETHUSDT']  # Limited symbols for testing
    )
    
    # Add development exchange configurations
    config.exchanges = {
        ExchangeType.BINANCE: ExchangeConfig(
            name=ExchangeType.BINANCE,
            api_config=APIKeyConfig(
                api_key=os.getenv('BINANCE_TESTNET_API_KEY', ''),
                api_secret=os.getenv('BINANCE_TESTNET_API_SECRET', ''),
                testnet=True
            )
        )
    }
    
    return config

def get_testing_config() -> TradingConfig:
    """Testing environment configuration"""
    config = TradingConfig(
        environment=Environment.TESTING,
        starting_capital=5000.0,
        enable_paper_trading=True,
        enable_automated_trading=True,
        risk_management=RiskManagementConfig(
            max_position_size=0.15,
            max_daily_trades=25,
            enable_risk_limits=True
        ),
        semantic=SemanticConfig(
            contradiction_threshold=0.4,
            semantic_confidence_threshold=0.7
        ),
        trading_symbols=['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'SOLUSDT']
    )
    
    # Add testing exchange configurations
    config.exchanges = {
        ExchangeType.BINANCE: ExchangeConfig(
            name=ExchangeType.BINANCE,
            api_config=APIKeyConfig(
                api_key=os.getenv('BINANCE_TESTNET_API_KEY', ''),
                api_secret=os.getenv('BINANCE_TESTNET_API_SECRET', ''),
                testnet=True
            )
        ),
        ExchangeType.PHEMEX: ExchangeConfig(
            name=ExchangeType.PHEMEX,
            api_config=APIKeyConfig(
                api_key=os.getenv('PHEMEX_TESTNET_API_KEY', ''),
                api_secret=os.getenv('PHEMEX_TESTNET_API_SECRET', ''),
                testnet=True
            )
        )
    }
    
    return config

def get_production_config() -> TradingConfig:
    """Production environment configuration"""
    config = TradingConfig(
        environment=Environment.PRODUCTION,
        starting_capital=float(os.getenv('STARTING_CAPITAL', '10000.0')),
        enable_paper_trading=False,  # Real trading
        enable_automated_trading=True,
        risk_management=RiskManagementConfig(
            max_position_size=0.20,
            max_daily_trades=50,
            enable_risk_limits=True,
            drawdown_limit=0.15,  # Stricter in production
            daily_loss_limit=0.03
        ),
        semantic=SemanticConfig(
            contradiction_threshold=0.5,  # Higher threshold for production
            semantic_confidence_threshold=0.75
        ),
        intelligence=IntelligenceConfig(
            enable_sentiment_analysis=True,
            enable_news_processing=True,
            enable_technical_analysis=True,
            enable_anomaly_detection=True
        )
    )
    
    # Add production exchange configurations
    config.exchanges = {
        ExchangeType.BINANCE: ExchangeConfig(
            name=ExchangeType.BINANCE,
            api_config=APIKeyConfig(
                api_key=os.getenv('BINANCE_API_KEY', ''),
                api_secret=os.getenv('BINANCE_API_SECRET', ''),
                testnet=False
            )
        ),
        ExchangeType.PHEMEX: ExchangeConfig(
            name=ExchangeType.PHEMEX,
            api_config=APIKeyConfig(
                api_key=os.getenv('PHEMEX_API_KEY', ''),
                api_secret=os.getenv('PHEMEX_API_SECRET', ''),
                testnet=False
            )
        )
    }
    
    return config

def get_aggressive_config() -> TradingConfig:
    """Aggressive trading configuration for maximum performance"""
    config = get_production_config()
    
    # Modify for aggressive trading
    config.risk_management.max_position_size = 0.35  # Larger positions
    config.risk_management.max_daily_trades = 100    # More trades
    config.risk_management.default_stop_loss = 0.015 # Tighter stops
    config.semantic.contradiction_threshold = 0.3    # More sensitive
    config.semantic.semantic_confidence_threshold = 0.6  # Lower threshold
    
    # Faster intervals
    config.market_data_interval = 1
    config.signal_generation_interval = 3
    config.position_management_interval = 5
    
    return config

def get_conservative_config() -> TradingConfig:
    """Conservative trading configuration for capital preservation"""
    config = get_production_config()
    
    # Modify for conservative trading
    config.risk_management.max_position_size = 0.10  # Smaller positions
    config.risk_management.max_daily_trades = 20     # Fewer trades
    config.risk_management.default_stop_loss = 0.03  # Wider stops
    config.semantic.contradiction_threshold = 0.7    # Less sensitive
    config.semantic.semantic_confidence_threshold = 0.85  # Higher threshold
    config.risk_management.drawdown_limit = 0.10     # Stricter drawdown
    config.risk_management.daily_loss_limit = 0.02   # Stricter daily loss
    
    # Slower intervals
    config.signal_generation_interval = 10
    config.position_management_interval = 30
    
    return config

# ===================== CONFIGURATION FACTORY =====================

def get_trading_config(environment: str = "development", 
                      custom_overrides: Dict[str, Any] = None) -> TradingConfig:
    """
    Get trading configuration for specified environment
    
    Args:
        environment: 'development', 'testing', 'production', 'aggressive', 'conservative'
        custom_overrides: Dictionary of custom configuration overrides
        
    Returns:
        TradingConfig object
    """
    config_map = {
        'development': get_development_config,
        'testing': get_testing_config,
        'production': get_production_config,
        'aggressive': get_aggressive_config,
        'conservative': get_conservative_config
    }
    
    if environment not in config_map:
        raise ValueError(f"Unknown environment: {environment}. Available: {list(config_map.keys())}")
    
    config = config_map[environment]()
    
    # Apply custom overrides
    if custom_overrides:
        for key, value in custom_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                print(f"Warning: Unknown configuration key: {key}")
    
    return config

def validate_config(config: TradingConfig) -> List[str]:
    """
    Validate trading configuration and return list of issues
    
    Args:
        config: TradingConfig to validate
        
    Returns:
        List of validation error messages
    """
    issues = []
    
    # Validate capital
    if config.starting_capital <= 0:
        issues.append("Starting capital must be positive")
    
    # Validate risk management
    if config.risk_management.max_position_size > 1.0:
        issues.append("Max position size cannot exceed 100%")
    
    if config.risk_management.max_total_risk > 0.5:
        issues.append("Max total risk should not exceed 50%")
    
    if config.risk_management.default_stop_loss > 0.1:
        issues.append("Default stop loss seems too large (>10%)")
    
    # Validate semantic parameters
    if not 0 <= config.semantic.contradiction_threshold <= 1:
        issues.append("Contradiction threshold must be between 0 and 1")
    
    if not 0 <= config.semantic.semantic_confidence_threshold <= 1:
        issues.append("Semantic confidence threshold must be between 0 and 1")
    
    # Validate exchange configurations
    if not config.exchanges:
        issues.append("At least one exchange must be configured")
    
    for exchange_type, exchange_config in config.exchanges.items():
        if not exchange_config.api_config.api_key:
            issues.append(f"API key missing for {exchange_type.value}")
        if not exchange_config.api_config.api_secret:
            issues.append(f"API secret missing for {exchange_type.value}")
    
    # Validate trading symbols
    if not config.trading_symbols:
        issues.append("At least one trading symbol must be specified")
    
    return issues

# ===================== ENVIRONMENT VARIABLE TEMPLATE =====================

def generate_env_template() -> str:
    """Generate .env template file content"""
    template = """
# KIMERA TRADING SYSTEM ENVIRONMENT VARIABLES
# ============================================

# General Configuration
STARTING_CAPITAL=10000.0
ENVIRONMENT=development

# Binance Configuration
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_API_SECRET=your_binance_api_secret_here
BINANCE_TESTNET_API_KEY=your_binance_testnet_api_key_here
BINANCE_TESTNET_API_SECRET=your_binance_testnet_api_secret_here

# Phemex Configuration
PHEMEX_API_KEY=your_phemex_api_key_here
PHEMEX_API_SECRET=your_phemex_api_secret_here
PHEMEX_TESTNET_API_KEY=your_phemex_testnet_api_key_here
PHEMEX_TESTNET_API_SECRET=your_phemex_testnet_api_secret_here

# Coinbase Configuration
COINBASE_API_KEY=your_coinbase_api_key_here
COINBASE_API_SECRET=your_coinbase_api_secret_here
COINBASE_PASSPHRASE=your_coinbase_passphrase_here

# Market Intelligence APIs
NEWS_API_KEY=your_news_api_key_here
REDDIT_CLIENT_ID=your_reddit_client_id_here
REDDIT_CLIENT_SECRET=your_reddit_client_secret_here
TWITTER_BEARER_TOKEN=your_twitter_bearer_token_here
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key_here
CRYPTOPANIC_API_KEY=your_cryptopanic_api_key_here
TAAPI_API_KEY=your_taapi_api_key_here

# Database Configuration
QUESTDB_HOST=localhost
QUESTDB_PORT=9009
KAFKA_SERVERS=localhost:9092

# Monitoring Configuration
DASHBOARD_PORT=8050
PROMETHEUS_PORT=9090

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=logs/kimera_trading.log
"""
    return template.strip()

# ===================== EXAMPLE USAGE =====================

if __name__ == "__main__":
    # Example usage
    print("Kimera Trading System Configuration Examples")
    print("=" * 50)
    
    # Development configuration
    dev_config = get_trading_config('development')
    print(f"Development Config - Starting Capital: ${dev_config.starting_capital}")
    print(f"Paper Trading: {dev_config.enable_paper_trading}")
    print(f"Max Position Size: {dev_config.risk_management.max_position_size * 100}%")
    print()
    
    # Production configuration
    prod_config = get_trading_config('production')
    print(f"Production Config - Starting Capital: ${prod_config.starting_capital}")
    print(f"Paper Trading: {prod_config.enable_paper_trading}")
    print(f"Max Position Size: {prod_config.risk_management.max_position_size * 100}%")
    print()
    
    # Custom configuration
    custom_config = get_trading_config('production', {
        'starting_capital': 50000.0,
        'trading_symbols': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
    })
    print(f"Custom Config - Starting Capital: ${custom_config.starting_capital}")
    print(f"Trading Symbols: {custom_config.trading_symbols}")
    print()
    
    # Validate configuration
    issues = validate_config(dev_config)
    if issues:
        print("Configuration Issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✅ Configuration is valid")
    
    # Generate .env template
    print("\n" + "=" * 50)
    print("Environment Variables Template:")
    print("=" * 50)
    print(generate_env_template()) 