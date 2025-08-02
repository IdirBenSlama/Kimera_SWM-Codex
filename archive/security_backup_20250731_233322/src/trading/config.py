"""
Trading Intelligence Configuration

Store API keys in .env file:
NEWS_API_KEY=your_key
REDDIT_CLIENT_ID=your_id
REDDIT_CLIENT_SECRET=your_secret
TWITTER_BEARER_TOKEN=your_token
ALPHA_VANTAGE_API_KEY=your_key
FRED_API_KEY=your_key
OPENWEATHER_API_KEY=your_key
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys (loaded from .env)
NEWS_API_KEY = os.getenv('NEWS_API_KEY', '')
REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID', '')
REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET', '')
REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT', 'KimeraAI/1.0')
TWITTER_BEARER_TOKEN = os.getenv('TWITTER_BEARER_TOKEN', '')
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', '')
FRED_API_KEY = os.getenv('FRED_API_KEY', '')
OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY', '')
FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY', '')
FINNHUB_WEBHOOK_SECRET = os.getenv('FINNHUB_WEBHOOK_SECRET', '')
TWELVE_DATA_API_KEY = os.getenv('TWELVE_DATA_API_KEY', '')

# Intelligence Weights (how much each factor influences decisions)
INTELLIGENCE_WEIGHTS = {
    'sentiment': 0.25,      # Social media & news sentiment
    'technical': 0.20,      # Traditional technical indicators
    'blockchain': 0.15,     # On-chain metrics
    'macro': 0.15,          # Macro economic factors
    'geopolitical': 0.10,   # Global events
    'weather': 0.05,        # Environmental factors
    'cognitive': 0.10       # Kimera's unique insights
}

# Sentiment Thresholds
SENTIMENT_THRESHOLDS = {
    'extreme_fear': -0.8,
    'fear': -0.5,
    'neutral': 0.0,
    'greed': 0.5,
    'extreme_greed': 0.8
}

# News Sources Priority
NEWS_SOURCES = {
    'coindesk': {'weight': 1.0, 'crypto_focus': True},
    'cointelegraph': {'weight': 0.9, 'crypto_focus': True},
    'reuters': {'weight': 0.8, 'crypto_focus': False},
    'bloomberg': {'weight': 0.9, 'crypto_focus': False},
    'wsj': {'weight': 0.8, 'crypto_focus': False}
}

# Reddit Subreddits to Monitor
REDDIT_SUBREDDITS = [
    'cryptocurrency',
    'bitcoin',
    'ethereum',
    'cryptomarkets',
    'wallstreetbets',  # For general market sentiment
    'investing',
    'stocks'
]

# Twitter Crypto Influencers (X handles)
TWITTER_INFLUENCERS = [
    'APompliano',
    'novogratz',
    'VitalikButerin',
    'cz_binance',
    'elonmusk',  # His tweets move markets
    'michael_saylor',
    'RaoulGMI'
]

# Blockchain Metrics to Track
BLOCKCHAIN_METRICS = [
    'hash_rate',
    'difficulty',
    'active_addresses',
    'transaction_volume',
    'exchange_inflows',
    'exchange_outflows',
    'miner_revenue',
    'network_fees'
]

# Macro Indicators
MACRO_INDICATORS = [
    'DGS10',     # 10-Year Treasury Rate
    'DFF',       # Federal Funds Rate
    'UNRATE',    # Unemployment Rate
    'CPIAUCSL',  # Consumer Price Index
    'DXY',       # Dollar Index
    'GOLD',      # Gold Price
    'OIL'        # Oil Price
]

# Geopolitical Event Keywords
GEOPOLITICAL_KEYWORDS = [
    'war', 'conflict', 'sanctions', 'regulation',
    'ban', 'adoption', 'legal tender', 'etf',
    'sec', 'federal reserve', 'ecb', 'china',
    'russia', 'ukraine', 'middle east'
]

# Weather Impact Regions (for mining)
MINING_REGIONS = {
    'china': {'lat': 35.86, 'lon': 104.19},
    'usa': {'lat': 39.83, 'lon': -98.58},
    'kazakhstan': {'lat': 48.01, 'lon': 66.92},
    'russia': {'lat': 61.52, 'lon': 105.31},
    'canada': {'lat': 56.13, 'lon': -106.34}
}

# Intelligence Update Intervals (seconds)
UPDATE_INTERVALS = {
    'sentiment': 300,      # 5 minutes
    'news': 600,          # 10 minutes
    'blockchain': 900,    # 15 minutes
    'macro': 3600,        # 1 hour
    'weather': 3600,      # 1 hour
    'geopolitical': 1800  # 30 minutes
}

# ==================== PREMIUM DATA PROVIDERS ====================

# Alpha Vantage Configuration
ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"
ALPHA_VANTAGE_RATE_LIMIT = 5  # calls per minute

# Finnhub Configuration
FINNHUB_BASE_URL = "https://finnhub.io/api/v1"
FINNHUB_RATE_LIMIT = 60  # calls per minute

# Twelve Data Configuration
TWELVE_DATA_BASE_URL = "https://api.twelvedata.com"
TWELVE_DATA_RATE_LIMIT = 8  # calls per minute

# Premium Trading Configuration
PREMIUM_MODE_ENABLED = True
PREMIUM_RISK_MULTIPLIER = 1.5  # Higher risk for premium signals
PREMIUM_CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence for premium trades
PREMIUM_DATA_SOURCES = [
    "alpha_vantage",
    "finnhub", 
    "twelve_data",
    "cryptopanic"
]

# Enhanced signal weights for premium data
SIGNAL_WEIGHTS = {
    "premium_sentiment": 3.0,
    "premium_contradiction": 2.5,
    "technical_rsi": 2.0,
    "enhanced_momentum": 1.8,
    "earnings_event": 1.5,
    "economic_indicator": 1.3,
    "standard_momentum": 1.0
}

# ==================== WEBHOOK CONFIGURATION ====================

# Finnhub Webhook Configuration
WEBHOOK_ENABLED = True
WEBHOOK_SECRET = FINNHUB_WEBHOOK_SECRET
WEBHOOK_ENDPOINTS = {
    "finnhub": {
        "url": "http://localhost:8080/webhook/finnhub",
        "secret": FINNHUB_WEBHOOK_SECRET,
        "events": ["earnings", "news", "splits", "dividends"]
    }
}

# ==================== ENHANCED TRADING PARAMETERS ====================

# Premium position sizing
PREMIUM_POSITION_SIZING = {
    "base_risk_per_trade": 0.20,  # 20% base risk
    "max_position_size": 0.60,    # Max 60% per position
    "premium_multiplier": 1.5,    # 1.5x for premium signals
    "volatility_position_size": 0.10,  # 10% for volatility trades
    "earnings_position_size": 0.15     # 15% for earnings plays
}

# Premium risk management
PREMIUM_RISK_MANAGEMENT = {
    "premium_signals": {
        "profit_target": 0.08,  # 8% profit target
        "stop_loss": 0.04,      # 4% stop loss
        "time_exit": 3          # 3 minutes max hold
    },
    "technical_signals": {
        "profit_target": 0.12,  # 12% profit target
        "stop_loss": 0.06,      # 6% stop loss
        "time_exit": 5          # 5 minutes max hold
    },
    "standard_signals": {
        "profit_target": 0.15,  # 15% profit target
        "stop_loss": 0.08,      # 8% stop loss
        "time_exit": 10         # 10 minutes max hold
    }
}

# ==================== ENHANCED WATCHLISTS ====================

# Premium watchlists with enhanced symbols
PREMIUM_WATCHLISTS = {
    "stocks": {
        "mega_cap": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
        "growth": ["TSLA", "NFLX", "CRM", "ADBE", "PYPL"],
        "value": ["BRK.B", "JPM", "JNJ", "PG", "KO"],
        "tech": ["META", "ORCL", "INTC", "AMD", "QCOM"]
    },
    "crypto": {
        "major": ["BTC-USD", "ETH-USD"],
        "alt": ["SOL-USD", "ADA-USD", "DOT-USD"],
        "defi": ["UNI-USD", "AAVE-USD", "SUSHI-USD"]
    },
    "forex": {
        "major": ["EUR/USD", "GBP/USD", "USD/JPY"],
        "minor": ["AUD/USD", "USD/CAD", "NZD/USD"]
    }
}

# ==================== PREMIUM INDICATORS ====================

# Technical indicators configuration for premium analysis
PREMIUM_INDICATORS = {
    "momentum": ["rsi", "macd", "stoch", "cci"],
    "trend": ["sma", "ema", "adx", "aroon"],
    "volatility": ["bbands", "atr", "keltner"],
    "volume": ["obv", "ad", "cmf"]
}

# Economic indicators to track
ECONOMIC_INDICATORS = {
    "us": ["GDP", "CPI", "UNEMPLOYMENT", "FEDERAL_FUNDS_RATE"],
    "global": ["CRUDE_OIL", "GOLD", "VIX"]
}

# ==================== PREMIUM MONITORING ====================

# Enhanced monitoring configuration
PREMIUM_MONITORING = {
    "performance_tracking": True,
    "real_time_alerts": True,
    "data_quality_monitoring": True,
    "signal_performance_analysis": True,
    "risk_monitoring": True,
    "webhook_monitoring": True
}

# Alerting thresholds
ALERT_THRESHOLDS = {
    "profit_target": 0.05,      # 5% profit alert
    "loss_threshold": -0.03,    # 3% loss alert
    "volatility_spike": 0.10,   # 10% volatility alert
    "news_sentiment": 20,       # High sentiment alert
    "contradiction_strength": 0.5  # Contradiction alert
}

# ==================== PREMIUM FEATURES ====================

# Feature flags for premium capabilities
PREMIUM_FEATURES = {
    "multi_source_sentiment": True,
    "economic_indicators": True,
    "earnings_calendar": True,
    "technical_analysis_suite": True,
    "contradiction_detection": True,
    "webhook_integration": True,
    "real_time_news": True,
    "advanced_risk_management": True,
    "performance_analytics": True,
    "data_quality_scoring": True
} 