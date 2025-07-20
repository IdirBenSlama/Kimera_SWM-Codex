"""
KIMERA Autonomous Trader Configuration
=====================================

Production-ready configuration for the autonomous profit trader.
Optimized for $2,000 profit target with risk management.
"""

import os
from typing import Dict, List, Any


class TradingConfig:
    """Configuration management for autonomous trading"""
    
    def __init__(self, mode: str = "production"):
        """
        Initialize trading configuration
        
        Args:
            mode: Trading mode ('production', 'testnet', 'simulation')
        """
        self.mode = mode
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration based on mode"""
        
        base_config = {
            # Core Settings
            'initial_balance': 10000.0,
            'profit_target': 2000.0,
            'max_drawdown': 0.10,  # 10% maximum drawdown
            'risk_per_trade': 0.02,  # 2% risk per trade
            'min_win_rate': 0.60,  # 60% minimum win rate
            'adaptive_sizing': True,
            'autonomous_mode': True,
            
            # Trading Symbols (prioritized by liquidity and volatility)
            'symbols': [
                'BTCUSDT',   # Bitcoin - highest liquidity
                'ETHUSDT',   # Ethereum - strong momentum
                'ADAUSDT',   # Cardano - good volatility
                'BNBUSDT',   # Binance Coin - exchange native
                'SOLUSDT',   # Solana - high growth potential
            ],
            
            # Timing Settings
            'update_interval': 5,       # Market data update (seconds)
            'analysis_interval': 15,    # Signal analysis (seconds)
            'rebalance_interval': 300,  # Portfolio rebalancing (seconds)
            'max_trade_duration': 3600, # Max trade duration (seconds)
            
            # Risk Management
            'max_position_pct': 0.20,      # Max 20% per position
            'max_portfolio_risk': 0.05,    # Max 5% portfolio risk
            'max_daily_loss': 0.05,        # Max 5% daily loss
            'max_consecutive_losses': 8,   # Max consecutive losses
            'emergency_stop_drawdown': 0.15, # Emergency stop at 15% drawdown
            
            # Signal Quality
            'min_signal_confidence': 0.6,  # Minimum signal confidence
            'min_market_score': 0.5,       # Minimum market condition score
            'signal_timeout': 300,          # Signal validity timeout (seconds)
            
            # Position Management
            'stop_loss_pct': 0.02,         # 2% stop loss
            'take_profit_pct': 0.06,       # 6% take profit
            'trailing_stop_pct': 0.015,    # 1.5% trailing stop
            'partial_profit_pct': 0.03,    # 3% partial profit taking
            
            # Performance Optimization
            'performance_review_interval': 1800,  # 30 minutes
            'strategy_adaptation_threshold': 0.1,  # 10% performance change
            'dynamic_risk_adjustment': True,
            'kelly_criterion_enabled': True,
            
            # Execution Settings
            'order_timeout': 30,           # Order timeout (seconds)
            'slippage_tolerance': 0.001,   # 0.1% slippage tolerance
            'partial_fill_threshold': 0.95, # 95% fill threshold
            'retry_attempts': 3,           # Order retry attempts
            
            # Logging and Monitoring
            'log_level': 'INFO',
            'log_trades': True,
            'log_performance': True,
            'log_interval': 300,           # Log progress every 5 minutes
            'save_state_interval': 900,    # Save state every 15 minutes
            
            # Safety Features
            'circuit_breaker_enabled': True,
            'max_daily_trades': 100,
            'cooling_off_period': 300,     # 5 minutes after losses
            'emergency_contacts': [],      # Emergency notification contacts
            
            # Cognitive Analysis
            'cognitive_engine_enabled': True,
            'sentiment_analysis_enabled': True,
            'anomaly_detection_enabled': True,
            'technical_analysis_enabled': True,
            'fundamental_analysis_enabled': True,
            
            # Exchange Configuration
            'exchanges': {
                'binance': {
                    'api_key': os.getenv('BINANCE_API_KEY', ''),
                    'private_key_path': os.getenv('BINANCE_PRIVATE_KEY_PATH', ''),
                    'testnet': self.mode != 'production',
                    'rate_limit': 1200,    # API calls per minute
                    'timeout': 30,         # Request timeout
                    'reconnect_attempts': 5,
                    'enable_websocket': True,
                    'websocket_reconnect': True,
                }
            }
        }
        
        # Mode-specific overrides
        if self.mode == 'testnet':
            base_config.update({
                'initial_balance': 1000.0,
                'profit_target': 200.0,
                'risk_per_trade': 0.01,    # More conservative for testing
                'max_drawdown': 0.05,      # Tighter drawdown limit
                'symbols': ['BTCUSDT', 'ETHUSDT'],  # Limited symbols for testing
                'log_level': 'DEBUG',
            })
        
        elif self.mode == 'simulation':
            base_config.update({
                'initial_balance': 10000.0,
                'profit_target': 2000.0,
                'risk_per_trade': 0.03,    # Slightly more aggressive
                'update_interval': 1,      # Faster simulation
                'analysis_interval': 5,
                'log_level': 'DEBUG',
                'exchanges': {
                    'binance': {
                        'api_key': 'simulation_key',
                        'private_key_path': 'test_key.pem',
                        'testnet': True,
                        'simulation_mode': True,
                    }
                }
            })
        
        return base_config
    
    def get_config(self) -> Dict[str, Any]:
        """Get the complete configuration"""
        return self.config.copy()
    
    def get_risk_config(self) -> Dict[str, Any]:
        """Get risk management configuration"""
        return {
            'max_position_pct': self.config['max_position_pct'],
            'max_portfolio_risk': self.config['max_portfolio_risk'],
            'max_drawdown': self.config['max_drawdown'],
            'risk_per_trade': self.config['risk_per_trade'],
            'stop_loss_pct': self.config['stop_loss_pct'],
            'take_profit_pct': self.config['take_profit_pct'],
            'max_consecutive_losses': self.config['max_consecutive_losses'],
            'emergency_stop_drawdown': self.config['emergency_stop_drawdown'],
        }
    
    def get_execution_config(self) -> Dict[str, Any]:
        """Get execution configuration"""
        return {
            'order_timeout': self.config['order_timeout'],
            'slippage_tolerance': self.config['slippage_tolerance'],
            'partial_fill_threshold': self.config['partial_fill_threshold'],
            'retry_attempts': self.config['retry_attempts'],
            'min_signal_confidence': self.config['min_signal_confidence'],
        }
    
    def get_symbols(self) -> List[str]:
        """Get trading symbols"""
        return self.config['symbols'].copy()
    
    def validate_config(self) -> bool:
        """Validate configuration"""
        try:
            # Check required fields
            required_fields = [
                'initial_balance', 'profit_target', 'symbols',
                'max_drawdown', 'risk_per_trade', 'exchanges'
            ]
            
            for field in required_fields:
                if field not in self.config:
                    raise ValueError(f"Missing required field: {field}")
            
            # Validate ranges
            if self.config['risk_per_trade'] <= 0 or self.config['risk_per_trade'] > 0.1:
                raise ValueError("Risk per trade must be between 0 and 0.1")
            
            if self.config['max_drawdown'] <= 0 or self.config['max_drawdown'] > 0.5:
                raise ValueError("Max drawdown must be between 0 and 0.5")
            
            if self.config['profit_target'] <= 0:
                raise ValueError("Profit target must be positive")
            
            if self.config['initial_balance'] <= 0:
                raise ValueError("Initial balance must be positive")
            
            # Validate symbols
            if not self.config['symbols']:
                raise ValueError("At least one symbol must be specified")
            
            # Validate exchange configuration
            if 'binance' not in self.config['exchanges']:
                raise ValueError("Binance exchange configuration required")
            
            return True
            
        except Exception as e:
            print(f"❌ Configuration validation failed: {e}")
            return False
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values"""
        self.config.update(updates)
    
    def save_config(self, filepath: str):
        """Save configuration to file"""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def load_config_from_file(self, filepath: str):
        """Load configuration from file"""
        import json
        with open(filepath, 'r') as f:
            file_config = json.load(f)
            self.config.update(file_config)


# Pre-configured setups for different scenarios
PRODUCTION_CONFIG = TradingConfig('production')
TESTNET_CONFIG = TradingConfig('testnet')
SIMULATION_CONFIG = TradingConfig('simulation')

# High-performance configuration for experienced traders
HIGH_PERFORMANCE_CONFIG = TradingConfig('production')
HIGH_PERFORMANCE_CONFIG.update_config({
    'risk_per_trade': 0.03,        # 3% risk per trade
    'max_position_pct': 0.25,      # 25% max position
    'profit_target': 5000.0,       # $5,000 target
    'min_signal_confidence': 0.7,  # Higher confidence requirement
    'symbols': ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'BNBUSDT', 'SOLUSDT', 'DOTUSDT'],
    'update_interval': 3,          # Faster updates
    'analysis_interval': 10,       # Faster analysis
})

# Conservative configuration for risk-averse traders
CONSERVATIVE_CONFIG = TradingConfig('production')
CONSERVATIVE_CONFIG.update_config({
    'risk_per_trade': 0.01,        # 1% risk per trade
    'max_position_pct': 0.15,      # 15% max position
    'max_drawdown': 0.05,          # 5% max drawdown
    'profit_target': 1000.0,       # $1,000 target
    'min_signal_confidence': 0.8,  # Very high confidence
    'symbols': ['BTCUSDT', 'ETHUSDT'],  # Only major pairs
    'stop_loss_pct': 0.015,        # Tighter stop loss
    'take_profit_pct': 0.04,       # Smaller take profit
})

# Scalping configuration for high-frequency trading
SCALPING_CONFIG = TradingConfig('production')
SCALPING_CONFIG.update_config({
    'risk_per_trade': 0.005,       # 0.5% risk per trade
    'max_position_pct': 0.10,      # 10% max position
    'profit_target': 500.0,        # $500 target
    'max_trade_duration': 900,     # 15 minutes max
    'update_interval': 1,          # 1 second updates
    'analysis_interval': 5,        # 5 second analysis
    'stop_loss_pct': 0.005,        # 0.5% stop loss
    'take_profit_pct': 0.015,      # 1.5% take profit
    'min_signal_confidence': 0.9,  # Very high confidence
    'symbols': ['BTCUSDT'],        # Single symbol focus
})


def get_config_for_mode(mode: str) -> TradingConfig:
    """Get configuration for specified mode"""
    if mode == 'production':
        return PRODUCTION_CONFIG
    elif mode == 'testnet':
        return TESTNET_CONFIG
    elif mode == 'simulation':
        return SIMULATION_CONFIG
    elif mode == 'high_performance':
        return HIGH_PERFORMANCE_CONFIG
    elif mode == 'conservative':
        return CONSERVATIVE_CONFIG
    elif mode == 'scalping':
        return SCALPING_CONFIG
    else:
        raise ValueError(f"Unknown configuration mode: {mode}")


def create_custom_config(base_mode: str = 'production', **kwargs) -> TradingConfig:
    """Create custom configuration based on base mode"""
    config = get_config_for_mode(base_mode)
    if kwargs:
        config.update_config(kwargs)
    return config


# Example usage
if __name__ == "__main__":
    # Test configuration validation
    config = TradingConfig('production')
    
    if config.validate_config():
        print("✅ Configuration validation passed")
        print(f"   Mode: {config.mode}")
        print(f"   Symbols: {config.get_symbols()}")
        print(f"   Profit Target: ${config.config['profit_target']:,.2f}")
        print(f"   Risk per Trade: {config.config['risk_per_trade']:.1%}")
        print(f"   Max Drawdown: {config.config['max_drawdown']:.1%}")
    else:
        print("❌ Configuration validation failed") 