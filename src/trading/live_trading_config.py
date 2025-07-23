"""
KIMERA LIVE TRADING CONFIGURATION
=================================

This configuration enables REAL trading execution with proper risk controls.
USE WITH EXTREME CAUTION - REAL MONEY IS AT RISK.

SAFETY FEATURES:
- Position size limits
- Daily loss limits  
- Emergency stop mechanisms
- Risk-based trade sizing
- Real-time monitoring

REQUIREMENTS:
- Valid exchange API keys
- Proper risk management understanding
- Sufficient capital for position sizes
- Active monitoring setup
"""

import os
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from decimal import Decimal
import json

logger = logging.getLogger(__name__)

@dataclass
class LiveTradingConfig:
    """Configuration for live trading with real money"""
    
    # Exchange Configuration
    binance_api_key: str
    binance_private_key_path: str
    
    # Trading Parameters
    initial_balance: float = 1000.0
    max_position_size_usd: float = 100.0  # Max $100 per position
    max_daily_loss_usd: float = 50.0      # Max $50 daily loss
    risk_per_trade_pct: float = 0.02      # 2% risk per trade
    
    # Safety Controls
    max_concurrent_positions: int = 3
    min_signal_confidence: float = 0.75
    emergency_stop_loss_pct: float = 0.10  # 10% portfolio loss emergency stop
    
    # Trading Symbols
    symbols: List[str] = None
    
    # Update Intervals
    analysis_interval_seconds: int = 30    # 30 second analysis
    execution_check_interval: int = 10     # 10 second execution check
    
    # Risk Management
    stop_loss_pct: float = 0.03           # 3% stop loss
    take_profit_pct: float = 0.06         # 6% take profit
    trailing_stop_pct: float = 0.02       # 2% trailing stop
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ['BTCUSDT', 'ETHUSDT']  # Only major pairs initially


class KimeraLiveTradingManager:
    """
    Manages live trading configuration and execution for Kimera.
    Bridges the gap between analysis and real market execution.
    """
    
    def __init__(self, config: LiveTradingConfig):
        self.config = config
        self.daily_pnl = 0.0
        self.active_positions = {}
        self.emergency_stop_active = False
        self.trading_enabled = False
        
        # Validate configuration
        if not self._validate_config():
            raise ValueError("Invalid live trading configuration")
        
        logger.info("🔴 LIVE TRADING MANAGER INITIALIZED")
        logger.info("⚠️  WARNING: REAL MONEY TRADING ENABLED")
        logger.info(f"   Max Position Size: ${self.config.max_position_size_usd}")
        logger.info(f"   Max Daily Loss: ${self.config.max_daily_loss_usd}")
        logger.info(f"   Risk Per Trade: {self.config.risk_per_trade_pct:.1%}")
    
    def _validate_config(self) -> bool:
        """Validate live trading configuration"""
        try:
            # Check API keys
            if not self.config.binance_api_key or len(self.config.binance_api_key) < 10:
                logger.error("❌ Invalid Binance API key")
                return False
            
            if not os.path.exists(self.config.binance_private_key_path):
                logger.error(f"❌ Private key file not found: {self.config.binance_private_key_path}")
                return False
            
            # Check risk parameters
            if self.config.risk_per_trade_pct > 0.05:  # Max 5% risk per trade
                logger.error("❌ Risk per trade too high (max 5%)")
                return False
            
            if self.config.max_position_size_usd > self.config.initial_balance * 0.5:
                logger.error("❌ Max position size too large (max 50% of balance)")
                return False
            
            if self.config.max_daily_loss_usd > self.config.initial_balance * 0.2:
                logger.error("❌ Daily loss limit too high (max 20% of balance)")
                return False
            
            logger.info("✅ Live trading configuration validated")
            return True
            
        except Exception as e:
            logger.error(f"❌ Configuration validation failed: {e}")
            return False
    
    def get_execution_config(self) -> Dict[str, Any]:
        """Get configuration for execution bridge"""
        return {
            'exchanges': {
                'binance': {
                    'api_key': self.config.binance_api_key,
                    'private_key_path': self.config.binance_private_key_path,
                    'testnet': False,  # LIVE TRADING - NOT TESTNET
                    'simulation_mode': False,  # REAL EXECUTION
                }
            },
            'max_order_size': self.config.max_position_size_usd,
            'max_daily_volume': self.config.max_daily_loss_usd * 5,  # 5x daily loss as volume limit
            'autonomous_mode': True,
            'risk_management': {
                'max_position_size': self.config.max_position_size_usd,
                'max_daily_loss': self.config.max_daily_loss_usd,
                'risk_per_trade': self.config.risk_per_trade_pct,
                'stop_loss_pct': self.config.stop_loss_pct,
                'take_profit_pct': self.config.take_profit_pct,
                'emergency_stop_loss_pct': self.config.emergency_stop_loss_pct,
            }
        }
    
    def get_signal_config(self) -> Dict[str, Any]:
        """Get configuration for signal generation"""
        return {
            'symbols': self.config.symbols,
            'min_confidence': self.config.min_signal_confidence,
            'max_concurrent_positions': self.config.max_concurrent_positions,
            'analysis_interval': self.config.analysis_interval_seconds,
        }
    
    def enable_trading(self) -> bool:
        """Enable live trading after final confirmation"""
        if self.emergency_stop_active:
            logger.error("❌ Cannot enable trading - emergency stop active")
            return False
        
        logger.warning("🔴 ENABLING LIVE TRADING")
        logger.warning("⚠️  REAL MONEY AT RISK")
        logger.warning("⚠️  MONITOR CLOSELY")
        
        self.trading_enabled = True
        return True
    
    def disable_trading(self):
        """Disable live trading"""
        logger.info("🛑 DISABLING LIVE TRADING")
        self.trading_enabled = False
    
    def emergency_stop(self):
        """Activate emergency stop"""
        logger.critical("🚨 EMERGENCY STOP ACTIVATED")
        self.emergency_stop_active = True
        self.trading_enabled = False
    
    def check_risk_limits(self, position_size: float) -> bool:
        """Check if new position respects risk limits"""
        if self.emergency_stop_active:
            return False
        
        if not self.trading_enabled:
            return False
        
        # Check position size limit
        if position_size > self.config.max_position_size_usd:
            logger.warning(f"❌ Position size ${position_size:.2f} exceeds limit ${self.config.max_position_size_usd:.2f}")
            return False
        
        # Check daily loss limit
        if self.daily_pnl < -self.config.max_daily_loss_usd:
            logger.warning(f"❌ Daily loss limit reached: ${self.daily_pnl:.2f}")
            self.emergency_stop()
            return False
        
        # Check concurrent positions
        if len(self.active_positions) >= self.config.max_concurrent_positions:
            logger.warning(f"❌ Max concurrent positions reached: {len(self.active_positions)}")
            return False
        
        return True
    
    def update_daily_pnl(self, pnl: float):
        """Update daily P&L and check emergency stop"""
        self.daily_pnl += pnl
        
        # Check emergency stop condition
        if self.daily_pnl < -self.config.max_daily_loss_usd:
            logger.critical(f"🚨 EMERGENCY STOP: Daily loss ${self.daily_pnl:.2f} exceeds limit ${self.config.max_daily_loss_usd:.2f}")
            self.emergency_stop()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current trading status"""
        return {
            'trading_enabled': self.trading_enabled,
            'emergency_stop_active': self.emergency_stop_active,
            'daily_pnl': self.daily_pnl,
            'active_positions': len(self.active_positions),
            'daily_loss_limit': self.config.max_daily_loss_usd,
            'position_size_limit': self.config.max_position_size_usd,
            'risk_per_trade': self.config.risk_per_trade_pct,
        }


def create_live_trading_config(
    binance_api_key: str,
    binance_private_key_path: str,
    initial_balance: float = 1000.0,
    max_position_size: float = 100.0,
    max_daily_loss: float = 50.0
) -> LiveTradingConfig:
    """Create a live trading configuration"""
    
    return LiveTradingConfig(
        binance_api_key=binance_api_key,
        binance_private_key_path=binance_private_key_path,
        initial_balance=initial_balance,
        max_position_size_usd=max_position_size,
        max_daily_loss_usd=max_daily_loss,
    )


def create_conservative_live_config(
    binance_api_key: str,
    binance_private_key_path: str
) -> LiveTradingConfig:
    """Create a conservative live trading configuration"""
    
    return LiveTradingConfig(
        binance_api_key=binance_api_key,
        binance_private_key_path=binance_private_key_path,
        initial_balance=500.0,
        max_position_size_usd=25.0,   # $25 max position
        max_daily_loss_usd=10.0,      # $10 max daily loss
        risk_per_trade_pct=0.01,      # 1% risk per trade
        min_signal_confidence=0.8,    # 80% confidence required
        symbols=['BTCUSDT'],          # Only Bitcoin
        max_concurrent_positions=1,   # Only one position at a time
    )


# Example usage - REPLACE WITH YOUR ACTUAL CREDENTIALS
if __name__ == "__main__":
    # ⚠️ WARNING: REPLACE WITH YOUR ACTUAL API CREDENTIALS
    # NEVER COMMIT REAL CREDENTIALS TO VERSION CONTROL
    
    # Example configuration (DO NOT USE THESE VALUES)
    config = create_conservative_live_config(
        binance_api_key="YOUR_BINANCE_API_KEY_HERE",
        binance_private_key_path="path/to/your/private_key.pem"
    )
    
    manager = KimeraLiveTradingManager(config)
    
    print("Live Trading Configuration:")
    print(json.dumps(manager.get_status(), indent=2))
    print()
    print("⚠️  To enable live trading, you must:")
    print("   1. Set your actual API credentials")
    print("   2. Ensure you have the private key file")
    print("   3. Understand the risks involved")
    print("   4. Monitor the system actively")
    print("   5. Have emergency stop procedures ready") 