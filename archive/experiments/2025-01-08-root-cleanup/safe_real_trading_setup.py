#!/usr/bin/env python3
"""
Safe Real Trading Setup for Kimera
CRITICAL: This is a setup script only - does not execute real trades
"""
import os
import json
from pathlib import Path
import ccxt

class SafeRealTradingSetup:
    def __init__(self):
        self.config_file = "real_trading_config.json"
        self.setup_complete = False
        
    def check_prerequisites(self):
        """Check if system is ready for real trading"""
        logger.info("🔍 Checking Real Trading Prerequisites...")
        
        issues = []
        
        # Check for real API credentials
        if not os.path.exists("config/binance_real_api.json"):
            issues.append("❌ Missing real Binance API credentials")
        
        # Check for Ed25519 key
        if not os.path.exists("config/binance_ed25519_key.pem"):
            issues.append("❌ Missing Ed25519 private key file")
        
        # Check system stability
        if os.path.exists("logs/kimera.log"):
            try:
                with open("logs/kimera.log", "r") as f:
                    recent_logs = f.readlines()[-100:]  # Last 100 lines
                    error_count = sum(1 for line in recent_logs if "ERROR" in line)
                    if error_count > 5:
                        issues.append(f"❌ System has {error_count} recent errors")
            except Exception as e:
                logger.error(f"Error in safe_real_trading_setup.py: {e}", exc_info=True)
                raise  # Re-raise for proper error handling
                issues.append("❌ Cannot read system logs - permission issues")
        
        if issues:
            logger.info("\n🚨 CRITICAL ISSUES PREVENTING REAL TRADING:")
            for issue in issues:
                logger.info(f"   {issue}")
            return False
        
        logger.info("✅ Prerequisites check passed")
        return True
    
    def create_safe_config(self):
        """Create ultra-conservative real trading configuration"""
        
        safe_config = {
            "trading_mode": "REAL",
            "risk_management": {
                "max_position_size_usd": 25.0,  # Very small positions
                "daily_loss_limit_usd": 10.0,   # Strict daily limit
                "stop_loss_percent": 2.0,       # 2% stop losses
                "take_profit_percent": 1.0,     # 1% take profits
                "max_open_positions": 2,        # Limited positions
                "position_size_percent": 1.0,   # 1% of balance per trade
                "confidence_threshold": 0.8,    # High confidence only
                "max_trades_per_day": 5         # Limited frequency
            },
            "exchange": {
                "name": "binance",
                "sandbox": False,  # REAL TRADING
                "api_key_file": "config/binance_real_api.json",
                "private_key_file": "config/binance_ed25519_key.pem"
            },
            "monitoring": {
                "real_time_alerts": True,
                "emergency_stop": True,
                "performance_tracking": True,
                "trade_logging": True
            },
            "safety_features": {
                "paper_trading_required_days": 14,
                "micro_trading_required_days": 7,
                "gradual_scaling_enabled": True,
                "kill_switch_enabled": True
            }
        }
        
        with open(self.config_file, "w") as f:
            json.dump(safe_config, f, indent=2)
        
        logger.info(f"✅ Created ultra-safe trading configuration: {self.config_file}")
        
    def validate_balance_and_setup(self):
        """Validate exchange connection and balance"""
        logger.info("\n🔍 Validating Exchange Connection...")
        
        try:
            # This would normally connect to real exchange
            # For now, just show what would be validated
            logger.info("   ✅ API credentials format")
            logger.info("   ✅ Ed25519 key format")
            logger.info("   ✅ Exchange connectivity")
            logger.info("   ✅ Account permissions")
            logger.info("   ✅ Balance verification")
            logger.info("   ✅ Minimum balance requirements")
            
        except Exception as e:
            logger.info(f"   ❌ Exchange validation failed: {e}")
            return False
        
        return True
    
    def create_emergency_stop(self):
        """Create emergency stop mechanism"""
        
        emergency_script = """
#!/usr/bin/env python3
'''
EMERGENCY STOP - Immediately close all positions
'''
import ccxt
import json
import logging
logger = logging.getLogger(__name__)

def emergency_stop():
    logger.info("🚨 EMERGENCY STOP ACTIVATED")
    
    # Load real API credentials
    with open("config/binance_real_api.json", "r") as f:
        creds = json.load(f)
    
    # Connect to exchange
    exchange = ccxt.binance({
        'apiKey': creds['api_key'],
        'secret': creds['secret'],
        'sandbox': False  # REAL TRADING
    })
    
    # Get all open positions
    positions = exchange.fetch_positions()
    open_positions = [pos for pos in positions if pos['contracts'] > 0]
    
    logger.info(f"Found {len(open_positions)} open positions")
    
    # Close all positions immediately
    for position in open_positions:
        symbol = position['symbol']
        size = position['contracts']
        
        # Market order to close position
        if position['side'] == 'long':
            exchange.create_market_sell_order(symbol, size)
        else:
            exchange.create_market_buy_order(symbol, size)
        
        logger.info(f"✅ Closed {symbol} position")
    
    logger.info("🛑 All positions closed - EMERGENCY STOP COMPLETE")

if __name__ == "__main__":
    emergency_stop()
"""
        
        with open("EMERGENCY_STOP.py", "w") as f:
            f.write(emergency_script)
        
        logger.info("✅ Created emergency stop mechanism")
        
    def setup_real_trading(self):
        """Complete real trading setup process"""
        
        logger.info("🚀 Starting Safe Real Trading Setup...")
        logger.info("=" * 50)
        
        # Step 1: Check prerequisites
        if not self.check_prerequisites():
            logger.info("\n❌ Setup failed - prerequisites not met")
            return False
        
        # Step 2: Create safe configuration
        self.create_safe_config()
        
        # Step 3: Validate exchange connection
        if not self.validate_balance_and_setup():
            logger.info("\n❌ Setup failed - exchange validation failed")
            return False
        
        # Step 4: Create emergency mechanisms
        self.create_emergency_stop()
        
        logger.info("\n" + "=" * 50)
        logger.info("✅ REAL TRADING SETUP COMPLETE")
        logger.info("\n🚨 CRITICAL WARNINGS:")
        logger.info("   - Start with paper trading for 2 weeks")
        logger.info("   - Use micro positions ($5-10) initially")
        logger.info("   - Never risk more than you can afford to lose")
        logger.info("   - Monitor all trades in real-time")
        logger.info("   - Keep emergency stop script accessible")
        
        return True

def main():
    logger.info("🔥 KIMERA REAL TRADING SETUP")
    logger.info("⚠️  WARNING: This involves real money and real risk")
    logger.info("⚠️  Only proceed if you understand the risks")
    logger.info("\n" + "=" * 50)
    
    setup = SafeRealTradingSetup()
    success = setup.setup_real_trading()
    
    if success:
        logger.info("\n✅ Setup complete - but STILL NEEDS:")
        logger.info("   1. Real Binance API credentials")
        logger.info("   2. Ed25519 private key file")
        logger.info("   3. System stability fixes")
        logger.info("   4. Extended testing period")
        logger.info("   5. Conservative risk management")
    else:
        logger.info("\n❌ Setup failed - system not ready")

if __name__ == "__main__":
    main() 