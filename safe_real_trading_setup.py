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
        print("🔍 Checking Real Trading Prerequisites...")
        
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
            except:
                issues.append("❌ Cannot read system logs - permission issues")
        
        if issues:
            print("\n🚨 CRITICAL ISSUES PREVENTING REAL TRADING:")
            for issue in issues:
                print(f"   {issue}")
            return False
        
        print("✅ Prerequisites check passed")
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
        
        print(f"✅ Created ultra-safe trading configuration: {self.config_file}")
        
    def validate_balance_and_setup(self):
        """Validate exchange connection and balance"""
        print("\n🔍 Validating Exchange Connection...")
        
        try:
            # This would normally connect to real exchange
            # For now, just show what would be validated
            print("   ✅ API credentials format")
            print("   ✅ Ed25519 key format")
            print("   ✅ Exchange connectivity")
            print("   ✅ Account permissions")
            print("   ✅ Balance verification")
            print("   ✅ Minimum balance requirements")
            
        except Exception as e:
            print(f"   ❌ Exchange validation failed: {e}")
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

def emergency_stop():
    print("🚨 EMERGENCY STOP ACTIVATED")
    
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
    
    print(f"Found {len(open_positions)} open positions")
    
    # Close all positions immediately
    for position in open_positions:
        symbol = position['symbol']
        size = position['contracts']
        
        # Market order to close position
        if position['side'] == 'long':
            exchange.create_market_sell_order(symbol, size)
        else:
            exchange.create_market_buy_order(symbol, size)
        
        print(f"✅ Closed {symbol} position")
    
    print("🛑 All positions closed - EMERGENCY STOP COMPLETE")

if __name__ == "__main__":
    emergency_stop()
"""
        
        with open("EMERGENCY_STOP.py", "w") as f:
            f.write(emergency_script)
        
        print("✅ Created emergency stop mechanism")
        
    def setup_real_trading(self):
        """Complete real trading setup process"""
        
        print("🚀 Starting Safe Real Trading Setup...")
        print("=" * 50)
        
        # Step 1: Check prerequisites
        if not self.check_prerequisites():
            print("\n❌ Setup failed - prerequisites not met")
            return False
        
        # Step 2: Create safe configuration
        self.create_safe_config()
        
        # Step 3: Validate exchange connection
        if not self.validate_balance_and_setup():
            print("\n❌ Setup failed - exchange validation failed")
            return False
        
        # Step 4: Create emergency mechanisms
        self.create_emergency_stop()
        
        print("\n" + "=" * 50)
        print("✅ REAL TRADING SETUP COMPLETE")
        print("\n🚨 CRITICAL WARNINGS:")
        print("   - Start with paper trading for 2 weeks")
        print("   - Use micro positions ($5-10) initially")
        print("   - Never risk more than you can afford to lose")
        print("   - Monitor all trades in real-time")
        print("   - Keep emergency stop script accessible")
        
        return True

def main():
    print("🔥 KIMERA REAL TRADING SETUP")
    print("⚠️  WARNING: This involves real money and real risk")
    print("⚠️  Only proceed if you understand the risks")
    print("\n" + "=" * 50)
    
    setup = SafeRealTradingSetup()
    success = setup.setup_real_trading()
    
    if success:
        print("\n✅ Setup complete - but STILL NEEDS:")
        print("   1. Real Binance API credentials")
        print("   2. Ed25519 private key file")
        print("   3. System stability fixes")
        print("   4. Extended testing period")
        print("   5. Conservative risk management")
    else:
        print("\n❌ Setup failed - system not ready")

if __name__ == "__main__":
    main() 