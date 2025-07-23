#!/usr/bin/env python3
"""
Kimera Aggressive Trader Launcher
=================================

Launches the ultra-aggressive 10-minute trader that attempts to turn $50 into $300.

⚠️ EXTREME RISK WARNING ⚠️
This is maximum risk trading. You could lose everything.
"""

import asyncio
import os
import sys
from datetime import datetime

def display_risk_warning():
    """Display comprehensive risk warning"""
    print("🚨" * 25)
    print("⚠️  EXTREME RISK WARNING - READ CAREFULLY ⚠️")
    print("🚨" * 25)
    print()
    print("📊 TRADING PARAMETERS:")
    print("   • Starting Capital: $50")
    print("   • Target Capital: $300")
    print("   • Target Return: 500% in 10 minutes")
    print("   • Risk Level: MAXIMUM")
    print("   • Strategy: Ultra-aggressive, high-frequency")
    print()
    print("⚠️  RISKS:")
    print("   • You could lose ALL $50 in minutes")
    print("   • Extremely high-risk trading strategies")
    print("   • Market volatility can cause rapid losses")
    print("   • No guarantees of profit")
    print("   • This is experimental trading")
    print()
    print("🎯 STRATEGY:")
    print("   • Scalping: Ultra-fast trades (20-60 seconds)")
    print("   • Momentum: Following strong price moves")
    print("   • Volatility: Trading on market chaos")
    print("   • Maximum position sizes (80-100% of capital)")
    print("   • Up to 120 trades in 10 minutes")
    print()
    print("💡 REQUIREMENTS:")
    print("   • Minimum $50 USDT in Binance account")
    print("   • Binance API keys with trading permissions")
    print("   • Stable internet connection")
    print("   • Ability to monitor for 10 minutes")
    print()
    print("🚨" * 25)

def get_user_confirmations():
    """Get multiple user confirmations"""
    print("\n🤔 MULTIPLE CONFIRMATIONS REQUIRED:")
    print()
    
    # First confirmation
    print("1️⃣ Do you understand this is EXTREMELY HIGH RISK?")
    response1 = input("   Type 'YES' to continue: ").strip().upper()
    if response1 != 'YES':
        return False
    
    # Second confirmation
    print("\n2️⃣ Do you understand you could lose ALL your money?")
    response2 = input("   Type 'YES' to continue: ").strip().upper()
    if response2 != 'YES':
        return False
    
    # Third confirmation
    print("\n3️⃣ Are you using money you can afford to lose completely?")
    response3 = input("   Type 'YES' to continue: ").strip().upper()
    if response3 != 'YES':
        return False
    
    # Final confirmation
    print("\n4️⃣ Type the exact phrase to start aggressive trading:")
    print("   'I ACCEPT ALL RISKS AND START AGGRESSIVE TRADING'")
    final_response = input("\nYour response: ").strip()
    
    return final_response == 'I ACCEPT ALL RISKS AND START AGGRESSIVE TRADING'

def check_prerequisites():
    """Check if prerequisites are met"""
    print("\n🔍 Checking Prerequisites...")
    
    # Check API credentials
    api_key = os.getenv('BINANCE_API_KEY')
    secret_key = os.getenv('BINANCE_SECRET_KEY')
    
    if not api_key or not secret_key:
        print("❌ Missing Binance API credentials")
        print("Please set environment variables:")
        print("  export BINANCE_API_KEY='your_api_key_here'")
        print("  export BINANCE_SECRET_KEY='your_secret_key_here'")
        return False
    
    print("✅ API credentials found")
    
    # Check dependencies
    try:
        import aiohttp
        print("✅ aiohttp available")
    except ImportError:
        print("❌ aiohttp not available - run: pip install aiohttp")
        return False
    
    print("✅ All prerequisites met")
    return True

def display_countdown():
    """Display countdown before starting"""
    print("\n🚀 STARTING AGGRESSIVE TRADING IN:")
    for i in range(5, 0, -1):
        print(f"   {i}...")
        import time
        time.sleep(1)
    print("   🔥 STARTING NOW!")

async def launch_aggressive_trader():
    """Launch the aggressive trader"""
    try:
        # Import and run the aggressive trader
        from kimera_aggressive_10min_trader import KimeraAggressive10MinTrader
        
        print("\n🚀 LAUNCHING KIMERA AGGRESSIVE TRADER (AUTO-LIQUIDATE + FORCE TRADE MODE)")
        print("=" * 50)
        print("🔄 All non-USDT assets will be sold for USDT at session start.")
        print("⚡ Kimera will always trade, even if the target is already met.")
        
        # Create and start trader
        trader = KimeraAggressive10MinTrader(
            starting_capital=50.0,
            target_capital=300.0,
            auto_liquidate=True,
            force_trade=True
        )
        
        print("⏰ 10-minute aggressive trading session starting...")
        print("📊 Monitor progress in real-time")
        print("🛑 Press Ctrl+C to emergency stop")
        print()
        
        await trader.start_aggressive_trading()
        
    except KeyboardInterrupt:
        print("\n🛑 EMERGENCY STOP - Trading halted by user")
        print("💰 Check your account balance manually")
    except Exception as e:
        print(f"\n❌ Trading error: {e}")
        print("💰 Check your account balance manually")
        print("📋 Check aggressive_trader.log for details")

def main():
    """Main launcher function"""
    print(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🚨 KIMERA AGGRESSIVE 10-MINUTE TRADER")
    print()
    
    # Display risk warning
    display_risk_warning()
    
    # Get user confirmations
    if not get_user_confirmations():
        print("\n👋 Trading cancelled. This was the smart choice for safety!")
        print("💡 Consider using the regular profit system instead:")
        print("   python start_kimera_profits.py")
        return
    
    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    
    # Final safety check
    print("\n⚠️  FINAL SAFETY CHECK:")
    print("   • Do you have exactly $50 or more in USDT?")
    print("   • Is your internet connection stable?")
    print("   • Can you monitor for the full 10 minutes?")
    print("   • Are you prepared for potential total loss?")
    
    final_check = input("\nAll checks passed? Type 'GO' to start: ").strip().upper()
    
    if final_check != 'GO':
        print("\n🛑 Trading cancelled")
        return
    
    # Countdown
    display_countdown()
    
    # Launch aggressive trader
    try:
        asyncio.run(launch_aggressive_trader())
    except KeyboardInterrupt:
        print("\n🛑 Launcher stopped")
    except Exception as e:
        print(f"\n❌ Launch error: {e}")
    
    print("\n📋 TRADING SESSION COMPLETE")
    print("💰 Check your final balance with: python check_kimera_profits.py")
    print("📊 Review trade log: aggressive_trader.log")

if __name__ == "__main__":
    main() 