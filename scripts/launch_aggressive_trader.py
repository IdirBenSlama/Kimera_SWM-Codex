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
    logger.info("🚨" * 25)
    logger.info("⚠️  EXTREME RISK WARNING - READ CAREFULLY ⚠️")
    logger.info("🚨" * 25)
    logger.info()
    logger.info("📊 TRADING PARAMETERS:")
    logger.info("   • Starting Capital: $50")
    logger.info("   • Target Capital: $300")
    logger.info("   • Target Return: 500% in 10 minutes")
    logger.info("   • Risk Level: MAXIMUM")
    logger.info("   • Strategy: Ultra-aggressive, high-frequency")
    logger.info()
    logger.info("⚠️  RISKS:")
    logger.info("   • You could lose ALL $50 in minutes")
    logger.info("   • Extremely high-risk trading strategies")
    logger.info("   • Market volatility can cause rapid losses")
    logger.info("   • No guarantees of profit")
    logger.info("   • This is experimental trading")
    logger.info()
    logger.info("🎯 STRATEGY:")
    logger.info("   • Scalping: Ultra-fast trades (20-60 seconds)")
    logger.info("   • Momentum: Following strong price moves")
    logger.info("   • Volatility: Trading on market chaos")
    logger.info("   • Maximum position sizes (80-100% of capital)")
    logger.info("   • Up to 120 trades in 10 minutes")
    logger.info()
    logger.info("💡 REQUIREMENTS:")
    logger.info("   • Minimum $50 USDT in Binance account")
    logger.info("   • Binance API keys with trading permissions")
    logger.info("   • Stable internet connection")
    logger.info("   • Ability to monitor for 10 minutes")
    logger.info()
    logger.info("🚨" * 25)

def get_user_confirmations():
    """Get multiple user confirmations"""
    logger.info("\n🤔 MULTIPLE CONFIRMATIONS REQUIRED:")
    logger.info()
    
    # First confirmation
    logger.info("1️⃣ Do you understand this is EXTREMELY HIGH RISK?")
    response1 = input("   Type 'YES' to continue: ").strip().upper()
    if response1 != 'YES':
        return False
    
    # Second confirmation
    logger.info("\n2️⃣ Do you understand you could lose ALL your money?")
    response2 = input("   Type 'YES' to continue: ").strip().upper()
    if response2 != 'YES':
        return False
    
    # Third confirmation
    logger.info("\n3️⃣ Are you using money you can afford to lose completely?")
    response3 = input("   Type 'YES' to continue: ").strip().upper()
    if response3 != 'YES':
        return False
    
    # Final confirmation
    logger.info("\n4️⃣ Type the exact phrase to start aggressive trading:")
    logger.info("   'I ACCEPT ALL RISKS AND START AGGRESSIVE TRADING'")
    final_response = input("\nYour response: ").strip()
    
    return final_response == 'I ACCEPT ALL RISKS AND START AGGRESSIVE TRADING'

def check_prerequisites():
    """Check if prerequisites are met"""
    logger.info("\n🔍 Checking Prerequisites...")
    
    # Check API credentials
    api_key = os.getenv('BINANCE_API_KEY')
    secret_key = os.getenv('BINANCE_SECRET_KEY')
    
    if not api_key or not secret_key:
        logger.info("❌ Missing Binance API credentials")
        logger.info("Please set environment variables:")
        logger.info("  export BINANCE_API_KEY='your_api_key_here'")
        logger.info("  export BINANCE_SECRET_KEY='your_secret_key_here'")
        return False
    
    logger.info("✅ API credentials found")
    
    # Check dependencies
    try:
        import aiohttp
        logger.info("✅ aiohttp available")
    except ImportError:
        logger.info("❌ aiohttp not available - run: pip install aiohttp")
        return False
    
    logger.info("✅ All prerequisites met")
    return True

def display_countdown():
    """Display countdown before starting"""
    logger.info("\n🚀 STARTING AGGRESSIVE TRADING IN:")
    for i in range(5, 0, -1):
        logger.info(f"   {i}...")
        import time
        time.sleep(1)
    logger.info("   🔥 STARTING NOW!")

async def launch_aggressive_trader():
    """Launch the aggressive trader"""
    try:
        # Import and run the aggressive trader
        from kimera_aggressive_10min_trader import KimeraAggressive10MinTrader
import logging
logger = logging.getLogger(__name__)
        
        logger.info("\n🚀 LAUNCHING KIMERA AGGRESSIVE TRADER (AUTO-LIQUIDATE + FORCE TRADE MODE)")
        logger.info("=" * 50)
        logger.info("🔄 All non-USDT assets will be sold for USDT at session start.")
        logger.info("⚡ Kimera will always trade, even if the target is already met.")
        
        # Create and start trader
        trader = KimeraAggressive10MinTrader(
            starting_capital=50.0,
            target_capital=300.0,
            auto_liquidate=True,
            force_trade=True
        )
        
        logger.info("⏰ 10-minute aggressive trading session starting...")
        logger.info("📊 Monitor progress in real-time")
        logger.info("🛑 Press Ctrl+C to emergency stop")
        logger.info()
        
        await trader.start_aggressive_trading()
        
    except KeyboardInterrupt:
        logger.info("\n🛑 EMERGENCY STOP - Trading halted by user")
        logger.info("💰 Check your account balance manually")
    except Exception as e:
        logger.info(f"\n❌ Trading error: {e}")
        logger.info("💰 Check your account balance manually")
        logger.info("📋 Check aggressive_trader.log for details")

def main():
    """Main launcher function"""
    logger.info(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("🚨 KIMERA AGGRESSIVE 10-MINUTE TRADER")
    logger.info()
    
    # Display risk warning
    display_risk_warning()
    
    # Get user confirmations
    if not get_user_confirmations():
        logger.info("\n👋 Trading cancelled. This was the smart choice for safety!")
        logger.info("💡 Consider using the regular profit system instead:")
        logger.info("   python start_kimera_profits.py")
        return
    
    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    
    # Final safety check
    logger.info("\n⚠️  FINAL SAFETY CHECK:")
    logger.info("   • Do you have exactly $50 or more in USDT?")
    logger.info("   • Is your internet connection stable?")
    logger.info("   • Can you monitor for the full 10 minutes?")
    logger.info("   • Are you prepared for potential total loss?")
    
    final_check = input("\nAll checks passed? Type 'GO' to start: ").strip().upper()
    
    if final_check != 'GO':
        logger.info("\n🛑 Trading cancelled")
        return
    
    # Countdown
    display_countdown()
    
    # Launch aggressive trader
    try:
        asyncio.run(launch_aggressive_trader())
    except KeyboardInterrupt:
        logger.info("\n🛑 Launcher stopped")
    except Exception as e:
        logger.info(f"\n❌ Launch error: {e}")
    
    logger.info("\n📋 TRADING SESSION COMPLETE")
    logger.info("💰 Check your final balance with: python check_kimera_profits.py")
    logger.info("📊 Review trade log: aggressive_trader.log")

if __name__ == "__main__":
    main() 