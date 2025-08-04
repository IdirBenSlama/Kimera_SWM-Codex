#!/usr/bin/env python3
"""
Kimera Profit System Launcher
=============================

Simple script to start the Kimera Autonomous Profit System.
This system will continuously analyze markets, execute profitable trades,
and compound profits automatically.

Usage:
    python start_kimera_profits.py
"""

import asyncio
import os
import sys
from datetime import datetime

def check_prerequisites():
    """Check if all prerequisites are met"""
    logger.info("🔍 Checking Prerequisites...")
    
    # Check API credentials
    api_key = os.getenv('BINANCE_API_KEY')
    secret_key = os.getenv('BINANCE_SECRET_KEY')
    
    if not api_key or not secret_key:
        logger.info("❌ Missing Binance API credentials")
        logger.info("Please set environment variables:")
        logger.info("  export BINANCE_API_KEY='your_api_key'")
        logger.info("  export BINANCE_SECRET_KEY='your_secret_key'")
        return False
    
    logger.info("✅ API credentials found")
    
    # Check dependencies
    try:
        import aiohttp
        logger.info("✅ aiohttp available")
    except ImportError:
        logger.info("❌ aiohttp not available - run: pip install aiohttp")
        return False
    
    return True

def display_startup_info():
    """Display startup information"""
    logger.info("\n" + "="*60)
    logger.info("🚀 KIMERA AUTONOMOUS PROFIT SYSTEM")
    logger.info("="*60)
    logger.info("📊 This system will:")
    logger.info("   • Continuously analyze Bitcoin and Ethereum markets")
    logger.info("   • Execute profitable trades automatically")
    logger.info("   • Manage risk with intelligent position sizing")
    logger.info("   • Compound profits for exponential growth")
    logger.info("   • Learn and adapt from each trade")
    logger.info("   • Operate 24/7 without human intervention")
    logger.info("\n⚠️  IMPORTANT:")
    logger.info("   • This system trades with REAL MONEY")
    logger.info("   • Start with small amounts to test")
    logger.info("   • Monitor performance regularly")
    logger.info("   • System will create profit_system.log file")
    logger.info("="*60)

def get_user_confirmation():
    """Get user confirmation to proceed"""
    logger.info("\n🤔 Are you ready to start making autonomous profits?")
    logger.info("   Type 'YES' to start the system")
    logger.info("   Type 'NO' to cancel")
    
    while True:
        response = input("\nYour choice: ").strip().upper()
        if response == 'YES':
            return True
        elif response == 'NO':
            return False
        else:
            logger.info("Please type 'YES' or 'NO'")

async def start_profit_system():
    """Start the autonomous profit system"""
    try:
        # Import the main system
        from kimera_autonomous_profit_system import KimeraAutonomousProfitSystem
import logging
logger = logging.getLogger(__name__)
        
        # Get starting capital
        logger.info("\n💰 How much capital would you like to start with?")
        logger.info("   Recommended: $50-100 for testing")
        
        while True:
            try:
                capital = float(input("Starting capital ($): "))
                if capital > 0:
                    break
                else:
                    logger.info("Please enter a positive amount")
            except ValueError:
                logger.info("Please enter a valid number")
        
        # Initialize system
        logger.info(f"\n🔄 Initializing Kimera Profit System with ${capital:.2f}")
        system = KimeraAutonomousProfitSystem(starting_capital=capital)
        
        # Start the system
        logger.info("🚀 Starting autonomous profit generation...")
        logger.info("📊 Monitor progress in the terminal and profit_system.log file")
        logger.info("🛑 Press Ctrl+C to stop the system")
        
        await system.start()
        
    except KeyboardInterrupt:
        logger.info("\n🛑 System stopped by user")
        await system.stop()
    except Exception as e:
        logger.info(f"\n❌ Error starting profit system: {e}")
        logger.info("Check the log file for details")

def main():
    """Main entry point"""
    logger.info(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    
    # Display startup information
    display_startup_info()
    
    # Get user confirmation
    if not get_user_confirmation():
        logger.info("\n👋 Goodbye! Run this script again when you're ready to start making profits.")
        sys.exit(0)
    
    # Start the system
    logger.info("\n🎯 Starting Kimera Autonomous Profit System...")
    try:
        asyncio.run(start_profit_system())
    except KeyboardInterrupt:
        logger.info("\n👋 System stopped. See you next time!")
    except Exception as e:
        logger.info(f"\n❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 