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
    print("🔍 Checking Prerequisites...")
    
    # Check API credentials
    api_key = os.getenv('BINANCE_API_KEY')
    secret_key = os.getenv('BINANCE_SECRET_KEY')
    
    if not api_key or not secret_key:
        print("❌ Missing Binance API credentials")
        print("Please set environment variables:")
        print("  export BINANCE_API_KEY='your_api_key'")
        print("  export BINANCE_SECRET_KEY='your_secret_key'")
        return False
    
    print("✅ API credentials found")
    
    # Check dependencies
    try:
        import aiohttp
        print("✅ aiohttp available")
    except ImportError:
        print("❌ aiohttp not available - run: pip install aiohttp")
        return False
    
    return True

def display_startup_info():
    """Display startup information"""
    print("\n" + "="*60)
    print("🚀 KIMERA AUTONOMOUS PROFIT SYSTEM")
    print("="*60)
    print("📊 This system will:")
    print("   • Continuously analyze Bitcoin and Ethereum markets")
    print("   • Execute profitable trades automatically")
    print("   • Manage risk with intelligent position sizing")
    print("   • Compound profits for exponential growth")
    print("   • Learn and adapt from each trade")
    print("   • Operate 24/7 without human intervention")
    print("\n⚠️  IMPORTANT:")
    print("   • This system trades with REAL MONEY")
    print("   • Start with small amounts to test")
    print("   • Monitor performance regularly")
    print("   • System will create profit_system.log file")
    print("="*60)

def get_user_confirmation():
    """Get user confirmation to proceed"""
    print("\n🤔 Are you ready to start making autonomous profits?")
    print("   Type 'YES' to start the system")
    print("   Type 'NO' to cancel")
    
    while True:
        response = input("\nYour choice: ").strip().upper()
        if response == 'YES':
            return True
        elif response == 'NO':
            return False
        else:
            print("Please type 'YES' or 'NO'")

async def start_profit_system():
    """Start the autonomous profit system"""
    try:
        # Import the main system
        from kimera_autonomous_profit_system import KimeraAutonomousProfitSystem
        
        # Get starting capital
        print("\n💰 How much capital would you like to start with?")
        print("   Recommended: $50-100 for testing")
        
        while True:
            try:
                capital = float(input("Starting capital ($): "))
                if capital > 0:
                    break
                else:
                    print("Please enter a positive amount")
            except ValueError:
                print("Please enter a valid number")
        
        # Initialize system
        print(f"\n🔄 Initializing Kimera Profit System with ${capital:.2f}")
        system = KimeraAutonomousProfitSystem(starting_capital=capital)
        
        # Start the system
        print("🚀 Starting autonomous profit generation...")
        print("📊 Monitor progress in the terminal and profit_system.log file")
        print("🛑 Press Ctrl+C to stop the system")
        
        await system.start()
        
    except KeyboardInterrupt:
        print("\n🛑 System stopped by user")
        await system.stop()
    except Exception as e:
        print(f"\n❌ Error starting profit system: {e}")
        print("Check the log file for details")

def main():
    """Main entry point"""
    print(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    
    # Display startup information
    display_startup_info()
    
    # Get user confirmation
    if not get_user_confirmation():
        print("\n👋 Goodbye! Run this script again when you're ready to start making profits.")
        sys.exit(0)
    
    # Start the system
    print("\n🎯 Starting Kimera Autonomous Profit System...")
    try:
        asyncio.run(start_profit_system())
    except KeyboardInterrupt:
        print("\n👋 System stopped. See you next time!")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 