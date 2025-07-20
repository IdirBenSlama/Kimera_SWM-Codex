"""
KIMERA Full System Startup Script
Initializes all components for the 24-hour trading simulation
"""

import asyncio
import sys
import os
import subprocess
import time
from datetime import datetime

# Initialize structured logger
from backend.utils.kimera_logger import get_system_logger
logger = get_system_logger(__name__)


# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'aiohttp', 'yfinance', 'ccxt', 'dash', 'plotly', 
        'pandas', 'numpy', 'python-dotenv'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"❌ Missing packages: {', '.join(missing_packages)
        logger.info("Installing missing packages...")
        for package in missing_packages:
            subprocess.run([sys.executable, '-m', 'pip', 'install', package])
    else:
        logger.info("✅ All dependencies satisfied")

def display_banner():
    """Display KIMERA startup banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║    ██╗  ██╗██╗███╗   ███╗███████╗██████╗  █████╗             ║
    ║    ██║ ██╔╝██║████╗ ████║██╔════╝██╔══██╗██╔══██╗            ║
    ║    █████╔╝ ██║██╔████╔██║█████╗  ██████╔╝███████║            ║
    ║    ██╔═██╗ ██║██║╚██╔╝██║██╔══╝  ██╔══██╗██╔══██║            ║
    ║    ██║  ██╗██║██║ ╚═╝ ██║███████╗██║  ██║██║  ██║            ║
    ║    ╚═╝  ╚═╝╚═╝╚═╝     ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝            ║
    ║                                                              ║
    ║         Semantic Trading System - 24 Hour Simulation        ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    
    🚀 Initializing KIMERA Semantic Trading System
    💰 Starting Capital: $1.00
    🎯 Goal: Maximum growth through semantic contradiction detection
    ⏰ Duration: 24 hours (or accelerated demo)
    
    """
    logger.info(banner)

def show_system_info():
    """Display system information"""
    logger.info("📊 SYSTEM CONFIGURATION:")
    logger.info(f"   🐍 Python Version: {sys.version.split()
    logger.info(f"   📁 Project Root: {project_root}")
    logger.info(f"   ⏰ Current Time: {datetime.now()
    logger.info()
    
    logger.debug("🔧 KIMERA COMPONENTS:")
    logger.info("   ✅ Semantic Trading Reactor")
    logger.info("   ✅ CryptoPanic News Connector")
    logger.info("   ✅ Yahoo Finance Market Data")
    logger.info("   ✅ Contradiction Detection Engine")
    logger.info("   ✅ Risk Management System")
    logger.info("   ✅ Performance Monitoring")
    logger.info()

async def test_system_connections():
    """Test all system connections before starting"""
    logger.debug("🔍 TESTING SYSTEM CONNECTIONS:")
    
    # Test CryptoPanic API
    try:
        sys.path.append(os.path.join(project_root, 'backend', 'trading'))
        from connectors.cryptopanic_connector import CryptoPanicConnector
        
        async with CryptoPanicConnector() as connector:
            news = await connector.get_posts()
            logger.info(f"   ✅ CryptoPanic API: {len(news)
    except Exception as e:
        logger.error(f"   ❌ CryptoPanic API: {str(e)
        
    # Test Yahoo Finance
    try:
        from connectors.data_providers import YahooFinanceConnector
        market_connector = YahooFinanceConnector()
        btc_price = await market_connector.get_current_price('BTC-USD')
        logger.info(f"   ✅ Yahoo Finance: BTC-USD @ ${btc_price:.2f}")
    except Exception as e:
        logger.error(f"   ❌ Yahoo Finance: {str(e)
        
    logger.info()

def show_trading_parameters():
    """Display trading parameters"""
    logger.info("⚙️  TRADING PARAMETERS:")
    logger.info("   💰 Starting Capital: $1.00")
    logger.info("   🎯 Risk per Trade: 5%")
    logger.info("   📊 Max Position Size: 30%")
    logger.info("   🛑 Stop Loss: 2%")
    logger.info("   💹 Take Profit: 6%")
    logger.info("   📈 Watchlist: BTC, ETH, SOL, BNB, ADA, DOT, AVAX, MATIC, LINK, UNI")
    logger.info()

def show_simulation_options():
    """Show available simulation options"""
    logger.info("🎮 SIMULATION OPTIONS:")
    logger.info("   1. Full 24-Hour Simulation (96 cycles, 15-minute intervals)
    logger.info("   2. Accelerated Demo (30 minutes, 2-minute intervals)
    logger.info("   3. Quick Test (5 cycles, 30-second intervals)
    logger.info("   4. System Test Only (verify connections)
    logger.info()

async def main():
    """Main startup sequence"""
    display_banner()
    
    # Check dependencies
    logger.debug("🔧 Checking dependencies...")
    check_dependencies()
    logger.info()
    
    # Show system info
    show_system_info()
    
    # Test connections
    await test_system_connections()
    
    # Show parameters
    show_trading_parameters()
    
    # Show options
    show_simulation_options()
    
    # Get user choice
    choice = input("Select simulation mode (1-4): ").strip()
    
    if choice == '4':
        logger.info("✅ System test complete!")
        return
    
    # Start the simulation
    logger.info("🚀 Starting KIMERA Trading Simulation...")
    logger.info("=" * 60)
    
    # Import and run the simulation
    simulation_path = os.path.join(project_root, 'backend', 'trading', 'examples', 'kimera_24h_simulation.py')
    
    if choice == '1':
        logger.info("⏰ Full 24-hour simulation starting...")
        subprocess.run([sys.executable, simulation_path], input='1\n', text=True)
    elif choice == '2':
        logger.info("⚡ Accelerated demo starting...")
        subprocess.run([sys.executable, simulation_path], input='2\n', text=True)
    elif choice == '3':
        logger.debug("🔬 Quick test starting...")
        subprocess.run([sys.executable, simulation_path], input='3\n', text=True)
    else:
        logger.info("❓ Invalid choice, running quick test...")
        subprocess.run([sys.executable, simulation_path], input='3\n', text=True)

if __name__ == "__main__":
    asyncio.run(main()) 