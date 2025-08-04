#!/usr/bin/env python3
"""
Kimera Profit System Setup
==========================

Setup script to install dependencies and configure the profit system.
"""

import os
import sys
import subprocess

def install_dependencies():
    """Install required dependencies"""
    logger.info("📦 Installing Dependencies...")
    
    dependencies = [
        'aiohttp',
        'numpy',
        'pandas',
        'scikit-learn',
        'requests'
    ]
    
    for dep in dependencies:
        try:
            logger.info(f"   Installing {dep}...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', dep], 
                         check=True, capture_output=True)
            logger.info(f"   ✅ {dep} installed successfully")
        except subprocess.CalledProcessError as e:
            logger.info(f"   ❌ Failed to install {dep}: {e}")
            return False
    
    return True

def setup_environment():
    """Setup environment variables"""
    logger.info("\n🔧 Setting Up Environment...")
    
    # Check if credentials are already set
    if os.getenv('BINANCE_API_KEY') and os.getenv('BINANCE_SECRET_KEY'):
        logger.info("   ✅ Binance credentials already configured")
        return True
    
    logger.info("   🔑 Binance API credentials needed")
    logger.info("   Go to https://binance.com/en/my/settings/api-management")
    logger.info("   Create new API key with trading permissions")
    logger.info("   Then set these environment variables:")
    logger.info()
    logger.info("   Windows:")
    logger.info("     set BINANCE_API_KEY=your_api_key_here")
    logger.info("     set BINANCE_SECRET_KEY=your_secret_key_here")
    logger.info()
    logger.info("   Linux/Mac:")
    logger.info("     export BINANCE_API_KEY=your_api_key_here")
    logger.info("     export BINANCE_SECRET_KEY=your_secret_key_here")
    logger.info()
    
    return False

def create_startup_files():
    """Create startup configuration files"""
    logger.info("\n📝 Creating Configuration Files...")
    
    # Create a simple config file
    config = {
        "system_name": "Kimera Autonomous Profit System",
        "version": "1.0.0",
        "default_capital": 50.0,
        "trading_symbols": ["BTCUSDT", "ETHUSDT"],
        "risk_management": {
            "max_risk_per_trade": 0.02,
            "max_total_risk": 0.2
        },
        "analysis_interval": 60,
        "profit_compounding": True
    }
    
    try:
        import json
import logging
logger = logging.getLogger(__name__)
        with open('kimera_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        logger.info("   ✅ Configuration file created")
    except Exception as e:
        logger.info(f"   ❌ Failed to create config file: {e}")
        return False
    
    return True

def display_next_steps():
    """Display next steps for the user"""
    logger.info("\n" + "="*60)
    logger.info("🎉 KIMERA PROFIT SYSTEM SETUP COMPLETE!")
    logger.info("="*60)
    logger.info()
    logger.info("📋 Next Steps:")
    logger.info("   1. Set up your Binance API credentials (if not done already)")
    logger.info("   2. Fund your Binance account with trading capital")
    logger.info("   3. Run: python start_kimera_profits.py")
    logger.info("   4. Monitor profits with: python check_kimera_profits.py")
    logger.info()
    logger.info("⚠️  Important Reminders:")
    logger.info("   • Start with small amounts for testing")
    logger.info("   • Monitor the system regularly")
    logger.info("   • Keep your API keys secure")
    logger.info("   • This system trades with real money")
    logger.info()
    logger.info("📚 Files Created:")
    logger.info("   • kimera_autonomous_profit_system.py - Main system")
    logger.info("   • start_kimera_profits.py - Easy startup script")
    logger.info("   • check_kimera_profits.py - Profit checker")
    logger.info("   • kimera_config.json - Configuration file")
    logger.info("   • setup_kimera_profits.py - This setup script")
    logger.info()
    logger.info("🚀 Ready to start making autonomous profits!")
    logger.info("="*60)

def main():
    """Main setup function"""
    logger.info("🚀 Kimera Autonomous Profit System Setup")
    logger.info("=" * 50)
    
    # Install dependencies
    if not install_dependencies():
        logger.info("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Setup environment
    credentials_ready = setup_environment()
    
    # Create configuration files
    if not create_startup_files():
        logger.info("❌ Failed to create configuration files")
        sys.exit(1)
    
    # Display next steps
    display_next_steps()
    
    if not credentials_ready:
        logger.info("\n⚠️  Please set up your Binance API credentials before running the system")
        logger.info("   Then run: python start_kimera_profits.py")
    else:
        logger.info("\n🎯 Setup complete! You can now run: python start_kimera_profits.py")

if __name__ == "__main__":
    main() 