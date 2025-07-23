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
    print("📦 Installing Dependencies...")
    
    dependencies = [
        'aiohttp',
        'numpy',
        'pandas',
        'scikit-learn',
        'requests'
    ]
    
    for dep in dependencies:
        try:
            print(f"   Installing {dep}...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', dep], 
                         check=True, capture_output=True)
            print(f"   ✅ {dep} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed to install {dep}: {e}")
            return False
    
    return True

def setup_environment():
    """Setup environment variables"""
    print("\n🔧 Setting Up Environment...")
    
    # Check if credentials are already set
    if os.getenv('BINANCE_API_KEY') and os.getenv('BINANCE_SECRET_KEY'):
        print("   ✅ Binance credentials already configured")
        return True
    
    print("   🔑 Binance API credentials needed")
    print("   Go to https://binance.com/en/my/settings/api-management")
    print("   Create new API key with trading permissions")
    print("   Then set these environment variables:")
    print()
    print("   Windows:")
    print("     set BINANCE_API_KEY=your_api_key_here")
    print("     set BINANCE_SECRET_KEY=your_secret_key_here")
    print()
    print("   Linux/Mac:")
    print("     export BINANCE_API_KEY=your_api_key_here")
    print("     export BINANCE_SECRET_KEY=your_secret_key_here")
    print()
    
    return False

def create_startup_files():
    """Create startup configuration files"""
    print("\n📝 Creating Configuration Files...")
    
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
        with open('kimera_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        print("   ✅ Configuration file created")
    except Exception as e:
        print(f"   ❌ Failed to create config file: {e}")
        return False
    
    return True

def display_next_steps():
    """Display next steps for the user"""
    print("\n" + "="*60)
    print("🎉 KIMERA PROFIT SYSTEM SETUP COMPLETE!")
    print("="*60)
    print()
    print("📋 Next Steps:")
    print("   1. Set up your Binance API credentials (if not done already)")
    print("   2. Fund your Binance account with trading capital")
    print("   3. Run: python start_kimera_profits.py")
    print("   4. Monitor profits with: python check_kimera_profits.py")
    print()
    print("⚠️  Important Reminders:")
    print("   • Start with small amounts for testing")
    print("   • Monitor the system regularly")
    print("   • Keep your API keys secure")
    print("   • This system trades with real money")
    print()
    print("📚 Files Created:")
    print("   • kimera_autonomous_profit_system.py - Main system")
    print("   • start_kimera_profits.py - Easy startup script")
    print("   • check_kimera_profits.py - Profit checker")
    print("   • kimera_config.json - Configuration file")
    print("   • setup_kimera_profits.py - This setup script")
    print()
    print("🚀 Ready to start making autonomous profits!")
    print("="*60)

def main():
    """Main setup function"""
    print("🚀 Kimera Autonomous Profit System Setup")
    print("=" * 50)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Setup environment
    credentials_ready = setup_environment()
    
    # Create configuration files
    if not create_startup_files():
        print("❌ Failed to create configuration files")
        sys.exit(1)
    
    # Display next steps
    display_next_steps()
    
    if not credentials_ready:
        print("\n⚠️  Please set up your Binance API credentials before running the system")
        print("   Then run: python start_kimera_profits.py")
    else:
        print("\n🎯 Setup complete! You can now run: python start_kimera_profits.py")

if __name__ == "__main__":
    main() 