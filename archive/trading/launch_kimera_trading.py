#!/usr/bin/env python3
"""
KIMERA TRADING LAUNCHER
======================
Automatically detects available credentials and launches the best trading engine
"""

import os
import sys
import asyncio
from dotenv import load_dotenv

def check_credentials():
    """Check which credentials are available"""
    
    # Check for Advanced Trade API
    load_dotenv('.env')
    advanced_key = os.getenv('COINBASE_ADVANCED_API_KEY')
    advanced_secret = os.getenv('COINBASE_ADVANCED_API_SECRET')
    
    # Check for CDP API
    load_dotenv('kimera_cdp_live.env')
    cdp_key = os.getenv('CDP_API_KEY_NAME')
    cdp_secret = os.getenv('CDP_API_KEY_PRIVATE_KEY')
    
    return {
        'advanced': bool(advanced_key and advanced_secret),
        'cdp': bool(cdp_key and cdp_secret)
    }

def main():
    print("\n🚀 KIMERA TRADING SYSTEM LAUNCHER")
    print("="*50)
    
    # Check available credentials
    creds = check_credentials()
    
    print("\n🔍 Credential Check:")
    print(f"   Advanced Trade API: {'✅ Available' if creds['advanced'] else '❌ Not found'}")
    print(f"   CDP API: {'✅ Available' if creds['cdp'] else '❌ Not found'}")
    
    # Determine which engine to run
    if creds['advanced']:
        print("\n✅ Using Advanced Trade API (Full Trading Capabilities)")
        print("🚀 Launching Full Autonomy Trading Engine...")
        
        # Import and run the full autonomy trader
        from kimera_full_autonomy_trader import main as run_full_autonomy
        asyncio.run(run_full_autonomy())
        
    elif creds['cdp']:
        print("\n⚠️  Using CDP API (Limited Trading - Simulation Mode)")
        print("🚀 Launching CDP-Compatible Trading Engine...")
        
        # Import and run CDP-compatible trader
        from kimera_cdp_compatible_trading import main as run_cdp_trading
        asyncio.run(run_cdp_trading())
        
    else:
        print("\n❌ No API credentials found!")
        print("\nTo get started:")
        print("1. For full trading: Create Advanced Trade API keys")
        print("   Visit: https://www.coinbase.com/settings/api")
        print("   Add to .env file:")
        print("   COINBASE_ADVANCED_API_KEY=your_key")
        print("   COINBASE_ADVANCED_API_SECRET=your_secret")
        print("\n2. Or use your existing CDP credentials")
        print("   Already in: kimera_cdp_live.env")
        
        # Offer to run simulation
        print("\n💡 Would you like to run a performance simulation instead?")
        response = input("Run simulation? (y/n): ").lower()
        
        if response == 'y':
            print("\n🎮 Launching Performance Simulation...")
            from kimera_performance_simulation import main as run_simulation
            asyncio.run(run_simulation())

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Trading stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc() 