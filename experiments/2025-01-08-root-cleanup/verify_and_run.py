#!/usr/bin/env python3
"""
VERIFY CREDENTIALS AND RUN AUTONOMOUS TRADING
=============================================

This script verifies your CDP credentials and launches autonomous trading.
"""

import os
import sys
from pathlib import Path

def verify_credentials():
    """Verify CDP credentials are configured"""
    print("🔍 VERIFYING CDP CREDENTIALS...")
    
    config_file = "kimera_cdp_live.env"
    
    if not Path(config_file).exists():
        print("❌ Configuration file not found!")
        print("Please ensure kimera_cdp_live.env exists")
        return False
    
    try:
        from dotenv import load_dotenv
        load_dotenv(config_file)
        
        api_key_name = os.getenv('CDP_API_KEY_NAME')
        private_key = os.getenv('CDP_API_KEY_PRIVATE_KEY')
        network_id = os.getenv('CDP_NETWORK_ID')
        
        print(f"✅ API Key Name: {api_key_name}")
        
        if not private_key or private_key == "YOUR_CDP_PRIVATE_KEY_HERE":
            print("❌ Private key not configured!")
            print("Please edit kimera_cdp_live.env and add your CDP private key")
            return False
        
        print(f"✅ Private Key: ***{private_key[-10:] if len(private_key) > 10 else '***'}")
        print(f"✅ Network: {network_id}")
        
        # Test CDP SDK
        try:
            from cdp import CdpClient
            print("✅ CDP SDK available")
        except ImportError:
            print("❌ CDP SDK not available. Run: pip install cdp-sdk")
            return False
        
        print("✅ Credentials verified successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Verification error: {e}")
        return False

def launch_autonomous_trading():
    """Launch the autonomous trading system"""
    print("\n🚀 LAUNCHING KIMERA AUTONOMOUS TRADING...")
    print("⚠️  Kimera will now have autonomous control of your wallet")
    print("⚠️  Starting with testnet for safety")
    print()
    
    try:
        # Import and run the live integration
        import asyncio
        from kimera_cdp_live_integration import main as live_trading_main
        
        print("🎯 Starting autonomous trading session...")
        asyncio.run(live_trading_main())
        
    except Exception as e:
        print(f"❌ Launch error: {e}")
        print("Please check the logs for details")

def main():
    """Main function"""
    print("🔐 KIMERA CDP AUTONOMOUS TRADING LAUNCHER")
    print("=" * 50)
    
    # Step 1: Verify credentials
    if not verify_credentials():
        print("\n❌ CREDENTIAL VERIFICATION FAILED")
        print("Please configure your credentials before proceeding.")
        print()
        print("📝 INSTRUCTIONS:")
        print("1. Edit kimera_cdp_live.env")
        print("2. Replace YOUR_CDP_PRIVATE_KEY_HERE with your actual CDP private key")
        print("3. Run this script again")
        return
    
    # Step 2: Launch autonomous trading
    print("\n✅ CREDENTIALS VERIFIED")
    
    proceed = input("\n🚀 Launch autonomous trading? (yes/no): ").strip().lower()
    
    if proceed == 'yes':
        launch_autonomous_trading()
    else:
        print("Launch cancelled.")

if __name__ == "__main__":
    main() 