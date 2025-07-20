#!/usr/bin/env python3
"""
KIMERA AUTONOMOUS LAUNCHER
=========================

Verifies CDP setup and launches real money autonomous trading
"""

import os
import sys
import subprocess
from dotenv import load_dotenv

def verify_and_launch():
    """Verify setup and launch Kimera"""
    print("🚀 KIMERA AUTONOMOUS LAUNCHER")
    print("=" * 60)
    
    # Load environment
    load_dotenv('kimera_cdp_live.env')
    
    # Check credentials
    api_key = os.getenv('CDP_API_KEY_NAME')
    private_key = os.getenv('CDP_API_KEY_PRIVATE_KEY')
    
    if not api_key or not private_key:
        print("❌ CDP credentials not found!")
        return False
    
    print(f"✅ API Key: {api_key}")
    print(f"✅ Private Key: {'*' * 40}{private_key[-10:]}")
    
    # Check CDP SDK
    try:
        import cdp
        print("✅ CDP SDK available")
    except ImportError:
        print("❌ CDP SDK not installed")
        print("Installing CDP SDK...")
        subprocess.run([sys.executable, "-m", "pip", "install", "cdp-sdk"])
    
    # Final confirmation
    print("\n" + "⚠️ " * 10)
    print("REAL MONEY TRADING - KIMERA WILL HAVE FULL CONTROL")
    print("The system will autonomously trade with your real wallet")
    print("⚠️ " * 10)
    
    response = input("\nProceed with REAL MONEY autonomous trading? (yes/no): ")
    
    if response.lower() == 'yes':
        print("\n🔥 LAUNCHING KIMERA AUTONOMOUS MISSION...")
        subprocess.run([sys.executable, "kimera_autonomous_real_money.py"])
    else:
        print("❌ Launch cancelled")

if __name__ == "__main__":
    verify_and_launch() 