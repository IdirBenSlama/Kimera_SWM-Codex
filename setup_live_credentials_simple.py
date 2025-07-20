#!/usr/bin/env python3
"""
SIMPLIFIED CDP CREDENTIALS SETUP
================================

Quick setup for live CDP credentials.
"""

import os
from datetime import datetime

def create_live_config():
    """Create live configuration file"""
    print("🔐 KIMERA CDP LIVE CREDENTIALS SETUP")
    print("=" * 50)
    
    # Use your provided API key
    api_key_name = "9268de76-b5f4-4683-b593-327fb2c19503"
    
    print(f"✅ Using your API Key: {api_key_name}")
    print()
    print("📝 Enter your CDP Private Key:")
    print("(Get this from your CDP dashboard)")
    
    private_key = input("CDP Private Key: ").strip()
    
    if not private_key:
        print("❌ Private key is required!")
        return False
    
    print()
    print("🌐 Network Selection:")
    print("1. base-sepolia (Testnet - SAFE)")
    print("2. base-mainnet (Mainnet - REAL MONEY)")
    
    network_choice = input("Select network [1]: ").strip() or "1"
    
    if network_choice == "1":
        network_id = "base-sepolia"
        is_testnet = True
        print("✅ Selected: Testnet (Safe)")
    else:
        network_id = "base-mainnet"
        is_testnet = False
        print("⚠️  Selected: Mainnet (REAL MONEY)")
    
    # Create configuration
    config_content = f"""# KIMERA CDP LIVE CONFIGURATION
# Generated: {datetime.now().isoformat()}

# CDP API Credentials
CDP_API_KEY_NAME={api_key_name}
CDP_API_KEY_PRIVATE_KEY={private_key}

# Network Configuration
CDP_NETWORK_ID={network_id}
CDP_USE_TESTNET={str(is_testnet).lower()}

# Risk Management (Safe Defaults)
KIMERA_CDP_MAX_POSITION_SIZE=0.1
KIMERA_CDP_MIN_CONFIDENCE=0.7
KIMERA_CDP_MAX_DAILY_TRADES=50
KIMERA_CDP_MAX_DAILY_LOSS_USD=100.0
KIMERA_CDP_EMERGENCY_STOP_LOSS_USD=500.0
KIMERA_CDP_MIN_WALLET_BALANCE_USD=10.0

# Safety Settings
KIMERA_CDP_GAS_LIMIT=200000
KIMERA_CDP_MAX_SLIPPAGE=0.02
KIMERA_CDP_OPERATION_TIMEOUT=300
KIMERA_CDP_ENABLE_LOGGING=true
"""
    
    # Save configuration
    config_file = "kimera_cdp_live.env"
    
    try:
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        print(f"\n✅ Configuration saved to: {config_file}")
        
        # Test the configuration
        print("\n🔍 Testing configuration...")
        
        from dotenv import load_dotenv
        load_dotenv(config_file)
        
        test_key = os.getenv('CDP_API_KEY_NAME')
        test_network = os.getenv('CDP_NETWORK_ID')
        
        if test_key and test_network:
            print("✅ Configuration file valid")
            print(f"✅ API Key: {test_key}")
            print(f"✅ Network: {test_network}")
            print(f"✅ Safety: {'Testnet' if is_testnet else 'MAINNET'}")
        
        print("\n🚀 READY FOR LIVE TRADING!")
        print("Run: python kimera_cdp_live_integration.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving configuration: {e}")
        return False

if __name__ == "__main__":
    create_live_config() 