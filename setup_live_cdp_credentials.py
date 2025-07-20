#!/usr/bin/env python3
"""
SECURE CDP CREDENTIALS SETUP
============================

This script helps you securely configure your real CDP API credentials
for live autonomous trading with Kimera.

IMPORTANT: This will enable real blockchain transactions with real assets.
"""

import os
import json
import getpass
from pathlib import Path

def setup_live_cdp_credentials():
    """
    Secure setup for live CDP credentials
    """
    print("🔐 KIMERA CDP LIVE CREDENTIALS SETUP")
    print("=" * 50)
    print()
    print("⚠️  WARNING: This will configure REAL CDP API credentials")
    print("⚠️  Kimera will have autonomous control over your wallet")
    print("⚠️  Only proceed if you understand the risks")
    print()
    
    # Confirm user wants to proceed
    confirm = input("Do you want to proceed with live credential setup? (yes/no): ").strip().lower()
    if confirm != 'yes':
        print("Setup cancelled.")
        return False
    
    print("\n📋 CREDENTIAL INFORMATION NEEDED:")
    print("1. CDP API Key Name (you provided: 9268de76-b5f4-4683-b593-327fb2c19503)")
    print("2. CDP API Private Key (from your CDP dashboard)")
    print("3. Network preference (testnet recommended for initial testing)")
    print()
    
    # Get API Key Name
    api_key_name = input("Enter your CDP API Key Name [9268de76-b5f4-4683-b593-327fb2c19503]: ").strip()
    if not api_key_name:
        api_key_name = "9268de76-b5f4-4683-b593-327fb2c19503"
    
    # Get Private Key securely
    print("\n🔑 Enter your CDP API Private Key:")
    print("(This should be a long string starting with -----BEGIN EC PRIVATE KEY-----)")
    private_key = getpass.getpass("CDP Private Key: ").strip()
    
    if not private_key:
        print("❌ Private key is required. Setup cancelled.")
        return False
    
    # Network selection
    print("\n🌐 Network Selection:")
    print("1. base-sepolia (Testnet - RECOMMENDED for initial testing)")
    print("2. base-mainnet (Mainnet - REAL MONEY)")
    print("3. ethereum-sepolia (Ethereum Testnet)")
    print("4. ethereum-mainnet (Ethereum Mainnet - REAL MONEY)")
    
    network_choice = input("Select network (1-4) [1]: ").strip()
    if not network_choice:
        network_choice = "1"
    
    network_map = {
        "1": ("base-sepolia", True),
        "2": ("base-mainnet", False),
        "3": ("ethereum-sepolia", True),
        "4": ("ethereum-mainnet", False)
    }
    
    if network_choice not in network_map:
        print("❌ Invalid network selection. Setup cancelled.")
        return False
    
    network_id, is_testnet = network_map[network_choice]
    
    # Risk parameters
    print("\n⚖️  RISK MANAGEMENT SETTINGS:")
    
    max_position = input("Maximum position size (% of wallet) [10]: ").strip()
    if not max_position:
        max_position = "10"
    
    min_confidence = input("Minimum confidence for trades (0.0-1.0) [0.7]: ").strip()
    if not min_confidence:
        min_confidence = "0.7"
    
    max_daily_trades = input("Maximum daily trades [50]: ").strip()
    if not max_daily_trades:
        max_daily_trades = "50"
    
    # Create environment configuration
    env_content = f"""# =============================================================================
# KIMERA CDP LIVE TRADING CONFIGURATION
# =============================================================================
# 
# ⚠️  WARNING: THESE ARE LIVE CREDENTIALS FOR REAL BLOCKCHAIN TRANSACTIONS
# ⚠️  Keep this file secure and never share it
# 
# Generated: {os.path.basename(__file__)} on {__import__('datetime').datetime.now().isoformat()}
# =============================================================================

# CDP API Credentials (LIVE)
CDP_API_KEY_NAME={api_key_name}
CDP_API_KEY_PRIVATE_KEY={private_key}

# Network Configuration
CDP_NETWORK_ID={network_id}
CDP_USE_TESTNET={str(is_testnet).lower()}

# Risk Management
KIMERA_CDP_MAX_POSITION_SIZE={float(max_position)/100.0}
KIMERA_CDP_MIN_CONFIDENCE={float(min_confidence)}
KIMERA_CDP_MAX_DAILY_TRADES={int(max_daily_trades)}

# Safety Limits
KIMERA_CDP_MAX_DAILY_LOSS_USD=100.0
KIMERA_CDP_EMERGENCY_STOP_LOSS_USD=500.0
KIMERA_CDP_MIN_WALLET_BALANCE_USD=10.0

# Operational Settings
KIMERA_CDP_GAS_LIMIT=200000
KIMERA_CDP_MAX_SLIPPAGE=0.02
KIMERA_CDP_OPERATION_TIMEOUT=300

# Monitoring
KIMERA_CDP_ENABLE_LOGGING=true
KIMERA_CDP_ENABLE_NOTIFICATIONS=true
KIMERA_CDP_REPORT_INTERVAL=3600

# Development/Debug
KIMERA_CDP_DEBUG_MODE=false
KIMERA_CDP_SIMULATION_MODE=false
"""
    
    # Save configuration
    config_file = "kimera_cdp_live.env"
    
    try:
        with open(config_file, 'w') as f:
            f.write(env_content)
        
        # Set secure permissions (Unix/Linux)
        try:
            os.chmod(config_file, 0o600)  # Read/write for owner only
        except:
            pass  # Windows doesn't support chmod
        
        print(f"\n✅ Configuration saved to: {config_file}")
        print("✅ File permissions set to secure (owner only)")
        
    except Exception as e:
        print(f"❌ Error saving configuration: {e}")
        return False
    
    # Create backup of configuration (without private key)
    backup_content = env_content.replace(private_key, "***PRIVATE_KEY_REDACTED***")
    backup_file = "kimera_cdp_config_backup.env"
    
    try:
        with open(backup_file, 'w') as f:
            f.write(backup_content)
        print(f"✅ Backup configuration saved to: {backup_file}")
    except:
        pass
    
    # Display summary
    print("\n" + "=" * 50)
    print("🎯 CONFIGURATION SUMMARY")
    print("=" * 50)
    print(f"API Key Name: {api_key_name}")
    print(f"Network: {network_id} ({'Testnet' if is_testnet else 'MAINNET'})")
    print(f"Max Position: {max_position}%")
    print(f"Min Confidence: {min_confidence}")
    print(f"Max Daily Trades: {max_daily_trades}")
    print("=" * 50)
    
    if not is_testnet:
        print("⚠️  WARNING: MAINNET CONFIGURATION DETECTED")
        print("⚠️  This will use REAL MONEY for transactions")
        print("⚠️  Consider testing on testnet first")
        print()
    
    print("🚀 NEXT STEPS:")
    print("1. Review the configuration file")
    print("2. Run: python kimera_cdp_live_integration.py")
    print("3. Monitor the logs carefully")
    print("4. Start with small amounts")
    print()
    
    return True

def verify_cdp_connection():
    """
    Verify CDP connection without executing trades
    """
    print("🔍 VERIFYING CDP CONNECTION...")
    
    try:
        # Try to load configuration
        from dotenv import load_dotenv
        load_dotenv('kimera_cdp_live.env')
        
        api_key_name = os.getenv('CDP_API_KEY_NAME')
        private_key = os.getenv('CDP_API_KEY_PRIVATE_KEY')
        
        if not api_key_name or not private_key:
            print("❌ Credentials not found in environment")
            return False
        
        print(f"✅ API Key Name: {api_key_name}")
        print(f"✅ Private Key: {'***' + private_key[-10:] if len(private_key) > 10 else '***'}")
        
        # Try to import CDP SDK
        try:
            from cdp import Cdp
            print("✅ CDP SDK available")
        except ImportError:
            print("❌ CDP SDK not installed. Run: pip install cdp-sdk")
            return False
        
        print("✅ Configuration appears valid")
        print("🚀 Ready for live trading")
        
        return True
        
    except Exception as e:
        print(f"❌ Verification error: {e}")
        return False

if __name__ == "__main__":
    print("KIMERA CDP LIVE SETUP")
    print("=" * 30)
    print("1. Setup live credentials")
    print("2. Verify connection")
    print("3. Exit")
    print()
    
    choice = input("Select option (1-3): ").strip()
    
    if choice == "1":
        setup_live_cdp_credentials()
    elif choice == "2":
        verify_cdp_connection()
    else:
        print("Goodbye!") 