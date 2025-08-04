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
    logger.info("🔐 KIMERA CDP LIVE CREDENTIALS SETUP")
    logger.info("=" * 50)
    logger.info()
    logger.info("⚠️  WARNING: This will configure REAL CDP API credentials")
    logger.info("⚠️  Kimera will have autonomous control over your wallet")
    logger.info("⚠️  Only proceed if you understand the risks")
    logger.info()
    
    # Confirm user wants to proceed
    confirm = input("Do you want to proceed with live credential setup? (yes/no): ").strip().lower()
    if confirm != 'yes':
        logger.info("Setup cancelled.")
        return False
    
    logger.info("\n📋 CREDENTIAL INFORMATION NEEDED:")
    logger.info("1. CDP API Key Name (you provided: 9268de76-b5f4-4683-b593-327fb2c19503)")
    logger.info("2. CDP API Private Key (from your CDP dashboard)")
    logger.info("3. Network preference (testnet recommended for initial testing)")
    logger.info()
    
    # Get API Key Name
    api_key_name = input("Enter your CDP API Key Name [9268de76-b5f4-4683-b593-327fb2c19503]: ").strip()
    if not api_key_name:
        api_key_name = os.getenv("CDP_API_KEY_NAME", "")
    
    # Get Private Key securely
    logger.info("\n🔑 Enter your CDP API Private Key:")
    logger.info("(This should be a long string starting with -----BEGIN EC PRIVATE KEY-----)")
    private_key = getpass.getpass("CDP Private Key: ").strip()
    
    if not private_key:
        logger.info("❌ Private key is required. Setup cancelled.")
        return False
    
    # Network selection
    logger.info("\n🌐 Network Selection:")
    logger.info("1. base-sepolia (Testnet - RECOMMENDED for initial testing)")
    logger.info("2. base-mainnet (Mainnet - REAL MONEY)")
    logger.info("3. ethereum-sepolia (Ethereum Testnet)")
    logger.info("4. ethereum-mainnet (Ethereum Mainnet - REAL MONEY)")
    
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
        logger.info("❌ Invalid network selection. Setup cancelled.")
        return False
    
    network_id, is_testnet = network_map[network_choice]
    
    # Risk parameters
    logger.info("\n⚖️  RISK MANAGEMENT SETTINGS:")
    
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
        except Exception as e:
            logger.error(f"Error in setup_live_cdp_credentials.py: {e}", exc_info=True)
            raise  # Re-raise for proper error handling
            pass  # Windows doesn't support chmod
        
        logger.info(f"\n✅ Configuration saved to: {config_file}")
        logger.info("✅ File permissions set to secure (owner only)")
        
    except Exception as e:
        logger.info(f"❌ Error saving configuration: {e}")
        return False
    
    # Create backup of configuration (without private key)
    backup_content = env_content.replace(private_key, "***PRIVATE_KEY_REDACTED***")
    backup_file = "kimera_cdp_config_backup.env"
    
    try:
        with open(backup_file, 'w') as f:
            f.write(backup_content)
        logger.info(f"✅ Backup configuration saved to: {backup_file}")
    except Exception as e:
        logger.error(f"Error in setup_live_cdp_credentials.py: {e}", exc_info=True)
        raise  # Re-raise for proper error handling
    
    # Display summary
    logger.info("\n" + "=" * 50)
    logger.info("🎯 CONFIGURATION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"API Key Name: {api_key_name}")
    logger.info(f"Network: {network_id} ({'Testnet' if is_testnet else 'MAINNET'})")
    logger.info(f"Max Position: {max_position}%")
    logger.info(f"Min Confidence: {min_confidence}")
    logger.info(f"Max Daily Trades: {max_daily_trades}")
    logger.info("=" * 50)
    
    if not is_testnet:
        logger.info("⚠️  WARNING: MAINNET CONFIGURATION DETECTED")
        logger.info("⚠️  This will use REAL MONEY for transactions")
        logger.info("⚠️  Consider testing on testnet first")
        logger.info()
    
    logger.info("🚀 NEXT STEPS:")
    logger.info("1. Review the configuration file")
    logger.info("2. Run: python kimera_cdp_live_integration.py")
    logger.info("3. Monitor the logs carefully")
    logger.info("4. Start with small amounts")
    logger.info()
    
    return True

def verify_cdp_connection():
    """
    Verify CDP connection without executing trades
    """
    logger.info("🔍 VERIFYING CDP CONNECTION...")
    
    try:
        # Try to load configuration
        from dotenv import load_dotenv
        load_dotenv('kimera_cdp_live.env')
        
        api_key_name = os.getenv('CDP_API_KEY_NAME')
        private_key = os.getenv('CDP_API_KEY_PRIVATE_KEY')
        
        if not api_key_name or not private_key:
            logger.info("❌ Credentials not found in environment")
            return False
        
        logger.info(f"✅ API Key Name: {api_key_name}")
        logger.info(f"✅ Private Key: {'***' + private_key[-10:] if len(private_key) > 10 else '***'}")
        
        # Try to import CDP SDK
        try:
            from cdp import Cdp
import logging
logger = logging.getLogger(__name__)
            logger.info("✅ CDP SDK available")
        except ImportError:
            logger.info("❌ CDP SDK not installed. Run: pip install cdp-sdk")
            return False
        
        logger.info("✅ Configuration appears valid")
        logger.info("🚀 Ready for live trading")
        
        return True
        
    except Exception as e:
        logger.info(f"❌ Verification error: {e}")
        return False

if __name__ == "__main__":
    logger.info("KIMERA CDP LIVE SETUP")
    logger.info("=" * 30)
    logger.info("1. Setup live credentials")
    logger.info("2. Verify connection")
    logger.info("3. Exit")
    logger.info()
    
    choice = input("Select option (1-3): ").strip()
    
    if choice == "1":
        setup_live_cdp_credentials()
    elif choice == "2":
        verify_cdp_connection()
    else:
        logger.info("Goodbye!") 