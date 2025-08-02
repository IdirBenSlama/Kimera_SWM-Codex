#!/usr/bin/env python3
"""
KIMERA AUTONOMOUS TRADER - DIRECT LAUNCH
========================================

Direct launch of Kimera autonomous trader without subprocess or confirmation.
"""

import os
import sys
import asyncio
import logging
from datetime import datetime

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from src.trading.autonomous_kimera_trader import create_autonomous_kimera

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - KIMERA-DIRECT - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/kimera_direct_launch.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('KIMERA_DIRECT')

def check_prerequisites():
    """Check system prerequisites"""
    logger.info("Checking prerequisites...")
    
    # Create required directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Check Python packages
    required_packages = ['numpy', 'pandas', 'scikit-learn', 'scipy', 'requests']
    package_import_map = {
        'scikit-learn': 'sklearn',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'scipy': 'scipy',
        'requests': 'requests'
    }
    
    missing_packages = []
    for package in required_packages:
        import_name = package_import_map.get(package, package)
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing packages: {missing_packages}")
        return False
    
    logger.info("All prerequisites satisfied")
    return True

async def main():
    """Main launcher function"""
    print("================================================================")
    print("                KIMERA AUTONOMOUS TRADER - DIRECT LAUNCH")
    print("================================================================")
    
    if not check_prerequisites():
        return
    
    # Configuration
    API_KEY = "9268de76-b5f4-4683-b593-327fb2c19503"
    TARGET_EUR = 100.0
    CYCLE_INTERVAL_MINUTES = 15
    
    logger.info("INITIALIZING KIMERA AUTONOMOUS TRADER")
    logger.info(f"   API Key: {API_KEY[:8]}...")
    logger.info(f"   Target: EUR {TARGET_EUR}")
    logger.info(f"   Cycle Interval: {CYCLE_INTERVAL_MINUTES} minutes")
    
    try:
        # Create autonomous trader
        trader = create_autonomous_kimera(API_KEY, TARGET_EUR)
        
        # Display initial status
        status = await trader.get_portfolio_status()
        logger.info("INITIAL STATUS:")
        logger.info(f"   Portfolio Value: EUR {status['portfolio_value_eur']:.2f}")
        logger.info(f"   Target: EUR {status['target_eur']}")
        logger.info(f"   Growth Required: {((status['target_eur']/status['portfolio_value_eur'])-1)*100:.0f}%")
        
        # Display confirmation
        print("\n" + "="*60)
        print("KIMERA AUTONOMOUS TRADING - DIRECT LAUNCH")
        print("="*60)
        print(f"Portfolio: EUR {status['portfolio_value_eur']:.2f}")
        print(f"Target: EUR {status['target_eur']}")
        print(f"Mode: FULLY AUTONOMOUS")
        print(f"Safety Limits: NONE")
        print(f"Decision Making: AI-CONTROLLED")
        print(f"Risk Level: USER ACCEPTED")
        print("="*60)
        
        # Skip confirmation - direct launch
        logger.info("DIRECT LAUNCH - UNLEASHING KIMERA")
        print("\nKIMERA AUTONOMOUS TRADER ACTIVE")
        print("   Monitor progress in logs/kimera_direct_launch.log")
        print("   Press Ctrl+C to stop")
        
        # Start autonomous trading
        await trader.run_autonomous_trader(CYCLE_INTERVAL_MINUTES)
        
    except KeyboardInterrupt:
        logger.info("Autonomous trading stopped by user")
        print("\nKimera autonomous trading stopped")
    
    except Exception as e:
        logger.error(f"Autonomous trader failed: {e}")
        print(f"\nError: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"Direct launch failed: {e}")
        import traceback
        traceback.print_exc() 