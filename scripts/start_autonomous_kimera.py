#!/usr/bin/env python3
"""
KIMERA AUTONOMOUS TRADER LAUNCHER
=================================

Launch Kimera with full autonomous trading capabilities.
NO SAFETY LIMITS - PURE COGNITIVE INTELLIGENCE.

Mission: Grow €5 to €100 using advanced AI decision-making.
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
    format='%(asctime)s - KIMERA-LAUNCHER - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/kimera_autonomous_launcher.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('KIMERA_LAUNCHER')

def display_banner():
    """Display Kimera autonomous trader banner"""
    banner = """
================================================================
                    KIMERA AUTONOMOUS TRADER                  
                                                              
    PURE COGNITIVE INTELLIGENCE - NO SAFETY LIMITS         
                                                              
    Mission: EUR 5 -> EUR 100                                     
    Mode: FULLY AUTONOMOUS                                    
    Constraints: NONE                                         
    Decision Making: ADVANCED AI                              
                                                              
    WARNING: USER ACCEPTS ALL RISKS                         
                                                              
================================================================
    """
    print(banner)

def check_prerequisites():
    """Check system prerequisites"""
    logger.info("Checking prerequisites...")
    
    # Create required directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Check Python packages
    required_packages = [
        'numpy', 'pandas', 'scikit-learn', 'scipy', 'requests'
    ]
    
    missing_packages = []
    package_import_map = {
        'scikit-learn': 'sklearn',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'scipy': 'scipy',
        'requests': 'requests'
    }
    
    for package in required_packages:
        import_name = package_import_map.get(package, package)
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing packages: {missing_packages}")
        logger.info("Install with: pip install " + " ".join(missing_packages))
        return False
    
    logger.info("All prerequisites satisfied")
    return True

async def main():
    """Main launcher function"""
    display_banner()
    
    if not check_prerequisites():
        return
    
    # Configuration
    API_KEY = os.getenv("CDP_API_KEY_NAME", "")
    TARGET_EUR = 100.0
    CYCLE_INTERVAL_MINUTES = 15  # Trade cycle every 15 minutes
    
    logger.info("INITIALIZING KIMERA AUTONOMOUS TRADER")
    logger.info(f"   API Key: {API_KEY[:8]}...")
    logger.info(f"   Target: EUR {TARGET_EUR}")
    logger.info(f"   Cycle Interval: {CYCLE_INTERVAL_MINUTES} minutes")
    logger.info(f"   Start Time: {datetime.now()}")
    
    try:
        # Create autonomous trader
        trader = create_autonomous_kimera(API_KEY, TARGET_EUR)
        
        # Display initial status
        status = await trader.get_portfolio_status()
        logger.info("INITIAL STATUS:")
        logger.info(f"   Portfolio Value: EUR {status['portfolio_value_eur']:.2f}")
        logger.info(f"   Target: EUR {status['target_eur']}")
        logger.info(f"   Growth Required: {((status['target_eur']/status['portfolio_value_eur'])-1)*100:.0f}%")
        
        # Confirm autonomous trading
        print("\n" + "="*60)
        print("KIMERA AUTONOMOUS TRADING - FINAL CONFIRMATION")
        print("="*60)
        print(f"Portfolio: EUR {status['portfolio_value_eur']:.2f}")
        print(f"Target: EUR {status['target_eur']}")
        print(f"Mode: FULLY AUTONOMOUS")
        print(f"Safety Limits: NONE")
        print(f"Decision Making: AI-CONTROLLED")
        print(f"Risk Level: USER ACCEPTED")
        print("="*60)
        
        confirm = input("\nType 'UNLEASH KIMERA' to start autonomous trading: ")
        
        if confirm.upper() == 'UNLEASH KIMERA':
            logger.info("USER CONFIRMED - UNLEASHING KIMERA")
            print("\nKIMERA AUTONOMOUS TRADER ACTIVE")
            print("   Monitor progress in logs/kimera_autonomous_launcher.log")
            print("   Press Ctrl+C to stop (will complete current cycle)")
            
            # Start autonomous trading
            await trader.run_autonomous_trader(CYCLE_INTERVAL_MINUTES)
            
        else:
            logger.info("User cancelled autonomous trading")
            print("Autonomous trading cancelled.")
    
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
        print(f"Launch failed: {e}") 