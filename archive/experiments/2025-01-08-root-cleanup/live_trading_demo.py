#!/usr/bin/env python3
"""
KIMERA SWM LIVE TRADING DEMONSTRATION
Real-world trading demonstration with Binance integration
"""

import sys
import os
import asyncio
import traceback
from datetime import datetime

# Add backend to path
sys.path.append('backend')

from trading.kimera_fixed_trading_system import create_simplified_kimera_trading_system
import logging
logger = logging.getLogger(__name__)

async def main():
    logger.info("=" * 60)
    logger.info("KIMERA SWM LIVE TRADING DEMONSTRATION")
    logger.info("=" * 60)
    logger.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info()

    # Initialize the trading system
    logger.info("Initializing Kimera Trading System...")
    
    # Configuration for live trading
    config = {
        'trading_symbols': ['BTCUSDT', 'ETHUSDT'],
        'max_position_size': 25.0,
        'risk_percentage': 0.005,
        'loop_interval': 5,
        'confidence_threshold': 0.6
    }
    
    trading_system = create_simplified_kimera_trading_system(config)
    
    try:
        # Step 1: Check system status
        logger.info("Checking Kimera system status...")
        status = trading_system.get_status()
        logger.info(f"   System Status: {status.get('system_status', 'Unknown')}")
        logger.info(f"   Kimera Available: {status.get('kimera_available', False)}")
        logger.info(f"   Active Signals: {status.get('active_signals', 0)}")
        logger.info("SUCCESS: Kimera Trading Engine operational")
        logger.info()
        
        # Step 2: Start the trading engine
        logger.info("Starting Kimera Trading Engine...")
        await trading_system.start()
        
        logger.info("SUCCESS: KIMERA TRADING ENGINE STARTED!")
        logger.info()
        
        # Step 3: Monitor for a short period
        logger.info("Monitoring trading activity for 30 seconds...")
        monitoring_start = datetime.now()
        
        while (datetime.now() - monitoring_start).seconds < 30:
            # Check active signals
            current_status = trading_system.get_status()
            active_signals = current_status.get('active_signals', 0)
            
            if active_signals > 0:
                logger.info(f"   SIGNAL: Active signals detected: {active_signals}")
                
                # Show signal details
                for symbol, signal in trading_system.active_signals.items():
                    logger.info(f"   {symbol}: {signal.action.upper()} "
                          f"(confidence: {signal.confidence:.2%}, "
                          f"strategy: {signal.strategy.value})")
            
            await asyncio.sleep(5)
        
        logger.info()
        logger.info("Stopping monitoring...")
        await trading_system.stop()
        
        logger.info("SUCCESS: LIVE TRADING DEMONSTRATION COMPLETED!")
        
    except Exception as e:
        logger.info(f"ERROR: Critical Error: {e}")
        logger.info("\nFull Error Details:")
        traceback.print_exc()
        
    finally:
        logger.info()
        logger.info("=" * 60)
        logger.info(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("Kimera SWM Trading System - Mission Complete")
        logger.info("=" * 60)

if __name__ == "__main__":
    asyncio.run(main()) 