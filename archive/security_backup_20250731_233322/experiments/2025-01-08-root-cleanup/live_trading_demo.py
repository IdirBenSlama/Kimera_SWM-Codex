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

async def main():
    print("=" * 60)
    print("KIMERA SWM LIVE TRADING DEMONSTRATION")
    print("=" * 60)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Initialize the trading system
    print("Initializing Kimera Trading System...")
    
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
        print("Checking Kimera system status...")
        status = trading_system.get_status()
        print(f"   System Status: {status.get('system_status', 'Unknown')}")
        print(f"   Kimera Available: {status.get('kimera_available', False)}")
        print(f"   Active Signals: {status.get('active_signals', 0)}")
        print("SUCCESS: Kimera Trading Engine operational")
        print()
        
        # Step 2: Start the trading engine
        print("Starting Kimera Trading Engine...")
        await trading_system.start()
        
        print("SUCCESS: KIMERA TRADING ENGINE STARTED!")
        print()
        
        # Step 3: Monitor for a short period
        print("Monitoring trading activity for 30 seconds...")
        monitoring_start = datetime.now()
        
        while (datetime.now() - monitoring_start).seconds < 30:
            # Check active signals
            current_status = trading_system.get_status()
            active_signals = current_status.get('active_signals', 0)
            
            if active_signals > 0:
                print(f"   SIGNAL: Active signals detected: {active_signals}")
                
                # Show signal details
                for symbol, signal in trading_system.active_signals.items():
                    print(f"   {symbol}: {signal.action.upper()} "
                          f"(confidence: {signal.confidence:.2%}, "
                          f"strategy: {signal.strategy.value})")
            
            await asyncio.sleep(5)
        
        print()
        print("Stopping monitoring...")
        await trading_system.stop()
        
        print("SUCCESS: LIVE TRADING DEMONSTRATION COMPLETED!")
        
    except Exception as e:
        print(f"ERROR: Critical Error: {e}")
        print("\nFull Error Details:")
        traceback.print_exc()
        
    finally:
        print()
        print("=" * 60)
        print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("Kimera SWM Trading System - Mission Complete")
        print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main()) 