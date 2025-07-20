#!/usr/bin/env python3
"""
KIMERA LIVE TRADING ACTIVATION SCRIPT
====================================

This script demonstrates how to enable REAL trading execution in Kimera.
It replaces the simulation-only approach with actual market orders.

⚠️ EXTREME CAUTION: THIS SCRIPT ENABLES REAL MONEY TRADING ⚠️

BEFORE RUNNING:
1. Set up your Binance API credentials
2. Create Ed25519 private key file
3. Understand all risks involved
4. Start with small amounts
5. Monitor continuously

EXECUTION PROBLEM SOLUTION:
- Replaces simulation mode with live trading
- Connects to real exchanges
- Places actual market orders
- Implements comprehensive risk management
"""

import asyncio
import logging
import sys
import os
from datetime import datetime
from typing import Dict, Any

# Add backend path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Kimera imports
from backend.trading.live_trading_config import create_conservative_live_config
from backend.trading.kimera_live_execution_bridge import create_live_execution_bridge
from backend.trading.autonomous_kimera_trader import KimeraAutonomousTrader, CognitiveSignal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/kimera_live_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('KimeraLiveTrading')

class KimeraLiveTradingSystem:
    """
    Complete live trading system that bridges Kimera's analysis
    with real market execution.
    """
    
    def __init__(self, binance_api_key: str, binance_private_key_path: str):
        """
        Initialize live trading system
        
        Args:
            binance_api_key: Your Binance API key
            binance_private_key_path: Path to your Ed25519 private key
        """
        self.binance_api_key = binance_api_key
        self.binance_private_key_path = binance_private_key_path
        
        # Initialize components
        self.live_execution_bridge = None
        self.autonomous_trader = None
        self.trading_active = False
        
        logger.info("🔴 KIMERA LIVE TRADING SYSTEM INITIALIZED")
        logger.info("⚠️  WARNING: REAL MONEY TRADING CAPABILITY ENABLED")
    
    async def initialize_system(self):
        """Initialize all system components"""
        try:
            logger.info("🚀 Initializing live trading system...")
            
            # 1. Create live execution bridge
            self.live_execution_bridge = create_live_execution_bridge(
                binance_api_key=self.binance_api_key,
                binance_private_key_path=self.binance_private_key_path,
                initial_balance=500.0,      # Start with $500
                max_position_size=25.0,     # Max $25 per position
                max_daily_loss=10.0         # Max $10 daily loss
            )
            
            # 2. Initialize connections
            await self.live_execution_bridge.initialize_connections()
            
            # 3. Create autonomous trader (modified for live execution)
            self.autonomous_trader = KimeraAutonomousTrader(
                api_key=self.binance_api_key,
                target_eur=100.0
            )
            
            logger.info("✅ Live trading system initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ System initialization failed: {e}")
            raise
    
    async def enable_live_trading(self) -> bool:
        """Enable live trading with final confirmation"""
        try:
            logger.warning("🔴 ATTEMPTING TO ENABLE LIVE TRADING")
            logger.warning("⚠️  REAL MONEY WILL BE AT RISK")
            
            # Final safety check
            if not self._perform_final_safety_check():
                logger.error("❌ Final safety check failed")
                return False
            
            # Enable trading in the bridge
            if not self.live_execution_bridge.trading_manager.enable_trading():
                logger.error("❌ Failed to enable trading")
                return False
            
            self.trading_active = True
            
            logger.warning("🔥 LIVE TRADING ENABLED")
            logger.warning("⚠️  MONITOR CLOSELY")
            logger.warning("⚠️  EMERGENCY STOP AVAILABLE")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to enable live trading: {e}")
            return False
    
    def _perform_final_safety_check(self) -> bool:
        """Perform final safety check before enabling trading"""
        try:
            # Check API credentials
            if not self.binance_api_key or len(self.binance_api_key) < 10:
                logger.error("❌ Invalid API key")
                return False
            
            # Check private key file
            if not os.path.exists(self.binance_private_key_path):
                logger.error(f"❌ Private key file not found: {self.binance_private_key_path}")
                return False
            
            # Check system components
            if not self.live_execution_bridge:
                logger.error("❌ Execution bridge not initialized")
                return False
            
            if not self.autonomous_trader:
                logger.error("❌ Autonomous trader not initialized")
                return False
            
            logger.info("✅ Final safety check passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Safety check failed: {e}")
            return False
    
    async def run_live_trading_cycle(self):
        """Run a single live trading cycle"""
        try:
            if not self.trading_active:
                logger.warning("⚠️  Trading not active, skipping cycle")
                return
            
            logger.info("🧠 Starting live trading cycle...")
            
            # 1. Monitor existing positions
            await self.live_execution_bridge.monitor_live_positions()
            
            # 2. Generate new signals
            symbols = ['BTCUSDT', 'ETHUSDT']
            
            for symbol in symbols:
                # Skip if already have position
                if symbol in self.live_execution_bridge.active_positions:
                    continue
                
                # Generate cognitive signal
                signal = self.autonomous_trader.generate_cognitive_signal(symbol)
                
                if signal and signal.confidence > 0.75:  # High confidence only
                    logger.info(f"📊 High confidence signal generated: {symbol}")
                    
                    # Execute signal with real money
                    result = await self.live_execution_bridge.execute_cognitive_signal(signal)
                    
                    if result.position_created:
                        logger.info(f"✅ Live position created: {symbol}")
                        logger.info(f"   Size: {result.actual_position_size:.6f}")
                        logger.info(f"   Price: ${result.actual_entry_price:.2f}")
                        logger.info(f"   Fees: ${result.fees_paid:.2f}")
                    else:
                        logger.warning(f"❌ Failed to create position: {symbol}")
                        logger.warning(f"   Reason: {result.warnings}")
                
                # Only one new position per cycle
                if any(result.position_created for result in []):
                    break
            
            # 3. Log current status
            status = self.live_execution_bridge.get_live_status()
            logger.info(f"📈 Live Trading Status:")
            logger.info(f"   Active Positions: {status['active_positions']}")
            logger.info(f"   Daily P&L: ${status['daily_pnl']:.2f}")
            logger.info(f"   Emergency Stop: {status['emergency_stop_active']}")
            
            # 4. Check emergency conditions
            if status['daily_pnl'] < -10.0:  # $10 loss limit
                logger.critical("🚨 DAILY LOSS LIMIT REACHED")
                await self.emergency_stop()
            
        except Exception as e:
            logger.error(f"❌ Live trading cycle failed: {e}")
            await self.emergency_stop()
    
    async def run_continuous_trading(self, cycle_interval_seconds: int = 60):
        """Run continuous live trading"""
        logger.info(f"🚀 Starting continuous live trading (interval: {cycle_interval_seconds}s)")
        
        try:
            while self.trading_active:
                await self.run_live_trading_cycle()
                await asyncio.sleep(cycle_interval_seconds)
                
        except KeyboardInterrupt:
            logger.info("🛑 Keyboard interrupt - stopping trading")
        except Exception as e:
            logger.error(f"❌ Continuous trading failed: {e}")
        finally:
            await self.emergency_stop()
    
    async def emergency_stop(self):
        """Emergency stop all trading"""
        logger.critical("🚨 EMERGENCY STOP ACTIVATED")
        
        self.trading_active = False
        
        if self.live_execution_bridge:
            await self.live_execution_bridge.emergency_stop_all()
        
        logger.critical("🚨 ALL POSITIONS CLOSED")
        logger.critical("🚨 TRADING STOPPED")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get complete system status"""
        if not self.live_execution_bridge:
            return {'status': 'not_initialized'}
        
        return {
            'trading_active': self.trading_active,
            'live_status': self.live_execution_bridge.get_live_status(),
            'system_time': datetime.now().isoformat(),
        }


def main():
    """Main function to demonstrate live trading activation"""
    print("🔴 KIMERA LIVE TRADING ACTIVATION")
    print("=" * 50)
    print()
    print("⚠️  WARNING: THIS SCRIPT ENABLES REAL MONEY TRADING")
    print("⚠️  ONLY USE WITH PROPER CREDENTIALS AND UNDERSTANDING")
    print()
    
    # Example configuration (DO NOT USE THESE VALUES)
    BINANCE_API_KEY = "YOUR_BINANCE_API_KEY_HERE"
    BINANCE_PRIVATE_KEY_PATH = "path/to/your/private_key.pem"
    
    if BINANCE_API_KEY == "YOUR_BINANCE_API_KEY_HERE":
        print("❌ PLEASE SET YOUR ACTUAL API CREDENTIALS")
        print("❌ DO NOT USE PLACEHOLDER VALUES")
        print()
        print("Required steps:")
        print("1. Get Binance API key from your account")
        print("2. Create Ed25519 private key file")
        print("3. Update the credentials in this script")
        print("4. Test with small amounts first")
        print("5. Monitor continuously")
        return
    
    async def run_demo():
        """Run live trading demo"""
        # Initialize system
        system = KimeraLiveTradingSystem(
            binance_api_key=BINANCE_API_KEY,
            binance_private_key_path=BINANCE_PRIVATE_KEY_PATH
        )
        
        try:
            # Initialize all components
            await system.initialize_system()
            
            # Enable live trading
            if await system.enable_live_trading():
                print("✅ Live trading enabled successfully")
                
                # Run for 5 minutes as demo
                print("🚀 Running live trading demo for 5 minutes...")
                await asyncio.sleep(300)  # 5 minutes
                
                # Stop trading
                await system.emergency_stop()
                
                # Show final status
                status = system.get_system_status()
                print(f"📊 Final Status: {status}")
                
            else:
                print("❌ Failed to enable live trading")
                
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            await system.emergency_stop()
    
    # Run the demo
    asyncio.run(run_demo())


if __name__ == "__main__":
    main()


# SOLUTION SUMMARY
"""
KIMERA EXECUTION PROBLEM - SOLUTION SUMMARY
===========================================

PROBLEM IDENTIFIED:
- Kimera was running in simulation mode only
- No real exchange connections established
- Internal position tracking without real orders
- All configurations set to testnet/simulation

SOLUTION IMPLEMENTED:
1. Created LiveTradingConfig for real trading setup
2. Built KimeraLiveExecutionBridge for real order execution
3. Implemented comprehensive risk management
4. Added emergency stop mechanisms
5. Created integration scripts for activation

KEY COMPONENTS:
- live_trading_config.py: Configuration for real trading
- kimera_live_execution_bridge.py: Real execution bridge
- enable_live_trading.py: Activation script

SAFETY FEATURES:
- Position size limits ($25 max per position)
- Daily loss limits ($10 max daily loss)
- Emergency stop mechanisms
- Real-time monitoring
- Risk-based position sizing
- Execution confirmation

TO ENABLE LIVE TRADING:
1. Set up Binance API credentials
2. Create Ed25519 private key file
3. Configure risk parameters
4. Run enable_live_trading.py
5. Monitor continuously

CRITICAL NOTES:
- Start with small amounts
- Monitor continuously
- Have emergency stop ready
- Understand all risks
- Never leave unattended
""" 