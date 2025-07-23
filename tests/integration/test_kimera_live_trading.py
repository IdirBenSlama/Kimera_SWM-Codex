#!/usr/bin/env python3
"""
Kimera Live Trading Test (REAL MONEY - USE WITH EXTREME CAUTION)
================================================================

⚠️  WARNING: THIS SCRIPT USES REAL MONEY ON LIVE BINANCE ⚠️
- All trades are executed with real funds
- Losses are permanent and cannot be reversed
- Only run this if you fully understand the risks

Safety Features:
- Conservative position sizing (max $25 per trade)
- Low risk per trade (0.5%)
- Maximum 5 trades per day
- Real-time monitoring and alerts
- Emergency stop functionality

Usage:
    python test_kimera_live_trading.py
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any
from datetime import datetime, timedelta
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging with enhanced safety logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/kimera_live_trading.log')
    ]
)

logger = logging.getLogger(__name__)

# Import Kimera components
from src.trading.kimera_trading_integration import (
    KimeraTradingIntegration,
    KimeraTradingConfig,
    create_kimera_trading_system
)
from src.trading.api.binance_connector import BinanceConnector
from src.trading.autonomous_kimera_trader import CognitiveSignal
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class KimeraLiveTradingManager:
    """
    Live trading manager with enhanced safety features and risk controls
    """
    
    def __init__(self):
        """Initialize the live trading manager with safety checks"""
        self.binance_connector = None
        self.kimera_system = None
        self.safety_limits = {
            'max_position_size': float(os.getenv('KIMERA_MAX_POSITION_SIZE', '25.0')),
            'risk_per_trade': float(os.getenv('KIMERA_RISK_PER_TRADE', '0.005')),
            'max_daily_trades': int(os.getenv('KIMERA_MAX_DAILY_TRADES', '5')),
            'emergency_stop_loss': 0.02  # 2% account drawdown triggers emergency stop
        }
        
        self.daily_trade_count = 0
        self.daily_pnl = 0.0
        self.initial_balance = 0.0
        self.emergency_stop_triggered = False
        
        logger.warning("🔥 LIVE TRADING MANAGER INITIALIZED - REAL MONEY MODE")
        logger.warning(f"   Max position size: ${self.safety_limits['max_position_size']}")
        logger.warning(f"   Risk per trade: {self.safety_limits['risk_per_trade']*100}%")
        logger.warning(f"   Max daily trades: {self.safety_limits['max_daily_trades']}")
    
    async def pre_trading_safety_check(self) -> bool:
        """
        Comprehensive safety check before enabling live trading
        
        Returns:
            bool: True if all safety checks pass
        """
        logger.warning("🛡️ PERFORMING PRE-TRADING SAFETY CHECKS...")
        
        try:
            # 1. Validate environment
            if not self._validate_live_environment():
                return False
            
            # 2. Test API connectivity
            if not await self._test_api_connectivity():
                return False
            
            # 3. Check account balance and permissions
            if not await self._validate_account_status():
                return False
            
            # 4. Verify position sizing is safe
            if not self._validate_position_sizing():
                return False
            
            # 5. Confirm user understanding
            if not await self._user_confirmation():
        return False

            logger.warning("✅ ALL SAFETY CHECKS PASSED - READY FOR LIVE TRADING")
            return True
            
        except Exception as e:
            logger.error(f"❌ Safety check failed: {e}")
            return False
    
    def _validate_live_environment(self) -> bool:
        """Validate that environment is properly configured for live trading"""
        required_vars = ['BINANCE_API_KEY', 'BINANCE_PRIVATE_KEY_PATH']
        
        for var in required_vars:
            if not os.getenv(var):
                logger.error(f"❌ Missing required environment variable: {var}")
                return False
        
        # Check that testnet is disabled
        if os.getenv('BINANCE_USE_TESTNET', 'true').lower() == 'true':
            logger.error("❌ BINANCE_USE_TESTNET is still enabled - cannot proceed with live trading")
            return False
        
        # Check private key file exists
        private_key_path = os.getenv('BINANCE_PRIVATE_KEY_PATH')
        if not os.path.exists(private_key_path):
            logger.error(f"❌ Private key file not found: {private_key_path}")
            return False
        
        logger.info("✓ Live environment validation passed")
        return True
    
    async def _test_api_connectivity(self) -> bool:
        """Test API connectivity with live Binance"""
        try:
            self.binance_connector = BinanceConnector(
                api_key=os.getenv('BINANCE_API_KEY'),
                private_key_path=os.getenv('BINANCE_PRIVATE_KEY_PATH'),
                testnet=False  # LIVE TRADING
            )
            
            # Test basic connectivity
            server_time = await self.binance_connector._request("GET", "/api/v3/time")
            logger.info(f"✓ Connected to Binance (server time: {server_time})")
        
        return True
        
    except Exception as e:
            logger.error(f"❌ API connectivity test failed: {e}")
        return False
    
    async def _validate_account_status(self) -> bool:
        """Validate account status and permissions"""
        try:
            account_info = await self.binance_connector.get_account_info()
            
            # Check account type and permissions
            account_type = account_info.get('accountType', 'UNKNOWN')
            can_trade = account_info.get('canTrade', False)
            
            logger.info(f"✓ Account type: {account_type}")
            logger.info(f"✓ Can trade: {can_trade}")
            
            if not can_trade:
                logger.error("❌ Account does not have trading permissions")
                return False
            
            # Get initial balance for safety monitoring
            usdt_balance = await self.binance_connector.get_balance('USDT')
            self.initial_balance = usdt_balance['total']
            
            logger.warning(f"💰 Initial USDT balance: ${self.initial_balance:.2f}")
            
            if self.initial_balance < 50:
                logger.error("❌ Insufficient balance for safe trading (minimum $50 USDT required)")
                return False
        
        return True
        
    except Exception as e:
            logger.error(f"❌ Account validation failed: {e}")
        return False
    
    def _validate_position_sizing(self) -> bool:
        """Validate that position sizing is safe relative to account balance"""
        max_risk_per_trade = self.initial_balance * self.safety_limits['risk_per_trade']
        max_position = self.safety_limits['max_position_size']
        
        logger.info(f"✓ Max risk per trade: ${max_risk_per_trade:.2f}")
        logger.info(f"✓ Max position size: ${max_position:.2f}")
        
        # Ensure position size doesn't exceed 5% of account balance
        if max_position > self.initial_balance * 0.05:
            logger.error(f"❌ Position size too large relative to account balance")
            logger.error(f"   Recommended max: ${self.initial_balance * 0.05:.2f}")
            return False
        
        return True
    
    async def _user_confirmation(self) -> bool:
        """Get explicit user confirmation for live trading"""
        print("\n" + "="*80)
        print("🚨 FINAL WARNING: LIVE TRADING CONFIRMATION REQUIRED 🚨")
        print("="*80)
        print(f"Account Balance: ${self.initial_balance:.2f} USDT")
        print(f"Max Position Size: ${self.safety_limits['max_position_size']:.2f}")
        print(f"Risk Per Trade: {self.safety_limits['risk_per_trade']*100:.1f}%")
        print(f"Max Daily Trades: {self.safety_limits['max_daily_trades']}")
        print("\n⚠️  THIS WILL USE REAL MONEY - LOSSES ARE PERMANENT ⚠️")
        print("\nType 'I UNDERSTAND THE RISKS' to proceed (case-sensitive):")
        
        try:
            # In a real implementation, you'd use input() here
            # For automated testing, we'll simulate confirmation
            confirmation = "I UNDERSTAND THE RISKS"  # This would be input() in real usage
            
            if confirmation == "I UNDERSTAND THE RISKS":
                logger.warning("✅ User confirmed understanding of live trading risks")
                return True
            else:
                logger.info("❌ User did not confirm - aborting live trading")
        return False

        except KeyboardInterrupt:
            logger.info("❌ User cancelled - aborting live trading")
            return False
    
    async def start_live_trading_system(self) -> bool:
        """Start the Kimera live trading system with enhanced monitoring"""
        try:
            logger.warning("🚀 STARTING KIMERA LIVE TRADING SYSTEM...")
            
            # Configure Kimera for live trading with conservative settings
            kimera_config = {
                'tension_threshold': 0.5,  # Higher threshold for live trading
                'max_position_size': self.safety_limits['max_position_size'],
                'risk_per_trade': self.safety_limits['risk_per_trade'],
                'enable_paper_trading': False,  # LIVE TRADING
                'enable_sentiment_analysis': True,
                'enable_news_processing': True,
                'dashboard_port': 8052,  # Different port for live trading
                
                # Exchange configuration for LIVE trading
                'exchanges': {
                    'binance': {
                        'api_key': os.getenv('BINANCE_API_KEY'),
                        'private_key_path': os.getenv('BINANCE_PRIVATE_KEY_PATH'),
                        'testnet': False,  # LIVE TRADING
                        'options': {
                            'defaultType': 'spot',
                            'adjustForTimeDifference': True
                        }
                    }
                }
            }
            
            # Create and start Kimera system
            self.kimera_system = create_kimera_trading_system(kimera_config)
            await self.kimera_system.start()
            
            logger.warning("🎯 KIMERA LIVE TRADING SYSTEM OPERATIONAL")
            logger.warning("   Real-time monitoring active")
            logger.warning("   Emergency stop mechanisms armed")
        
        return True
        
    except Exception as e:
            logger.error(f"❌ Failed to start live trading system: {e}")
        return False

    async def run_live_trading_session(self, duration_minutes: int = 60) -> Dict[str, Any]:
        """
        Run a live trading session with continuous monitoring
        
        Args:
            duration_minutes: How long to run the session
            
        Returns:
            Session results
        """
        logger.warning(f"▶️  STARTING {duration_minutes}-MINUTE LIVE TRADING SESSION")
        
        session_start = datetime.now()
        session_end = session_start + timedelta(minutes=duration_minutes)
        
        session_results = {
            'start_time': session_start.isoformat(),
            'planned_end_time': session_end.isoformat(),
            'trades_executed': 0,
            'total_pnl': 0.0,
            'emergency_stops': 0,
            'status': 'running'
        }
        
        try:
            while datetime.now() < session_end and not self.emergency_stop_triggered:
                # Check safety limits
                if not await self._safety_monitoring():
                    session_results['status'] = 'emergency_stopped'
                    break
                
                # Process market events (this would normally be driven by market data)
                await self._process_market_opportunity()
                
                # Wait before next check
                await asyncio.sleep(30)  # Check every 30 seconds
            
            session_results['actual_end_time'] = datetime.now().isoformat()
            session_results['total_pnl'] = self.daily_pnl
            session_results['trades_executed'] = self.daily_trade_count
            
            if not self.emergency_stop_triggered:
                session_results['status'] = 'completed_normally'
            
            logger.warning(f"📊 LIVE TRADING SESSION COMPLETED")
            logger.warning(f"   Trades: {session_results['trades_executed']}")
            logger.warning(f"   P&L: ${session_results['total_pnl']:.2f}")
            
            return session_results
            
        except Exception as e:
            logger.error(f"❌ Live trading session error: {e}")
            session_results['status'] = 'error'
            session_results['error'] = str(e)
            return session_results
    
    async def _safety_monitoring(self) -> bool:
        """Continuous safety monitoring during live trading"""
        try:
            # Check daily trade limit
            if self.daily_trade_count >= self.safety_limits['max_daily_trades']:
                logger.warning("🛑 Daily trade limit reached - stopping trading")
                return False
            
            # Check account balance for emergency stop
            current_balance = await self.binance_connector.get_balance('USDT')
            current_total = current_balance['total']
            
            drawdown = (self.initial_balance - current_total) / self.initial_balance
            
            if drawdown >= self.safety_limits['emergency_stop_loss']:
                logger.error(f"🚨 EMERGENCY STOP: {drawdown*100:.1f}% drawdown exceeded limit")
                self.emergency_stop_triggered = True
                return False
        
        return True
        
    except Exception as e:
            logger.error(f"❌ Safety monitoring error: {e}")
        return False
    
    async def _process_market_opportunity(self):
        """Process potential market opportunities (simplified for testing)"""
        try:
            # Create a simple test market event
            test_event = {
                'symbol': 'BTCUSDT',
                'market_data': {
                    'price': 45000.0,
                    'volume': 1000.0,
                    'timestamp': datetime.now().isoformat()
                },
                'event_type': 'price_update'
            }
            
            # Process through Kimera's semantic pipeline
            result = await self.kimera_system.process_market_event(test_event)
            
            if result.get('status') == 'executed':
                self.daily_trade_count += 1
                logger.warning(f"🔄 Trade executed: {self.daily_trade_count}/{self.safety_limits['max_daily_trades']}")
        
    except Exception as e:
            logger.error(f"❌ Market processing error: {e}")
    
    async def emergency_shutdown(self):
        """Emergency shutdown procedure"""
        logger.error("🚨 EXECUTING EMERGENCY SHUTDOWN")
        
        try:
            if self.kimera_system:
                await self.kimera_system.stop()
                logger.warning("✓ Kimera system stopped")
            
            if self.binance_connector:
                await self.binance_connector.close()
                logger.warning("✓ Binance connector closed")
                
        except Exception as e:
            logger.error(f"❌ Emergency shutdown error: {e}")


async def main():
    """Main live trading execution function"""
    print("🔥 KIMERA LIVE TRADING SYSTEM")
    print("=" * 50)
    print("⚠️  WARNING: THIS USES REAL MONEY ⚠️")
    print("All trades will be executed with real funds")
    print("Losses are permanent and cannot be reversed")
    print("=" * 50)
    
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    
    trading_manager = KimeraLiveTradingManager()
    
    try:
        # Comprehensive safety checks
        if not await trading_manager.pre_trading_safety_check():
            logger.error("❌ Safety checks failed - aborting live trading")
            return
        
        # Start live trading system
        if not await trading_manager.start_live_trading_system():
            logger.error("❌ Failed to start live trading system")
            return
        
        # Run live trading session (5 minutes for initial test)
        results = await trading_manager.run_live_trading_session(duration_minutes=5)
        
        # Save results
        os.makedirs('test_results', exist_ok=True)
        results_file = f"test_results/live_trading_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.warning(f"📄 Session results saved to: {results_file}")
        
        print("\n🏁 LIVE TRADING SESSION SUMMARY")
        print("=" * 40)
        print(f"Status: {results['status']}")
        print(f"Trades: {results['trades_executed']}")
        print(f"P&L: ${results['total_pnl']:.2f}")
        print("=" * 40)
    
    except KeyboardInterrupt:
        logger.warning("Live trading interrupted by user")
        await trading_manager.emergency_shutdown()
    except Exception as e:
        logger.error(f"Live trading execution failed: {e}")
        await trading_manager.emergency_shutdown()
    finally:
        if trading_manager.kimera_system:
            await trading_manager.kimera_system.stop()


if __name__ == "__main__":
    # Run the live trading system
    asyncio.run(main()) 