#!/usr/bin/env python3
"""
KIMERA CONSERVATIVE TRADE TEST
=============================

Ultra-conservative trade test to verify real trading functionality.
Uses minimal position sizes and strict safety controls.
"""

import os
import sys
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ConservativeTradeTest:
    """
    Ultra-conservative trade test with maximum safety controls
    """
    
    def __init__(self):
        """Initialize with ultra-conservative settings"""
        self.test_config = {
            # Ultra-conservative test parameters
            "max_position_usd": 5.0,      # Maximum $5 position
            "target_profit_pct": 0.5,     # 0.5% profit target
            "stop_loss_pct": 0.3,         # 0.3% stop loss
            "test_symbol": "BTCUSDT",     # Most liquid pair
            "max_test_duration_minutes": 5,  # Maximum 5 minutes
            
            # Safety overrides
            "require_confirmation": True,
            "simulation_mode": False,  # We want real trading for this test
            "emergency_stop_enabled": True
        }
        
        self.trading_manager = None
        self.trade_active = False
        self.start_time = None
        
    async def initialize_system(self) -> bool:
        """Initialize Kimera trading system for conservative test"""
        try:
            logger.info("🔧 Initializing Kimera trading system...")
            
            # Import required components
            from src.trading.core.live_trading_manager import LiveTradingManager, LiveTradingConfig, TradingMode, TradingPhase
            from src.vault.vault_manager import VaultManager
            
            # Create ultra-conservative configuration
            config = LiveTradingConfig(
                mode=TradingMode.LIVE_MINIMAL,
                phase=TradingPhase.PROOF_OF_CONCEPT,
                starting_capital=self.test_config["max_position_usd"],
                current_capital=self.test_config["max_position_usd"],
                max_daily_risk=0.01,  # 1% max daily risk
                max_position_size=0.50,  # 50% max position (of our tiny capital)
                primary_exchange="binance",
                use_testnet=False,  # REAL TRADING MODE
                enable_circuit_breakers=True,
                max_consecutive_losses=1,  # Stop after 1 loss
                daily_loss_limit=0.02,  # 2% daily loss limit
                emergency_stop_loss=0.05,  # 5% portfolio stop loss
                require_ethical_approval=True,
                min_confidence_threshold=0.8,  # High confidence required
            )
            
            # Initialize trading manager
            self.trading_manager = LiveTradingManager(config)
            
            logger.info("✅ Trading system initialized")
            logger.info(f"   Mode: {config.mode.value}")
            logger.info(f"   Capital: ${config.starting_capital}")
            logger.info(f"   Real Trading: {not config.use_testnet}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize trading system: {e}")
            return False
    
    async def verify_system_status(self) -> bool:
        """Verify system is ready for trading"""
        try:
            logger.info("🔍 Verifying system status...")
            
            # Check if we're in real trading mode
            from src.trading.core.live_trading_manager import LiveTradingConfig
            config = LiveTradingConfig()
            
            if config.use_testnet:
                logger.warning("⚠️ System is in TESTNET mode - not real trading")
                logger.info("To enable real trading, ensure KIMERA_USE_TESTNET is not set to 'true'")
                return False
            
            logger.info("✅ System verified - REAL TRADING MODE ACTIVE")
            
            # Additional safety checks
            logger.info("🛡️ Safety systems status:")
            logger.info(f"   Circuit breakers: {config.enable_circuit_breakers}")
            logger.info(f"   Ethical approval: {config.require_ethical_approval}")
            logger.info(f"   Daily loss limit: {config.daily_loss_limit * 100}%")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ System verification failed: {e}")
            return False
    
    async def execute_conservative_trade(self) -> Dict[str, Any]:
        """Execute ultra-conservative test trade"""
        try:
            logger.info("🎯 Executing conservative trade test...")
            self.start_time = datetime.now()
            
            # Import trading components
            from src.trading.api.binance_connector import BinanceConnector
            from src.trading.core.trading_engine import TradingDecision, MarketState
            
            # Get real API credentials (you'll need to set these)
            api_key = os.getenv('BINANCE_API_KEY')
            private_key_path = os.getenv('BINANCE_PRIVATE_KEY_PATH')
            
            if not api_key or not private_key_path:
                logger.error("❌ Missing API credentials:")
                logger.error("   Set BINANCE_API_KEY environment variable")
                logger.error("   Set BINANCE_PRIVATE_KEY_PATH environment variable")
                return {
                    "success": False,
                    "error": "Missing API credentials",
                    "message": "Please set BINANCE_API_KEY and BINANCE_PRIVATE_KEY_PATH"
                }
            
            # Create connector for real trading
            async with BinanceConnector(api_key, private_key_path, testnet=False) as connector:
                logger.info("🔗 Connected to Binance (LIVE)")
                
                # Get current market data
                ticker = await connector.get_ticker(self.test_config["test_symbol"])
                current_price = float(ticker['lastPrice'])
                
                logger.info(f"📊 Current {self.test_config['test_symbol']} price: ${current_price:,.2f}")
                
                # Calculate ultra-conservative position size
                max_usd = self.test_config["max_position_usd"]
                position_size = max_usd / current_price
                
                # Round to valid precision (Binance has minimum notionals)
                position_size = round(position_size, 6)
                
                logger.info(f"💰 Conservative position size: {position_size} BTC (${max_usd})")
                
                # Get account balance first
                account = await connector.get_account()
                usdt_balance = 0
                for balance in account.get('balances', []):
                    if balance['asset'] == 'USDT':
                        usdt_balance = float(balance['free'])
                        break
                
                if usdt_balance < max_usd:
                    logger.error(f"❌ Insufficient balance: ${usdt_balance:.2f} USDT available, need ${max_usd}")
                    return {
                        "success": False,
                        "error": "Insufficient balance",
                        "balance": usdt_balance,
                        "required": max_usd
                    }
                
                logger.info(f"💵 Available balance: ${usdt_balance:.2f} USDT")
                
                # Calculate targets
                target_price = current_price * (1 + self.test_config["target_profit_pct"] / 100)
                stop_price = current_price * (1 - self.test_config["stop_loss_pct"] / 100)
                
                logger.info(f"🎯 Trade targets:")
                logger.info(f"   Entry: ${current_price:,.2f}")
                logger.info(f"   Target: ${target_price:,.2f} (+{self.test_config['target_profit_pct']}%)")
                logger.info(f"   Stop: ${stop_price:,.2f} (-{self.test_config['stop_loss_pct']}%)")
                
                # Confirm with user
                if self.test_config["require_confirmation"]:
                    response = input(f"\n🚨 CONFIRM REAL TRADE: Buy ${max_usd} worth of BTC? (yes/no): ")
                    if response.lower() != 'yes':
                        logger.info("❌ Trade cancelled by user")
                        return {"success": False, "error": "User cancelled trade"}
                
                # Execute the trade
                logger.info("🚀 EXECUTING REAL TRADE...")
                self.trade_active = True
                
                # Place market buy order
                buy_order = await connector.place_order(
                    symbol=self.test_config["test_symbol"],
                    side="BUY",
                    order_type="MARKET",
                    quantity=position_size
                )
                
                logger.info("✅ BUY ORDER EXECUTED!")
                logger.info(f"   Order ID: {buy_order.get('orderId')}")
                logger.info(f"   Status: {buy_order.get('status')}")
                
                # Wait a moment for fill
                await asyncio.sleep(2)
                
                # Check order status
                order_status = await connector.get_order(
                    symbol=self.test_config["test_symbol"],
                    order_id=buy_order['orderId']
                )
                
                if order_status.get('status') == 'FILLED':
                    filled_price = float(order_status.get('price', current_price))
                    filled_qty = float(order_status.get('executedQty', position_size))
                    
                    logger.info("✅ ORDER FILLED!")
                    logger.info(f"   Filled Price: ${filled_price:,.2f}")
                    logger.info(f"   Filled Quantity: {filled_qty} BTC")
                    
                    # Immediately place take profit order
                    tp_order = await connector.place_order(
                        symbol=self.test_config["test_symbol"],
                        side="SELL",
                        order_type="LIMIT",
                        quantity=filled_qty,
                        price=target_price,
                        time_in_force="GTC"
                    )
                    
                    logger.info("✅ TAKE PROFIT ORDER PLACED!")
                    logger.info(f"   TP Order ID: {tp_order.get('orderId')}")
                    
                    self.trade_active = False
                    
                    return {
                        "success": True,
                        "buy_order": buy_order,
                        "tp_order": tp_order,
                        "filled_price": filled_price,
                        "filled_quantity": filled_qty,
                        "target_price": target_price,
                        "message": "Conservative trade executed successfully!"
                    }
                
                else:
                    logger.error(f"❌ Order not filled: {order_status.get('status')}")
                    return {
                        "success": False,
                        "error": "Order not filled",
                        "order_status": order_status
                    }
                
        except Exception as e:
            logger.error(f"❌ Trade execution failed: {e}")
            self.trade_active = False
            return {
                "success": False,
                "error": str(e)
            }
    
    async def run_test(self) -> bool:
        """Run complete conservative trade test"""
        try:
            logger.info("🧪 KIMERA CONSERVATIVE TRADE TEST")
            logger.info("=" * 50)
            
            # Step 1: Initialize system
            if not await self.initialize_system():
                return False
            
            # Step 2: Verify system status
            if not await self.verify_system_status():
                return False
            
            # Step 3: Execute conservative trade
            result = await self.execute_conservative_trade()
            
            # Step 4: Report results
            logger.info("\n" + "=" * 50)
            logger.info("📊 TEST RESULTS:")
            
            if result["success"]:
                logger.info("✅ CONSERVATIVE TRADE TEST PASSED!")
                logger.info("🎉 Kimera trading system is working with real trading!")
                logger.info(f"   Trade value: ~${self.test_config['max_position_usd']}")
                logger.info(f"   Target profit: {self.test_config['target_profit_pct']}%")
                return True
            else:
                logger.error("❌ CONSERVATIVE TRADE TEST FAILED!")
                logger.error(f"   Error: {result.get('error', 'Unknown error')}")
                return False
            
        except Exception as e:
            logger.error(f"❌ Test execution failed: {e}")
            return False

async def main():
    """Main test execution"""
    test = ConservativeTradeTest()
    
    # Important safety warnings
    print("\n🚨 REAL TRADING TEST WARNING 🚨")
    print("This will execute a REAL TRADE with REAL MONEY!")
    print(f"Maximum risk: ${test.test_config['max_position_usd']}")
    print("Ensure you have:")
    print("1. Set BINANCE_API_KEY environment variable")
    print("2. Set BINANCE_PRIVATE_KEY_PATH environment variable")
    print("3. Sufficient USDT balance in your account")
    print("4. Are ready to risk the test amount")
    
    confirm = input("\nProceed with REAL TRADING TEST? (type 'CONFIRM' to proceed): ")
    if confirm != 'CONFIRM':
        print("❌ Test cancelled")
        return False
    
    success = await test.run_test()
    return success

if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n❌ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        sys.exit(1) 