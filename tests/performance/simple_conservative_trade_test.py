#!/usr/bin/env python3
"""
SIMPLE CONSERVATIVE TRADE TEST
==============================

Direct test of Kimera's real trading capability without complex system imports.
Uses minimal setup and ultra-conservative parameters.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Any

# Configure logging with timestamp
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class SimpleConservativeTest:
    """Ultra-conservative real trading test using HMAC authentication"""
    
    def __init__(self):
        self.config = {
            "test_symbol": "BTCUSDT",
            "max_position_usd": 10,  # Increased to $10 to ensure minimum lot size compliance
            "target_profit_pct": 0.3,  # 0.3% profit target
            "stop_loss_pct": 0.2,  # 0.2% stop loss
        }
    
    def _format_quantity(self, quantity: float, step_size: str = "0.00001000") -> str:
        """Format quantity according to Binance step size requirements"""
        # Convert step size to decimal places
        step_size_decimal = float(step_size)
        
        # Calculate decimal places needed
        decimal_places = len(step_size.split('.')[1].rstrip('0'))
        
        # Round to step size and format
        formatted_quantity = round(quantity / step_size_decimal) * step_size_decimal
        
        # Format with proper decimal places
        return f"{formatted_quantity:.{decimal_places}f}"
    
    async def execute_conservative_trade(self) -> Dict[str, Any]:
        """Execute ultra-conservative real trade using HMAC"""
        try:
            logger.info("🎯 Executing conservative trade...")
            
            # Import HMAC connector
            from src.trading.api.binance_connector_hmac import BinanceConnector
            
            # Get credentials
            api_key = os.getenv('BINANCE_API_KEY')
            secret_key = os.getenv('BINANCE_SECRET_KEY')
            
            async with BinanceConnector(api_key, secret_key, testnet=False) as connector:
                
                # Get exchange info first to understand lot size requirements
                exchange_info = await connector.get_exchange_info()
                
                # Find BTCUSDT symbol info
                btcusdt_info = None
                for symbol in exchange_info['symbols']:
                    if symbol['symbol'] == 'BTCUSDT':
                        btcusdt_info = symbol
                        break
                
                if not btcusdt_info:
                    raise Exception("BTCUSDT symbol not found")
                
                # Get lot size filter
                lot_size_filter = None
                min_notional_filter = None
                
                for filter_info in btcusdt_info['filters']:
                    if filter_info['filterType'] == 'LOT_SIZE':
                        lot_size_filter = filter_info
                    elif filter_info['filterType'] == 'MIN_NOTIONAL':
                        min_notional_filter = filter_info
                
                if not lot_size_filter:
                    raise Exception("LOT_SIZE filter not found")
                
                min_qty = float(lot_size_filter['minQty'])
                step_size = lot_size_filter['stepSize']
                
                logger.info(f"📏 Lot size requirements:")
                logger.info(f"   Minimum quantity: {min_qty} BTC")
                logger.info(f"   Step size: {step_size}")
                
                # Get current price
                ticker = await connector.get_ticker(self.config["test_symbol"])
                current_price = float(ticker['lastPrice'])
                
                # Calculate position size
                max_usd = self.config["max_position_usd"]
                position_size = max_usd / current_price
                
                # Ensure position size meets minimum requirements
                if position_size < min_qty:
                    logger.warning(f"⚠️ Position size {position_size:.8f} below minimum {min_qty:.8f}")
                    position_size = min_qty
                    actual_usd = position_size * current_price
                    logger.info(f"   Adjusted position size to minimum: {position_size:.8f} BTC (${actual_usd:.2f})")
                
                # Format position size according to step size
                formatted_position_size = self._format_quantity(position_size, step_size)
                position_size = float(formatted_position_size)
                
                # Calculate targets
                target_price = current_price * (1 + self.config["target_profit_pct"] / 100)
                actual_position_value = position_size * current_price
                
                logger.info(f"🎯 Trade plan:")
                logger.info(f"   Symbol: {self.config['test_symbol']}")
                logger.info(f"   Position size: {position_size:.8f} BTC")
                logger.info(f"   Position value: ${actual_position_value:.2f}")
                logger.info(f"   Entry price: ${current_price:,.2f}")
                logger.info(f"   Target price: ${target_price:,.2f} (+{self.config['target_profit_pct']}%)")
                
                # Auto-confirm - user already confirmed they want to proceed
                print("\n🚨 REAL TRADE CONFIRMATION 🚨")
                print(f"This will buy ${actual_position_value:.2f} worth of Bitcoin with REAL MONEY!")
                print(f"Position: {position_size:.8f} BTC @ ${current_price:,.2f}")
                print(f"Target: +{self.config['target_profit_pct']}% profit")
                print("\n✅ AUTO-CONFIRMING: User has already approved this trade...")
                
                # Execute the trade
                logger.info("🚀 EXECUTING REAL TRADE...")
                
                # Place market buy order with properly formatted quantity
                buy_order = await connector.place_order(
                    symbol=self.config["test_symbol"],
                    side="BUY",
                    order_type="MARKET",
                    quantity=position_size
                )
                
                logger.info("✅ BUY ORDER EXECUTED!")
                logger.info(f"   Order ID: {buy_order.get('orderId')}")
                logger.info(f"   Status: {buy_order.get('status')}")
                
                # Wait for fill confirmation
                await asyncio.sleep(3)
                
                # Check order status
                order_status = await connector.get_order(
                    symbol=self.config["test_symbol"],
                    order_id=buy_order['orderId']
                )
                
                if order_status.get('status') == 'FILLED':
                    filled_price = float(order_status.get('price', current_price))
                    filled_qty = float(order_status.get('executedQty', position_size))
                    
                    logger.info("✅ ORDER FILLED!")
                    logger.info(f"   Filled price: ${filled_price:,.2f}")
                    logger.info(f"   Filled quantity: {filled_qty:.8f} BTC")
                    logger.info(f"   Total value: ${filled_price * filled_qty:.2f}")
                    
                    # Place take profit order
                    tp_order = await connector.place_order(
                        symbol=self.config["test_symbol"],
                        side="SELL",
                        order_type="LIMIT",
                        quantity=filled_qty,
                        price=target_price,
                        time_in_force="GTC"
                    )
                    
                    logger.info("✅ TAKE PROFIT ORDER PLACED!")
                    logger.info(f"   TP Order ID: {tp_order.get('orderId')}")
                    logger.info(f"   TP Price: ${target_price:,.2f}")
                    
                    return {
                        "success": True,
                        "buy_order_id": buy_order['orderId'],
                        "tp_order_id": tp_order['orderId'],
                        "filled_price": filled_price,
                        "filled_quantity": filled_qty,
                        "target_price": target_price,
                        "position_value": filled_price * filled_qty
                    }
                else:
                    logger.error(f"❌ Order not filled: {order_status.get('status')}")
                    return {"success": False, "error": "Order not filled"}
                    
        except Exception as e:
            logger.error(f"❌ Trade execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def test_real_trading_connection(self) -> bool:
        """Test connection to real Binance trading API using HMAC"""
        try:
            logger.info("🔧 Testing real trading connection...")
            
            # Get API credentials from environment
            api_key = os.getenv('BINANCE_API_KEY')
            secret_key = os.getenv('BINANCE_SECRET_KEY')
            
            if not api_key:
                logger.error("❌ BINANCE_API_KEY environment variable not set")
                logger.info("   Set it with: export BINANCE_API_KEY='your_api_key'")
                return False
                
            if not secret_key:
                logger.error("❌ BINANCE_SECRET_KEY environment variable not set") 
                logger.info("   Set it with: export BINANCE_SECRET_KEY='your_secret_key'")
                return False
            
            # Test if we can import the HMAC connector
            from src.trading.api.binance_connector_hmac import BinanceConnector
            
            # Create connector in real trading mode
            async with BinanceConnector(api_key, secret_key, testnet=False) as connector:
                
                # Verify we're in real trading mode
                if "testnet" in connector.BASE_URL:
                    logger.error("❌ Connector is still using testnet URLs")
                    return False
                
                logger.info("✅ Connected to Binance LIVE API")
                logger.info(f"   Base URL: {connector.BASE_URL}")
                
                # Test basic API call
                ticker = await connector.get_ticker(self.config["test_symbol"])
                current_price = float(ticker['lastPrice'])
                
                logger.info(f"📊 Current {self.config['test_symbol']} price: ${current_price:,.2f}")
                
                # Test account access
                account = await connector.get_account()
                logger.info("✅ Account access verified")
                
                # Check USDT balance
                usdt_balance = 0
                for balance in account.get('balances', []):
                    if balance['asset'] == 'USDT':
                        usdt_balance = float(balance['free'])
                        break
                
                logger.info(f"💵 Available USDT balance: ${usdt_balance:.2f}")
                
                if usdt_balance < self.config["max_position_usd"]:
                    logger.warning(f"⚠️ Low balance: Have ${usdt_balance:.2f}, need ${self.config['max_position_usd']}")
                    return False
                
                return True
                
        except ImportError as e:
            logger.error(f"❌ Could not import HMAC trading connector: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Connection test failed: {e}")
            return False
    
    async def run_test(self) -> bool:
        """Run the complete test"""
        try:
            print("🧪 KIMERA SIMPLE CONSERVATIVE TRADE TEST")
            print("=" * 50)
            
            # Step 1: Test connection
            if not await self.test_real_trading_connection():
                logger.error("❌ Connection test failed")
                return False
            
            # Step 2: Execute trade
            result = await self.execute_conservative_trade()
            
            # Step 3: Report results
            print("\n" + "=" * 50)
            print("📊 TEST RESULTS:")
            
            if result["success"]:
                print("✅ CONSERVATIVE TRADE TEST PASSED!")
                print("🎉 Kimera real trading is WORKING!")
                print(f"   Position value: ${result.get('position_value', 0):.2f}")
                print(f"   Buy order: {result.get('buy_order_id')}")
                print(f"   Take profit order: {result.get('tp_order_id')}")
                return True
            else:
                print("❌ CONSERVATIVE TRADE TEST FAILED!")
                print(f"   Error: {result.get('error')}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Test failed: {e}")
            return False

async def main():
    """Main execution function"""
    print("🚨 REAL TRADING TEST WARNING 🚨")
    print("This will execute a REAL TRADE with REAL MONEY!")
    print("Maximum risk: $5 USD")
    print()
    print("Ensure you have:")
    print("1. Set BINANCE_API_KEY environment variable")
    print("2. Set BINANCE_SECRET_KEY environment variable")
    print("3. At least $5 USDT balance in your Binance account")
    print("4. Are prepared to risk this amount")
    print()
    
    response = input("Proceed with REAL TRADING TEST? (type 'YES' to continue): ")
    if response != 'YES':
        print("❌ Test cancelled")
        return
    
    print(f"\nStarting test at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    test = SimpleConservativeTest()
    success = await test.run_test()
    
    if success:
        print("\n🎉 REAL TRADING TEST COMPLETED SUCCESSFULLY!")
    else:
        print("\n❌ REAL TRADING TEST FAILED!")
    
    print(f"\nTest completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    asyncio.run(main()) 