#!/usr/bin/env python3
"""
KIMERA EMERGENCY STOP PROTOCOL
==============================

Immediately stops all trading activities and closes open positions.
Use this script in case of emergency or unexpected behavior.

USAGE: python emergency_stop.py
"""

import asyncio
import os
import sys
import logging
from datetime import datetime

# Add backend to path
sys.path.append('backend')

from trading.api.binance_connector_hmac import BinanceConnector

# Setup emergency logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - EMERGENCY - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'emergency_stop_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)

logger = logging.getLogger(__name__)

class EmergencyStopProtocol:
    """Emergency stop protocol for immediate trading halt."""
    
    def __init__(self):
        self.connector = None
        self.stopped_orders = []
        self.closed_positions = []
        
    async def initialize(self):
        """Initialize emergency connection."""
        try:
            # Load credentials
            if not os.path.exists('kimera_binance_hmac.env'):
                logger.error("❌ Credentials file not found!")
                return False
                
            with open('kimera_binance_hmac.env', 'r') as f:
                for line in f:
                    if '=' in line and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value
                        
            api_key = os.environ.get('BINANCE_API_KEY')
            secret_key = os.environ.get('BINANCE_SECRET_KEY')
            testnet = os.environ.get('BINANCE_USE_TESTNET', 'false').lower() == 'true'
            
            self.connector = BinanceConnector(
                api_key=api_key,
                secret_key=secret_key,
                testnet=testnet
            )
            
            logger.info("🔗 Emergency connection established")
            return True
            
        except Exception as e:
            logger.error(f"❌ Emergency initialization failed: {e}")
            return False
            
    async def cancel_all_open_orders(self):
        """Cancel all open orders immediately."""
        try:
            logger.info("🛑 Cancelling all open orders...")
            
            # Get all open orders
            open_orders = await self.connector.get_open_orders()
            
            if not open_orders:
                logger.info("✅ No open orders to cancel")
                return True
                
            # Cancel each order
            for order in open_orders:
                try:
                    symbol = order['symbol']
                    order_id = order['orderId']
                    
                    result = await self.connector.cancel_order(symbol, order_id)
                    
                    if result:
                        self.stopped_orders.append({
                            'symbol': symbol,
                            'order_id': order_id,
                            'side': order['side'],
                            'quantity': order['origQty'],
                            'status': 'CANCELLED'
                        })
                        logger.info(f"✅ Cancelled order {order_id} for {symbol}")
                    else:
                        logger.warning(f"⚠️ Failed to cancel order {order_id}")
                        
                except Exception as e:
                    logger.error(f"❌ Error cancelling order {order.get('orderId')}: {e}")
                    
            logger.info(f"🛑 Emergency stop: {len(self.stopped_orders)} orders cancelled")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to cancel orders: {e}")
            return False
            
    async def close_all_positions(self):
        """Close all open positions immediately."""
        try:
            logger.info("🔄 Closing all positions...")
            
            # Get account info to find non-zero balances
            account_info = await self.connector.get_account_info()
            
            if not account_info:
                logger.error("❌ Could not get account info")
                return False
                
            # Find all non-zero balances (potential positions)
            positions_to_close = []
            
            for balance in account_info.get('balances', []):
                asset = balance['asset']
                free_balance = float(balance['free'])
                
                # Skip USDT and very small balances
                if asset == 'USDT' or free_balance < 0.001:
                    continue
                    
                # Check if this asset has a USDT trading pair
                symbol = f"{asset}USDT"
                
                try:
                    # Test if symbol exists by getting ticker
                    ticker = await self.connector.get_ticker_price(symbol)
                    
                    if ticker:
                        positions_to_close.append({
                            'asset': asset,
                            'symbol': symbol,
                            'quantity': free_balance,
                            'current_price': float(ticker['price'])
                        })
                        
                except Exception as e:
                    logger.error(f"Error in emergency_stop.py: {e}", exc_info=True)
                    raise  # Re-raise for proper error handling
                    # Symbol doesn't exist or not tradeable
                    continue
                    
            if not positions_to_close:
                logger.info("✅ No positions to close")
                return True
                
            # Close each position
            for position in positions_to_close:
                try:
                    symbol = position['symbol']
                    quantity = position['quantity']
                    
                    # Round quantity appropriately
                    if symbol == 'TRXUSDT':
                        quantity = round(quantity, 2)
                    else:
                        quantity = round(quantity, 6)
                        
                    # Execute market sell order
                    order = await self.connector.create_market_order(
                        symbol=symbol,
                        side='SELL',
                        quantity=quantity
                    )
                    
                    if order and order.get('status') == 'FILLED':
                        filled_qty = float(order['executedQty'])
                        filled_price = float(order['fills'][0]['price'])
                        value_usd = filled_qty * filled_price
                        
                        self.closed_positions.append({
                            'symbol': symbol,
                            'asset': position['asset'],
                            'quantity': filled_qty,
                            'price': filled_price,
                            'value_usd': value_usd,
                            'order_id': order['orderId']
                        })
                        
                        logger.info(f"✅ Closed {filled_qty} {position['asset']} for ${value_usd:.2f}")
                    else:
                        logger.warning(f"⚠️ Failed to close {position['asset']} position")
                        
                except Exception as e:
                    logger.error(f"❌ Error closing {position['asset']}: {e}")
                    
            logger.info(f"🔄 Emergency close: {len(self.closed_positions)} positions closed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to close positions: {e}")
            return False
            
    async def generate_emergency_report(self):
        """Generate emergency stop report."""
        try:
            total_value_closed = sum(p['value_usd'] for p in self.closed_positions)
            
            report = {
                'emergency_stop_time': datetime.now().isoformat(),
                'orders_cancelled': len(self.stopped_orders),
                'positions_closed': len(self.closed_positions),
                'total_value_closed_usd': total_value_closed,
                'cancelled_orders': self.stopped_orders,
                'closed_positions': self.closed_positions,
                'status': 'EMERGENCY_STOP_COMPLETE'
            }
            
            # Save report
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"emergency_stop_report_{timestamp}.json"
            
            with open(filename, 'w') as f:
                import json
                json.dump(report, f, indent=2)
                
            logger.info(f"📄 Emergency report saved: {filename}")
            
            # Print summary
            print("\n" + "="*60)
            print("🚨 EMERGENCY STOP PROTOCOL COMPLETE")
            print("="*60)
            print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"🛑 Orders Cancelled: {len(self.stopped_orders)}")
            print(f"🔄 Positions Closed: {len(self.closed_positions)}")
            print(f"💰 Total Value Closed: ${total_value_closed:.2f}")
            print("="*60)
            
            if self.closed_positions:
                print("📊 CLOSED POSITIONS:")
                for pos in self.closed_positions:
                    print(f"   • {pos['quantity']:.4f} {pos['asset']} → ${pos['value_usd']:.2f}")
                    
            if self.stopped_orders:
                print("🛑 CANCELLED ORDERS:")
                for order in self.stopped_orders:
                    print(f"   • {order['symbol']} {order['side']} {order['quantity']}")
                    
            print("="*60)
            print("✅ All trading activities have been stopped.")
            print("📄 Detailed report saved to:", filename)
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Failed to generate emergency report: {e}")
            return None
            
    async def execute_emergency_stop(self):
        """Execute complete emergency stop protocol."""
        print("\n" + "="*60)
        print("🚨 KIMERA EMERGENCY STOP PROTOCOL ACTIVATED")
        print("="*60)
        print("⚠️  STOPPING ALL TRADING ACTIVITIES...")
        print()
        
        success = True
        
        # Step 1: Cancel all open orders
        print("🛑 Step 1: Cancelling all open orders...")
        if await self.cancel_all_open_orders():
            print("✅ All orders cancelled successfully")
        else:
            print("❌ Some orders may not have been cancelled")
            success = False
            
        print()
        
        # Step 2: Close all positions
        print("🔄 Step 2: Closing all open positions...")
        if await self.close_all_positions():
            print("✅ All positions closed successfully")
        else:
            print("❌ Some positions may not have been closed")
            success = False
            
        print()
        
        # Step 3: Generate report
        print("📄 Step 3: Generating emergency report...")
        report = await self.generate_emergency_report()
        
        if success:
            print("🎉 EMERGENCY STOP COMPLETED SUCCESSFULLY")
        else:
            print("⚠️  EMERGENCY STOP COMPLETED WITH WARNINGS")
            print("   Please check the logs and verify manually")
            
        # Close connection
        if self.connector:
            await self.connector.close()
            
        return report

async def main():
    """Main emergency stop execution."""
    print("🚨 KIMERA EMERGENCY STOP PROTOCOL")
    print("This will immediately stop all trading and close positions.")
    print()
    
    # Confirmation
    try:
        confirm = input("Type 'EMERGENCY' to confirm: ").strip()
        if confirm != 'EMERGENCY':
            print("❌ Emergency stop cancelled.")
            return
    except KeyboardInterrupt:
        print("\n❌ Emergency stop cancelled.")
        return
        
    # Execute emergency stop
    protocol = EmergencyStopProtocol()
    
    if await protocol.initialize():
        await protocol.execute_emergency_stop()
    else:
        print("❌ Failed to initialize emergency stop protocol")

if __name__ == "__main__":
    asyncio.run(main()) 