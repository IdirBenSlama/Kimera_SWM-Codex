import os
#!/usr/bin/env python3
"""
Check exact TRX trading requirements and fix precision
"""

from binance import Client
import json
import logging
logger = logging.getLogger(__name__)

def main():
    # API credentials
    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = "qUn5JqSpYz1GDxFj2X3UF23TYgtxKrTsCbDZEoBMYCPbYZgP4siVLyspkB5HAPl7"
    
    client = Client(api_key, api_secret)
    
    # Get exchange info for TRXUSDT
    exchange_info = client.get_exchange_info()
    
    for symbol_info in exchange_info['symbols']:
        if symbol_info['symbol'] == 'TRXUSDT':
            logger.info("🔍 TRXUSDT Trading Requirements:")
            logger.info("="*50)
            
            for filter_info in symbol_info['filters']:
                if filter_info['filterType'] in ['LOT_SIZE', 'NOTIONAL', 'PRICE_FILTER']:
                    logger.info(f"\n📊 {filter_info['filterType']}:")
                    for key, value in filter_info.items():
                        if key != 'filterType':
                            logger.info(f"   {key}: {value}")
            
            # Get current price
            ticker = client.get_symbol_ticker(symbol='TRXUSDT')
            current_price = float(ticker['price'])
            logger.info(f"\n💰 Current Price: ${current_price:.6f}")
            
            # Calculate proper order size
            for filter_info in symbol_info['filters']:
                if filter_info['filterType'] == 'LOT_SIZE':
                    min_qty = float(filter_info['minQty'])
                    step_size = float(filter_info['stepSize'])
                elif filter_info['filterType'] == 'NOTIONAL':
                    min_notional = float(filter_info['minNotional'])
            
            # Calculate minimum quantity for notional
            min_qty_for_notional = min_notional / current_price
            
            # Use larger requirement
            required_qty = max(min_qty, min_qty_for_notional)
            
            # Round to proper step size
            steps = round(required_qty / step_size)
            if steps * step_size < required_qty:
                steps += 1
            
            final_qty = steps * step_size
            final_value = final_qty * current_price
            
            logger.info(f"\n🎯 CALCULATED ORDER:")
            logger.info(f"   Minimum Quantity: {min_qty}")
            logger.info(f"   Step Size: {step_size}")
            logger.info(f"   Minimum Notional: ${min_notional}")
            logger.info(f"   Qty for Notional: {min_qty_for_notional:.1f}")
            logger.info(f"   Required Steps: {steps}")
            logger.info(f"   Final Quantity: {final_qty}")
            logger.info(f"   Final Value: ${final_value:.2f}")
            
            # Test with this quantity
            logger.info(f"\n⚡ TESTING ORDER WITH {final_qty} TRX...")
            
            try:
                # Test order first
                test_order = client.create_test_order(
                    symbol='TRXUSDT',
                    side='SELL',
                    type='MARKET',
                    quantity=f"{final_qty:.0f}"
                )
                logger.info("✅ Test order successful!")
                
                # Real order
                confirm = input(f"\n⚠️  Execute REAL trade of {final_qty:.0f} TRX? (type 'YES'): ")
                if confirm == 'YES':
                    order = client.order_market_sell(
                        symbol='TRXUSDT',
                        quantity=f"{final_qty:.0f}"
                    )
                    logger.info("🎉 REAL ORDER EXECUTED!")
                    logger.info(json.dumps(order, indent=2))
                    
                    # Save success
                    with open('kimera_first_success.json', 'w') as f:
                        json.dump(order, f, indent=2)
                else:
                    logger.info("❌ Real trade cancelled")
                    
            except Exception as e:
                logger.info(f"❌ Order failed: {e}")
            
            break

if __name__ == "__main__":
    main() 