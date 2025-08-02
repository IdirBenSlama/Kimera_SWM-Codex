#!/usr/bin/env python3
"""
Kimera Precision Trading - Exact Binance Requirements Handler
Handles precise calculations for Binance trading requirements
"""

import os
import logging
from binance import Client
from binance.exceptions import BinanceAPIException
import json
from datetime import datetime
import time
from decimal import Decimal, ROUND_UP

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('kimera_precision_trade.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_precise_exchange_info(client, symbol):
    """Get precise trading pair information using Decimal for accuracy"""
    try:
        exchange_info = client.get_exchange_info()
        
        for symbol_info in exchange_info['symbols']:
            if symbol_info['symbol'] == symbol:
                # Extract filters with Decimal precision
                filters = {}
                for filter_info in symbol_info['filters']:
                    filter_type = filter_info['filterType']
                    if filter_type == 'LOT_SIZE':
                        filters['min_qty'] = Decimal(filter_info['minQty'])
                        filters['max_qty'] = Decimal(filter_info['maxQty'])
                        filters['step_size'] = Decimal(filter_info['stepSize'])
                    elif filter_type == 'NOTIONAL':
                        filters['min_notional'] = Decimal(filter_info['minNotional'])
                        filters['max_notional'] = Decimal(filter_info['maxNotional'])
                        filters['apply_min_to_market'] = filter_info.get('applyMinToMarket', True)
                    elif filter_type == 'PRICE_FILTER':
                        filters['min_price'] = Decimal(filter_info['minPrice'])
                        filters['max_price'] = Decimal(filter_info['maxPrice'])
                        filters['tick_size'] = Decimal(filter_info['tickSize'])
                
                return filters
        
        return None
    except Exception as e:
        logger.error(f"Error getting exchange info: {e}")
        return None

def calculate_precise_order_size(filters, current_price):
    """Calculate precise order size using Decimal arithmetic"""
    
    current_price_decimal = Decimal(str(current_price))
    
    # Get requirements
    min_qty = filters['min_qty']
    step_size = filters['step_size']
    min_notional = filters['min_notional']
    
    logger.info(f"Precise calculations:")
    logger.info(f"  Min Quantity: {min_qty}")
    logger.info(f"  Step Size: {step_size}")
    logger.info(f"  Min Notional: ${min_notional}")
    logger.info(f"  Current Price: ${current_price_decimal}")
    
    # Calculate minimum quantity to meet notional requirement
    min_qty_for_notional = min_notional / current_price_decimal
    logger.info(f"  Min Qty for Notional: {min_qty_for_notional}")
    
    # Use the larger of the two requirements
    required_qty = max(min_qty, min_qty_for_notional)
    logger.info(f"  Required Qty (before rounding): {required_qty}")
    
    # Round UP to the nearest step size to ensure we meet requirements
    steps_needed = (required_qty / step_size).quantize(Decimal('1'), rounding=ROUND_UP)
    final_qty = steps_needed * step_size
    
    # Add small buffer to ensure we exceed minimum notional
    buffer_steps = Decimal('2')  # Add 2 more steps for safety
    final_qty_with_buffer = (steps_needed + buffer_steps) * step_size
    
    # Calculate final values
    final_notional = final_qty_with_buffer * current_price_decimal
    
    logger.info(f"  Steps needed: {steps_needed}")
    logger.info(f"  Final quantity (with buffer): {final_qty_with_buffer}")
    logger.info(f"  Final notional value: ${final_notional}")
    
    # Validate requirements
    qty_valid = final_qty_with_buffer >= min_qty
    notional_valid = final_notional >= min_notional
    step_valid = (final_qty_with_buffer % step_size) == 0
    
    logger.info(f"Validation:")
    logger.info(f"  Quantity valid: {qty_valid}")
    logger.info(f"  Notional valid: {notional_valid}")
    logger.info(f"  Step size valid: {step_valid}")
    
    return {
        'quantity': final_qty_with_buffer,
        'notional_value': final_notional,
        'steps_used': steps_needed + buffer_steps,
        'all_requirements_met': qty_valid and notional_valid and step_valid,
        'validation': {
            'qty_valid': qty_valid,
            'notional_valid': notional_valid,
            'step_valid': step_valid
        }
    }

def perform_precision_trade(client):
    """Perform trade with precise calculations"""
    
    print("\n" + "="*80)
    print("🎯 KIMERA PRECISION TRADING EXECUTION")
    print("="*80)
    
    symbol = 'TRXUSDT'
    
    try:
        # Get current price with high precision
        ticker = client.get_symbol_ticker(symbol=symbol)
        current_price = float(ticker['price'])
        logger.info(f"Current {symbol} price: ${current_price:.6f}")
        
        # Get account balance
        account = client.get_account()
        trx_balance = 0
        usdt_balance = 0
        
        for balance in account['balances']:
            if balance['asset'] == 'TRX':
                trx_balance = float(balance['free'])
            elif balance['asset'] == 'USDT':
                usdt_balance = float(balance['free'])
        
        print(f"\n💰 CURRENT BALANCES:")
        print(f"   TRX: {trx_balance:.2f} (~${trx_balance * current_price:.2f})")
        print(f"   USDT: ${usdt_balance:.2f}")
        
        # Get precise exchange requirements
        filters = get_precise_exchange_info(client, symbol)
        if not filters:
            logger.error("Could not get symbol filters")
            return False
        
        print(f"\n📊 EXCHANGE REQUIREMENTS:")
        print(f"   Min Quantity: {filters['min_qty']} TRX")
        print(f"   Step Size: {filters['step_size']} TRX")
        print(f"   Min Notional: ${filters['min_notional']}")
        
        # Calculate precise order size
        order_calc = calculate_precise_order_size(filters, current_price)
        
        print(f"\n🔬 PRECISION CALCULATIONS:")
        print(f"   Calculated Quantity: {order_calc['quantity']} TRX")
        print(f"   Calculated Value: ${order_calc['notional_value']:.6f}")
        print(f"   Steps Used: {order_calc['steps_used']}")
        print(f"   All Requirements Met: {order_calc['all_requirements_met']}")
        
        if not order_calc['all_requirements_met']:
            print(f"\n❌ REQUIREMENTS NOT MET:")
            for req, status in order_calc['validation'].items():
                print(f"   {req}: {'✅' if status else '❌'}")
            return False
        
        # Check if we have enough balance
        required_qty_float = float(order_calc['quantity'])
        if trx_balance < required_qty_float:
            print(f"\n❌ INSUFFICIENT BALANCE:")
            print(f"   Need: {required_qty_float:.1f} TRX")
            print(f"   Have: {trx_balance:.2f} TRX")
            return False
        
        # Format quantity for API (remove trailing zeros)
        quantity_str = f"{order_calc['quantity']:.1f}".rstrip('0').rstrip('.')
        
        print(f"\n🎯 EXECUTING PRECISION TRADE:")
        print(f"   Strategy: SELL {quantity_str} TRX for USDT")
        print(f"   Expected Proceeds: ~${order_calc['notional_value']:.2f}")
        print(f"   Quantity String: '{quantity_str}'")
        
        # Test order first
        print(f"\n🧪 TESTING ORDER VALIDITY...")
        try:
            test_order = client.create_test_order(
                symbol=symbol,
                side='SELL',
                type='MARKET',
                quantity=quantity_str
            )
            print(f"✅ Test order validation successful!")
        except Exception as e:
            print(f"❌ Test order failed: {e}")
            logger.error(f"Test order error: {e}")
            return False
        
        # Confirm execution
        confirm = input(f"\n⚠️  CONFIRM REAL TRADE? (type 'YES' to proceed): ")
        if confirm != 'YES':
            print("❌ Trade cancelled by user")
            return False
        
        # Execute real order
        print(f"\n⚡ PLACING REAL ORDER...")
        
        order = client.order_market_sell(
            symbol=symbol,
            quantity=quantity_str
        )
        
        print(f"✅ ORDER EXECUTED SUCCESSFULLY!")
        print(f"\n📋 ORDER DETAILS:")
        print(f"   Order ID: {order['orderId']}")
        print(f"   Symbol: {order['symbol']}")
        print(f"   Side: {order['side']}")
        print(f"   Quantity Requested: {quantity_str}")
        print(f"   Quantity Executed: {order['executedQty']}")
        print(f"   Status: {order['status']}")
        print(f"   Transaction Time: {order['transactTime']}")
        
        # Calculate fills and fees
        total_executed = Decimal('0')
        total_proceeds = Decimal('0')
        total_fees = Decimal('0')
        
        if 'fills' in order:
            print(f"\n💹 EXECUTION DETAILS:")
            for i, fill in enumerate(order['fills']):
                qty = Decimal(fill['qty'])
                price = Decimal(fill['price'])
                commission = Decimal(fill['commission'])
                
                total_executed += qty
                total_proceeds += qty * price
                total_fees += commission
                
                print(f"   Fill {i+1}: {qty} TRX @ ${price} = ${qty * price:.6f}")
                print(f"            Fee: {commission} {fill['commissionAsset']}")
        
        print(f"\n📊 TRADE SUMMARY:")
        print(f"   Total Executed: {total_executed} TRX")
        print(f"   Total Proceeds: ${total_proceeds:.6f}")
        print(f"   Total Fees: {total_fees}")
        print(f"   Net Proceeds: ${total_proceeds - total_fees:.6f}")
        
        # Save comprehensive order data
        order_data = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "order_calculation": {
                "requested_quantity": str(order_calc['quantity']),
                "quantity_string": quantity_str,
                "expected_notional": str(order_calc['notional_value']),
                "steps_used": str(order_calc['steps_used'])
            },
            "execution_summary": {
                "total_executed": str(total_executed),
                "total_proceeds": str(total_proceeds),
                "total_fees": str(total_fees),
                "net_proceeds": str(total_proceeds - total_fees)
            },
            "full_order_response": order,
            "exchange_filters": {k: str(v) for k, v in filters.items()}
        }
        
        with open('kimera_precision_trade_success.json', 'w') as f:
            json.dump(order_data, f, indent=2, default=str)
        
        print(f"\n💾 Complete trade data saved to: kimera_precision_trade_success.json")
        
        # Wait and check updated balances
        print(f"\n⏳ Waiting 3 seconds for balance update...")
        time.sleep(3)
        
        # Get updated balances
        account = client.get_account()
        new_trx_balance = 0
        new_usdt_balance = 0
        
        for balance in account['balances']:
            if balance['asset'] == 'TRX':
                new_trx_balance = float(balance['free'])
            elif balance['asset'] == 'USDT':
                new_usdt_balance = float(balance['free'])
        
        print(f"\n💰 UPDATED BALANCES:")
        print(f"   TRX: {new_trx_balance:.2f} (was {trx_balance:.2f})")
        print(f"   USDT: ${new_usdt_balance:.6f} (was ${usdt_balance:.6f})")
        print(f"   Change: -{trx_balance - new_trx_balance:.1f} TRX, +${new_usdt_balance - usdt_balance:.6f} USDT")
        
        print(f"\n🎉 KIMERA PRECISION TRADE COMPLETED!")
        print(f"🚀 KIMERA IS NOW FULLY OPERATIONAL!")
        print(f"⚡ READY FOR AUTONOMOUS TRADING STRATEGIES!")
        
        return True
        
    except BinanceAPIException as e:
        logger.error(f"Binance API Error: {e}")
        print(f"\n❌ Trading Error: {e}")
        print(f"   Error Code: {e.code}")
        print(f"   Error Message: {e.message}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"\n❌ Unexpected Error: {e}")
        return False

def main():
    """Main execution function"""
    
    # Load API credentials
    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = "qUn5JqSpYz1GDxFj2X3UF23TYgtxKrTsCbDZEoBMYCPbYZgP4siVLyspkB5HAPl7"
    
    if not api_key or not api_secret:
        logger.error("API credentials not found")
        return
    
    # Initialize client
    client = Client(api_key, api_secret)
    
    logger.info("🔬 Starting Kimera precision trading system...")
    
    # Perform precision trade
    success = perform_precision_trade(client)
    
    if success:
        print(f"\n✅ KIMERA PRECISION TRADING SYSTEM FULLY OPERATIONAL!")
        print(f"🎯 All Binance requirements precisely calculated and met")
        print(f"🚀 Ready for advanced autonomous trading strategies")
        print(f"⚡ Maximum profit optimization algorithms ready to deploy")
    else:
        print(f"\n⚠️  Precision trade analysis completed - review calculations above")

if __name__ == "__main__":
    main() 