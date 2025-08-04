#!/usr/bin/env python3
"""
Kimera First Real Trade - Proper Order Size Validation
Performs the first real trade with correct Binance requirements
"""

import os
import logging
from binance import Client
from binance.exceptions import BinanceAPIException
import json
from datetime import datetime
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('kimera_first_trade.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_exchange_info(client, symbol):
    """Get trading pair information and requirements"""
    try:
        exchange_info = client.get_exchange_info()
        
        for symbol_info in exchange_info['symbols']:
            if symbol_info['symbol'] == symbol:
                return symbol_info
        
        return None
    except Exception as e:
        logger.error(f"Error getting exchange info: {e}")
        return None

def calculate_minimum_order(symbol_info, current_price):
    """Calculate minimum valid order size"""
    min_qty = 0
    min_notional = 0
    step_size = 0
    
    for filter_info in symbol_info['filters']:
        if filter_info['filterType'] == 'LOT_SIZE':
            min_qty = float(filter_info['minQty'])
            step_size = float(filter_info['stepSize'])
        elif filter_info['filterType'] == 'NOTIONAL':
            min_notional = float(filter_info['minNotional'])
    
    # Calculate minimum quantity needed to meet notional requirement
    min_qty_for_notional = min_notional / current_price
    
    # Use the larger of the two requirements
    required_qty = max(min_qty, min_qty_for_notional)
    
    # Round up to nearest step size
    required_qty = round(required_qty / step_size) * step_size
    
    # Add small buffer (5%)
    required_qty *= 1.05
    
    return required_qty, min_notional, step_size

def perform_first_trade(client):
    """Perform Kimera's first real trade"""
    
    logger.info("\n" + "="*80)
    logger.info("🚀 KIMERA FIRST REAL TRADE EXECUTION")
    logger.info("="*80)
    
    symbol = 'TRXUSDT'
    
    try:
        # Get current price
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
        
        logger.info(f"\n💰 CURRENT BALANCES:")
        logger.info(f"   TRX: {trx_balance:.2f} (~${trx_balance * current_price:.2f})")
        logger.info(f"   USDT: ${usdt_balance:.2f}")
        
        # Get exchange requirements
        symbol_info = get_exchange_info(client, symbol)
        if not symbol_info:
            logger.error("Could not get symbol information")
            return False
        
        # Calculate minimum order requirements
        min_qty, min_notional, step_size = calculate_minimum_order(symbol_info, current_price)
        
        logger.info(f"\n📊 TRADING REQUIREMENTS:")
        logger.info(f"   Minimum Quantity: {min_qty:.2f} TRX")
        logger.info(f"   Minimum Notional: ${min_notional:.2f}")
        logger.info(f"   Step Size: {step_size}")
        logger.info(f"   Required Order Value: ${min_qty * current_price:.2f}")
        
        # Check if we have enough balance
        if trx_balance < min_qty:
            logger.info(f"\n❌ INSUFFICIENT BALANCE:")
            logger.info(f"   Need: {min_qty:.2f} TRX")
            logger.info(f"   Have: {trx_balance:.2f} TRX")
            logger.info(f"   Consider converting some TRX to USDT first")
            return False
        
        # Execute the trade
        logger.info(f"\n🎯 EXECUTING KIMERA'S FIRST TRADE:")
        logger.info(f"   Strategy: SELL {min_qty:.2f} TRX for USDT")
        logger.info(f"   Expected Proceeds: ~${min_qty * current_price:.2f}")
        
        # Confirm execution
        confirm = input(f"\n⚠️  CONFIRM REAL TRADE? (type 'YES' to proceed): ")
        if confirm != 'YES':
            logger.info("❌ Trade cancelled by user")
            return False
        
        # Place the order
        logger.info(f"\n⚡ PLACING ORDER...")
        
        order = client.order_market_sell(
            symbol=symbol,
            quantity=f"{min_qty:.2f}"
        )
        
        logger.info(f"✅ ORDER EXECUTED SUCCESSFULLY!")
        logger.info(f"\n📋 ORDER DETAILS:")
        logger.info(f"   Order ID: {order['orderId']}")
        logger.info(f"   Symbol: {order['symbol']}")
        logger.info(f"   Side: {order['side']}")
        logger.info(f"   Quantity: {order['executedQty']}")
        logger.info(f"   Status: {order['status']}")
        
        # Save order details
        order_data = {
            "timestamp": datetime.now().isoformat(),
            "order_id": order['orderId'],
            "symbol": order['symbol'],
            "side": order['side'],
            "quantity": order['executedQty'],
            "status": order['status'],
            "full_order": order
        }
        
        with open('kimera_first_trade_success.json', 'w') as f:
            json.dump(order_data, f, indent=2)
        
        logger.info(f"\n💾 Order details saved to: kimera_first_trade_success.json")
        
        # Wait and check new balances
        logger.info(f"\n⏳ Waiting 3 seconds for balance update...")
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
        
        logger.info(f"\n💰 UPDATED BALANCES:")
        logger.info(f"   TRX: {new_trx_balance:.2f} (was {trx_balance:.2f})")
        logger.info(f"   USDT: ${new_usdt_balance:.2f} (was ${usdt_balance:.2f})")
        logger.info(f"   Change: -{trx_balance - new_trx_balance:.2f} TRX, +${new_usdt_balance - usdt_balance:.2f} USDT")
        
        logger.info(f"\n🎉 KIMERA'S FIRST TRADE COMPLETED SUCCESSFULLY!")
        logger.info(f"🚀 KIMERA IS NOW LIVE AND TRADING!")
        
        return True
        
    except BinanceAPIException as e:
        logger.error(f"Binance API Error: {e}")
        logger.info(f"\n❌ Trading Error: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.info(f"\n❌ Unexpected Error: {e}")
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
    
    # Perform first trade
    success = perform_first_trade(client)
    
    if success:
        logger.info(f"\n✅ KIMERA TRADING SYSTEM FULLY OPERATIONAL!")
        logger.info(f"🎯 Ready for autonomous trading strategies")
        logger.info(f"🚀 All systems go for maximum profit optimization")
    else:
        logger.info(f"\n⚠️  Trade not completed - check requirements above")

if __name__ == "__main__":
    main() 