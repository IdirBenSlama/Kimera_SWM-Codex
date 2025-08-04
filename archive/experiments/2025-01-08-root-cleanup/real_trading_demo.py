#!/usr/bin/env python3
"""
DIRECT BINANCE LIVE TRADING DEMONSTRATION
This bypasses the complex Kimera backend and shows actual live trading
"""

import os
import asyncio
import ccxt
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DirectBinanceTradingDemo:
    def __init__(self):
        # Get API credentials from environment
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.secret_key = os.getenv('BINANCE_SECRET_KEY')
        
        if not self.api_key or not self.secret_key:
            raise ValueError("Missing Binance API credentials in .env file")
        
        # Initialize Binance exchange
        self.exchange = ccxt.binance({
            'apiKey': self.api_key,
            'secret': self.secret_key,
            'sandbox': False,  # Set to True for testnet
            'enableRateLimit': True,
        })
    
    async def check_connection(self):
        """Test Binance connection"""
        try:
            logger.info("Testing Binance connection...")
            # Use sync method for balance check
            account = self.exchange.fetch_balance()
            usdt_balance = account.get('USDT', {}).get('free', 0)
            logger.info(f"SUCCESS: Connected to Binance")
            logger.info(f"USDT Balance: ${usdt_balance:.2f}")
            return True
        except Exception as e:
            logger.info(f"ERROR: Failed to connect to Binance: {e}")
            return False
    
    async def get_market_data(self, symbol='BTCUSDT'):
        """Get current market data"""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            logger.info(f"\n{symbol} Market Data:")
            logger.info(f"  Price: ${ticker['last']:.2f}")
            logger.info(f"  24h Change: {ticker['percentage']:.2f}%")
            logger.info(f"  Volume: {ticker['baseVolume']:.2f}")
            return ticker
        except Exception as e:
            logger.info(f"ERROR: Failed to get market data: {e}")
            return None
    
    async def place_small_order(self, symbol='BTCUSDT', side='buy', amount_usd=10):
        """Place a small test order"""
        try:
            logger.info(f"\nPlacing {side.upper()} order for ${amount_usd} of {symbol}...")
            
            # Get current price
            ticker = self.exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            
            # Calculate quantity
            quantity = amount_usd / current_price
            
            # Place order
            order = self.exchange.create_market_order(
                symbol=symbol,
                side=side,
                amount=quantity
            )
            
            logger.info(f"SUCCESS: Order placed!")
            logger.info(f"  Order ID: {order['id']}")
            logger.info(f"  Side: {order['side'].upper()}")
            logger.info(f"  Amount: {order['amount']:.6f} {symbol[:3]}")
            logger.info(f"  Price: ${order['price']:.2f}")
            logger.info(f"  Status: {order['status']}")
            
            return order
            
        except Exception as e:
            logger.info(f"ERROR: Failed to place order: {e}")
            return None
    
    async def run_demo(self):
        """Run the live trading demonstration"""
        logger.info("=" * 60)
        logger.info("DIRECT BINANCE LIVE TRADING DEMONSTRATION")
        logger.info("=" * 60)
        logger.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info()
        
        # Step 1: Test connection
        if not await self.check_connection():
            return
        
        # Step 2: Get market data
        btc_data = await self.get_market_data('BTCUSDT')
        eth_data = await self.get_market_data('ETHUSDT')
        
        if not btc_data:
            logger.info("Cannot proceed without market data")
            return
        
        # Step 3: Ask user for confirmation before live trading
        logger.info("\n" + "!" * 60)
        logger.info("WARNING: This will place REAL ORDERS with REAL MONEY")
        logger.info("!" * 60)
        
        response = input("Do you want to proceed with live trading? (yes/no): ")
        
        if response.lower() != 'yes':
            logger.info("Demo cancelled by user")
            return
        
        # Step 4: Place small test orders
        logger.info("\nProceeding with live trading demonstration...")
        
        # Small BTC buy order
        btc_order = await self.place_small_order('BTCUSDT', 'buy', 10)
        
        await asyncio.sleep(2)
        
        # Small ETH buy order  
        eth_order = await self.place_small_order('ETHUSDT', 'buy', 10)
        
        # Step 5: Show final status
        logger.info("\n" + "=" * 60)
        logger.info("LIVE TRADING DEMONSTRATION COMPLETED")
        logger.info("=" * 60)
        logger.info(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info()
        logger.info("Orders placed:")
        if btc_order:
            logger.info(f"  BTC Order: {btc_order['id']} - {btc_order['status']}")
        if eth_order:
            logger.info(f"  ETH Order: {eth_order['id']} - {eth_order['status']}")
        
        logger.info("\nNOTE: This was REAL TRADING with REAL MONEY")
        logger.info("Check your Binance account for order details")

async def main():
    try:
        demo = DirectBinanceTradingDemo()
        await demo.run_demo()
    except Exception as e:
        logger.info(f"CRITICAL ERROR: {e}")
        import traceback
import logging
logger = logging.getLogger(__name__)
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 