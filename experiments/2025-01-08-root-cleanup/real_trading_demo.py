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
            print("Testing Binance connection...")
            # Use sync method for balance check
            account = self.exchange.fetch_balance()
            usdt_balance = account.get('USDT', {}).get('free', 0)
            print(f"SUCCESS: Connected to Binance")
            print(f"USDT Balance: ${usdt_balance:.2f}")
            return True
        except Exception as e:
            print(f"ERROR: Failed to connect to Binance: {e}")
            return False
    
    async def get_market_data(self, symbol='BTCUSDT'):
        """Get current market data"""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            print(f"\n{symbol} Market Data:")
            print(f"  Price: ${ticker['last']:.2f}")
            print(f"  24h Change: {ticker['percentage']:.2f}%")
            print(f"  Volume: {ticker['baseVolume']:.2f}")
            return ticker
        except Exception as e:
            print(f"ERROR: Failed to get market data: {e}")
            return None
    
    async def place_small_order(self, symbol='BTCUSDT', side='buy', amount_usd=10):
        """Place a small test order"""
        try:
            print(f"\nPlacing {side.upper()} order for ${amount_usd} of {symbol}...")
            
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
            
            print(f"SUCCESS: Order placed!")
            print(f"  Order ID: {order['id']}")
            print(f"  Side: {order['side'].upper()}")
            print(f"  Amount: {order['amount']:.6f} {symbol[:3]}")
            print(f"  Price: ${order['price']:.2f}")
            print(f"  Status: {order['status']}")
            
            return order
            
        except Exception as e:
            print(f"ERROR: Failed to place order: {e}")
            return None
    
    async def run_demo(self):
        """Run the live trading demonstration"""
        print("=" * 60)
        print("DIRECT BINANCE LIVE TRADING DEMONSTRATION")
        print("=" * 60)
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Step 1: Test connection
        if not await self.check_connection():
            return
        
        # Step 2: Get market data
        btc_data = await self.get_market_data('BTCUSDT')
        eth_data = await self.get_market_data('ETHUSDT')
        
        if not btc_data:
            print("Cannot proceed without market data")
            return
        
        # Step 3: Ask user for confirmation before live trading
        print("\n" + "!" * 60)
        print("WARNING: This will place REAL ORDERS with REAL MONEY")
        print("!" * 60)
        
        response = input("Do you want to proceed with live trading? (yes/no): ")
        
        if response.lower() != 'yes':
            print("Demo cancelled by user")
            return
        
        # Step 4: Place small test orders
        print("\nProceeding with live trading demonstration...")
        
        # Small BTC buy order
        btc_order = await self.place_small_order('BTCUSDT', 'buy', 10)
        
        await asyncio.sleep(2)
        
        # Small ETH buy order  
        eth_order = await self.place_small_order('ETHUSDT', 'buy', 10)
        
        # Step 5: Show final status
        print("\n" + "=" * 60)
        print("LIVE TRADING DEMONSTRATION COMPLETED")
        print("=" * 60)
        print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        print("Orders placed:")
        if btc_order:
            print(f"  BTC Order: {btc_order['id']} - {btc_order['status']}")
        if eth_order:
            print(f"  ETH Order: {eth_order['id']} - {eth_order['status']}")
        
        print("\nNOTE: This was REAL TRADING with REAL MONEY")
        print("Check your Binance account for order details")

async def main():
    try:
        demo = DirectBinanceTradingDemo()
        await demo.run_demo()
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 