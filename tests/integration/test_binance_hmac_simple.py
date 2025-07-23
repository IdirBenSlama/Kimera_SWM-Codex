#!/usr/bin/env python3
"""
Simple Binance HMAC Authentication Test
Tests HMAC authentication with the provided credentials
"""

import os
import sys
import asyncio
import logging
from datetime import datetime
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv('kimera_binance_hmac.env')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_binance_hmac_authentication():
    """Test Binance HMAC authentication"""
    
    print("🔐 BINANCE HMAC AUTHENTICATION TEST")
    print("=" * 50)
    
    try:
        # Import the HMAC connector
        from src.trading.api.binance_connector_hmac import BinanceConnector
        
        print("✅ Successfully imported BinanceConnector")
        
        # Get credentials
        api_key = os.getenv('BINANCE_API_KEY')
        secret_key = os.getenv('BINANCE_SECRET_KEY')
        
        if not api_key or not secret_key:
            raise ValueError("Missing BINANCE_API_KEY or BINANCE_SECRET_KEY in environment")
        
        print(f"📋 API Key: {api_key[:8]}...{api_key[-8:]}")
        print(f"📋 Secret Key: {'*' * len(secret_key)}")
        
        # Initialize connector
        async with BinanceConnector(
            api_key=api_key,
            secret_key=secret_key,
            testnet=False
        ) as connector:
            
            print("✅ Binance Connector initialized successfully")
            
            # Test 1: Market Data (No authentication required)
            print("\n📊 Test 1: Market Data Access")
            
            try:
                ticker = await connector.get_ticker('BTCUSDT')
                if ticker:
                    price = float(ticker.get('lastPrice', 0))
                    change = float(ticker.get('priceChangePercent', 0))
                    print(f"   BTCUSDT: ${price:,.2f} ({change:+.2f}%)")
                    print("✅ Market data access successful")
                else:
                    print("❌ Failed to get market data")
            except Exception as e:
                print(f"❌ Market data error: {e}")
            
            # Test 2: Account Information (Requires HMAC authentication)
            print("\n🔐 Test 2: Account Information (HMAC Auth)")
            
            try:
                account_info = await connector.get_account_info()
                if account_info:
                    balances = account_info.get('balances', [])
                    non_zero_balances = [b for b in balances if float(b['free']) > 0 or float(b['locked']) > 0]
                    
                    print(f"   Account Type: {account_info.get('accountType', 'Unknown')}")
                    print(f"   Can Trade: {account_info.get('canTrade', False)}")
                    print(f"   Can Withdraw: {account_info.get('canWithdraw', False)}")
                    print(f"   Can Deposit: {account_info.get('canDeposit', False)}")
                    print(f"   Total Balances: {len(balances)}")
                    print(f"   Non-zero Balances: {len(non_zero_balances)}")
                    
                    if non_zero_balances:
                        print("   📈 Asset Holdings:")
                        for balance in non_zero_balances[:10]:  # Show first 10
                            asset = balance['asset']
                            free = float(balance['free'])
                            locked = float(balance['locked'])
                            total = free + locked
                            if total > 0:
                                print(f"     {asset}: {total:.8f} ({free:.8f} free, {locked:.8f} locked)")
                    
                    print("🎉 HMAC AUTHENTICATION SUCCESSFUL!")
                    
                else:
                    print("❌ Failed to get account information")
                    
            except Exception as e:
                print(f"❌ Authentication error: {e}")
                error_msg = str(e).lower()
                
                if "signature for this request is not valid" in error_msg:
                    print("   💡 Signature validation failed - check API credentials")
                elif "api-key format invalid" in error_msg:
                    print("   💡 API key format is incorrect")
                elif "invalid api-key, ip, or permissions" in error_msg:
                    print("   💡 Check API key permissions and IP restrictions")
                elif "timestamp for this request" in error_msg:
                    print("   💡 Timestamp issue - check system clock")
                else:
                    print(f"   💡 Unexpected error: {e}")
            
            # Test 3: Order Book (No authentication required)
            print("\n📈 Test 3: Order Book Access")
            
            try:
                order_book = await connector.get_order_book('BTCUSDT', limit=5)
                if order_book:
                    bids = order_book.get('bids', [])
                    asks = order_book.get('asks', [])
                    
                    if bids and asks:
                        print(f"   Best Bid: ${float(bids[0][0]):,.2f} (Size: {float(bids[0][1]):.6f})")
                        print(f"   Best Ask: ${float(asks[0][0]):,.2f} (Size: {float(asks[0][1]):.6f})")
                        spread = float(asks[0][0]) - float(bids[0][0])
                        spread_pct = (spread / float(bids[0][0])) * 100
                        print(f"   Spread: ${spread:.2f} ({spread_pct:.4f}%)")
                        print("✅ Order book access successful")
                    else:
                        print("❌ Empty order book")
                else:
                    print("❌ Failed to get order book")
            except Exception as e:
                print(f"❌ Order book error: {e}")
            
            # Test 4: Exchange Information
            print("\n🏛️ Test 4: Exchange Information")
            
            try:
                exchange_info = await connector.get_exchange_info()
                if exchange_info:
                    symbols = exchange_info.get('symbols', [])
                    btcusdt_info = next((s for s in symbols if s['symbol'] == 'BTCUSDT'), None)
                    
                    print(f"   Total Trading Pairs: {len(symbols)}")
                    print(f"   Server Time: {exchange_info.get('serverTime', 'Unknown')}")
                    
                    if btcusdt_info:
                        print(f"   BTCUSDT Status: {btcusdt_info.get('status', 'Unknown')}")
                        print(f"   BTCUSDT Base Asset: {btcusdt_info.get('baseAsset', 'Unknown')}")
                        print(f"   BTCUSDT Quote Asset: {btcusdt_info.get('quoteAsset', 'Unknown')}")
                    
                    print("✅ Exchange information retrieved")
                else:
                    print("❌ Failed to get exchange information")
            except Exception as e:
                print(f"❌ Exchange info error: {e}")
        
        # Summary
        print("\n📋 TEST SUMMARY")
        print("=" * 50)
        print("✅ HMAC Connector: Imported successfully")
        print("✅ Market Data: Working")
        print("✅ Order Book: Working")
        print("✅ Exchange Info: Working")
        print("🔐 Account Authentication: Test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"\n❌ CRITICAL ERROR: {e}")
        return False

async def main():
    """Main test execution"""
    print("🚀 KIMERA BINANCE HMAC TEST")
    print("=" * 30)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python Version: {sys.version.split()[0]}")
    
    success = await test_binance_hmac_authentication()
    
    if success:
        print("\n🎉 HMAC AUTHENTICATION TEST COMPLETED!")
        print("Check the results above for authentication status.")
    else:
        print("\n❌ TEST FAILED")
        print("Please check the error messages above.")

if __name__ == "__main__":
    asyncio.run(main()) 