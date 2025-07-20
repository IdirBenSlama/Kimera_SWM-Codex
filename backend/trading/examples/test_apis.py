"""
Simple API Test Script
=====================

Tests CryptoPanic and TAAPI APIs individually
"""

import asyncio
import os
from backend.trading.connectors.cryptopanic_connector import create_cryptopanic_connector
from backend.trading.connectors.taapi_connector import create_taapi_connector, Indicator, Timeframe
from dotenv import load_dotenv
import argparse

# Initialize structured logger
from backend.utils.kimera_logger import get_system_logger
logger = get_system_logger(__name__)



async def test_cryptopanic():
    """Test CryptoPanic API"""
    logger.info("\n" + "="*50)
    logger.info("Testing CryptoPanic API")
    logger.info("="*50)
    
    api_key = os.getenv('CRYPTOPANIC_API_KEY', '23675a49e161477a7b2b3c8c4a25743ba6777e8e')
    connector = create_cryptopanic_connector(api_key, testnet=True)
    
    async with connector:
        try:
            # Just get a few posts
            posts = await connector.get_posts(limit=5)
            logger.info(f"\n✅ Successfully fetched {len(posts)}")
            
            if posts:
                logger.info("\nFirst post:")
                logger.info(f"  Title: {posts[0].title}")
                logger.info(f"  Sentiment: {posts[0].sentiment_score:.2f}")
                logger.info(f"  Source: {posts[0].source.get('title', 'Unknown')}")
        except Exception as e:
            logger.error(f"\n❌ CryptoPanic API error: {e}")


async def test_taapi():
    """Test TAAPI API"""
    logger.info("\n" + "="*50)
    logger.info("Testing TAAPI API")
    logger.info("="*50)
    
    api_key = os.getenv('TAAPI_API_KEY', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJjbHVlIjoiNjg1NjAxODg4MDZmZjE2NTFlYTE1ZDk5IiwiaWF0IjoxNzUwNDY2OTcwLCJleHAiOjMzMjU0OTMwOTcwfQ.vNLwdY6pKkmcT-Hm1pSjaKnJuw3B0daeDoPvvY4TGfQ')
    connector = create_taapi_connector(api_key)
    
    async with connector:
        try:
            # Get just RSI to test
            logger.info("\nFetching RSI for BTC/USDT...")
            rsi = await connector.get_indicator(
                Indicator.RSI,
                'BTC/USDT',
                exchange='binance',
                timeframe=Timeframe.ONE_HOUR
            )
            
            logger.info(f"\n✅ Successfully fetched RSI")
            logger.info(f"  Value: {rsi.value}")
            logger.info(f"  Timeframe: {rsi.timeframe}")
            
            # Wait before next request due to rate limit
            await asyncio.sleep(2)
            
            # Get MACD
            logger.info("\nFetching MACD for BTC/USDT...")
            macd = await connector.get_indicator(
                Indicator.MACD,
                'BTC/USDT',
                exchange='binance',
                timeframe=Timeframe.ONE_HOUR
            )
            
            logger.info(f"\n✅ Successfully fetched MACD")
            logger.info(f"  Value: {macd.value}")
            
        except Exception as e:
            logger.error(f"\n❌ TAAPI API error: {e}")


async def test_binance_api(args):
    """Test Binance API"""
    logger.info("\n" + "="*50)
    logger.info("Testing Binance API")
    logger.info("="*50)

    # Load environment variables
    load_dotenv(dotenv_path=args.env)
    
    api_key = os.getenv("BINANCE_API_KEY")
    private_key_path = os.getenv("BINANCE_PRIVATE_KEY_PATH")
    use_testnet = os.getenv("BINANCE_USE_TESTNET", "true").lower() == "true"

    if not api_key or not private_key_path:
        logger.error("❌ Binance API Key or Private Key Path not found in environment variables.")
        return

    logger.info(f"🔑 Loaded Binance API Key: ...{api_key[-4:]}")
    logger.info(f"📄 Using private key from: {private_key_path}")
    logger.info(f"🌐 Using Testnet: {use_testnet}")

    try:
        connector = BinanceConnector(
            api_key=api_key,
            private_key_path=private_key_path,
            testnet=use_testnet
        )
        
        # Test 1: Get account info
        logger.info("\n--- 1. Fetching Account Information ---")
        account_info = await connector.get_account()
        logger.info(f"✅ Account information retrieved successfully. Account type: {account_info.get('accountType', 'N/A')}")
        
        # Test 2: Get balances
        logger.info("\n--- 2. Fetching Balances for Key Assets ---")
        assets_to_check = ["BTC", "ETH", "USDT", "BNB"]
        balances = account_info.get("balances", [])
        
        if not balances:
            logger.warning("No balance information found in account details.")
        else:
            for asset in assets_to_check:
                found = False
                for balance in balances:
                    if balance['asset'] == asset:
                        logger.info(f"  - {asset}: Free={balance['free']}, Locked={balance['locked']}")
                        found = True
                        break
                if not found:
                    logger.info(f"  - {asset}: 0 (Not found in balances)")
        
        # Test 3: Get a ticker
        logger.info("\n--- 3. Fetching BTC/USDT Ticker ---")
        ticker = await connector.get_ticker('BTC/USDT')
        logger.info(f"✅ Ticker for BTC/USDT: {ticker['lastPrice']}")

    except FileNotFoundError as e:
        logger.error(f"FATAL: Private key file not found. {e}")
    except Exception as e:
        logger.error(f"❌ An error occurred during the test: {e}")
    finally:
        if 'connector' in locals() and connector:
            await connector.close()
            logger.info("\n--- Connection closed ---")


async def test_phemex_api(args):
    """Test Phemex API"""
    logger.info("\n" + "="*50)
    logger.info("Testing Phemex API")
    logger.info("="*50)

    api_key = os.getenv("PHEMEX_API_KEY")
    api_secret = os.getenv("PHEMEX_API_SECRET")

    if not api_key or not api_secret:
        logger.error("❌ Phemex API Key or Secret not found in environment variables.")
        return

    logger.info(f"Loaded Phemex API Key: ...{api_key[-4:]}")

    try:
        connector = PhemexConnector(
            api_key=api_key,
            api_secret=api_secret
        )

        # Test 1: Get account info
        logger.info("\n--- 1. Fetching Account Information ---")
        account_info = await connector.get_account()
        logger.info(f"✅ Account information retrieved successfully. Account type: {account_info.get('accountType', 'N/A')}")

        # Test 2: Get balances
        logger.info("\n--- 2. Fetching Balances for Key Assets ---")
        assets_to_check = ["BTC", "ETH", "USDT", "BNB"]
        balances = account_info.get("balances", [])
        
        if not balances:
            logger.warning("No balance information found in account details.")
        else:
            for asset in assets_to_check:
                found = False
                for balance in balances:
                    if balance['asset'] == asset:
                        logger.info(f"  - {asset}: Free={balance['free']}, Locked={balance['locked']}")
                        found = True
                        break
                if not found:
                    logger.info(f"  - {asset}: 0 (Not found in balances)")

        # Test 3: Get a ticker
        logger.info("\n--- 3. Fetching BTC/USDT Ticker ---")
        ticker = await connector.get_ticker('BTC/USDT')
        logger.info(f"✅ Ticker for BTC/USDT: {ticker['lastPrice']}")

    except Exception as e:
        logger.error(f"❌ An error occurred during the test: {e}")
    finally:
        if 'connector' in locals() and connector:
            await connector.close()
            logger.info("\n--- Connection closed ---")


async def main():
    """Run all tests"""
    logger.info("API Connection Test")
    logger.info("==================")
    
    await test_cryptopanic()
    await test_taapi()
    
    logger.info("\n" + "="*50)
    logger.info("Test completed!")
    logger.info("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test CryptoPanic, TAAPI, Binance, and Phemex APIs.")
    parser.add_argument("--exchange", type=str, required=True, choices=['binance', 'phemex'], help="The exchange to test.")
    parser.add_argument("--env", type=str, default=".env", help="Path to your .env file.")
    
    args = parser.parse_args()

    if args.exchange == 'binance':
        asyncio.run(test_binance_api(args))
    elif args.exchange == 'phemex':
        asyncio.run(test_phemex_api(args)) 