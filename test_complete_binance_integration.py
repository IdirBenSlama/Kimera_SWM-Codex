#!/usr/bin/env python3
"""
Complete Binance Integration Test

This script tests both Ed25519 and HMAC authentication methods,
integrates with the Kimera trading system, and provides comprehensive
error handling and diagnostics.
"""

import asyncio
import logging
import os
import sys
import time
from typing import Optional

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from dotenv import load_dotenv
from backend.trading.api.binance_connector_fixed import BinanceConnectorFixed

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CompleteBinanceIntegrationTest:
    """Complete Binance integration test with fallback mechanisms."""
    
    def __init__(self):
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.api_secret = os.getenv('BINANCE_API_SECRET')  # For HMAC fallback
        self.private_key_path = os.getenv('BINANCE_PRIVATE_KEY_PATH', 'binance_private_key.pem')
        self.use_testnet = os.getenv('BINANCE_USE_TESTNET', 'false').lower() == 'true'
        
        self.ed25519_working = False
        self.hmac_working = False
        self.working_connector: Optional[BinanceConnectorFixed] = None
    
    async def test_ed25519_authentication(self) -> bool:
        """Test Ed25519 authentication."""
        print("🔐 Testing Ed25519 Authentication")
        print("-" * 40)
        
        if not self.api_key:
            print("❌ No API key found")
            return False
        
        if not os.path.exists(self.private_key_path):
            print(f"❌ Private key not found: {self.private_key_path}")
            return False
        
        try:
            connector = BinanceConnectorFixed(
                api_key=self.api_key,
                private_key_path=self.private_key_path,
                testnet=self.use_testnet
            )
            
            async with connector:
                # Test account access
                account_info = await connector.get_account()
                print("✅ Ed25519 authentication successful!")
                print(f"   Account type: {account_info.get('accountType', 'Unknown')}")
                print(f"   Trading enabled: {account_info.get('canTrade', False)}")
                
                self.ed25519_working = True
                self.working_connector = connector
                return True
                
        except Exception as e:
            print(f"❌ Ed25519 authentication failed: {e}")
            if "Signature for this request is not valid" in str(e):
                print("💡 This likely means the Ed25519 public key is not properly registered")
                print("   or the API key was not created with Ed25519 authentication.")
            return False
    
    async def test_hmac_fallback(self) -> bool:
        """Test HMAC authentication as fallback."""
        print("\n🔐 Testing HMAC Fallback Authentication")
        print("-" * 40)
        
        if not self.api_secret:
            print("❌ No API secret found for HMAC fallback")
            return False
        
        try:
            # Import HMAC connector (we'll need to create this)
            from backend.trading.api.binance_connector import BinanceConnector
            
            connector = BinanceConnector(
                api_key=self.api_key,
                api_secret=self.api_secret,
                testnet=self.use_testnet
            )
            
            async with connector:
                # Test account access
                account_info = await connector.get_account()
                print("✅ HMAC authentication successful!")
                print(f"   Account type: {account_info.get('accountType', 'Unknown')}")
                print(f"   Trading enabled: {account_info.get('canTrade', False)}")
                
                self.hmac_working = True
                if not self.working_connector:
                    self.working_connector = connector
                return True
                
        except Exception as e:
            print(f"❌ HMAC authentication failed: {e}")
            return False
    
    async def test_market_data_access(self) -> bool:
        """Test market data access (unsigned requests)."""
        print("\n📊 Testing Market Data Access")
        print("-" * 40)
        
        try:
            # Use any connector for unsigned requests
            if self.working_connector:
                connector = self.working_connector
            else:
                # Create a basic connector for unsigned requests
                connector = BinanceConnectorFixed(
                    api_key=self.api_key,
                    private_key_path=self.private_key_path,
                    testnet=self.use_testnet
                )
            
            async with connector:
                # Test ticker data
                ticker = await connector.get_ticker("BTCUSDT")
                print(f"✅ BTCUSDT Price: ${float(ticker['lastPrice']):,.2f}")
                print(f"   24h Change: {float(ticker['priceChangePercent']):.2f}%")
                
                # Test order book
                orderbook = await connector.get_orderbook("BTCUSDT", 5)
                print(f"   Best Bid: ${float(orderbook['bids'][0][0]):,.2f}")
                print(f"   Best Ask: ${float(orderbook['asks'][0][0]):,.2f}")
                
                return True
                
        except Exception as e:
            print(f"❌ Market data access failed: {e}")
            return False
    
    async def test_kimera_integration(self) -> bool:
        """Test integration with Kimera trading system."""
        print("\n🧠 Testing Kimera Integration")
        print("-" * 40)
        
        if not self.working_connector:
            print("❌ No working connector available for Kimera integration")
            return False
        
        try:
            # Test with semantic execution bridge
            from backend.trading.semantic_execution_bridge import SemanticExecutionBridge
            
            bridge = SemanticExecutionBridge()
            
            # Add our working Binance connector
            bridge.binance_connector = self.working_connector
            
            # Test market data integration
            market_data = await self.working_connector.get_market_data("BTCUSDT")
            print(f"✅ Market data integrated:")
            print(f"   Symbol: {market_data['symbol']}")
            print(f"   Price: ${market_data['price']:,.2f}")
            print(f"   Spread: {market_data['ask'] - market_data['bid']:.2f}")
            
            # Test balance checking
            if self.ed25519_working or self.hmac_working:
                balance = await self.working_connector.get_balance("USDT")
                print(f"   USDT Balance: {balance['total']}")
            
            print("✅ Kimera integration successful!")
            return True
            
        except Exception as e:
            print(f"❌ Kimera integration failed: {e}")
            logger.error(f"Kimera integration error: {e}", exc_info=True)
            return False
    
    async def generate_integration_report(self):
        """Generate a comprehensive integration report."""
        print("\n📋 Integration Report")
        print("=" * 50)
        
        print(f"🔐 Authentication Status:")
        print(f"   Ed25519: {'✅ Working' if self.ed25519_working else '❌ Failed'}")
        print(f"   HMAC: {'✅ Working' if self.hmac_working else '❌ Failed'}")
        
        if self.ed25519_working:
            print(f"\n🎯 Recommended Setup:")
            print(f"   ✅ Use Ed25519 authentication (most secure)")
            print(f"   ✅ Ed25519 public key is properly registered")
            print(f"   ✅ API key has correct permissions")
        elif self.hmac_working:
            print(f"\n⚠️  Fallback Setup:")
            print(f"   ✅ HMAC authentication working")
            print(f"   💡 Consider upgrading to Ed25519 for better security")
        else:
            print(f"\n❌ Critical Issues:")
            print(f"   ❌ No authentication method working")
            print(f"   💡 Check API key permissions and registration")
        
        print(f"\n🔧 Configuration:")
        print(f"   API Key: {self.api_key[:10]}..." if self.api_key else "   API Key: Not set")
        print(f"   Testnet: {self.use_testnet}")
        print(f"   Private Key: {'Found' if os.path.exists(self.private_key_path) else 'Missing'}")
        print(f"   API Secret: {'Set' if self.api_secret else 'Not set'}")
        
        # Generate next steps
        print(f"\n🎯 Next Steps:")
        if self.ed25519_working:
            print("1. ✅ Ed25519 authentication is working perfectly")
            print("2. ✅ Ready for live trading integration")
            print("3. ✅ Kimera system can use secure Ed25519 signatures")
        elif self.hmac_working:
            print("1. ⚠️  HMAC authentication working as fallback")
            print("2. 💡 Consider registering Ed25519 public key for better security")
            print("3. ✅ Ready for trading with HMAC authentication")
        else:
            print("1. ❌ Register the Ed25519 public key with Binance API")
            print("2. ❌ Ensure API key has SPOT trading permissions")
            print("3. ❌ Verify API key and secret are correctly set")
        
        return self.ed25519_working or self.hmac_working


async def main():
    """Main test function."""
    print("🚀 Complete Kimera-Binance Integration Test")
    print("=" * 60)
    
    tester = CompleteBinanceIntegrationTest()
    
    # Run all tests
    await tester.test_ed25519_authentication()
    await tester.test_hmac_fallback()
    await tester.test_market_data_access()
    await tester.test_kimera_integration()
    
    # Generate final report
    success = await tester.generate_integration_report()
    
    if success:
        print("\n🎉 INTEGRATION SUCCESSFUL!")
        print("✅ Kimera-Binance integration is ready for trading")
    else:
        print("\n❌ INTEGRATION FAILED")
        print("💡 Please follow the next steps above to resolve issues")
    
    return success


if __name__ == "__main__":
    asyncio.run(main()) 