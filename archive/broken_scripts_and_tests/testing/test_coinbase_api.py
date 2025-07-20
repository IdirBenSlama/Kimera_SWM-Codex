#!/usr/bin/env python3
"""
COINBASE PRO API TEST - FUNDS VERIFICATION
=========================================

This script tests the Coinbase Pro API connection and checks available funds
for autonomous trading deployment.

MISSION: Verify API connectivity and available trading capital
"""

import os
import json
import time
import hmac
import hashlib
import base64
import requests
from datetime import datetime
from typing import Dict, List, Any
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - COINBASE TEST - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CoinbaseProAPITest:
    """Test Coinbase Pro API connection and check available funds"""
    
    def __init__(self, api_key: str, api_secret: str, passphrase: str, sandbox: bool = True):
        """Initialize Coinbase Pro API test"""
        
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.sandbox = sandbox
        
        # API endpoints
        if sandbox:
            self.base_url = "https://api-public.sandbox.pro.coinbase.com"
            logger.info("🧪 Using SANDBOX environment (test mode)")
        else:
            self.base_url = "https://api.pro.coinbase.com"
            logger.warning("💰 Using LIVE environment (real money)")
        
        logger.info("Coinbase Pro API Test initialized")
    
    def _generate_signature(self, timestamp: str, method: str, path: str, body: str = '') -> str:
        """Generate API signature for authentication"""
        
        message = timestamp + method + path + body
        hmac_key = base64.b64decode(self.api_secret)
        signature = hmac.new(hmac_key, message.encode(), hashlib.sha256)
        signature_b64 = base64.b64encode(signature.digest()).decode()
        
        return signature_b64
    
    def _make_request(self, method: str, endpoint: str, params: Dict = None) -> Dict:
        """Make authenticated API request"""
        
        timestamp = str(time.time())
        path = endpoint
        
        # Generate signature
        signature = self._generate_signature(timestamp, method, path)
        
        # Headers
        headers = {
            'CB-ACCESS-KEY': self.api_key,
            'CB-ACCESS-SIGN': signature,
            'CB-ACCESS-TIMESTAMP': timestamp,
            'CB-ACCESS-PASSPHRASE': self.passphrase,
            'Content-Type': 'application/json'
        }
        
        # Make request
        url = self.base_url + endpoint
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, params=params, timeout=10)
            elif method == 'POST':
                response = requests.post(url, headers=headers, json=params, timeout=10)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            raise
    
    async def test_api_connection(self) -> Dict[str, Any]:
        """Test API connection and authentication"""
        
        logger.info("🔗 TESTING API CONNECTION")
        
        test_results = {
            'connection_status': 'unknown',
            'authentication_status': 'unknown',
            'api_permissions': [],
            'server_time': None,
            'error_details': None
        }
        
        try:
            # Test 1: Server time (no auth required)
            logger.info("Testing server connectivity...")
            time_response = requests.get(f"{self.base_url}/time", timeout=10)
            time_response.raise_for_status()
            server_time_data = time_response.json()
            
            test_results['connection_status'] = 'success'
            test_results['server_time'] = server_time_data
            logger.info("✅ Server connection successful")
            logger.info(f"   Server time: {server_time_data}")
            
            # Test 2: Authentication (requires auth)
            logger.info("Testing API authentication...")
            accounts_data = self._make_request('GET', '/accounts')
            
            test_results['authentication_status'] = 'success'
            logger.info("✅ API authentication successful")
            logger.info(f"   Found {len(accounts_data)} accounts")
            
            # Test 3: Check permissions
            logger.info("Checking API permissions...")
            try:
                # Try to access different endpoints to check permissions
                orders_data = self._make_request('GET', '/orders')
                test_results['api_permissions'].append('view_orders')
                logger.info("✅ View orders permission confirmed")
            except Exception as e:
                logger.warning(f"⚠️ View orders permission: {str(e)}")
            
            try:
                fills_data = self._make_request('GET', '/fills')
                test_results['api_permissions'].append('view_fills')
                logger.info("✅ View fills permission confirmed")
            except Exception as e:
                logger.warning(f"⚠️ View fills permission: {str(e)}")
            
        except Exception as e:
            test_results['connection_status'] = 'failed'
            test_results['authentication_status'] = 'failed'
            test_results['error_details'] = str(e)
            logger.error(f"❌ API connection test failed: {str(e)}")
            raise
        
        return test_results
    
    async def check_available_funds(self) -> Dict[str, Any]:
        """Check available funds in all accounts"""
        
        logger.info("💰 CHECKING AVAILABLE FUNDS")
        
        funds_data = {
            'total_accounts': 0,
            'accounts_with_balance': 0,
            'total_usd_value': 0.0,
            'account_details': [],
            'trading_pairs_available': [],
            'recommended_trading_capital': 0.0
        }
        
        try:
            # Get all accounts
            accounts = self._make_request('GET', '/accounts')
            funds_data['total_accounts'] = len(accounts)
            
            logger.info(f"Found {len(accounts)} accounts")
            
            # Process each account
            for account in accounts:
                currency = account['currency']
                balance = float(account['balance'])
                available = float(account['available'])
                hold = float(account['hold'])
                
                if balance > 0 or available > 0:
                    funds_data['accounts_with_balance'] += 1
                    
                    account_info = {
                        'currency': currency,
                        'balance': balance,
                        'available': available,
                        'hold': hold,
                        'usd_value': 0.0
                    }
                    
                    # Get USD value for major currencies
                    if currency == 'USD':
                        account_info['usd_value'] = available
                        funds_data['total_usd_value'] += available
                        
                    elif currency in ['BTC', 'ETH', 'LTC', 'BCH'] and available > 0:
                        try:
                            # Get current price
                            ticker = self._make_request('GET', f'/products/{currency}-USD/ticker')
                            price = float(ticker['price'])
                            usd_value = available * price
                            account_info['usd_value'] = usd_value
                            account_info['current_price'] = price
                            funds_data['total_usd_value'] += usd_value
                            
                        except Exception as e:
                            logger.warning(f"Could not get price for {currency}: {str(e)}")
                    
                    funds_data['account_details'].append(account_info)
                    
                    logger.info(f"💰 {currency}: {available:.8f} available (${account_info['usd_value']:.2f})")
            
            # Check available trading pairs
            logger.info("Checking available trading pairs...")
            try:
                products = self._make_request('GET', '/products')
                major_pairs = ['BTC-USD', 'ETH-USD', 'LTC-USD', 'BCH-USD']
                
                for product in products:
                    if product['id'] in major_pairs and product['status'] == 'online':
                        funds_data['trading_pairs_available'].append({
                            'pair': product['id'],
                            'base_currency': product['base_currency'],
                            'quote_currency': product['quote_currency'],
                            'min_market_funds': float(product.get('min_market_funds', '0')),
                            'max_market_funds': float(product.get('max_market_funds', '1000000'))
                        })
                        
                logger.info(f"✅ {len(funds_data['trading_pairs_available'])} major trading pairs available")
                
            except Exception as e:
                logger.warning(f"Could not check trading pairs: {str(e)}")
            
            # Calculate recommended trading capital
            usd_available = funds_data['total_usd_value']
            if usd_available >= 100:
                # Conservative: use up to 50% of available funds for trading
                funds_data['recommended_trading_capital'] = min(usd_available * 0.5, 10000)
            else:
                funds_data['recommended_trading_capital'] = usd_available * 0.8
            
            logger.info(f"💰 TOTAL USD VALUE: ${funds_data['total_usd_value']:.2f}")
            logger.info(f"📊 RECOMMENDED TRADING CAPITAL: ${funds_data['recommended_trading_capital']:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Failed to check funds: {str(e)}")
            raise
        
        return funds_data
    
    async def generate_trading_assessment(self, funds_data: Dict) -> Dict[str, Any]:
        """Generate trading readiness assessment"""
        
        logger.info("📊 GENERATING TRADING ASSESSMENT")
        
        assessment = {
            'trading_ready': False,
            'readiness_score': 0,
            'recommendations': [],
            'risk_assessment': 'unknown',
            'deployment_options': []
        }
        
        total_usd = funds_data['total_usd_value']
        recommended_capital = funds_data['recommended_trading_capital']
        
        # Assess trading readiness
        if total_usd >= 1000:
            assessment['trading_ready'] = True
            assessment['readiness_score'] = 100
            assessment['risk_assessment'] = 'low'
            assessment['recommendations'].append("Excellent capital base for autonomous trading")
            assessment['deployment_options'].extend([
                "Conservative: $100-500 per session",
                "Moderate: $500-2000 per session", 
                "Aggressive: $2000+ per session"
            ])
            
        elif total_usd >= 500:
            assessment['trading_ready'] = True
            assessment['readiness_score'] = 85
            assessment['risk_assessment'] = 'moderate'
            assessment['recommendations'].append("Good capital base for conservative trading")
            assessment['deployment_options'].extend([
                "Conservative: $50-200 per session",
                "Moderate: $200-500 per session"
            ])
            
        elif total_usd >= 100:
            assessment['trading_ready'] = True
            assessment['readiness_score'] = 70
            assessment['risk_assessment'] = 'moderate-high'
            assessment['recommendations'].append("Adequate capital for conservative trading")
            assessment['deployment_options'].append("Conservative: $25-100 per session")
            
        elif total_usd >= 25:
            assessment['trading_ready'] = True
            assessment['readiness_score'] = 50
            assessment['risk_assessment'] = 'high'
            assessment['recommendations'].append("Minimal capital - use extreme caution")
            assessment['deployment_options'].append("Micro-trading: $5-25 per session")
            
        else:
            assessment['trading_ready'] = False
            assessment['readiness_score'] = 0
            assessment['risk_assessment'] = 'very-high'
            assessment['recommendations'].append("Insufficient capital for safe autonomous trading")
        
        # Additional recommendations
        if len(funds_data['trading_pairs_available']) >= 4:
            assessment['recommendations'].append("All major trading pairs available")
        else:
            assessment['recommendations'].append("Limited trading pairs - may affect strategy")
        
        if funds_data['accounts_with_balance'] == 1 and any(acc['currency'] == 'USD' for acc in funds_data['account_details']):
            assessment['recommendations'].append("USD-only account - good for direct trading")
        elif funds_data['accounts_with_balance'] > 1:
            assessment['recommendations'].append("Multiple currency balances - consider consolidation")
        
        logger.info(f"📊 TRADING READINESS: {'✅ READY' if assessment['trading_ready'] else '❌ NOT READY'}")
        logger.info(f"📊 READINESS SCORE: {assessment['readiness_score']}/100")
        logger.info(f"📊 RISK ASSESSMENT: {assessment['risk_assessment'].upper()}")
        
        return assessment
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive API test and funds check"""
        
        logger.info("=" * 80)
        logger.info("🚀 COINBASE PRO API COMPREHENSIVE TEST")
        logger.info("=" * 80)
        logger.info(f"Environment: {'SANDBOX (Test)' if self.sandbox else 'LIVE (Real Money)'}")
        logger.info("Testing API connection, authentication, and available funds")
        logger.info("=" * 80)
        
        test_start = datetime.now()
        
        try:
            # Test 1: API Connection
            connection_results = await self.test_api_connection()
            
            # Test 2: Check Funds
            funds_results = await self.check_available_funds()
            
            # Test 3: Trading Assessment
            trading_assessment = await self.generate_trading_assessment(funds_results)
            
            # Compile comprehensive results
            comprehensive_results = {
                'test_timestamp': test_start.isoformat(),
                'test_duration': (datetime.now() - test_start).total_seconds(),
                'environment': 'sandbox' if self.sandbox else 'live',
                'connection_test': connection_results,
                'funds_analysis': funds_results,
                'trading_assessment': trading_assessment,
                'overall_status': 'ready' if trading_assessment['trading_ready'] else 'not_ready'
            }
            
            # Save results
            filename = f"coinbase_api_test_{int(time.time())}.json"
            with open(filename, 'w') as f:
                json.dump(comprehensive_results, f, indent=2, default=str)
            
            # Generate summary report
            logger.info("\n" + "=" * 80)
            logger.info("🏆 COINBASE PRO API TEST COMPLETE")
            logger.info("=" * 80)
            logger.info(f"Connection Status: {'✅ SUCCESS' if connection_results['connection_status'] == 'success' else '❌ FAILED'}")
            logger.info(f"Authentication: {'✅ SUCCESS' if connection_results['authentication_status'] == 'success' else '❌ FAILED'}")
            logger.info(f"Total USD Value: ${funds_results['total_usd_value']:.2f}")
            logger.info(f"Recommended Capital: ${funds_results['recommended_trading_capital']:.2f}")
            logger.info(f"Trading Ready: {'✅ YES' if trading_assessment['trading_ready'] else '❌ NO'}")
            logger.info(f"Readiness Score: {trading_assessment['readiness_score']}/100")
            logger.info(f"Risk Level: {trading_assessment['risk_assessment'].upper()}")
            
            if trading_assessment['deployment_options']:
                logger.info("\n💰 DEPLOYMENT OPTIONS:")
                for option in trading_assessment['deployment_options']:
                    logger.info(f"   - {option}")
            
            if trading_assessment['recommendations']:
                logger.info("\n📋 RECOMMENDATIONS:")
                for rec in trading_assessment['recommendations']:
                    logger.info(f"   - {rec}")
            
            logger.info(f"\n📊 Detailed results saved: {filename}")
            logger.info("=" * 80)
            
            return comprehensive_results
            
        except Exception as e:
            logger.error(f"❌ Comprehensive test failed: {str(e)}")
            raise

async def main():
    """Main test execution"""
    
    print("COINBASE PRO API FUNDS VERIFICATION")
    print("=" * 80)
    print("This will test your Coinbase Pro API connection and check available funds")
    print("for autonomous trading deployment.")
    print("=" * 80)
    
    # Get API credentials
    print("\n🔧 API CONFIGURATION:")
    
    # Option to use environment variables
    use_env = input("Use environment variables for API credentials? (y/n): ").strip().lower() == 'y'
    
    if use_env:
        api_key = os.getenv('COINBASE_API_KEY')
        api_secret = os.getenv('COINBASE_API_SECRET')
        passphrase = os.getenv('COINBASE_PASSPHRASE')
        
        if not all([api_key, api_secret, passphrase]):
            print("❌ Environment variables not found. Please set:")
            print("   COINBASE_API_KEY")
            print("   COINBASE_API_SECRET")
            print("   COINBASE_PASSPHRASE")
            return
    else:
        api_key = input("API Key: ").strip()
        api_secret = input("API Secret: ").strip()
        passphrase = input("Passphrase: ").strip()
        
        if not all([api_key, api_secret, passphrase]):
            print("❌ All credentials are required")
            return
    
    # Environment selection
    environment = input("Use sandbox environment? (y/n, default: y): ").strip().lower()
    use_sandbox = environment != 'n'
    
    if not use_sandbox:
        print("\n⚠️  LIVE ENVIRONMENT WARNING:")
        print("This will connect to the LIVE Coinbase Pro API with real money.")
        confirmation = input("Type 'LIVE TEST' to proceed: ").strip()
        
        if confirmation != "LIVE TEST":
            print("Test cancelled - using sandbox instead")
            use_sandbox = True
    
    print(f"\n🚀 STARTING API TEST:")
    print(f"   Environment: {'SANDBOX (Test)' if use_sandbox else 'LIVE (Real Money)'}")
    print(f"   Testing: Connection, Authentication, Funds")
    
    input("Press Enter to start test...")
    
    # Initialize test system
    api_test = CoinbaseProAPITest(
        api_key=api_key,
        api_secret=api_secret,
        passphrase=passphrase,
        sandbox=use_sandbox
    )
    
    try:
        # Run comprehensive test
        results = await api_test.run_comprehensive_test()
        
        print(f"\n✅ Test completed successfully!")
        print(f"Check the generated JSON file for detailed results.")
        
    except KeyboardInterrupt:
        logger.info("\n\nTest interrupted by user")
        
    except Exception as e:
        logger.error(f"\n\nTest failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import asyncio
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nTest cancelled by user")
    except Exception as e:
        print(f"\n\nTest failed: {str(e)}")
        import traceback
        traceback.print_exc() 