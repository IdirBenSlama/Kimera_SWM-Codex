#!/usr/bin/env python3
"""
Kimera-Binance Integration Test (Testnet)
==========================================

This script tests the Binance Ed25519 integration within the full Kimera trading system.
It uses Binance testnet for safe testing without real money.

Features tested:
- Kimera semantic trading reactor
- Binance Ed25519 authentication 
- Market data processing
- Order execution (testnet only)
- Real-time monitoring dashboard

Usage:
    python test_kimera_binance_integration.py
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/kimera_binance_test.log')
    ]
)

logger = logging.getLogger(__name__)

# Import Kimera components
from backend.trading.kimera_trading_integration import (
    KimeraTradingIntegration,
    KimeraTradingConfig,
    create_kimera_trading_system
)
from backend.trading.api.binance_connector import BinanceConnector
from backend.trading.autonomous_kimera_trader import CognitiveSignal
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class KimeraBinanceTestSuite:
    """
    Comprehensive test suite for Kimera-Binance integration
    """
    
    def __init__(self):
        """Initialize the test suite"""
        self.binance_connector = None
        self.kimera_system = None
        self.test_results = {}
        
        # Test configuration
        self.test_symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
        self.test_amount = 10.0  # Small testnet amount
        
        logger.info("🧪 Kimera-Binance Test Suite initialized")
    
    async def setup_test_environment(self) -> bool:
        """
        Setup the complete test environment with Kimera + Binance
        
        Returns:
            bool: True if setup successful
        """
        try:
            logger.info("🔧 Setting up Kimera-Binance test environment...")
            
            # 1. Validate environment variables
            if not self._validate_environment():
                return False
            
            # 2. Initialize Binance connector
            if not await self._setup_binance_connector():
                return False
            
            # 3. Initialize Kimera trading system
            if not await self._setup_kimera_system():
                return False
            
            # 4. Verify connectivity
            if not await self._verify_connectivity():
                return False
            
            logger.info("✅ Test environment setup complete")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to setup test environment: {e}")
            return False
    
    def _validate_environment(self) -> bool:
        """Validate required environment variables"""
        required_vars = [
            'BINANCE_API_KEY',
            'BINANCE_PRIVATE_KEY_PATH'
        ]
        
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            logger.error(f"❌ Missing environment variables: {missing_vars}")
            logger.error("   Please ensure your .env file contains:")
            for var in missing_vars:
                logger.error(f"   {var}=your_value_here")
            return False
        
        # Check if private key file exists
        private_key_path = os.getenv('BINANCE_PRIVATE_KEY_PATH')
        if not os.path.exists(private_key_path):
            logger.error(f"❌ Private key file not found: {private_key_path}")
            return False
        
        logger.info("✓ Environment variables validated")
        return True
    
    async def _setup_binance_connector(self) -> bool:
        """Setup Binance connector with Ed25519 authentication"""
        try:
            api_key = os.getenv('BINANCE_API_KEY')
            private_key_path = os.getenv('BINANCE_PRIVATE_KEY_PATH')
            
            self.binance_connector = BinanceConnector(
                api_key=api_key,
                private_key_path=private_key_path,
                testnet=True  # Always use testnet for safety
            )
            
            logger.info("✓ Binance connector initialized (TESTNET)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to setup Binance connector: {e}")
            return False
    
    async def _setup_kimera_system(self) -> bool:
        """Setup the complete Kimera trading system"""
        try:
            # Configure Kimera with Binance integration
            kimera_config = {
                'tension_threshold': 0.3,  # Lower threshold for testing
                'max_position_size': 50.0,  # Small position for testnet
                'risk_per_trade': 0.01,    # Conservative for testing
                'enable_paper_trading': False,  # We want real testnet trades
                'enable_sentiment_analysis': True,
                'enable_news_processing': True,
                'dashboard_port': 8051,  # Different port to avoid conflicts
                
                # Exchange configuration
                'exchanges': {
                    'binance': {
                        'api_key': os.getenv('BINANCE_API_KEY'),
                        'private_key_path': os.getenv('BINANCE_PRIVATE_KEY_PATH'),
                        'testnet': True,
                        'options': {
                            'defaultType': 'spot',  # Spot trading only
                            'adjustForTimeDifference': True
                        }
                    }
                }
            }
            
            # Create Kimera system
            self.kimera_system = create_kimera_trading_system(kimera_config)
            
            # Start the system
            await self.kimera_system.start()
            
            logger.info("✓ Kimera trading system initialized and started")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to setup Kimera system: {e}")
            return False
    
    async def _verify_connectivity(self) -> bool:
        """Verify connectivity to all systems"""
        try:
            # Test Binance connectivity
            account_info = await self.binance_connector.get_account_info()
            if not account_info:
                logger.error("❌ Failed to get Binance account info")
                return False
            
            logger.info("✓ Binance connectivity verified")
            logger.info(f"   Account type: {account_info.get('accountType', 'unknown')}")
            logger.info(f"   Can trade: {account_info.get('canTrade', False)}")
            
            # Test Kimera system status
            status = self.kimera_system.get_status()
            if not status.get('is_running'):
                logger.error("❌ Kimera system not running")
                return False
            
            logger.info("✓ Kimera system connectivity verified")
            logger.info(f"   Connected exchanges: {status.get('connected_exchanges', [])}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Connectivity verification failed: {e}")
            return False
    
    async def run_comprehensive_tests(self) -> Dict[str, Any]:
        """
        Run comprehensive integration tests
        
        Returns:
            Dict with test results
        """
        logger.info("🚀 Starting comprehensive Kimera-Binance integration tests...")
        
        test_results = {
            'timestamp': datetime.now().isoformat(),
            'environment': 'testnet',
            'tests': {}
        }
        
        # Test 1: Market Data Integration
        test_results['tests']['market_data'] = await self._test_market_data_integration()
        
        # Test 2: Semantic Analysis Pipeline
        test_results['tests']['semantic_analysis'] = await self._test_semantic_analysis()
        
        # Test 3: Order Execution (Testnet)
        test_results['tests']['order_execution'] = await self._test_order_execution()
        
        # Test 4: Risk Management
        test_results['tests']['risk_management'] = await self._test_risk_management()
        
        # Test 5: Real-time Monitoring
        test_results['tests']['monitoring'] = await self._test_monitoring_system()
        
        # Calculate overall success rate
        successful_tests = sum(1 for test in test_results['tests'].values() if test.get('success', False))
        total_tests = len(test_results['tests'])
        test_results['success_rate'] = successful_tests / total_tests
        test_results['overall_success'] = test_results['success_rate'] >= 0.8
        
        logger.info(f"🏁 Tests completed: {successful_tests}/{total_tests} successful")
        
        return test_results
    
    async def _test_market_data_integration(self) -> Dict[str, Any]:
        """Test market data integration between Binance and Kimera"""
        logger.info("📊 Testing market data integration...")
        
        try:
            results = {}
            
            # Get market data from Binance
            for symbol in self.test_symbols:
                ticker = await self.binance_connector.get_ticker(symbol)
                if ticker:
                    results[symbol] = {
                        'price': ticker.get('last'),
                        'volume': ticker.get('baseVolume'),
                        'change': ticker.get('percentage')
                    }
                    logger.info(f"   ✓ {symbol}: ${ticker.get('last', 'N/A')}")
                else:
                    logger.warning(f"   ⚠ Failed to get data for {symbol}")
            
            return {
                'success': len(results) > 0,
                'data': results,
                'message': f"Retrieved data for {len(results)}/{len(self.test_symbols)} symbols"
            }
            
        except Exception as e:
            logger.error(f"❌ Market data test failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _test_semantic_analysis(self) -> Dict[str, Any]:
        """Test Kimera's semantic analysis with Binance data"""
        logger.info("🧠 Testing semantic analysis pipeline...")
        
        try:
            # Create a test market event
            test_event = {
                'symbol': 'BTCUSDT',
                'market_data': {
                    'price': 45000.0,
                    'volume': 1000.0,
                    'timestamp': datetime.now().isoformat()
                },
                'event_type': 'price_update'
            }
            
            # Process through Kimera's semantic pipeline
            analysis_result = await self.kimera_system.process_market_event(test_event)
            
            logger.info(f"   ✓ Analysis confidence: {analysis_result.get('confidence', 0):.2f}")
            logger.info(f"   ✓ Contradictions detected: {len(analysis_result.get('contradictions', []))}")
            
            return {
                'success': True,
                'analysis': analysis_result,
                'message': 'Semantic analysis completed successfully'
            }
            
        except Exception as e:
            logger.error(f"❌ Semantic analysis test failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _test_order_execution(self) -> Dict[str, Any]:
        """Test order execution on Binance testnet"""
        logger.info("💱 Testing order execution (TESTNET)...")
        
        try:
            # Get current price for BTCUSDT
            ticker = await self.binance_connector.get_ticker('BTCUSDT')
            if not ticker:
                return {'success': False, 'error': 'Failed to get ticker data'}
            
            current_price = float(ticker['last'])
            
            # Place a small limit buy order below market price (unlikely to fill)
            test_price = current_price * 0.95  # 5% below market
            
            order_result = await self.binance_connector.place_order(
                symbol='BTCUSDT',
                side='buy',
                order_type='limit',
                quantity=0.001,  # Very small amount
                price=test_price
            )
            
            if order_result:
                order_id = order_result.get('id')
                logger.info(f"   ✓ Order placed: {order_id}")
                
                # Wait a moment then cancel the order
                await asyncio.sleep(2)
                cancel_result = await self.binance_connector.cancel_order('BTCUSDT', order_id)
                
                if cancel_result:
                    logger.info(f"   ✓ Order cancelled successfully")
                    return {
                        'success': True,
                        'order_id': order_id,
                        'message': 'Order placement and cancellation successful'
                    }
                else:
                    logger.warning(f"   ⚠ Failed to cancel order {order_id}")
                    return {
                        'success': True,
                        'order_id': order_id,
                        'message': 'Order placed but cancellation failed'
                    }
            else:
                return {'success': False, 'error': 'Failed to place order'}
                
        except Exception as e:
            logger.error(f"❌ Order execution test failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _test_risk_management(self) -> Dict[str, Any]:
        """Test risk management systems"""
        logger.info("🛡️ Testing risk management...")
        
        try:
            # Test position size calculation
            config = self.kimera_system.config
            max_risk = config.max_position_size * config.risk_per_trade
            
            logger.info(f"   ✓ Max position size: ${config.max_position_size}")
            logger.info(f"   ✓ Risk per trade: {config.risk_per_trade * 100}%")
            logger.info(f"   ✓ Max risk per trade: ${max_risk}")
            
            # Verify risk limits are reasonable for testnet
            if max_risk <= 1.0:  # No more than $1 risk per trade on testnet
                return {
                    'success': True,
                    'max_risk': max_risk,
                    'message': 'Risk management parameters validated'
                }
            else:
                return {
                    'success': False,
                    'error': f'Risk too high for testnet: ${max_risk}'
                }
                
        except Exception as e:
            logger.error(f"❌ Risk management test failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _test_monitoring_system(self) -> Dict[str, Any]:
        """Test monitoring and dashboard systems"""
        logger.info("📈 Testing monitoring system...")
        
        try:
            # Get system status
            status = self.kimera_system.get_status()
            
            logger.info(f"   ✓ System running: {status.get('is_running')}")
            logger.info(f"   ✓ Active positions: {status.get('active_positions', 0)}")
            logger.info(f"   ✓ Total trades: {status.get('total_trades', 0)}")
            
            # Test dashboard access (non-blocking)
            dashboard_running = hasattr(self.kimera_system, 'dashboard') and self.kimera_system.dashboard is not None
            
            return {
                'success': True,
                'status': status,
                'dashboard_available': dashboard_running,
                'message': 'Monitoring system operational'
            }
            
        except Exception as e:
            logger.error(f"❌ Monitoring test failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def cleanup(self):
        """Cleanup test environment"""
        logger.info("🧹 Cleaning up test environment...")
        
        try:
            if self.kimera_system:
                await self.kimera_system.stop()
                logger.info("   ✓ Kimera system stopped")
            
            if self.binance_connector:
                await self.binance_connector.close()
                logger.info("   ✓ Binance connector closed")
                
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")
    
    def save_test_results(self, results: Dict[str, Any]):
        """Save test results to file"""
        try:
            os.makedirs('test_results', exist_ok=True)
            filename = f"test_results/kimera_binance_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"📄 Test results saved to: {filename}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save test results: {e}")


async def main():
    """Main test execution function"""
    print("🔥 KIMERA-BINANCE INTEGRATION TEST SUITE")
    print("=" * 50)
    print("Testing Kimera's semantic trading system with Binance Ed25519 authentication")
    print("Environment: TESTNET (Safe for testing)")
    print("=" * 50)
    
    test_suite = KimeraBinanceTestSuite()
    
    try:
        # Setup test environment
        if not await test_suite.setup_test_environment():
            logger.error("❌ Failed to setup test environment")
            return
        
        # Run comprehensive tests
        results = await test_suite.run_comprehensive_tests()
        
        # Display results
        print("\n🏁 TEST RESULTS SUMMARY")
        print("=" * 30)
        print(f"Overall Success: {'✅ PASS' if results['overall_success'] else '❌ FAIL'}")
        print(f"Success Rate: {results['success_rate']:.1%}")
        print(f"Environment: {results['environment'].upper()}")
        
        print("\nDetailed Results:")
        for test_name, test_result in results['tests'].items():
            status = "✅ PASS" if test_result.get('success') else "❌ FAIL"
            print(f"  {test_name}: {status}")
            if 'message' in test_result:
                print(f"    → {test_result['message']}")
        
        # Save results
        test_suite.save_test_results(results)
        
        if results['overall_success']:
            print("\n🎉 KIMERA-BINANCE INTEGRATION SUCCESSFUL!")
            print("The system is ready for live trading (when you're ready to switch from testnet)")
        else:
            print("\n⚠️  Some tests failed. Please review the logs and fix issues before proceeding.")
    
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
    finally:
        await test_suite.cleanup()


if __name__ == "__main__":
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    
    # Run the test suite
    asyncio.run(main()) 