"""
KIMERA Autonomous Trading System Test Runner
==========================================

Comprehensive test runner for the autonomous trading system.
Tests the complete pipeline from initialization to execution.
"""

import asyncio
import logging
import json
import time
from datetime import datetime
from typing import Dict, Any

from autonomous_trader_config import TradingConfig
from kimera_autonomous_profit_trader import KimeraAutonomousProfitTrader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AutonomousTradingSystemTest:
    """Test runner for the autonomous trading system"""
    
    def __init__(self):
        """Initialize the test runner"""
        self.test_results = {
            'system_initialization': False,
            'configuration_validation': False,
            'component_integration': False,
            'risk_management': False,
            'execution_system': False,
            'cognitive_analysis': False,
            'profit_target_system': False,
            'overall_status': 'UNKNOWN'
        }
        
        logger.info("🧪 Autonomous Trading System Test Runner initialized")
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive system test"""
        logger.info("🚀 Starting comprehensive autonomous trading system test...")
        
        try:
            # Test 1: System Initialization
            await self._test_system_initialization()
            
            # Test 2: Configuration Validation
            await self._test_configuration_validation()
            
            # Test 3: Component Integration
            await self._test_component_integration()
            
            # Test 4: Risk Management
            await self._test_risk_management()
            
            # Test 5: Execution System
            await self._test_execution_system()
            
            # Test 6: Cognitive Analysis
            await self._test_cognitive_analysis()
            
            # Test 7: Profit Target System
            await self._test_profit_target_system()
            
            # Generate final report
            report = self._generate_test_report()
            
            logger.info("✅ Comprehensive system test completed")
            return report
            
        except Exception as e:
            logger.error(f"❌ System test failed: {e}")
            self.test_results['overall_status'] = 'FAILED'
            return self._generate_test_report()
    
    async def _test_system_initialization(self):
        """Test system initialization"""
        logger.info("🧪 Testing system initialization...")
        
        try:
            # Test different configuration modes
            configs = {
                'simulation': TradingConfig('simulation'),
                'testnet': TradingConfig('testnet'),
                'production': TradingConfig('production')
            }
            
            for mode, config in configs.items():
                logger.info(f"   Testing {mode} configuration...")
                
                # Validate configuration
                if not config.validate_config():
                    raise ValueError(f"Configuration validation failed for {mode}")
                
                # Test trader initialization
                trader = KimeraAutonomousProfitTrader(config.get_config())
                
                # Verify components are initialized
                if not trader.portfolio:
                    raise ValueError("Portfolio not initialized")
                
                if not trader.risk_manager:
                    raise ValueError("Risk manager not initialized")
                
                if not trader.execution_bridge:
                    raise ValueError("Execution bridge not initialized")
                
                if not trader.action_interface:
                    raise ValueError("Action interface not initialized")
                
                logger.info(f"   ✅ {mode} configuration initialized successfully")
            
            self.test_results['system_initialization'] = True
            logger.info("✅ System initialization test passed")
            
        except Exception as e:
            logger.error(f"❌ System initialization test failed: {e}")
            self.test_results['system_initialization'] = False
    
    async def _test_configuration_validation(self):
        """Test configuration validation"""
        logger.info("🧪 Testing configuration validation...")
        
        try:
            # Test valid configurations
            valid_configs = [
                {'mode': 'simulation', 'initial_balance': 10000.0, 'profit_target': 2000.0},
                {'mode': 'testnet', 'initial_balance': 1000.0, 'profit_target': 200.0},
                {'mode': 'production', 'initial_balance': 50000.0, 'profit_target': 10000.0}
            ]
            
            for config_params in valid_configs:
                config = TradingConfig(config_params['mode'])
                config.update_config({
                    'initial_balance': config_params['initial_balance'],
                    'profit_target': config_params['profit_target']
                })
                
                if not config.validate_config():
                    raise ValueError(f"Valid configuration failed validation: {config_params}")
                
                logger.info(f"   ✅ Valid configuration passed: {config_params['mode']}")
            
            # Test invalid configurations
            invalid_configs = [
                {'mode': 'simulation', 'initial_balance': -1000.0},  # Negative balance
                {'mode': 'testnet', 'profit_target': -100.0},        # Negative target
                {'mode': 'production', 'risk_per_trade': 0.15},      # Too high risk
                {'mode': 'simulation', 'max_drawdown': 0.8},         # Too high drawdown
            ]
            
            for config_params in invalid_configs:
                config = TradingConfig(config_params['mode'])
                config.update_config({k: v for k, v in config_params.items() if k != 'mode'})
                
                if config.validate_config():
                    raise ValueError(f"Invalid configuration passed validation: {config_params}")
                
                logger.info(f"   ✅ Invalid configuration properly rejected: {config_params}")
            
            self.test_results['configuration_validation'] = True
            logger.info("✅ Configuration validation test passed")
            
        except Exception as e:
            logger.error(f"❌ Configuration validation test failed: {e}")
            self.test_results['configuration_validation'] = False
    
    async def _test_component_integration(self):
        """Test component integration"""
        logger.info("🧪 Testing component integration...")
        
        try:
            # Initialize system
            config = TradingConfig('simulation')
            trader = KimeraAutonomousProfitTrader(config.get_config())
            
            # Test portfolio integration
            initial_balance = trader.portfolio.cash
            if initial_balance != config.config['initial_balance']:
                raise ValueError("Portfolio balance mismatch")
            
            # Test risk manager integration
            risk_summary = trader.risk_manager.get_risk_summary()
            if 'max_position_pct' not in risk_summary:
                raise ValueError("Risk manager integration failed")
            
            # Test execution bridge integration
            execution_analytics = trader.execution_bridge.get_execution_analytics()
            if 'total_orders' not in execution_analytics:
                raise ValueError("Execution bridge integration failed")
            
            # Test action interface integration
            action_summary = trader.action_interface.get_action_summary()
            if 'total_actions' not in action_summary:
                raise ValueError("Action interface integration failed")
            
            logger.info("   ✅ All components integrated successfully")
            
            self.test_results['component_integration'] = True
            logger.info("✅ Component integration test passed")
            
        except Exception as e:
            logger.error(f"❌ Component integration test failed: {e}")
            self.test_results['component_integration'] = False
    
    async def _test_risk_management(self):
        """Test risk management functionality"""
        logger.info("🧪 Testing risk management...")
        
        try:
            # Initialize system
            config = TradingConfig('simulation')
            trader = KimeraAutonomousProfitTrader(config.get_config())
            
            # Test position size validation
            valid_size = trader.risk_manager.validate_position_size('BTCUSDT', 0.01, 45000.0)
            if not valid_size:
                raise ValueError("Valid position size rejected")
            
            invalid_size = trader.risk_manager.validate_position_size('BTCUSDT', 1.0, 45000.0)
            if invalid_size:
                raise ValueError("Invalid position size accepted")
            
            # Test risk score validation
            valid_risk = trader.risk_manager.validate_risk_score(0.5)
            if not valid_risk:
                raise ValueError("Valid risk score rejected")
            
            invalid_risk = trader.risk_manager.validate_risk_score(0.95)
            if invalid_risk:
                raise ValueError("Invalid risk score accepted")
            
            # Test position sizing calculation
            position_size = trader.risk_manager.calculate_position_size('BTCUSDT', 10000.0, 0.02, 45000.0)
            if position_size <= 0 or position_size > 2000.0:  # Should be around 2% of 10k
                raise ValueError(f"Invalid position size calculated: {position_size}")
            
            logger.info("   ✅ Risk management functions working correctly")
            
            self.test_results['risk_management'] = True
            logger.info("✅ Risk management test passed")
            
        except Exception as e:
            logger.error(f"❌ Risk management test failed: {e}")
            self.test_results['risk_management'] = False
    
    async def _test_execution_system(self):
        """Test execution system functionality"""
        logger.info("🧪 Testing execution system...")
        
        try:
            # Initialize system
            config = TradingConfig('simulation')
            trader = KimeraAutonomousProfitTrader(config.get_config())
            
            # Test market data retrieval
            market_data = await trader._get_market_data('BTCUSDT')
            if not market_data or 'price' not in market_data:
                raise ValueError("Market data retrieval failed")
            
            # Test market condition analysis
            conditions = await trader._analyze_market_conditions('BTCUSDT', market_data)
            if not conditions or conditions.overall_score < 0 or conditions.overall_score > 1:
                raise ValueError("Market condition analysis failed")
            
            # Test signal generation
            signal = await trader._generate_trading_signal('BTCUSDT')
            if signal and (signal.confidence < 0 or signal.confidence > 1):
                raise ValueError("Signal generation produced invalid confidence")
            
            # Test execution analytics
            analytics = trader.execution_bridge.get_execution_analytics()
            required_metrics = ['total_orders', 'successful_orders', 'average_latency']
            for metric in required_metrics:
                if metric not in analytics:
                    raise ValueError(f"Missing execution metric: {metric}")
            
            logger.info("   ✅ Execution system functions working correctly")
            
            self.test_results['execution_system'] = True
            logger.info("✅ Execution system test passed")
            
        except Exception as e:
            logger.error(f"❌ Execution system test failed: {e}")
            self.test_results['execution_system'] = False
    
    async def _test_cognitive_analysis(self):
        """Test cognitive analysis functionality"""
        logger.info("🧪 Testing cognitive analysis...")
        
        try:
            # Initialize system
            config = TradingConfig('simulation')
            trader = KimeraAutonomousProfitTrader(config.get_config())
            
            # Test cognitive engine availability
            if not trader.cognitive_engine:
                raise ValueError("Cognitive engine not available")
            
            # Test market analysis with cognitive components
            market_data = await trader._get_market_data('BTCUSDT')
            conditions = await trader._analyze_market_conditions('BTCUSDT', market_data)
            
            # Verify cognitive metrics are present
            if conditions.cognitive_pressure < 0 or conditions.cognitive_pressure > 1:
                raise ValueError("Invalid cognitive pressure score")
            
            if conditions.sentiment_score < 0 or conditions.sentiment_score > 1:
                raise ValueError("Invalid sentiment score")
            
            logger.info("   ✅ Cognitive analysis functions working correctly")
            
            self.test_results['cognitive_analysis'] = True
            logger.info("✅ Cognitive analysis test passed")
            
        except Exception as e:
            logger.error(f"❌ Cognitive analysis test failed: {e}")
            self.test_results['cognitive_analysis'] = False
    
    async def _test_profit_target_system(self):
        """Test profit target system functionality"""
        logger.info("🧪 Testing profit target system...")
        
        try:
            # Initialize system
            config = TradingConfig('simulation')
            trader = KimeraAutonomousProfitTrader(config.get_config())
            
            # Test profit target initialization
            if trader.profit_target.target_amount != config.config['profit_target']:
                raise ValueError("Profit target not initialized correctly")
            
            # Test progress calculation
            trader.profit_target.current_profit = 500.0
            progress = (trader.profit_target.current_profit / trader.profit_target.target_amount) * 100
            if progress != 25.0:  # 500/2000 = 25%
                raise ValueError(f"Profit progress calculation incorrect: {progress}")
            
            # Test profit target reached detection
            trader.profit_target.current_profit = trader.profit_target.target_amount
            if trader.profit_target.current_profit < trader.profit_target.target_amount:
                raise ValueError("Profit target reached detection failed")
            
            # Test performance metrics
            required_metrics = ['total_trades', 'win_rate', 'total_pnl']
            for metric in required_metrics:
                if metric not in trader.performance_metrics:
                    raise ValueError(f"Missing performance metric: {metric}")
            
            logger.info("   ✅ Profit target system functions working correctly")
            
            self.test_results['profit_target_system'] = True
            logger.info("✅ Profit target system test passed")
            
        except Exception as e:
            logger.error(f"❌ Profit target system test failed: {e}")
            self.test_results['profit_target_system'] = False
    
    def _generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        passed_tests = sum(1 for result in self.test_results.values() if result is True)
        total_tests = len(self.test_results) - 1  # Exclude overall_status
        pass_rate = (passed_tests / total_tests) * 100
        
        # Determine overall status
        if pass_rate == 100:
            overall_status = 'EXCELLENT'
            readiness = 'PRODUCTION_READY'
        elif pass_rate >= 85:
            overall_status = 'GOOD'
            readiness = 'NEARLY_READY'
        elif pass_rate >= 70:
            overall_status = 'NEEDS_IMPROVEMENT'
            readiness = 'REQUIRES_FIXES'
        else:
            overall_status = 'POOR'
            readiness = 'NOT_READY'
        
        self.test_results['overall_status'] = overall_status
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'test_results': self.test_results.copy(),
            'summary': {
                'passed_tests': passed_tests,
                'total_tests': total_tests,
                'pass_rate': pass_rate,
                'overall_status': overall_status,
                'readiness': readiness
            },
            'component_status': {
                'system_initialization': '✅ PASSED' if self.test_results['system_initialization'] else '❌ FAILED',
                'configuration_validation': '✅ PASSED' if self.test_results['configuration_validation'] else '❌ FAILED',
                'component_integration': '✅ PASSED' if self.test_results['component_integration'] else '❌ FAILED',
                'risk_management': '✅ PASSED' if self.test_results['risk_management'] else '❌ FAILED',
                'execution_system': '✅ PASSED' if self.test_results['execution_system'] else '❌ FAILED',
                'cognitive_analysis': '✅ PASSED' if self.test_results['cognitive_analysis'] else '❌ FAILED',
                'profit_target_system': '✅ PASSED' if self.test_results['profit_target_system'] else '❌ FAILED',
            },
            'recommendations': self._generate_recommendations(overall_status, pass_rate),
            'deployment_readiness': {
                'ready_for_testnet': pass_rate >= 85,
                'ready_for_production': pass_rate >= 95,
                'recommended_next_steps': self._get_next_steps(readiness)
            }
        }
        
        return report
    
    def _generate_recommendations(self, overall_status: str, pass_rate: float) -> list:
        """Generate recommendations based on test results"""
        recommendations = []
        
        if overall_status == 'EXCELLENT':
            recommendations.extend([
                "System is ready for autonomous trading",
                "Consider starting with testnet for final validation",
                "Monitor performance closely during initial trades",
                "Implement progressive capital allocation"
            ])
        elif overall_status == 'GOOD':
            recommendations.extend([
                "Address failing components before production",
                "Conduct extended testing on testnet",
                "Review risk management settings",
                "Monitor system performance metrics"
            ])
        else:
            recommendations.extend([
                "Fix all failing components before proceeding",
                "Conduct comprehensive system review",
                "Test individual components separately",
                "Consider system redesign if multiple failures"
            ])
        
        # Component-specific recommendations
        if not self.test_results['risk_management']:
            recommendations.append("Critical: Fix risk management system immediately")
        
        if not self.test_results['execution_system']:
            recommendations.append("Critical: Fix execution system before any trading")
        
        if not self.test_results['cognitive_analysis']:
            recommendations.append("Enhance cognitive analysis capabilities")
        
        return recommendations
    
    def _get_next_steps(self, readiness: str) -> list:
        """Get recommended next steps based on readiness"""
        steps = {
            'PRODUCTION_READY': [
                "1. Deploy to testnet for final validation",
                "2. Start with conservative position sizing",
                "3. Monitor performance for 24-48 hours",
                "4. Gradually increase position sizes",
                "5. Implement full autonomous trading"
            ],
            'NEARLY_READY': [
                "1. Fix remaining issues",
                "2. Conduct additional testing",
                "3. Deploy to testnet",
                "4. Monitor performance",
                "5. Re-evaluate for production"
            ],
            'REQUIRES_FIXES': [
                "1. Fix all failing components",
                "2. Re-run comprehensive tests",
                "3. Conduct component-level testing",
                "4. Review system architecture",
                "5. Consider phased deployment"
            ],
            'NOT_READY': [
                "1. Conduct thorough system review",
                "2. Fix all critical issues",
                "3. Test components individually",
                "4. Re-run full test suite",
                "5. Consider system redesign"
            ]
        }
        
        return steps.get(readiness, ["Contact support for assistance"])


async def run_system_test():
    """Run the complete system test"""
    logger.info("🎯 KIMERA AUTONOMOUS TRADING SYSTEM TEST")
    logger.info("=" * 60)
    
    # Initialize test runner
    test_runner = AutonomousTradingSystemTest()
    
    # Run comprehensive test
    results = await test_runner.run_comprehensive_test()
    
    # Save results
    with open('autonomous_trading_system_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("AUTONOMOUS TRADING SYSTEM TEST RESULTS")
    logger.info("=" * 60)
    
    summary = results['summary']
    logger.info(f"Overall Status: {summary['overall_status']}")
    logger.info(f"Pass Rate: {summary['pass_rate']:.1f}%")
    logger.info(f"Passed Tests: {summary['passed_tests']}/{summary['total_tests']}")
    logger.info(f"Readiness: {summary['readiness']}")
    
    logger.info("\nComponent Status:")
    for component, status in results['component_status'].items():
        logger.info(f"  {component}: {status}")
    
    logger.info("\nRecommendations:")
    for rec in results['recommendations']:
        logger.info(f"  • {rec}")
    
    logger.info("\nNext Steps:")
    for step in results['deployment_readiness']['recommended_next_steps']:
        logger.info(f"  {step}")
    
    logger.info("\nDeployment Readiness:")
    logger.info(f"  Ready for Testnet: {'✅' if results['deployment_readiness']['ready_for_testnet'] else '❌'}")
    logger.info(f"  Ready for Production: {'✅' if results['deployment_readiness']['ready_for_production'] else '❌'}")
    
    logger.info("=" * 60)
    logger.info("🎯 Test completed successfully!")
    
    return results


if __name__ == "__main__":
    asyncio.run(run_system_test()) 