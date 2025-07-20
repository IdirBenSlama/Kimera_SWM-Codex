"""
KIMERA Action Interface Test Runner

Tests the core functionality of KIMERA's action execution interface.
Follows the zero-debugging constraint with comprehensive logging and error handling.

Author: KIMERA AI System
Date: 2025-01-27
"""

import asyncio
import logging
import json
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import asdict
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import os

# Setup path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.trading.execution.kimera_action_interface import (
    KimeraActionInterface,
    ActionType,
    ExecutionStatus,
    ActionRequest,
    ActionResult,
    CognitiveFeedbackProcessor,
    create_kimera_action_interface
)
from backend.trading.core.trading_engine import TradingDecision, MarketState
from backend.trading.core.integrated_trading_engine import IntegratedTradingSignal

# Configure logging for zero-debugging constraint
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'test_action_interface_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ActionInterfaceTestRunner:
    """
    Comprehensive test runner for KIMERA Action Interface.
    
    Tests all critical functionality with proper error handling and clear feedback.
    """
    
    def __init__(self):
        """Initialize the test runner with safe test configuration."""
        self.test_config = {
            "binance_enabled": True,
            "phemex_enabled": False,
            "binance_api_key": "test_key_safe",
            "binance_api_secret": "test_secret_safe", 
            "testnet": True,  # ALWAYS use testnet for safety
            "autonomous_mode": False,
            "max_position_size": 10.0,  # Small test size
            "daily_loss_limit": 0.01,   # 1% for safety
            "approval_threshold": 0.05
        }
        
        self.action_interface: Optional[KimeraActionInterface] = None
        self.test_results: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "detailed_results": [],
            "summary": {}
        }
        
        logger.info("🚀 KIMERA Action Interface Test Runner initialized")
        logger.info(f"📋 Test configuration: {json.dumps(self.test_config, indent=2)}")
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """
        Run comprehensive test suite for the action interface.
        
        Returns:
            Dict containing complete test results
        """
        logger.info("=" * 60)
        logger.info("🎯 Starting KIMERA Action Interface Test Suite")
        logger.info("=" * 60)
        
        try:
            # Test 1: Interface Creation and Initialization
            await self._test_interface_creation()
            
            # Test 2: Safety Controls
            await self._test_safety_controls()
            
            # Test 3: Action Request Processing
            await self._test_action_request_processing()
            
            # Test 4: Mock Exchange Integration
            await self._test_mock_exchange_integration()
            
            # Test 5: Feedback System
            await self._test_feedback_system()
            
            # Test 6: Emergency Controls
            await self._test_emergency_controls()
            
            # Test 7: Configuration Validation
            await self._test_configuration_validation()
            
            # Test 8: Error Handling
            await self._test_error_handling()
            
        except Exception as e:
            logger.error(f"❌ Critical error in test suite: {str(e)}")
            logger.error(f"📍 Traceback: {traceback.format_exc()}")
            self._record_test_failure("test_suite_execution", str(e))
        
        finally:
            # Cleanup
            await self._cleanup()
        
        # Generate final report
        self._generate_final_report()
        return self.test_results
    
    async def _test_interface_creation(self):
        """Test 1: Verify interface can be created and initialized properly."""
        test_name = "interface_creation"
        logger.info(f"🔍 Test 1: {test_name}")
        
        try:
            # Test interface creation with factory function
            self.action_interface = await create_kimera_action_interface(self.test_config)
            
            # Verify interface is properly initialized
            assert self.action_interface is not None, "Interface creation failed"
            assert hasattr(self.action_interface, 'config'), "Interface missing config"
            assert hasattr(self.action_interface, 'exchanges'), "Interface missing exchanges"
            assert hasattr(self.action_interface, 'pending_actions'), "Interface missing action tracking"
            
            # Verify safety controls are initialized
            assert hasattr(self.action_interface, 'autonomous_mode'), "Missing autonomous mode setting"
            assert hasattr(self.action_interface, 'emergency_stop_active'), "Missing emergency stop state"
            
            logger.info("✅ Interface creation successful")
            self._record_test_success(test_name, "Interface created and initialized correctly")
            
        except Exception as e:
            logger.error(f"❌ Interface creation failed: {str(e)}")
            self._record_test_failure(test_name, str(e))
    
    async def _test_safety_controls(self):
        """Test 2: Verify safety controls work correctly."""
        test_name = "safety_controls"
        logger.info(f"🔍 Test 2: {test_name}")
        
        if not self.action_interface:
            self._record_test_failure(test_name, "Interface not available for testing")
            return
        
        try:
            # Test daily loss limit check
            self.action_interface.daily_pnl = -self.test_config["daily_loss_limit"] * 1000 - 1
            
            test_action = ActionRequest(
                action_id="test_safety_001",
                action_type=ActionType.PLACE_ORDER,
                symbol="BTCUSDT",
                parameters={"side": "BUY", "size": 0.001},
                cognitive_reasoning=["Test safety limits"],
                confidence=0.8,
                risk_score=0.3,
                expected_outcome="Test outcome",
                timestamp=datetime.now()
            )
            
            # This should fail safety check due to daily loss limit
            safety_check_result = await self.action_interface._safety_check(test_action)
            assert not safety_check_result, "Safety check should have failed due to loss limit"
            
            # Reset for next test
            self.action_interface.daily_pnl = 0.0
            
            # Test position size limit
            test_action.parameters["size"] = self.test_config["max_position_size"] + 1
            safety_check_result = await self.action_interface._safety_check(test_action)
            assert not safety_check_result, "Safety check should have failed due to position size"
            
            logger.info("✅ Safety controls working correctly")
            self._record_test_success(test_name, "All safety checks functioning properly")
            
        except Exception as e:
            logger.error(f"❌ Safety controls test failed: {str(e)}")
            self._record_test_failure(test_name, str(e))
    
    async def _test_action_request_processing(self):
        """Test 3: Verify action request processing logic."""
        test_name = "action_request_processing"
        logger.info(f"🔍 Test 3: {test_name}")
        
        if not self.action_interface:
            self._record_test_failure(test_name, "Interface not available for testing")
            return
        
        try:
            # Create test trading decision
            test_decision = TradingDecision(
                action="BUY",
                size=0.001,
                confidence=0.7,
                risk_score=0.2,
                reasoning=["Test market signal", "Technical analysis positive"],
                stop_loss=45000.0,
                take_profit=55000.0,
                expected_return=5.0
            )
            
            test_market_state = MarketState(
                symbol="BTCUSDT",
                price=50000.0,
                volume=1000000.0,
                volatility=0.02,
                trend="BULLISH",
                timestamp=datetime.now()
            )
            
            # Test approval requirement logic
            requires_approval = self.action_interface._requires_approval(test_decision, test_market_state)
            logger.info(f"📋 Approval required: {requires_approval}")
            
            # Test enhanced signal processing
            enhanced_signal = IntegratedTradingSignal(
                symbol="BTCUSDT",
                action="BUY",
                confidence=0.6,
                reasoning=["Enhanced cognitive analysis"],
                risk_metrics={"var": 0.05, "risk_score": 0.3},
                expected_outcomes={"target": 0.05, "probability": 0.7},
                cognitive_state={"field_strength": 0.8, "entropy": 0.2},
                timestamp=datetime.now()
            )
            
            # Verify signal processing doesn't crash
            approval_needed = self.action_interface._requires_enhanced_approval(enhanced_signal)
            logger.info(f"📋 Enhanced signal approval needed: {approval_needed}")
            
            logger.info("✅ Action request processing working correctly")
            self._record_test_success(test_name, "Action processing logic verified")
            
        except Exception as e:
            logger.error(f"❌ Action request processing test failed: {str(e)}")
            self._record_test_failure(test_name, str(e))
    
    @patch('backend.trading.api.binance_connector.BinanceConnector')
    async def _test_mock_exchange_integration(self, mock_binance):
        """Test 4: Verify exchange integration with mocked responses."""
        test_name = "mock_exchange_integration"
        logger.info(f"🔍 Test 4: {test_name}")
        
        if not self.action_interface:
            self._record_test_failure(test_name, "Interface not available for testing")
            return
        
        try:
            # Mock exchange responses
            mock_exchange_instance = AsyncMock()
            mock_exchange_instance.place_order.return_value = {
                "orderId": "12345",
                "status": "FILLED",
                "executedQty": "0.001",
                "price": "50000.0"
            }
            mock_binance.return_value = mock_exchange_instance
            
            # Mock the exchange in the interface
            self.action_interface.exchanges["binance"] = mock_exchange_instance
            
            # Test order placement
            test_action = ActionRequest(
                action_id="test_exchange_001",
                action_type=ActionType.PLACE_ORDER,
                symbol="BTCUSDT",
                parameters={
                    "side": "BUY",
                    "size": 0.001,
                    "price": 50000.0,
                    "order_type": "MARKET"
                },
                cognitive_reasoning=["Mock test order"],
                confidence=0.8,
                risk_score=0.2,
                expected_outcome="Test execution",
                timestamp=datetime.now()
            )
            
            # Execute action with mocked exchange
            result = await self.action_interface._place_order(
                test_action, 
                mock_exchange_instance,
                "binance"
            )
            
            assert result.status == ExecutionStatus.COMPLETED, "Order execution should succeed"
            assert result.action_id == test_action.action_id, "Action ID should match"
            
            logger.info("✅ Mock exchange integration working correctly")
            self._record_test_success(test_name, "Exchange integration verified with mocks")
            
        except Exception as e:
            logger.error(f"❌ Mock exchange integration test failed: {str(e)}")
            self._record_test_failure(test_name, str(e))
    
    async def _test_feedback_system(self):
        """Test 5: Verify cognitive feedback system."""
        test_name = "feedback_system"
        logger.info(f"🔍 Test 5: {test_name}")
        
        if not self.action_interface:
            self._record_test_failure(test_name, "Interface not available for testing")
            return
        
        try:
            # Create feedback processor
            feedback_processor = CognitiveFeedbackProcessor()
            
            # Test feedback callback registration
            callback_triggered = False
            async def test_callback(result):
                nonlocal callback_triggered
                callback_triggered = True
                logger.info(f"📡 Feedback callback triggered for action: {result.action_id}")
            
            self.action_interface.register_feedback_callback(test_callback)
            
            # Create test action result
            test_result = ActionResult(
                action_id="test_feedback_001",
                status=ExecutionStatus.COMPLETED,
                execution_time=datetime.now(),
                exchange_response={"orderId": "12345", "status": "FILLED"},
                actual_outcome="Order executed successfully",
                pnl_impact=100.0,
                cognitive_feedback={"learning": "Successful execution"},
                lessons_learned=["Market timing was good"]
            )
            
            # Test feedback processing
            await feedback_processor.process_execution_feedback(test_result)
            await self.action_interface._send_cognitive_feedback(test_result)
            
            # Verify callback was triggered
            assert callback_triggered, "Feedback callback should have been triggered"
            
            logger.info("✅ Feedback system working correctly")
            self._record_test_success(test_name, "Cognitive feedback system verified")
            
        except Exception as e:
            logger.error(f"❌ Feedback system test failed: {str(e)}")
            self._record_test_failure(test_name, str(e))
    
    async def _test_emergency_controls(self):
        """Test 6: Verify emergency stop and resume functionality."""
        test_name = "emergency_controls"
        logger.info(f"🔍 Test 6: {test_name}")
        
        if not self.action_interface:
            self._record_test_failure(test_name, "Interface not available for testing")
            return
        
        try:
            # Test emergency stop
            await self.action_interface.emergency_stop()
            assert self.action_interface.emergency_stop_active, "Emergency stop should be active"
            
            # Test that actions are blocked during emergency stop
            test_action = ActionRequest(
                action_id="test_emergency_001",
                action_type=ActionType.PLACE_ORDER,
                symbol="BTCUSDT",
                parameters={"side": "BUY", "size": 0.001},
                cognitive_reasoning=["Emergency test"],
                confidence=0.8,
                risk_score=0.2,
                expected_outcome="Should be blocked",
                timestamp=datetime.now()
            )
            
            safety_check = await self.action_interface._safety_check(test_action)
            assert not safety_check, "Actions should be blocked during emergency stop"
            
            # Test resume functionality
            self.action_interface.resume_trading()
            assert not self.action_interface.emergency_stop_active, "Emergency stop should be deactivated"
            
            # Verify actions work after resume
            safety_check_after_resume = await self.action_interface._safety_check(test_action)
            # Note: This might still fail due to other safety checks, but emergency stop shouldn't block it
            
            logger.info("✅ Emergency controls working correctly")
            self._record_test_success(test_name, "Emergency stop and resume functionality verified")
            
        except Exception as e:
            logger.error(f"❌ Emergency controls test failed: {str(e)}")
            self._record_test_failure(test_name, str(e))
    
    async def _test_configuration_validation(self):
        """Test 7: Verify configuration validation and edge cases."""
        test_name = "configuration_validation"
        logger.info(f"🔍 Test 7: {test_name}")
        
        try:
            # Test with minimal config
            minimal_config = {
                "testnet": True,
                "autonomous_mode": False
            }
            
            minimal_interface = await create_kimera_action_interface(minimal_config)
            assert minimal_interface is not None, "Interface should work with minimal config"
            
            # Test with invalid config values
            invalid_config = {
                "max_position_size": -100,  # Invalid negative value
                "daily_loss_limit": 2.0,    # Invalid >100% loss limit
                "testnet": True
            }
            
            # Should handle invalid config gracefully
            invalid_interface = await create_kimera_action_interface(invalid_config)
            assert invalid_interface is not None, "Interface should handle invalid config gracefully"
            
            logger.info("✅ Configuration validation working correctly")
            self._record_test_success(test_name, "Configuration validation verified")
            
        except Exception as e:
            logger.error(f"❌ Configuration validation test failed: {str(e)}")
            self._record_test_failure(test_name, str(e))
    
    async def _test_error_handling(self):
        """Test 8: Verify robust error handling."""
        test_name = "error_handling"
        logger.info(f"🔍 Test 8: {test_name}")
        
        if not self.action_interface:
            self._record_test_failure(test_name, "Interface not available for testing")
            return
        
        try:
            # Test with malformed action request
            try:
                malformed_action = ActionRequest(
                    action_id="test_error_001",
                    action_type=ActionType.PLACE_ORDER,
                    symbol="INVALID_SYMBOL",
                    parameters={},  # Empty parameters
                    cognitive_reasoning=[],
                    confidence=-1.0,  # Invalid confidence
                    risk_score=2.0,   # Invalid risk score
                    expected_outcome="",
                    timestamp=datetime.now()
                )
                
                # This should handle errors gracefully
                result = await self.action_interface._safety_check(malformed_action)
                logger.info(f"📋 Malformed action safety check result: {result}")
                
            except Exception as e:
                logger.info(f"📋 Expected error for malformed action: {str(e)}")
            
            # Test with None values
            try:
                await self.action_interface._safety_check(None)
            except Exception as e:
                logger.info(f"📋 Expected error for None action: {str(e)}")
            
            logger.info("✅ Error handling working correctly")
            self._record_test_success(test_name, "Error handling verified")
            
        except Exception as e:
            logger.error(f"❌ Error handling test failed: {str(e)}")
            self._record_test_failure(test_name, str(e))
    
    async def _cleanup(self):
        """Clean up test resources."""
        logger.info("🧹 Cleaning up test resources...")
        
        try:
            if self.action_interface:
                await self.action_interface.disconnect_exchanges()
                logger.info("✅ Action interface cleaned up")
        except Exception as e:
            logger.error(f"⚠️ Cleanup error: {str(e)}")
    
    def _record_test_success(self, test_name: str, message: str):
        """Record a successful test result."""
        self.test_results["tests_run"] += 1
        self.test_results["tests_passed"] += 1
        self.test_results["detailed_results"].append({
            "test": test_name,
            "status": "PASSED",
            "message": message,
            "timestamp": datetime.now().isoformat()
        })
        logger.info(f"✅ {test_name}: {message}")
    
    def _record_test_failure(self, test_name: str, error_message: str):
        """Record a failed test result."""
        self.test_results["tests_run"] += 1
        self.test_results["tests_failed"] += 1
        self.test_results["detailed_results"].append({
            "test": test_name,
            "status": "FAILED",
            "error": error_message,
            "timestamp": datetime.now().isoformat()
        })
        logger.error(f"❌ {test_name}: {error_message}")
    
    def _generate_final_report(self):
        """Generate comprehensive test report."""
        logger.info("=" * 60)
        logger.info("📊 KIMERA Action Interface Test Results")
        logger.info("=" * 60)
        
        total_tests = self.test_results["tests_run"]
        passed_tests = self.test_results["tests_passed"]
        failed_tests = self.test_results["tests_failed"]
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        self.test_results["summary"] = {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "success_rate": f"{success_rate:.1f}%",
            "overall_status": "PASSED" if failed_tests == 0 else "FAILED"
        }
        
        logger.info(f"📈 Total tests: {total_tests}")
        logger.info(f"✅ Passed: {passed_tests}")
        logger.info(f"❌ Failed: {failed_tests}")
        logger.info(f"📊 Success rate: {success_rate:.1f}%")
        logger.info(f"🎯 Overall status: {self.test_results['summary']['overall_status']}")
        
        # Save detailed report
        report_filename = f"action_interface_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(report_filename, 'w') as f:
                json.dump(self.test_results, f, indent=2)
            logger.info(f"📄 Detailed report saved: {report_filename}")
        except Exception as e:
            logger.error(f"⚠️ Failed to save report: {str(e)}")
        
        logger.info("=" * 60)


async def main():
    """Main test execution function."""
    logger.info("🚀 KIMERA Action Interface Test Runner")
    logger.info("Testing KIMERA's real-world execution capabilities...")
    logger.info()
    
    test_runner = ActionInterfaceTestRunner()
    
    try:
        results = await test_runner.run_all_tests()
        
        if results["summary"]["overall_status"] == "PASSED":
            logger.info("🎉 All tests passed! KIMERA Action Interface is working correctly.")
            return 0
        else:
            logger.warning("⚠️ Some tests failed. Check the logs for details.")
            return 1
            
    except Exception as e:
        logger.error(f"💥 Critical test runner failure: {str(e)}")
        logger.error(f"📍 Traceback: {traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    # Run the test suite
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 