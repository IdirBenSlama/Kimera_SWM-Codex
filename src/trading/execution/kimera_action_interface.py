"""
KIMERA Action Execution Interface

This module provides KIMERA with the "arms" to execute real-world trading actions.
It bridges the gap between cognitive analysis and actual market execution.

Key Features:
- Safe execution with built-in risk controls
- Real-time feedback to KIMERA's cognitive systems
- Autonomous decision execution with human oversight options
- Action logging and performance tracking
"""

import os
import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import json

from src.trading.core.trading_engine import TradingDecision, MarketState
from src.trading.core.integrated_trading_engine import IntegratedTradingSignal
from src.trading.api.binance_connector import BinanceConnector
from src.trading.api.phemex_connector import PhemexConnector

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Types of actions KIMERA can execute"""
    PLACE_ORDER = "place_order"
    CANCEL_ORDER = "cancel_order"
    MODIFY_POSITION = "modify_position"
    SET_STOP_LOSS = "set_stop_loss"
    SET_TAKE_PROFIT = "set_take_profit"
    CLOSE_POSITION = "close_position"
    REBALANCE_PORTFOLIO = "rebalance_portfolio"
    EMERGENCY_STOP = "emergency_stop"


class ExecutionStatus(Enum):
    """Status of action execution"""
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REQUIRES_APPROVAL = "requires_approval"


@dataclass
class ActionRequest:
    """Request for KIMERA to execute an action"""
    action_id: str
    action_type: ActionType
    symbol: str
    parameters: Dict[str, Any]
    cognitive_reasoning: List[str]
    confidence: float
    risk_score: float
    expected_outcome: str
    timestamp: datetime
    requires_approval: bool = False


@dataclass
class ActionResult:
    """Result of executed action"""
    action_id: str
    status: ExecutionStatus
    execution_time: datetime
    exchange_response: Optional[Dict[str, Any]]
    actual_outcome: str
    pnl_impact: Optional[float]
    cognitive_feedback: Dict[str, Any]
    lessons_learned: List[str]


class KimeraActionInterface:
    """
    Interface that gives KIMERA the ability to execute real-world trading actions.
    
    This is KIMERA's "arms" - it translates cognitive decisions into market actions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize KIMERA's action interface.
        
        Args:
            config: Configuration including exchange settings, safety limits
        """
        self.config = config
        
        # Initialize exchange connectors
        self.exchanges = {}
        if config.get("binance_enabled", False):
            self.exchanges["binance"] = BinanceConnector(
                api_key=config.get("binance_api_key", ""),
                api_secret=config.get("binance_api_secret", ""),
                testnet=config.get("testnet", os.getenv('KIMERA_USE_TESTNET', 'false').lower() == 'true')
            )
        
        if config.get("phemex_enabled", False):
            self.exchanges["phemex"] = PhemexConnector(
                api_key=config.get("phemex_api_key", ""),
                api_secret=config.get("phemex_api_secret", ""),
                testnet=config.get("testnet", os.getenv('KIMERA_USE_TESTNET', 'false').lower() == 'true')
            )
        
        # Safety controls
        self.autonomous_mode = config.get("autonomous_mode", False)
        self.max_position_size = config.get("max_position_size", 1000.0)
        self.daily_loss_limit = config.get("daily_loss_limit", 0.05)  # 5%
        self.approval_threshold = config.get("approval_threshold", 0.1)  # Above 10% of balance
        
        # Action tracking
        self.pending_actions: Dict[str, ActionRequest] = {}
        self.completed_actions: List[ActionResult] = []
        self.approval_queue: List[ActionRequest] = []
        
        # Cognitive feedback system
        self.feedback_callbacks: List[Callable] = []
        
        # Safety state
        self.emergency_stop_active = False
        self.daily_pnl = 0.0
        self.action_count_today = 0
        
        logger.info("🎯 KIMERA Action Interface initialized - Ready to execute!")
    
    async def connect_exchanges(self):
        """Connect to all configured exchanges"""
        for name, exchange in self.exchanges.items():
            try:
                await exchange.__aenter__()
                logger.info(f"✅ Connected to {name}")
            except Exception as e:
                logger.error(f"❌ Failed to connect to {name}: {str(e)}")
    
    async def disconnect_exchanges(self):
        """Disconnect from all exchanges"""
        for name, exchange in self.exchanges.items():
            try:
                await exchange.close()
                logger.info(f"Disconnected from {name}")
            except Exception as e:
                logger.error(f"Error disconnecting from {name}: {str(e)}")
    
    async def execute_trading_decision(
        self,
        decision: TradingDecision,
        market_state: MarketState,
        exchange: str = "binance"
    ) -> ActionResult:
        """
        Execute a trading decision made by KIMERA's cognitive system.
        
        This is where KIMERA's thoughts become real-world actions!
        
        Args:
            decision: Trading decision from KIMERA
            market_state: Current market state
            exchange: Exchange to execute on
            
        Returns:
            Result of the execution
        """
        # Create action request
        action_request = ActionRequest(
            action_id=f"kimera_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            action_type=ActionType.PLACE_ORDER,
            symbol=market_state.symbol,
            parameters={
                "side": decision.action,
                "size": decision.size,
                "price": market_state.price,
                "order_type": "MARKET" if decision.confidence > 0.8 else "LIMIT",
                "stop_loss": decision.stop_loss,
                "take_profit": decision.take_profit
            },
            cognitive_reasoning=decision.reasoning,
            confidence=decision.confidence,
            risk_score=decision.risk_score,
            expected_outcome=f"Expected return: {decision.expected_return:.2f}%",
            timestamp=datetime.now(),
            requires_approval=self._requires_approval(decision, market_state)
        )
        
        return await self.execute_action(action_request, exchange)
    
    async def execute_enhanced_signal(
        self,
        signal: IntegratedTradingSignal,
        symbol: str,
        exchange: str = "binance"
    ) -> ActionResult:
        """
        Execute an enhanced trading signal from KIMERA.
        
        Args:
            signal: Enhanced signal with cognitive analysis
            symbol: Trading symbol
            exchange: Exchange to execute on
            
        Returns:
            Result of the execution
        """
        action_request = ActionRequest(
            action_id=f"kimera_enhanced_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            action_type=ActionType.PLACE_ORDER,
            symbol=symbol,
            parameters={
                "side": signal.action,
                "size": signal.position_size,
                "order_type": "MARKET",
                "cognitive_pressure": signal.cognitive_pressure,
                "contradiction_level": signal.contradiction_level,
                "semantic_temperature": signal.semantic_temperature
            },
            cognitive_reasoning=[signal.explanation],
            confidence=signal.confidence,
            risk_score=signal.anomaly_score,
            expected_outcome=f"Cognitive signal: {signal.action} with {signal.confidence:.2f} confidence",
            timestamp=signal.timestamp,
            requires_approval=self._requires_enhanced_approval(signal)
        )
        
        return await self.execute_action(action_request, exchange)
    
    async def execute_action(
        self,
        action_request: ActionRequest,
        exchange: str = "binance"
    ) -> ActionResult:
        """
        Execute a specific action request.
        
        This is KIMERA's primary execution method - where decisions become reality.
        
        Args:
            action_request: The action to execute
            exchange: Exchange to execute on
            
        Returns:
            Result of the execution
        """
        try:
            # Safety checks first - KIMERA's safety protocols
            if not await self._safety_check(action_request):
                return ActionResult(
                    action_id=action_request.action_id,
                    status=ExecutionStatus.FAILED,
                    execution_time=datetime.now(),
                    exchange_response=None,
                    actual_outcome="Failed safety checks",
                    pnl_impact=None,
                    cognitive_feedback={"safety_violation": True},
                    lessons_learned=["Action blocked by safety systems"]
                )
            
            # Check if approval required
            if action_request.requires_approval and not self.autonomous_mode:
                self.approval_queue.append(action_request)
                logger.info(f"⏳ Action {action_request.action_id} requires approval")
                return ActionResult(
                    action_id=action_request.action_id,
                    status=ExecutionStatus.REQUIRES_APPROVAL,
                    execution_time=datetime.now(),
                    exchange_response=None,
                    actual_outcome="Waiting for human approval",
                    pnl_impact=None,
                    cognitive_feedback={"approval_required": True},
                    lessons_learned=["High-risk action requires approval"]
                )
            
            # Add to pending actions
            self.pending_actions[action_request.action_id] = action_request
            
            # Execute the action - This is where KIMERA acts!
            logger.info(f"🎯 KIMERA executing action: {action_request.action_type.value}")
            logger.info(f"   Symbol: {action_request.symbol}")
            logger.info(f"   Confidence: {action_request.confidence:.2f}")
            logger.info(f"   Reasoning: {'; '.join(action_request.cognitive_reasoning)}")
            
            result = await self._execute_on_exchange(action_request, exchange)
            
            # Remove from pending
            self.pending_actions.pop(action_request.action_id, None)
            
            # Store result
            self.completed_actions.append(result)
            
            # Update performance tracking
            await self._update_performance(result)
            
            # Send cognitive feedback to KIMERA
            await self._send_cognitive_feedback(result)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Action execution failed: {str(e)}")
            
            # Remove from pending
            self.pending_actions.pop(action_request.action_id, None)
            
            return ActionResult(
                action_id=action_request.action_id,
                status=ExecutionStatus.FAILED,
                execution_time=datetime.now(),
                exchange_response=None,
                actual_outcome=f"Execution error: {str(e)}",
                pnl_impact=None,
                cognitive_feedback={"execution_error": str(e)},
                lessons_learned=["Technical execution failure - investigate system"]
            )
    
    async def _execute_on_exchange(
        self,
        action_request: ActionRequest,
        exchange_name: str
    ) -> ActionResult:
        """Execute action on specific exchange"""
        exchange = self.exchanges.get(exchange_name)
        if not exchange:
            raise ValueError(f"Exchange {exchange_name} not available")
        
        if action_request.action_type == ActionType.PLACE_ORDER:
            return await self._place_order(action_request, exchange, exchange_name)
        elif action_request.action_type == ActionType.CANCEL_ORDER:
            return await self._cancel_order(action_request, exchange)
        elif action_request.action_type == ActionType.CLOSE_POSITION:
            return await self._close_position(action_request, exchange)
        else:
            raise ValueError(f"Unsupported action type: {action_request.action_type}")
    
    async def _place_order(
        self,
        action_request: ActionRequest,
        exchange,
        exchange_name: str
    ) -> ActionResult:
        """Place an order on the exchange - KIMERA's actual trading execution"""
        params = action_request.parameters
        
        try:
            if exchange_name == "binance":
                # Calculate quantity for Binance (price-based)
                if params["side"].upper() == "BUY":
                    # For buy orders, convert size (USD value) to quantity
                    price = params.get("price", 0)
                    if price > 0:
                        quantity = params["size"] / price
                    else:
                        raise ValueError("Price required for buy orders")
                else:
                    # For sell orders, use size as quantity directly
                    quantity = params["size"]
                
                order_response = await exchange.place_order(
                    symbol=action_request.symbol,
                    side=params["side"].upper(),
                    order_type=params.get("order_type", "MARKET"),
                    quantity=quantity,
                    price=params.get("price") if params.get("order_type") == "LIMIT" else None
                )
            else:  # phemex
                order_response = await exchange.place_contract_order(
                    symbol=action_request.symbol,
                    side=params["side"],
                    order_type=params.get("order_type", "Market"),
                    contracts=int(params["size"]),
                    price=params.get("price")
                )
            
            # Calculate P&L impact (simplified)
            pnl_impact = self._estimate_pnl_impact(action_request, order_response)
            
            logger.info(f"✅ KIMERA successfully executed order!")
            logger.info(f"   Order ID: {order_response.get('orderId', 'N/A')}")
            logger.info(f"   Filled: {order_response.get('executedQty', 'N/A')}")
            
            return ActionResult(
                action_id=action_request.action_id,
                status=ExecutionStatus.COMPLETED,
                execution_time=datetime.now(),
                exchange_response=order_response,
                actual_outcome=f"Order placed successfully: {order_response.get('orderId')}",
                pnl_impact=pnl_impact,
                cognitive_feedback={
                    "execution_success": True,
                    "order_id": order_response.get('orderId'),
                    "filled_quantity": order_response.get('executedQty'),
                    "cognitive_reasoning_validated": True
                },
                lessons_learned=["Successful order execution - cognitive analysis effective"]
            )
            
        except Exception as e:
            logger.error(f"❌ Order execution failed: {str(e)}")
            raise
    
    async def _cancel_order(self, action_request: ActionRequest, exchange) -> ActionResult:
        """Cancel an order"""
        params = action_request.parameters
        order_id = params.get("order_id")
        
        try:
            cancel_response = await exchange.cancel_order(
                symbol=action_request.symbol,
                order_id=order_id
            )
            
            return ActionResult(
                action_id=action_request.action_id,
                status=ExecutionStatus.COMPLETED,
                execution_time=datetime.now(),
                exchange_response=cancel_response,
                actual_outcome=f"Order {order_id} cancelled successfully",
                pnl_impact=0.0,
                cognitive_feedback={"cancellation_success": True},
                lessons_learned=["Order cancellation executed"]
            )
            
        except Exception as e:
            logger.error(f"❌ Order cancellation failed: {str(e)}")
            raise
    
    async def _close_position(self, action_request: ActionRequest, exchange) -> ActionResult:
        """Close a position"""
        params = action_request.parameters
        
        try:
            # Get current position size first
            # This would need to be implemented based on exchange API
            position_size = params.get("size", 0)
            
            close_response = await exchange.place_order(
                symbol=action_request.symbol,
                side="SELL" if params.get("current_side") == "BUY" else "BUY",
                order_type="MARKET",
                quantity=position_size
            )
            
            return ActionResult(
                action_id=action_request.action_id,
                status=ExecutionStatus.COMPLETED,
                execution_time=datetime.now(),
                exchange_response=close_response,
                actual_outcome=f"Position closed successfully",
                pnl_impact=self._estimate_pnl_impact(action_request, close_response),
                cognitive_feedback={"position_closed": True},
                lessons_learned=["Position closure executed"]
            )
            
        except Exception as e:
            logger.error(f"❌ Position closure failed: {str(e)}")
            raise
    
    async def _safety_check(self, action_request: ActionRequest) -> bool:
        """Comprehensive safety checks before execution"""
        
        # Emergency stop check
        if self.emergency_stop_active:
            logger.warning("🛑 Emergency stop active - blocking all actions")
            return False
        
        # Daily loss limit check
        if abs(self.daily_pnl) > self.daily_loss_limit * 10000:  # Assuming 10k starting balance
            logger.warning(f"🛑 Daily loss limit reached: {self.daily_pnl}")
            return False
        
        # Position size check
        if action_request.parameters.get("size", 0) > self.max_position_size:
            logger.warning(f"🛑 Position size too large: {action_request.parameters['size']}")
            return False
        
        # Risk score check
        if action_request.risk_score > 0.8:
            logger.warning(f"🛑 Risk score too high: {action_request.risk_score}")
            return False
        
        # Daily action limit
        if self.action_count_today > 100:  # Max 100 actions per day
            logger.warning("🛑 Daily action limit reached")
            return False
        
        # Confidence check
        if action_request.confidence < 0.3:
            logger.warning(f"🛑 Confidence too low: {action_request.confidence}")
            return False
        
        return True
    
    def _requires_approval(self, decision: TradingDecision, market_state: MarketState) -> bool:
        """Check if decision requires human approval"""
        # Large position
        if decision.size > self.approval_threshold * 10000:
            return True
        # High risk
        if decision.risk_score > 0.7:
            return True
        # Extreme market conditions
        if market_state.cognitive_pressure > 0.9:
            return True
        # Low confidence
        if decision.confidence < 0.5:
            return True
        return False
    
    def _requires_enhanced_approval(self, signal: IntegratedTradingSignal) -> bool:
        """Check if enhanced signal requires approval"""
        # High anomaly
        if signal.anomaly_score > 0.7:
            return True
        # Large position
        if signal.position_size > self.approval_threshold * 10000:
            return True
        # Low confidence
        if signal.confidence < 0.5:
            return True
        return False
    
    def _estimate_pnl_impact(self, action_request: ActionRequest, order_response: Dict) -> float:
        """Estimate P&L impact of executed action"""
        try:
            size = float(order_response.get('executedQty', action_request.parameters.get('size', 0)))
            price = float(order_response.get('price', action_request.parameters.get('price', 0)))
            
            # Simple P&L estimate (this would be more sophisticated in production)
            if action_request.parameters.get("side", "").upper() == "BUY":
                return -size * price  # Negative because we spent money
            else:
                return size * price   # Positive because we received money
        except (ValueError, TypeError):
            return 0.0
    
    async def _update_performance(self, result: ActionResult):
        """Update performance metrics"""
        if result.pnl_impact:
            self.daily_pnl += result.pnl_impact
        self.action_count_today += 1
        
        # Log performance updates
        if result.status == ExecutionStatus.COMPLETED:
            logger.info(f"📈 Performance update: Daily P&L: ${self.daily_pnl:.2f}")
    
    async def _send_cognitive_feedback(self, result: ActionResult):
        """Send feedback to KIMERA's cognitive systems"""
        feedback_data = {
            "action_id": result.action_id,
            "status": result.status.value,
            "pnl_impact": result.pnl_impact,
            "lessons_learned": result.lessons_learned,
            "timestamp": result.execution_time.isoformat()
        }
        
        # Send to all registered feedback callbacks
        for callback in self.feedback_callbacks:
            try:
                await callback(feedback_data)
            except Exception as e:
                logger.error(f"Feedback callback error: {str(e)}")
        
        # Log cognitive feedback
        logger.info(f"🧠 Cognitive feedback sent: {result.cognitive_feedback}")
    
    def register_feedback_callback(self, callback: Callable):
        """Register a callback for cognitive feedback"""
        self.feedback_callbacks.append(callback)
        logger.info("🔗 New cognitive feedback callback registered")
    
    async def approve_action(self, action_id: str) -> bool:
        """Manually approve a pending action"""
        for i, action in enumerate(self.approval_queue):
            if action.action_id == action_id:
                approved_action = self.approval_queue.pop(i)
                result = await self.execute_action(approved_action)
                logger.info(f"✅ Manually approved and executed action {action_id}")
                return True
        logger.warning(f"⚠️ Action {action_id} not found in approval queue")
        return False
    
    async def emergency_stop(self):
        """Emergency stop all trading activities"""
        self.emergency_stop_active = True
        logger.critical("🛑 EMERGENCY STOP ACTIVATED - ALL TRADING HALTED")
        
        # Cancel all pending actions
        cancelled_count = len(self.pending_actions)
        self.pending_actions.clear()
        
        logger.info(f"🛑 Cancelled {cancelled_count} pending actions")
    
    def resume_trading(self):
        """Resume trading after emergency stop"""
        self.emergency_stop_active = False
        logger.info("✅ Trading resumed - Emergency stop deactivated")
    
    def get_action_summary(self) -> Dict[str, Any]:
        """Get summary of KIMERA's actions"""
        successful_actions = [a for a in self.completed_actions if a.status == ExecutionStatus.COMPLETED]
        failed_actions = [a for a in self.completed_actions if a.status == ExecutionStatus.FAILED]
        
        return {
            "total_actions": len(self.completed_actions),
            "successful_actions": len(successful_actions),
            "failed_actions": len(failed_actions),
            "success_rate": len(successful_actions) / len(self.completed_actions) if self.completed_actions else 0,
            "pending_actions": len(self.pending_actions),
            "approval_queue": len(self.approval_queue),
            "daily_pnl": self.daily_pnl,
            "action_count_today": self.action_count_today,
            "emergency_stop_active": self.emergency_stop_active,
            "autonomous_mode": self.autonomous_mode,
            "system_status": "OPERATIONAL" if not self.emergency_stop_active else "EMERGENCY_STOP"
        }
    
    def get_pending_approvals(self) -> List[Dict[str, Any]]:
        """Get list of actions waiting for approval"""
        return [
            {
                "action_id": action.action_id,
                "symbol": action.symbol,
                "action_type": action.action_type.value,
                "confidence": action.confidence,
                "risk_score": action.risk_score,
                "reasoning": action.cognitive_reasoning,
                "expected_outcome": action.expected_outcome
            }
            for action in self.approval_queue
        ]


async def create_kimera_action_interface(config: Dict[str, Any]) -> KimeraActionInterface:
    """Factory function to create KIMERA's action interface"""
    interface = KimeraActionInterface(config)
    await interface.connect_exchanges()
    return interface


# Cognitive Feedback Integration
class CognitiveFeedbackProcessor:
    """Processes execution results and feeds them back to KIMERA's cognitive systems"""
    
    def __init__(self, cognitive_field=None, contradiction_engine=None):
        self.cognitive_field = cognitive_field
        self.contradiction_engine = contradiction_engine
        self.feedback_history = []
        
    async def process_execution_feedback(self, result: ActionResult):
        """Process execution result and update cognitive systems"""
        logger.info(f"🧠 Processing cognitive feedback for action {result.action_id}")
        
        feedback_data = {
            "action_id": result.action_id,
            "timestamp": result.execution_time,
            "status": result.status,
            "cognitive_impact": self._calculate_cognitive_impact(result)
        }
        
        # Update cognitive field with execution outcome
        if self.cognitive_field and result.status == ExecutionStatus.COMPLETED:
            # Positive reinforcement for successful actions
            logger.info("✅ Positive cognitive reinforcement - action successful")
            feedback_data["reinforcement"] = "positive"
            
        elif result.status == ExecutionStatus.FAILED:
            # Learning from failures
            logger.info("📚 Learning from execution failure")
            feedback_data["reinforcement"] = "negative"
            
        # Store lessons learned for future decisions
        if result.lessons_learned:
            logger.info(f"📝 Lessons learned: {'; '.join(result.lessons_learned)}")
            feedback_data["lessons"] = result.lessons_learned
        
        # Store feedback history
        self.feedback_history.append(feedback_data)
        
        # Keep only recent feedback (last 1000 actions)
        if len(self.feedback_history) > 1000:
            self.feedback_history.pop(0)
    
    def _calculate_cognitive_impact(self, result: ActionResult) -> float:
        """Calculate the cognitive impact of the action result"""
        if result.status == ExecutionStatus.COMPLETED:
            # Successful execution has positive impact
            base_impact = 0.8
            if result.pnl_impact and result.pnl_impact > 0:
                base_impact += 0.2  # Extra positive for profitable trades
            return base_impact
        elif result.status == ExecutionStatus.FAILED:
            # Failed execution has negative impact
            return -0.5
        else:
            # Neutral for other statuses
            return 0.0
    
    def get_cognitive_performance_summary(self) -> Dict[str, Any]:
        """Get summary of cognitive performance from feedback"""
        if not self.feedback_history:
            return {"message": "No feedback history available"}
        
        successful_actions = [f for f in self.feedback_history if f["status"] == ExecutionStatus.COMPLETED]
        failed_actions = [f for f in self.feedback_history if f["status"] == ExecutionStatus.FAILED]
        
        avg_cognitive_impact = sum(f["cognitive_impact"] for f in self.feedback_history) / len(self.feedback_history)
        
        return {
            "total_feedback_entries": len(self.feedback_history),
            "successful_actions": len(successful_actions),
            "failed_actions": len(failed_actions),
            "success_rate": len(successful_actions) / len(self.feedback_history),
            "average_cognitive_impact": avg_cognitive_impact,
            "cognitive_learning_trend": "improving" if avg_cognitive_impact > 0 else "needs_attention"
        }


if __name__ == "__main__":
    # Test the action interface
    logger.info("🚀 Testing KIMERA Action Interface - The Bridge Between Mind and Market")
    
    config = {
        "binance_enabled": True,
        "binance_api_key": "test_key",
        "binance_api_secret": "test_secret",
        "testnet": os.getenv('KIMERA_USE_TESTNET', 'false').lower() == 'true',  # Default to real trading
        "autonomous_mode": False,
        "max_position_size": 100.0,
        "daily_loss_limit": 0.05
    }
    
    async def test_interface():
        logger.info("Creating KIMERA's action interface...")
        interface = await create_kimera_action_interface(config)
        
        # Test creating an action request
        from src.trading.core.trading_engine import TradingDecision, MarketState
        
        test_decision = TradingDecision(
            action="BUY",
            confidence=0.75,
            size=50.0,
            reasoning=["KIMERA detected market opportunity", "High cognitive pressure indicates potential"],
            risk_score=0.3,
            cognitive_alignment=0.8,
            expected_return=2.5
        )
        
        test_market_state = MarketState(
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            price=50000.0,
            volume=1000.0,
            bid_ask_spread=0.001,
            volatility=0.02,
            cognitive_pressure=0.7,
            contradiction_level=0.4,
            semantic_temperature=0.6,
            insight_signals=["Market inefficiency detected"]
        )
        
        logger.info("📊 Testing KIMERA's decision execution...")
        result = await interface.execute_trading_decision(test_decision, test_market_state)
        
        logger.info(f"✅ Execution Result: {result.status.value}")
        logger.info(f"📝 Outcome: {result.actual_outcome}")
        logger.info(f"🧠 Cognitive Feedback: {result.cognitive_feedback}")
        
        summary = interface.get_action_summary()
        logger.info(f"\n📈 KIMERA Action Summary:")
        logger.info(f"   Total Actions: {summary['total_actions']}")
        logger.info(f"   Success Rate: {summary['success_rate']:.2%}")
        logger.info(f"   System Status: {summary['system_status']}")
        logger.info(f"   Autonomous Mode: {summary['autonomous_mode']}")
        
        await interface.disconnect_exchanges()
        logger.info("\n🎯 KIMERA now has the 'arms' to execute in the real world!")
    
    asyncio.run(test_interface()) 