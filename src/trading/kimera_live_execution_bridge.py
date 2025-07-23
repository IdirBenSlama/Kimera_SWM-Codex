"""
KIMERA LIVE EXECUTION BRIDGE
============================

This module bridges Kimera's cognitive analysis with REAL trade execution.
It replaces the simulation-only approach with actual market orders.

⚠️ WARNING: THIS MODULE PLACES REAL TRADES WITH REAL MONEY ⚠️

CRITICAL SAFETY FEATURES:
- Pre-execution risk checks
- Position size validation
- Daily loss limits
- Emergency stop mechanisms
- Real-time monitoring
- Execution confirmation
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

# Kimera imports
from src.trading.execution.semantic_execution_bridge import (
    SemanticExecutionBridge, ExecutionRequest, ExecutionResult, OrderType, OrderStatus
)
from src.trading.execution.kimera_action_interface import (
    KimeraActionInterface, ActionRequest, ActionResult, ActionType, ExecutionStatus
)
from src.trading.live_trading_config import LiveTradingConfig, KimeraLiveTradingManager
from src.trading.autonomous_kimera_trader import CognitiveSignal, AutonomousPosition
from src.trading.api.binance_connector import BinanceConnector

logger = logging.getLogger(__name__)

@dataclass
class LiveExecutionResult:
    """Result of live execution with enhanced tracking"""
    signal: CognitiveSignal
    execution_result: ExecutionResult
    position_created: bool
    risk_checks_passed: bool
    actual_position_size: float
    actual_entry_price: float
    fees_paid: float
    execution_time: float
    exchange_order_id: str
    warnings: List[str]
    metadata: Dict[str, Any]


class KimeraLiveExecutionBridge:
    """
    Bridges Kimera's cognitive signals to real trade execution.
    
    This is the critical component that translates Kimera's analysis
    into actual market orders with real money.
    """
    
    def __init__(self, live_config: LiveTradingConfig):
        """
        Initialize live execution bridge
        
        Args:
            live_config: Live trading configuration with real credentials
        """
        self.live_config = live_config
        self.trading_manager = KimeraLiveTradingManager(live_config)
        
        # Initialize execution components
        self.execution_bridge = None
        self.action_interface = None
        self.binance_connector = None
        
        # Trading state
        self.active_positions: Dict[str, AutonomousPosition] = {}
        self.execution_history: List[LiveExecutionResult] = []
        self.daily_stats = {
            'trades_executed': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'total_fees': 0.0,
            'gross_pnl': 0.0,
            'net_pnl': 0.0,
            'start_time': datetime.now()
        }
        
        # Risk monitoring
        self.risk_violations = []
        self.last_risk_check = datetime.now()
        
        logger.info("🔴 KIMERA LIVE EXECUTION BRIDGE INITIALIZED")
        logger.info("⚠️  REAL MONEY TRADING ENABLED")
        logger.info(f"   Max Position Size: ${live_config.max_position_size_usd}")
        logger.info(f"   Max Daily Loss: ${live_config.max_daily_loss_usd}")
    
    async def initialize_connections(self):
        """Initialize all trading connections"""
        try:
            # Get execution configuration
            exec_config = self.trading_manager.get_execution_config()
            
            # Initialize execution bridge with LIVE configuration
            self.execution_bridge = SemanticExecutionBridge(exec_config)
            
            # Initialize action interface
            self.action_interface = KimeraActionInterface(exec_config)
            
            # Initialize direct Binance connection
            self.binance_connector = BinanceConnector(
                api_key=self.live_config.binance_api_key,
                private_key_path=self.live_config.binance_private_key_path,
                testnet=False  # LIVE TRADING
            )
            
            # Connect to exchanges
            await self.action_interface.connect_exchanges()
            
            logger.info("✅ Live execution connections initialized")
            logger.info("🔴 READY FOR LIVE TRADING")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize connections: {e}")
            raise
    
    async def execute_cognitive_signal(self, signal: CognitiveSignal) -> LiveExecutionResult:
        """
        Execute a cognitive signal with real money
        
        Args:
            signal: Cognitive signal from Kimera's analysis
            
        Returns:
            LiveExecutionResult with execution details
        """
        start_time = time.time()
        warnings = []
        
        try:
            logger.info(f"🧠 EXECUTING COGNITIVE SIGNAL: {signal.symbol}")
            logger.info(f"   Action: {signal.action}")
            logger.info(f"   Confidence: {signal.confidence:.2f}")
            logger.info(f"   Conviction: {signal.conviction:.2f}")
            
            # 1. Pre-execution risk checks
            risk_checks_passed = await self._perform_risk_checks(signal)
            if not risk_checks_passed:
                return LiveExecutionResult(
                    signal=signal,
                    execution_result=None,
                    position_created=False,
                    risk_checks_passed=False,
                    actual_position_size=0.0,
                    actual_entry_price=0.0,
                    fees_paid=0.0,
                    execution_time=time.time() - start_time,
                    exchange_order_id="",
                    warnings=["Risk checks failed"],
                    metadata={'reason': 'risk_checks_failed'}
                )
            
            # 2. Calculate actual position size
            actual_position_size = await self._calculate_live_position_size(signal)
            
            # 3. Create execution request
            execution_request = ExecutionRequest(
                order_id=f"kimera_live_{signal.symbol}_{int(time.time())}",
                symbol=signal.symbol,
                side=signal.action.lower(),
                quantity=actual_position_size,
                order_type=OrderType.MARKET if signal.confidence > 0.8 else OrderType.LIMIT,
                price=signal.entry_price if signal.confidence <= 0.8 else None,
                metadata={
                    'signal_confidence': signal.confidence,
                    'signal_conviction': signal.conviction,
                    'cognitive_reasoning': signal.reasoning,
                    'strategy': signal.strategy.value,
                    'live_trading': True
                }
            )
            
            # 4. Execute the order
            logger.warning(f"🔥 PLACING REAL ORDER: {signal.symbol} {signal.action}")
            execution_result = await self.execution_bridge.execute_order(execution_request)
            
            # 5. Process execution result
            if execution_result.status == OrderStatus.FILLED:
                # Create live position
                position = await self._create_live_position(signal, execution_result)
                self.active_positions[signal.symbol] = position
                
                # Update trading statistics
                self.daily_stats['trades_executed'] += 1
                self.daily_stats['successful_trades'] += 1
                self.daily_stats['total_fees'] += execution_result.fees
                
                # Update trading manager
                self.trading_manager.active_positions[signal.symbol] = position
                
                logger.info(f"✅ LIVE ORDER EXECUTED:")
                logger.info(f"   Order ID: {execution_result.exchange_order_id}")
                logger.info(f"   Size: {execution_result.filled_quantity}")
                logger.info(f"   Price: ${execution_result.average_price:.2f}")
                logger.info(f"   Fees: ${execution_result.fees:.2f}")
                
                return LiveExecutionResult(
                    signal=signal,
                    execution_result=execution_result,
                    position_created=True,
                    risk_checks_passed=True,
                    actual_position_size=execution_result.filled_quantity,
                    actual_entry_price=execution_result.average_price,
                    fees_paid=execution_result.fees,
                    execution_time=time.time() - start_time,
                    exchange_order_id=execution_result.metadata.get('exchange_order_id', ''),
                    warnings=warnings,
                    metadata={
                        'execution_status': 'success',
                        'order_type': execution_request.order_type.value,
                        'exchange': execution_result.exchange
                    }
                )
            
            else:
                # Execution failed
                self.daily_stats['failed_trades'] += 1
                
                logger.error(f"❌ LIVE ORDER FAILED:")
                logger.error(f"   Status: {execution_result.status}")
                logger.error(f"   Reason: {execution_result.metadata}")
                
                return LiveExecutionResult(
                    signal=signal,
                    execution_result=execution_result,
                    position_created=False,
                    risk_checks_passed=True,
                    actual_position_size=0.0,
                    actual_entry_price=0.0,
                    fees_paid=0.0,
                    execution_time=time.time() - start_time,
                    exchange_order_id="",
                    warnings=[f"Execution failed: {execution_result.status}"],
                    metadata={'reason': 'execution_failed', 'status': execution_result.status}
                )
        
        except Exception as e:
            logger.error(f"❌ CRITICAL EXECUTION ERROR: {e}")
            self.daily_stats['failed_trades'] += 1
            
            return LiveExecutionResult(
                signal=signal,
                execution_result=None,
                position_created=False,
                risk_checks_passed=False,
                actual_position_size=0.0,
                actual_entry_price=0.0,
                fees_paid=0.0,
                execution_time=time.time() - start_time,
                exchange_order_id="",
                warnings=[f"Critical error: {str(e)}"],
                metadata={'error': str(e)}
            )
    
    async def _perform_risk_checks(self, signal: CognitiveSignal) -> bool:
        """Perform comprehensive risk checks before execution"""
        try:
            # Check if trading is enabled
            if not self.trading_manager.trading_enabled:
                logger.warning("❌ Trading is disabled")
                return False
            
            # Check emergency stop
            if self.trading_manager.emergency_stop_active:
                logger.warning("❌ Emergency stop is active")
                return False
            
            # Check signal confidence
            if signal.confidence < self.live_config.min_signal_confidence:
                logger.warning(f"❌ Signal confidence {signal.confidence:.2f} below threshold {self.live_config.min_signal_confidence:.2f}")
                return False
            
            # Check position size
            position_size = signal.suggested_allocation_pct * self.live_config.initial_balance
            if not self.trading_manager.check_risk_limits(position_size):
                logger.warning(f"❌ Position size ${position_size:.2f} violates risk limits")
                return False
            
            # Check concurrent positions
            if len(self.active_positions) >= self.live_config.max_concurrent_positions:
                logger.warning(f"❌ Max concurrent positions reached: {len(self.active_positions)}")
                return False
            
            # Check daily loss limit
            if self.trading_manager.daily_pnl < -self.live_config.max_daily_loss_usd:
                logger.warning(f"❌ Daily loss limit exceeded: ${self.trading_manager.daily_pnl:.2f}")
                return False
            
            logger.info("✅ Risk checks passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Risk check failed: {e}")
            return False
    
    async def _calculate_live_position_size(self, signal: CognitiveSignal) -> float:
        """Calculate actual position size for live trading"""
        try:
            # Base position size from signal
            base_size = signal.suggested_allocation_pct * self.live_config.initial_balance
            
            # Apply position size limits
            max_size = self.live_config.max_position_size_usd
            actual_size = min(base_size, max_size)
            
            # Apply risk-based sizing
            risk_adjusted_size = actual_size * signal.conviction
            
            # Final position size
            final_size = min(risk_adjusted_size, max_size)
            
            # Convert to quantity (simplified - in production, use real price)
            quantity = final_size / signal.entry_price
            
            logger.info(f"📊 Position Sizing:")
            logger.info(f"   Base Size: ${base_size:.2f}")
            logger.info(f"   Risk Adjusted: ${risk_adjusted_size:.2f}")
            logger.info(f"   Final Size: ${final_size:.2f}")
            logger.info(f"   Quantity: {quantity:.6f}")
            
            return quantity
            
        except Exception as e:
            logger.error(f"❌ Position sizing failed: {e}")
            return 0.0
    
    async def _create_live_position(self, signal: CognitiveSignal, execution_result: ExecutionResult) -> AutonomousPosition:
        """Create a live position from execution result"""
        return AutonomousPosition(
            symbol=signal.symbol,
            side=signal.action,
            amount_eur=execution_result.filled_quantity * execution_result.average_price,
            amount_crypto=execution_result.filled_quantity,
            entry_price=execution_result.average_price,
            current_price=execution_result.average_price,
            unrealized_pnl=0.0,
            stop_loss=signal.stop_loss,
            profit_targets=signal.profit_targets,
            targets_hit=[False] * len(signal.profit_targets),
            strategy=signal.strategy,
            conviction=signal.conviction,
            entry_reasoning=signal.reasoning,
            entry_time=datetime.now(),
            max_holding_hours=signal.holding_period_hours,
            is_active=True
        )
    
    async def monitor_live_positions(self):
        """Monitor and manage live positions"""
        try:
            for symbol, position in list(self.active_positions.items()):
                if not position.is_active:
                    continue
                
                # Get current price
                current_price = await self._get_current_price(symbol)
                position.current_price = current_price
                
                # Calculate unrealized P&L
                if position.side == 'buy':
                    position.unrealized_pnl = (current_price - position.entry_price) * position.amount_crypto
                else:
                    position.unrealized_pnl = (position.entry_price - current_price) * position.amount_crypto
                
                # Check exit conditions
                should_close, reason = await self._check_exit_conditions(position)
                
                if should_close:
                    await self._close_live_position(symbol, reason)
        
        except Exception as e:
            logger.error(f"❌ Position monitoring failed: {e}")
    
    async def _get_current_price(self, symbol: str) -> float:
        """Get current price for symbol"""
        try:
            ticker = await self.binance_connector.get_ticker(symbol)
            return float(ticker['lastPrice'])
        except Exception as e:
            logger.error(f"❌ Price fetch failed for {symbol}: {e}")
            return 0.0
    
    async def _check_exit_conditions(self, position: AutonomousPosition) -> tuple[bool, str]:
        """Check if position should be closed"""
        try:
            # Check stop loss
            if position.stop_loss:
                if position.side == 'buy' and position.current_price <= position.stop_loss:
                    return True, "Stop loss hit"
                elif position.side == 'sell' and position.current_price >= position.stop_loss:
                    return True, "Stop loss hit"
            
            # Check profit targets
            for i, target in enumerate(position.profit_targets):
                if not position.targets_hit[i]:
                    if position.side == 'buy' and position.current_price >= target:
                        return True, f"Profit target {i+1} hit"
                    elif position.side == 'sell' and position.current_price <= target:
                        return True, f"Profit target {i+1} hit"
            
            # Check holding period
            holding_hours = (datetime.now() - position.entry_time).total_seconds() / 3600
            if holding_hours >= position.max_holding_hours:
                return True, "Maximum holding period reached"
            
            return False, "Position within limits"
            
        except Exception as e:
            logger.error(f"❌ Exit condition check failed: {e}")
            return False, "Check failed"
    
    async def _close_live_position(self, symbol: str, reason: str):
        """Close a live position"""
        try:
            position = self.active_positions[symbol]
            
            # Create close order
            close_action = 'sell' if position.side == 'buy' else 'buy'
            
            execution_request = ExecutionRequest(
                order_id=f"kimera_close_{symbol}_{int(time.time())}",
                symbol=symbol,
                side=close_action,
                quantity=position.amount_crypto,
                order_type=OrderType.MARKET,
                metadata={'reason': reason, 'live_trading': True}
            )
            
            # Execute close order
            logger.warning(f"🔄 CLOSING LIVE POSITION: {symbol} - {reason}")
            result = await self.execution_bridge.execute_order(execution_request)
            
            if result.status == OrderStatus.FILLED:
                # Calculate final P&L
                final_pnl = position.unrealized_pnl - result.fees
                
                # Update statistics
                self.daily_stats['total_fees'] += result.fees
                self.daily_stats['gross_pnl'] += position.unrealized_pnl
                self.daily_stats['net_pnl'] += final_pnl
                
                # Update trading manager
                self.trading_manager.update_daily_pnl(final_pnl)
                
                # Remove position
                position.is_active = False
                del self.active_positions[symbol]
                
                logger.info(f"✅ LIVE POSITION CLOSED:")
                logger.info(f"   Symbol: {symbol}")
                logger.info(f"   Reason: {reason}")
                logger.info(f"   P&L: ${final_pnl:.2f}")
                logger.info(f"   Fees: ${result.fees:.2f}")
                
            else:
                logger.error(f"❌ Failed to close position: {symbol}")
                
        except Exception as e:
            logger.error(f"❌ Position closure failed: {e}")
    
    def get_live_status(self) -> Dict[str, Any]:
        """Get current live trading status"""
        return {
            'trading_enabled': self.trading_manager.trading_enabled,
            'emergency_stop_active': self.trading_manager.emergency_stop_active,
            'active_positions': len(self.active_positions),
            'daily_pnl': self.trading_manager.daily_pnl,
            'daily_stats': self.daily_stats,
            'position_details': [
                {
                    'symbol': pos.symbol,
                    'side': pos.side,
                    'size': pos.amount_crypto,
                    'entry_price': pos.entry_price,
                    'current_price': pos.current_price,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'strategy': pos.strategy.value
                }
                for pos in self.active_positions.values()
            ]
        }
    
    async def emergency_stop_all(self):
        """Emergency stop - close all positions immediately"""
        logger.critical("🚨 EMERGENCY STOP - CLOSING ALL POSITIONS")
        
        self.trading_manager.emergency_stop()
        
        # Close all positions
        for symbol in list(self.active_positions.keys()):
            await self._close_live_position(symbol, "Emergency stop")
        
        logger.critical("🚨 EMERGENCY STOP COMPLETED")


# Factory function to create live execution bridge
def create_live_execution_bridge(
    binance_api_key: str,
    binance_private_key_path: str,
    initial_balance: float = 1000.0,
    max_position_size: float = 100.0,
    max_daily_loss: float = 50.0
) -> KimeraLiveExecutionBridge:
    """Create a live execution bridge with the specified configuration"""
    
    from src.trading.live_trading_config import create_live_trading_config
    
    config = create_live_trading_config(
        binance_api_key=binance_api_key,
        binance_private_key_path=binance_private_key_path,
        initial_balance=initial_balance,
        max_position_size=max_position_size,
        max_daily_loss=max_daily_loss
    )
    
    return KimeraLiveExecutionBridge(config)


# Example usage
if __name__ == "__main__":
    print("🔴 KIMERA LIVE EXECUTION BRIDGE")
    print("⚠️  WARNING: THIS MODULE TRADES WITH REAL MONEY")
    print()
    print("To use this module:")
    print("1. Set up your Binance API credentials")
    print("2. Create your Ed25519 private key file")
    print("3. Configure your risk parameters")
    print("4. Initialize the bridge with your credentials")
    print("5. Enable trading and monitor closely")
    print()
    print("NEVER leave live trading running unattended!") 