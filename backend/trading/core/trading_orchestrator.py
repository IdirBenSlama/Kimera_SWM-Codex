"""
Trading Orchestrator

Main orchestration layer that coordinates between Kimera's cognitive engine,
market analysis, and exchange execution for real-time crypto trading.
"""

import os
import asyncio
import logging
from typing import Dict, List, Optional, Any, Set, Callable
from datetime import datetime, timedelta
import json
from dataclasses import dataclass, asdict
import numpy as np

from backend.trading.core.trading_engine import KimeraTradingEngine, MarketState, TradingDecision
from backend.trading.api.binance_connector import BinanceConnector
from backend.trading.monitoring.performance_tracker import PerformanceTracker
from backend.trading.strategies.strategy_manager import StrategyManager

logger = logging.getLogger(__name__)


@dataclass
class TradingSession:
    """Trading session information"""
    session_id: str
    start_time: datetime
    symbols: List[str]
    initial_balance: float
    current_balance: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    status: str  # ACTIVE, PAUSED, STOPPED


class TradingOrchestrator:
    """
    Main orchestrator that coordinates all trading components.
    Handles real-time market monitoring, decision making, and execution.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the trading orchestrator.
        
        Args:
            config: Trading configuration including:
                - api_key: Binance API key
                - api_secret: Binance API secret
                - symbols: List of trading symbols
                - initial_balance: Starting balance
                - risk_params: Risk management parameters
        """
        self.config = config
        self.trading_engine = KimeraTradingEngine(config)
        self.performance_tracker = PerformanceTracker()
        self.strategy_manager = StrategyManager()
        
        # Initialize exchange connector
        self.exchange = BinanceConnector(
            api_key=config.get("api_key", ""),
            api_secret=config.get("api_secret", ""),
            testnet=os.getenv('KIMERA_USE_TESTNET', 'false').lower() == 'true'  # Default to real trading
        )
        
        # Trading state
        self.active_symbols: Set[str] = set(config.get("symbols", ["BTCUSDT"]))
        self.market_states: Dict[str, MarketState] = {}
        self.pending_orders: Dict[str, Any] = {}
        self.session: Optional[TradingSession] = None
        
        # Control flags
        self.is_running = False
        self.pause_trading = False
        
        # Performance monitoring
        self.update_interval = config.get("update_interval", 5)  # seconds
        self.decision_interval = config.get("decision_interval", 30)  # seconds
        
        logger.info("Trading Orchestrator initialized")
    
    async def start_trading(self) -> None:
        """Start the trading system"""
        try:
            logger.info("Starting trading system...")
            
            # Initialize session
            await self._initialize_session()
            
            # Connect to exchange
            await self.exchange.__aenter__()
            
            # Start background tasks
            self.is_running = True
            
            # Create concurrent tasks
            tasks = [
                asyncio.create_task(self._market_monitor_loop()),
                asyncio.create_task(self._decision_loop()),
                asyncio.create_task(self._order_monitor_loop()),
                asyncio.create_task(self._performance_monitor_loop()),
            ]
            
            # Subscribe to real-time data
            for symbol in self.active_symbols:
                await self._subscribe_market_data(symbol)
            
            logger.info("Trading system started successfully")
            
            # Run until stopped
            await asyncio.gather(*tasks)
            
        except Exception as e:
            logger.error(f"Failed to start trading: {str(e)}")
            await self.stop_trading()
            raise
    
    async def stop_trading(self) -> None:
        """Stop the trading system gracefully"""
        logger.info("Stopping trading system...")
        
        self.is_running = False
        
        # Cancel all pending orders
        await self._cancel_all_orders()
        
        # Close all positions (optional, depends on strategy)
        if self.config.get("close_on_stop", False):
            await self._close_all_positions()
        
        # Close exchange connection
        await self.exchange.close()
        
        # Save session data
        await self._save_session()
        
        logger.info("Trading system stopped")
    
    async def pause_trading(self) -> None:
        """Pause trading (monitoring continues)"""
        self.pause_trading = True
        logger.info("Trading paused")
    
    async def resume_trading(self) -> None:
        """Resume trading"""
        self.pause_trading = False
        logger.info("Trading resumed")
    
    async def _initialize_session(self) -> None:
        """Initialize trading session"""
        # Get initial balance
        account = await self.exchange.get_account()
        initial_balance = self._calculate_total_balance(account)
        
        self.session = TradingSession(
            session_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
            start_time=datetime.now(),
            symbols=list(self.active_symbols),
            initial_balance=initial_balance,
            current_balance=initial_balance,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            status="ACTIVE"
        )
        
        logger.info(f"Session initialized: {self.session.session_id}")
    
    async def _market_monitor_loop(self) -> None:
        """Continuously monitor market conditions"""
        while self.is_running:
            try:
                # Update market data for all symbols
                for symbol in self.active_symbols:
                    market_data = await self.exchange.get_market_data(symbol)
                    
                    # Analyze with Kimera
                    market_state = await self.trading_engine.analyze_market(
                        symbol, market_data
                    )
                    
                    self.market_states[symbol] = market_state
                    
                    # Log significant changes
                    if market_state.cognitive_pressure > 0.8:
                        logger.warning(
                            f"{symbol}: High cognitive pressure detected: "
                            f"{market_state.cognitive_pressure}"
                        )
                    
                    if market_state.contradiction_level > 0.7:
                        logger.info(
                            f"{symbol}: Market inefficiency detected: "
                            f"{market_state.contradiction_level}"
                        )
                
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Market monitoring error: {str(e)}")
                await asyncio.sleep(self.update_interval)
    
    async def _decision_loop(self) -> None:
        """Make trading decisions at regular intervals"""
        while self.is_running:
            try:
                if not self.pause_trading:
                    # Get current portfolio state
                    portfolio_state = await self._get_portfolio_state()
                    
                    # Make decisions for each symbol
                    for symbol in self.active_symbols:
                        if symbol in self.market_states:
                            market_state = self.market_states[symbol]
                            
                            # Get trading decision from Kimera
                            decision = await self.trading_engine.make_trading_decision(
                                symbol, market_state, portfolio_state
                            )
                            
                            # Execute decision
                            await self._execute_decision(symbol, decision)
                
                await asyncio.sleep(self.decision_interval)
                
            except Exception as e:
                logger.error(f"Decision loop error: {str(e)}")
                await asyncio.sleep(self.decision_interval)
    
    async def _execute_decision(self, symbol: str, decision: TradingDecision) -> None:
        """Execute trading decision"""
        try:
            if decision.action == "HOLD":
                logger.info(f"{symbol}: Holding position")
                return
            
            # Check risk limits
            if not await self._check_risk_limits(decision):
                logger.warning(f"{symbol}: Decision rejected by risk management")
                return
            
            # Place order
            if decision.action == "BUY":
                order = await self.exchange.place_order(
                    symbol=symbol,
                    side="BUY",
                    order_type="MARKET",
                    quantity=decision.size / self.market_states[symbol].price
                )
                
                # Place stop loss if specified
                if decision.stop_loss:
                    sl_order = await self.exchange.place_order(
                        symbol=symbol,
                        side="SELL",
                        order_type="STOP_LOSS_LIMIT",
                        quantity=order["executedQty"],
                        price=decision.stop_loss * 0.995,  # Slightly below stop
                        stop_price=decision.stop_loss
                    )
                    self.pending_orders[sl_order["orderId"]] = sl_order
                
                # Place take profit if specified
                if decision.take_profit:
                    tp_order = await self.exchange.place_order(
                        symbol=symbol,
                        side="SELL",
                        order_type="LIMIT",
                        quantity=order["executedQty"],
                        price=decision.take_profit
                    )
                    self.pending_orders[tp_order["orderId"]] = tp_order
                
            elif decision.action == "SELL":
                order = await self.exchange.place_order(
                    symbol=symbol,
                    side="SELL",
                    order_type="MARKET",
                    quantity=self.trading_engine.active_positions[symbol]["size"]
                )
            
            # Update session stats
            self.session.total_trades += 1
            
            # Log execution
            logger.info(
                f"{symbol}: Executed {decision.action} order - "
                f"Size: {decision.size}, Confidence: {decision.confidence:.2f}"
            )
            
            # Store decision for analysis
            await self.performance_tracker.record_decision(symbol, decision, order)
            
        except Exception as e:
            logger.error(f"Failed to execute decision: {str(e)}")
    
    async def _order_monitor_loop(self) -> None:
        """Monitor order status and handle fills"""
        while self.is_running:
            try:
                # Check pending orders
                for order_id in list(self.pending_orders.keys()):
                    order = await self.exchange.get_order(
                        symbol=self.pending_orders[order_id]["symbol"],
                        order_id=order_id
                    )
                    
                    if order["status"] == "FILLED":
                        # Handle filled order
                        await self._handle_filled_order(order)
                        del self.pending_orders[order_id]
                    
                    elif order["status"] in ["CANCELED", "REJECTED", "EXPIRED"]:
                        # Remove from pending
                        del self.pending_orders[order_id]
                
                await asyncio.sleep(2)  # Check every 2 seconds
                
            except Exception as e:
                logger.error(f"Order monitoring error: {str(e)}")
                await asyncio.sleep(2)
    
    async def _performance_monitor_loop(self) -> None:
        """Monitor and log performance metrics"""
        while self.is_running:
            try:
                # Get current metrics
                metrics = await self._calculate_performance_metrics()
                
                # Update session
                self.session.current_balance = metrics["current_balance"]
                
                # Log performance
                logger.info(
                    f"Performance Update - "
                    f"P&L: ${metrics['total_pnl']:.2f} ({metrics['pnl_percent']:.2f}%), "
                    f"Win Rate: {metrics['win_rate']:.2f}%, "
                    f"Sharpe: {metrics['sharpe_ratio']:.2f}"
                )
                
                # Check for issues
                if metrics["total_pnl"] < -self.config.get("max_daily_loss", 0.05) * self.session.initial_balance:
                    logger.error("Daily loss limit reached! Pausing trading.")
                    await self.pause_trading()
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {str(e)}")
                await asyncio.sleep(60)
    
    async def _subscribe_market_data(self, symbol: str) -> None:
        """Subscribe to real-time market data"""
        # Subscribe to ticker updates
        await self.exchange.subscribe_ticker(
            symbol, 
            lambda data: asyncio.create_task(self._handle_ticker_update(symbol, data))
        )
        
        # Subscribe to order book updates
        await self.exchange.subscribe_orderbook(
            symbol,
            lambda data: asyncio.create_task(self._handle_orderbook_update(symbol, data))
        )
    
    async def _handle_ticker_update(self, symbol: str, data: Dict[str, Any]) -> None:
        """Handle real-time ticker updates"""
        # Quick update of price in market state
        if symbol in self.market_states:
            self.market_states[symbol].price = float(data.get("c", data.get("lastPrice", 0)))
    
    async def _handle_orderbook_update(self, symbol: str, data: Dict[str, Any]) -> None:
        """Handle real-time orderbook updates"""
        # Could trigger immediate decisions on significant changes
        pass
    
    async def _handle_filled_order(self, order: Dict[str, Any]) -> None:
        """Handle filled order"""
        symbol = order["symbol"]
        side = order["side"]
        
        # Update positions
        if side == "BUY":
            self.trading_engine.active_positions[symbol] = {
                "size": float(order["executedQty"]),
                "entry_price": float(order["price"]),
                "entry_time": datetime.now(),
                "value": float(order["executedQty"]) * float(order["price"])
            }
        else:  # SELL
            if symbol in self.trading_engine.active_positions:
                position = self.trading_engine.active_positions[symbol]
                pnl = (float(order["price"]) - position["entry_price"]) * position["size"]
                
                # Update session stats
                if pnl > 0:
                    self.session.winning_trades += 1
                else:
                    self.session.losing_trades += 1
                
                # Clear position
                del self.trading_engine.active_positions[symbol]
                
                # Update performance
                await self.trading_engine.update_performance({
                    "pnl": pnl,
                    "total_trades": self.session.total_trades,
                    "winning_trades": self.session.winning_trades
                })
    
    async def _get_portfolio_state(self) -> Dict[str, Any]:
        """Get current portfolio state"""
        account = await self.exchange.get_account()
        
        total_value = self._calculate_total_balance(account)
        
        return {
            "total_value": total_value,
            "free_balance": self._get_free_balance(account),
            "positions": self.trading_engine.active_positions,
            "daily_pnl": total_value - self.session.initial_balance,
            "margin_used": self._calculate_margin_used()
        }
    
    async def _check_risk_limits(self, decision: TradingDecision) -> bool:
        """Check if decision passes risk management rules"""
        portfolio_state = await self._get_portfolio_state()
        
        # Check daily loss limit
        if portfolio_state["daily_pnl"] < -self.config.get("max_daily_loss", 0.05) * self.session.initial_balance:
            return False
        
        # Check position size limit
        if decision.size > self.config.get("max_position_size", 0.1) * portfolio_state["total_value"]:
            return False
        
        # Check number of concurrent positions
        if len(self.trading_engine.active_positions) >= self.config.get("max_positions", 5):
            return False
        
        return True
    
    async def _cancel_all_orders(self) -> None:
        """Cancel all pending orders"""
        for order_id, order in self.pending_orders.items():
            try:
                await self.exchange.cancel_order(order["symbol"], order_id)
            except Exception as e:
                logger.error(f"Failed to cancel order {order_id}: {str(e)}")
    
    async def _close_all_positions(self) -> None:
        """Close all open positions"""
        for symbol, position in list(self.trading_engine.active_positions.items()):
            try:
                await self.exchange.place_order(
                    symbol=symbol,
                    side="SELL",
                    order_type="MARKET",
                    quantity=position["size"]
                )
            except Exception as e:
                logger.error(f"Failed to close position {symbol}: {str(e)}")
    
    async def _calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate current performance metrics"""
        account = await self.exchange.get_account()
        current_balance = self._calculate_total_balance(account)
        
        total_pnl = current_balance - self.session.initial_balance
        pnl_percent = (total_pnl / self.session.initial_balance) * 100
        
        total_trades = self.session.total_trades
        win_rate = (self.session.winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Calculate Sharpe ratio (simplified)
        returns = []  # Would need to track historical returns
        sharpe_ratio = 0.0  # Placeholder
        
        return {
            "current_balance": current_balance,
            "total_pnl": total_pnl,
            "pnl_percent": pnl_percent,
            "win_rate": win_rate,
            "sharpe_ratio": sharpe_ratio,
            "total_trades": total_trades
        }
    
    def _calculate_total_balance(self, account: Dict[str, Any]) -> float:
        """Calculate total account balance in USDT"""
        total = 0.0
        
        for balance in account["balances"]:
            asset = balance["asset"]
            free = float(balance["free"])
            locked = float(balance["locked"])
            
            if asset == "USDT":
                total += free + locked
            # Add conversion for other assets if needed
        
        return total
    
    def _get_free_balance(self, account: Dict[str, Any]) -> float:
        """Get free USDT balance"""
        for balance in account["balances"]:
            if balance["asset"] == "USDT":
                return float(balance["free"])
        return 0.0
    
    def _calculate_margin_used(self) -> float:
        """Calculate margin currently in use"""
        total_margin = 0.0
        
        for position in self.trading_engine.active_positions.values():
            total_margin += position["value"]
        
        return total_margin
    
    async def _save_session(self) -> None:
        """Save session data to file"""
        if self.session:
            self.session.status = "STOPPED"
            
            session_data = {
                **asdict(self.session),
                "end_time": datetime.now().isoformat(),
                "performance_metrics": self.trading_engine.performance_metrics,
                "market_states": {
                    symbol: {
                        "timestamp": state.timestamp.isoformat(),
                        "price": state.price,
                        "cognitive_pressure": state.cognitive_pressure,
                        "contradiction_level": state.contradiction_level,
                        "semantic_temperature": state.semantic_temperature
                    }
                    for symbol, state in self.market_states.items()
                }
            }
            
            filename = f"trading_session_{self.session.session_id}.json"
            
            with open(filename, "w") as f:
                json.dump(session_data, f, indent=2, default=str)
            
            logger.info(f"Session saved to {filename}") 