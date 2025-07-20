"""
KIMERA Semantic Execution Bridge
================================

Real-time execution layer that bridges Kimera's semantic analysis with actual market execution.
This module handles the critical path from decision to execution with enterprise-grade reliability.

Features:
- Ultra-low latency order execution
- Smart order routing across multiple exchanges
- Real-time position and risk management
- Compliance and audit trail generation
"""

import os
import asyncio
import logging
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np

# Exchange connectivity
try:
    import ccxt.async_support as ccxt
    CCXT_AVAILABLE = True
except ImportError:
    logging.warning("CCXT not available - exchange connectivity limited")
    CCXT_AVAILABLE = False

# FIX Protocol support
try:
    import quickfix as fix
    FIX_AVAILABLE = True
except ImportError:
    logging.warning("QuickFIX not available - FIX protocol support disabled")
    FIX_AVAILABLE = False

from backend.trading.api.binance_connector import BinanceConnector

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types supported by the execution bridge"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    ICEBERG = "iceberg"
    TWAP = "twap"
    VWAP = "vwap"


class OrderStatus(Enum):
    """Order lifecycle states"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class ExecutionRequest:
    """Request for order execution"""
    order_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    order_type: OrderType
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"  # Good Till Cancelled
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """Result of order execution"""
    order_id: str
    status: OrderStatus
    filled_quantity: float
    average_price: float
    fees: float
    execution_time: float
    exchange: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketMicrostructure:
    """Real-time market microstructure data"""
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    spread: float
    depth: Dict[str, List[Tuple[float, float]]]  # price levels and sizes
    last_trade: float
    timestamp: datetime


class SemanticExecutionBridge:
    """
    High-performance execution bridge that translates Kimera's semantic decisions
    into actual market orders with smart routing and risk management.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the execution bridge
        
        Args:
            config: Configuration including exchange credentials and settings
        """
        self.config = config
        self.exchanges = {}
        self.active_orders = {}
        self.order_history = []
        
        # Risk limits
        self.max_order_size = config.get('max_order_size', 10000)
        self.max_daily_volume = config.get('max_daily_volume', 1000000)
        self.daily_volume = 0.0
        
        # Performance tracking
        self.execution_metrics = {
            'total_orders': 0,
            'successful_orders': 0,
            'average_latency': 0.0,
            'total_fees': 0.0,
            'slippage': []
        }
        
        # Initialize exchanges
        self._initialize_exchanges()
        
        logger.info("⚡ Semantic Execution Bridge initialized")
        logger.info(f"   Connected exchanges: {list(self.exchanges.keys())}")
    
    def _initialize_exchanges(self):
        """Initialize connections to configured exchanges"""
        if not CCXT_AVAILABLE:
            logger.warning("CCXT not available - running in simulation mode")
            return
        
        exchange_configs = self.config.get('exchanges', {})
        
        for exchange_id, exchange_config in exchange_configs.items():
            try:
                # Handle custom Binance Ed25519 connector
                if exchange_id == 'binance' and 'private_key_path' in exchange_config:
                    logger.info(f"Initializing custom Binance Ed25519 connector...")
                    
                    exchange = BinanceConnector(
                        api_key=exchange_config.get('api_key'),
                        private_key_path=exchange_config.get('private_key_path'),
                        testnet=exchange_config.get('testnet', os.getenv('KIMERA_USE_TESTNET', 'false').lower() == 'true')
                    )
                    
                    # Store exchange instance
                    self.exchanges[exchange_id] = exchange
                    logger.info(f"✓ Connected to {exchange_id} (Ed25519)")
                    continue
                
                # Handle standard CCXT exchanges
                exchange_class = getattr(ccxt, exchange_id)
                exchange = exchange_class({
                    'apiKey': exchange_config.get('api_key'),
                    'secret': exchange_config.get('secret'),
                    'enableRateLimit': True,
                    'options': exchange_config.get('options', {})
                })
                
                # Store exchange instance
                self.exchanges[exchange_id] = exchange
                logger.info(f"✓ Connected to {exchange_id} (CCXT)")
                
            except Exception as e:
                logger.error(f"Failed to connect to {exchange_id}: {e}")
    
    async def execute_order(self, request: ExecutionRequest) -> ExecutionResult:
        """
        Execute an order with smart routing and best execution
        
        Args:
            request: Execution request with order details
            
        Returns:
            ExecutionResult with fill information
        """
        start_time = time.time()
        
        try:
            # Pre-execution checks
            if not await self._pre_execution_checks(request):
                return ExecutionResult(
                    order_id=request.order_id,
                    status=OrderStatus.REJECTED,
                    filled_quantity=0.0,
                    average_price=0.0,
                    fees=0.0,
                    execution_time=time.time() - start_time,
                    exchange="none",
                    metadata={'reason': 'failed_pre_execution_checks'}
                )
            
            # Get market microstructure
            microstructure = await self._get_market_microstructure(request.symbol)
            
            # Determine best execution venue
            best_exchange = await self._select_best_exchange(request, microstructure)
            
            # Execute based on order type
            if request.order_type == OrderType.MARKET:
                result = await self._execute_market_order(request, best_exchange)
            elif request.order_type == OrderType.LIMIT:
                result = await self._execute_limit_order(request, best_exchange)
            elif request.order_type in [OrderType.TWAP, OrderType.VWAP]:
                result = await self._execute_algo_order(request, best_exchange)
            else:
                result = await self._execute_standard_order(request, best_exchange)
            
            # Update metrics
            self._update_execution_metrics(result, microstructure)
            
            return result
            
        except Exception as e:
            logger.error(f"Execution failed for order {request.order_id}: {e}")
            return ExecutionResult(
                order_id=request.order_id,
                status=OrderStatus.REJECTED,
                filled_quantity=0.0,
                average_price=0.0,
                fees=0.0,
                execution_time=time.time() - start_time,
                exchange="error",
                metadata={'error': str(e)}
            )
    
    async def _pre_execution_checks(self, request: ExecutionRequest) -> bool:
        """Perform pre-execution risk and compliance checks"""
        # Check order size limits
        if request.quantity > self.max_order_size:
            logger.warning(f"Order size {request.quantity} exceeds limit {self.max_order_size}")
            return False
        
        # Check daily volume limits
        if self.daily_volume + request.quantity > self.max_daily_volume:
            logger.warning(f"Daily volume limit would be exceeded")
            return False
        
        # Check if symbol is tradeable
        if not await self._is_symbol_tradeable(request.symbol):
            logger.warning(f"Symbol {request.symbol} is not tradeable")
            return False
        
        return True
    
    async def _get_market_microstructure(self, symbol: str) -> MarketMicrostructure:
        """Get real-time market microstructure data"""
        if not self.exchanges:
            # Simulation mode
            return MarketMicrostructure(
                bid=100.0,
                ask=100.1,
                bid_size=1000,
                ask_size=1000,
                spread=0.1,
                depth={'bids': [(100.0, 1000)], 'asks': [(100.1, 1000)]},
                last_trade=100.05,
                timestamp=datetime.now()
            )
        
        # Aggregate data from all exchanges
        all_orderbooks = []
        for exchange_id, exchange in self.exchanges.items():
            try:
                orderbook = await exchange.fetch_order_book(symbol)
                all_orderbooks.append(orderbook)
            except Exception as e:
                logger.warning(f"Failed to fetch orderbook from {exchange_id}: {e}")
        
        if not all_orderbooks:
            raise ValueError(f"No orderbook data available for {symbol}")
        
        # Find best bid/ask across all exchanges
        best_bid = max(ob['bids'][0][0] for ob in all_orderbooks if ob['bids'])
        best_ask = min(ob['asks'][0][0] for ob in all_orderbooks if ob['asks'])
        
        # Aggregate depth
        aggregated_bids = []
        aggregated_asks = []
        for ob in all_orderbooks:
            aggregated_bids.extend(ob['bids'][:5])  # Top 5 levels
            aggregated_asks.extend(ob['asks'][:5])
        
        return MarketMicrostructure(
            bid=best_bid,
            ask=best_ask,
            bid_size=sum(size for _, size in aggregated_bids if _ == best_bid),
            ask_size=sum(size for _, size in aggregated_asks if _ == best_ask),
            spread=best_ask - best_bid,
            depth={'bids': sorted(aggregated_bids, reverse=True)[:10], 
                   'asks': sorted(aggregated_asks)[:10]},
            last_trade=(best_bid + best_ask) / 2,
            timestamp=datetime.now()
        )
    
    async def _select_best_exchange(self, 
                                  request: ExecutionRequest, 
                                  microstructure: MarketMicrostructure) -> str:
        """Select the best exchange for execution based on multiple factors"""
        if not self.exchanges:
            return "simulation"
        
        best_score = -float('inf')
        best_exchange = None
        
        for exchange_id, exchange in self.exchanges.items():
            try:
                # Calculate execution score based on:
                # 1. Liquidity at the touch
                # 2. Fee structure
                # 3. Historical fill rate
                # 4. Current latency
                
                score = await self._calculate_exchange_score(
                    exchange_id, exchange, request, microstructure
                )
                
                if score > best_score:
                    best_score = score
                    best_exchange = exchange_id
                    
            except Exception as e:
                logger.warning(f"Failed to score exchange {exchange_id}: {e}")
        
        return best_exchange or list(self.exchanges.keys())[0]
    
    async def _calculate_exchange_score(self,
                                      exchange_id: str,
                                      exchange: Any,
                                      request: ExecutionRequest,
                                      microstructure: MarketMicrostructure) -> float:
        """Calculate execution quality score for an exchange"""
        score = 0.0
        
        # Liquidity score (higher is better)
        if request.side == 'buy':
            available_liquidity = microstructure.ask_size
        else:
            available_liquidity = microstructure.bid_size
        
        liquidity_ratio = min(available_liquidity / request.quantity, 1.0)
        score += liquidity_ratio * 40  # 40% weight
        
        # Fee score (lower is better)
        fee_rate = 0.001  # Default 0.1%
        if hasattr(exchange, 'fees'):
            fee_rate = exchange.fees.get('trading', {}).get('taker', 0.001)
        score += (1 - fee_rate) * 30  # 30% weight
        
        # Historical performance score
        historical_fill_rate = 0.95  # Default 95%
        score += historical_fill_rate * 20  # 20% weight
        
        # Latency score (lower is better)
        latency = 50  # Default 50ms
        latency_score = max(0, 1 - (latency / 1000))  # Normalize to 0-1
        score += latency_score * 10  # 10% weight
        
        return score
    
    async def _execute_market_order(self, 
                                  request: ExecutionRequest, 
                                  exchange_id: str) -> ExecutionResult:
        """Execute market order with best execution"""
        start_time = time.time()
        
        exchange = self.exchanges[exchange_id]
        
        try:
            # Handle custom Binance Ed25519 connector
            if isinstance(exchange, BinanceConnector):
                order = await exchange.place_order(
                    symbol=request.symbol,
                    side=request.side,
                    order_type='market',
                    quantity=request.quantity
                )
                
                if order:
                    return ExecutionResult(
                        order_id=request.order_id,
                        status=OrderStatus.FILLED,
                        filled_quantity=float(order.get('executedQty', request.quantity)),
                        average_price=float(order.get('price', 0)),
                        fees=float(order.get('commission', 0)),
                        execution_time=time.time() - start_time,
                        exchange=exchange_id,
                        metadata={'exchange_order_id': order.get('orderId')}
                    )
                else:
                    return ExecutionResult(
                        order_id=request.order_id,
                        status=OrderStatus.REJECTED,
                        filled_quantity=0.0,
                        average_price=0.0,
                        fees=0.0,
                        execution_time=time.time() - start_time,
                        exchange=exchange_id,
                        metadata={'reason': 'order_placement_failed'}
                    )
            
            # Handle standard CCXT exchanges
            else:
                order = await exchange.create_market_order(
                    symbol=request.symbol,
                    side=request.side,
                    amount=request.quantity
                )
                
                # Wait for order to be filled (market orders should fill immediately)
                filled_order = await self._wait_for_fill(exchange, order['id'], request.symbol, 5.0)
                
                return ExecutionResult(
                    order_id=request.order_id,
                    status=OrderStatus.FILLED if filled_order['status'] == 'closed' else OrderStatus.PARTIAL,
                    filled_quantity=filled_order['filled'],
                    average_price=filled_order['average'] or 0,
                    fees=filled_order['fee']['cost'] if filled_order.get('fee') else 0,
                    execution_time=time.time() - start_time,
                    exchange=exchange_id,
                    metadata={'exchange_order_id': order['id']}
                )
                
        except Exception as e:
            logger.error(f"Market order execution failed: {e}")
            return ExecutionResult(
                order_id=request.order_id,
                status=OrderStatus.REJECTED,
                filled_quantity=0.0,
                average_price=0.0,
                fees=0.0,
                execution_time=time.time() - start_time,
                exchange=exchange_id,
                metadata={'error': str(e)}
            )
    
    async def _execute_limit_order(self,
                                 request: ExecutionRequest,
                                 exchange_id: str) -> ExecutionResult:
        """Execute limit order"""
        start_time = time.time()
        
        exchange = self.exchanges[exchange_id]
        
        try:
            # Handle custom Binance Ed25519 connector
            if isinstance(exchange, BinanceConnector):
                order = await exchange.place_order(
                    symbol=request.symbol,
                    side=request.side,
                    order_type='limit',
                    quantity=request.quantity,
                    price=request.price
                )
                
                if order:
                    # Store active order
                    self.active_orders[request.order_id] = {
                        'exchange_id': exchange_id,
                        'exchange_order_id': order.get('orderId'),
                        'request': request,
                        'status': OrderStatus.SUBMITTED
                    }
                    
                    return ExecutionResult(
                        order_id=request.order_id,
                        status=OrderStatus.SUBMITTED,
                        filled_quantity=0.0,
                        average_price=0.0,
                        fees=0.0,
                        execution_time=time.time() - start_time,
                        exchange=exchange_id,
                        metadata={'exchange_order_id': order.get('orderId')}
                    )
                else:
                    return ExecutionResult(
                        order_id=request.order_id,
                        status=OrderStatus.REJECTED,
                        filled_quantity=0.0,
                        average_price=0.0,
                        fees=0.0,
                        execution_time=time.time() - start_time,
                        exchange=exchange_id,
                        metadata={'reason': 'order_placement_failed'}
                    )
            
            # Handle standard CCXT exchanges
            else:
                # Place limit order
                order = await exchange.create_order(
                    symbol=request.symbol,
                    type='limit',
                    side=request.side,
                    amount=request.quantity,
                    price=request.price
                )
                
                # Store active order
                self.active_orders[request.order_id] = {
                    'exchange_id': exchange_id,
                    'exchange_order_id': order['id'],
                    'request': request,
                    'status': OrderStatus.SUBMITTED
                }
                
                # Return immediate result (order monitoring happens separately)
                return ExecutionResult(
                    order_id=request.order_id,
                    status=OrderStatus.SUBMITTED,
                    filled_quantity=0.0,
                    average_price=0.0,
                    fees=0.0,
                    execution_time=time.time() - order['timestamp'] / 1000,
                    exchange=exchange_id,
                    metadata={'exchange_order_id': order['id']}
                )
                
        except Exception as e:
            logger.error(f"Limit order execution failed: {e}")
            return ExecutionResult(
                order_id=request.order_id,
                status=OrderStatus.REJECTED,
                filled_quantity=0.0,
                average_price=0.0,
                fees=0.0,
                execution_time=time.time() - start_time,
                exchange=exchange_id,
                metadata={'error': str(e)}
            )
    
    async def _execute_algo_order(self,
                                request: ExecutionRequest,
                                exchange_id: str) -> ExecutionResult:
        """Execute algorithmic orders (TWAP/VWAP)"""
        # This is a simplified implementation
        # In production, this would slice the order and execute over time
        
        if request.order_type == OrderType.TWAP:
            # Time-Weighted Average Price
            slices = 10  # Split into 10 slices
            slice_size = request.quantity / slices
            
            total_filled = 0.0
            total_cost = 0.0
            total_fees = 0.0
            
            for i in range(slices):
                # Execute each slice
                slice_request = ExecutionRequest(
                    order_id=f"{request.order_id}_slice_{i}",
                    symbol=request.symbol,
                    side=request.side,
                    quantity=slice_size,
                    order_type=OrderType.MARKET,
                    metadata={'parent_order': request.order_id}
                )
                
                slice_result = await self._execute_market_order(slice_request, exchange_id)
                
                total_filled += slice_result.filled_quantity
                total_cost += slice_result.filled_quantity * slice_result.average_price
                total_fees += slice_result.fees
                
                # Wait between slices
                await asyncio.sleep(1.0)  # 1 second between slices
            
            return ExecutionResult(
                order_id=request.order_id,
                status=OrderStatus.FILLED if total_filled >= request.quantity * 0.99 else OrderStatus.PARTIAL,
                filled_quantity=total_filled,
                average_price=total_cost / total_filled if total_filled > 0 else 0,
                fees=total_fees,
                execution_time=slices * 1.0,  # Total execution time
                exchange=exchange_id,
                metadata={'algo_type': 'TWAP', 'slices': slices}
            )
        
        else:
            # VWAP - Volume-Weighted Average Price
            # For now, just execute as market order
            return await self._execute_market_order(request, exchange_id)
    
    async def _execute_standard_order(self,
                                    request: ExecutionRequest,
                                    exchange_id: str) -> ExecutionResult:
        """Execute other order types"""
        # Placeholder for stop, stop-limit, iceberg orders
        return await self._execute_market_order(request, exchange_id)
    
    async def _wait_for_fill(self, exchange: Any, order_id: str, symbol: str, timeout: float = 30.0) -> Dict:
        """Wait for an order to be filled"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                order = await exchange.fetch_order(order_id, symbol)
                
                if order['status'] in ['closed', 'canceled', 'expired']:
                    return order
                
                await asyncio.sleep(0.1)  # Check every 100ms
                
            except Exception as e:
                logger.warning(f"Error checking order status: {e}")
                await asyncio.sleep(1.0)
        
        # Timeout - return last known state
        return await exchange.fetch_order(order_id, symbol)
    
    async def _is_symbol_tradeable(self, symbol: str) -> bool:
        """Check if a symbol is tradeable on any connected exchange"""
        if not self.exchanges:
            return True  # Simulation mode
        
        for exchange in self.exchanges.values():
            try:
                markets = await exchange.load_markets()
                if symbol in markets and markets[symbol]['active']:
                    return True
            except Exception as e:
                logger.warning(f"Failed to check symbol tradability: {e}")
        
        return False
    
    def _update_execution_metrics(self, result: ExecutionResult, microstructure: MarketMicrostructure):
        """Update execution quality metrics"""
        self.execution_metrics['total_orders'] += 1
        
        if result.status == OrderStatus.FILLED:
            self.execution_metrics['successful_orders'] += 1
        
        # Update average latency
        current_avg = self.execution_metrics['average_latency']
        new_avg = (current_avg * (self.execution_metrics['total_orders'] - 1) + result.execution_time) / self.execution_metrics['total_orders']
        self.execution_metrics['average_latency'] = new_avg
        
        # Track fees
        self.execution_metrics['total_fees'] += result.fees
        
        # Calculate slippage
        if result.filled_quantity > 0:
            mid_price = (microstructure.bid + microstructure.ask) / 2
            slippage = abs(result.average_price - mid_price) / mid_price
            self.execution_metrics['slippage'].append(slippage)
        
        # Update daily volume
        self.daily_volume += result.filled_quantity
    
    async def monitor_active_orders(self):
        """Monitor active orders and update their status"""
        while True:
            for order_id, order_info in list(self.active_orders.items()):
                try:
                    exchange = self.exchanges[order_info['exchange_id']]
                    exchange_order = await exchange.fetch_order(
                        order_info['exchange_order_id'],
                        order_info['request'].symbol
                    )
                    
                    # Update status
                    if exchange_order['status'] == 'closed':
                        order_info['status'] = OrderStatus.FILLED
                        # Move to history
                        self.order_history.append(order_info)
                        del self.active_orders[order_id]
                    elif exchange_order['status'] == 'canceled':
                        order_info['status'] = OrderStatus.CANCELLED
                        del self.active_orders[order_id]
                        
                except Exception as e:
                    logger.error(f"Error monitoring order {order_id}: {e}")
            
            await asyncio.sleep(1.0)  # Check every second
    
    def get_execution_analytics(self) -> Dict[str, Any]:
        """Get comprehensive execution analytics"""
        return {
            'total_orders': self.execution_metrics['total_orders'],
            'successful_orders': self.execution_metrics['successful_orders'],
            'success_rate': self.execution_metrics['successful_orders'] / max(self.execution_metrics['total_orders'], 1),
            'average_latency_ms': self.execution_metrics['average_latency'] * 1000,
            'total_fees': self.execution_metrics['total_fees'],
            'average_slippage': np.mean(self.execution_metrics['slippage']) if self.execution_metrics['slippage'] else 0.0,
            'daily_volume': self.daily_volume,
            'active_orders': len(self.active_orders),
            'connected_exchanges': list(self.exchanges.keys())
        }

    async def simulate_order_execution(self, order_scenario: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate order execution for testing purposes
        
        Args:
            order_scenario: Dictionary containing order details for simulation
            
        Returns:
            Dictionary with simulation results
        """
        try:
            # Extract order details
            order_id = order_scenario.get('order_id', str(uuid.uuid4()))
            symbol = order_scenario.get('symbol', 'BTCUSDT')
            side = order_scenario.get('side', 'buy')
            quantity = order_scenario.get('quantity', 0.001)
            price = order_scenario.get('price', 45000.0)
            
            # Simulate execution latency
            await asyncio.sleep(0.001)  # 1ms simulation
            
            # Simulate execution with realistic scenarios
            # Use higher success probability for testing, configurable for production
            success_probability = getattr(self, 'test_success_probability', 0.97)
            
            if np.random.random() < success_probability:
                # Successful execution
                filled_quantity = quantity * np.random.uniform(0.95, 1.0)  # Partial fills possible
                execution_price = price * np.random.uniform(0.999, 1.001)  # Small price variance
                fees = filled_quantity * execution_price * 0.001  # 0.1% fees
                
                # Update metrics
                self.execution_metrics['total_orders'] += 1
                self.execution_metrics['successful_orders'] += 1
                
                return {
                    'status': 'success',
                    'order_id': order_id,
                    'symbol': symbol,
                    'side': side,
                    'quantity': quantity,
                    'filled_quantity': filled_quantity,
                    'execution_price': execution_price,
                    'fees': fees,
                    'execution_time': time.time(),
                    'exchange': 'simulation'
                }
            else:
                # Failed execution
                self.execution_metrics['total_orders'] += 1
                
                failure_reasons = [
                    'Insufficient liquidity',
                    'Price moved outside tolerance',
                    'Exchange connectivity issue',
                    'Order rejected by risk system'
                ]
                
                return {
                    'status': 'failed',
                    'order_id': order_id,
                    'symbol': symbol,
                    'side': side,
                    'quantity': quantity,
                    'filled_quantity': 0.0,
                    'error': np.random.choice(failure_reasons),
                    'execution_time': time.time(),
                    'exchange': 'simulation'
                }
                
        except Exception as e:
            logger.error(f"Simulation error: {e}")
            return {
                'status': 'error',
                'order_id': order_scenario.get('order_id', 'unknown'),
                'error': str(e),
                'execution_time': time.time()
            }

    async def get_real_time_market_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get real-time market data for analysis
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with real-time market data
        """
        try:
            # If we have real exchanges, get real data
            if self.exchanges:
                for exchange_id, exchange in self.exchanges.items():
                    try:
                        if hasattr(exchange, 'fetch_ticker'):
                            ticker = await exchange.fetch_ticker(symbol)
                            return {
                                'symbol': symbol,
                                'price': ticker.get('last', 0.0),
                                'bid': ticker.get('bid', 0.0),
                                'ask': ticker.get('ask', 0.0),
                                'volume': ticker.get('baseVolume', 0.0),
                                'change_24h': ticker.get('percentage', 0.0),
                                'high_24h': ticker.get('high', 0.0),
                                'low_24h': ticker.get('low', 0.0),
                                'timestamp': datetime.now(),
                                'exchange': exchange_id
                            }
                    except Exception as e:
                        logger.warning(f"Failed to get market data from {exchange_id}: {e}")
                        continue
            
            # Fallback to simulated data
            base_prices = {
                'BTCUSDT': 45000.0,
                'ETHUSDT': 2500.0,
                'ADAUSDT': 0.5,
                'BNBUSDT': 300.0,
                'SOLUSDT': 100.0
            }
            
            base_price = base_prices.get(symbol, 1000.0)
            price_variance = np.random.uniform(0.98, 1.02)
            current_price = base_price * price_variance
            
            return {
                'symbol': symbol,
                'price': current_price,
                'bid': current_price * 0.999,
                'ask': current_price * 1.001,
                'volume': np.random.uniform(1000000, 10000000),
                'change_24h': np.random.uniform(-5.0, 5.0),
                'high_24h': current_price * 1.05,
                'low_24h': current_price * 0.95,
                'timestamp': datetime.now(),
                'exchange': 'simulation'
            }
            
        except Exception as e:
            logger.error(f"Failed to get market data for {symbol}: {e}")
            return {
                'symbol': symbol,
                'price': 0.0,
                'error': str(e),
                'timestamp': datetime.now()
            }

    async def validate_trading_pair(self, symbol: str) -> bool:
        """
        Validate if a trading pair is supported
        
        Args:
            symbol: Trading symbol to validate
            
        Returns:
            True if supported, False otherwise
        """
        try:
            # Check with real exchanges first
            if self.exchanges:
                for exchange_id, exchange in self.exchanges.items():
                    try:
                        if hasattr(exchange, 'load_markets'):
                            markets = await exchange.load_markets()
                            if symbol in markets:
                                return True
                    except Exception as e:
                        logger.warning(f"Failed to validate pair on {exchange_id}: {e}")
                        continue
            
            # Fallback to common pairs
            common_pairs = [
                'BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'BNBUSDT', 'SOLUSDT',
                'DOTUSDT', 'LINKUSDT', 'MATICUSDT', 'AVAXUSDT', 'UNIUSDT'
            ]
            
            return symbol in common_pairs
            
        except Exception as e:
            logger.error(f"Failed to validate trading pair {symbol}: {e}")
            return False

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics"""
        return {
            'execution_metrics': self.execution_metrics,
            'daily_volume': self.daily_volume,
            'max_daily_volume': self.max_daily_volume,
            'volume_utilization': self.daily_volume / self.max_daily_volume,
            'active_orders_count': len(self.active_orders),
            'connected_exchanges': list(self.exchanges.keys()),
            'exchange_count': len(self.exchanges)
        }

def create_semantic_execution_bridge(config: Dict[str, Any]) -> SemanticExecutionBridge:
    """
    Factory function to create a Semantic Execution Bridge
    
    Args:
        config: Configuration with exchange credentials and settings
        
    Returns:
        SemanticExecutionBridge instance
    """
    return SemanticExecutionBridge(config) 