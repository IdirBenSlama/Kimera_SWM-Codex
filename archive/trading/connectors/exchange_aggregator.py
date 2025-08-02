"""
Exchange Aggregator for Kimera Ultimate Trading System

Unified interface to multiple cryptocurrency exchanges with:
- Liquidity aggregation across exchanges
- Smart order routing
- Best execution analysis
- Cross-exchange arbitrage detection
- Real-time market data consolidation
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from collections import defaultdict

# Exchange connectors
from src.trading.api.binance_connector import BinanceConnector

logger = logging.getLogger(__name__)

@dataclass
class ExchangeOrderBook:
    """Order book data from an exchange"""
    exchange: str
    symbol: str
    bids: List[Tuple[float, float]]  # [(price, quantity), ...]
    asks: List[Tuple[float, float]]  # [(price, quantity), ...]
    timestamp: datetime
    latency_ms: float

@dataclass
class BestExecutionVenue:
    """Best execution venue analysis"""
    exchange: str
    symbol: str
    side: str  # 'buy' or 'sell'
    price: float
    available_quantity: float
    total_cost: float
    expected_slippage: float
    execution_quality_score: float
    reasoning: str

@dataclass
class ArbitrageOpportunity:
    """Cross-exchange arbitrage opportunity"""
    buy_exchange: str
    sell_exchange: str
    symbol: str
    buy_price: float
    sell_price: float
    profit_percent: float
    max_quantity: float
    estimated_profit: float
    risk_score: float

class ExchangeAggregator:
    """
    Unified interface to multiple cryptocurrency exchanges
    with intelligent order routing and liquidity aggregation
    """
    
    def __init__(self):
        # Initialize exchange connectors
        self.exchanges = {}
        
        # Try to initialize Binance connector with demo credentials
        try:
            self.exchanges['binance'] = BinanceConnector('demo_key', 'demo_secret')
        except Exception as e:
            logger.warning(f"Failed to initialize Binance connector: {e}")
            # Create a mock connector for demo purposes
            self.exchanges['binance'] = self.create_mock_connector('binance')
        
        # Exchange status tracking
        self.exchange_status = {}
        self.connection_latencies = defaultdict(list)
        self.order_books = {}
        self.last_update_times = {}
        
        # Performance metrics
        self.total_requests = 0
        self.successful_requests = 0
        self.arbitrage_opportunities_found = 0
        
        logger.info(f"🔗 Exchange Aggregator initialized with {len(self.exchanges)} exchanges")
    
    async def connect_all_exchanges(self):
        """Connect to all configured exchanges"""
        logger.info("🔌 Connecting to all exchanges...")
        
        connection_tasks = []
        for exchange_name, exchange in self.exchanges.items():
            task = self.connect_exchange(exchange_name, exchange)
            connection_tasks.append(task)
        
        # Connect to all exchanges concurrently
        results = await asyncio.gather(*connection_tasks, return_exceptions=True)
        
        # Process results
        connected_count = 0
        for i, (exchange_name, result) in enumerate(zip(self.exchanges.keys(), results)):
            if isinstance(result, Exception):
                logger.error(f"❌ Failed to connect to {exchange_name}: {result}")
                self.exchange_status[exchange_name] = 'disconnected'
            else:
                logger.info(f"✅ Connected to {exchange_name}")
                self.exchange_status[exchange_name] = 'connected'
                connected_count += 1
        
        logger.info(f"📊 Connected to {connected_count}/{len(self.exchanges)} exchanges")
        return connected_count > 0
    
    async def connect_exchange(self, exchange_name: str, exchange) -> bool:
        """Connect to a specific exchange"""
        try:
            start_time = time.time()
            
            # Test connection with a simple API call
            if hasattr(exchange, 'get_ticker'):
                test_ticker = await exchange.get_ticker('BTCUSDT')
                if test_ticker:
                    connection_time = (time.time() - start_time) * 1000
                    self.connection_latencies[exchange_name].append(connection_time)
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Connection failed for {exchange_name}: {e}")
            return False
    
    async def get_consolidated_order_book(self, symbol: str) -> Dict[str, ExchangeOrderBook]:
        """Get order books from all connected exchanges"""
        order_book_tasks = []
        
        for exchange_name, exchange in self.exchanges.items():
            if self.exchange_status.get(exchange_name) == 'connected':
                task = self.get_exchange_order_book(exchange_name, exchange, symbol)
                order_book_tasks.append(task)
        
        if not order_book_tasks:
            logger.warning(f"⚠️ No connected exchanges for {symbol}")
            return {}
        
        # Fetch order books concurrently
        results = await asyncio.gather(*order_book_tasks, return_exceptions=True)
        
        consolidated_books = {}
        for result in results:
            if isinstance(result, ExchangeOrderBook):
                consolidated_books[result.exchange] = result
            elif isinstance(result, Exception):
                logger.warning(f"Order book fetch failed: {result}")
        
        self.order_books[symbol] = consolidated_books
        self.last_update_times[symbol] = datetime.now()
        
        return consolidated_books
    
    async def get_exchange_order_book(self, exchange_name: str, exchange, symbol: str) -> ExchangeOrderBook:
        """Get order book from specific exchange"""
        try:
            start_time = time.time()
            
            # Get order book data (simplified - in real implementation, get full depth)
            ticker = await exchange.get_ticker(symbol)
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Simulate order book structure (in real implementation, get actual order book)
            bid_price = float(ticker.get('bid', ticker.get('price', 0)))
            ask_price = float(ticker.get('ask', ticker.get('price', 0)))
            
            # Create simplified order book
            bids = [(bid_price, 1.0)] if bid_price > 0 else []
            asks = [(ask_price, 1.0)] if ask_price > 0 else []
            
            return ExchangeOrderBook(
                exchange=exchange_name,
                symbol=symbol,
                bids=bids,
                asks=asks,
                timestamp=datetime.now(),
                latency_ms=latency_ms
            )
            
        except Exception as e:
            logger.error(f"Failed to get order book from {exchange_name}: {e}")
            raise
    
    async def find_best_execution_venue(self, symbol: str, side: str, quantity: float) -> Optional[BestExecutionVenue]:
        """Find optimal exchange for order execution"""
        # Get current order books
        order_books = await self.get_consolidated_order_book(symbol)
        
        if not order_books:
            logger.warning(f"⚠️ No order book data available for {symbol}")
            return None
        
        best_venues = []
        
        for exchange_name, order_book in order_books.items():
            venue = self.analyze_execution_venue(order_book, side, quantity)
            if venue:
                best_venues.append(venue)
        
        if not best_venues:
            return None
        
        # Sort by execution quality score (higher is better)
        best_venues.sort(key=lambda x: x.execution_quality_score, reverse=True)
        
        return best_venues[0]
    
    def analyze_execution_venue(self, order_book: ExchangeOrderBook, side: str, quantity: float) -> Optional[BestExecutionVenue]:
        """Analyze execution quality for a specific venue"""
        try:
            if side == 'buy':
                # Analyze asks for buying
                orders = order_book.asks
                price_key = 0  # Price is first element
                sort_ascending = True  # Best ask is lowest price
            else:
                # Analyze bids for selling
                orders = order_book.bids
                price_key = 0  # Price is first element
                sort_ascending = False  # Best bid is highest price
            
            if not orders:
                return None
            
            # Sort orders by price
            sorted_orders = sorted(orders, key=lambda x: x[price_key], reverse=not sort_ascending)
            
            # Calculate execution metrics
            total_cost = 0
            remaining_quantity = quantity
            weighted_avg_price = 0
            
            for price, available_qty in sorted_orders:
                if remaining_quantity <= 0:
                    break
                
                execute_qty = min(remaining_quantity, available_qty)
                total_cost += price * execute_qty
                weighted_avg_price += price * execute_qty
                remaining_quantity -= execute_qty
            
            if quantity > remaining_quantity:
                # Partial fill possible
                filled_quantity = quantity - remaining_quantity
                avg_price = weighted_avg_price / (filled_quantity * quantity) if filled_quantity > 0 else 0
                
                # Calculate execution quality score
                execution_quality_score = self.calculate_execution_quality(
                    order_book, avg_price, filled_quantity, quantity
                )
                
                # Estimate slippage
                best_price = sorted_orders[0][0] if sorted_orders else 0
                expected_slippage = abs(avg_price - best_price) / best_price if best_price > 0 else 0
                
                return BestExecutionVenue(
                    exchange=order_book.exchange,
                    symbol=order_book.symbol,
                    side=side,
                    price=avg_price,
                    available_quantity=filled_quantity,
                    total_cost=total_cost,
                    expected_slippage=expected_slippage,
                    execution_quality_score=execution_quality_score,
                    reasoning=f"Can fill {filled_quantity:.6f}/{quantity:.6f} at avg price {avg_price:.2f}"
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Execution analysis failed for {order_book.exchange}: {e}")
            return None
    
    def calculate_execution_quality(self, order_book: ExchangeOrderBook, avg_price: float, 
                                  filled_quantity: float, requested_quantity: float) -> float:
        """Calculate execution quality score (0-1, higher is better)"""
        try:
            # Factors affecting execution quality:
            # 1. Fill ratio (how much of the order can be filled)
            fill_ratio = filled_quantity / requested_quantity
            
            # 2. Latency (lower is better)
            latency_score = max(0, 1 - (order_book.latency_ms / 1000))  # Normalize to 1 second
            
            # 3. Spread quality (tighter spreads are better)
            if order_book.bids and order_book.asks:
                best_bid = max(order_book.bids, key=lambda x: x[0])[0]
                best_ask = min(order_book.asks, key=lambda x: x[0])[0]
                spread = (best_ask - best_bid) / best_bid if best_bid > 0 else 1
                spread_score = max(0, 1 - spread * 100)  # Normalize spread
            else:
                spread_score = 0.5
            
            # 4. Liquidity depth
            total_liquidity = sum(qty for _, qty in order_book.bids + order_book.asks)
            liquidity_score = min(1.0, total_liquidity / (requested_quantity * 10))
            
            # Combined execution quality score
            quality_score = (
                fill_ratio * 0.4 +           # 40% weight on fill ratio
                latency_score * 0.2 +        # 20% weight on latency
                spread_score * 0.2 +         # 20% weight on spread
                liquidity_score * 0.2        # 20% weight on liquidity
            )
            
            return min(1.0, max(0.0, quality_score))
            
        except Exception as e:
            logger.error(f"Quality calculation failed: {e}")
            return 0.0
    
    async def detect_arbitrage_opportunities(self, symbol: str, min_profit_percent: float = 0.1) -> List[ArbitrageOpportunity]:
        """Detect cross-exchange arbitrage opportunities"""
        order_books = await self.get_consolidated_order_book(symbol)
        
        if len(order_books) < 2:
            return []  # Need at least 2 exchanges for arbitrage
        
        opportunities = []
        exchanges = list(order_books.keys())
        
        # Compare all exchange pairs
        for i in range(len(exchanges)):
            for j in range(i + 1, len(exchanges)):
                exchange_a = exchanges[i]
                exchange_b = exchanges[j]
                
                book_a = order_books[exchange_a]
                book_b = order_books[exchange_b]
                
                # Check both directions
                opp_a_to_b = self.analyze_arbitrage_pair(book_a, book_b, symbol, min_profit_percent)
                opp_b_to_a = self.analyze_arbitrage_pair(book_b, book_a, symbol, min_profit_percent)
                
                if opp_a_to_b:
                    opportunities.append(opp_a_to_b)
                if opp_b_to_a:
                    opportunities.append(opp_b_to_a)
        
        # Sort by profit potential
        opportunities.sort(key=lambda x: x.profit_percent, reverse=True)
        
        if opportunities:
            self.arbitrage_opportunities_found += len(opportunities)
            logger.info(f"🎯 Found {len(opportunities)} arbitrage opportunities for {symbol}")
        
        return opportunities
    
    def analyze_arbitrage_pair(self, buy_book: ExchangeOrderBook, sell_book: ExchangeOrderBook, 
                              symbol: str, min_profit_percent: float) -> Optional[ArbitrageOpportunity]:
        """Analyze arbitrage opportunity between two exchanges"""
        try:
            if not buy_book.asks or not sell_book.bids:
                return None
            
            # Best price to buy from buy_exchange
            best_ask = min(buy_book.asks, key=lambda x: x[0])
            buy_price = best_ask[0]
            max_buy_quantity = best_ask[1]
            
            # Best price to sell on sell_exchange
            best_bid = max(sell_book.bids, key=lambda x: x[0])
            sell_price = best_bid[0]
            max_sell_quantity = best_bid[1]
            
            # Calculate profit
            if sell_price <= buy_price:
                return None  # No profit opportunity
            
            profit_percent = ((sell_price - buy_price) / buy_price) * 100
            
            if profit_percent < min_profit_percent:
                return None  # Profit too small
            
            # Maximum quantity for arbitrage
            max_quantity = min(max_buy_quantity, max_sell_quantity)
            estimated_profit = (sell_price - buy_price) * max_quantity
            
            # Calculate risk score (higher latency = higher risk)
            avg_latency = (buy_book.latency_ms + sell_book.latency_ms) / 2
            risk_score = min(1.0, avg_latency / 1000)  # Normalize to 1 second
            
            return ArbitrageOpportunity(
                buy_exchange=buy_book.exchange,
                sell_exchange=sell_book.exchange,
                symbol=symbol,
                buy_price=buy_price,
                sell_price=sell_price,
                profit_percent=profit_percent,
                max_quantity=max_quantity,
                estimated_profit=estimated_profit,
                risk_score=risk_score
            )
            
        except Exception as e:
            logger.error(f"Arbitrage analysis failed: {e}")
            return None
    
    async def get_best_price(self, symbol: str, side: str) -> Dict[str, Any]:
        """Get best price across all exchanges"""
        order_books = await self.get_consolidated_order_book(symbol)
        
        if not order_books:
            return {'error': 'No order book data available'}
        
        best_prices = {}
        
        for exchange_name, order_book in order_books.items():
            try:
                if side == 'buy' and order_book.asks:
                    # Best ask price for buying
                    best_ask = min(order_book.asks, key=lambda x: x[0])
                    best_prices[exchange_name] = {
                        'price': best_ask[0],
                        'quantity': best_ask[1],
                        'side': 'ask'
                    }
                elif side == 'sell' and order_book.bids:
                    # Best bid price for selling
                    best_bid = max(order_book.bids, key=lambda x: x[0])
                    best_prices[exchange_name] = {
                        'price': best_bid[0],
                        'quantity': best_bid[1],
                        'side': 'bid'
                    }
            except Exception as e:
                logger.warning(f"Price extraction failed for {exchange_name}: {e}")
        
        if not best_prices:
            return {'error': 'No valid prices found'}
        
        # Find overall best price
        if side == 'buy':
            # Lowest ask price is best for buying
            best_exchange = min(best_prices.items(), key=lambda x: x[1]['price'])
        else:
            # Highest bid price is best for selling
            best_exchange = max(best_prices.items(), key=lambda x: x[1]['price'])
        
        return {
            'best_exchange': best_exchange[0],
            'best_price': best_exchange[1]['price'],
            'available_quantity': best_exchange[1]['quantity'],
            'all_prices': best_prices,
            'symbol': symbol,
            'side': side,
            'timestamp': datetime.now()
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get aggregator performance metrics"""
        connected_exchanges = sum(1 for status in self.exchange_status.values() if status == 'connected')
        
        avg_latencies = {}
        for exchange, latencies in self.connection_latencies.items():
            if latencies:
                avg_latencies[exchange] = sum(latencies) / len(latencies)
        
        success_rate = 0
        if self.total_requests > 0:
            success_rate = self.successful_requests / self.total_requests
        
        return {
            'total_exchanges': len(self.exchanges),
            'connected_exchanges': connected_exchanges,
            'exchange_status': dict(self.exchange_status),
            'average_latencies_ms': avg_latencies,
            'total_requests': self.total_requests,
            'success_rate': success_rate,
            'arbitrage_opportunities_found': self.arbitrage_opportunities_found,
            'symbols_tracked': len(self.order_books)
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all exchanges"""
        health_status = {}
        
        for exchange_name, exchange in self.exchanges.items():
            try:
                start_time = time.time()
                test_ticker = await exchange.get_ticker('BTCUSDT')
                response_time = (time.time() - start_time) * 1000
                
                health_status[exchange_name] = {
                    'status': 'healthy',
                    'response_time_ms': response_time,
                    'last_check': datetime.now()
                }
                
            except Exception as e:
                health_status[exchange_name] = {
                    'status': 'unhealthy',
                    'error': str(e),
                    'last_check': datetime.now()
                }
        
        return health_status
    
    def create_mock_connector(self, exchange_name: str):
        """Create a mock exchange connector for demo purposes"""
        class MockConnector:
            def __init__(self, name):
                self.name = name
            
            async def get_ticker(self, symbol):
                """Mock ticker data"""
                import random
                base_price = {'BTCUSDT': 45000, 'ETHUSDT': 2500, 'ADAUSDT': 0.5, 'SOLUSDT': 100}.get(symbol, 1000)
                variation = random.uniform(-0.05, 0.05)  # ±5% variation
                price = base_price * (1 + variation)
                
                return {
                    'symbol': symbol,
                    'price': price,
                    'bid': price * 0.999,
                    'ask': price * 1.001,
                    'volume': random.uniform(1000000, 10000000)
                }
        
        return MockConnector(exchange_name)

# Factory function
def create_exchange_aggregator() -> ExchangeAggregator:
    """Create and return exchange aggregator instance"""
    return ExchangeAggregator() 