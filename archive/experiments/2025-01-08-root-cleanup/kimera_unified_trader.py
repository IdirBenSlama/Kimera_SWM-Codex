import os
#!/usr/bin/env python3
"""
KIMERA UNIFIED TRADING SYSTEM
State-of-the-Art Multi-Strategy Trading Platform

This unified module combines all Kimera trading capabilities:
- Ultra-Low Latency Execution
- Smart Order Routing
- Multiple Trading Strategies
- Real-Time Risk Management
- Nanosecond Precision Timing
- Advanced Market Analysis
- Performance Analytics
"""

import logging
import time
import json
import asyncio
import threading
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN
from collections import deque, defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import statistics
import signal
import sys
import argparse

from binance import Client, ThreadedWebsocketManager
from binance.exceptions import BinanceAPIException
import numpy as np

# Configure high-performance logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler('kimera_unified.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Trading Modes
class TradingMode(Enum):
    """Available trading modes"""
    ULTRA_LOW_LATENCY = "ultra_low_latency"
    AGGRESSIVE = "aggressive"
    CONSERVATIVE = "conservative"
    SCALPING = "scalping"
    MARKET_MAKING = "market_making"
    DEMONSTRATION = "demonstration"
    MICRO = "micro"

# Execution Strategies
class ExecutionStrategy(Enum):
    """Order execution strategies"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    ICEBERG = "ICEBERG"
    TWAP = "TWAP"
    VWAP = "VWAP"
    SMART = "SMART"

# Risk Levels
class RiskLevel(Enum):
    """Risk management levels"""
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"
    CRITICAL = "CRITICAL"

@dataclass
class LatencyMetrics:
    """Nanosecond precision latency measurements"""
    timestamp_ns: int
    market_data_latency_ns: int
    decision_latency_ns: int
    execution_latency_ns: int
    total_latency_ns: int
    
@dataclass
class MarketData:
    """Real-time market data structure"""
    symbol: str
    timestamp_ns: int
    best_bid: Decimal
    best_ask: Decimal
    bid_qty: Decimal
    ask_qty: Decimal
    spread: Decimal
    spread_bps: int
    mid_price: Decimal
    volume_imbalance: float
    momentum_score: float
    
@dataclass
class TradingSignal:
    """Trading signal with confidence score"""
    symbol: str
    timestamp_ns: int
    action: str  # BUY/SELL
    confidence: float
    strategy: ExecutionStrategy
    target_price: Decimal
    stop_loss: Decimal
    take_profit: Decimal
    
@dataclass
class OrderExecution:
    """Order execution record"""
    timestamp_ns: int
    symbol: str
    side: str
    strategy: ExecutionStrategy
    quantity: Decimal
    price: Decimal
    value: Decimal
    latency_metrics: LatencyMetrics
    order_id: str
    success: bool
    pnl: Optional[Decimal] = None

class HighPrecisionTimer:
    """Nanosecond precision timing system"""
    
    @staticmethod
    def now_ns() -> int:
        """Get current time in nanoseconds"""
        return time.perf_counter_ns()
    
    @staticmethod
    def now_us() -> float:
        """Get current time in microseconds"""
        return time.perf_counter_ns() / 1000.0
    
    @staticmethod
    def latency_ns(start_ns: int) -> int:
        """Calculate latency in nanoseconds"""
        return time.perf_counter_ns() - start_ns

class UnifiedRiskManager:
    """Comprehensive risk management system"""
    
    def __init__(self, mode: TradingMode):
        self.mode = mode
        self.configure_risk_parameters(mode)
        self.daily_pnl = Decimal('0')
        self.current_positions = defaultdict(Decimal)
        self.risk_level = RiskLevel.GREEN
        self.circuit_breaker_active = False
        self.trade_count = 0
        self.last_trade_time = 0
        
    def configure_risk_parameters(self, mode: TradingMode):
        """Configure risk parameters based on trading mode"""
        if mode == TradingMode.AGGRESSIVE:
            self.position_limits = {
                'max_position_usd': Decimal('8.0'),
                'max_daily_loss': Decimal('0.50'),
                'max_trade_size': Decimal('5.0'),  # Increased to allow full balance usage
                'max_trades_per_minute': 20
            }
        elif mode == TradingMode.CONSERVATIVE:
            self.position_limits = {
                'max_position_usd': Decimal('5.0'),
                'max_daily_loss': Decimal('0.20'),
                'max_trade_size': Decimal('2.0'),
                'max_trades_per_minute': 5
            }
        elif mode == TradingMode.MICRO:
            self.position_limits = {
                'max_position_usd': Decimal('5.0'),
                'max_daily_loss': Decimal('0.10'),
                'max_trade_size': Decimal('5.0'),
                'max_trades_per_minute': 10
            }
        else:  # Default/Ultra-low latency
            self.position_limits = {
                'max_position_usd': Decimal('10.0'),
                'max_daily_loss': Decimal('1.0'),
                'max_trade_size': Decimal('5.0'),
                'max_trades_per_minute': 15
            }
    
    def check_pre_trade_risk(self, symbol: str, side: str, value: Decimal) -> bool:
        """Comprehensive pre-trade risk assessment"""
        current_time = time.time()
        
        # Rate limiting
        if self.mode != TradingMode.DEMONSTRATION:
            if current_time - self.last_trade_time < 60 / self.position_limits['max_trades_per_minute']:
                logger.warning("RISK: Rate limit exceeded")
                return False
        
        # Position size check
        if value > self.position_limits['max_trade_size']:
            logger.warning(f"RISK: Trade size ${value} exceeds limit ${self.position_limits['max_trade_size']}")
            return False
            
        # Daily loss check
        if self.daily_pnl < -self.position_limits['max_daily_loss']:
            logger.warning(f"RISK: Daily loss ${abs(self.daily_pnl)} exceeds limit")
            self.circuit_breaker_active = True
            return False
            
        # Circuit breaker check
        if self.circuit_breaker_active:
            logger.warning("RISK: Circuit breaker active")
            return False
            
        return True
    
    def update_pnl(self, pnl: Decimal):
        """Update daily P&L and risk level"""
        self.daily_pnl += pnl
        self.trade_count += 1
        self.last_trade_time = time.time()
        
        # Update risk level based on P&L
        loss_pct = abs(self.daily_pnl) / self.position_limits['max_daily_loss'] if self.daily_pnl < 0 else 0
        if loss_pct > 0.9:
            self.risk_level = RiskLevel.CRITICAL
        elif loss_pct > 0.7:
            self.risk_level = RiskLevel.RED
        elif loss_pct > 0.5:
            self.risk_level = RiskLevel.YELLOW
        else:
            self.risk_level = RiskLevel.GREEN

class MarketAnalyzer:
    """Advanced market analysis and signal generation"""
    
    def __init__(self):
        self.order_books = {}
        self.price_history = defaultdict(lambda: deque(maxlen=100))
        self.volume_history = defaultdict(lambda: deque(maxlen=50))
        self.spread_history = defaultdict(lambda: deque(maxlen=50))
        self.signal_history = defaultdict(lambda: deque(maxlen=20))
        
    def analyze_market_data(self, symbol: str, depth_data: dict) -> Optional[MarketData]:
        """Comprehensive market data analysis"""
        analysis_start_ns = HighPrecisionTimer.now_ns()
        
        try:
            bids = depth_data.get('bids', [])
            asks = depth_data.get('asks', [])
            
            if not bids or not asks:
                return None
                
            best_bid = Decimal(bids[0][0])
            best_ask = Decimal(asks[0][0])
            bid_qty = Decimal(bids[0][1])
            ask_qty = Decimal(asks[0][1])
            
            # Calculate spread metrics
            spread = best_ask - best_bid
            spread_bps = int((spread / best_bid) * 10000)
            mid_price = (best_bid + best_ask) / 2
            
            # Volume imbalance analysis
            total_bid_vol = sum(Decimal(bid[1]) for bid in bids[:5])
            total_ask_vol = sum(Decimal(ask[1]) for ask in asks[:5])
            volume_imbalance = float((total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol))
            
            # Price momentum analysis
            self.price_history[symbol].append(float(mid_price))
            momentum_score = 0.0
            if len(self.price_history[symbol]) >= 10:
                prices = list(self.price_history[symbol])
                short_term_avg = sum(prices[-5:]) / 5
                long_term_avg = sum(prices[-10:]) / 10
                momentum_score = (short_term_avg - long_term_avg) / long_term_avg * 10000  # bps
            
            # Store historical data
            self.spread_history[symbol].append(spread_bps)
            self.volume_history[symbol].append(volume_imbalance)
            
            market_data = MarketData(
                symbol=symbol,
                timestamp_ns=analysis_start_ns,
                best_bid=best_bid,
                best_ask=best_ask,
                bid_qty=bid_qty,
                ask_qty=ask_qty,
                spread=spread,
                spread_bps=spread_bps,
                mid_price=mid_price,
                volume_imbalance=volume_imbalance,
                momentum_score=momentum_score
            )
            
            self.order_books[symbol] = market_data
            return market_data
            
        except Exception as e:
            logger.error(f"Market analysis failed for {symbol}: {e}")
            return None
    
    def generate_trading_signal(self, market_data: MarketData, mode: TradingMode) -> Optional[TradingSignal]:
        """Generate trading signals based on market conditions and mode"""
        try:
            confidence = 0.0
            action = None
            strategy = ExecutionStrategy.MARKET
            
            # Mode-specific signal generation
            if mode == TradingMode.AGGRESSIVE:
                # Aggressive scalping signals
                if market_data.spread_bps <= 5 and abs(market_data.volume_imbalance) > 0.2:
                    confidence = 0.8
                    action = "BUY" if market_data.volume_imbalance > 0 else "SELL"
                    strategy = ExecutionStrategy.MARKET
                elif abs(market_data.momentum_score) > 10:
                    confidence = 0.7
                    action = "BUY" if market_data.momentum_score > 0 else "SELL"
                    strategy = ExecutionStrategy.LIMIT
                    
            elif mode == TradingMode.CONSERVATIVE:
                # Conservative signals with higher thresholds
                if market_data.spread_bps <= 3 and abs(market_data.momentum_score) > 5:
                    confidence = 0.6
                    action = "BUY" if market_data.momentum_score > 0 else "SELL"
                    strategy = ExecutionStrategy.LIMIT
                    
            elif mode == TradingMode.SCALPING:
                # Pure scalping on tight spreads
                if market_data.spread_bps <= 10:
                    confidence = 0.5 + (10 - market_data.spread_bps) / 20
                    action = "BUY"  # Always buy first in scalping
                    strategy = ExecutionStrategy.MARKET
                    
            elif mode == TradingMode.MARKET_MAKING:
                # Market making on both sides
                if market_data.spread_bps >= 5:
                    confidence = 0.7
                    action = "BUY"  # Place both buy and sell limits
                    strategy = ExecutionStrategy.LIMIT
                    
            else:  # Default ultra-low latency
                # Balanced approach
                if market_data.spread_bps <= 8 and abs(market_data.volume_imbalance) > 0.15:
                    confidence = 0.65
                    action = "BUY" if market_data.volume_imbalance > 0 else "SELL"
                    strategy = ExecutionStrategy.SMART
            
            if action and confidence > 0.5:
                # Calculate target prices
                if action == "BUY":
                    target_price = market_data.best_ask * Decimal('1.001')  # 0.1% profit target
                    stop_loss = market_data.best_ask * Decimal('0.998')    # 0.2% stop loss
                else:
                    target_price = market_data.best_bid * Decimal('0.999')  # 0.1% profit target
                    stop_loss = market_data.best_bid * Decimal('1.002')     # 0.2% stop loss
                
                take_profit = target_price
                
                signal = TradingSignal(
                    symbol=market_data.symbol,
                    timestamp_ns=HighPrecisionTimer.now_ns(),
                    action=action,
                    confidence=confidence,
                    strategy=strategy,
                    target_price=target_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
                
                self.signal_history[market_data.symbol].append(signal)
                return signal
                
        except Exception as e:
            logger.error(f"Signal generation failed: {e}")
            
        return None 

class SmartOrderRouter:
    """AI-enhanced Smart Order Routing system"""
    
    def __init__(self, client: Client):
        self.client = client
        self.execution_history = deque(maxlen=1000)
        self.strategy_performance = defaultdict(list)
        
    def select_optimal_strategy(self, signal: TradingSignal, market_data: MarketData, 
                              balance: Decimal) -> ExecutionStrategy:
        """AI-enhanced strategy selection based on market conditions"""
        
        # Override with signal strategy if specified
        if signal.strategy != ExecutionStrategy.SMART:
            return signal.strategy
        
        # Smart strategy selection based on multiple factors
        spread_bps = market_data.spread_bps
        volume_imbalance = abs(market_data.volume_imbalance)
        momentum = abs(market_data.momentum_score)
        
        # Ultra-tight spread - use market orders
        if spread_bps <= 2:
            return ExecutionStrategy.MARKET
        
        # Tight spread with volume imbalance - use limit orders
        elif spread_bps <= 5 and volume_imbalance > 0.2:
            return ExecutionStrategy.LIMIT
        
        # Moderate spread with momentum - use iceberg
        elif spread_bps <= 10 and momentum > 10:
            return ExecutionStrategy.ICEBERG
        
        # Wide spread - use TWAP
        elif spread_bps > 15:
            return ExecutionStrategy.TWAP
        
        # Default to market orders
        else:
            return ExecutionStrategy.MARKET
    
    def calculate_optimal_price(self, market_data: MarketData, side: str, 
                              strategy: ExecutionStrategy) -> Decimal:
        """Calculate optimal execution price based on strategy"""
        
        if strategy == ExecutionStrategy.MARKET:
            return market_data.best_ask if side == 'BUY' else market_data.best_bid
            
        elif strategy == ExecutionStrategy.LIMIT:
            # Aggressive limit pricing to ensure execution
            if side == 'BUY':
                return market_data.best_bid + (market_data.spread * Decimal('0.3'))
            else:
                return market_data.best_ask - (market_data.spread * Decimal('0.3'))
                
        elif strategy == ExecutionStrategy.ICEBERG:
            # Use mid-price for iceberg orders
            return market_data.mid_price
            
        elif strategy == ExecutionStrategy.TWAP:
            # Start at mid-price for TWAP
            return market_data.mid_price
            
        else:
            return market_data.mid_price
    
    def record_execution(self, execution: OrderExecution):
        """Record execution for performance analysis"""
        self.execution_history.append(execution)
        self.strategy_performance[execution.strategy].append(execution)

class PerformanceAnalyzer:
    """Comprehensive performance analytics system"""
    
    def __init__(self):
        self.latency_history = deque(maxlen=10000)
        self.execution_history = deque(maxlen=1000)
        self.pnl_history = deque(maxlen=1000)
        self.performance_metrics = {}
        
    def record_latency(self, metrics: LatencyMetrics):
        """Record latency measurements"""
        self.latency_history.append(metrics)
        
    def record_execution(self, execution: OrderExecution):
        """Record execution details"""
        self.execution_history.append(execution)
        if execution.pnl is not None:
            self.pnl_history.append(execution.pnl)
        
    def calculate_performance_stats(self) -> dict:
        """Calculate comprehensive performance statistics"""
        stats = {}
        
        # Latency statistics
        if self.latency_history:
            total_latencies = [m.total_latency_ns for m in self.latency_history]
            market_data_latencies = [m.market_data_latency_ns for m in self.latency_history]
            execution_latencies = [m.execution_latency_ns for m in self.latency_history]
            
            stats['latency'] = {
                'total': {
                    'min_ns': min(total_latencies),
                    'max_ns': max(total_latencies),
                    'mean_ns': statistics.mean(total_latencies),
                    'median_ns': statistics.median(total_latencies),
                    'p95_ns': np.percentile(total_latencies, 95),
                    'p99_ns': np.percentile(total_latencies, 99)
                },
                'market_data': {
                    'mean_ns': statistics.mean(market_data_latencies),
                    'median_ns': statistics.median(market_data_latencies)
                },
                'execution': {
                    'mean_ns': statistics.mean(execution_latencies),
                    'median_ns': statistics.median(execution_latencies)
                }
            }
        
        # Trading performance
        if self.execution_history:
            successful_trades = [e for e in self.execution_history if e.success]
            stats['trading'] = {
                'total_trades': len(self.execution_history),
                'successful_trades': len(successful_trades),
                'success_rate': len(successful_trades) / len(self.execution_history) * 100,
                'trades_per_minute': len(self.execution_history) / max(1, (self.execution_history[-1].timestamp_ns - self.execution_history[0].timestamp_ns) / 60_000_000_000)
            }
        
        # P&L statistics
        if self.pnl_history:
            pnl_list = list(self.pnl_history)
            winning_trades = [p for p in pnl_list if p > 0]
            losing_trades = [p for p in pnl_list if p < 0]
            
            stats['pnl'] = {
                'total_pnl': sum(pnl_list),
                'average_pnl': statistics.mean(pnl_list),
                'win_rate': len(winning_trades) / len(pnl_list) * 100,
                'average_win': statistics.mean(winning_trades) if winning_trades else 0,
                'average_loss': statistics.mean(losing_trades) if losing_trades else 0,
                'profit_factor': abs(sum(winning_trades) / sum(losing_trades)) if losing_trades else float('inf')
            }
        
        return stats
    
    def generate_report(self) -> str:
        """Generate human-readable performance report"""
        stats = self.calculate_performance_stats()
        
        report = "\n" + "=" * 80 + "\n"
        report += "KIMERA PERFORMANCE REPORT\n"
        report += "=" * 80 + "\n"
        
        if 'latency' in stats:
            report += "\nLATENCY PERFORMANCE:\n"
            report += f"  Average Total: {stats['latency']['total']['mean_ns']:,.0f} ns ({stats['latency']['total']['mean_ns']/1_000_000:.2f} ms)\n"
            report += f"  Minimum: {stats['latency']['total']['min_ns']:,.0f} ns\n"
            report += f"  Maximum: {stats['latency']['total']['max_ns']:,.0f} ns\n"
            report += f"  95th Percentile: {stats['latency']['total']['p95_ns']:,.0f} ns\n"
        
        if 'trading' in stats:
            report += "\nTRADING PERFORMANCE:\n"
            report += f"  Total Trades: {stats['trading']['total_trades']}\n"
            report += f"  Success Rate: {stats['trading']['success_rate']:.2f}%\n"
            report += f"  Trades/Minute: {stats['trading']['trades_per_minute']:.2f}\n"
        
        if 'pnl' in stats:
            report += "\nP&L PERFORMANCE:\n"
            report += f"  Total P&L: ${stats['pnl']['total_pnl']:.4f}\n"
            report += f"  Win Rate: {stats['pnl']['win_rate']:.2f}%\n"
            report += f"  Profit Factor: {stats['pnl']['profit_factor']:.2f}\n"
        
        report += "=" * 80 + "\n"
        return report

class KimeraUnifiedTrader:
    """Unified state-of-the-art trading system"""
    
    def __init__(self, api_key: str, api_secret: str, mode: TradingMode = TradingMode.ULTRA_LOW_LATENCY):
        self.client = Client(api_key, api_secret)
        self.mode = mode
        self.timer = HighPrecisionTimer()
        self.risk_manager = UnifiedRiskManager(mode)
        self.market_analyzer = MarketAnalyzer()
        self.order_router = SmartOrderRouter(self.client)
        self.performance_analyzer = PerformanceAnalyzer()
        
        # Trading configuration based on mode
        self.configure_trading_parameters(mode)
        
        # Symbol information
        self.symbol_info = {}
        self.running = False
        
        # Performance tracking
        self.total_trades = 0
        self.successful_trades = 0
        self.total_pnl = Decimal('0')
        
        # Load symbol information
        self.load_symbol_info()
        
    def configure_trading_parameters(self, mode: TradingMode):
        """Configure trading parameters based on mode"""
        if mode == TradingMode.AGGRESSIVE:
            self.trading_pairs = ['DOGEUSDT', 'TRXUSDT', 'SHIBUSDT', 'PEPEUSDT']
            self.min_trade_percentage = Decimal('0.95')
            self.target_profit_bps = 2  # 0.02%
            self.max_spread_bps = 20
            
        elif mode == TradingMode.CONSERVATIVE:
            self.trading_pairs = ['BTCUSDT', 'ETHUSDT']
            self.min_trade_percentage = Decimal('0.5')
            self.target_profit_bps = 10  # 0.1%
            self.max_spread_bps = 5
            
        elif mode == TradingMode.SCALPING:
            self.trading_pairs = ['DOGEUSDT', 'TRXUSDT']
            self.min_trade_percentage = Decimal('0.9')
            self.target_profit_bps = 1  # 0.01%
            self.max_spread_bps = 30
            
        elif mode == TradingMode.MICRO:
            self.trading_pairs = ['DOGEUSDT', 'TRXUSDT']
            self.min_trade_percentage = Decimal('0.8')
            self.target_profit_bps = 5  # 0.05%
            self.max_spread_bps = 50
            
        elif mode == TradingMode.DEMONSTRATION:
            self.trading_pairs = ['DOGEUSDT']
            self.min_trade_percentage = Decimal('0.9')
            self.target_profit_bps = 0  # No profit requirement
            self.max_spread_bps = 100
            
        else:  # Default ULTRA_LOW_LATENCY
            self.trading_pairs = ['TRXUSDT', 'DOGEUSDT', 'ADAUSDT', 'XRPUSDT']
            self.min_trade_percentage = Decimal('0.8')
            self.target_profit_bps = 10  # 0.1%
            self.max_spread_bps = 15
    
    def load_symbol_info(self):
        """Load trading rules with performance optimization"""
        start_ns = self.timer.now_ns()
        
        try:
            exchange_info = self.client.get_exchange_info()
            
            for symbol_data in exchange_info['symbols']:
                symbol = symbol_data['symbol']
                if symbol in self.trading_pairs:
                    
                    filters = {}
                    for f in symbol_data['filters']:
                        filters[f['filterType']] = f
                    
                    self.symbol_info[symbol] = {
                        'min_qty': Decimal(filters.get('LOT_SIZE', {}).get('minQty', '0')),
                        'step_size': Decimal(filters.get('LOT_SIZE', {}).get('stepSize', '0')),
                        'min_notional': Decimal(filters.get('NOTIONAL', {}).get('minNotional', '0')),
                        'tick_size': Decimal(filters.get('PRICE_FILTER', {}).get('tickSize', '0'))
                    }
            
            load_latency_ns = self.timer.latency_ns(start_ns)
            logger.info(f"Symbol info loaded in {load_latency_ns:,} ns ({load_latency_ns/1_000_000:.2f} ms)")
            logger.info(f"Trading mode: {self.mode.value}")
            logger.info(f"Trading pairs: {', '.join(self.trading_pairs)}")
            
            for symbol, info in self.symbol_info.items():
                logger.info(f"{symbol}: Min Notional ${info['min_notional']}")
            
        except Exception as e:
            logger.error(f"Failed to load symbol info: {e}")
    
    def get_account_balance(self) -> Decimal:
        """Get USDT balance with latency measurement"""
        start_ns = self.timer.now_ns()
        
        try:
            account = self.client.get_account()
            for balance in account['balances']:
                if balance['asset'] == 'USDT':
                    usdt_balance = Decimal(balance['free'])
                    latency_ns = self.timer.latency_ns(start_ns)
                    logger.debug(f"Balance query latency: {latency_ns:,} ns")
                    return usdt_balance
            return Decimal('0')
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return Decimal('0') 
    
    def scan_for_opportunities(self) -> List[Tuple[MarketData, Optional[TradingSignal]]]:
        """Scan all trading pairs for opportunities"""
        opportunities = []
        
        for symbol in self.trading_pairs:
            try:
                # Get market data
                depth = self.client.get_order_book(symbol=symbol, limit=10)
                market_data = self.market_analyzer.analyze_market_data(symbol, depth)
                
                if market_data:
                    # Generate trading signal
                    signal = self.market_analyzer.generate_trading_signal(market_data, self.mode)
                    
                    # Filter by spread requirements
                    if market_data.spread_bps <= self.max_spread_bps:
                        opportunities.append((market_data, signal))
                        
            except Exception as e:
                logger.error(f"Failed to scan {symbol}: {e}")
        
        # Sort by opportunity score (signal confidence if available)
        opportunities.sort(key=lambda x: x[1].confidence if x[1] else 0, reverse=True)
        return opportunities
    
    def execute_trade(self, symbol: str, side: str, quantity: Decimal, 
                     strategy: ExecutionStrategy) -> Optional[OrderExecution]:
        """Execute trade with nanosecond precision tracking"""
        execution_start_ns = self.timer.now_ns()
        
        try:
            # Execute order based on strategy
            if strategy in [ExecutionStrategy.MARKET, ExecutionStrategy.SMART]:
                if side == 'BUY':
                    order = self.client.order_market_buy(
                        symbol=symbol,
                        quantity=f"{quantity:.8f}".rstrip('0').rstrip('.')
                    )
                else:
                    order = self.client.order_market_sell(
                        symbol=symbol,
                        quantity=f"{quantity:.8f}".rstrip('0').rstrip('.')
                    )
            else:
                # For now, fallback to market orders for other strategies
                # In production, implement limit, iceberg, TWAP, VWAP
                if side == 'BUY':
                    order = self.client.order_market_buy(
                        symbol=symbol,
                        quantity=f"{quantity:.8f}".rstrip('0').rstrip('.')
                    )
                else:
                    order = self.client.order_market_sell(
                        symbol=symbol,
                        quantity=f"{quantity:.8f}".rstrip('0').rstrip('.')
                    )
            
            execution_latency_ns = self.timer.latency_ns(execution_start_ns)
            
            if order['status'] == 'FILLED':
                executed_qty = Decimal(order['executedQty'])
                executed_value = Decimal(order['cummulativeQuoteQty'])
                avg_price = executed_value / executed_qty
                
                # Create latency metrics
                latency_metrics = LatencyMetrics(
                    timestamp_ns=execution_start_ns,
                    market_data_latency_ns=0,  # Set by caller
                    decision_latency_ns=0,     # Set by caller
                    execution_latency_ns=execution_latency_ns,
                    total_latency_ns=execution_latency_ns
                )
                
                execution_record = OrderExecution(
                    timestamp_ns=execution_start_ns,
                    symbol=symbol,
                    side=side,
                    strategy=strategy,
                    quantity=executed_qty,
                    price=avg_price,
                    value=executed_value,
                    latency_metrics=latency_metrics,
                    order_id=order['orderId'],
                    success=True
                )
                
                logger.info(f"ORDER EXECUTED: {side} {executed_qty} {symbol} @ ${avg_price} "
                          f"[{execution_latency_ns:,} ns] Strategy: {strategy.value}")
                
                return execution_record
                
        except Exception as e:
            logger.error(f"Order execution failed: {e}")
            
        return None
    
    def execute_trading_cycle(self, market_data: MarketData, signal: Optional[TradingSignal]) -> bool:
        """Execute complete trading cycle with full tracking"""
        cycle_start_ns = self.timer.now_ns()
        
        try:
            symbol = market_data.symbol
            
            # Check if we should trade
            if not signal or signal.confidence < 0.5:
                return False
            
            # Get current balance
            balance = self.get_account_balance()
            if balance < self.symbol_info[symbol]['min_notional']:
                logger.warning(f"Insufficient balance: ${balance}")
                return False
            
            # Calculate trade size
            trade_value = balance * self.min_trade_percentage
            
            # Risk check
            if not self.risk_manager.check_pre_trade_risk(symbol, signal.action, trade_value):
                return False
            
            # Select optimal strategy
            strategy = self.order_router.select_optimal_strategy(signal, market_data, balance)
            
            # Calculate quantity
            price = self.order_router.calculate_optimal_price(market_data, signal.action, strategy)
            quantity = trade_value / price
            
            # Apply step size precision
            step_size = self.symbol_info[symbol]['step_size']
            if step_size > 0:
                steps = int(quantity / step_size)
                quantity = steps * step_size
            
            if quantity < self.symbol_info[symbol]['min_qty']:
                logger.warning(f"Quantity too small: {quantity}")
                return False
            
            # Measure decision latency
            decision_latency_ns = self.timer.latency_ns(cycle_start_ns)
            
            # Execute initial order (buy for scalping)
            initial_execution = self.execute_trade(symbol, signal.action, quantity, strategy)
            if not initial_execution:
                return False
            
            # Update latency metrics
            initial_execution.latency_metrics.decision_latency_ns = decision_latency_ns
            
            # For scalping/demo modes, execute opposite side immediately
            if self.mode in [TradingMode.SCALPING, TradingMode.DEMONSTRATION, TradingMode.AGGRESSIVE]:
                # Brief pause
                time.sleep(0.05)  # 50ms
                
                # Get asset balance for opposite side
                if signal.action == 'BUY':
                    asset = symbol.replace('USDT', '')
                    account = self.client.get_account()
                    asset_balance = Decimal('0')
                    
                    for balance in account['balances']:
                        if balance['asset'] == asset:
                            asset_balance = Decimal(balance['free'])
                            break
                    
                    if asset_balance > 0:
                        # Execute sell
                        sell_execution = self.execute_trade(symbol, 'SELL', asset_balance, strategy)
                        if sell_execution:
                            # Calculate P&L
                            pnl = sell_execution.value - initial_execution.value
                            sell_execution.pnl = pnl
                            
                            # Update metrics
                            self.risk_manager.update_pnl(pnl)
                            self.total_pnl += pnl
                            self.successful_trades += 1
                            
                            # Record executions
                            self.order_router.record_execution(initial_execution)
                            self.order_router.record_execution(sell_execution)
                            self.performance_analyzer.record_execution(initial_execution)
                            self.performance_analyzer.record_execution(sell_execution)
                            
                            # Calculate total cycle latency
                            total_latency_ns = self.timer.latency_ns(cycle_start_ns)
                            
                            # Record complete latency metrics
                            complete_metrics = LatencyMetrics(
                                timestamp_ns=cycle_start_ns,
                                market_data_latency_ns=0,  # Would be set if we tracked it
                                decision_latency_ns=decision_latency_ns,
                                execution_latency_ns=initial_execution.latency_metrics.execution_latency_ns + sell_execution.latency_metrics.execution_latency_ns,
                                total_latency_ns=total_latency_ns
                            )
                            self.performance_analyzer.record_latency(complete_metrics)
                            
                            profit_bps = int((pnl / initial_execution.value) * 10000)
                            logger.info(f"CYCLE COMPLETED: {symbol} | P&L: ${pnl:.4f} ({profit_bps} bps) | "
                                      f"Total Latency: {total_latency_ns:,} ns")
                            
                            return True
            else:
                # For other modes, just record the single execution
                self.order_router.record_execution(initial_execution)
                self.performance_analyzer.record_execution(initial_execution)
                self.performance_analyzer.record_latency(initial_execution.latency_metrics)
                self.successful_trades += 1
                return True
                
        except Exception as e:
            logger.error(f"Trading cycle failed: {e}")
        
        self.total_trades += 1
        return False
    
    def run_trading_session(self, runtime_minutes: int = 15):
        """Run unified trading session"""
        try:
            start_time = datetime.now()
            end_time = start_time + timedelta(minutes=runtime_minutes)
            
            logger.info("=" * 100)
            logger.info("KIMERA UNIFIED TRADING SYSTEM")
            logger.info(f"Mode: {self.mode.value.upper()}")
            logger.info("=" * 100)
            logger.info(f"Runtime: {runtime_minutes} minutes")
            logger.info(f"Trading Pairs: {', '.join(self.trading_pairs)}")
            logger.info(f"Target Profit: {self.target_profit_bps} bps")
            logger.info(f"Max Spread: {self.max_spread_bps} bps")
            logger.info(f"Risk Level: {self.risk_manager.risk_level.value}")
            
            initial_balance = self.get_account_balance()
            logger.info(f"Starting Balance: ${initial_balance}")
            logger.info("=" * 100)
            
            self.running = True
            logger.info(f"TRADING SESSION STARTED - Mode: {self.mode.value}")
            
            cycle_count = 0
            status_interval = 20 if self.mode == TradingMode.AGGRESSIVE else 10
            
            while self.running and datetime.now() < end_time:
                cycle_start_ns = self.timer.now_ns()
                cycle_count += 1
                
                if self.risk_manager.circuit_breaker_active:
                    logger.warning("CIRCUIT BREAKER ACTIVE - TRADING HALTED")
                    break
                
                # Scan for opportunities
                opportunities = self.scan_for_opportunities()
                
                # Execute best opportunities
                max_trades_per_cycle = 3 if self.mode == TradingMode.AGGRESSIVE else 1
                for market_data, signal in opportunities[:max_trades_per_cycle]:
                    if not self.running:
                        break
                    
                    success = self.execute_trading_cycle(market_data, signal)
                    if success:
                        # Brief pause between trades
                        time.sleep(0.02 if self.mode == TradingMode.AGGRESSIVE else 0.1)
                
                # Status update
                if cycle_count % status_interval == 0:
                    remaining_minutes = (end_time - datetime.now()).total_seconds() / 60
                    current_balance = self.get_account_balance()
                    
                    # Get performance stats
                    perf_stats = self.performance_analyzer.calculate_performance_stats()
                    avg_latency_us = 0
                    if 'latency' in perf_stats and 'total' in perf_stats['latency']:
                        avg_latency_us = perf_stats['latency']['total'].get('mean_ns', 0) / 1000
                    
                    logger.info(f"STATUS - Cycle: {cycle_count} | "
                              f"Trades: {self.successful_trades}/{self.total_trades} | "
                              f"Balance: ${current_balance:.4f} | "
                              f"P&L: ${self.total_pnl:.4f} | "
                              f"Avg Latency: {avg_latency_us:.1f} μs | "
                              f"Risk: {self.risk_manager.risk_level.value} | "
                              f"Time: {remaining_minutes:.1f}min")
                
                # Cycle pause based on mode
                if self.mode == TradingMode.AGGRESSIVE:
                    time.sleep(0.05)  # 50ms
                elif self.mode == TradingMode.SCALPING:
                    time.sleep(0.1)   # 100ms
                else:
                    time.sleep(0.2)   # 200ms
            
            # Generate final report
            self.generate_final_report(start_time, initial_balance)
            
        except KeyboardInterrupt:
            logger.info("TRADING SESSION STOPPED BY USER")
        except Exception as e:
            logger.error(f"TRADING SESSION ERROR: {e}")
        finally:
            self.running = False
    
    def generate_final_report(self, start_time: datetime, initial_balance: Decimal):
        """Generate comprehensive final report"""
        end_time = datetime.now()
        runtime_seconds = (end_time - start_time).total_seconds()
        final_balance = self.get_account_balance()
        
        total_profit = final_balance - initial_balance
        profit_pct = (total_profit / initial_balance * 100) if initial_balance > 0 else 0
        
        # Print summary report
        logger.info("\n" + "=" * 100)
        logger.info(f"KIMERA UNIFIED TRADING SYSTEM - FINAL REPORT")
        logger.info(f"Trading Mode: {self.mode.value.upper()}")
        logger.info("=" * 100)
        logger.info(f"Runtime: {runtime_seconds:.1f} seconds")
        logger.info(f"Initial Balance: ${initial_balance:.6f}")
        logger.info(f"Final Balance: ${final_balance:.6f}")
        logger.info(f"Total Profit: ${total_profit:.6f}")
        logger.info(f"Profit Percentage: {profit_pct:.4f}%")
        logger.info(f"Successful Trades: {self.successful_trades}")
        logger.info(f"Total Trade Attempts: {self.total_trades}")
        logger.info(f"Success Rate: {(self.successful_trades/max(self.total_trades,1)*100):.2f}%")
        logger.info(f"Final Risk Level: {self.risk_manager.risk_level.value}")
        
        # Print detailed performance report
        logger.info(self.performance_analyzer.generate_report())
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"kimera_unified_results_{self.mode.value}_{timestamp}.json"
        
        perf_stats = self.performance_analyzer.calculate_performance_stats()
        
        results_data = {
            'summary': {
                'mode': self.mode.value,
                'runtime_seconds': runtime_seconds,
                'initial_balance': str(initial_balance),
                'final_balance': str(final_balance),
                'total_profit': str(total_profit),
                'profit_percentage': float(profit_pct),
                'successful_trades': self.successful_trades,
                'total_trades': self.total_trades,
                'success_rate': float(self.successful_trades/max(self.total_trades,1)*100),
                'final_risk_level': self.risk_manager.risk_level.value
            },
            'configuration': {
                'trading_pairs': self.trading_pairs,
                'target_profit_bps': self.target_profit_bps,
                'max_spread_bps': self.max_spread_bps,
                'min_trade_percentage': str(self.min_trade_percentage)
            },
            'performance_metrics': perf_stats
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        logger.info(f"RESULTS SAVED: {results_file}")
        logger.info("=" * 100)

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Kimera Unified Trading System')
    parser.add_argument('--mode', type=str, default='ultra_low_latency',
                       choices=['ultra_low_latency', 'aggressive', 'conservative', 
                               'scalping', 'market_making', 'demonstration', 'micro'],
                       help='Trading mode to use')
    parser.add_argument('--runtime', type=int, default=15,
                       help='Runtime in minutes (default: 15)')
    parser.add_argument('--api-key', type=str, 
                       default=os.getenv("BINANCE_API_KEY", ""),
                       help='Binance API key')
    parser.add_argument('--api-secret', type=str,
                       default="qUn5JqSpYz1GDxFj2X3UF23TYgtxKrTsCbDZEoBMYCPbYZgP4siVLyspkB5HAPl7",
                       help='Binance API secret')
    
    args = parser.parse_args()
    
    # Convert mode string to enum
    mode_map = {
        'ultra_low_latency': TradingMode.ULTRA_LOW_LATENCY,
        'aggressive': TradingMode.AGGRESSIVE,
        'conservative': TradingMode.CONSERVATIVE,
        'scalping': TradingMode.SCALPING,
        'market_making': TradingMode.MARKET_MAKING,
        'demonstration': TradingMode.DEMONSTRATION,
        'micro': TradingMode.MICRO
    }
    
    trading_mode = mode_map.get(args.mode, TradingMode.ULTRA_LOW_LATENCY)
    
    # Create unified trader
    trader = KimeraUnifiedTrader(args.api_key, args.api_secret, trading_mode)
    
    # Setup emergency stop handlers
    def emergency_handler(signum, frame):
        logger.info("EMERGENCY STOP ACTIVATED")
        trader.running = False
        sys.exit(0)
    
    signal.signal(signal.SIGINT, emergency_handler)
    signal.signal(signal.SIGTERM, emergency_handler)
    
    # Run trading session
    trader.run_trading_session(runtime_minutes=args.runtime)

if __name__ == "__main__":
    main() 