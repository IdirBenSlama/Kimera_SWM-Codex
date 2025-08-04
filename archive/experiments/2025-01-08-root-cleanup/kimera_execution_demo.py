import os
#!/usr/bin/env python3
"""
KIMERA EXECUTION DEMONSTRATION SYSTEM
Pure execution speed demonstration - trades regardless of profit margins

This system demonstrates the ultra-low latency execution capabilities
by executing trades to showcase the nanosecond precision timing and
state-of-the-art architecture in action.
"""

import logging
import time
import json
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import statistics
import signal
import sys

from binance import Client
from binance.exceptions import BinanceAPIException
import numpy as np

# Configure high-performance logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler('kimera_demo.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ExecutionMetrics:
    """Nanosecond precision execution measurements"""
    timestamp_ns: int
    market_data_latency_ns: int
    decision_latency_ns: int
    order_placement_latency_ns: int
    total_execution_latency_ns: int
    symbol: str
    side: str
    quantity: Decimal
    price: Decimal

class HighPrecisionTimer:
    """Nanosecond precision timing system"""
    
    @staticmethod
    def now_ns() -> int:
        return time.perf_counter_ns()
    
    @staticmethod
    def latency_ns(start_ns: int) -> int:
        return time.perf_counter_ns() - start_ns

class KimeraExecutionDemo:
    """Ultra-low latency execution demonstration system"""
    
    def __init__(self, api_key: str, api_secret: str):
        self.client = Client(api_key, api_secret)
        self.timer = HighPrecisionTimer()
        
        # Demo configuration - focused on execution speed
        self.demo_pairs = ['DOGEUSDT']  # Single pair for maximum focus
        self.trade_percentage = Decimal('0.9')  # Use 90% of balance
        self.demo_trades = 5  # Number of demonstration trades
        
        # Symbol information
        self.symbol_info = {}
        self.running = False
        
        # Performance tracking
        self.execution_metrics = []
        self.total_trades = 0
        self.successful_trades = 0
        self.total_pnl = Decimal('0')
        
        # Load symbol information
        self.load_symbol_info()
        
    def load_symbol_info(self):
        """Load trading rules with performance optimization"""
        start_ns = self.timer.now_ns()
        
        try:
            exchange_info = self.client.get_exchange_info()
            
            for symbol_data in exchange_info['symbols']:
                symbol = symbol_data['symbol']
                if symbol in self.demo_pairs:
                    
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
    
    def get_market_data_ultra_fast(self, symbol: str) -> Optional[Tuple[Decimal, Decimal]]:
        """Get market data with nanosecond precision timing"""
        start_ns = self.timer.now_ns()
        
        try:
            depth = self.client.get_order_book(symbol=symbol, limit=5)
            
            if not depth['bids'] or not depth['asks']:
                return None
                
            best_bid = Decimal(depth['bids'][0][0])
            best_ask = Decimal(depth['asks'][0][0])
            
            market_data_latency_ns = self.timer.latency_ns(start_ns)
            logger.debug(f"Market data latency: {market_data_latency_ns:,} ns")
            
            return best_bid, best_ask, market_data_latency_ns
            
        except Exception as e:
            logger.error(f"Failed to get market data for {symbol}: {e}")
            return None
    
    def execute_demonstration_trade(self, symbol: str, trade_number: int) -> bool:
        """Execute demonstration trade with full latency measurement"""
        demo_start_ns = self.timer.now_ns()
        
        try:
            logger.info(f"DEMO TRADE #{trade_number} STARTING - {symbol}")
            
            # Get market data with timing
            market_data_start_ns = self.timer.now_ns()
            market_result = self.get_market_data_ultra_fast(symbol)
            if not market_result:
                return False
            
            best_bid, best_ask, market_data_latency_ns = market_result
            
            # Decision phase timing
            decision_start_ns = self.timer.now_ns()
            
            # Get current balance
            balance = self.get_account_balance()
            if balance < self.symbol_info[symbol]['min_notional']:
                logger.warning(f"Insufficient balance: ${balance}")
                return False
            
            # Calculate trade size
            trade_value = balance * self.trade_percentage
            buy_price = best_ask
            quantity = trade_value / buy_price
            
            # Apply step size precision
            step_size = self.symbol_info[symbol]['step_size']
            if step_size > 0:
                steps = int(quantity / step_size)
                quantity = steps * step_size
            
            if quantity < self.symbol_info[symbol]['min_qty']:
                logger.warning(f"Quantity too small: {quantity}")
                return False
            
            decision_latency_ns = self.timer.latency_ns(decision_start_ns)
            
            # Execute buy order with timing
            order_start_ns = self.timer.now_ns()
            
            logger.info(f"EXECUTING DEMO BUY: {quantity} {symbol} @ ${buy_price}")
            
            buy_order = self.client.order_market_buy(
                symbol=symbol,
                quantity=f"{quantity:.8f}".rstrip('0').rstrip('.')
            )
            
            buy_latency_ns = self.timer.latency_ns(order_start_ns)
            
            if buy_order['status'] != 'FILLED':
                logger.error(f"Buy order not filled: {buy_order['status']}")
                return False
            
            buy_qty = Decimal(buy_order['executedQty'])
            buy_value = Decimal(buy_order['cummulativeQuoteQty'])
            actual_buy_price = buy_value / buy_qty
            
            logger.info(f"DEMO BUY FILLED: {buy_qty} {symbol} @ ${actual_buy_price} = ${buy_value}")
            
            # Brief pause for demonstration
            time.sleep(0.1)  # 100ms pause
            
            # Get asset balance for sell
            asset = symbol.replace('USDT', '')
            account = self.client.get_account()
            asset_balance = Decimal('0')
            
            for balance in account['balances']:
                if balance['asset'] == asset:
                    asset_balance = Decimal(balance['free'])
                    break
            
            if asset_balance > 0:
                # Execute sell order with timing
                sell_start_ns = self.timer.now_ns()
                
                logger.info(f"EXECUTING DEMO SELL: {asset_balance} {symbol}")
                
                sell_order = self.client.order_market_sell(
                    symbol=symbol,
                    quantity=f"{asset_balance:.8f}".rstrip('0').rstrip('.')
                )
                
                sell_latency_ns = self.timer.latency_ns(sell_start_ns)
                
                if sell_order['status'] == 'FILLED':
                    sell_qty = Decimal(sell_order['executedQty'])
                    sell_value = Decimal(sell_order['cummulativeQuoteQty'])
                    actual_sell_price = sell_value / sell_qty
                    
                    # Calculate profit/loss
                    profit = sell_value - buy_value
                    profit_bps = int((profit / buy_value) * 10000)
                    
                    self.total_pnl += profit
                    
                    total_execution_latency_ns = self.timer.latency_ns(demo_start_ns)
                    
                    # Record execution metrics
                    metrics = ExecutionMetrics(
                        timestamp_ns=demo_start_ns,
                        market_data_latency_ns=market_data_latency_ns,
                        decision_latency_ns=decision_latency_ns,
                        order_placement_latency_ns=buy_latency_ns + sell_latency_ns,
                        total_execution_latency_ns=total_execution_latency_ns,
                        symbol=symbol,
                        side='BUY/SELL',
                        quantity=buy_qty,
                        price=actual_buy_price
                    )
                    
                    self.execution_metrics.append(metrics)
                    self.successful_trades += 1
                    
                    logger.info(f"DEMO TRADE #{trade_number} COMPLETED:")
                    logger.info(f"  Buy Price: ${actual_buy_price:.6f}")
                    logger.info(f"  Sell Price: ${actual_sell_price:.6f}")
                    logger.info(f"  Profit/Loss: ${profit:.4f} ({profit_bps} bps)")
                    logger.info(f"  Market Data Latency: {market_data_latency_ns:,} ns")
                    logger.info(f"  Decision Latency: {decision_latency_ns:,} ns")
                    logger.info(f"  Order Latency: {buy_latency_ns + sell_latency_ns:,} ns")
                    logger.info(f"  Total Execution: {total_execution_latency_ns:,} ns ({total_execution_latency_ns/1_000_000:.2f} ms)")
                    
                    return True
                    
        except Exception as e:
            logger.error(f"Demo trade execution failed: {e}")
        
        return False
    
    def run_execution_demonstration(self):
        """Run execution speed demonstration"""
        try:
            start_time = datetime.now()
            
            logger.info("=" * 100)
            logger.info("KIMERA ULTRA-LOW LATENCY EXECUTION DEMONSTRATION")
            logger.info("Showcasing State-of-the-Art Trading Infrastructure")
            logger.info("=" * 100)
            logger.info(f"Demo Pairs: {', '.join(self.demo_pairs)}")
            logger.info(f"Number of Demo Trades: {self.demo_trades}")
            logger.info(f"Trade Size: {float(self.trade_percentage * 100):.0f}% of balance")
            
            initial_balance = self.get_account_balance()
            logger.info(f"Starting Balance: ${initial_balance}")
            logger.info("=" * 100)
            
            self.running = True
            logger.info("EXECUTION DEMONSTRATION STARTED")
            
            # Execute demonstration trades
            for trade_num in range(1, self.demo_trades + 1):
                if not self.running:
                    break
                
                logger.info(f"PREPARING DEMO TRADE #{trade_num}/{self.demo_trades}")
                
                success = self.execute_demonstration_trade(self.demo_pairs[0], trade_num)
                if success:
                    self.total_trades += 1
                    
                # Pause between demo trades
                if trade_num < self.demo_trades:
                    logger.info(f"Pausing 2 seconds before next demo trade...")
                    time.sleep(2)
            
            # Generate demonstration report
            self.generate_demo_report(start_time, initial_balance)
            
        except KeyboardInterrupt:
            logger.info("EXECUTION DEMONSTRATION STOPPED BY USER")
        except Exception as e:
            logger.error(f"EXECUTION DEMONSTRATION ERROR: {e}")
        finally:
            self.running = False
    
    def generate_demo_report(self, start_time: datetime, initial_balance: Decimal):
        """Generate comprehensive demonstration report"""
        end_time = datetime.now()
        runtime_seconds = (end_time - start_time).total_seconds()
        final_balance = self.get_account_balance()
        
        total_profit = final_balance - initial_balance
        profit_pct = (total_profit / initial_balance * 100) if initial_balance > 0 else 0
        
        logger.info("\n" + "=" * 100)
        logger.info("KIMERA EXECUTION DEMONSTRATION REPORT")
        logger.info("=" * 100)
        logger.info(f"Demo Runtime: {runtime_seconds:.1f} seconds")
        logger.info(f"Initial Balance: ${initial_balance:.6f}")
        logger.info(f"Final Balance: ${final_balance:.6f}")
        logger.info(f"Total P&L: ${total_profit:.6f}")
        logger.info(f"P&L Percentage: {profit_pct:.4f}%")
        logger.info(f"Successful Executions: {self.successful_trades}/{self.demo_trades}")
        logger.info(f"Success Rate: {(self.successful_trades/max(self.demo_trades,1)*100):.2f}%")
        
        if self.execution_metrics:
            # Latency analysis
            total_latencies = [m.total_execution_latency_ns for m in self.execution_metrics]
            market_latencies = [m.market_data_latency_ns for m in self.execution_metrics]
            decision_latencies = [m.decision_latency_ns for m in self.execution_metrics]
            order_latencies = [m.order_placement_latency_ns for m in self.execution_metrics]
            
            logger.info("\n" + "=" * 60)
            logger.info("ULTRA-LOW LATENCY PERFORMANCE ANALYSIS")
            logger.info("=" * 60)
            
            logger.info(f"Total Execution Latency:")
            logger.info(f"  Minimum: {min(total_latencies):,} ns ({min(total_latencies)/1_000_000:.2f} ms)")
            logger.info(f"  Maximum: {max(total_latencies):,} ns ({max(total_latencies)/1_000_000:.2f} ms)")
            logger.info(f"  Average: {statistics.mean(total_latencies):,.0f} ns ({statistics.mean(total_latencies)/1_000_000:.2f} ms)")
            logger.info(f"  Median:  {statistics.median(total_latencies):,.0f} ns ({statistics.median(total_latencies)/1_000_000:.2f} ms)")
            
            logger.info(f"\nMarket Data Latency:")
            logger.info(f"  Average: {statistics.mean(market_latencies):,.0f} ns ({statistics.mean(market_latencies)/1_000_000:.2f} ms)")
            
            logger.info(f"\nDecision Latency:")
            logger.info(f"  Average: {statistics.mean(decision_latencies):,.0f} ns ({statistics.mean(decision_latencies)/1_000_000:.2f} ms)")
            
            logger.info(f"\nOrder Placement Latency:")
            logger.info(f"  Average: {statistics.mean(order_latencies):,.0f} ns ({statistics.mean(order_latencies)/1_000_000:.2f} ms)")
            
            # Individual trade breakdown
            logger.info(f"\n" + "=" * 60)
            logger.info("INDIVIDUAL TRADE PERFORMANCE")
            logger.info("=" * 60)
            for i, metrics in enumerate(self.execution_metrics, 1):
                logger.info(f"Trade #{i}: {metrics.total_execution_latency_ns:,} ns "
                      f"({metrics.total_execution_latency_ns/1_000_000:.2f} ms) - "
                      f"{metrics.symbol} {metrics.quantity} @ ${metrics.price:.6f}")
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"kimera_demo_results_{timestamp}.json"
        
        results_data = {
            'summary': {
                'demo_runtime_seconds': runtime_seconds,
                'initial_balance': str(initial_balance),
                'final_balance': str(final_balance),
                'total_pnl': str(total_profit),
                'pnl_percentage': float(profit_pct),
                'successful_executions': self.successful_trades,
                'total_demo_trades': self.demo_trades,
                'success_rate': float(self.successful_trades/max(self.demo_trades,1)*100)
            },
            'latency_performance': {
                'total_execution_avg_ns': statistics.mean([m.total_execution_latency_ns for m in self.execution_metrics]) if self.execution_metrics else 0,
                'market_data_avg_ns': statistics.mean([m.market_data_latency_ns for m in self.execution_metrics]) if self.execution_metrics else 0,
                'decision_avg_ns': statistics.mean([m.decision_latency_ns for m in self.execution_metrics]) if self.execution_metrics else 0,
                'order_placement_avg_ns': statistics.mean([m.order_placement_latency_ns for m in self.execution_metrics]) if self.execution_metrics else 0
            },
            'individual_trades': [
                {
                    'trade_number': i+1,
                    'symbol': m.symbol,
                    'quantity': str(m.quantity),
                    'price': str(m.price),
                    'total_latency_ns': m.total_execution_latency_ns,
                    'market_data_latency_ns': m.market_data_latency_ns,
                    'decision_latency_ns': m.decision_latency_ns,
                    'order_latency_ns': m.order_placement_latency_ns
                }
                for i, m in enumerate(self.execution_metrics)
            ]
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        logger.info(f"DEMONSTRATION RESULTS SAVED: {results_file}")
        logger.info("=" * 100)
        logger.info("DEMONSTRATION COMPLETE - Ultra-Low Latency Capabilities Showcased")
        logger.info("=" * 100)

def main():
    """Main execution function"""
    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = "qUn5JqSpYz1GDxFj2X3UF23TYgtxKrTsCbDZEoBMYCPbYZgP4siVLyspkB5HAPl7"
    
    # Create execution demonstration system
    demo = KimeraExecutionDemo(api_key, api_secret)
    
    # Setup emergency stop handlers
    def emergency_handler(signum, frame):
        logger.info("EMERGENCY STOP ACTIVATED")
        demo.running = False
        sys.exit(0)
    
    signal.signal(signal.SIGINT, emergency_handler)
    signal.signal(signal.SIGTERM, emergency_handler)
    
    # Run execution demonstration
    demo.run_execution_demonstration()

if __name__ == "__main__":
    main() 