import os
#!/usr/bin/env python3
"""
KIMERA ULTRA-AGGRESSIVE 5-MINUTE PROFIT MAXIMIZER
Target: Maximum profit extraction from $25 allocation in 5 minutes
Mode: ULTRA-AGGRESSIVE PERFORMANCE - NO LIMITS
"""

import asyncio
import logging
import time
import json
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_UP, ROUND_DOWN
from binance import Client
from binance.exceptions import BinanceAPIException
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import signal
import sys

# Configure ultra-performance logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler('kimera_ultra_aggressive.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class KimeraUltraAggressive:
    def __init__(self, api_key, api_secret, allocation_usd=25, runtime_minutes=5):
        self.client = Client(api_key, api_secret)
        self.allocation_usd = Decimal(str(allocation_usd))
        self.runtime_minutes = runtime_minutes
        self.start_time = None
        self.end_time = None
        
        # Ultra-aggressive parameters
        self.min_profit_threshold = Decimal('0.0005')  # 0.05% minimum profit
        self.max_positions = 6  # Multiple simultaneous positions
        self.execution_interval = 0.2  # 200ms execution cycles
        self.price_update_interval = 0.1  # 100ms price updates
        
        # Market data storage
        self.price_data = {}
        self.order_book_data = {}
        self.active_positions = {}
        self.completed_trades = []
        self.price_queue = queue.Queue()
        
        # Performance tracking
        self.total_profit = Decimal('0')
        self.total_trades = 0
        self.successful_trades = 0
        self.trade_frequency = 0
        
        # Trading pairs for ultra-aggressive scanning
        self.ultra_pairs = [
            'TRXUSDT', 'ADAUSDT', 'DOGEUSDT', 'SHIBUSDT', 'PEPEUSDT',
            'XRPUSDT', 'SOLUSDT', 'MATICUSDT', 'DOTUSDT', 'AVAXUSDT'
        ]
        
        # Risk management
        self.max_loss_per_trade = self.allocation_usd * Decimal('0.02')  # 2% max loss
        self.position_size_per_trade = self.allocation_usd / Decimal(str(self.max_positions))
        
        self.running = False
        
    def setup_signal_handlers(self):
        """Setup graceful shutdown handlers"""
        def signal_handler(signum, frame):
            logger.info("🛑 EMERGENCY STOP SIGNAL RECEIVED")
            self.emergency_stop()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def get_account_balances(self):
        """Get current account balances"""
        try:
            account = self.client.get_account()
            balances = {}
            
            for balance in account['balances']:
                asset = balance['asset']
                free = Decimal(balance['free'])
                locked = Decimal(balance['locked'])
                
                if free > 0 or locked > 0:
                    balances[asset] = {
                        'free': free,
                        'locked': locked,
                        'total': free + locked
                    }
            
            return balances
        except Exception as e:
            logger.error(f"Error getting balances: {e}")
            return {}
    
    def calculate_available_usdt(self, balances):
        """Calculate total USDT available for trading"""
        usdt_balance = balances.get('USDT', {}).get('free', Decimal('0'))
        
        # Convert other assets to USDT equivalent
        total_usdt_equivalent = usdt_balance
        
        for asset, balance_info in balances.items():
            if asset != 'USDT' and balance_info['free'] > 0:
                try:
                    # Get current price for conversion
                    symbol = f"{asset}USDT"
                    if symbol in [pair for pair in self.ultra_pairs]:
                        ticker = self.client.get_symbol_ticker(symbol=symbol)
                        price = Decimal(ticker['price'])
                        asset_value = balance_info['free'] * price
                        total_usdt_equivalent += asset_value
                        logger.info(f"💰 {asset}: {balance_info['free']} (~${asset_value:.2f})")
                except Exception as e:
                    logger.error(f"Error in kimera_ultra_aggressive_5min.py: {e}", exc_info=True)
                    raise  # Re-raise for proper error handling
                    continue
        
        return total_usdt_equivalent
    
    def get_market_data_ultra_fast(self, symbol):
        """Ultra-fast market data retrieval"""
        try:
            # Get ticker and order book simultaneously
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            depth = self.client.get_order_book(symbol=symbol, limit=5)
            
            current_price = Decimal(ticker['price'])
            
            # Calculate bid-ask spread
            best_bid = Decimal(depth['bids'][0][0])
            best_ask = Decimal(depth['asks'][0][0])
            spread = (best_ask - best_bid) / best_bid * 100
            
            # Calculate order book pressure
            bid_volume = sum(Decimal(bid[1]) for bid in depth['bids'])
            ask_volume = sum(Decimal(ask[1]) for ask in depth['asks'])
            pressure = (bid_volume - ask_volume) / (bid_volume + ask_volume) if (bid_volume + ask_volume) > 0 else 0
            
            return {
                'symbol': symbol,
                'price': current_price,
                'best_bid': best_bid,
                'best_ask': best_ask,
                'spread': spread,
                'pressure': pressure,
                'timestamp': time.time()
            }
        except Exception as e:
            logger.warning(f"Failed to get market data for {symbol}: {e}")
            return None
    
    def detect_ultra_opportunities(self, market_data):
        """Detect ultra-aggressive trading opportunities"""
        opportunities = []
        
        for symbol, data in market_data.items():
            if not data:
                continue
            
            # Ultra-aggressive opportunity detection
            spread = data['spread']
            pressure = data['pressure']
            
            # Opportunity 1: Tight spread with strong pressure
            if spread < 0.2 and abs(pressure) > 0.2:
                direction = 'BUY' if pressure > 0 else 'SELL'
                confidence = min(abs(pressure) * 3, 0.9)
                
                opportunities.append({
                    'symbol': symbol,
                    'type': 'PRESSURE_PLAY',
                    'direction': direction,
                    'confidence': confidence,
                    'expected_profit': spread * 1.5,
                    'entry_price': data['best_ask'] if direction == 'BUY' else data['best_bid'],
                    'urgency': 'ULTRA_HIGH'
                })
            
            # Opportunity 2: Very tight spread for quick scalping
            if spread < 0.1:
                opportunities.append({
                    'symbol': symbol,
                    'type': 'SCALP_OPPORTUNITY',
                    'direction': 'BUY',
                    'confidence': 0.7,
                    'expected_profit': spread * 0.5,
                    'entry_price': data['best_bid'],
                    'urgency': 'HIGH'
                })
        
        # Sort by urgency and expected profit
        opportunities.sort(key=lambda x: (x['urgency'] == 'ULTRA_HIGH', x['expected_profit']), reverse=True)
        
        return opportunities[:self.max_positions]  # Limit to max positions
    
    def calculate_position_size(self, symbol, entry_price, available_usdt):
        """Calculate optimal position size for ultra-aggressive trading"""
        try:
            # Get exchange info for precision
            exchange_info = self.client.get_exchange_info()
            symbol_info = None
            
            for s in exchange_info['symbols']:
                if s['symbol'] == symbol:
                    symbol_info = s
                    break
            
            if not symbol_info:
                return None
            
            # Extract filters
            min_qty = Decimal('0')
            step_size = Decimal('0')
            min_notional = Decimal('0')
            
            for filter_info in symbol_info['filters']:
                if filter_info['filterType'] == 'LOT_SIZE':
                    min_qty = Decimal(filter_info['minQty'])
                    step_size = Decimal(filter_info['stepSize'])
                elif filter_info['filterType'] == 'NOTIONAL':
                    min_notional = Decimal(filter_info['minNotional'])
            
            # Calculate position size (use smaller allocation for multiple positions)
            position_value = min(self.position_size_per_trade, available_usdt * Decimal('0.15'))  # 15% per position
            quantity = position_value / entry_price
            
            # Ensure minimum notional
            if quantity * entry_price < min_notional:
                quantity = min_notional / entry_price * Decimal('1.1')  # 10% buffer
            
            # Round to step size
            steps = (quantity / step_size).quantize(Decimal('1'), rounding=ROUND_DOWN)
            final_quantity = steps * step_size
            
            # Validate
            if final_quantity >= min_qty and final_quantity * entry_price >= min_notional:
                return {
                    'quantity': final_quantity,
                    'value': final_quantity * entry_price,
                    'quantity_str': f"{final_quantity:.8f}".rstrip('0').rstrip('.')
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}")
            return None
    
    def execute_ultra_aggressive_trade(self, opportunity, available_usdt):
        """Execute ultra-aggressive trade with maximum speed"""
        symbol = opportunity['symbol']
        direction = opportunity['direction']
        
        try:
            # Calculate position size
            entry_price = opportunity['entry_price']
            position_info = self.calculate_position_size(symbol, entry_price, available_usdt)
            
            if not position_info:
                logger.warning(f"⚠️ Cannot calculate position size for {symbol}")
                return None
            
            quantity_str = position_info['quantity_str']
            
            logger.info(f"⚡ EXECUTING {direction} {quantity_str} {symbol} @ ${entry_price}")
            
            # Execute trade based on direction
            if direction == 'BUY':
                order = self.client.order_market_buy(
                    symbol=symbol,
                    quantity=quantity_str
                )
            elif direction == 'SELL':
                order = self.client.order_market_sell(
                    symbol=symbol,
                    quantity=quantity_str
                )
            
            # Track the trade
            trade_data = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'direction': direction,
                'quantity': quantity_str,
                'opportunity_type': opportunity['type'],
                'confidence': opportunity['confidence'],
                'order_id': order['orderId'],
                'status': order['status'],
                'order_data': order
            }
            
            self.completed_trades.append(trade_data)
            self.total_trades += 1
            
            if order['status'] == 'FILLED':
                self.successful_trades += 1
                logger.info(f"✅ TRADE EXECUTED: {symbol} {direction} - Order ID: {order['orderId']}")
            
            return trade_data
            
        except BinanceAPIException as e:
            logger.error(f"❌ Trade execution failed for {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Unexpected error executing {symbol}: {e}")
            return None
    
    def ultra_aggressive_trading_loop(self):
        """Main ultra-aggressive trading loop"""
        logger.info("🚀 STARTING ULTRA-AGGRESSIVE TRADING LOOP")
        
        last_balance_check = 0
        balance_check_interval = 15  # Check balances every 15 seconds
        
        while self.running and datetime.now() < self.end_time:
            loop_start = time.time()
            
            try:
                # Get current balances periodically
                current_time = time.time()
                if current_time - last_balance_check > balance_check_interval:
                    balances = self.get_account_balances()
                    available_usdt = self.calculate_available_usdt(balances)
                    last_balance_check = current_time
                    
                    logger.info(f"💰 Available USDT Equivalent: ${available_usdt:.2f}")
                    
                    if available_usdt < Decimal('5'):
                        logger.warning("⚠️ Low funds for aggressive trading")
                        time.sleep(2)
                        continue
                
                # Get market data for all pairs simultaneously
                market_data = {}
                with ThreadPoolExecutor(max_workers=8) as executor:
                    futures = {executor.submit(self.get_market_data_ultra_fast, symbol): symbol 
                              for symbol in self.ultra_pairs}
                    
                    for future in futures:
                        symbol = futures[future]
                        try:
                            data = future.result(timeout=1)
                            if data:
                                market_data[symbol] = data
                        except Exception as e:
                            logger.warning(f"Market data timeout for {symbol}")
                
                # Detect opportunities
                opportunities = self.detect_ultra_opportunities(market_data)
                
                if opportunities:
                    logger.info(f"🎯 DETECTED {len(opportunities)} ULTRA OPPORTUNITIES")
                    
                    # Execute top opportunities
                    for opp in opportunities[:3]:  # Execute top 3 opportunities
                        if len(self.active_positions) < self.max_positions:
                            result = self.execute_ultra_aggressive_trade(opp, available_usdt)
                            if result:
                                self.active_positions[result['order_id']] = result
                
                # Performance metrics
                elapsed = (datetime.now() - self.start_time).total_seconds()
                self.trade_frequency = self.total_trades / elapsed if elapsed > 0 else 0
                
                # Show progress
                remaining = (self.end_time - datetime.now()).total_seconds()
                if remaining > 0:
                    logger.info(f"⏱️ {remaining:.1f}s remaining | 📊 {self.total_trades} trades | ⚡ {self.trade_frequency:.1f}/min")
                
                # Ultra-fast loop timing
                loop_time = time.time() - loop_start
                if loop_time < self.execution_interval:
                    time.sleep(self.execution_interval - loop_time)
                
            except Exception as e:
                logger.error(f"❌ Error in trading loop: {e}")
                time.sleep(0.5)
    
    def calculate_final_performance(self):
        """Calculate final performance metrics"""
        try:
            # Get final balances
            final_balances = self.get_account_balances()
            final_usdt_equivalent = self.calculate_available_usdt(final_balances)
            
            # Calculate profit
            profit = final_usdt_equivalent - self.allocation_usd
            profit_percentage = (profit / self.allocation_usd) * 100
            
            # Trading statistics
            success_rate = (self.successful_trades / self.total_trades * 100) if self.total_trades > 0 else 0
            
            runtime_seconds = (datetime.now() - self.start_time).total_seconds()
            trades_per_minute = (self.total_trades / runtime_seconds * 60) if runtime_seconds > 0 else 0
            
            return {
                'initial_allocation': float(self.allocation_usd),
                'final_value': float(final_usdt_equivalent),
                'profit': float(profit),
                'profit_percentage': float(profit_percentage),
                'total_trades': self.total_trades,
                'successful_trades': self.successful_trades,
                'success_rate': float(success_rate),
                'runtime_seconds': runtime_seconds,
                'trades_per_minute': float(trades_per_minute),
                'final_balances': {k: {'free': float(v['free']), 'total': float(v['total'])} 
                                 for k, v in final_balances.items()}
            }
        except Exception as e:
            logger.error(f"Error calculating performance: {e}")
            return {}
    
    def emergency_stop(self):
        """Emergency stop all trading"""
        logger.info("🛑 EMERGENCY STOP ACTIVATED")
        self.running = False
        
        # Cancel all open orders
        try:
            open_orders = self.client.get_open_orders()
            for order in open_orders:
                try:
                    self.client.cancel_order(symbol=order['symbol'], orderId=order['orderId'])
                    logger.info(f"❌ Cancelled order {order['orderId']}")
                except Exception as e:
                    logger.error(f"Error in kimera_ultra_aggressive_5min.py: {e}", exc_info=True)
                    raise  # Re-raise for proper error handling
        except Exception as e:
            logger.error(f"Error in kimera_ultra_aggressive_5min.py: {e}", exc_info=True)
            raise  # Re-raise for proper error handling
    
    def run_ultra_aggressive_session(self):
        """Run the complete ultra-aggressive trading session"""
        self.setup_signal_handlers()
        
        print("\n" + "="*80)
        print("🔥 KIMERA ULTRA-AGGRESSIVE 5-MINUTE PROFIT MAXIMIZER")
        print("="*80)
        print(f"💰 Allocation: ${self.allocation_usd}")
        print(f"⏱️  Runtime: {self.runtime_minutes} minutes")
        print(f"🎯 Target: MAXIMUM PROFIT EXTRACTION")
        print(f"⚡ Mode: ULTRA-AGGRESSIVE PERFORMANCE")
        
        # Get initial balances
        initial_balances = self.get_account_balances()
        initial_usdt = self.calculate_available_usdt(initial_balances)
        
        print(f"\n💰 INITIAL PORTFOLIO VALUE: ${initial_usdt:.2f}")
        
        if initial_usdt < self.allocation_usd:
            print(f"⚠️ WARNING: Available funds (${initial_usdt:.2f}) less than allocation (${self.allocation_usd})")
            self.allocation_usd = initial_usdt * Decimal('0.9')  # Use 90% of available
            print(f"🔄 Adjusted allocation to: ${self.allocation_usd:.2f}")
        
        # Set runtime
        self.start_time = datetime.now()
        self.end_time = self.start_time + timedelta(minutes=self.runtime_minutes)
        
        print(f"\n🚀 ULTRA-AGGRESSIVE TRADING STARTS: {self.start_time.strftime('%H:%M:%S')}")
        print(f"🏁 SESSION ENDS: {self.end_time.strftime('%H:%M:%S')}")
        
        confirm = input(f"\n⚠️ CONFIRM ULTRA-AGGRESSIVE TRADING? (type 'ULTRA' to proceed): ")
        if confirm != 'ULTRA':
            print("❌ Session cancelled")
            return
        
        print(f"\n⚡ KIMERA ULTRA-AGGRESSIVE MODE ACTIVATED!")
        
        # Start trading
        self.running = True
        
        try:
            self.ultra_aggressive_trading_loop()
        except KeyboardInterrupt:
            logger.info("🛑 Manual stop requested")
        except Exception as e:
            logger.error(f"❌ Fatal error: {e}")
        finally:
            self.running = False
        
        # Calculate final performance
        print(f"\n🏁 ULTRA-AGGRESSIVE SESSION COMPLETED!")
        performance = self.calculate_final_performance()
        
        print(f"\n📊 FINAL PERFORMANCE REPORT:")
        print("="*50)
        print(f"💰 Initial Value: ${performance.get('initial_allocation', 0):.2f}")
        print(f"💰 Final Value: ${performance.get('final_value', 0):.2f}")
        print(f"📈 Profit: ${performance.get('profit', 0):.2f}")
        print(f"📊 Profit %: {performance.get('profit_percentage', 0):.2f}%")
        print(f"🔢 Total Trades: {performance.get('total_trades', 0)}")
        print(f"✅ Successful: {performance.get('successful_trades', 0)}")
        print(f"📈 Success Rate: {performance.get('success_rate', 0):.1f}%")
        print(f"⚡ Trades/Min: {performance.get('trades_per_minute', 0):.1f}")
        print(f"⏱️  Runtime: {performance.get('runtime_seconds', 0):.1f}s")
        
        # Save detailed results
        session_data = {
            'session_info': {
                'start_time': self.start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'allocation': float(self.allocation_usd),
                'runtime_minutes': self.runtime_minutes
            },
            'performance': performance,
            'completed_trades': self.completed_trades,
            'parameters': {
                'min_profit_threshold': float(self.min_profit_threshold),
                'max_positions': self.max_positions,
                'execution_interval': self.execution_interval,
                'trading_pairs': self.ultra_pairs
            }
        }
        
        filename = f"kimera_ultra_aggressive_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=2, default=str)
        
        print(f"\n💾 Session data saved to: {filename}")
        
        if performance.get('profit', 0) > 0:
            print(f"\n🎉 PROFIT TARGET ACHIEVED!")
            print(f"🚀 KIMERA ULTRA-AGGRESSIVE MODE: SUCCESS!")
        else:
            print(f"\n📊 Session completed - Market conditions analyzed")
        
        return performance

def main():
    """Main execution function"""
    
    # API credentials
    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = "qUn5JqSpYz1GDxFj2X3UF23TYgtxKrTsCbDZEoBMYCPbYZgP4siVLyspkB5HAPl7"
    
    # Initialize ultra-aggressive trader
    kimera = KimeraUltraAggressive(
        api_key=api_key,
        api_secret=api_secret,
        allocation_usd=25,
        runtime_minutes=5
    )
    
    # Run the session
    performance = kimera.run_ultra_aggressive_session()
    
    return performance

if __name__ == "__main__":
    main() 