#!/usr/bin/env python3
"""
KIMERA WEBSOCKET TRADER - RATE LIMIT OPTIMIZED
Uses WebSocket streams for real-time data to avoid API bans
Implements aggressive but intelligent trading strategies
"""

import logging
import time
import json
import asyncio
import websockets
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN
from binance import Client, ThreadedWebsocketManager
from binance.exceptions import BinanceAPIException
import threading
import queue
import signal
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('kimera_websocket_trader.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class KimeraWebSocketTrader:
    def __init__(self, api_key, api_secret, runtime_minutes=10):
        self.client = Client(api_key, api_secret)
        self.runtime_minutes = runtime_minutes
        self.start_time = None
        self.end_time = None
        
        # Trading parameters
        self.min_profit_target = Decimal('0.001')  # 0.1% minimum profit
        self.max_simultaneous_trades = 2
        self.trade_size_percentage = Decimal('0.05')  # 5% of portfolio per trade
        
        # Performance tracking
        self.starting_usdt_value = Decimal('0')
        self.completed_trades = []
        self.total_profit = Decimal('0')
        self.trade_count = 0
        
        # Real-time data storage
        self.ticker_data = {}
        self.orderbook_data = {}
        self.kline_data = {}
        
        # Active trading pairs
        self.trading_pairs = [
            'trxusdt', 'dogeusdt', 'shibusdt', 'pepeusdt', 'adausdt'
        ]
        
        self.running = False
        self.twm = None
        
    def setup_emergency_stop(self):
        """Setup emergency stop"""
        def emergency_handler(signum, frame):
            logger.info("EMERGENCY STOP - STOPPING ALL TRADES")
            self.running = False
            if self.twm:
                self.twm.stop()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, emergency_handler)
        signal.signal(signal.SIGTERM, emergency_handler)
    
    def handle_ticker_message(self, msg):
        """Handle ticker WebSocket messages"""
        try:
            symbol = msg['s'].lower()
            if symbol in self.trading_pairs:
                self.ticker_data[symbol] = {
                    'price': Decimal(msg['c']),
                    'change_24h': Decimal(msg['P']),
                    'volume': Decimal(msg['v']),
                    'timestamp': datetime.now()
                }
        except Exception as e:
            logger.warning(f"Error processing ticker data: {e}")
    
    def handle_depth_message(self, msg):
        """Handle order book depth messages"""
        try:
            symbol = msg['s'].lower()
            if symbol in self.trading_pairs:
                self.orderbook_data[symbol] = {
                    'bids': [[Decimal(bid[0]), Decimal(bid[1])] for bid in msg['b'][:5]],
                    'asks': [[Decimal(ask[0]), Decimal(ask[1])] for ask in msg['a'][:5]],
                    'timestamp': datetime.now()
                }
        except Exception as e:
            logger.warning(f"Error processing depth data: {e}")
    
    def handle_kline_message(self, msg):
        """Handle kline/candlestick messages"""
        try:
            symbol = msg['s'].lower()
            if symbol in self.trading_pairs:
                kline = msg['k']
                if symbol not in self.kline_data:
                    self.kline_data[symbol] = []
                
                self.kline_data[symbol].append({
                    'open': Decimal(kline['o']),
                    'high': Decimal(kline['h']),
                    'low': Decimal(kline['l']),
                    'close': Decimal(kline['c']),
                    'volume': Decimal(kline['v']),
                    'timestamp': datetime.now()
                })
                
                # Keep only last 20 klines
                if len(self.kline_data[symbol]) > 20:
                    self.kline_data[symbol] = self.kline_data[symbol][-20:]
        except Exception as e:
            logger.warning(f"Error processing kline data: {e}")
    
    def setup_websocket_streams(self):
        """Setup WebSocket streams for real-time data"""
        try:
            self.twm = ThreadedWebsocketManager(api_key=self.client.API_KEY, api_secret=self.client.API_SECRET)
            self.twm.start()
            
            # Start ticker streams
            for symbol in self.trading_pairs:
                symbol_upper = symbol.upper()
                self.twm.start_symbol_ticker_socket(callback=self.handle_ticker_message, symbol=symbol_upper)
                self.twm.start_depth_socket(callback=self.handle_depth_message, symbol=symbol_upper, depth=5)
                self.twm.start_kline_socket(callback=self.handle_kline_message, symbol=symbol_upper, interval='1m')
            
            logger.info(f"WebSocket streams started for {len(self.trading_pairs)} pairs")
            
            # Wait for initial data
            time.sleep(3)
            
        except Exception as e:
            logger.error(f"Error setting up WebSocket streams: {e}")
    
    def get_portfolio_value(self):
        """Get current portfolio value using minimal API calls"""
        try:
            account = self.client.get_account()
            total_value = Decimal('0')
            
            for balance in account['balances']:
                asset = balance['asset']
                free_amount = Decimal(balance['free'])
                
                if free_amount > 0:
                    if asset == 'USDT':
                        total_value += free_amount
                    else:
                        symbol = f"{asset.lower()}usdt"
                        if symbol in self.ticker_data:
                            price = self.ticker_data[symbol]['price']
                            value = free_amount * price
                            total_value += value
            
            return total_value
        except Exception as e:
            logger.error(f"Error calculating portfolio value: {e}")
            return Decimal('0')
    
    def find_trading_opportunity(self, symbol):
        """Find trading opportunities using real-time WebSocket data"""
        try:
            symbol_lower = symbol.lower()
            
            # Check if we have sufficient data
            if (symbol_lower not in self.ticker_data or 
                symbol_lower not in self.orderbook_data or 
                symbol_lower not in self.kline_data):
                return None
            
            ticker = self.ticker_data[symbol_lower]
            orderbook = self.orderbook_data[symbol_lower]
            klines = self.kline_data[symbol_lower]
            
            if not orderbook['bids'] or not orderbook['asks'] or len(klines) < 5:
                return None
            
            # Calculate key metrics
            best_bid = orderbook['bids'][0][0]
            best_ask = orderbook['asks'][0][0]
            spread = (best_ask - best_bid) / best_bid
            
            # Volume analysis
            recent_volume = sum([k['volume'] for k in klines[-5:]])
            avg_volume = recent_volume / 5
            
            # Price momentum
            prices = [k['close'] for k in klines[-10:]]
            if len(prices) >= 5:
                short_ma = sum(prices[-3:]) / 3
                long_ma = sum(prices[-5:]) / 5
                momentum = (short_ma - long_ma) / long_ma
                
                # Strong upward momentum + good spread = BUY
                if momentum > Decimal('0.002') and spread > self.min_profit_target:
                    return {
                        'symbol': symbol_lower,
                        'action': 'BUY',
                        'entry_price': best_ask,
                        'target_price': best_ask * (Decimal('1') + self.min_profit_target),
                        'spread': spread,
                        'momentum': momentum,
                        'confidence': min(float(momentum * 1000), 0.9),
                        'volume_score': min(float(avg_volume / 1000), 1.0)
                    }
                
                # Strong downward momentum = SELL (if we have the asset)
                elif momentum < Decimal('-0.002') and spread > self.min_profit_target:
                    return {
                        'symbol': symbol_lower,
                        'action': 'SELL',
                        'entry_price': best_bid,
                        'target_price': best_bid * (Decimal('1') - self.min_profit_target),
                        'spread': spread,
                        'momentum': momentum,
                        'confidence': min(float(abs(momentum) * 1000), 0.9),
                        'volume_score': min(float(avg_volume / 1000), 1.0)
                    }
            
            return None
            
        except Exception as e:
            logger.warning(f"Error analyzing {symbol}: {e}")
            return None
    
    def calculate_trade_quantity(self, symbol, price, action):
        """Calculate optimal trade quantity"""
        try:
            portfolio_value = self.get_portfolio_value()
            trade_value = portfolio_value * self.trade_size_percentage
            
            if action == 'BUY':
                quantity = trade_value / price
            else:  # SELL
                # Check available balance
                account = self.client.get_account()
                asset = symbol.upper().replace('USDT', '')
                available = Decimal('0')
                
                for balance in account['balances']:
                    if balance['asset'] == asset:
                        available = Decimal(balance['free'])
                        break
                
                if available == 0:
                    return None
                
                quantity = min(available, trade_value / price)
            
            # Get symbol info for precision (cache this to avoid API calls)
            if not hasattr(self, '_symbol_info'):
                self._symbol_info = {}
            
            if symbol not in self._symbol_info:
                exchange_info = self.client.get_exchange_info()
                for s in exchange_info['symbols']:
                    if s['symbol'] == symbol.upper():
                        self._symbol_info[symbol] = s
                        break
            
            if symbol not in self._symbol_info:
                return None
            
            symbol_info = self._symbol_info[symbol]
            
            # Apply trading rules
            min_qty = Decimal('0')
            step_size = Decimal('0')
            min_notional = Decimal('0')
            
            for filter_info in symbol_info['filters']:
                if filter_info['filterType'] == 'LOT_SIZE':
                    min_qty = Decimal(filter_info['minQty'])
                    step_size = Decimal(filter_info['stepSize'])
                elif filter_info['filterType'] == 'NOTIONAL':
                    min_notional = Decimal(filter_info['minNotional'])
            
            # Apply precision
            if step_size > 0:
                quantity = quantity.quantize(step_size, rounding=ROUND_DOWN)
            
            # Check requirements
            if quantity < min_qty or quantity * price < min_notional:
                return None
            
            return quantity
            
        except Exception as e:
            logger.error(f"Error calculating quantity for {symbol}: {e}")
            return None
    
    def execute_trade(self, opportunity):
        """Execute trade with minimal API calls"""
        try:
            symbol = opportunity['symbol'].upper()
            action = opportunity['action']
            entry_price = opportunity['entry_price']
            
            quantity = self.calculate_trade_quantity(opportunity['symbol'], entry_price, action)
            if not quantity:
                logger.warning(f"Cannot calculate valid quantity for {symbol}")
                return None
            
            logger.info(f"EXECUTING: {action} {quantity} {symbol} @ ${entry_price}")
            
            # Execute order
            if action == 'BUY':
                order = self.client.order_market_buy(symbol=symbol, quantity=str(quantity))
            else:
                order = self.client.order_market_sell(symbol=symbol, quantity=str(quantity))
            
            logger.info(f"ORDER EXECUTED: {order['orderId']} - {order['status']}")
            
            # Record trade
            executed_qty = Decimal(order['executedQty'])
            executed_value = Decimal(order['cummulativeQuoteQty'])
            
            if executed_qty > 0:
                avg_price = executed_value / executed_qty
                
                trade_record = {
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'action': action,
                    'quantity': executed_qty,
                    'price': avg_price,
                    'value': executed_value,
                    'order_id': order['orderId']
                }
                
                self.completed_trades.append(trade_record)
                self.trade_count += 1
                
                logger.info(f"TRADE COMPLETED: {action} {executed_qty} {symbol} @ ${avg_price}")
                return trade_record
            
        except BinanceAPIException as e:
            logger.error(f"API error executing {action} for {symbol}: {e}")
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
        
        return None
    
    def trading_loop(self):
        """Main trading loop using WebSocket data"""
        logger.info("STARTING WEBSOCKET TRADING LOOP")
        
        last_scan = 0
        scan_interval = 2  # Scan every 2 seconds
        
        while self.running and datetime.now() < self.end_time:
            try:
                current_time = time.time()
                
                if current_time - last_scan >= scan_interval:
                    opportunities = []
                    
                    # Scan all pairs for opportunities
                    for symbol in self.trading_pairs:
                        opportunity = self.find_trading_opportunity(symbol)
                        if opportunity:
                            opportunities.append(opportunity)
                    
                    # Sort by confidence * volume score
                    opportunities.sort(key=lambda x: x['confidence'] * x['volume_score'], reverse=True)
                    
                    # Execute best opportunities
                    active_trades = len([t for t in self.completed_trades 
                                       if (datetime.now() - t['timestamp']).seconds < 300])
                    
                    available_slots = self.max_simultaneous_trades - active_trades
                    
                    for opp in opportunities[:available_slots]:
                        logger.info(f"OPPORTUNITY: {opp['action']} {opp['symbol'].upper()} "
                                  f"Spread: {opp['spread']:.4f} Momentum: {opp['momentum']:.4f} "
                                  f"Confidence: {opp['confidence']:.3f}")
                        
                        trade_result = self.execute_trade(opp)
                        if trade_result:
                            logger.info(f"PROFIT TRADE: {trade_result['action']} "
                                      f"{trade_result['quantity']} {trade_result['symbol']}")
                    
                    # Performance update
                    if self.trade_count > 0:
                        current_value = self.get_portfolio_value()
                        profit = current_value - self.starting_usdt_value
                        
                        elapsed = (datetime.now() - self.start_time).total_seconds() / 60
                        remaining = self.runtime_minutes - elapsed
                        
                        logger.info(f"PERFORMANCE - Profit: ${profit:.2f} | "
                                  f"Trades: {self.trade_count} | "
                                  f"Portfolio: ${current_value:.2f} | "
                                  f"Time: {remaining:.1f}min")
                    
                    last_scan = current_time
                
                time.sleep(0.1)  # Small sleep to prevent CPU overload
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(1)
        
        logger.info("TRADING SESSION COMPLETED")
    
    def run_trading_session(self):
        """Run complete trading session"""
        try:
            self.setup_emergency_stop()
            self.start_time = datetime.now()
            self.end_time = self.start_time + timedelta(minutes=self.runtime_minutes)
            
            logger.info("=" * 80)
            logger.info("KIMERA WEBSOCKET TRADER - RATE LIMIT OPTIMIZED")
            logger.info("=" * 80)
            logger.info(f"Runtime: {self.runtime_minutes} minutes")
            logger.info(f"Target: REAL PROFIT WITH WEBSOCKET EFFICIENCY")
            logger.info(f"Mode: INTELLIGENT AGGRESSIVE TRADING")
            
            # Setup WebSocket streams
            logger.info("Setting up WebSocket streams...")
            self.setup_websocket_streams()
            
            # Get starting value
            self.starting_usdt_value = self.get_portfolio_value()
            logger.info(f"STARTING PORTFOLIO: ${self.starting_usdt_value:.2f}")
            
            if self.starting_usdt_value < Decimal('5'):
                logger.error("INSUFFICIENT FUNDS - Need at least $5")
                return
            
            # Start trading
            self.running = True
            logger.info("STARTING WEBSOCKET TRADING")
            
            self.trading_loop()
            
            # Final results
            final_value = self.get_portfolio_value()
            total_profit = final_value - self.starting_usdt_value
            
            logger.info("\n" + "=" * 80)
            logger.info("WEBSOCKET TRADING RESULTS")
            logger.info("=" * 80)
            logger.info(f"Starting Value: ${self.starting_usdt_value:.2f}")
            logger.info(f"Final Value: ${final_value:.2f}")
            logger.info(f"TOTAL PROFIT: ${total_profit:.2f}")
            logger.info(f"Profit %: {(total_profit / self.starting_usdt_value * 100):.2f}%")
            logger.info(f"Total Trades: {self.trade_count}")
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"kimera_websocket_results_{timestamp}.json"
            
            with open(results_file, 'w') as f:
                json.dump({
                    'performance': {
                        'starting_value': float(self.starting_usdt_value),
                        'final_value': float(final_value),
                        'total_profit': float(total_profit),
                        'profit_percentage': float(total_profit / self.starting_usdt_value * 100),
                        'total_trades': self.trade_count,
                        'runtime_minutes': self.runtime_minutes
                    },
                    'trades': [
                        {
                            'timestamp': trade['timestamp'].isoformat(),
                            'symbol': trade['symbol'],
                            'action': trade['action'],
                            'quantity': str(trade['quantity']),
                            'price': str(trade['price']),
                            'value': str(trade['value']),
                            'order_id': trade['order_id']
                        }
                        for trade in self.completed_trades
                    ]
                }, f, indent=2)
            
            logger.info(f"RESULTS SAVED: {results_file}")
            
            if total_profit > 0:
                logger.info(f"SUCCESS: ${total_profit:.2f} PROFIT GENERATED!")
            
        except Exception as e:
            logger.error(f"Critical error: {e}")
        finally:
            self.running = False
            if self.twm:
                self.twm.stop()

def main():
    """Main execution"""
    api_key = "Y9WyflPyK1tVXnET3CTMvSdCbPia3Nhtd89VYWjS9RaAbQ0KEhHezkcGSCySQ8cL"
    api_secret = "qUn5JqSpYz1GDxFj2X3UF23TYgtxKrTsCbDZEoBMYCPbYZgP4siVLyspkB5HAPl7"
    
    trader = KimeraWebSocketTrader(
        api_key=api_key,
        api_secret=api_secret,
        runtime_minutes=10
    )
    
    trader.run_trading_session()

if __name__ == "__main__":
    main() 