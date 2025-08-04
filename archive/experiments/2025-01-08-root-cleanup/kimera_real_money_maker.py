import os
#!/usr/bin/env python3
"""
KIMERA REAL MONEY MAKER - NO BULLSHIT VERSION
Actual aggressive trading to generate REAL profits
No portfolio valuation tricks - REAL TRADES ONLY
"""

import logging
import time
import json
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN
from binance import Client
from binance.exceptions import BinanceAPIException
from concurrent.futures import ThreadPoolExecutor
import signal
import sys

# Configure aggressive logging with ASCII only
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('kimera_real_money.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class KimeraRealMoneyMaker:
    def __init__(self, api_key, api_secret, runtime_minutes=10):
        self.client = Client(api_key, api_secret)
        self.runtime_minutes = runtime_minutes
        self.start_time = None
        self.end_time = None
        
        # REAL trading parameters
        self.min_profit_target = Decimal('0.002')  # 0.2% minimum profit per trade
        self.max_simultaneous_trades = 3
        self.execution_speed = 0.5  # 500ms between scans
        
        # Track REAL performance
        self.starting_usdt_value = Decimal('0')
        self.completed_real_trades = []
        self.total_real_profit = Decimal('0')
        self.trade_count = 0
        
        # High-volume pairs for REAL opportunities
        self.trading_pairs = [
            'TRXUSDT', 'DOGEUSDT', 'SHIBUSDT', 'PEPEUSDT', 'ADAUSDT',
            'XRPUSDT', 'SOLUSDT', 'DOTUSDT', 'AVAXUSDT', 'LINKUSDT'
        ]
        
        self.running = False
        
    def setup_emergency_stop(self):
        """Setup emergency stop"""
        def emergency_handler(signum, frame):
            logger.info("EMERGENCY STOP - STOPPING ALL TRADES")
            self.running = False
            sys.exit(0)
        
        signal.signal(signal.SIGINT, emergency_handler)
        signal.signal(signal.SIGTERM, emergency_handler)
    
    def get_real_usdt_value(self):
        """Calculate REAL USDT equivalent value of entire portfolio"""
        try:
            account = self.client.get_account()
            total_usdt_value = Decimal('0')
            
            for balance in account['balances']:
                asset = balance['asset']
                free_amount = Decimal(balance['free'])
                
                if free_amount > 0:
                    if asset == 'USDT':
                        total_usdt_value += free_amount
                        logger.info(f"BALANCE {asset}: {free_amount} USDT")
                    else:
                        try:
                            symbol = f"{asset}USDT"
                            ticker = self.client.get_symbol_ticker(symbol=symbol)
                            price = Decimal(ticker['price'])
                            value = free_amount * price
                            total_usdt_value += value
                            logger.info(f"BALANCE {asset}: {free_amount} @ ${price} = ${value:.2f}")
                        except Exception as e:
                            logger.error(f"Error in kimera_real_money_maker.py: {e}", exc_info=True)
                            raise  # Re-raise for proper error handling
                            # Skip assets we can't price
                            continue
            
            return total_usdt_value
        except Exception as e:
            logger.error(f"Error calculating portfolio value: {e}")
            return Decimal('0')
    
    def find_real_scalping_opportunity(self, symbol):
        """Find REAL scalping opportunities with actual profit potential"""
        try:
            # Get order book depth
            depth = self.client.get_order_book(symbol=symbol, limit=10)
            
            if not depth['bids'] or not depth['asks']:
                return None
            
            best_bid = Decimal(depth['bids'][0][0])
            best_ask = Decimal(depth['asks'][0][0])
            bid_volume = Decimal(depth['bids'][0][1])
            ask_volume = Decimal(depth['asks'][0][1])
            
            # Calculate spread
            spread = (best_ask - best_bid) / best_bid
            
            # Look for REAL opportunities
            if spread > self.min_profit_target and min(bid_volume, ask_volume) > 100:
                
                # Check recent price movement
                klines = self.client.get_klines(symbol=symbol, interval='1m', limit=5)
                recent_prices = [Decimal(k[4]) for k in klines]  # Closing prices
                
                # Calculate momentum
                if len(recent_prices) >= 2:
                    momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
                    
                    # Strong upward momentum + good spread = BUY opportunity
                    if momentum > Decimal('0.001') and spread > Decimal('0.003'):
                        return {
                            'symbol': symbol,
                            'action': 'BUY',
                            'entry_price': best_ask,
                            'target_price': best_ask * (Decimal('1') + self.min_profit_target),
                            'spread': spread,
                            'momentum': momentum,
                            'confidence': min(float(momentum * 100), 0.8)
                        }
                    
                    # Strong downward momentum + good spread = SELL opportunity  
                    elif momentum < Decimal('-0.001') and spread > Decimal('0.003'):
                        return {
                            'symbol': symbol,
                            'action': 'SELL',
                            'entry_price': best_bid,
                            'target_price': best_bid * (Decimal('1') - self.min_profit_target),
                            'spread': spread,
                            'momentum': momentum,
                            'confidence': min(float(abs(momentum) * 100), 0.8)
                        }
            
            return None
            
        except Exception as e:
            logger.warning(f"Error analyzing {symbol}: {e}")
            return None
    
    def calculate_trade_size(self, symbol, entry_price, action):
        """Calculate optimal trade size for REAL profit"""
        try:
            # Get current portfolio value
            portfolio_value = self.get_real_usdt_value()
            
            # Use 10% of portfolio per trade (aggressive but not reckless)
            trade_value = portfolio_value * Decimal('0.1')
            
            # Get symbol info for precision
            exchange_info = self.client.get_exchange_info()
            symbol_info = None
            
            for s in exchange_info['symbols']:
                if s['symbol'] == symbol:
                    symbol_info = s
                    break
            
            if not symbol_info:
                return None
            
            # Extract trading rules
            min_qty = Decimal('0')
            step_size = Decimal('0')
            min_notional = Decimal('0')
            
            for filter_info in symbol_info['filters']:
                if filter_info['filterType'] == 'LOT_SIZE':
                    min_qty = Decimal(filter_info['minQty'])
                    step_size = Decimal(filter_info['stepSize'])
                elif filter_info['filterType'] == 'NOTIONAL':
                    min_notional = Decimal(filter_info['minNotional'])
            
            # Calculate quantity
            if action == 'BUY':
                quantity = trade_value / entry_price
            else:  # SELL - check available balance
                account = self.client.get_account()
                asset = symbol.replace('USDT', '')
                available = Decimal('0')
                
                for balance in account['balances']:
                    if balance['asset'] == asset:
                        available = Decimal(balance['free'])
                        break
                
                if available == 0:
                    return None
                
                quantity = min(available, trade_value / entry_price)
            
            # Apply step size precision
            if step_size > 0:
                quantity = quantity.quantize(step_size, rounding=ROUND_DOWN)
            
            # Check minimum requirements
            notional_value = quantity * entry_price
            if quantity < min_qty or notional_value < min_notional:
                return None
            
            return quantity
            
        except Exception as e:
            logger.error(f"Error calculating trade size for {symbol}: {e}")
            return None
    
    def execute_real_trade(self, opportunity):
        """Execute REAL trade on Binance - NO SIMULATION"""
        try:
            symbol = opportunity['symbol']
            action = opportunity['action']
            entry_price = opportunity['entry_price']
            
            # Calculate trade size
            quantity = self.calculate_trade_size(symbol, entry_price, action)
            if not quantity:
                logger.warning(f"Cannot calculate valid trade size for {symbol}")
                return None
            
            logger.info(f"EXECUTING REAL TRADE: {action} {quantity} {symbol} @ ${entry_price}")
            
            # Execute REAL order
            if action == 'BUY':
                order = self.client.order_market_buy(
                    symbol=symbol,
                    quantity=str(quantity)
                )
            else:  # SELL
                order = self.client.order_market_sell(
                    symbol=symbol,
                    quantity=str(quantity)
                )
            
            # Log REAL execution
            logger.info(f"REAL ORDER EXECUTED: {order['orderId']} - {order['status']}")
            
            # Calculate REAL execution details
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
                    'order_id': order['orderId'],
                    'opportunity': opportunity
                }
                
                self.completed_real_trades.append(trade_record)
                self.trade_count += 1
                
                logger.info(f"REAL TRADE COMPLETED: {action} {executed_qty} {symbol} @ ${avg_price}")
                
                return trade_record
            
        except BinanceAPIException as e:
            logger.error(f"Binance API error executing {action} for {symbol}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error executing trade: {e}")
        
        return None
    
    def real_money_making_loop(self):
        """Main aggressive trading loop - REAL MONEY ONLY"""
        logger.info("STARTING REAL MONEY MAKING LOOP")
        
        while self.running and datetime.now() < self.end_time:
            try:
                # Scan all pairs simultaneously for opportunities
                opportunities = []
                
                with ThreadPoolExecutor(max_workers=5) as executor:
                    future_to_symbol = {
                        executor.submit(self.find_real_scalping_opportunity, symbol): symbol 
                        for symbol in self.trading_pairs
                    }
                    
                    for future in future_to_symbol:
                        try:
                            opportunity = future.result(timeout=2)
                            if opportunity:
                                opportunities.append(opportunity)
                        except Exception as e:
                            logger.warning(f"Error scanning {future_to_symbol[future]}: {e}")
                
                # Sort by confidence and execute best opportunities
                opportunities.sort(key=lambda x: x['confidence'], reverse=True)
                
                # Execute top opportunities (up to max simultaneous)
                active_trades = len([t for t in self.completed_real_trades 
                                   if (datetime.now() - t['timestamp']).seconds < 300])
                
                available_slots = self.max_simultaneous_trades - active_trades
                
                for i, opp in enumerate(opportunities[:available_slots]):
                    logger.info(f"OPPORTUNITY FOUND: {opp['action']} {opp['symbol']} "
                              f"Spread: {opp['spread']:.4f} Momentum: {opp['momentum']:.4f} "
                              f"Confidence: {opp['confidence']:.2f}")
                    
                    # Execute REAL trade
                    trade_result = self.execute_real_trade(opp)
                    
                    if trade_result:
                        logger.info(f"REAL PROFIT TRADE EXECUTED: {trade_result['action']} "
                                  f"{trade_result['quantity']} {trade_result['symbol']}")
                
                # Calculate and display REAL performance
                current_value = self.get_real_usdt_value()
                real_profit = current_value - self.starting_usdt_value
                
                elapsed = (datetime.now() - self.start_time).total_seconds() / 60
                remaining = self.runtime_minutes - elapsed
                
                logger.info(f"REAL PERFORMANCE - Profit: ${real_profit:.2f} | "
                          f"Trades: {self.trade_count} | "
                          f"Portfolio: ${current_value:.2f} | "
                          f"Time Remaining: {remaining:.1f}min")
                
                # Aggressive execution speed
                time.sleep(self.execution_speed)
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(1)
        
        logger.info("REAL MONEY MAKING SESSION COMPLETED")
    
    def calculate_real_performance(self):
        """Calculate REAL performance metrics"""
        current_value = self.get_real_usdt_value()
        total_profit = current_value - self.starting_usdt_value
        
        performance = {
            'starting_value': float(self.starting_usdt_value),
            'ending_value': float(current_value),
            'total_profit': float(total_profit),
            'profit_percentage': float((total_profit / self.starting_usdt_value) * 100) if self.starting_usdt_value > 0 else 0,
            'total_trades': self.trade_count,
            'runtime_minutes': self.runtime_minutes,
            'trades_executed': len(self.completed_real_trades)
        }
        
        return performance
    
    def run_real_money_session(self):
        """Run complete REAL money making session"""
        try:
            # Setup
            self.setup_emergency_stop()
            self.start_time = datetime.now()
            self.end_time = self.start_time + timedelta(minutes=self.runtime_minutes)
            
            logger.info("=" * 80)
            logger.info("KIMERA REAL MONEY MAKER - NO BULLSHIT VERSION")
            logger.info("=" * 80)
            logger.info(f"Runtime: {self.runtime_minutes} minutes")
            logger.info(f"Target: REAL PROFIT GENERATION")
            logger.info(f"Mode: ACTUAL AGGRESSIVE TRADING")
            
            # Get starting portfolio value
            self.starting_usdt_value = self.get_real_usdt_value()
            logger.info(f"STARTING PORTFOLIO VALUE: ${self.starting_usdt_value:.2f}")
            
            if self.starting_usdt_value < Decimal('10'):
                logger.error("INSUFFICIENT FUNDS - Need at least $10 to start trading")
                return
            
            # Start REAL trading
            self.running = True
            logger.info("STARTING AGGRESSIVE REAL MONEY TRADING")
            
            self.real_money_making_loop()
            
            # Calculate final performance
            performance = self.calculate_real_performance()
            
            logger.info("\n" + "=" * 80)
            logger.info("FINAL REAL MONEY RESULTS")
            logger.info("=" * 80)
            logger.info(f"Starting Value: ${performance['starting_value']:.2f}")
            logger.info(f"Ending Value: ${performance['ending_value']:.2f}")
            logger.info(f"REAL PROFIT: ${performance['total_profit']:.2f}")
            logger.info(f"Profit %: {performance['profit_percentage']:.2f}%")
            logger.info(f"Total Trades: {performance['total_trades']}")
            logger.info(f"Runtime: {performance['runtime_minutes']} minutes")
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"kimera_real_money_results_{timestamp}.json"
            
            with open(results_file, 'w') as f:
                json.dump({
                    'performance': performance,
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
                        for trade in self.completed_real_trades
                    ]
                }, f, indent=2)
            
            logger.info(f"REAL RESULTS SAVED TO: {results_file}")
            
            if performance['total_profit'] > 0:
                logger.info(f"SUCCESS: GENERATED ${performance['total_profit']:.2f} REAL PROFIT")
            else:
                logger.info(f"LOSS: ${abs(performance['total_profit']):.2f} - ANALYZE AND IMPROVE")
            
        except Exception as e:
            logger.error(f"Critical error in real money session: {e}")
            self.running = False

def main():
    """Main execution function"""
    # Load API credentials
    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = "qUn5JqSpYz1GDxFj2X3UF23TYgtxKrTsCbDZEoBMYCPbYZgP4siVLyspkB5HAPl7"
    
    # Create and run REAL money maker
    kimera = KimeraRealMoneyMaker(
        api_key=api_key,
        api_secret=api_secret,
        runtime_minutes=10
    )
    
    kimera.run_real_money_session()

if __name__ == "__main__":
    main() 