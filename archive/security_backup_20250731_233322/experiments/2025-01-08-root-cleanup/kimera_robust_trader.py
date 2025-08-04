#!/usr/bin/env python3
"""
KIMERA ROBUST TRADER - PRODUCTION READY
Handles all Binance trading rules and executes real trades
"""

import logging
import time
import json
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN
from binance import Client
from binance.exceptions import BinanceAPIException
import signal
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler('kimera_robust.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class KimeraRobustTrader:
    def __init__(self, api_key, api_secret):
        self.client = Client(api_key, api_secret)
        self.symbol_info = {}
        self.execution_history = []
        self.running = False
        
        # Trading parameters
        self.min_trade_value = Decimal('6')  # Minimum $6 per trade
        self.max_trade_value = Decimal('20')  # Maximum $20 per trade
        self.target_profit = Decimal('0.001')  # 0.1% profit target
        
        # High-liquidity pairs
        self.trading_pairs = ['TRXUSDT', 'DOGEUSDT', 'ADAUSDT', 'XRPUSDT']
        
        # Load symbol information
        self.load_symbol_info()
    
    def load_symbol_info(self):
        """Load trading rules for all symbols"""
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
                        'tick_size': Decimal(filters.get('PRICE_FILTER', {}).get('tickSize', '0')),
                        'base_precision': symbol_data['baseAssetPrecision'],
                        'quote_precision': symbol_data['quoteAssetPrecision']
                    }
            
            logger.info(f"Loaded trading rules for {len(self.symbol_info)} symbols")
            
        except Exception as e:
            logger.error(f"Failed to load symbol info: {e}")
    
    def get_account_balance(self, asset='USDT'):
        """Get available balance for specific asset"""
        try:
            account = self.client.get_account()
            for balance in account['balances']:
                if balance['asset'] == asset:
                    return Decimal(balance['free'])
            return Decimal('0')
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return Decimal('0')
    
    def get_current_price(self, symbol):
        """Get current market price"""
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return Decimal(ticker['price'])
        except Exception as e:
            logger.error(f"Failed to get price for {symbol}: {e}")
            return None
    
    def calculate_quantity(self, symbol, usdt_amount, price):
        """Calculate valid quantity based on trading rules"""
        try:
            if symbol not in self.symbol_info:
                logger.error(f"No symbol info for {symbol}")
                return None
            
            info = self.symbol_info[symbol]
            
            # Calculate base quantity
            quantity = usdt_amount / price
            
            # Apply step size precision
            if info['step_size'] > 0:
                steps = int(quantity / info['step_size'])
                quantity = steps * info['step_size']
            
            # Check minimum quantity
            if quantity < info['min_qty']:
                logger.warning(f"Quantity {quantity} below minimum {info['min_qty']} for {symbol}")
                return None
            
            # Check minimum notional
            notional = quantity * price
            if notional < info['min_notional']:
                logger.warning(f"Notional {notional} below minimum {info['min_notional']} for {symbol}")
                return None
            
            return quantity
            
        except Exception as e:
            logger.error(f"Error calculating quantity: {e}")
            return None
    
    def execute_buy_order(self, symbol, usdt_amount):
        """Execute market buy order"""
        try:
            current_price = self.get_current_price(symbol)
            if not current_price:
                return None
            
            quantity = self.calculate_quantity(symbol, usdt_amount, current_price)
            if not quantity:
                return None
            
            logger.info(f"EXECUTING BUY: {quantity} {symbol} (~${usdt_amount})")
            
            order = self.client.order_market_buy(
                symbol=symbol,
                quantity=f"{quantity:.8f}".rstrip('0').rstrip('.')
            )
            
            if order['status'] == 'FILLED':
                executed_qty = Decimal(order['executedQty'])
                executed_value = Decimal(order['cummulativeQuoteQty'])
                avg_price = executed_value / executed_qty
                
                trade_record = {
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'side': 'BUY',
                    'quantity': executed_qty,
                    'price': avg_price,
                    'value': executed_value,
                    'order_id': order['orderId']
                }
                
                self.execution_history.append(trade_record)
                
                logger.info(f"BUY FILLED: {executed_qty} {symbol} @ ${avg_price} = ${executed_value}")
                return trade_record
            
        except BinanceAPIException as e:
            logger.error(f"Buy order failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected buy error: {e}")
        
        return None
    
    def execute_sell_order(self, symbol, quantity):
        """Execute market sell order"""
        try:
            logger.info(f"EXECUTING SELL: {quantity} {symbol}")
            
            order = self.client.order_market_sell(
                symbol=symbol,
                quantity=f"{quantity:.8f}".rstrip('0').rstrip('.')
            )
            
            if order['status'] == 'FILLED':
                executed_qty = Decimal(order['executedQty'])
                executed_value = Decimal(order['cummulativeQuoteQty'])
                avg_price = executed_value / executed_qty
                
                trade_record = {
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'side': 'SELL',
                    'quantity': executed_qty,
                    'price': avg_price,
                    'value': executed_value,
                    'order_id': order['orderId']
                }
                
                self.execution_history.append(trade_record)
                
                logger.info(f"SELL FILLED: {executed_qty} {symbol} @ ${avg_price} = ${executed_value}")
                return trade_record
            
        except BinanceAPIException as e:
            logger.error(f"Sell order failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected sell error: {e}")
        
        return None
    
    def check_spread_opportunity(self, symbol):
        """Check if symbol has profitable spread"""
        try:
            depth = self.client.get_order_book(symbol=symbol, limit=5)
            
            if not depth['bids'] or not depth['asks']:
                return False, 0
            
            best_bid = Decimal(depth['bids'][0][0])
            best_ask = Decimal(depth['asks'][0][0])
            spread = (best_ask - best_bid) / best_bid
            
            return spread > self.target_profit, float(spread * 100)
            
        except Exception as e:
            logger.error(f"Error checking spread for {symbol}: {e}")
            return False, 0
    
    def execute_scalping_cycle(self, symbol):
        """Execute complete buy-sell scalping cycle"""
        try:
            # Check if we have enough USDT
            usdt_balance = self.get_account_balance('USDT')
            if usdt_balance < self.min_trade_value:
                logger.warning(f"Insufficient USDT: ${usdt_balance}")
                return False
            
            # Use portion of balance for this trade
            trade_amount = min(self.max_trade_value, usdt_balance * Decimal('0.5'))
            
            logger.info(f"STARTING SCALPING CYCLE: {symbol} with ${trade_amount}")
            
            # Execute buy order
            buy_trade = self.execute_buy_order(symbol, trade_amount)
            if not buy_trade:
                logger.error(f"Buy order failed for {symbol}")
                return False
            
            # Wait briefly for market movement
            time.sleep(1)
            
            # Execute sell order
            sell_trade = self.execute_sell_order(symbol, buy_trade['quantity'])
            if not sell_trade:
                logger.error(f"Sell order failed for {symbol}")
                return False
            
            # Calculate profit/loss
            profit = sell_trade['value'] - buy_trade['value']
            profit_pct = (profit / buy_trade['value']) * 100
            
            logger.info(f"SCALPING COMPLETED: {symbol} - "
                      f"Buy: ${buy_trade['price']:.6f} Sell: ${sell_trade['price']:.6f} "
                      f"Profit: ${profit:.4f} ({profit_pct:.2f}%)")
            
            return True
            
        except Exception as e:
            logger.error(f"Scalping cycle failed for {symbol}: {e}")
            return False
    
    def run_trading_session(self, runtime_minutes=10):
        """Run main trading session"""
        try:
            start_time = datetime.now()
            end_time = start_time + timedelta(minutes=runtime_minutes)
            
            logger.info("=" * 80)
            logger.info("KIMERA ROBUST TRADER - PRODUCTION READY")
            logger.info("=" * 80)
            logger.info(f"Runtime: {runtime_minutes} minutes")
            logger.info(f"Trading Pairs: {', '.join(self.trading_pairs)}")
            logger.info(f"Trade Size: ${self.min_trade_value} - ${self.max_trade_value}")
            logger.info(f"Target Profit: {float(self.target_profit * 100):.1f}%")
            
            self.running = True
            logger.info("ROBUST TRADING SESSION STARTED")
            
            cycle_count = 0
            successful_trades = 0
            
            while self.running and datetime.now() < end_time:
                cycle_count += 1
                
                # Check each trading pair for opportunities
                for symbol in self.trading_pairs:
                    if not self.running:
                        break
                    
                    has_opportunity, spread_pct = self.check_spread_opportunity(symbol)
                    
                    if has_opportunity:
                        logger.info(f"OPPORTUNITY DETECTED: {symbol} - Spread: {spread_pct:.3f}%")
                        
                        success = self.execute_scalping_cycle(symbol)
                        if success:
                            successful_trades += 1
                            logger.info(f"SUCCESSFUL TRADE #{successful_trades}")
                        
                        # Brief pause between trades
                        time.sleep(2)
                
                # Performance update every 10 cycles
                if cycle_count % 10 == 0:
                    remaining = (end_time - datetime.now()).total_seconds() / 60
                    usdt_balance = self.get_account_balance('USDT')
                    
                    logger.info(f"STATUS - Cycle: {cycle_count} | "
                              f"Successful Trades: {successful_trades} | "
                              f"USDT Balance: ${usdt_balance:.2f} | "
                              f"Time Left: {remaining:.1f}min")
                
                # Wait between cycles
                time.sleep(5)
            
            # Generate final report
            self.generate_final_report(start_time, successful_trades)
            
        except KeyboardInterrupt:
            logger.info("TRADING STOPPED BY USER")
        except Exception as e:
            logger.error(f"TRADING SESSION ERROR: {e}")
        finally:
            self.running = False
    
    def generate_final_report(self, start_time, successful_trades):
        """Generate final trading report"""
        end_time = datetime.now()
        runtime = (end_time - start_time).total_seconds()
        
        # Calculate P&L
        buy_trades = [t for t in self.execution_history if t['side'] == 'BUY']
        sell_trades = [t for t in self.execution_history if t['side'] == 'SELL']
        
        total_pnl = Decimal('0')
        for i in range(min(len(buy_trades), len(sell_trades))):
            pnl = sell_trades[i]['value'] - buy_trades[i]['value']
            total_pnl += pnl
        
        final_balance = self.get_account_balance('USDT')
        
        logger.info("\n" + "=" * 80)
        logger.info("ROBUST TRADER FINAL REPORT")
        logger.info("=" * 80)
        logger.info(f"Runtime: {runtime:.1f} seconds")
        logger.info(f"Total Executions: {len(self.execution_history)}")
        logger.info(f"Successful Cycles: {successful_trades}")
        logger.info(f"Total P&L: ${total_pnl:.4f}")
        logger.info(f"Final USDT Balance: ${final_balance:.2f}")
        logger.info(f"Success Rate: {(successful_trades / max(1, len(buy_trades)) * 100):.1f}%")
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"kimera_robust_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump({
                'summary': {
                    'runtime_seconds': runtime,
                    'total_executions': len(self.execution_history),
                    'successful_cycles': successful_trades,
                    'total_pnl': str(total_pnl),
                    'final_balance': str(final_balance)
                },
                'trades': [
                    {
                        'timestamp': trade['timestamp'].isoformat(),
                        'symbol': trade['symbol'],
                        'side': trade['side'],
                        'quantity': str(trade['quantity']),
                        'price': str(trade['price']),
                        'value': str(trade['value']),
                        'order_id': trade['order_id']
                    }
                    for trade in self.execution_history
                ]
            }, f, indent=2)
        
        logger.info(f"RESULTS SAVED: {results_file}")

def main():
    """Main execution"""
    api_key = "Y9WyflPyK1tVXnET3CTMvSdCbPia3Nhtd89VYWjS9RaAbQ0KEhHezkcGSCySQ8cL"
    api_secret = "qUn5JqSpYz1GDxFj2X3UF23TYgtxKrTsCbDZEoBMYCPbYZgP4siVLyspkB5HAPl7"
    
    # Create robust trader
    trader = KimeraRobustTrader(api_key, api_secret)
    
    # Setup emergency stop
    def emergency_handler(signum, frame):
        logger.info("EMERGENCY STOP")
        trader.running = False
        sys.exit(0)
    
    signal.signal(signal.SIGINT, emergency_handler)
    signal.signal(signal.SIGTERM, emergency_handler)
    
    # Run trading session
    trader.run_trading_session(runtime_minutes=10)

if __name__ == "__main__":
    main() 