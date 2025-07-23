#!/usr/bin/env python3
"""
KIMERA MICRO TRADER - WORKS WITH SMALL BALANCES
Optimized for trading with minimal USDT amounts
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
        logging.FileHandler('kimera_micro.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class KimeraMicroTrader:
    def __init__(self, api_key, api_secret):
        self.client = Client(api_key, api_secret)
        self.symbol_info = {}
        self.execution_history = []
        self.running = False
        
        # Micro trading parameters - work with any amount
        self.min_trade_percentage = Decimal('0.8')  # Use 80% of available balance
        self.target_profit = Decimal('0.0005')  # 0.05% profit target (very achievable)
        
        # High-volume, low-price pairs for micro trading
        self.trading_pairs = ['TRXUSDT', 'DOGEUSDT']  # Focus on 2 most liquid pairs
        
        # Load symbol information
        self.load_symbol_info()
    
    def load_symbol_info(self):
        """Load trading rules for symbols"""
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
            
            logger.info(f"Loaded trading rules for {len(self.symbol_info)} symbols")
            for symbol, info in self.symbol_info.items():
                logger.info(f"{symbol}: Min Notional: ${info['min_notional']}")
            
        except Exception as e:
            logger.error(f"Failed to load symbol info: {e}")
    
    def get_usdt_balance(self):
        """Get available USDT balance"""
        try:
            account = self.client.get_account()
            for balance in account['balances']:
                if balance['asset'] == 'USDT':
                    return Decimal(balance['free'])
            return Decimal('0')
        except Exception as e:
            logger.error(f"Failed to get USDT balance: {e}")
            return Decimal('0')
    
    def get_asset_balance(self, asset):
        """Get balance for specific asset"""
        try:
            account = self.client.get_account()
            for balance in account['balances']:
                if balance['asset'] == asset:
                    return Decimal(balance['free'])
            return Decimal('0')
        except Exception as e:
            logger.error(f"Failed to get {asset} balance: {e}")
            return Decimal('0')
    
    def get_current_price(self, symbol):
        """Get current market price"""
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return Decimal(ticker['price'])
        except Exception as e:
            logger.error(f"Failed to get price for {symbol}: {e}")
            return None
    
    def calculate_buy_quantity(self, symbol, usdt_amount):
        """Calculate maximum valid buy quantity"""
        try:
            if symbol not in self.symbol_info:
                return None
            
            current_price = self.get_current_price(symbol)
            if not current_price:
                return None
            
            info = self.symbol_info[symbol]
            
            # Calculate base quantity
            quantity = usdt_amount / current_price
            
            # Apply step size precision
            if info['step_size'] > 0:
                steps = int(quantity / info['step_size'])
                quantity = steps * info['step_size']
            
            # Check minimum requirements
            if quantity < info['min_qty']:
                return None
            
            notional = quantity * current_price
            if notional < info['min_notional']:
                return None
            
            return quantity, current_price
            
        except Exception as e:
            logger.error(f"Error calculating buy quantity: {e}")
            return None
    
    def execute_micro_buy(self, symbol):
        """Execute micro buy order using available balance"""
        try:
            usdt_balance = self.get_usdt_balance()
            if usdt_balance <= 0:
                logger.warning("No USDT balance available")
                return None
            
            # Use most of available balance
            trade_amount = usdt_balance * self.min_trade_percentage
            
            result = self.calculate_buy_quantity(symbol, trade_amount)
            if not result:
                logger.warning(f"Cannot calculate valid quantity for {symbol} with ${trade_amount}")
                return None
            
            quantity, price = result
            
            logger.info(f"EXECUTING MICRO BUY: {quantity} {symbol} @ ${price} = ${trade_amount:.2f}")
            
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
                
                logger.info(f"MICRO BUY FILLED: {executed_qty} {symbol} @ ${avg_price} = ${executed_value}")
                return trade_record
            
        except BinanceAPIException as e:
            logger.error(f"Micro buy failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected buy error: {e}")
        
        return None
    
    def execute_micro_sell(self, symbol):
        """Execute micro sell order for all available assets"""
        try:
            asset = symbol.replace('USDT', '')
            asset_balance = self.get_asset_balance(asset)
            
            if asset_balance <= 0:
                logger.warning(f"No {asset} balance to sell")
                return None
            
            logger.info(f"EXECUTING MICRO SELL: {asset_balance} {symbol}")
            
            order = self.client.order_market_sell(
                symbol=symbol,
                quantity=f"{asset_balance:.8f}".rstrip('0').rstrip('.')
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
                
                logger.info(f"MICRO SELL FILLED: {executed_qty} {symbol} @ ${avg_price} = ${executed_value}")
                return trade_record
            
        except BinanceAPIException as e:
            logger.error(f"Micro sell failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected sell error: {e}")
        
        return None
    
    def check_micro_opportunity(self, symbol):
        """Check for micro trading opportunity"""
        try:
            depth = self.client.get_order_book(symbol=symbol, limit=5)
            
            if not depth['bids'] or not depth['asks']:
                return False, 0
            
            best_bid = Decimal(depth['bids'][0][0])
            best_ask = Decimal(depth['asks'][0][0])
            spread = (best_ask - best_bid) / best_bid
            
            return spread > self.target_profit, float(spread * 100)
            
        except Exception as e:
            logger.error(f"Error checking opportunity for {symbol}: {e}")
            return False, 0
    
    def execute_micro_cycle(self, symbol):
        """Execute complete micro trading cycle"""
        try:
            logger.info(f"STARTING MICRO CYCLE: {symbol}")
            
            # Execute buy
            buy_trade = self.execute_micro_buy(symbol)
            if not buy_trade:
                return False
            
            # Brief wait for price movement
            time.sleep(2)
            
            # Execute sell
            sell_trade = self.execute_micro_sell(symbol)
            if not sell_trade:
                return False
            
            # Calculate profit
            profit = sell_trade['value'] - buy_trade['value']
            profit_pct = (profit / buy_trade['value']) * 100
            
            logger.info(f"MICRO CYCLE COMPLETED: {symbol} - "
                      f"Buy: ${buy_trade['price']:.6f} Sell: ${sell_trade['price']:.6f} "
                      f"Profit: ${profit:.4f} ({profit_pct:.3f}%)")
            
            return True
            
        except Exception as e:
            logger.error(f"Micro cycle failed: {e}")
            return False
    
    def run_micro_trading(self, runtime_minutes=10):
        """Run micro trading session"""
        try:
            start_time = datetime.now()
            end_time = start_time + timedelta(minutes=runtime_minutes)
            
            print("=" * 80)
            print("KIMERA MICRO TRADER - SMALL BALANCE OPTIMIZED")
            print("=" * 80)
            print(f"Runtime: {runtime_minutes} minutes")
            print(f"Trading Pairs: {', '.join(self.trading_pairs)}")
            print(f"Target Profit: {float(self.target_profit * 100):.2f}%")
            
            initial_balance = self.get_usdt_balance()
            print(f"Starting Balance: ${initial_balance}")
            
            self.running = True
            logger.info("MICRO TRADING SESSION STARTED")
            
            cycle_count = 0
            successful_trades = 0
            
            while self.running and datetime.now() < end_time:
                cycle_count += 1
                
                for symbol in self.trading_pairs:
                    if not self.running:
                        break
                    
                    # Check if we have enough balance to trade
                    current_balance = self.get_usdt_balance()
                    if current_balance < Decimal('1'):  # Need at least $1
                        logger.warning(f"Balance too low: ${current_balance}")
                        continue
                    
                    has_opportunity, spread_pct = self.check_micro_opportunity(symbol)
                    
                    if has_opportunity:
                        logger.info(f"MICRO OPPORTUNITY: {symbol} - Spread: {spread_pct:.3f}%")
                        
                        success = self.execute_micro_cycle(symbol)
                        if success:
                            successful_trades += 1
                            logger.info(f"SUCCESSFUL MICRO TRADE #{successful_trades}")
                        
                        # Pause between trades
                        time.sleep(3)
                
                # Status update
                if cycle_count % 5 == 0:
                    remaining = (end_time - datetime.now()).total_seconds() / 60
                    current_balance = self.get_usdt_balance()
                    
                    logger.info(f"MICRO STATUS - Cycle: {cycle_count} | "
                              f"Trades: {successful_trades} | "
                              f"Balance: ${current_balance:.4f} | "
                              f"Time: {remaining:.1f}min")
                
                time.sleep(5)
            
            # Final report
            self.generate_micro_report(start_time, initial_balance, successful_trades)
            
        except KeyboardInterrupt:
            logger.info("MICRO TRADING STOPPED")
        except Exception as e:
            logger.error(f"MICRO TRADING ERROR: {e}")
        finally:
            self.running = False
    
    def generate_micro_report(self, start_time, initial_balance, successful_trades):
        """Generate micro trading report"""
        end_time = datetime.now()
        runtime = (end_time - start_time).total_seconds()
        final_balance = self.get_usdt_balance()
        
        total_profit = final_balance - initial_balance
        profit_pct = (total_profit / initial_balance * 100) if initial_balance > 0 else 0
        
        print("\n" + "=" * 80)
        print("MICRO TRADER FINAL REPORT")
        print("=" * 80)
        print(f"Runtime: {runtime:.1f} seconds")
        print(f"Initial Balance: ${initial_balance:.4f}")
        print(f"Final Balance: ${final_balance:.4f}")
        print(f"Total Profit: ${total_profit:.4f}")
        print(f"Profit %: {profit_pct:.3f}%")
        print(f"Successful Trades: {successful_trades}")
        print(f"Total Executions: {len(self.execution_history)}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"kimera_micro_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump({
                'summary': {
                    'runtime_seconds': runtime,
                    'initial_balance': str(initial_balance),
                    'final_balance': str(final_balance),
                    'total_profit': str(total_profit),
                    'profit_percentage': float(profit_pct),
                    'successful_trades': successful_trades,
                    'total_executions': len(self.execution_history)
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
        
        logger.info(f"MICRO RESULTS SAVED: {results_file}")

def main():
    """Main execution"""
    api_key = "Y9WyflPyK1tVXnET3CTMvSdCbPia3Nhtd89VYWjS9RaAbQ0KEhHezkcGSCySQ8cL"
    api_secret = "qUn5JqSpYz1GDxFj2X3UF23TYgtxKrTsCbDZEoBMYCPbYZgP4siVLyspkB5HAPl7"
    
    # Create micro trader
    trader = KimeraMicroTrader(api_key, api_secret)
    
    # Setup emergency stop
    def emergency_handler(signum, frame):
        logger.info("EMERGENCY STOP")
        trader.running = False
        sys.exit(0)
    
    signal.signal(signal.SIGINT, emergency_handler)
    signal.signal(signal.SIGTERM, emergency_handler)
    
    # Run micro trading
    trader.run_micro_trading(runtime_minutes=10)

if __name__ == "__main__":
    main() 