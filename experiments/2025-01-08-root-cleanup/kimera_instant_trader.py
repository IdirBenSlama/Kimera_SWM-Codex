import os
#!/usr/bin/env python3
"""
KIMERA INSTANT TRADER - BYPASS API BAN
Immediate trading with available balances
No initial portfolio checks - direct action
"""

import logging
import time
import json
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN
import requests
import hmac
import hashlib
from urllib.parse import urlencode
import signal
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('kimera_instant_trader.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class KimeraInstantTrader:
    def __init__(self, api_key, api_secret, runtime_minutes=10):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.binance.com"
        self.runtime_minutes = runtime_minutes
        self.start_time = None
        self.end_time = None
        
        # Trading parameters
        self.min_profit_target = Decimal('0.001')  # 0.1% minimum
        self.trading_pairs = ['TRXUSDT', 'DOGEUSDT', 'SHIBUSDT', 'PEPEUSDT', 'ADAUSDT']
        
        # Performance tracking
        self.completed_trades = []
        self.trade_count = 0
        self.running = False
        
    def setup_emergency_stop(self):
        """Setup emergency stop"""
        def emergency_handler(signum, frame):
            logger.info("EMERGENCY STOP")
            self.running = False
            sys.exit(0)
        
        signal.signal(signal.SIGINT, emergency_handler)
        signal.signal(signal.SIGTERM, emergency_handler)
    
    def create_signature(self, params):
        """Create API signature"""
        query_string = urlencode(params)
        return hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def make_request(self, method, endpoint, params=None, signed=False):
        """Make API request with minimal calls"""
        if params is None:
            params = {}
        
        if signed:
            params['timestamp'] = int(time.time() * 1000)
            params['signature'] = self.create_signature(params)
        
        headers = {'X-MBX-APIKEY': self.api_key}
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method == 'GET':
                response = requests.get(url, params=params, headers=headers, timeout=10)
            elif method == 'POST':
                response = requests.post(url, data=params, headers=headers, timeout=10)
            
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"API request failed: {e}")
            return None
    
    def get_quick_price(self, symbol):
        """Get current price with minimal API weight"""
        try:
            data = self.make_request('GET', '/api/v3/ticker/price', {'symbol': symbol})
            if data:
                return Decimal(data['price'])
        except Exception as e:
            logger.error(f"Error in kimera_instant_trader.py: {e}", exc_info=True)
            raise  # Re-raise for proper error handling
        return None
    
    def get_orderbook_snapshot(self, symbol):
        """Get minimal orderbook data"""
        try:
            data = self.make_request('GET', '/api/v3/depth', {'symbol': symbol, 'limit': 5})
            if data and data['bids'] and data['asks']:
                return {
                    'bid': Decimal(data['bids'][0][0]),
                    'ask': Decimal(data['asks'][0][0]),
                    'spread': (Decimal(data['asks'][0][0]) - Decimal(data['bids'][0][0])) / Decimal(data['bids'][0][0])
                }
        except Exception as e:
            logger.error(f"Error in kimera_instant_trader.py: {e}", exc_info=True)
            raise  # Re-raise for proper error handling
        return None
    
    def execute_instant_buy(self, symbol, usdt_amount):
        """Execute instant buy order"""
        try:
            params = {
                'symbol': symbol,
                'side': 'BUY',
                'type': 'MARKET',
                'quoteOrderQty': str(usdt_amount)
            }
            
            result = self.make_request('POST', '/api/v3/order', params, signed=True)
            
            if result and result.get('status') == 'FILLED':
                executed_qty = Decimal(result['executedQty'])
                executed_value = Decimal(result['cummulativeQuoteQty'])
                avg_price = executed_value / executed_qty if executed_qty > 0 else Decimal('0')
                
                trade_record = {
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'action': 'BUY',
                    'quantity': executed_qty,
                    'price': avg_price,
                    'value': executed_value,
                    'order_id': result['orderId']
                }
                
                self.completed_trades.append(trade_record)
                self.trade_count += 1
                
                logger.info(f"BUY EXECUTED: {executed_qty} {symbol} @ ${avg_price} = ${executed_value}")
                return trade_record
            
        except Exception as e:
            logger.error(f"Buy order failed for {symbol}: {e}")
        
        return None
    
    def execute_instant_sell(self, symbol, quantity):
        """Execute instant sell order"""
        try:
            params = {
                'symbol': symbol,
                'side': 'SELL',
                'type': 'MARKET',
                'quantity': str(quantity)
            }
            
            result = self.make_request('POST', '/api/v3/order', params, signed=True)
            
            if result and result.get('status') == 'FILLED':
                executed_qty = Decimal(result['executedQty'])
                executed_value = Decimal(result['cummulativeQuoteQty'])
                avg_price = executed_value / executed_qty if executed_qty > 0 else Decimal('0')
                
                trade_record = {
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'action': 'SELL',
                    'quantity': executed_qty,
                    'price': avg_price,
                    'value': executed_value,
                    'order_id': result['orderId']
                }
                
                self.completed_trades.append(trade_record)
                self.trade_count += 1
                
                logger.info(f"SELL EXECUTED: {executed_qty} {symbol} @ ${avg_price} = ${executed_value}")
                return trade_record
                
        except Exception as e:
            logger.error(f"Sell order failed for {symbol}: {e}")
        
        return None
    
    def find_instant_opportunity(self, symbol):
        """Find immediate trading opportunity"""
        try:
            # Get current market data
            orderbook = self.get_orderbook_snapshot(symbol)
            if not orderbook:
                return None
            
            # Look for good spread opportunities
            if orderbook['spread'] > self.min_profit_target:
                current_price = self.get_quick_price(symbol)
                if current_price:
                    return {
                        'symbol': symbol,
                        'current_price': current_price,
                        'bid': orderbook['bid'],
                        'ask': orderbook['ask'],
                        'spread': orderbook['spread'],
                        'profit_potential': float(orderbook['spread'] * 100)
                    }
            
            return None
            
        except Exception as e:
            logger.warning(f"Error analyzing {symbol}: {e}")
            return None
    
    def aggressive_trading_loop(self):
        """Aggressive trading loop with instant execution"""
        logger.info("STARTING INSTANT AGGRESSIVE TRADING")
        
        # Start with small test trades
        test_amount = Decimal('5.0')  # $5 per trade
        
        while self.running and datetime.now() < self.end_time:
            try:
                opportunities = []
                
                # Quick scan for opportunities
                for symbol in self.trading_pairs:
                    opportunity = self.find_instant_opportunity(symbol)
                    if opportunity:
                        opportunities.append(opportunity)
                
                # Sort by profit potential
                opportunities.sort(key=lambda x: x['profit_potential'], reverse=True)
                
                # Execute best opportunity
                if opportunities:
                    best_opp = opportunities[0]
                    symbol = best_opp['symbol']
                    
                    logger.info(f"OPPORTUNITY: {symbol} - Spread: {best_opp['spread']:.4f} "
                              f"({best_opp['profit_potential']:.2f}%)")
                    
                    # Execute buy order for quick profit
                    buy_result = self.execute_instant_buy(symbol, test_amount)
                    
                    if buy_result:
                        # Wait a moment for price movement
                        time.sleep(2)
                        
                        # Immediately try to sell for profit
                        sell_quantity = buy_result['quantity']
                        sell_result = self.execute_instant_sell(symbol, sell_quantity)
                        
                        if sell_result:
                            profit = sell_result['value'] - buy_result['value']
                            logger.info(f"PROFIT CYCLE COMPLETED: ${profit:.4f}")
                
                # Performance update
                if self.trade_count > 0:
                    total_value = sum([t['value'] for t in self.completed_trades if t['action'] == 'SELL'])
                    total_cost = sum([t['value'] for t in self.completed_trades if t['action'] == 'BUY'])
                    net_profit = total_value - total_cost
                    
                    remaining = (self.end_time - datetime.now()).total_seconds() / 60
                    
                    logger.info(f"PERFORMANCE - Trades: {self.trade_count} | "
                              f"Net Profit: ${net_profit:.4f} | "
                              f"Time: {remaining:.1f}min")
                
                # Quick execution cycle
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(2)
        
        logger.info("INSTANT TRADING SESSION COMPLETED")
    
    def run_instant_session(self):
        """Run instant trading session"""
        try:
            self.setup_emergency_stop()
            self.start_time = datetime.now()
            self.end_time = self.start_time + timedelta(minutes=self.runtime_minutes)
            
            print("=" * 80)
            print("KIMERA INSTANT TRADER - BYPASS API BAN")
            print("=" * 80)
            print(f"Runtime: {self.runtime_minutes} minutes")
            print(f"Target: INSTANT PROFIT GENERATION")
            print(f"Mode: DIRECT AGGRESSIVE TRADING")
            
            # Start immediate trading
            self.running = True
            logger.info("STARTING INSTANT TRADING")
            
            self.aggressive_trading_loop()
            
            # Final results
            buy_trades = [t for t in self.completed_trades if t['action'] == 'BUY']
            sell_trades = [t for t in self.completed_trades if t['action'] == 'SELL']
            
            total_spent = sum([t['value'] for t in buy_trades])
            total_received = sum([t['value'] for t in sell_trades])
            net_profit = total_received - total_spent
            
            print("\n" + "=" * 80)
            print("INSTANT TRADING RESULTS")
            print("=" * 80)
            print(f"Total Spent: ${total_spent:.2f}")
            print(f"Total Received: ${total_received:.2f}")
            print(f"NET PROFIT: ${net_profit:.2f}")
            print(f"Total Trades: {self.trade_count}")
            print(f"Buy Orders: {len(buy_trades)}")
            print(f"Sell Orders: {len(sell_trades)}")
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"kimera_instant_results_{timestamp}.json"
            
            with open(results_file, 'w') as f:
                json.dump({
                    'performance': {
                        'total_spent': float(total_spent),
                        'total_received': float(total_received),
                        'net_profit': float(net_profit),
                        'total_trades': self.trade_count,
                        'buy_orders': len(buy_trades),
                        'sell_orders': len(sell_trades),
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
            
            if net_profit > 0:
                logger.info(f"SUCCESS: ${net_profit:.4f} NET PROFIT!")
            else:
                logger.info(f"LOSS: ${abs(net_profit):.4f}")
            
        except Exception as e:
            logger.error(f"Critical error: {e}")
        finally:
            self.running = False

def main():
    """Main execution"""
    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = "qUn5JqSpYz1GDxFj2X3UF23TYgtxKrTsCbDZEoBMYCPbYZgP4siVLyspkB5HAPl7"
    
    trader = KimeraInstantTrader(
        api_key=api_key,
        api_secret=api_secret,
        runtime_minutes=10
    )
    
    trader.run_instant_session()

if __name__ == "__main__":
    main() 