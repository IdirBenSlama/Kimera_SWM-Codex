#!/usr/bin/env python3
"""
REAL COINBASE PRO TRADER
Actual Coinbase Pro API integration with real trading
"""

import asyncio
import json
import time
import hmac
import hashlib
import base64
import requests
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CoinbaseProAPI:
    """Real Coinbase Pro API client"""
    
    def __init__(self, api_key: str, api_secret: str, passphrase: str, sandbox: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        
        # Use sandbox for testing, production for real trading
        if sandbox:
            self.base_url = "https://api-public.sandbox.pro.coinbase.com"
        else:
            self.base_url = "https://api.pro.coinbase.com"
        
        logger.info(f"Coinbase Pro API initialized ({'SANDBOX' if sandbox else 'PRODUCTION'})")
    
    def _generate_signature(self, timestamp: str, method: str, path: str, body: str = '') -> str:
        """Generate API signature"""
        message = timestamp + method + path + body
        signature = hmac.new(
            base64.b64decode(self.api_secret),
            message.encode('utf-8'),
            hashlib.sha256
        ).digest()
        return base64.b64encode(signature).decode('utf-8')
    
    def _make_request(self, method: str, path: str, params: Dict = None, data: Dict = None) -> Dict:
        """Make authenticated API request"""
        timestamp = str(time.time())
        body = json.dumps(data) if data else ''
        
        headers = {
            'CB-ACCESS-KEY': self.api_key,
            'CB-ACCESS-SIGN': self._generate_signature(timestamp, method, path, body),
            'CB-ACCESS-TIMESTAMP': timestamp,
            'CB-ACCESS-PASSPHRASE': self.passphrase,
            'Content-Type': 'application/json'
        }
        
        url = self.base_url + path
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, params=params, timeout=30)
            elif method == 'POST':
                response = requests.post(url, headers=headers, json=data, timeout=30)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"API Error {response.status_code}: {response.text}")
                return {}
                
        except Exception as e:
            logger.error(f"Request error: {e}")
            return {}
    
    def get_accounts(self) -> List[Dict]:
        """Get account balances"""
        return self._make_request('GET', '/accounts')
    
    def get_products(self) -> List[Dict]:
        """Get available trading products"""
        return self._make_request('GET', '/products')
    
    def get_ticker(self, product_id: str) -> Dict:
        """Get ticker for product"""
        return self._make_request('GET', f'/products/{product_id}/ticker')
    
    def get_order_book(self, product_id: str, level: int = 1) -> Dict:
        """Get order book"""
        return self._make_request('GET', f'/products/{product_id}/book', {'level': level})
    
    def place_order(self, product_id: str, side: str, order_type: str, size: float = None, funds: float = None, price: float = None) -> Dict:
        """Place order"""
        order_data = {
            'product_id': product_id,
            'side': side,  # 'buy' or 'sell'
            'type': order_type  # 'market', 'limit'
        }
        
        if order_type == 'market':
            if side == 'buy':
                order_data['funds'] = str(funds)  # USD amount for buy
            else:
                order_data['size'] = str(size)   # Crypto amount for sell
        elif order_type == 'limit':
            order_data['size'] = str(size)
            order_data['price'] = str(price)
        
        return self._make_request('POST', '/orders', data=order_data)
    
    def get_orders(self, status: str = 'open') -> List[Dict]:
        """Get orders"""
        return self._make_request('GET', '/orders', {'status': status})
    
    def cancel_order(self, order_id: str) -> Dict:
        """Cancel order"""
        return self._make_request('DELETE', f'/orders/{order_id}')

class RealCoinbaseTrader:
    """Real Coinbase Pro trading system"""
    
    def __init__(self, api_key: str, api_secret: str, passphrase: str, starting_balance: float = 1.0):
        self.api = CoinbaseProAPI(api_key, api_secret, passphrase, sandbox=True)  # Start with sandbox
        self.starting_balance = starting_balance
        self.session_start = datetime.now()
        self.session_end = self.session_start + timedelta(hours=6)
        
        # Trading pairs (Coinbase Pro format)
        self.trading_pairs = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'MATIC-USD']
        
        # Risk management
        self.max_trade_amount = 0.50  # Max $0.50 per trade
        self.min_trade_amount = 0.10  # Min $0.10 per trade
        self.max_positions = 3
        
        # Track positions and trades
        self.positions = {}
        self.trades = []
        
        logger.info("Real Coinbase Pro Trader initialized")
        logger.info(f"Session: {self.session_start.strftime('%H:%M')} - {self.session_end.strftime('%H:%M')}")
    
    async def get_account_balance(self) -> float:
        """Get USD account balance"""
        accounts = self.api.get_accounts()
        
        for account in accounts:
            if account.get('currency') == 'USD':
                return float(account.get('balance', 0))
        
        return 0.0
    
    async def get_market_data(self) -> Dict[str, Dict]:
        """Get real market data from Coinbase"""
        market_data = {}
        
        for pair in self.trading_pairs:
            ticker = self.api.get_ticker(pair)
            
            if ticker and 'price' in ticker:
                asset = pair.split('-')[0].lower()
                market_data[asset] = {
                    'pair': pair,
                    'price': float(ticker['price']),
                    'bid': float(ticker.get('bid', ticker['price'])),
                    'ask': float(ticker.get('ask', ticker['price'])),
                    'volume': float(ticker.get('volume', 0))
                }
        
        return market_data
    
    def analyze_market_opportunity(self, market_data: Dict) -> tuple:
        """Analyze market for trading opportunities"""
        best_asset = None
        best_action = "hold"
        best_score = 0.0
        
        for asset, data in market_data.items():
            price = data['price']
            volume = data['volume']
            
            # Simple momentum analysis
            score = 0.0
            action = "hold"
            
            # Volume-based scoring
            if volume > 1000:  # High volume
                score += 0.3
            
            # Price action analysis (simplified)
            # In real implementation, you'd use historical data
            spread = (data['ask'] - data['bid']) / data['bid'] * 100
            
            if spread < 0.1:  # Tight spread = good liquidity
                score += 0.2
            
            # Random factor for demonstration
            import random
            momentum_score = random.uniform(0, 0.5)
            score += momentum_score
            
            # Determine action
            if score > 0.6 and len(self.positions) < self.max_positions:
                action = "buy"
            elif asset in self.positions and score > 0.7:
                action = "sell"
            
            if score > best_score:
                best_score = score
                best_asset = asset
                best_action = action
        
        return best_asset, best_action, best_score
    
    async def execute_buy_order(self, asset: str, usd_amount: float) -> bool:
        """Execute real buy order"""
        pair = f"{asset.upper()}-USD"
        
        try:
            # Place market buy order
            result = self.api.place_order(
                product_id=pair,
                side='buy',
                order_type='market',
                funds=usd_amount
            )
            
            if result and 'id' in result:
                order_id = result['id']
                
                # Wait for order to fill
                await asyncio.sleep(2)
                
                # Check order status
                orders = self.api.get_orders(status='done')
                
                for order in orders:
                    if order.get('id') == order_id:
                        filled_size = float(order.get('filled_size', 0))
                        executed_value = float(order.get('executed_value', 0))
                        
                        if filled_size > 0:
                            # Record position
                            if asset in self.positions:
                                # Average position
                                existing = self.positions[asset]
                                total_size = existing['size'] + filled_size
                                avg_price = (existing['size'] * existing['price'] + executed_value) / total_size
                                
                                self.positions[asset] = {
                                    'size': total_size,
                                    'price': avg_price,
                                    'entry_time': existing['entry_time']
                                }
                            else:
                                self.positions[asset] = {
                                    'size': filled_size,
                                    'price': executed_value / filled_size,
                                    'entry_time': datetime.now()
                                }
                            
                            # Record trade
                            self.trades.append({
                                'action': 'buy',
                                'asset': asset,
                                'size': filled_size,
                                'price': executed_value / filled_size,
                                'usd_value': executed_value,
                                'time': datetime.now(),
                                'order_id': order_id
                            })
                            
                            logger.info(f"BUY EXECUTED: {filled_size:.6f} {asset.upper()} for ${executed_value:.4f}")
                            return True
            
            logger.warning(f"Buy order failed: {result}")
            return False
            
        except Exception as e:
            logger.error(f"Buy order error: {e}")
            return False
    
    async def execute_sell_order(self, asset: str, percentage: float) -> bool:
        """Execute real sell order"""
        if asset not in self.positions:
            return False
        
        pair = f"{asset.upper()}-USD"
        position = self.positions[asset]
        sell_size = position['size'] * (percentage / 100)
        
        try:
            # Place market sell order
            result = self.api.place_order(
                product_id=pair,
                side='sell',
                order_type='market',
                size=sell_size
            )
            
            if result and 'id' in result:
                order_id = result['id']
                
                # Wait for order to fill
                await asyncio.sleep(2)
                
                # Check order status
                orders = self.api.get_orders(status='done')
                
                for order in orders:
                    if order.get('id') == order_id:
                        filled_size = float(order.get('filled_size', 0))
                        executed_value = float(order.get('executed_value', 0))
                        
                        if filled_size > 0:
                            # Calculate profit
                            avg_sell_price = executed_value / filled_size
                            profit = (avg_sell_price - position['price']) * filled_size
                            
                            # Update position
                            if percentage >= 99:  # Sell all
                                del self.positions[asset]
                            else:
                                self.positions[asset]['size'] -= filled_size
                            
                            # Record trade
                            self.trades.append({
                                'action': 'sell',
                                'asset': asset,
                                'size': filled_size,
                                'price': avg_sell_price,
                                'usd_value': executed_value,
                                'profit': profit,
                                'time': datetime.now(),
                                'order_id': order_id
                            })
                            
                            logger.info(f"SELL EXECUTED: {filled_size:.6f} {asset.upper()} for ${executed_value:.4f} | Profit: ${profit:+.4f}")
                            return True
            
            logger.warning(f"Sell order failed: {result}")
            return False
            
        except Exception as e:
            logger.error(f"Sell order error: {e}")
            return False
    
    async def trading_cycle(self) -> bool:
        """Execute one trading cycle"""
        if datetime.now() >= self.session_end:
            return False
        
        try:
            # Get current balance
            balance = await self.get_account_balance()
            
            # Get market data
            market_data = await self.get_market_data()
            if not market_data:
                logger.warning("No market data available")
                return True
            
            # Analyze opportunities
            best_asset, action, score = self.analyze_market_opportunity(market_data)
            
            if action == "buy" and score > 0.6 and balance > self.min_trade_amount:
                # Calculate trade size
                trade_amount = min(
                    balance * (0.2 + 0.3 * score),  # 20-50% of balance based on confidence
                    self.max_trade_amount
                )
                trade_amount = max(trade_amount, self.min_trade_amount)
                
                if trade_amount <= balance:
                    logger.info(f"Attempting BUY: ${trade_amount:.4f} of {best_asset.upper()} (confidence: {score:.3f})")
                    await self.execute_buy_order(best_asset, trade_amount)
            
            elif action == "sell" and score > 0.7 and best_asset in self.positions:
                # Sell based on confidence
                sell_percentage = 50 + (50 * score)  # 50-100%
                logger.info(f"Attempting SELL: {sell_percentage:.1f}% of {best_asset.upper()} (confidence: {score:.3f})")
                await self.execute_sell_order(best_asset, sell_percentage)
            
            return True
            
        except Exception as e:
            logger.error(f"Trading cycle error: {e}")
            return True
    
    async def generate_report(self) -> Dict:
        """Generate comprehensive trading report"""
        balance = await self.get_account_balance()
        market_data = await self.get_market_data()
        
        # Calculate position values
        total_position_value = 0
        position_details = {}
        
        for asset, position in self.positions.items():
            if asset in market_data:
                current_price = market_data[asset]['price']
                current_value = position['size'] * current_price
                profit_loss = (current_price - position['price']) * position['size']
                profit_pct = (current_price / position['price'] - 1) * 100
                
                total_position_value += current_value
                position_details[asset] = {
                    'size': position['size'],
                    'entry_price': position['price'],
                    'current_price': current_price,
                    'current_value': current_value,
                    'profit_loss': profit_loss,
                    'profit_pct': profit_pct
                }
        
        total_value = balance + total_position_value
        total_return = (total_value / self.starting_balance - 1) * 100
        
        # Trading statistics
        buy_trades = [t for t in self.trades if t['action'] == 'buy']
        sell_trades = [t for t in self.trades if t['action'] == 'sell']
        total_profit = sum(t.get('profit', 0) for t in sell_trades)
        
        duration = (datetime.now() - self.session_start).total_seconds() / 3600
        
        return {
            'session': {
                'start_time': self.session_start.isoformat(),
                'duration_hours': round(duration, 2),
                'starting_balance': self.starting_balance,
                'current_balance': round(balance, 6),
                'position_value': round(total_position_value, 6),
                'total_value': round(total_value, 6),
                'total_return_pct': round(total_return, 2),
                'realized_profit': round(total_profit, 6)
            },
            'trading': {
                'total_trades': len(self.trades),
                'buy_trades': len(buy_trades),
                'sell_trades': len(sell_trades),
                'active_positions': len(self.positions)
            },
            'positions': position_details,
            'recent_trades': self.trades[-5:] if self.trades else []
        }

async def run_real_coinbase_session():
    """Run real Coinbase Pro trading session"""
    # Your actual Coinbase Pro API credentials
    API_KEY = "f7360d36-8068-4b75-8169-6d016b96d810"
    API_SECRET = "BiCUFOxZ4J4Fi8F6mcyzuzreXaGZeBLHxr7q8Puo6VHcSEgyqJ6mIx29RbbAJGAjq6SHBt5K4PieiymRhEWVHw=="
    PASSPHRASE = "your_passphrase_here"  # You need to provide this
    
    logger.info("🚀 REAL COINBASE PRO TRADER")
    logger.info("=" * 50)
    logger.warning("⚠️  REAL MONEY TRADING - PROCEED WITH CAUTION")
    logger.info("Starting Balance: $1.00 | Duration: 6 hours")
    logger.info("=" * 50)
    
    # Note: This requires a valid passphrase which wasn't provided
    # For demonstration, we'll show the structure
    
    try:
        trader = RealCoinbaseTrader(API_KEY, API_SECRET, PASSPHRASE, 1.0)
        last_report = time.time()
        
        while True:
            # Execute trading
            if not await trader.trading_cycle():
                break
            
            # Periodic reports
            if time.time() - last_report > 1800:  # Every 30 minutes
                report = await trader.generate_report()
                
                logger.info(f"\n⏰ HOUR {report['session']['duration_hours']:.1f}")
                logger.info(f"💰 Total Value: ${report['session']['total_value']:.6f}")
                logger.info(f"💵 Cash Balance: ${report['session']['current_balance']:.6f}")
                logger.info(f"📈 Return: {report['session']['total_return_pct']:+.2f}%")
                logger.info(f"🔄 Trades: {report['trading']['total_trades']}")
                logger.info(f"📊 Positions: {report['trading']['active_positions']}")
                
                if report['positions']:
                    logger.info("🎯 Active Positions:")
                    for asset, pos in report['positions'].items():
                        logger.info(f"   {asset.upper()
                
                last_report = time.time()
            
            # Wait between cycles
            await asyncio.sleep(60)  # 1 minute between cycles
    
    except KeyboardInterrupt:
        logger.info("\n🛑 Session stopped by user")
    
    except Exception as e:
        logger.error(f"Session error: {e}")
        logger.error(f"\n❌ Session error: {e}")
    
    finally:
        logger.info("\n🏁 Real Coinbase Pro session ended")

if __name__ == "__main__":
    logger.warning("⚠️  WARNING: This requires valid Coinbase Pro API credentials")
    logger.warning("⚠️  Including API Key, Secret, and Passphrase")
    logger.warning("⚠️  Currently configured for SANDBOX mode")
    logger.info("\nTo run with real money:")
    logger.info("1. Add your passphrase")
    logger.info("2. Change sandbox=False in CoinbaseProAPI init")
    logger.info("3. Ensure you have proper risk management")
    
    # Uncomment to run (requires valid passphrase)
    # asyncio.run(run_real_coinbase_session()) 