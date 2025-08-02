"""
Coinbase Pro Trading Connector for KIMERA
=========================================

Real-world integration with Coinbase Pro API for live trading
Supports the $1 to infinity challenge with full order management
"""

import asyncio
import json
import time
import hmac
import hashlib
import base64
from datetime import datetime
from typing import Dict, List, Any, Optional
import aiohttp
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class CoinbaseOrder:
    order_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    amount: float
    price: float
    status: str
    created_at: datetime
    filled_amount: float = 0.0
    fees: float = 0.0

class CoinbaseProConnector:
    """
    Coinbase Pro API connector for real-world crypto trading
    Supports high-frequency trading for the $1 challenge
    """
    
    def __init__(self, api_key: str = None, api_secret: str = None, passphrase: str = None, sandbox: bool = True):
        """
        Initialize Coinbase Pro connector
        
        Args:
            api_key: Coinbase Pro API key
            api_secret: Coinbase Pro API secret
            passphrase: Coinbase Pro API passphrase
            sandbox: Use sandbox environment for testing
        """
        self.api_key = api_key or "demo_key"
        self.api_secret = api_secret or "demo_secret"
        self.passphrase = passphrase or "demo_passphrase"
        self.sandbox = sandbox
        
        # API endpoints
        if sandbox:
            self.base_url = "https://api-public.sandbox.pro.coinbase.com"
        else:
            self.base_url = "https://api.pro.coinbase.com"
        
        self.session = None
        self.active_orders = {}
        self.balance = {"USD": 1.0}  # Start with $1
        
        logger.info(f"í¿¦ Coinbase Pro Connector initialized ({'SANDBOX' if sandbox else 'LIVE'})")
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def _generate_signature(self, timestamp: str, method: str, path: str, body: str = '') -> str:
        """Generate CB-ACCESS-SIGN header"""
        message = timestamp + method + path + body
        signature = hmac.new(
            base64.b64decode(self.api_secret),
            message.encode('utf-8'),
            hashlib.sha256
        )
        return base64.b64encode(signature.digest()).decode()
    
    def _get_headers(self, method: str, path: str, body: str = '') -> Dict[str, str]:
        """Get authentication headers for Coinbase Pro API"""
        timestamp = str(time.time())
        signature = self._generate_signature(timestamp, method, path, body)
        
        return {
            'CB-ACCESS-KEY': self.api_key,
            'CB-ACCESS-SIGN': signature,
            'CB-ACCESS-TIMESTAMP': timestamp,
            'CB-ACCESS-PASSPHRASE': self.passphrase,
            'Content-Type': 'application/json'
        }
    
    async def get_account_balance(self) -> Dict[str, float]:
        """Get account balances for all currencies"""
        try:
            if self.sandbox:
                # Return mock balance for sandbox
                return self.balance
            
            path = '/accounts'
            headers = self._get_headers('GET', path)
            
            async with self.session.get(f"{self.base_url}{path}", headers=headers) as response:
                if response.status == 200:
                    accounts = await response.json()
                    balances = {}
                    for account in accounts:
                        currency = account['currency']
                        balance = float(account['available'])
                        balances[currency] = balance
                    return balances
                else:
                    logger.error(f"Failed to get balances: {response.status}")
                    return self.balance
                    
        except Exception as e:
            logger.error(f"Error getting account balance: {e}")
            return self.balance
    
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get current ticker data for a symbol"""
        try:
            path = f'/products/{symbol}/ticker'
            
            if self.sandbox:
                # Return mock ticker data
                base_prices = {
                    'BTC-USD': 43000, 'ETH-USD': 2600, 'SOL-USD': 95,
                    'DOGE-USD': 0.08, 'ADA-USD': 0.45, 'MATIC-USD': 0.85,
                    'AVAX-USD': 35, 'DOT-USD': 7.2, 'LINK-USD': 14.5
                }
                import numpy as np
                base_price = base_prices.get(symbol, 100)
                current_price = base_price * np.random.uniform(0.995, 1.005)
                
                return {
                    'price': str(current_price),
                    'size': str(np.random.uniform(0.1, 10)),
                    'bid': str(current_price * 0.999),
                    'ask': str(current_price * 1.001),
                    'volume': str(np.random.uniform(1000, 10000)),
                    'time': datetime.now().isoformat()
                }
            
            async with self.session.get(f"{self.base_url}{path}") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to get ticker for {symbol}: {response.status}")
                    return {}
                    
        except Exception as e:
            logger.error(f"Error getting ticker for {symbol}: {e}")
            return {}
    
    async def place_market_order(self, symbol: str, side: str, amount: float) -> Optional[CoinbaseOrder]:
        """
        Place a market order
        
        Args:
            symbol: Trading pair (e.g., 'BTC-USD')
            side: 'buy' or 'sell'
            amount: Amount to trade (in quote currency for buy, base currency for sell)
        """
        try:
            order_data = {
                'type': 'market',
                'side': side,
                'product_id': symbol,
            }
            
            if side == 'buy':
                order_data['funds'] = str(amount)  # USD amount for buying
            else:
                order_data['size'] = str(amount)   # Crypto amount for selling
            
            if self.sandbox:
                # Simulate order execution
                ticker = await self.get_ticker(symbol)
                price = float(ticker.get('price', 100))
                
                order_id = f"sim_{int(time.time() * 1000)}"
                order = CoinbaseOrder(
                    order_id=order_id,
                    symbol=symbol,
                    side=side,
                    amount=amount,
                    price=price,
                    status='filled',
                    created_at=datetime.now(),
                    filled_amount=amount,
                    fees=amount * 0.005  # 0.5% fee
                )
                
                # Update mock balance
                if side == 'buy':
                    self.balance['USD'] -= amount + order.fees
                    crypto_symbol = symbol.split('-')[0]
                    crypto_amount = amount / price
                    self.balance[crypto_symbol] = self.balance.get(crypto_symbol, 0) + crypto_amount
                else:
                    crypto_symbol = symbol.split('-')[0]
                    self.balance[crypto_symbol] -= amount
                    usd_received = amount * price - order.fees
                    self.balance['USD'] += usd_received
                
                self.active_orders[order_id] = order
                
                logger.info(f"í´„ SIMULATED ORDER: {side.upper()} {amount:.4f} {symbol} @ ${price:.4f}")
                return order
            
            # Real API call
            path = '/orders'
            body = json.dumps(order_data)
            headers = self._get_headers('POST', path, body)
            
            async with self.session.post(f"{self.base_url}{path}", data=body, headers=headers) as response:
                if response.status == 200:
                    order_response = await response.json()
                    order = CoinbaseOrder(
                        order_id=order_response['id'],
                        symbol=symbol,
                        side=side,
                        amount=amount,
                        price=float(order_response.get('price', 0)),
                        status=order_response['status'],
                        created_at=datetime.fromisoformat(order_response['created_at'].replace('Z', '+00:00')),
                        filled_amount=float(order_response.get('filled_size', 0)),
                        fees=float(order_response.get('fill_fees', 0))
                    )
                    
                    self.active_orders[order.order_id] = order
                    return order
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to place order: {response.status} - {error_text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error placing market order: {e}")
            return None
    
    async def get_order_status(self, order_id: str) -> Optional[CoinbaseOrder]:
        """Get the status of an order"""
        try:
            if order_id in self.active_orders:
                return self.active_orders[order_id]
            
            if self.sandbox:
                return None
            
            path = f'/orders/{order_id}'
            headers = self._get_headers('GET', path)
            
            async with self.session.get(f"{self.base_url}{path}", headers=headers) as response:
                if response.status == 200:
                    order_data = await response.json()
                    order = CoinbaseOrder(
                        order_id=order_data['id'],
                        symbol=order_data['product_id'],
                        side=order_data['side'],
                        amount=float(order_data['size']),
                        price=float(order_data.get('price', 0)),
                        status=order_data['status'],
                        created_at=datetime.fromisoformat(order_data['created_at'].replace('Z', '+00:00')),
                        filled_amount=float(order_data.get('filled_size', 0)),
                        fees=float(order_data.get('fill_fees', 0))
                    )
                    return order
                else:
                    logger.error(f"Failed to get order status: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting order status: {e}")
            return None
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        try:
            if self.sandbox:
                if order_id in self.active_orders:
                    self.active_orders[order_id].status = 'cancelled'
                    return True
                return False
            
            path = f'/orders/{order_id}'
            headers = self._get_headers('DELETE', path)
            
            async with self.session.delete(f"{self.base_url}{path}", headers=headers) as response:
                return response.status == 200
                
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False
    
    async def get_available_pairs(self) -> List[str]:
        """Get all available trading pairs"""
        try:
            if self.sandbox:
                return [
                    'BTC-USD', 'ETH-USD', 'SOL-USD', 'DOGE-USD', 'ADA-USD',
                    'MATIC-USD', 'AVAX-USD', 'DOT-USD', 'LINK-USD'
                ]
            
            path = '/products'
            async with self.session.get(f"{self.base_url}{path}") as response:
                if response.status == 200:
                    products = await response.json()
                    return [product['id'] for product in products if product['status'] == 'online']
                else:
                    logger.error(f"Failed to get products: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error getting available pairs: {e}")
            return []
    
    async def get_order_book(self, symbol: str, level: int = 1) -> Dict[str, Any]:
        """Get order book for a symbol"""
        try:
            path = f'/products/{symbol}/book'
            params = {'level': level}
            
            if self.sandbox:
                # Mock order book
                ticker = await self.get_ticker(symbol)
                price = float(ticker.get('price', 100))
                
                return {
                    'sequence': int(time.time()),
                    'bids': [[str(price * 0.999), str(10.0), 1]],
                    'asks': [[str(price * 1.001), str(10.0), 1]]
                }
            
            async with self.session.get(f"{self.base_url}{path}", params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to get order book: {response.status}")
                    return {}
                    
        except Exception as e:
            logger.error(f"Error getting order book: {e}")
            return {}

def create_coinbase_pro_connector(api_key: str = None, api_secret: str = None, 
                                 passphrase: str = None, sandbox: bool = True) -> CoinbaseProConnector:
    """
    Factory function to create a Coinbase Pro connector
    
    Args:
        api_key: Coinbase Pro API key
        api_secret: Coinbase Pro API secret  
        passphrase: Coinbase Pro API passphrase
        sandbox: Use sandbox environment
        
    Returns:
        CoinbaseProConnector instance
    """
    return CoinbaseProConnector(api_key, api_secret, passphrase, sandbox)

async def demo_coinbase_trading():
    """Demo function showing Coinbase Pro integration"""
    logger.info("í¿¦ COINBASE PRO CONNECTOR DEMO")
    logger.info("=" * 40)
    
    async with create_coinbase_pro_connector(sandbox=True) as connector:
        # Get account balance
        balance = await connector.get_account_balance()
        logger.info(f"í²° Account Balance: {balance}")
        
        # Get ticker for BTC-USD
        ticker = await connector.get_ticker('BTC-USD')
        logger.info(f"í³Š BTC-USD Ticker: ${float(ticker['price']):.2f}")
        
        # Place a small buy order
        order = await connector.place_market_order('BTC-USD', 'buy', 0.50)  # $0.50 buy
        if order:
            logger.info(f"íº€ Order Placed: {order.order_id}")
            logger.info(f"   Side: {order.side}")
            logger.info(f"   Amount: ${order.amount:.4f}")
            logger.info(f"   Price: ${order.price:.2f}")
            logger.info(f"   Status: {order.status}")
        
        # Check updated balance
        new_balance = await connector.get_account_balance()
        logger.info(f"í²° Updated Balance: {new_balance}")

if __name__ == "__main__":
    asyncio.run(demo_coinbase_trading())
