"""
Binance API Connector (HMAC Implementation)

Handles all interactions with Binance exchange for crypto trading.
This version uses traditional HMAC-SHA256 authentication.
"""

import asyncio
import logging
import time
import hashlib
import hmac
from typing import Dict, List, Optional, Any, Callable
from urllib.parse import urlencode
import aiohttp
import websockets
import json

logger = logging.getLogger(__name__)


class BinanceConnector:
    """
    Binance exchange connector with WebSocket support for real-time data.
    Uses HMAC-SHA256 for signing requests.
    """
    
    BASE_URL = "https://api.binance.com"
    WS_BASE_URL = "wss://stream.binance.com:9443/ws"
    
    def __init__(self, api_key: str, secret_key: str, testnet: bool = False):
        """
        Initialize Binance connector with HMAC authentication.
        
        Args:
            api_key: Your Binance API Key.
            secret_key: Your Binance Secret Key.
            testnet: Use testnet if True.
        """
        if not api_key:
            raise ValueError("API key is required.")
        if not secret_key:
            raise ValueError("Secret key is required.")

        self.api_key = api_key
        self.secret_key = secret_key
        
        if testnet:
            self.BASE_URL = "https://testnet.binance.vision"
            self.WS_BASE_URL = "wss://testnet.binance.vision/ws"
            
        self.session: Optional[aiohttp.ClientSession] = None
        self.ws_connections: Dict[str, websockets.WebSocketClientProtocol] = {}
        self.callbacks: Dict[str, List[Callable]] = {}
        
        logger.info(f"Binance connector initialized with HMAC auth (testnet={testnet})")

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def close(self):
        """Close all connections"""
        for ws in self.ws_connections.values():
            await ws.close()
        self.ws_connections.clear()
        
        if self.session and not self.session.closed:
            await self.session.close()
    
    def _sign_request(self, params: Dict[str, Any]) -> str:
        """Sign request parameters with HMAC-SHA256."""
        # Remove signature from params if it exists to avoid double signing
        sign_params = {k: v for k, v in params.items() if k != 'signature'}
        query_string = urlencode(sign_params, safe='~')
        return hmac.new(
            self.secret_key.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    async def _request(
        self, 
        method: str, 
        endpoint: str, 
        params: Optional[Dict[str, Any]] = None,
        signed: bool = False
    ) -> Dict[str, Any]:
        """Make an HTTP request to the Binance API."""
        if not self.session or self.session.closed:
            self.session = aiohttp.ClientSession()
        
        if params is None:
            params = {}
        
        headers = {"X-MBX-APIKEY": self.api_key}
        
        if signed:
            params["timestamp"] = int(time.time() * 1000)
            params["recvWindow"] = 5000
            signature = self._sign_request(params)
            params["signature"] = signature

        try:
            if method.upper() == "GET":
                # For GET requests, parameters go in the query string
                query_string = urlencode(params, safe='~')
                url = f"{self.BASE_URL}{endpoint}?{query_string}" if query_string else f"{self.BASE_URL}{endpoint}"
                
                async with self.session.get(url, headers=headers) as response:
                    data = await response.json()
                    
            elif method.upper() == "POST":
                # For POST requests, parameters go in the request body
                url = f"{self.BASE_URL}{endpoint}"
                
                async with self.session.post(url, data=params, headers=headers) as response:
                    data = await response.json()
                    
            elif method.upper() == "DELETE":
                # For DELETE requests, parameters go in the query string
                query_string = urlencode(params, safe='~')
                url = f"{self.BASE_URL}{endpoint}?{query_string}" if query_string else f"{self.BASE_URL}{endpoint}"
                
                async with self.session.delete(url, headers=headers) as response:
                    data = await response.json()
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
                
            if response.status != 200:
                logger.error(f"API error: {response.status} - {data}")
                raise Exception(f"Binance API error: {response.status} - {data.get('msg', 'Unknown Error')}")
            
            return data
                
        except aiohttp.ClientError as e:
            logger.error(f"AIOHTTP request failed: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response: {e}")
            try:
                raw_text = await response.text()
                logger.error(f"Raw response text: {raw_text}")
            except Exception as e:
                logger.error(f"Error in binance_connector_hmac.py: {e}", exc_info=True)
                raise  # Re-raise for proper error handling
                logger.error("Could not get raw response text")
            raise
        except Exception as e:
            logger.error(f"An unexpected request error occurred: {e}")
            raise

    # Market Data Endpoints (No authentication required)
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get 24hr ticker for symbol"""
        return await self._request("GET", "/api/v3/ticker/24hr", {"symbol": symbol})
    
    async def get_order_book(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """Get order book for symbol"""
        return await self._request("GET", "/api/v3/depth", {
            "symbol": symbol,
            "limit": limit
        })
    
    async def get_klines(
        self, 
        symbol: str, 
        interval: str = "1m",
        limit: int = 100
    ) -> List[List[Any]]:
        """Get candlestick data"""
        data = await self._request("GET", "/api/v3/klines", {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        })
        return data
    
    # Account Endpoints (Require authentication)
    async def get_account(self) -> Dict[str, Any]:
        """Get account information"""
        return await self._request("GET", "/api/v3/account", signed=True)
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information (alias for get_account for compatibility)"""
        return await self.get_account()

    async def get_balance(self, asset: str) -> Dict[str, float]:
        """Get balance for specific asset"""
        account = await self.get_account()
        
        for balance in account["balances"]:
            if balance["asset"] == asset:
                return {
                    "free": float(balance["free"]),
                    "locked": float(balance["locked"])
                }
        
        return {"free": 0.0, "locked": 0.0}
    
    # Trading Endpoints
    async def place_order(
        self,
        symbol: str,
        side: str,  # BUY or SELL
        order_type: str,  # MARKET, LIMIT, etc.
        quantity: float,
        price: Optional[float] = None,
        time_in_force: str = "GTC",
        stop_price: Optional[float] = None
    ) -> Dict[str, Any]:
        """Place a new order"""
        params = {
            "symbol": symbol,
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": self._format_quantity(symbol, quantity)
        }
        
        if order_type.upper() in ["LIMIT", "STOP_LOSS_LIMIT", "TAKE_PROFIT_LIMIT"]:
            if price is None:
                raise ValueError(f"Price is required for {order_type} orders")
            params["price"] = self._format_price(symbol, price)
            params["timeInForce"] = time_in_force
        
        if stop_price is not None:
            params["stopPrice"] = self._format_price(symbol, stop_price)
        
        return await self._request("POST", "/api/v3/order", params, signed=True)
    
    async def cancel_order(self, symbol: str, order_id: int) -> Dict[str, Any]:
        """Cancel an active order"""
        params = {"symbol": symbol, "orderId": order_id}
        return await self._request("DELETE", "/api/v3/order", params, signed=True)
    
    async def get_order(self, symbol: str, order_id: int) -> Dict[str, Any]:
        """Get order status"""
        params = {"symbol": symbol, "orderId": order_id}
        return await self._request("GET", "/api/v3/order", params, signed=True)
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all open orders"""
        params = {}
        if symbol:
            params["symbol"] = symbol
        return await self._request("GET", "/api/v3/openOrders", params, signed=True)

    # WebSocket Methods (Simplified for this implementation)
    async def subscribe_ticker(self, symbol: str, callback: Callable) -> None:
        """Subscribe to ticker updates"""
        await self._subscribe_stream(f"{symbol.lower()}@ticker", callback)

    async def subscribe_orderbook(self, symbol: str, callback: Callable) -> None:
        """Subscribe to order book updates"""
        await self._subscribe_stream(f"{symbol.lower()}@depth", callback)

    async def subscribe_trades(self, symbol: str, callback: Callable) -> None:
        """Subscribe to trade updates"""
        await self._subscribe_stream(f"{symbol.lower()}@trade", callback)

    async def subscribe_user_data(self, callback: Callable) -> None:
        """Subscribe to user data stream"""
        # Get listen key first
        listen_key_data = await self._request("POST", "/api/v3/userDataStream", signed=True)
        listen_key = listen_key_data["listenKey"]
        
        await self._subscribe_stream(listen_key, callback)
        
        # Keep alive task
        asyncio.create_task(self._keep_alive_user_stream(listen_key))

    async def _subscribe_stream(self, stream_name: str, callback: Callable):
        """Subscribe to a WebSocket stream"""
        if stream_name in self.ws_connections:
            return  # Already subscribed
        
        ws_url = f"{self.WS_BASE_URL}/{stream_name}"
        
        try:
            ws = await websockets.connect(ws_url)
            self.ws_connections[stream_name] = ws
            
            if stream_name not in self.callbacks:
                self.callbacks[stream_name] = []
            self.callbacks[stream_name].append(callback)
            
            # Start listening task
            asyncio.create_task(self._listen_stream(stream_name, ws))
            
        except Exception as e:
            logger.error(f"Failed to subscribe to {stream_name}: {e}")

    async def _listen_stream(self, stream_name: str, ws: websockets.WebSocketClientProtocol):
        """Listen to WebSocket stream"""
        try:
            async for message in ws:
                try:
                    data = json.loads(message)
                    
                    # Call all callbacks for this stream
                    for callback in self.callbacks.get(stream_name, []):
                        try:
                            await callback(data)
                        except Exception as e:
                            logger.error(f"Callback error for {stream_name}: {e}")
                            
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode WebSocket message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"WebSocket connection closed for {stream_name}")
        except Exception as e:
            logger.error(f"WebSocket error for {stream_name}: {e}")
        finally:
            if stream_name in self.ws_connections:
                del self.ws_connections[stream_name]

    async def _keep_alive_user_stream(self, listen_key: str):
        """Keep user data stream alive"""
        while listen_key in [s for s in self.ws_connections.keys() if len(s) == 60]:  # Listen keys are 60 chars
            try:
                await asyncio.sleep(30 * 60)  # Every 30 minutes
                await self._request("PUT", "/api/v3/userDataStream", {"listenKey": listen_key}, signed=True)
                logger.debug(f"User data stream keep-alive sent for {listen_key}")
            except Exception as e:
                logger.error(f"Failed to keep alive user stream: {e}")
                break

    # Utility methods
    def _format_quantity(self, symbol: str, quantity: float) -> str:
        """Format quantity for the symbol (simplified implementation)"""
        # This is a simplified implementation
        # A robust version would use exchange info to get proper precision
        return f"{quantity:.8f}".rstrip('0').rstrip('.')
    
    def _format_price(self, symbol: str, price: float) -> str:
        """Format price for the symbol (simplified implementation)"""
        # This is a simplified implementation
        # A robust version would use exchange info to get proper precision
        return f"{price:.8f}".rstrip('0').rstrip('.')
    
    async def get_exchange_info(self) -> Dict[str, Any]:
        """Get exchange trading rules and symbol information"""
        return await self._request("GET", "/api/v3/exchangeInfo")
    
    async def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive market data for symbol"""
        ticker = await self.get_ticker(symbol)
        order_book = await self.get_order_book(symbol, limit=5)
        
        return {
            "symbol": symbol,
            "price": float(ticker["lastPrice"]),
            "change_24h": float(ticker["priceChangePercent"]),
            "volume_24h": float(ticker["volume"]),
            "high_24h": float(ticker["highPrice"]),
            "low_24h": float(ticker["lowPrice"]),
            "bid": float(order_book["bids"][0][0]) if order_book["bids"] else 0,
            "ask": float(order_book["asks"][0][0]) if order_book["asks"] else 0,
            "timestamp": int(time.time() * 1000)
        }

    async def get_ticker_price(self, symbol: str) -> Dict[str, Any]:
        """Get current price for symbol"""
        return await self._request("GET", "/api/v3/ticker/price", {"symbol": symbol})

    async def create_market_order(self, symbol: str, side: str, quantity: float) -> Dict[str, Any]:
        """Create a market order"""
        return await self.place_order(
            symbol=symbol,
            side=side,
            order_type="MARKET",
            quantity=quantity
        ) 