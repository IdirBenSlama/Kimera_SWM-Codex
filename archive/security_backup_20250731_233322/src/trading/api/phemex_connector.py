"""
Phemex API Connector

Handles all interactions with Phemex exchange for crypto trading.
Supports perpetual contracts and spot trading with real-time data streaming.
"""

import os
import asyncio
import logging
import time
import hmac
import hashlib
import json
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from decimal import Decimal
import aiohttp
import websockets
from urllib.parse import urlencode

logger = logging.getLogger(__name__)


class PhemexConnector:
    """
    Phemex exchange connector with WebSocket support for real-time data.
    Handles both spot and perpetual contract trading.
    """
    
    BASE_URL = "https://api.phemex.com"
    WS_BASE_URL = "wss://phemex.com/ws"
    TESTNET_BASE_URL = "https://testnet-api.phemex.com"
    TESTNET_WS_URL = "wss://testnet.phemex.com/ws"
    
    # Phemex uses scaled values
    PRICE_SCALE = 10000  # For contracts
    QTY_SCALE = 100000000  # For quantity
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = None):
        """
        Initialize Phemex connector.
        
        Args:
            api_key: Phemex API key
            api_secret: Phemex API secret
            testnet: Use testnet if True. Defaults to environment variable KIMERA_USE_TESTNET or False for real trading.
        """
        self.api_key = api_key
        self.api_secret = api_secret.encode('utf-8')
        
        # Default to real trading unless explicitly set to testnet
        if testnet is None:
            testnet = os.getenv('KIMERA_USE_TESTNET', 'false').lower() == 'true'
        
        if testnet:
            self.BASE_URL = self.TESTNET_BASE_URL
            self.WS_BASE_URL = self.TESTNET_WS_URL
            logger.warning("ðŸ§ª TESTNET MODE ENABLED - No real trades will be executed")
        else:
            logger.info("ðŸš€ LIVE TRADING MODE ENABLED - Real trades will be executed")
            
        self.session: Optional[aiohttp.ClientSession] = None
        self.ws_connection: Optional[websockets.WebSocketClientProtocol] = None
        self.subscriptions: Dict[str, List[Callable]] = {}
        
        # Message ID for WebSocket
        self._msg_id = 1
        
        logger.info(f"Phemex connector initialized (testnet={testnet})")
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def close(self):
        """Close all connections"""
        if self.ws_connection:
            await self.ws_connection.close()
            
        if self.session:
            await self.session.close()
    
    def _sign_request(self, path: str, query_string: str = "", expiry: int = 60) -> Dict[str, str]:
        """
        Sign request with API secret.
        
        Phemex uses: signature = HmacSHA256(URL Path + QueryString + Expiry + BodyString)
        """
        expiry_time = int(time.time() + expiry)
        
        # Construct the message to sign
        message = path + query_string + str(expiry_time)
        
        # Generate signature
        signature = hmac.new(
            self.api_secret,
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return {
            "x-phemex-access-token": self.api_key,
            "x-phemex-request-signature": signature,
            "x-phemex-request-expiry": str(expiry_time)
        }
    
    async def _request(
        self, 
        method: str, 
        endpoint: str, 
        params: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None,
        signed: bool = False
    ) -> Dict[str, Any]:
        """Make HTTP request to Phemex API"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        url = f"{self.BASE_URL}{endpoint}"
        headers = {"Content-Type": "application/json"}
        
        # Prepare query string
        query_string = ""
        if params:
            query_string = "?" + urlencode(sorted(params.items()))
            url += query_string
        
        # Sign request if needed
        if signed:
            sign_headers = self._sign_request(endpoint, query_string)
            headers.update(sign_headers)
        
        try:
            kwargs = {"headers": headers}
            if body:
                kwargs["json"] = body
                
            async with self.session.request(method, url, **kwargs) as response:
                data = await response.json()
                
                if response.status != 200 or data.get("code") != 0:
                    error_msg = data.get("msg", "Unknown error")
                    logger.error(f"API error: {response.status} - {error_msg}")
                    raise Exception(f"Phemex API error: {error_msg}")
                
                return data.get("data", data)
                
        except Exception as e:
            logger.error(f"Request failed: {str(e)}")
            raise
    
    # Market Data Endpoints
    async def get_products(self) -> List[Dict[str, Any]]:
        """Get all trading products"""
        data = await self._request("GET", "/public/products")
        return data.get("products", [])
    
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get 24hr ticker for symbol"""
        endpoint = f"/md/v2/ticker/24hr"
        params = {"symbol": symbol}
        return await self._request("GET", endpoint, params=params)
    
    async def get_orderbook(self, symbol: str) -> Dict[str, Any]:
        """Get order book for symbol"""
        endpoint = f"/md/v2/orderbook"
        params = {"symbol": symbol}
        return await self._request("GET", endpoint, params=params)
    
    async def get_klines(
        self, 
        symbol: str,
        resolution: int = 300,  # 5 minutes
        limit: int = 100
    ) -> List[List[Any]]:
        """Get candlestick data"""
        endpoint = "/exchange/public/md/v2/kline"
        
        # Calculate time range
        to_time = int(time.time())
        from_time = to_time - (resolution * limit)
        
        params = {
            "symbol": symbol,
            "resolution": resolution,
            "from": from_time,
            "to": to_time
        }
        
        data = await self._request("GET", endpoint, params=params)
        return data.get("rows", [])
    
    # Account Endpoints
    async def get_account_positions(self, currency: str = "BTC") -> Dict[str, Any]:
        """Get account positions"""
        endpoint = f"/accounts/accountPositions"
        params = {"currency": currency}
        return await self._request("GET", endpoint, params=params, signed=True)
    
    async def get_wallet_balance(self, currency: str = "USDT") -> Dict[str, Any]:
        """Get wallet balance"""
        endpoint = "/phemex-user/users/children"
        data = await self._request("GET", endpoint, signed=True)
        
        # Extract balance for specified currency
        for account in data.get("accounts", []):
            if account.get("currency") == currency:
                return {
                    "currency": currency,
                    "balance": account.get("accountBalance", 0) / self.QTY_SCALE,
                    "available": account.get("availableBalance", 0) / self.QTY_SCALE
                }
        
        return {"currency": currency, "balance": 0, "available": 0}
    
    # Trading Endpoints (Perpetual Contracts)
    async def place_contract_order(
        self,
        symbol: str,
        side: str,  # Buy or Sell
        order_type: str,  # Market, Limit
        contracts: int,  # Number of contracts
        price: Optional[float] = None,
        reduce_only: bool = False,
        time_in_force: str = "GoodTillCancel",
        stop_px: Optional[float] = None,
        trigger_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Place a perpetual contract order"""
        endpoint = "/orders"
        
        # Scale price if provided
        scaled_price = int(price * self.PRICE_SCALE) if price else None
        scaled_stop = int(stop_px * self.PRICE_SCALE) if stop_px else None
        
        body = {
            "symbol": symbol,
            "side": side,
            "ordType": order_type,
            "orderQty": contracts,
            "reduceOnly": reduce_only,
            "timeInForce": time_in_force
        }
        
        if order_type == "Limit" and scaled_price:
            body["priceRp"] = scaled_price
            
        if scaled_stop:
            body["stopPxRp"] = scaled_stop
            if trigger_type:
                body["triggerType"] = trigger_type
        
        return await self._request("POST", endpoint, body=body, signed=True)
    
    # Trading Endpoints (Spot)
    async def place_spot_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        time_in_force: str = "GoodTillCancel"
    ) -> Dict[str, Any]:
        """Place a spot order"""
        endpoint = "/spot/orders"
        
        body = {
            "symbol": symbol,
            "side": side,
            "ordType": order_type,
            "qtyType": "ByBase",
            "baseQtyRq": str(quantity),
            "timeInForce": time_in_force
        }
        
        if order_type == "Limit" and price:
            body["priceRp"] = str(price)
        
        return await self._request("POST", endpoint, body=body, signed=True)
    
    async def cancel_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """Cancel an order"""
        endpoint = f"/orders/cancel"
        params = {"symbol": symbol, "orderID": order_id}
        return await self._request("DELETE", endpoint, params=params, signed=True)
    
    async def get_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """Get order status"""
        endpoint = "/exchange/order"
        params = {"symbol": symbol, "orderID": order_id}
        return await self._request("GET", endpoint, params=params, signed=True)
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all open orders"""
        endpoint = "/orders/activeList"
        params = {}
        if symbol:
            params["symbol"] = symbol
            
        data = await self._request("GET", endpoint, params=params, signed=True)
        return data.get("rows", [])
    
    # WebSocket Methods
    async def connect_websocket(self) -> None:
        """Connect to WebSocket"""
        if self.ws_connection:
            return
            
        self.ws_connection = await websockets.connect(self.WS_BASE_URL)
        
        # Start listening task
        asyncio.create_task(self._listen_websocket())
        
        # Authenticate
        await self._authenticate_websocket()
    
    async def _authenticate_websocket(self) -> None:
        """Authenticate WebSocket connection"""
        expiry = int(time.time() + 60)
        signature = hmac.new(
            self.api_secret,
            f"{self.api_key}{expiry}".encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        auth_msg = {
            "method": "user.auth",
            "params": ["API", self.api_key, signature, expiry],
            "id": self._get_msg_id()
        }
        
        await self.ws_connection.send(json.dumps(auth_msg))
    
    async def subscribe_ticker(self, symbol: str, callback: Callable) -> None:
        """Subscribe to ticker updates"""
        await self._subscribe("tick", symbol, callback)
    
    async def subscribe_orderbook(self, symbol: str, callback: Callable) -> None:
        """Subscribe to orderbook updates"""
        await self._subscribe("orderbook", symbol, callback)
    
    async def subscribe_trades(self, symbol: str, callback: Callable) -> None:
        """Subscribe to trade updates"""
        await self._subscribe("trade", symbol, callback)
    
    async def subscribe_account(self, callback: Callable) -> None:
        """Subscribe to account updates (orders, positions)"""
        await self._subscribe("aop", None, callback)
    
    async def _subscribe(self, channel: str, symbol: Optional[str], callback: Callable) -> None:
        """Subscribe to a channel"""
        if not self.ws_connection:
            await self.connect_websocket()
        
        # Build subscription key
        sub_key = f"{channel}.{symbol}" if symbol else channel
        
        # Store callback
        if sub_key not in self.subscriptions:
            self.subscriptions[sub_key] = []
        self.subscriptions[sub_key].append(callback)
        
        # Send subscription message
        sub_msg = {
            "id": self._get_msg_id(),
            "method": f"{channel}.subscribe",
            "params": []
        }
        
        if symbol:
            sub_msg["params"].append(symbol)
        
        await self.ws_connection.send(json.dumps(sub_msg))
        logger.info(f"Subscribed to {sub_key}")
    
    async def _listen_websocket(self) -> None:
        """Listen to WebSocket messages"""
        try:
            async for message in self.ws_connection:
                data = json.loads(message)
                
                # Handle different message types
                if "method" in data:
                    await self._handle_ws_notification(data)
                elif "error" in data:
                    logger.error(f"WebSocket error: {data['error']}")
                elif "result" in data:
                    logger.debug(f"WebSocket result: {data}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed")
            self.ws_connection = None
            # Try to reconnect
            await asyncio.sleep(5)
            await self.connect_websocket()
    
    async def _handle_ws_notification(self, data: Dict[str, Any]) -> None:
        """Handle WebSocket notifications"""
        method = data.get("method", "")
        params = data.get("params", [])
        
        # Extract channel and symbol
        parts = method.split(".")
        if len(parts) >= 2:
            channel = parts[0]
            
            # Find matching subscriptions
            for sub_key, callbacks in self.subscriptions.items():
                if sub_key.startswith(channel):
                    for callback in callbacks:
                        try:
                            await callback(params)
                        except Exception as e:
                            logger.error(f"Callback error: {str(e)}")
    
    def _get_msg_id(self) -> int:
        """Get next message ID"""
        msg_id = self._msg_id
        self._msg_id += 1
        return msg_id
    
    # Helper Methods
    async def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive market data for Kimera analysis"""
        try:
            # Gather all market data in parallel
            ticker_task = self.get_ticker(symbol)
            orderbook_task = self.get_orderbook(symbol)
            klines_task = self.get_klines(symbol, resolution=300, limit=100)
            
            ticker, orderbook, klines = await asyncio.gather(
                ticker_task, orderbook_task, klines_task
            )
            
            # Process ticker data
            last_price = float(ticker.get("lastPrice", 0))
            volume = float(ticker.get("volume", 0))
            
            # Process klines for price and volume history
            price_history = []
            volume_history = []
            
            for kline in klines:
                # Kline format: [timestamp, interval, last_price, open, high, low, close, volume, turnover]
                if len(kline) >= 8:
                    price_history.append(float(kline[6]))  # Close price
                    volume_history.append(float(kline[7]))  # Volume
            
            # Calculate order book imbalance
            bids = orderbook.get("bids", [])
            asks = orderbook.get("asks", [])
            
            total_bid_volume = sum(float(bid[1]) for bid in bids) if bids else 0
            total_ask_volume = sum(float(ask[1]) for ask in asks) if asks else 0
            
            if total_bid_volume + total_ask_volume > 0:
                order_book_imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)
            else:
                order_book_imbalance = 0
            
            # Determine sentiment
            price_change = float(ticker.get("changeRatio", 0))
            sentiment = "bullish" if price_change > 0 else "bearish" if price_change < 0 else "neutral"
            
            return {
                "symbol": symbol,
                "price": last_price,
                "volume": volume,
                "bid": float(bids[0][0]) if bids else 0,
                "ask": float(asks[0][0]) if asks else 0,
                "price_history": price_history,
                "volume_history": volume_history,
                "order_book": orderbook,
                "order_book_imbalance": order_book_imbalance,
                "price_change_percent": price_change * 100,
                "high_24h": float(ticker.get("high", 0)),
                "low_24h": float(ticker.get("low", 0)),
                "sentiment": sentiment
            }
            
        except Exception as e:
            logger.error(f"Failed to get market data: {str(e)}")
            raise
    
    def scale_price(self, price: float, is_contract: bool = True) -> int:
        """Scale price for Phemex API"""
        return int(price * self.PRICE_SCALE) if is_contract else price
    
    def unscale_price(self, scaled_price: int, is_contract: bool = True) -> float:
        """Unscale price from Phemex API"""
        return scaled_price / self.PRICE_SCALE if is_contract else scaled_price
    
    def scale_quantity(self, quantity: float) -> int:
        """Scale quantity for Phemex API"""
        return int(quantity * self.QTY_SCALE)
    
    def unscale_quantity(self, scaled_qty: int) -> float:
        """Unscale quantity from Phemex API"""
        return scaled_qty / self.QTY_SCALE 