"""
Fixed Binance API Connector - Ed25519 Implementation (Official Format)

Based on official Binance documentation:
https://developers.binance.com/docs/binance-spot-api-docs/rest-api/endpoint-security-type

Key fixes:
1. Ed25519 signatures must be sent as POST data, not query parameters
2. Use proper request body format for signed endpoints
3. Follow exact signature generation from Binance Python example
"""

import asyncio
import logging
import time
import base64
import json
import os
from typing import Dict, List, Optional, Any, Callable
from urllib.parse import urlencode
import aiohttp
import websockets

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

logger = logging.getLogger(__name__)


class BinanceConnectorFixed:
    """
    Fixed Binance connector implementing official Ed25519 signature format.
    
    Based on official Binance API documentation example:
    https://developers.binance.com/docs/binance-spot-api-docs/rest-api/endpoint-security-type
    """
    
    BASE_URL = "https://api.binance.com"
    WS_BASE_URL = "wss://stream.binance.com:9443/ws"
    
    def __init__(self, api_key: str, private_key_path: str, testnet: bool = True):
        """
        Initialize Binance connector with Ed25519 key-pair.
        
        Args:
            api_key: Your Binance API Key.
            private_key_path: Filesystem path to your Ed25519 private key PEM file.
            testnet: Use testnet if True (recommended for testing).
        """
        if not api_key:
            raise ValueError("API key is required.")
        if not private_key_path:
            raise ValueError("Private key path is required.")

        self.api_key = api_key
        self._private_key = self._load_private_key(private_key_path)
        
        if testnet:
            self.BASE_URL = "https://testnet.binance.vision"
            self.WS_BASE_URL = "wss://testnet.binance.vision/ws"
            
        self.session: Optional[aiohttp.ClientSession] = None
        
        logger.info(f"Fixed Binance connector initialized with Ed25519 auth (testnet={testnet})")

    def _load_private_key(self, path: str) -> ed25519.Ed25519PrivateKey:
        """Loads the Ed25519 private key from a PEM file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Private key file not found at: {path}")
        try:
            with open(path, 'rb') as f:
                private_key = serialization.load_pem_private_key(
                    f.read(),
                    password=None  # No password protection on the key file
                )
            if not isinstance(private_key, ed25519.Ed25519PrivateKey):
                raise TypeError("The provided key is not a valid Ed25519 private key.")
            logger.info(f"Successfully loaded Ed25519 private key from {path}")
            return private_key
        except Exception as e:
            logger.error(f"Failed to load or parse private key from {path}: {e}")
            raise

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def close(self):
        """Close all connections"""
        if self.session and not self.session.closed:
            await self.session.close()
    
    def _sign_request(self, params: Dict[str, Any]) -> str:
        """
        Sign request using Ed25519 following official Binance documentation.
        
        Official example from Binance docs:
        payload = '&'.join([f'{param}={value}' for param, value in params.items()])
        signature = base64.b64encode(private_key.sign(payload.encode('ASCII')))
        """
        # Create payload exactly as shown in Binance documentation
        payload = '&'.join([f'{param}={value}' for param, value in params.items()])
        
        # Sign with Ed25519 and base64 encode (official format)
        signature_bytes = self._private_key.sign(payload.encode('ASCII'))
        signature = base64.b64encode(signature_bytes).decode('utf-8')
        
        logger.debug(f"Ed25519 signature generated for payload: {payload}")
        logger.debug(f"Signature: {signature}")
        
        return signature
    
    async def _request(
        self, 
        method: str, 
        endpoint: str, 
        params: Optional[Dict[str, Any]] = None,
        signed: bool = False
    ) -> Dict[str, Any]:
        """
        Make an HTTP request using official Binance Ed25519 format.
        
        Key difference: For signed requests, format depends on HTTP method.
        """
        if not self.session or self.session.closed:
            self.session = aiohttp.ClientSession()
        
        if params is None:
            params = {}
        
        headers = {"X-MBX-APIKEY": self.api_key}
        url = f"{self.BASE_URL}{endpoint}"
        
        if signed:
            # Add timestamp
            params["timestamp"] = int(time.time() * 1000)
            
            # Generate signature
            signature = self._sign_request(params)
            params["signature"] = signature
            
            # For signed requests, format depends on method
            logger.debug(f"Sending signed {method} request to {url}")
            logger.debug(f"Params: {params}")
            
            try:
                if method.upper() in ["POST", "PUT", "DELETE"]:
                    # For POST/PUT/DELETE, send as form data
                    async with self.session.request(
                        method, 
                        url, 
                        headers=headers,
                        data=params
                    ) as response:
                        response_text = await response.text()
                        logger.debug(f"Response status: {response.status}")
                        logger.debug(f"Response text: {response_text}")
                        
                        if response.status == 200:
                            return json.loads(response_text)
                        else:
                            try:
                                error_data = json.loads(response_text)
                            except Exception as e:
                                logger.error(f"Error in binance_connector_fixed.py: {e}", exc_info=True)
                                raise  # Re-raise for proper error handling
                                error_data = {"msg": response_text, "code": response.status}
                            
                            logger.error(f"API error: {response.status} - {error_data}")
                            raise Exception(f"Binance API error: {response.status} - {error_data.get('msg', 'Unknown Error')}")
                else:
                    # For GET requests, send as query parameters
                    async with self.session.request(
                        method, 
                        url, 
                        headers=headers,
                        params=params
                    ) as response:
                        response_text = await response.text()
                        logger.debug(f"Response status: {response.status}")
                        logger.debug(f"Response text: {response_text}")
                        
                        if response.status == 200:
                            return json.loads(response_text)
                        else:
                            try:
                                error_data = json.loads(response_text)
                            except Exception as e:
                                logger.error(f"Error in binance_connector_fixed.py: {e}", exc_info=True)
                                raise  # Re-raise for proper error handling
                                error_data = {"msg": response_text, "code": response.status}
                            
                            logger.error(f"API error: {response.status} - {error_data}")
                            raise Exception(f"Binance API error: {response.status} - {error_data.get('msg', 'Unknown Error')}")
                        
            except aiohttp.ClientError as e:
                logger.error(f"AIOHTTP request failed: {e}")
                raise
            except Exception as e:
                logger.error(f"Request error: {e}")
                raise
        else:
            # For unsigned requests, use query parameters
            try:
                async with self.session.request(
                    method, 
                    url, 
                    headers=headers,
                    params=params
                ) as response:
                    data = await response.json()
                    
                    if response.status != 200:
                        logger.error(f"API error: {response.status} - {data}")
                        raise Exception(f"Binance API error: {response.status} - {data.get('msg', 'Unknown Error')}")
                    
                    return data
                    
            except Exception as e:
                logger.error(f"Request error: {e}")
                raise

    # Market Data Endpoints
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get 24hr ticker for symbol"""
        return await self._request("GET", "/api/v3/ticker/24hr", {"symbol": symbol})
    
    async def get_orderbook(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """Get order book for symbol"""
        return await self._request("GET", "/api/v3/depth", {
            "symbol": symbol,
            "limit": limit
        })
    
    # Account Endpoints
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
                    "locked": float(balance["locked"]),
                    "total": float(balance["free"]) + float(balance["locked"])
                }
        
        return {"free": 0.0, "locked": 0.0, "total": 0.0}
    
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
        """Place a trading order"""
        params = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "quantity": str(quantity)  # Convert to string for proper formatting
        }
        
        if order_type == "LIMIT":
            if price is None:
                raise ValueError("Price required for LIMIT orders")
            params["price"] = str(price)
            params["timeInForce"] = time_in_force
        
        if stop_price is not None:
            params["stopPrice"] = str(stop_price)
        
        return await self._request("POST", "/api/v3/order", params, signed=True)
    
    async def cancel_order(self, symbol: str, order_id: int) -> Dict[str, Any]:
        """Cancel an order"""
        return await self._request("DELETE", "/api/v3/order", {
            "symbol": symbol,
            "orderId": order_id
        }, signed=True)
    
    async def get_order(self, symbol: str, order_id: int) -> Dict[str, Any]:
        """Get order status"""
        return await self._request("GET", "/api/v3/order", {
            "symbol": symbol,
            "orderId": order_id
        }, signed=True)
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all open orders"""
        params = {}
        if symbol:
            params["symbol"] = symbol
        return await self._request("GET", "/api/v3/openOrders", params, signed=True)
    
    # Utility methods for compatibility
    def _format_quantity(self, symbol: str, quantity: float) -> str:
        """Format quantity - simplified version"""
        return f"{quantity:.8f}".rstrip('0').rstrip('.')
    
    def _format_price(self, symbol: str, price: float) -> str:
        """Format price - simplified version"""
        return f"{price:.8f}".rstrip('0').rstrip('.')
    
    async def get_exchange_info(self) -> Dict[str, Any]:
        """Get exchange information"""
        return await self._request("GET", "/api/v3/exchangeInfo")
    
    async def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive market data for symbol"""
        ticker = await self.get_ticker(symbol)
        orderbook = await self.get_orderbook(symbol, 10)
        
        return {
            "symbol": symbol,
            "price": float(ticker["lastPrice"]),
            "bid": float(orderbook["bids"][0][0]) if orderbook["bids"] else 0,
            "ask": float(orderbook["asks"][0][0]) if orderbook["asks"] else 0,
            "volume": float(ticker["volume"]),
            "change": float(ticker["priceChangePercent"])
        } 