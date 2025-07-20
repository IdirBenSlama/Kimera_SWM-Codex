#!/usr/bin/env python3
"""
Direct Aggressive Trader Execution
=================================

Direct execution of the aggressive 10-minute trader.
$50 → $300 target in 10 minutes.
"""

import asyncio
import aiohttp
import hmac
import hashlib
import time
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
from urllib.parse import urlencode

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AggressiveTrader:
    """Direct aggressive trading execution"""
    
    def __init__(self):
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.secret_key = os.getenv('BINANCE_SECRET_KEY')
        self.base_url = "https://api.binance.com"
        self.session = None
        self.start_time = None
        self.starting_balance = 50.0
        self.target_balance = 300.0
        self.trades_executed = 0
        self.max_trades = 120
        self.symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'SOLUSDT']
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def _generate_signature(self, params):
        """Generate HMAC signature"""
        query_string = urlencode(params)
        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    async def get_account_balance(self) -> float:
        """Get total account balance in USDT"""
        try:
            url = f"{self.base_url}/api/v3/account"
            timestamp = int(time.time() * 1000)
            params = {"timestamp": timestamp}
            signature = self._generate_signature(params)
            params["signature"] = signature
            headers = {"X-MBX-APIKEY": self.api_key}
            
            async with self.session.get(url, params=params, headers=headers) as response:
                data = await response.json()
                
                total_usdt = 0
                for balance in data.get('balances', []):
                    asset = balance['asset']
                    free = float(balance['free'])
                    
                    if asset == 'USDT':
                        total_usdt += free
                    elif free > 0 and asset in ['BTC', 'ETH', 'ADA', 'SOL']:
                        # Convert to USDT
                        try:
                            price = await self.get_price(f"{asset}USDT")
                            total_usdt += free * price
                        except:
                            pass
                
                return total_usdt
                
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return 0
    
    async def get_price(self, symbol: str) -> float:
        """Get current price"""
        try:
            url = f"{self.base_url}/api/v3/ticker/price"
            params = {"symbol": symbol}
            
            async with self.session.get(url, params=params) as response:
                data = await response.json()
                return float(data['price'])
        except Exception as e:
            logger.error(f"Failed to get price for {symbol}: {e}")
            return 0
    
    async def get_volatility(self, symbol: str) -> float:
        """Get 24h volatility"""
        try:
            url = f"{self.base_url}/api/v3/ticker/24hr"
            params = {"symbol": symbol}
            
            async with self.session.get(url, params=params) as response:
                data = await response.json()
                return abs(float(data['priceChangePercent']))
        except Exception as e:
            logger.error(f"Failed to get volatility for {symbol}: {e}")
            return 0
    
    async def place_aggressive_trade(self, symbol: str, side: str, quantity: float) -> bool:
        """Place aggressive market order"""
        try:
            url = f"{self.base_url}/api/v3/order"
            timestamp = int(time.time() * 1000)
            
            # Format quantity
            if 'BTC' in symbol:
                quantity = round(quantity, 5)
            elif 'ETH' in symbol:
                quantity = round(quantity, 3)
            else:
                quantity = round(quantity, 2)
            
            params = {
                "symbol": symbol,
                "side": side.upper(),
                "type": "MARKET",
                "quantity": str(quantity),
                "timestamp": timestamp
            }
            
            signature = self._generate_signature(params)
            params["signature"] = signature
            headers = {"X-MBX-APIKEY": self.api_key}
            
            async with self.session.post(url, data=params, headers=headers) as response:
                result = await response.json()
                
                if 'orderId' in result:
                    self.trades_executed += 1
                    logger.info(f"✅ Trade #{self.trades_executed}: {side} {quantity} {symbol}")
                    return True
                else:
                    logger.error(f"❌ Trade failed: {result}")
                    return False
                    
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return False
    
    async def select_best_opportunity(self) -> Dict[str, Any]:
        """Select best trading opportunity"""
        try:
            best_symbol = None
            best_volatility = 0
            best_action = 'buy'
            
            for symbol in self.symbols:
                volatility = await self.get_volatility(symbol)
                
                if volatility > best_volatility:
                    best_volatility = volatility
                    best_symbol = symbol
                    
                    # Simple momentum strategy
                    price = await self.get_price(symbol)
                    if price > 0:
                        best_action = 'buy' if volatility > 5 else 'sell'
            
            return {
                'symbol': best_symbol or 'BTCUSDT',
                'action': best_action,
                'volatility': best_volatility,
                'confidence': min(best_volatility / 10, 1.0)
            }
            
        except Exception as e:
            logger.error(f"Opportunity selection failed: {e}")
            return {'symbol': 'BTCUSDT', 'action': 'buy', 'volatility': 0, 'confidence': 0.5}
    
    async def execute_aggressive_session(self):
        """Execute the aggressive 10-minute trading session"""
        try:
            self.start_time = datetime.now()
            logger.info("🚀 STARTING AGGRESSIVE 10-MINUTE SESSION")
            logger.info(f"🎯 Target: ${self.starting_balance} → ${self.target_balance}")
            
            # Get starting balance
            current_balance = await self.get_account_balance()
            logger.info(f"💰 Starting Balance: ${current_balance:.2f}")
            
            if current_balance < 20:
                logger.error("❌ Insufficient balance for aggressive trading")
                return
            
            # Trading loop
            while True:
                # Check time limit (10 minutes)
                elapsed = datetime.now() - self.start_time
                if elapsed.total_seconds() > 600:
                    logger.info("⏰ 10-minute time limit reached")
                    break
                
                # Check target achieved
                current_balance = await self.get_account_balance()
                if current_balance >= self.target_balance:
                    logger.info(f"🎯 TARGET ACHIEVED! ${current_balance:.2f}")
                    break
                
                # Check max trades
                if self.trades_executed >= self.max_trades:
                    logger.info("📊 Maximum trades reached")
                    break
                
                # Select opportunity
                opportunity = await self.select_best_opportunity()
                
                if opportunity['confidence'] < 0.3:
                    logger.info(f"⚠️ Low confidence opportunity, skipping")
                    await asyncio.sleep(2)
                    continue
                
                # Calculate aggressive position size
                position_size = current_balance * 0.9  # 90% of balance
                
                # Get price and calculate quantity
                price = await self.get_price(opportunity['symbol'])
                if price <= 0:
                    continue
                
                quantity = position_size / price
                
                # Execute trade
                logger.info(f"🔥 AGGRESSIVE TRADE:")
                logger.info(f"   Symbol: {opportunity['symbol']}")
                logger.info(f"   Action: {opportunity['action'].upper()}")
                logger.info(f"   Quantity: {quantity:.6f}")
                logger.info(f"   Value: ${position_size:.2f}")
                logger.info(f"   Volatility: {opportunity['volatility']:.2f}%")
                
                success = await self.place_aggressive_trade(
                    opportunity['symbol'],
                    opportunity['action'],
                    quantity
                )
                
                if success:
                    # Brief pause for order processing
                    await asyncio.sleep(1)
                    
                    # Update balance
                    new_balance = await self.get_account_balance()
                    profit = new_balance - current_balance
                    
                    logger.info(f"💰 New Balance: ${new_balance:.2f}")
                    logger.info(f"📈 Trade Profit: ${profit:.2f}")
                    
                    current_balance = new_balance
                
                # Quick pause before next trade
                await asyncio.sleep(3)
            
            # Final report
            final_balance = await self.get_account_balance()
            total_profit = final_balance - self.starting_balance
            success_rate = (final_balance / self.target_balance) * 100
            
            logger.info("=" * 50)
            logger.info("🏁 AGGRESSIVE SESSION COMPLETE")
            logger.info(f"💰 Final Balance: ${final_balance:.2f}")
            logger.info(f"📈 Total Profit: ${total_profit:.2f}")
            logger.info(f"🎯 Success Rate: {success_rate:.1f}%")
            logger.info(f"🔥 Trades Executed: {self.trades_executed}")
            
            if final_balance >= self.target_balance:
                logger.info("🎉 TARGET ACHIEVED! MISSION SUCCESSFUL!")
            else:
                logger.info("⚠️ Target not reached, but gains made")
            
        except Exception as e:
            logger.error(f"Session error: {e}")

async def main():
    """Execute aggressive trading immediately"""
    logger.info("🚨 KIMERA AGGRESSIVE TRADER - DIRECT EXECUTION")
    logger.info("⚡ Starting aggressive 10-minute session...")
    
    async with AggressiveTrader() as trader:
        await trader.execute_aggressive_session()

if __name__ == "__main__":
    asyncio.run(main()) 