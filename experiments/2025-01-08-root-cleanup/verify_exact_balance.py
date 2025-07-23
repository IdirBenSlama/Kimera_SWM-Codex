#!/usr/bin/env python3
"""
Exact Balance Verification
=========================

Precise balance verification to check current account status.
"""

import asyncio
import aiohttp
import hmac
import hashlib
import time
import os
from urllib.parse import urlencode

class ExactBalanceChecker:
    def __init__(self):
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.secret_key = os.getenv('BINANCE_SECRET_KEY')
        self.base_url = "https://api.binance.com"
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def _generate_signature(self, params):
        query_string = urlencode(params)
        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    async def get_precise_balance(self):
        """Get precise account balance"""
        try:
            url = f"{self.base_url}/api/v3/account"
            timestamp = int(time.time() * 1000)
            params = {"timestamp": timestamp}
            signature = self._generate_signature(params)
            params["signature"] = signature
            headers = {"X-MBX-APIKEY": self.api_key}
            
            async with self.session.get(url, params=params, headers=headers) as response:
                data = await response.json()
                
                print("🔍 EXACT BALANCE VERIFICATION")
                print("=" * 50)
                
                total_value_usdt = 0
                positions = []
                
                for balance in data.get('balances', []):
                    asset = balance['asset']
                    free = float(balance['free'])
                    locked = float(balance['locked'])
                    total = free + locked
                    
                    if total > 0:
                        if asset == 'USDT':
                            value_usdt = total
                            total_value_usdt += value_usdt
                            print(f"💰 {asset}: {total:.8f} = ${value_usdt:.2f}")
                        else:
                            # Get current price for non-USDT assets
                            try:
                                price = await self.get_price(f"{asset}USDT")
                                value_usdt = total * price
                                total_value_usdt += value_usdt
                                
                                positions.append({
                                    'asset': asset,
                                    'amount': total,
                                    'price': price,
                                    'value': value_usdt
                                })
                                
                                print(f"🎯 {asset}: {total:.8f} @ ${price:.2f} = ${value_usdt:.2f}")
                            except Exception as e:
                                print(f"⚠️ Could not get price for {asset}: {e}")
                
                print("=" * 50)
                print(f"📈 TOTAL PORTFOLIO VALUE: ${total_value_usdt:.2f}")
                print("=" * 50)
                
                # Target analysis
                target_value = 300.0
                starting_value = 50.0
                
                if total_value_usdt >= target_value:
                    print(f"🎉 TARGET ACHIEVED! ${total_value_usdt:.2f} >= ${target_value:.2f}")
                    success_rate = (total_value_usdt / target_value) * 100
                    print(f"🎯 Success Rate: {success_rate:.1f}%")
                else:
                    print(f"⚠️ Target not reached: ${total_value_usdt:.2f} < ${target_value:.2f}")
                    progress = (total_value_usdt / target_value) * 100
                    print(f"📊 Progress: {progress:.1f}% to target")
                
                # Profit calculation
                profit = total_value_usdt - starting_value
                profit_pct = (profit / starting_value) * 100
                print(f"💰 Profit: ${profit:.2f} ({profit_pct:+.1f}%)")
                
                return total_value_usdt
                
        except Exception as e:
            print(f"❌ Error checking balance: {e}")
            return 0
    
    async def get_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        try:
            url = f"{self.base_url}/api/v3/ticker/price"
            params = {"symbol": symbol}
            
            async with self.session.get(url, params=params) as response:
                data = await response.json()
                return float(data['price'])
        except Exception as e:
            print(f"Error getting price for {symbol}: {e}")
            return 0

async def main():
    print("🔍 KIMERA EXACT BALANCE VERIFICATION")
    print("====================================")
    
    async with ExactBalanceChecker() as checker:
        await checker.get_precise_balance()

if __name__ == "__main__":
    asyncio.run(main()) 