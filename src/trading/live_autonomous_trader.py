#!/usr/bin/env python3
"""
KIMERA LIVE AUTONOMOUS TRADER
=============================

Real autonomous trader that places actual orders on Coinbase via CDP API.
NO SIMULATION - REAL MONEY TRADING.
"""

import os
import sys
import json
import asyncio
import requests
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from .autonomous_kimera_trader import (
    KimeraAutonomousTrader, 
    CognitiveSignal, 
    TradingStrategy, 
    MarketRegime,
    create_autonomous_kimera
)

import logging
logger = logging.getLogger('KIMERA_LIVE')

@dataclass
class LivePosition:
    """Real position on Coinbase"""
    symbol: str
    side: str
    amount_eur: float
    amount_crypto: float
    entry_price: float
    order_id: str
    status: str
    cdp_account_id: str
    timestamp: datetime

class LiveCDPTrader(KimeraAutonomousTrader):
    """
    Live autonomous trader that places real orders via CDP API
    """
    
    def __init__(self, api_key: str, target_eur: float = 100.0):
        super().__init__(api_key, target_eur)
        
        # CDP API configuration
        self.cdp_api_key = api_key
        self.cdp_base_url = "https://api.coinbase.com/v2/"
        self.live_positions: Dict[str, LivePosition] = {}
        
        # Real account balance
        self.real_balance_eur = 0.0
        
        logger.info("🔥 LIVE CDP TRADER INITIALIZED - REAL MONEY TRADING")
        logger.info(f"   API Key: {api_key[:8]}...")
        logger.info(f"   Target: EUR {target_eur}")
        logger.info("   WARNING: ALL TRADES ARE REAL")
    
    async def get_real_account_balance(self) -> float:
        """Get real EUR balance from Coinbase account"""
        try:
            headers = {
                'Authorization': f'Bearer {self.cdp_api_key}',
                'Content-Type': 'application/json'
            }
            
            response = requests.get(f"{self.cdp_base_url}accounts", headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                
                for account in data.get('data', []):
                    if account.get('currency') == 'EUR':
                        balance = float(account.get('balance', {}).get('amount', 0))
                        self.real_balance_eur = balance
                        self.portfolio_value = balance
                        
                        logger.info(f"💰 Real EUR balance: €{balance:.2f}")
                        return balance
            
            logger.warning(f"⚠️ Failed to get account balance: {response.status_code}")
            return 0.0
            
        except Exception as e:
            logger.error(f"❌ Failed to get real balance: {e}")
            return 0.0
    
    def get_cdp_symbol(self, symbol: str) -> str:
        """Convert internal symbol to CDP trading pair"""
        symbol_map = {
            'bitcoin': 'BTC-EUR',
            'ethereum': 'ETH-EUR', 
            'solana': 'SOL-EUR',
            'cardano': 'ADA-EUR',
            'polkadot': 'DOT-EUR'
        }
        return symbol_map.get(symbol, f"{symbol.upper()}-EUR")
    
    async def place_real_order(self, signal: CognitiveSignal) -> Optional[str]:
        """Place real order on Coinbase via CDP API"""
        try:
            # Get real balance first
            await self.get_real_account_balance()
            
            if self.real_balance_eur < 1.0:
                logger.warning("⚠️ Insufficient EUR balance for trading")
                return None
            
            # Calculate position size based on real balance
            position_eur = min(
                self.real_balance_eur * signal.suggested_allocation_pct,
                self.real_balance_eur * 0.8  # Max 80% of real balance
            )
            
            if position_eur < 0.1:  # Minimum €0.10 trade
                logger.warning("⚠️ Position size too small for real trading")
                return None
            
            cdp_symbol = self.get_cdp_symbol(signal.symbol)
            
            # Prepare order data
            order_data = {
                'type': 'market',  # Market order for immediate execution
                'side': signal.action,  # 'buy' or 'sell'
                'product_id': cdp_symbol,
                'funds': str(position_eur) if signal.action == 'buy' else None,
                'size': str(position_eur / signal.entry_price) if signal.action == 'sell' else None
            }
            
            headers = {
                'Authorization': f'Bearer {self.cdp_api_key}',
                'Content-Type': 'application/json'
            }
            
            logger.info(f"🚀 PLACING REAL ORDER:")
            logger.info(f"   Symbol: {cdp_symbol}")
            logger.info(f"   Side: {signal.action}")
            logger.info(f"   Amount: €{position_eur:.2f}")
            logger.info(f"   Strategy: {signal.strategy.value}")
            logger.info(f"   Confidence: {signal.confidence:.2f}")
            
            # Place the actual order
            response = requests.post(
                f"{self.cdp_base_url}orders",
                headers=headers,
                json=order_data
            )
            
            if response.status_code in [200, 201]:
                order = response.json().get('data', {})
                order_id = order.get('id')
                
                logger.info(f"✅ REAL ORDER PLACED SUCCESSFULLY")
                logger.info(f"   Order ID: {order_id}")
                logger.info(f"   Status: {order.get('status')}")
                
                # Create live position record
                live_position = LivePosition(
                    symbol=signal.symbol,
                    side=signal.action,
                    amount_eur=position_eur,
                    amount_crypto=position_eur / signal.entry_price,
                    entry_price=signal.entry_price,
                    order_id=order_id,
                    status=order.get('status', 'pending'),
                    cdp_account_id=order.get('account_id', ''),
                    timestamp=datetime.now()
                )
                
                self.live_positions[signal.symbol] = live_position
                
                # Update portfolio with real balance
                await self.get_real_account_balance()
                
                return order_id
            
            else:
                logger.error(f"❌ Order failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to place real order: {e}")
            return None
    
    async def execute_autonomous_trade(self, signal: CognitiveSignal) -> bool:
        """Execute real autonomous trade via CDP API"""
        try:
            logger.info(f"🔥 EXECUTING REAL AUTONOMOUS TRADE:")
            logger.info(f"   Symbol: {signal.symbol}")
            logger.info(f"   Action: {signal.action}")
            logger.info(f"   Confidence: {signal.confidence:.2f}")
            logger.info(f"   Conviction: {signal.conviction:.2f}")
            logger.info("   THIS IS REAL MONEY TRADING")
            
            # Place real order
            order_id = await self.place_real_order(signal)
            
            if order_id:
                # Update statistics
                self.total_trades += 1
                
                # Save state
                self._save_autonomous_state()
                
                logger.info(f"✅ REAL TRADE EXECUTED - Order ID: {order_id}")
                return True
            else:
                logger.error("❌ Failed to place real order")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to execute real trade: {e}")
            return False
    
    async def check_order_status(self, order_id: str) -> Dict[str, Any]:
        """Check status of real order"""
        try:
            headers = {
                'Authorization': f'Bearer {self.cdp_api_key}',
                'Content-Type': 'application/json'
            }
            
            response = requests.get(
                f"{self.cdp_base_url}orders/{order_id}",
                headers=headers
            )
            
            if response.status_code == 200:
                return response.json().get('data', {})
            
            return {}
            
        except Exception as e:
            logger.error(f"❌ Failed to check order status: {e}")
            return {}
    
    async def manage_live_positions(self):
        """Manage real positions on Coinbase"""
        try:
            for symbol, position in list(self.live_positions.items()):
                # Check order status
                order_status = await self.check_order_status(position.order_id)
                
                if order_status.get('status') == 'done':
                    # Order completed
                    logger.info(f"✅ Order completed for {symbol}: {position.order_id}")
                    
                    # Update real balance
                    await self.get_real_account_balance()
                    
                elif order_status.get('status') == 'cancelled':
                    # Order cancelled
                    logger.warning(f"⚠️ Order cancelled for {symbol}: {position.order_id}")
                    del self.live_positions[symbol]
                
                # Could add more sophisticated position management here
                # like trailing stops, partial profit taking, etc.
                
        except Exception as e:
            logger.error(f"❌ Failed to manage live positions: {e}")
    
    async def autonomous_trading_cycle(self):
        """Enhanced autonomous cycle with real trading"""
        logger.info("🔥 Starting LIVE autonomous trading cycle...")
        
        # Update real balance first
        await self.get_real_account_balance()
        
        if self.real_balance_eur < 0.5:
            logger.warning("⚠️ Insufficient balance for trading")
            return False
        
        # Symbols to trade
        symbols = ['bitcoin', 'ethereum', 'solana', 'cardano', 'polkadot']
        
        try:
            # Fetch market data for all symbols
            for symbol in symbols:
                await self.fetch_market_data(symbol)
            
            # Manage existing live positions
            await self.manage_live_positions()
            
            # Generate new signals if not at position limit
            if len(self.live_positions) < 3:  # Max 3 concurrent live positions
                for symbol in symbols:
                    if symbol not in self.live_positions:
                        signal = self.generate_cognitive_signal(symbol)
                        
                        if signal and signal.confidence > 0.7:  # Higher threshold for real money
                            await self.execute_autonomous_trade(signal)
                            break  # Only one new real position per cycle
            
            # Update portfolio value with real balance
            await self.get_real_account_balance()
            
            # Log current status
            status = await self.get_portfolio_status()
            logger.info(f"🔥 LIVE Portfolio Status:")
            logger.info(f"   Real Balance: €{self.real_balance_eur:.2f}")
            logger.info(f"   Progress: {status['progress_pct']:.1f}%")
            logger.info(f"   Active Live Positions: {len(self.live_positions)}")
            logger.info(f"   Total Trades: {self.total_trades}")
            
            # Check if target reached
            if self.real_balance_eur >= self.target_eur:
                logger.info(f"🎉 TARGET REACHED! Real Balance: €{self.real_balance_eur:.2f}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Live autonomous trading cycle failed: {e}")
            return False

def create_live_autonomous_kimera(api_key: str, target_eur: float = 100.0) -> LiveCDPTrader:
    """Create live autonomous Kimera trader instance"""
    return LiveCDPTrader(api_key, target_eur)

if __name__ == "__main__":
    # Example usage - REAL MONEY TRADING
    API_KEY = os.getenv("CDP_API_KEY_NAME", "")
    
    trader = create_live_autonomous_kimera(API_KEY, target_eur=100.0)
    
    # Run live autonomous trader
    asyncio.run(trader.run_autonomous_trader(cycle_interval_minutes=15)) 