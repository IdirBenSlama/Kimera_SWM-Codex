#!/usr/bin/env python3
"""
KIMERA FIXED AUTONOMOUS TRADING SYSTEM
=====================================
Fixed version that properly handles actual balances and minimum order sizes
"""

import os
import asyncio
import ccxt
import time
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict, List, Any

load_dotenv()

class KimeraFixedAutonomousTrader:
    """Fixed autonomous trading system"""
    
    def __init__(self):
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.secret_key = os.getenv('BINANCE_SECRET_KEY')
        
        self.exchange = ccxt.binance({
            'apiKey': self.api_key,
            'secret': self.secret_key,
            'sandbox': False,
            'enableRateLimit': True,
        })
        
        # Trading parameters
        self.session_duration = 300  # 5 minutes
        self.min_order_value = 6.0   # $6 minimum (above $5 requirement)
        self.max_position_size = 25.0  # $25 max per trade
        self.profit_target = 0.02    # 2% profit target per trade
        self.stop_loss = -0.015      # 1.5% stop loss
        
        # State
        self.session_start = None
        self.starting_value = 0.0
        self.trades_executed = 0
        self.total_profit = 0.0
        self.active_positions = {}
        self.running = False
        
        print("🤖 KIMERA FIXED AUTONOMOUS TRADER")
        print("✅ Configured for your actual balances")
    
    def get_current_balances(self) -> Dict[str, float]:
        """Get current balances"""
        try:
            balance = self.exchange.fetch_balance()
            balances = {}
            
            for asset, info in balance.items():
                if asset not in ['free', 'used', 'total', 'info'] and isinstance(info, dict):
                    free = float(info.get('free', 0))
                    if free > 0:
                        balances[asset] = free
            
            return balances
        except Exception as e:
            print(f"❌ Error getting balances: {e}")
            return {}
    
    def calculate_portfolio_value(self) -> float:
        """Calculate total portfolio value in USD"""
        try:
            balances = self.get_current_balances()
            total_value = 0.0
            
            # Get all tickers
            tickers = self.exchange.fetch_tickers()
            
            for asset, amount in balances.items():
                if asset == 'USDT':
                    total_value += amount
                else:
                    symbol = f"{asset}/USDT"
                    if symbol in tickers:
                        price = tickers[symbol]['last']
                        total_value += amount * price
            
            return total_value
        except Exception as e:
            print(f"❌ Portfolio calculation error: {e}")
            return 0.0
    
    def find_trading_opportunities(self) -> List[Dict]:
        """Find profitable trading opportunities"""
        opportunities = []
        
        try:
            tickers = self.exchange.fetch_tickers()
            balances = self.get_current_balances()
            
            # Strategy 1: Use available USDT to buy trending assets
            usdt_balance = balances.get('USDT', 0)
            
            if usdt_balance >= self.min_order_value:
                # Look for assets with positive momentum
                trending_symbols = ['TRX/USDT', 'ADA/USDT', 'BNB/USDT', 'BTC/USDT']
                
                for symbol in trending_symbols:
                    if symbol in tickers:
                        ticker = tickers[symbol]
                        change_24h = ticker.get('percentage', 0)
                        volume = ticker.get('quoteVolume', 0)
                        
                        # Look for moderate positive momentum with good volume
                        if 0.5 <= change_24h <= 5.0 and volume > 100000:
                            opportunities.append({
                                'type': 'BUY_MOMENTUM',
                                'symbol': symbol,
                                'confidence': min(change_24h / 5.0, 0.9),
                                'reason': f"Positive momentum: +{change_24h:.1f}%",
                                'max_invest': min(usdt_balance * 0.3, self.max_position_size)
                            })
            
            # Strategy 2: Sell assets that have gained significantly
            for asset, amount in balances.items():
                if asset in ['USDT', 'BTC']:  # Keep BTC and USDT
                    continue
                
                symbol = f"{asset}/USDT"
                if symbol in tickers:
                    ticker = tickers[symbol]
                    change_24h = ticker.get('percentage', 0)
                    value_usd = amount * ticker['last']
                    
                    # Sell if asset has gained >3% and we have >$6 worth
                    if change_24h > 3.0 and value_usd >= self.min_order_value:
                        opportunities.append({
                            'type': 'SELL_PROFIT',
                            'symbol': symbol,
                            'confidence': min(change_24h / 10.0, 0.8),
                            'reason': f"Take profit: +{change_24h:.1f}%",
                            'amount': amount * 0.5,  # Sell 50%
                            'value_usd': value_usd * 0.5
                        })
            
            # Sort by confidence
            opportunities.sort(key=lambda x: x['confidence'], reverse=True)
            
        except Exception as e:
            print(f"❌ Error finding opportunities: {e}")
        
        return opportunities[:3]  # Top 3 opportunities
    
    def execute_buy_trade(self, opportunity: Dict) -> bool:
        """Execute a buy trade"""
        try:
            symbol = opportunity['symbol']
            max_invest = opportunity['max_invest']
            
            print(f"\n🚀 EXECUTING BUY: {symbol}")
            print(f"   Reason: {opportunity['reason']}")
            print(f"   Investment: ${max_invest:.2f}")
            
            # Get current price
            ticker = self.exchange.fetch_ticker(symbol)
            price = ticker['last']
            
            # Calculate quantity
            quantity = max_invest / price
            
            # Execute market buy order
            order = self.exchange.create_market_buy_order(symbol, quantity)
            
            print(f"   ✅ BOUGHT: {quantity:.6f} {symbol.split('/')[0]}")
            print(f"   💰 Cost: ${order.get('cost', max_invest):.2f}")
            print(f"   📋 Order ID: {order['id']}")
            
            # Track position
            self.active_positions[symbol] = {
                'type': 'long',
                'quantity': quantity,
                'entry_price': price,
                'entry_time': time.time(),
                'order_id': order['id']
            }
            
            self.trades_executed += 1
            return True
            
        except Exception as e:
            print(f"   ❌ Buy trade failed: {e}")
            return False
    
    def execute_sell_trade(self, opportunity: Dict) -> bool:
        """Execute a sell trade"""
        try:
            symbol = opportunity['symbol']
            amount = opportunity['amount']
            
            print(f"\n💰 EXECUTING SELL: {symbol}")
            print(f"   Reason: {opportunity['reason']}")
            print(f"   Amount: {amount:.6f}")
            
            # Execute market sell order
            order = self.exchange.create_market_sell_order(symbol, amount)
            
            profit = order.get('cost', 0)
            self.total_profit += profit * 0.02  # Estimate 2% profit
            
            print(f"   ✅ SOLD: {amount:.6f} {symbol.split('/')[0]}")
            print(f"   💰 Received: ${profit:.2f}")
            print(f"   📋 Order ID: {order['id']}")
            
            self.trades_executed += 1
            return True
            
        except Exception as e:
            print(f"   ❌ Sell trade failed: {e}")
            return False
    
    def monitor_positions(self):
        """Monitor and exit positions if needed"""
        for symbol, position in list(self.active_positions.items()):
            try:
                ticker = self.exchange.fetch_ticker(symbol)
                current_price = ticker['last']
                entry_price = position['entry_price']
                
                # Calculate profit/loss
                pnl_pct = (current_price - entry_price) / entry_price
                
                # Exit conditions
                should_exit = False
                reason = ""
                
                if pnl_pct >= self.profit_target:
                    should_exit = True
                    reason = f"PROFIT_TARGET ({pnl_pct:.2%})"
                elif pnl_pct <= self.stop_loss:
                    should_exit = True
                    reason = f"STOP_LOSS ({pnl_pct:.2%})"
                elif time.time() - position['entry_time'] > 120:  # 2 minutes max hold
                    should_exit = True
                    reason = f"TIME_EXIT ({pnl_pct:.2%})"
                
                if should_exit:
                    # Get current balance and sell
                    base_asset = symbol.split('/')[0]
                    balances = self.get_current_balances()
                    available = balances.get(base_asset, 0)
                    
                    if available > 0:
                        order = self.exchange.create_market_sell_order(symbol, available)
                        
                        profit_usd = pnl_pct * position['quantity'] * entry_price
                        self.total_profit += profit_usd
                        
                        print(f"   🎯 EXITED {symbol}: {reason}")
                        print(f"   💰 P&L: ${profit_usd:+.2f}")
                        
                        del self.active_positions[symbol]
                
            except Exception as e:
                print(f"   ⚠️ Position monitoring error: {e}")
    
    async def run_autonomous_session(self):
        """Run the autonomous trading session"""
        print("\n" + "=" * 60)
        print("🤖 KIMERA AUTONOMOUS TRADING - FIXED VERSION")
        print("⏱️ DURATION: 5 MINUTES")
        print("💰 WORKING WITH YOUR ACTUAL BALANCES")
        print("=" * 60)
        
        self.session_start = time.time()
        self.starting_value = self.calculate_portfolio_value()
        self.running = True
        
        print(f"💰 Starting Portfolio: ${self.starting_value:.2f}")
        print(f"🎯 Available USDT: ${self.get_current_balances().get('USDT', 0):.2f}")
        
        # Main trading loop
        while self.running and (time.time() - self.session_start) < self.session_duration:
            try:
                elapsed = time.time() - self.session_start
                remaining = self.session_duration - elapsed
                
                print(f"\n⏱️ Time: {remaining:.0f}s | Trades: {self.trades_executed} | Profit: ${self.total_profit:.2f}")
                
                # Monitor existing positions
                if self.active_positions:
                    self.monitor_positions()
                
                # Look for new opportunities
                if len(self.active_positions) < 2:  # Max 2 positions
                    opportunities = self.find_trading_opportunities()
                    
                    for opp in opportunities[:1]:  # Execute best opportunity
                        if opp['type'] == 'BUY_MOMENTUM':
                            self.execute_buy_trade(opp)
                        elif opp['type'] == 'SELL_PROFIT':
                            self.execute_sell_trade(opp)
                        break
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                print(f"⚠️ Trading loop error: {e}")
                await asyncio.sleep(5)
        
        # Session complete
        await self.close_session()
    
    async def close_session(self):
        """Close the trading session"""
        print(f"\n🔚 CLOSING SESSION...")
        
        # Close all positions
        for symbol in list(self.active_positions.keys()):
            try:
                base_asset = symbol.split('/')[0]
                balances = self.get_current_balances()
                available = balances.get(base_asset, 0)
                
                if available > 0:
                    order = self.exchange.create_market_sell_order(symbol, available)
                    print(f"   ✅ Closed {symbol}")
            except Exception as e:
                print(f"   ⚠️ Error closing {symbol}: {e}")
        
        # Final report
        final_value = self.calculate_portfolio_value()
        total_profit = final_value - self.starting_value
        profit_pct = (total_profit / self.starting_value) * 100
        
        print("\n" + "=" * 60)
        print("📊 KIMERA AUTONOMOUS SESSION COMPLETE")
        print("=" * 60)
        print(f"💰 Starting Value: ${self.starting_value:.2f}")
        print(f"💰 Final Value: ${final_value:.2f}")
        print(f"📈 Total Profit: ${total_profit:+.2f}")
        print(f"📊 Profit %: {profit_pct:+.2f}%")
        print(f"🔄 Trades: {self.trades_executed}")
        print("=" * 60)

async def main():
    print("🚀 STARTING KIMERA FIXED AUTONOMOUS TRADER...")
    
    response = input("\nStart 5-minute autonomous trading session? (yes/no): ")
    
    if response.lower() == 'yes':
        trader = KimeraFixedAutonomousTrader()
        await trader.run_autonomous_session()
    else:
        print("🛑 Session cancelled")

if __name__ == "__main__":
    asyncio.run(main()) 