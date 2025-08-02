#!/usr/bin/env python3
"""
KIMERA ULTRA-AGGRESSIVE AUTONOMOUS TRADER
========================================
🔥 MAXIMUM AGGRESSION MODE 🔥
TARGET: 100% PROFIT
FULL WALLET CONTROL
ULTRA-HIGH FREQUENCY TRADING
"""

import os
import asyncio
import ccxt
import time
import random
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict, List, Any
import json

load_dotenv()

class KimeraUltraAggressiveTrader:
    """ULTRA-AGGRESSIVE TRADING SYSTEM - 100% PROFIT TARGET"""
    
    def __init__(self):
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.secret_key = os.getenv('BINANCE_SECRET_KEY')
        
        self.exchange = ccxt.binance({
            'apiKey': self.api_key,
            'secret': self.secret_key,
            'sandbox': False,
            'enableRateLimit': False,  # MAXIMUM SPEED
        })
        
        # 🔥 ULTRA-AGGRESSIVE PARAMETERS 🔥
        self.profit_target = 1.0        # 100% PROFIT TARGET
        self.max_position_ratio = 0.8   # Use 80% of wallet per trade
        self.min_profit_per_trade = 0.005  # 0.5% minimum profit
        self.max_loss_per_trade = -0.03    # 3% max loss
        self.trade_frequency = 2        # Trade every 2 seconds
        self.max_concurrent_trades = 5  # 5 simultaneous positions
        self.scalping_mode = True       # Ultra-fast scalping
        
        # State tracking
        self.session_start = None
        self.starting_portfolio_value = 0.0
        self.current_portfolio_value = 0.0
        self.total_profit = 0.0
        self.trades_executed = 0
        self.successful_trades = 0
        self.failed_trades = 0
        self.active_positions = {}
        self.trade_history = []
        self.running = False
        
        # Performance metrics
        self.max_drawdown = 0.0
        self.peak_value = 0.0
        self.trades_per_minute = 0
        
        print("🔥" * 60)
        print("🤖 KIMERA ULTRA-AGGRESSIVE AUTONOMOUS TRADER")
        print("🎯 TARGET: 100% PROFIT")
        print("⚡ MAXIMUM AGGRESSION MODE")
        print("💀 FULL WALLET CONTROL")
        print("🔥" * 60)
    
    def get_full_portfolio(self) -> Dict[str, Any]:
        """Get complete portfolio with values"""
        try:
            balance = self.exchange.fetch_balance()
            tickers = self.exchange.fetch_tickers()
            
            portfolio = {}
            total_value = 0.0
            
            for asset, info in balance.items():
                if asset not in ['free', 'used', 'total', 'info'] and isinstance(info, dict):
                    free = float(info.get('free', 0))
                    if free > 0:
                        if asset == 'USDT':
                            usd_value = free
                            price = 1.0
                        else:
                            symbol = f"{asset}/USDT"
                            if symbol in tickers:
                                price = tickers[symbol]['last']
                                usd_value = free * price
                            else:
                                continue
                        
                        portfolio[asset] = {
                            'amount': free,
                            'price': price,
                            'value_usd': usd_value,
                            'tradeable': usd_value >= 5.0
                        }
                        total_value += usd_value
            
            return {'assets': portfolio, 'total_value': total_value}
            
        except Exception as e:
            print(f"❌ Portfolio error: {e}")
            return {'assets': {}, 'total_value': 0.0}
    
    def find_ultra_aggressive_opportunities(self) -> List[Dict]:
        """Find ULTRA-AGGRESSIVE trading opportunities"""
        opportunities = []
        
        try:
            tickers = self.exchange.fetch_tickers()
            portfolio = self.get_full_portfolio()
            
            # 🔥 STRATEGY 1: MOMENTUM SCALPING 🔥
            # Look for ANY price movement to scalp
            scalp_targets = [
                'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'TRX/USDT', 'ADA/USDT',
                'SOL/USDT', 'DOGE/USDT', 'XRP/USDT', 'MATIC/USDT', 'DOT/USDT'
            ]
            
            for symbol in scalp_targets:
                if symbol in tickers:
                    ticker = tickers[symbol]
                    change_1h = ticker.get('percentage', 0)
                    volume = ticker.get('quoteVolume', 0)
                    
                    # ANY movement is opportunity in ultra-aggressive mode
                    if abs(change_1h) > 0.1 and volume > 50000:
                        confidence = min(abs(change_1h) / 2.0, 0.95)
                        
                        opportunities.append({
                            'type': 'SCALP_MOMENTUM',
                            'symbol': symbol,
                            'direction': 'BUY' if change_1h > 0 else 'SELL',
                            'confidence': confidence,
                            'urgency': 0.9,  # ULTRA HIGH URGENCY
                            'expected_profit': abs(change_1h) * 0.3,
                            'risk_level': 0.8,
                            'reason': f"Scalp {change_1h:+.2f}% movement"
                        })
            
            # 🔥 STRATEGY 2: PORTFOLIO ROTATION 🔥
            # Constantly rotate between assets for maximum profit
            for asset, data in portfolio['assets'].items():
                if asset == 'USDT' or not data['tradeable']:
                    continue
                
                symbol = f"{asset}/USDT"
                if symbol in tickers:
                    ticker = tickers[symbol]
                    change_24h = ticker.get('percentage', 0)
                    
                    # Rotate if asset gained >2% or lost >1%
                    if change_24h > 2.0:
                        opportunities.append({
                            'type': 'PROFIT_ROTATION',
                            'symbol': symbol,
                            'direction': 'SELL',
                            'confidence': 0.85,
                            'urgency': 0.8,
                            'expected_profit': change_24h * 0.5,
                            'amount': data['amount'] * 0.7,  # Sell 70%
                            'reason': f"Rotate profit: +{change_24h:.1f}%"
                        })
                    elif change_24h < -1.0:
                        opportunities.append({
                            'type': 'LOSS_ROTATION',
                            'symbol': symbol,
                            'direction': 'SELL',
                            'confidence': 0.7,
                            'urgency': 0.9,
                            'expected_profit': 2.0,  # Expected recovery profit
                            'amount': data['amount'] * 0.5,  # Sell 50%
                            'reason': f"Cut loss: {change_24h:.1f}%"
                        })
            
            # 🔥 STRATEGY 3: WHALE HUNTING 🔥
            # Look for high-volume spikes to ride
            for symbol, ticker in tickers.items():
                if '/USDT' in symbol and symbol.count('/') == 1:
                    volume_24h = ticker.get('quoteVolume', 0)
                    change_1h = ticker.get('percentage', 0)
                    
                    # Hunt for volume spikes with price movement
                    if volume_24h > 10000000 and abs(change_1h) > 1.0:  # >10M volume, >1% move
                        opportunities.append({
                            'type': 'WHALE_HUNT',
                            'symbol': symbol,
                            'direction': 'BUY' if change_1h > 0 else 'SELL',
                            'confidence': 0.9,
                            'urgency': 1.0,  # MAXIMUM URGENCY
                            'expected_profit': abs(change_1h) * 0.6,
                            'reason': f"Whale activity: {change_1h:+.1f}%, Vol: {volume_24h/1000000:.1f}M"
                        })
            
            # Sort by urgency and confidence
            opportunities.sort(key=lambda x: x['urgency'] * x['confidence'], reverse=True)
            
        except Exception as e:
            print(f"❌ Opportunity scan error: {e}")
        
        return opportunities[:10]  # Top 10 opportunities
    
    async def execute_ultra_aggressive_trade(self, opportunity: Dict) -> bool:
        """Execute ULTRA-AGGRESSIVE trade"""
        try:
            symbol = opportunity['symbol']
            direction = opportunity['direction']
            
            print(f"\n🔥 EXECUTING: {opportunity['type']}")
            print(f"   Symbol: {symbol}")
            print(f"   Direction: {direction}")
            print(f"   Confidence: {opportunity['confidence']:.1%}")
            print(f"   Urgency: {opportunity['urgency']:.1%}")
            print(f"   Reason: {opportunity['reason']}")
            
            portfolio = self.get_full_portfolio()
            max_trade_value = portfolio['total_value'] * self.max_position_ratio
            
            if direction == 'BUY':
                # Use available USDT or convert assets
                usdt_available = portfolio['assets'].get('USDT', {}).get('amount', 0)
                
                if usdt_available < 10:
                    # Convert largest non-USDT asset to USDT
                    await self.convert_to_usdt_for_trading(max_trade_value)
                    portfolio = self.get_full_portfolio()
                    usdt_available = portfolio['assets'].get('USDT', {}).get('amount', 0)
                
                if usdt_available >= 6:
                    trade_amount_usdt = min(usdt_available * 0.9, max_trade_value)
                    
                    # Get current price and calculate quantity
                    ticker = self.exchange.fetch_ticker(symbol)
                    price = ticker['last']
                    quantity = trade_amount_usdt / price
                    
                    # Execute BUY order
                    order = self.exchange.create_market_buy_order(symbol, quantity)
                    
                    print(f"   ✅ BOUGHT: {quantity:.6f} {symbol.split('/')[0]}")
                    print(f"   💰 Cost: ${trade_amount_usdt:.2f}")
                    print(f"   📋 Order: {order['id']}")
                    
                    # Track position
                    self.active_positions[symbol] = {
                        'type': 'long',
                        'quantity': quantity,
                        'entry_price': price,
                        'entry_time': time.time(),
                        'target_profit': opportunity['expected_profit'],
                        'strategy': opportunity['type']
                    }
                    
                    self.trades_executed += 1
                    return True
            
            elif direction == 'SELL':
                # Sell specified asset
                base_asset = symbol.split('/')[0]
                
                if 'amount' in opportunity:
                    sell_amount = opportunity['amount']
                else:
                    asset_data = portfolio['assets'].get(base_asset, {})
                    sell_amount = asset_data.get('amount', 0) * 0.7  # Sell 70%
                
                if sell_amount > 0:
                    order = self.exchange.create_market_sell_order(symbol, sell_amount)
                    
                    profit_usd = order.get('cost', 0)
                    
                    print(f"   ✅ SOLD: {sell_amount:.6f} {base_asset}")
                    print(f"   💰 Received: ${profit_usd:.2f}")
                    print(f"   📋 Order: {order['id']}")
                    
                    self.trades_executed += 1
                    self.total_profit += profit_usd * 0.01  # Estimate profit
                    return True
                    
        except Exception as e:
            print(f"   ❌ Trade failed: {e}")
            self.failed_trades += 1
            return False
        
        return False
    
    async def convert_to_usdt_for_trading(self, needed_amount: float):
        """Convert assets to USDT for trading"""
        try:
            portfolio = self.get_full_portfolio()
            
            # Find largest non-USDT asset
            largest_asset = None
            largest_value = 0
            
            for asset, data in portfolio['assets'].items():
                if asset != 'USDT' and data['value_usd'] > largest_value:
                    largest_value = data['value_usd']
                    largest_asset = asset
            
            if largest_asset and largest_value > 10:
                symbol = f"{largest_asset}/USDT"
                convert_amount = min(needed_amount / portfolio['assets'][largest_asset]['price'], 
                                   portfolio['assets'][largest_asset]['amount'] * 0.5)
                
                if convert_amount > 0:
                    order = self.exchange.create_market_sell_order(symbol, convert_amount)
                    print(f"   🔄 Converted {convert_amount:.4f} {largest_asset} to USDT")
                    
        except Exception as e:
            print(f"   ⚠️ Conversion failed: {e}")
    
    def monitor_ultra_aggressive_positions(self):
        """Monitor positions with ULTRA-AGGRESSIVE exit strategy"""
        for symbol, position in list(self.active_positions.items()):
            try:
                ticker = self.exchange.fetch_ticker(symbol)
                current_price = ticker['last']
                entry_price = position['entry_price']
                
                # Calculate P&L
                pnl_pct = (current_price - entry_price) / entry_price
                hold_time = time.time() - position['entry_time']
                
                # ULTRA-AGGRESSIVE exit conditions
                should_exit = False
                reason = ""
                
                # Quick profit taking (even 0.5%)
                if pnl_pct >= self.min_profit_per_trade:
                    should_exit = True
                    reason = f"QUICK_PROFIT ({pnl_pct:.2%})"
                
                # Tight stop loss
                elif pnl_pct <= self.max_loss_per_trade:
                    should_exit = True
                    reason = f"STOP_LOSS ({pnl_pct:.2%})"
                
                # Ultra-fast time exit (30 seconds max)
                elif hold_time > 30:
                    should_exit = True
                    reason = f"TIME_EXIT ({pnl_pct:.2%})"
                
                # Market reversal detection
                elif pnl_pct > 0.01 and hold_time > 10:  # If >1% profit and held >10sec
                    # Check if momentum is reversing
                    change_short = ticker.get('percentage', 0)
                    if (position['type'] == 'long' and change_short < -0.5) or \
                       (position['type'] == 'short' and change_short > 0.5):
                        should_exit = True
                        reason = f"MOMENTUM_REVERSAL ({pnl_pct:.2%})"
                
                if should_exit:
                    # Exit position immediately
                    base_asset = symbol.split('/')[0]
                    portfolio = self.get_full_portfolio()
                    available = portfolio['assets'].get(base_asset, {}).get('amount', 0)
                    
                    if available > 0:
                        order = self.exchange.create_market_sell_order(symbol, available)
                        
                        profit_usd = pnl_pct * position['quantity'] * entry_price
                        self.total_profit += profit_usd
                        
                        if profit_usd > 0:
                            self.successful_trades += 1
                        
                        print(f"   🎯 EXITED {symbol}: {reason}")
                        print(f"   💰 P&L: ${profit_usd:+.2f}")
                        
                        # Record trade
                        self.trade_history.append({
                            'symbol': symbol,
                            'strategy': position['strategy'],
                            'pnl_pct': pnl_pct,
                            'pnl_usd': profit_usd,
                            'hold_time': hold_time,
                            'exit_reason': reason
                        })
                        
                        del self.active_positions[symbol]
                
            except Exception as e:
                print(f"   ⚠️ Position monitoring error: {e}")
    
    async def run_ultra_aggressive_session(self, duration_minutes: int = 10):
        """Run ULTRA-AGGRESSIVE autonomous session"""
        print(f"\n🔥 STARTING ULTRA-AGGRESSIVE SESSION 🔥")
        print(f"⏱️ DURATION: {duration_minutes} MINUTES")
        print(f"🎯 TARGET: 100% PROFIT ({self.profit_target:.0%})")
        print(f"💀 FULL WALLET CONTROL ACTIVATED")
        print("🔥" * 60)
        
        self.session_start = time.time()
        portfolio = self.get_full_portfolio()
        self.starting_portfolio_value = portfolio['total_value']
        self.peak_value = self.starting_portfolio_value
        self.running = True
        
        print(f"💰 Starting Portfolio: ${self.starting_portfolio_value:.2f}")
        print(f"🎯 Target Value: ${self.starting_portfolio_value * (1 + self.profit_target):.2f}")
        
        session_duration = duration_minutes * 60
        last_trade_time = 0
        
        # ULTRA-AGGRESSIVE MAIN LOOP
        while self.running and (time.time() - self.session_start) < session_duration:
            try:
                elapsed = time.time() - self.session_start
                remaining = session_duration - elapsed
                
                # Update portfolio value
                current_portfolio = self.get_full_portfolio()
                self.current_portfolio_value = current_portfolio['total_value']
                
                # Calculate performance metrics
                current_profit = self.current_portfolio_value - self.starting_portfolio_value
                current_profit_pct = (current_profit / self.starting_portfolio_value) * 100
                
                # Update peak and drawdown
                if self.current_portfolio_value > self.peak_value:
                    self.peak_value = self.current_portfolio_value
                
                drawdown = (self.peak_value - self.current_portfolio_value) / self.peak_value * 100
                if drawdown > self.max_drawdown:
                    self.max_drawdown = drawdown
                
                # Calculate trades per minute
                self.trades_per_minute = self.trades_executed / max(elapsed / 60, 0.1)
                
                print(f"\n⚡ Time: {remaining:.0f}s | Portfolio: ${self.current_portfolio_value:.2f} | "
                      f"Profit: {current_profit_pct:+.2f}% | Trades: {self.trades_executed} | "
                      f"Active: {len(self.active_positions)} | TPM: {self.trades_per_minute:.1f}")
                
                # Check if target achieved
                if current_profit_pct >= self.profit_target * 100:
                    print(f"\n🎯 TARGET ACHIEVED! {current_profit_pct:.2f}% PROFIT!")
                    break
                
                # Monitor existing positions
                if self.active_positions:
                    self.monitor_ultra_aggressive_positions()
                
                # Execute new trades (ultra-high frequency)
                if time.time() - last_trade_time >= self.trade_frequency:
                    if len(self.active_positions) < self.max_concurrent_trades:
                        opportunities = self.find_ultra_aggressive_opportunities()
                        
                        # Execute top opportunities
                        for opp in opportunities[:2]:  # Execute top 2
                            if opp['urgency'] > 0.7:  # Only high urgency
                                await self.execute_ultra_aggressive_trade(opp)
                                last_trade_time = time.time()
                                await asyncio.sleep(0.5)  # Brief pause between trades
                
                await asyncio.sleep(1)  # Ultra-fast loop
                
        except Exception as e:
                print(f"⚠️ Trading loop error: {e}")
                await asyncio.sleep(2)
        
        # Session complete
        await self.close_ultra_aggressive_session()
    
    async def close_ultra_aggressive_session(self):
        """Close session and generate report"""
        print(f"\n🔚 CLOSING ULTRA-AGGRESSIVE SESSION...")
        
        # Close all positions
        for symbol in list(self.active_positions.keys()):
            try:
                base_asset = symbol.split('/')[0]
                portfolio = self.get_full_portfolio()
                available = portfolio['assets'].get(base_asset, {}).get('amount', 0)
                
                if available > 0:
                    order = self.exchange.create_market_sell_order(symbol, available)
                    print(f"   ✅ Closed {symbol}")
                    
            except Exception as e:
                print(f"   ⚠️ Error closing {symbol}: {e}")
        
        # Final calculations
        final_portfolio = self.get_full_portfolio()
        final_value = final_portfolio['total_value']
        total_profit = final_value - self.starting_portfolio_value
        total_profit_pct = (total_profit / self.starting_portfolio_value) * 100
        session_time = (time.time() - self.session_start) / 60
        
        # Generate comprehensive report
        print("\n" + "🔥" * 60)
        print("📊 KIMERA ULTRA-AGGRESSIVE SESSION COMPLETE")
        print("🔥" * 60)
        print(f"⏱️ Session Duration: {session_time:.1f} minutes")
        print(f"💰 Starting Value: ${self.starting_portfolio_value:.2f}")
        print(f"💰 Final Value: ${final_value:.2f}")
        print(f"📈 Total Profit: ${total_profit:+.2f}")
        print(f"🎯 Profit Percentage: {total_profit_pct:+.2f}%")
        print(f"📊 Peak Value: ${self.peak_value:.2f}")
        print(f"📉 Max Drawdown: {self.max_drawdown:.2f}%")
        print(f"🔄 Total Trades: {self.trades_executed}")
        print(f"✅ Successful: {self.successful_trades}")
        print(f"❌ Failed: {self.failed_trades}")
        print(f"⚡ Trades/Minute: {self.trades_per_minute:.1f}")
        
        if self.successful_trades > 0:
            win_rate = (self.successful_trades / self.trades_executed) * 100
            print(f"🎯 Win Rate: {win_rate:.1f}%")
        
        print(f"\n🏆 PERFORMANCE RATING:")
        if total_profit_pct >= 100:
            print("🔥🔥🔥 LEGENDARY PERFORMANCE! TARGET ACHIEVED! 🔥🔥🔥")
        elif total_profit_pct >= 50:
            print("🔥🔥 EXCELLENT PERFORMANCE! 🔥🔥")
        elif total_profit_pct >= 20:
            print("🔥 GOOD PERFORMANCE! 🔥")
        elif total_profit_pct >= 0:
            print("✅ PROFITABLE SESSION")
        else:
            print("📊 LEARNING EXPERIENCE")
        
        print("🔥" * 60)

async def main():
    print("🔥 INITIALIZING KIMERA ULTRA-AGGRESSIVE TRADER 🔥")
    
    print("\n" + "⚠️" * 60)
    print("🚨 ULTRA-AGGRESSIVE TRADING MODE")
    print("🎯 TARGET: 100% PROFIT")
    print("💀 FULL WALLET CONTROL")
    print("⚡ MAXIMUM RISK - MAXIMUM REWARD")
    print("🔥 REAL MONEY - REAL CONSEQUENCES")
    print("⚠️" * 60)
    
    response = input("\nActivate ULTRA-AGGRESSIVE mode? (yes/no): ")
    
    if response.lower() == 'yes':
        duration = input("Session duration in minutes (default 10): ")
        try:
            duration_minutes = int(duration) if duration else 10
        except Exception as e:
            logger.error(f"Error in kimera_ultra_aggressive_trader.py: {e}", exc_info=True)
            raise  # Re-raise for proper error handling
            duration_minutes = 10
        
        trader = KimeraUltraAggressiveTrader()
        await trader.run_ultra_aggressive_session(duration_minutes)
    else:
        print("🛑 Ultra-aggressive mode cancelled")

if __name__ == "__main__":
    asyncio.run(main()) 