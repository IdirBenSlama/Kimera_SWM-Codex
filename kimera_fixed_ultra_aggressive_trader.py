#!/usr/bin/env python3
"""
KIMERA FIXED ULTRA-AGGRESSIVE AUTONOMOUS TRADER
==============================================
🔥 MAXIMUM AGGRESSION MODE - FIXED VERSION 🔥
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
import traceback

load_dotenv()

class KimeraFixedUltraAggressiveTrader:
    """ULTRA-AGGRESSIVE TRADING SYSTEM - FIXED VERSION"""
    
    def __init__(self):
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.secret_key = os.getenv('BINANCE_SECRET_KEY')
        
        self.exchange = ccxt.binance({
            'apiKey': self.api_key,
            'secret': self.secret_key,
            'sandbox': False,
            'enableRateLimit': True,  # Enable rate limiting for stability
        })
        
        # 🔥 ULTRA-AGGRESSIVE PARAMETERS 🔥
        self.profit_target = 1.0        # 100% PROFIT TARGET
        self.max_position_ratio = 0.6   # Use 60% of wallet per trade (more conservative)
        self.min_profit_per_trade = 0.008  # 0.8% minimum profit
        self.max_loss_per_trade = -0.02    # 2% max loss
        self.trade_frequency = 3        # Trade every 3 seconds (more stable)
        self.max_concurrent_trades = 3  # 3 simultaneous positions
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
        print("🤖 KIMERA FIXED ULTRA-AGGRESSIVE TRADER")
        print("🎯 TARGET: 100% PROFIT")
        print("⚡ MAXIMUM AGGRESSION MODE - FIXED")
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
                            'tradeable': usd_value >= 5.0,
                            'symbol': symbol if asset != 'USDT' else None
                        }
                        total_value += usd_value
            
            return {'assets': portfolio, 'total_value': total_value}
            
        except Exception as e:
            print(f"❌ Portfolio error: {e}")
            return {'assets': {}, 'total_value': 0.0}
    
    def find_ultra_aggressive_opportunities(self) -> List[Dict]:
        """Find ULTRA-AGGRESSIVE trading opportunities - FIXED VERSION"""
        opportunities = []
        
        try:
            tickers = self.exchange.fetch_tickers()
            portfolio = self.get_full_portfolio()
            
            # Get current USDT balance
            usdt_balance = portfolio['assets'].get('USDT', {}).get('amount', 0)
            
            # 🔥 STRATEGY 1: MOMENTUM SCALPING 🔥
            scalp_targets = [
                'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'TRX/USDT', 'ADA/USDT',
                'SOL/USDT', 'DOGE/USDT', 'XRP/USDT', 'MATIC/USDT', 'DOT/USDT'
            ]
            
            for symbol in scalp_targets:
                if symbol in tickers:
                    ticker = tickers[symbol]
                    change_1h = ticker.get('percentage', 0)
                    volume = ticker.get('quoteVolume', 0)
                    price = ticker.get('last', 0)
                    
                    # Calculate trade size
                    max_trade_value = portfolio['total_value'] * self.max_position_ratio
                    trade_size = min(usdt_balance * 0.8, max_trade_value, 50)  # Max $50 per trade
                    
                    # Only trade if we have enough USDT and movement is significant
                    if abs(change_1h) > 0.3 and volume > 100000 and trade_size >= 5.1:
                        confidence = min(abs(change_1h) / 3.0, 0.9)
                        
                        opportunities.append({
                            'type': 'SCALP_MOMENTUM',
                            'symbol': symbol,
                            'direction': 'BUY' if change_1h > 0 else 'SELL',
                            'confidence': confidence,
                            'urgency': 0.8,
                            'expected_profit': abs(change_1h) * 0.4,
                            'trade_size_usdt': trade_size,
                            'price': price,
                            'reason': f"Scalp {change_1h:+.2f}% movement"
                        })
            
            # 🔥 STRATEGY 2: PROFIT TAKING 🔥
            for asset, data in portfolio['assets'].items():
                if asset == 'USDT' or not data['tradeable']:
                    continue
                
                symbol = data.get('symbol')
                if symbol and symbol in tickers:
                    ticker = tickers[symbol]
                    change_24h = ticker.get('percentage', 0)
                    
                    # Take profits on gains >3% or cut losses >2%
                    if change_24h > 3.0:
                        sell_amount = data['amount'] * 0.5  # Sell 50%
                        opportunities.append({
                            'type': 'PROFIT_TAKING',
                            'symbol': symbol,
                            'direction': 'SELL',
                            'confidence': 0.9,
                            'urgency': 0.7,
                            'expected_profit': change_24h * 0.3,
                            'sell_amount': sell_amount,
                            'reason': f"Take profit: +{change_24h:.1f}%"
                        })
                    elif change_24h < -2.0:
                        sell_amount = data['amount'] * 0.3  # Sell 30%
                        opportunities.append({
                            'type': 'LOSS_CUTTING',
                            'symbol': symbol,
                            'direction': 'SELL',
                            'confidence': 0.8,
                            'urgency': 0.9,
                            'expected_profit': 1.0,  # Expected recovery
                            'sell_amount': sell_amount,
                            'reason': f"Cut loss: {change_24h:.1f}%"
                        })
            
            # 🔥 STRATEGY 3: HIGH VOLUME HUNTING 🔥
            for symbol, ticker in tickers.items():
                if '/USDT' in symbol and symbol.count('/') == 1:
                    volume_24h = ticker.get('quoteVolume', 0)
                    change_1h = ticker.get('percentage', 0)
                    price = ticker.get('last', 0)
                    
                    # Hunt for massive volume with price movement
                    if volume_24h > 50000000 and abs(change_1h) > 2.0:  # >50M volume, >2% move
                        trade_size = min(usdt_balance * 0.6, 30)  # Max $30 for high-risk trades
                        
                        if trade_size >= 5.1:
                            opportunities.append({
                                'type': 'VOLUME_HUNT',
                                'symbol': symbol,
                                'direction': 'BUY' if change_1h > 0 else 'SELL',
                                'confidence': 0.85,
                                'urgency': 1.0,
                                'expected_profit': abs(change_1h) * 0.5,
                                'trade_size_usdt': trade_size,
                                'price': price,
                                'reason': f"Volume spike: {change_1h:+.1f}%, Vol: {volume_24h/1000000:.0f}M"
                            })
            
            # Sort by urgency and confidence
            opportunities.sort(key=lambda x: x['urgency'] * x['confidence'], reverse=True)
            
            return opportunities[:5]  # Return top 5 opportunities
            
        except Exception as e:
            print(f"❌ Opportunity finding error: {e}")
            return []
    
    async def execute_ultra_aggressive_trade(self, opportunity: Dict) -> bool:
        """Execute ULTRA-AGGRESSIVE trade - FIXED VERSION"""
        try:
            symbol = opportunity['symbol']
            direction = opportunity['direction']
            
            print(f"\n🔥 EXECUTING: {opportunity['type']}")
            print(f"   Symbol: {symbol}")
            print(f"   Direction: {direction}")
            print(f"   Confidence: {opportunity['confidence']:.1%}")
            print(f"   Reason: {opportunity['reason']}")
            
            if direction == 'BUY':
                trade_size_usdt = opportunity.get('trade_size_usdt', 0)
                price = opportunity.get('price', 0)
                
                if trade_size_usdt >= 5.1 and price > 0:
                    # Calculate exact quantity
                    quantity = trade_size_usdt / price
                    
                    # Check minimum quantity requirements
                    market = self.exchange.market(symbol)
                    min_amount = market.get('limits', {}).get('amount', {}).get('min', 0)
                    
                    if quantity >= min_amount:
                        # Execute BUY order
                        order = self.exchange.create_market_buy_order(symbol, quantity)
                        
                        print(f"   ✅ BOUGHT: {quantity:.8f} {symbol.split('/')[0]}")
                        print(f"   💰 Cost: ${trade_size_usdt:.2f}")
                        print(f"   📋 Order: {order['id']}")
                        
                        # Track position
                        self.active_positions[symbol] = {
                            'type': 'long',
                            'quantity': quantity,
                            'entry_price': price,
                            'entry_time': time.time(),
                            'target_profit': opportunity['expected_profit'],
                            'strategy': opportunity['type'],
                            'order_id': order['id']
                        }
                        
                        self.trades_executed += 1
                        return True
                    else:
                        print(f"   ❌ Quantity {quantity:.8f} below minimum {min_amount}")
                else:
                    print(f"   ❌ Invalid trade parameters: size=${trade_size_usdt:.2f}, price=${price:.2f}")
            
            elif direction == 'SELL':
                sell_amount = opportunity.get('sell_amount', 0)
                
                if sell_amount > 0:
                    # Check minimum quantity
                    market = self.exchange.market(symbol)
                    min_amount = market.get('limits', {}).get('amount', {}).get('min', 0)
                    
                    if sell_amount >= min_amount:
                        order = self.exchange.create_market_sell_order(symbol, sell_amount)
                        
                        profit_usd = order.get('cost', 0)
                        
                        print(f"   ✅ SOLD: {sell_amount:.8f} {symbol.split('/')[0]}")
                        print(f"   💰 Received: ${profit_usd:.2f}")
                        print(f"   📋 Order: {order['id']}")
                        
                        self.trades_executed += 1
                        self.total_profit += profit_usd * 0.005  # Conservative profit estimate
                        return True
                    else:
                        print(f"   ❌ Sell amount {sell_amount:.8f} below minimum {min_amount}")
                else:
                    print(f"   ❌ Invalid sell amount: {sell_amount}")
            
        except ccxt.InsufficientFunds as e:
            print(f"   ❌ Insufficient funds: {e}")
            self.failed_trades += 1
        except ccxt.InvalidOrder as e:
            print(f"   ❌ Invalid order: {e}")
            self.failed_trades += 1
        except ccxt.NetworkError as e:
            print(f"   ❌ Network error: {e}")
            self.failed_trades += 1
        except Exception as e:
            print(f"   ❌ Trade failed: {e}")
            print(f"   📊 Error details: {traceback.format_exc()}")
            self.failed_trades += 1
        
        return False
    
    def monitor_ultra_aggressive_positions(self):
        """Monitor positions with ULTRA-AGGRESSIVE exit strategy - FIXED"""
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
                
                # Quick profit taking
                if pnl_pct >= self.min_profit_per_trade:
                    should_exit = True
                    reason = f"PROFIT_TARGET ({pnl_pct:.2%})"
                
                # Stop loss
                elif pnl_pct <= self.max_loss_per_trade:
                    should_exit = True
                    reason = f"STOP_LOSS ({pnl_pct:.2%})"
                
                # Time-based exit (45 seconds max)
                elif hold_time > 45:
                    should_exit = True
                    reason = f"TIME_EXIT ({pnl_pct:.2%})"
                
                # Momentum reversal
                elif pnl_pct > 0.005 and hold_time > 15:  # If >0.5% profit and held >15sec
                    change_short = ticker.get('percentage', 0)
                    if (position['type'] == 'long' and change_short < -0.3) or \
                       (position['type'] == 'short' and change_short > 0.3):
                        should_exit = True
                        reason = f"REVERSAL ({pnl_pct:.2%})"
                
                if should_exit:
                    # Exit position
                    base_asset = symbol.split('/')[0]
                    portfolio = self.get_full_portfolio()
                    available = portfolio['assets'].get(base_asset, {}).get('amount', 0)
                    
                    if available > 0:
                        market = self.exchange.market(symbol)
                        min_amount = market.get('limits', {}).get('amount', {}).get('min', 0)
                        
                        if available >= min_amount:
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
                                'exit_reason': reason,
                                'timestamp': datetime.now().isoformat()
                            })
                            
                            del self.active_positions[symbol]
                        else:
                            print(f"   ⚠️ Cannot exit {symbol}: amount {available:.8f} below minimum {min_amount}")
                
            except Exception as e:
                print(f"   ⚠️ Position monitoring error for {symbol}: {e}")
    
    async def run_ultra_aggressive_session(self, duration_minutes: int = 5):
        """Run ULTRA-AGGRESSIVE autonomous session - FIXED"""
        print(f"\n🔥 STARTING FIXED ULTRA-AGGRESSIVE SESSION 🔥")
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
        loop_count = 0
        
        # ULTRA-AGGRESSIVE MAIN LOOP
        while self.running and (time.time() - self.session_start) < session_duration:
            try:
                loop_count += 1
                elapsed = time.time() - self.session_start
                remaining = session_duration - elapsed
                
                # Update portfolio value every 10 loops
                if loop_count % 10 == 0:
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
                          f"Active: {len(self.active_positions)} | Success: {self.successful_trades}/{self.trades_executed}")
                    
                    # Check if target achieved
                    if current_profit_pct >= self.profit_target * 100:
                        print(f"\n🎯 TARGET ACHIEVED! {current_profit_pct:.2f}% PROFIT!")
                        break
                
                # Monitor existing positions
                if self.active_positions:
                    self.monitor_ultra_aggressive_positions()
                
                # Execute new trades
                if time.time() - last_trade_time >= self.trade_frequency:
                    if len(self.active_positions) < self.max_concurrent_trades:
                        opportunities = self.find_ultra_aggressive_opportunities()
                        
                        if opportunities:
                            # Execute best opportunity
                            best_opp = opportunities[0]
                            if best_opp['urgency'] > 0.6:
                                success = await self.execute_ultra_aggressive_trade(best_opp)
                                if success:
                                    last_trade_time = time.time()
                                await asyncio.sleep(1)  # Brief pause after trade
                
                await asyncio.sleep(2)  # Main loop delay
                
            except KeyboardInterrupt:
                print("\n🛑 MANUAL STOP REQUESTED")
                break
            except Exception as e:
                print(f"⚠️ Trading loop error: {e}")
                await asyncio.sleep(3)
        
        # Session complete
        await self.close_ultra_aggressive_session()
    
    async def close_ultra_aggressive_session(self):
        """Close session and generate report - FIXED"""
        print(f"\n🔚 CLOSING ULTRA-AGGRESSIVE SESSION...")
        
        # Close all positions
        for symbol in list(self.active_positions.keys()):
            try:
                base_asset = symbol.split('/')[0]
                portfolio = self.get_full_portfolio()
                available = portfolio['assets'].get(base_asset, {}).get('amount', 0)
                
                if available > 0:
                    market = self.exchange.market(symbol)
                    min_amount = market.get('limits', {}).get('amount', {}).get('min', 0)
                    
                    if available >= min_amount:
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
        print("📊 KIMERA FIXED ULTRA-AGGRESSIVE SESSION COMPLETE")
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
        
        if self.trades_executed > 0:
            win_rate = (self.successful_trades / self.trades_executed) * 100
            print(f"🎯 Win Rate: {win_rate:.1f}%")
        
        print(f"\n🏆 PERFORMANCE RATING:")
        if total_profit_pct >= 100:
            print("🔥🔥🔥 LEGENDARY! TARGET ACHIEVED! 🔥🔥🔥")
        elif total_profit_pct >= 50:
            print("🔥🔥 EXCELLENT PERFORMANCE! 🔥🔥")
        elif total_profit_pct >= 20:
            print("🔥 GOOD PERFORMANCE! 🔥")
        elif total_profit_pct >= 5:
            print("✅ SOLID GAINS!")
        elif total_profit_pct >= 0:
            print("📊 PROFITABLE SESSION")
        else:
            print("📚 LEARNING EXPERIENCE")
        
        # Save detailed results
        results = {
            'session_start': datetime.fromtimestamp(self.session_start).isoformat(),
            'session_duration_minutes': session_time,
            'starting_value': self.starting_portfolio_value,
            'final_value': final_value,
            'total_profit': total_profit,
            'profit_percentage': total_profit_pct,
            'peak_value': self.peak_value,
            'max_drawdown': self.max_drawdown,
            'total_trades': self.trades_executed,
            'successful_trades': self.successful_trades,
            'failed_trades': self.failed_trades,
            'trades_per_minute': self.trades_per_minute,
            'trade_history': self.trade_history,
            'final_portfolio': final_portfolio
        }
        
        # Save to file
        timestamp = int(time.time())
        filename = f"kimera_ultra_aggressive_results_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Results saved to: {filename}")
        print("🔥" * 60)

async def main():
    print("🔥 INITIALIZING KIMERA FIXED ULTRA-AGGRESSIVE TRADER 🔥")
    
    print("\n" + "⚠️" * 60)
    print("🚨 ULTRA-AGGRESSIVE TRADING MODE - FIXED VERSION")
    print("🎯 TARGET: 100% PROFIT")
    print("💀 FULL WALLET CONTROL")
    print("⚡ MAXIMUM RISK - MAXIMUM REWARD")
    print("🔥 REAL MONEY - REAL CONSEQUENCES")
    print("🛠️ ENHANCED ERROR HANDLING & STABILITY")
    print("⚠️" * 60)
    
    response = input("\nActivate FIXED ULTRA-AGGRESSIVE mode? (yes/no): ")
    
    if response.lower() == 'yes':
        duration = input("Session duration in minutes (default 5): ")
        try:
            duration_minutes = int(duration) if duration else 5
        except:
            duration_minutes = 5
        
        trader = KimeraFixedUltraAggressiveTrader()
        await trader.run_ultra_aggressive_session(duration_minutes)
    else:
        print("🛑 Ultra-aggressive mode cancelled")

if __name__ == "__main__":
    asyncio.run(main()) 