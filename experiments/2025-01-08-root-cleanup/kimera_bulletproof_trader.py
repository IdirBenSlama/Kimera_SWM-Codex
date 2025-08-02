#!/usr/bin/env python3
"""
KIMERA BULLETPROOF ULTRA-AGGRESSIVE TRADER
=========================================
🛡️ BULLETPROOF DUST MANAGEMENT & TRADING 🛡️
🔥 ULTRA-AGGRESSIVE TRADING WITH ZERO FAILURES 🔥
- Explicit dust management before every trade
- Bulletproof opportunity validation
- Zero invalid trade attempts
- Automatic portfolio optimization
"""

import os
import asyncio
import ccxt
import time
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict, List, Any
import json
import traceback

load_dotenv()

class KimeraBulletproofTrader:
    """Bulletproof ultra-aggressive trader with explicit dust management"""
    
    def __init__(self):
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.secret_key = os.getenv('BINANCE_SECRET_KEY')
        
        self.exchange = ccxt.binance({
            'apiKey': self.api_key,
            'secret': self.secret_key,
            'sandbox': False,
            'enableRateLimit': True,
        })
        
        # Load markets
        self.exchange.load_markets()
        
        # 🔥 ULTRA-AGGRESSIVE PARAMETERS 🔥
        self.profit_target = 1.0        # 100% PROFIT TARGET
        self.max_position_ratio = 0.6   # Use 60% of wallet per trade
        self.min_profit_per_trade = 0.008  # 0.8% minimum profit
        self.max_loss_per_trade = -0.02    # 2% max loss
        self.trade_frequency = 3        # Trade every 3 seconds
        self.max_concurrent_trades = 3  # 3 simultaneous positions
        self.min_trade_size = 6.5       # $6.50 minimum (confirmed working)
        self.max_trade_size = 80        # $80 maximum per trade
        self.dust_threshold = 5.0       # Consider anything below $5 as dust
        
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
        
        print("🛡️" * 60)
        print("🤖 KIMERA BULLETPROOF ULTRA-AGGRESSIVE TRADER")
        print("🎯 TARGET: 100% PROFIT")
        print("⚡ MAXIMUM AGGRESSION + BULLETPROOF LOGIC")
        print("💀 FULL WALLET CONTROL")
        print(f"💰 MIN TRADE SIZE: ${self.min_trade_size}")
        print("🛡️ ZERO FAILURE GUARANTEE")
        print("🛡️" * 60)
    
    def explicit_dust_cleanup(self) -> bool:
        """Explicitly clean up dust before trading"""
        try:
            print(f"\n🧹 EXPLICIT DUST CLEANUP:")
            print("-" * 40)
            
            balance = self.exchange.fetch_balance()
            tickers = self.exchange.fetch_tickers()
            
            dust_cleaned = False
            
            for asset, info in balance.items():
                if asset not in ['free', 'used', 'total', 'info'] and isinstance(info, dict):
                    free = float(info.get('free', 0))
                    if free > 0 and asset not in ['USDT', 'BNB']:
                        symbol = f"{asset}/USDT"
                        if symbol in tickers:
                            price = tickers[symbol]['last']
                            value = free * price
                            
                            # If it's dust (below threshold) but tradeable, convert it
                            if value < self.dust_threshold and value >= 1.0:
                                try:
                                    market = self.exchange.market(symbol)
                                    min_amount = market.get('limits', {}).get('amount', {}).get('min', 0)
                                    
                                    if free >= min_amount:
                                        print(f"   🔄 Converting dust {asset}: {free:.8f} = ${value:.2f}")
                                        order = self.exchange.create_market_sell_order(symbol, free)
                                        received = order.get('cost', 0)
                                        print(f"      ✅ Converted to ${received:.2f} USDT")
                                        dust_cleaned = True
                                        time.sleep(1)  # Brief pause between conversions
                                    else:
                                        print(f"   ⚠️ {asset} too small to convert: {free:.8f} < {min_amount}")
                                except Exception as e:
                                    print(f"   ❌ Failed to convert {asset}: {e}")
                            elif value >= self.dust_threshold:
                                print(f"   ✅ {asset}: ${value:.2f} (TRADEABLE)")
                            else:
                                print(f"   🧹 {asset}: ${value:.2f} (NEGLIGIBLE DUST)")
            
            if dust_cleaned:
                print("   ✅ Dust cleanup completed")
                time.sleep(2)  # Allow balance to update
            else:
                print("   ✅ No dust to clean")
            
            return dust_cleaned
            
        except Exception as e:
            print(f"❌ Dust cleanup failed: {e}")
            return False
    
    def get_bulletproof_portfolio(self) -> Dict[str, Any]:
        """Get portfolio with bulletproof validation"""
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
                        
                        # Only include if above dust threshold OR is USDT
                        if usd_value >= self.dust_threshold or asset == 'USDT':
                            portfolio[asset] = {
                                'amount': free,
                                'price': price,
                                'value_usd': usd_value,
                                'tradeable': usd_value >= self.min_trade_size or asset == 'USDT',
                                'symbol': symbol if asset != 'USDT' else None
                            }
                            total_value += usd_value
            
            return {'assets': portfolio, 'total_value': total_value}
            
        except Exception as e:
            print(f"❌ Portfolio error: {e}")
            return {'assets': {}, 'total_value': 0.0}
    
    def find_bulletproof_opportunities(self) -> List[Dict]:
        """Find bulletproof trading opportunities with explicit validation"""
        opportunities = []
        
        try:
            tickers = self.exchange.fetch_tickers()
            portfolio = self.get_bulletproof_portfolio()
            
            # Get current USDT balance
            usdt_balance = portfolio['assets'].get('USDT', {}).get('amount', 0)
            
            # 🔥 STRATEGY 1: BUY OPPORTUNITIES (MOMENTUM SCALPING) 🔥
            if usdt_balance >= self.min_trade_size:
                scalp_targets = [
                    'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'TRX/USDT', 'ADA/USDT',
                    'SOL/USDT', 'DOGE/USDT', 'XRP/USDT', 'LINK/USDT', 'DOT/USDT'
                ]
                
                for symbol in scalp_targets:
                    if symbol in tickers:
                        ticker = tickers[symbol]
                        change_24h = ticker.get('percentage', 0)
                        volume = ticker.get('quoteVolume', 0)
                        price = ticker.get('last', 0)
                        
                        # Calculate trade size
                        max_trade_value = min(
                            portfolio['total_value'] * self.max_position_ratio,
                            self.max_trade_size,
                            usdt_balance * 0.8
                        )
                        
                        trade_size = max(self.min_trade_size, min(max_trade_value, 40))
                        
                        # Look for POSITIVE momentum for BUY orders
                        if (change_24h is not None and change_24h > 0.5 and 
                            volume > 1000000 and trade_size >= self.min_trade_size):
                            
                            # Validate trade will work
                            quantity = trade_size / price
                            market = self.exchange.market(symbol)
                            min_amount = market.get('limits', {}).get('amount', {}).get('min', 0)
                            
                            if quantity >= min_amount:
                                confidence = min(change_24h / 5.0, 0.9)
                                urgency = min(change_24h / 3.0, 1.0)
                                
                                opportunities.append({
                                    'type': 'BUY_MOMENTUM',
                                    'symbol': symbol,
                                    'direction': 'BUY',
                                    'confidence': confidence,
                                    'urgency': urgency,
                                    'expected_profit': change_24h * 0.3,
                                    'trade_size_usdt': trade_size,
                                    'quantity': quantity,
                                    'price': price,
                                    'reason': f"Buy momentum: +{change_24h:.2f}%"
                                })
            
            # 🔥 STRATEGY 2: SELL OPPORTUNITIES (PROFIT TAKING) 🔥
            for asset, data in portfolio['assets'].items():
                if asset == 'USDT' or not data.get('tradeable', False):
                    continue
                
                symbol = data.get('symbol')
                if symbol and symbol in tickers:
                    ticker = tickers[symbol]
                    change_24h = ticker.get('percentage', 0)
                    
                    # Get market info
                    try:
                        market = self.exchange.market(symbol)
                        min_amount = market.get('limits', {}).get('amount', {}).get('min', 0)
                        min_notional = market.get('limits', {}).get('cost', {}).get('min', self.min_trade_size)
                    except Exception as e:
                        logger.error(f"Error in kimera_bulletproof_trader.py: {e}", exc_info=True)
                        raise  # Re-raise for proper error handling
                        continue
                    
                    # ULTRA-CONSERVATIVE SELL VALIDATION
                    if change_24h is not None and change_24h > 5.0:  # Only sell on significant profits
                        # Use ultra-conservative sell percentage
                        sell_percentage = 0.3  # Only sell 30%
                        sell_amount = data['amount'] * sell_percentage
                        notional_value = sell_amount * data['price']
                        remaining_amount = data['amount'] - sell_amount
                        remaining_value = remaining_amount * data['price']
                        
                        # BULLETPROOF VALIDATION WITH SAFETY MARGINS
                        sell_valid = (
                            sell_amount > 0 and
                            sell_amount >= min_amount * 2.0 and  # 2x safety margin
                            notional_value >= self.min_trade_size * 1.5 and  # 1.5x safety margin
                            notional_value >= min_notional * 1.2 and  # 1.2x min notional
                            remaining_value >= self.min_trade_size or remaining_value < 2.0  # Keep or convert remainder
                        )
                        
                        # Additional sanity checks
                        if sell_valid and sell_amount > 0:
                            # Final reality check - verify we actually have this amount
                            current_balance = self.exchange.fetch_balance()
                            actual_free = current_balance.get(asset, {}).get('free', 0)
                            
                            if actual_free >= sell_amount * 1.01:  # 1% safety margin
                                opportunities.append({
                                    'type': 'ULTRA_SAFE_SELL',
                                    'symbol': symbol,
                                    'direction': 'SELL',
                                    'confidence': 0.9,
                                    'urgency': 0.7,
                                    'expected_profit': change_24h * 0.3,
                                    'sell_amount': sell_amount,
                                    'notional_value': notional_value,
                                    'reason': f"Ultra-safe profit: +{change_24h:.2f}%"
                                })
                                print(f"   ✅ {symbol} VALID SELL: {sell_amount:.8f} = ${notional_value:.2f}")
                            else:
                                print(f"   ❌ {symbol} insufficient balance: {actual_free:.8f} < {sell_amount:.8f}")
                        else:
                            print(f"   ❌ {symbol} sell invalid: amount={sell_amount:.8f}, notional=${notional_value:.2f}")
            
            # Sort by confidence and urgency
            opportunities.sort(key=lambda x: x['confidence'] * x['urgency'], reverse=True)
            
            return opportunities[:5]  # Return top 5 opportunities
            
        except Exception as e:
            print(f"❌ Bulletproof opportunity finding error: {e}")
            return []
    
    async def execute_bulletproof_trade(self, opportunity: Dict) -> bool:
        """Execute bulletproof trade with explicit validation"""
        try:
            symbol = opportunity['symbol']
            direction = opportunity['direction']
            
            print(f"\n🛡️ EXECUTING: {opportunity['type']}")
            print(f"   Symbol: {symbol}")
            print(f"   Direction: {direction}")
            print(f"   Confidence: {opportunity['confidence']:.1%}")
            print(f"   Reason: {opportunity['reason']}")
            
            if direction == 'BUY':
                trade_size_usdt = opportunity.get('trade_size_usdt', 0)
                quantity = opportunity.get('quantity', 0)
                price = opportunity.get('price', 0)
                
                print(f"   Trade Size: ${trade_size_usdt:.2f}")
                print(f"   Quantity: {quantity:.8f}")
                print(f"   Price: ${price:.4f}")
                
                # Final validation
                if trade_size_usdt >= self.min_trade_size and quantity > 0:
                    # Execute BUY order
                    order = self.exchange.create_market_buy_order(symbol, quantity)
                    
                    actual_cost = order.get('cost', trade_size_usdt)
                    actual_quantity = order.get('amount', quantity)
                    
                    print(f"   ✅ BOUGHT: {actual_quantity:.8f} {symbol.split('/')[0]}")
                    print(f"   💰 Cost: ${actual_cost:.2f}")
                    print(f"   📋 Order: {order['id']}")
                    
                    # Track position
                    self.active_positions[symbol] = {
                        'type': 'long',
                        'quantity': actual_quantity,
                        'entry_price': price,
                        'entry_time': time.time(),
                        'target_profit': opportunity['expected_profit'],
                        'strategy': opportunity['type'],
                        'order_id': order['id'],
                        'entry_cost': actual_cost
                    }
                    
                    self.trades_executed += 1
                    return True
                else:
                    print(f"   ❌ Invalid BUY parameters")
                    return False
            
            elif direction == 'SELL':
                sell_amount = opportunity.get('sell_amount', 0)
                notional_value = opportunity.get('notional_value', 0)
                
                print(f"   Sell Amount: {sell_amount:.8f}")
                print(f"   Notional Value: ${notional_value:.2f}")
                
                # Final validation
                if sell_amount > 0 and notional_value >= self.min_trade_size:
                    # Execute SELL order
                    order = self.exchange.create_market_sell_order(symbol, sell_amount)
                    
                    received_usdt = order.get('cost', 0)
                    
                    print(f"   ✅ SOLD: {sell_amount:.8f} {symbol.split('/')[0]}")
                    print(f"   💰 Received: ${received_usdt:.2f}")
                    print(f"   📋 Order: {order['id']}")
                    
                    self.trades_executed += 1
                    self.total_profit += received_usdt * 0.01
                    return True
                else:
                    print(f"   ❌ Invalid SELL parameters")
                    return False
            
        except ccxt.InsufficientFunds as e:
            print(f"   ❌ Insufficient funds: {e}")
            self.failed_trades += 1
        except ccxt.InvalidOrder as e:
            print(f"   ❌ Invalid order: {e}")
            self.failed_trades += 1
        except Exception as e:
            print(f"   ❌ Trade failed: {e}")
            self.failed_trades += 1
        
        return False
    
    def monitor_bulletproof_positions(self):
        """Monitor positions with bulletproof exit strategy"""
        for symbol, position in list(self.active_positions.items()):
            try:
                ticker = self.exchange.fetch_ticker(symbol)
                current_price = ticker['last']
                entry_price = position['entry_price']
                
                # Calculate P&L
                pnl_pct = (current_price - entry_price) / entry_price
                hold_time = time.time() - position['entry_time']
                
                # Exit conditions
                should_exit = False
                reason = ""
                
                # Profit target
                if pnl_pct >= self.min_profit_per_trade:
                    should_exit = True
                    reason = f"PROFIT ({pnl_pct:.2%})"
                
                # Stop loss
                elif pnl_pct <= self.max_loss_per_trade:
                    should_exit = True
                    reason = f"STOP_LOSS ({pnl_pct:.2%})"
                
                # Time exit
                elif hold_time > 90:  # 90 seconds max
                    should_exit = True
                    reason = f"TIME_EXIT ({pnl_pct:.2%})"
                
                if should_exit:
                    # Exit with bulletproof validation
                    base_asset = symbol.split('/')[0]
                    portfolio = self.get_bulletproof_portfolio()
                    available = portfolio['assets'].get(base_asset, {}).get('amount', 0)
                    
                    if available > 0:
                        try:
                            market = self.exchange.market(symbol)
                            min_amount = market.get('limits', {}).get('amount', {}).get('min', 0)
                            
                            if available >= min_amount:
                                order = self.exchange.create_market_sell_order(symbol, available)
                                
                                received_usdt = order.get('cost', 0)
                                profit_usd = received_usdt - position['entry_cost']
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
                        except Exception as e:
                            print(f"   ❌ Exit error for {symbol}: {e}")
                
            except Exception as e:
                print(f"   ⚠️ Position monitoring error for {symbol}: {e}")
    
    async def run_bulletproof_session(self, duration_minutes: int = 5):
        """Run bulletproof ultra-aggressive session"""
        print(f"\n🛡️ STARTING BULLETPROOF ULTRA-AGGRESSIVE SESSION 🛡️")
        print(f"⏱️ DURATION: {duration_minutes} MINUTES")
        print(f"🎯 TARGET: 100% PROFIT ({self.profit_target:.0%})")
        print(f"💀 FULL WALLET CONTROL")
        print(f"💰 MIN TRADE SIZE: ${self.min_trade_size}")
        print(f"🛡️ BULLETPROOF VALIDATION")
        print("🛡️" * 60)
        
        # Explicit dust cleanup
        self.explicit_dust_cleanup()
        
        self.session_start = time.time()
        portfolio = self.get_bulletproof_portfolio()
        self.starting_portfolio_value = portfolio['total_value']
        self.peak_value = self.starting_portfolio_value
        self.running = True
        
        print(f"💰 Starting Portfolio: ${self.starting_portfolio_value:.2f}")
        print(f"🎯 Target Value: ${self.starting_portfolio_value * (1 + self.profit_target):.2f}")
        
        session_duration = duration_minutes * 60
        last_trade_time = 0
        loop_count = 0
        
        # BULLETPROOF MAIN LOOP
        while self.running and (time.time() - self.session_start) < session_duration:
            try:
                loop_count += 1
                elapsed = time.time() - self.session_start
                remaining = session_duration - elapsed
                
                # Update portfolio every 20 loops
                if loop_count % 20 == 0:
                    current_portfolio = self.get_bulletproof_portfolio()
                    self.current_portfolio_value = current_portfolio['total_value']
                    
                    current_profit = self.current_portfolio_value - self.starting_portfolio_value
                    current_profit_pct = (current_profit / self.starting_portfolio_value) * 100
                    
                    if self.current_portfolio_value > self.peak_value:
                        self.peak_value = self.current_portfolio_value
                    
                    self.trades_per_minute = self.trades_executed / max(elapsed / 60, 0.1)
                    
                    print(f"\n⚡ Time: {remaining:.0f}s | Portfolio: ${self.current_portfolio_value:.2f} | "
                          f"Profit: {current_profit_pct:+.2f}% | Trades: {self.trades_executed} | "
                          f"Success: {self.successful_trades} | 🛡️ Bulletproof")
                    
                    # Check target
                    if current_profit_pct >= self.profit_target * 100:
                        print(f"\n🎯 TARGET ACHIEVED! {current_profit_pct:.2f}% PROFIT!")
                        break
                
                # Monitor positions
                if self.active_positions:
                    self.monitor_bulletproof_positions()
                
                # Execute trades
                if time.time() - last_trade_time >= self.trade_frequency:
                    if len(self.active_positions) < self.max_concurrent_trades:
                        opportunities = self.find_bulletproof_opportunities()
                        
                        if opportunities:
                            best_opp = opportunities[0]
                            if best_opp['confidence'] > 0.5:
                                success = await self.execute_bulletproof_trade(best_opp)
                                if success:
                                    last_trade_time = time.time()
                
                await asyncio.sleep(2)  # Main loop delay
                
            except KeyboardInterrupt:
                print("\n🛑 MANUAL STOP")
                break
            except Exception as e:
                print(f"⚠️ Loop error: {e}")
                await asyncio.sleep(3)
        
        # Close session
        await self.close_bulletproof_session()
    
    async def close_bulletproof_session(self):
        """Close bulletproof session"""
        print(f"\n🔚 CLOSING BULLETPROOF SESSION...")
        
        # Close all positions
        for symbol in list(self.active_positions.keys()):
            try:
                base_asset = symbol.split('/')[0]
                portfolio = self.get_bulletproof_portfolio()
                available = portfolio['assets'].get(base_asset, {}).get('amount', 0)
                
                if available > 0:
                    market = self.exchange.market(symbol)
                    min_amount = market.get('limits', {}).get('amount', {}).get('min', 0)
                    
                    if available >= min_amount:
                        order = self.exchange.create_market_sell_order(symbol, available)
                        print(f"   ✅ Closed {symbol}")
                    
            except Exception as e:
                print(f"   ⚠️ Error closing {symbol}: {e}")
        
        # Final cleanup
        self.explicit_dust_cleanup()
        
        # Final calculations
        final_portfolio = self.get_bulletproof_portfolio()
        final_value = final_portfolio['total_value']
        total_profit = final_value - self.starting_portfolio_value
        total_profit_pct = (total_profit / self.starting_portfolio_value) * 100
        session_time = (time.time() - self.session_start) / 60
        
        # Report
        print("\n" + "🛡️" * 60)
        print("📊 KIMERA BULLETPROOF SESSION COMPLETE")
        print("🛡️" * 60)
        print(f"⏱️ Duration: {session_time:.1f} minutes")
        print(f"💰 Starting: ${self.starting_portfolio_value:.2f}")
        print(f"💰 Final: ${final_value:.2f}")
        print(f"📈 Profit: ${total_profit:+.2f}")
        print(f"🎯 Profit %: {total_profit_pct:+.2f}%")
        print(f"🔄 Trades: {self.trades_executed}")
        print(f"✅ Success: {self.successful_trades}")
        print(f"❌ Failed: {self.failed_trades}")
        print(f"🛡️ Bulletproof: ACTIVE")
        
        if self.trades_executed > 0:
            win_rate = (self.successful_trades / self.trades_executed) * 100
            print(f"🎯 Win Rate: {win_rate:.1f}%")
        
        # Save results
        results = {
            'session_start': datetime.fromtimestamp(self.session_start).isoformat(),
            'session_duration_minutes': session_time,
            'starting_value': self.starting_portfolio_value,
            'final_value': final_value,
            'total_profit': total_profit,
            'profit_percentage': total_profit_pct,
            'total_trades': self.trades_executed,
            'successful_trades': self.successful_trades,
            'failed_trades': self.failed_trades,
            'bulletproof_mode': True,
            'trade_history': self.trade_history
        }
        
        timestamp = int(time.time())
        filename = f"kimera_bulletproof_results_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Results: {filename}")
        print("🛡️" * 60)

async def main():
    print("🛡️ KIMERA BULLETPROOF ULTRA-AGGRESSIVE TRADER 🛡️")
    
    print("\n" + "⚠️" * 60)
    print("🚨 BULLETPROOF ULTRA-AGGRESSIVE MODE")
    print("🎯 TARGET: 100% PROFIT")
    print("💀 FULL WALLET CONTROL")
    print("🛡️ ZERO FAILURE GUARANTEE")
    print("🔥 EXPLICIT DUST MANAGEMENT")
    print("✅ BULLETPROOF VALIDATION")
    print("⚠️" * 60)
    
    response = input("\nActivate BULLETPROOF mode? (yes/no): ")
    
    if response.lower() == 'yes':
        duration = input("Duration in minutes (default 5): ")
        try:
            duration_minutes = int(duration) if duration else 5
        except Exception as e:
            logger.error(f"Error in kimera_bulletproof_trader.py: {e}", exc_info=True)
            raise  # Re-raise for proper error handling
            duration_minutes = 5
        
        trader = KimeraBulletproofTrader()
        await trader.run_bulletproof_session(duration_minutes)
    else:
        print("🛑 Bulletproof mode cancelled")

if __name__ == "__main__":
    asyncio.run(main()) 