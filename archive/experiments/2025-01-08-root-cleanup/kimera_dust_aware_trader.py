#!/usr/bin/env python3
"""
KIMERA DUST-AWARE ULTRA-AGGRESSIVE TRADER
========================================
🧹 AUTOMATIC DUST MANAGEMENT INTEGRATED 🧹
🔥 ULTRA-AGGRESSIVE TRADING WITH DUST PREVENTION 🔥
- Automatically detects and manages dust before trading
- Prevents dust-related trading failures
- Optimizes portfolio continuously
- Maintains clean tradeable assets
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
from kimera_dust_manager import KimeraDustManager
import logging
logger = logging.getLogger(__name__)

load_dotenv()

class KimeraDustAwareTrader:
    """Ultra-aggressive trader with integrated dust management"""
    
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
        
        # Initialize dust manager
        self.dust_manager = KimeraDustManager()
        
        # 🔥 ULTRA-AGGRESSIVE PARAMETERS 🔥
        self.profit_target = 1.0        # 100% PROFIT TARGET
        self.max_position_ratio = 0.7   # Use 70% of wallet per trade
        self.min_profit_per_trade = 0.006  # 0.6% minimum profit
        self.max_loss_per_trade = -0.015   # 1.5% max loss
        self.trade_frequency = 2        # Trade every 2 seconds
        self.max_concurrent_trades = 4  # 4 simultaneous positions
        self.min_trade_size = 6.5       # $6.50 minimum (confirmed working)
        self.max_trade_size = 100       # $100 maximum per trade
        self.dust_check_interval = 60   # Check for dust every 60 seconds
        
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
        self.last_dust_check = 0
        
        # Performance metrics
        self.max_drawdown = 0.0
        self.peak_value = 0.0
        self.trades_per_minute = 0
        
        logger.info("🧹" * 60)
        logger.info("🤖 KIMERA DUST-AWARE ULTRA-AGGRESSIVE TRADER")
        logger.info("🎯 TARGET: 100% PROFIT")
        logger.info("⚡ MAXIMUM AGGRESSION + DUST MANAGEMENT")
        logger.info("💀 FULL WALLET CONTROL")
        logger.info(f"💰 MIN TRADE SIZE: ${self.min_trade_size}")
        logger.info("🧹 AUTOMATIC DUST PREVENTION")
        logger.info("🧹" * 60)
    
    def get_dust_free_portfolio(self) -> Dict[str, Any]:
        """Get portfolio with dust filtering applied"""
        try:
            # Use dust manager to get clean portfolio
            clean_portfolio = self.dust_manager.create_dust_free_portfolio_snapshot()
            
            if clean_portfolio:
                return clean_portfolio
            else:
                # Fallback to regular portfolio if dust manager fails
                return self.get_full_portfolio()
                
        except Exception as e:
            logger.info(f"❌ Dust-free portfolio error: {e}")
            return self.get_full_portfolio()
    
    def get_full_portfolio(self) -> Dict[str, Any]:
        """Get complete portfolio with values (fallback method)"""
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
                            'tradeable': usd_value >= self.min_trade_size,
                            'symbol': symbol if asset != 'USDT' else None
                        }
                        total_value += usd_value
            
            return {'assets': portfolio, 'total_value': total_value}
            
        except Exception as e:
            logger.info(f"❌ Portfolio error: {e}")
            return {'assets': {}, 'total_value': 0.0}
    
    def periodic_dust_check(self):
        """Periodically check and manage dust during trading"""
        try:
            current_time = time.time()
            
            if current_time - self.last_dust_check >= self.dust_check_interval:
                logger.info(f"\n🧹 PERIODIC DUST CHECK:")
                
                # Quick dust analysis
                analysis = self.dust_manager.analyze_dust()
                dust_count = len(analysis.get('dust_assets', []))
                dust_value = analysis.get('total_dust_value', 0)
                
                if dust_count > 0:
                    logger.info(f"   Found {dust_count} dust assets worth ${dust_value:.2f}")
                    
                    # Auto-manage dust if significant
                    if dust_count > 1 or dust_value > 5:
                        logger.info("   🔄 Managing dust automatically...")
                        self.dust_manager.consolidate_dust_by_trading(analysis.get('dust_assets', []))
                else:
                    logger.info("   ✅ No dust detected")
                
                self.last_dust_check = current_time
                
        except Exception as e:
            logger.info(f"⚠️ Periodic dust check error: {e}")
    
    def find_dust_aware_opportunities(self) -> List[Dict]:
        """Find trading opportunities with dust awareness"""
        opportunities = []
        
        try:
            tickers = self.exchange.fetch_tickers()
            portfolio = self.get_dust_free_portfolio()  # Use dust-free portfolio
            
            # Get current USDT balance
            usdt_balance = portfolio['assets'].get('USDT', {}).get('amount', 0)
            
            # 🔥 STRATEGY 1: MOMENTUM SCALPING 🔥
            scalp_targets = [
                'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'TRX/USDT', 'ADA/USDT',
                'SOL/USDT', 'DOGE/USDT', 'XRP/USDT', 'MATIC/USDT', 'DOT/USDT',
                'AVAX/USDT', 'LINK/USDT', 'UNI/USDT', 'LTC/USDT'
            ]
            
            for symbol in scalp_targets:
                if symbol in tickers:
                    ticker = tickers[symbol]
                    change_1h = ticker.get('percentage', 0)
                    volume = ticker.get('quoteVolume', 0)
                    price = ticker.get('last', 0)
                    
                    # Calculate trade size
                    max_trade_value = min(
                        portfolio['total_value'] * self.max_position_ratio,
                        self.max_trade_size,
                        usdt_balance * 0.9
                    )
                    
                    trade_size = max(self.min_trade_size, min(max_trade_value, 50))
                    
                    # Look for significant movement and volume
                    if change_1h is not None and abs(change_1h) > 0.5 and volume > 500000 and trade_size >= self.min_trade_size and usdt_balance >= trade_size:
                        confidence = min(abs(change_1h) / 4.0, 0.95)
                        urgency = min(abs(change_1h) / 2.0, 1.0)
                        
                        opportunities.append({
                            'type': 'SCALP_MOMENTUM',
                            'symbol': symbol,
                            'direction': 'BUY' if change_1h > 0 else 'SELL',
                            'confidence': confidence,
                            'urgency': urgency,
                            'expected_profit': abs(change_1h or 0) * 0.3,
                            'trade_size_usdt': trade_size,
                            'price': price,
                            'reason': f"Scalp {change_1h:+.2f}% movement, Vol: {volume/1000000:.1f}M"
                        })
            
            # 🔥 STRATEGY 2: DUST-AWARE PROFIT TAKING & LOSS CUTTING 🔥
            for asset, data in portfolio['assets'].items():
                if asset == 'USDT' or not data.get('tradeable', False) or data.get('amount', 0) <= 0:
                    continue
                
                symbol = data.get('symbol')
                if symbol and symbol in tickers:
                    ticker = tickers[symbol]
                    change_24h = ticker.get('percentage', 0)
                    
                    # Get market info for validation
                    try:
                        market = self.exchange.market(symbol)
                        min_amount = market.get('limits', {}).get('amount', {}).get('min', 0)
                    except Exception as e:
                        logger.error(f"Error in kimera_dust_aware_trader.py: {e}", exc_info=True)
                        raise  # Re-raise for proper error handling
                        continue
                    
                    # Take profits on gains >2% or cut losses >1.5%
                    if change_24h is not None and change_24h > 2.0:
                        sell_amount = data['amount'] * 0.6  # Sell 60%
                        notional_value = sell_amount * data['price']
                        
                        # Enhanced validation to prevent dust creation
                        remaining_amount = data['amount'] - sell_amount
                        remaining_value = remaining_amount * data['price']
                        
                        # Only sell if both sell amount AND remaining amount are valid
                        if (sell_amount > 0 and 
                            sell_amount >= min_amount and 
                            notional_value >= self.min_trade_size and
                            (remaining_value >= self.min_trade_size or remaining_value < 1.0)):  # Either tradeable or negligible
                            
                            opportunities.append({
                                'type': 'DUST_AWARE_PROFIT_TAKING',
                                'symbol': symbol,
                                'direction': 'SELL',
                                'confidence': 0.95,
                                'urgency': 0.8,
                                'expected_profit': change_24h * 0.4,
                                'sell_amount': sell_amount,
                                'reason': f"Dust-aware profit: +{change_24h:.1f}%"
                            })
                    elif change_24h is not None and change_24h < -1.5:
                        sell_amount = data['amount'] * 0.4  # Sell 40%
                        notional_value = sell_amount * data['price']
                        
                        # Enhanced validation to prevent dust creation
                        remaining_amount = data['amount'] - sell_amount
                        remaining_value = remaining_amount * data['price']
                        
                        if (sell_amount > 0 and 
                            sell_amount >= min_amount and 
                            notional_value >= self.min_trade_size and
                            (remaining_value >= self.min_trade_size or remaining_value < 1.0)):
                            
                            opportunities.append({
                                'type': 'DUST_AWARE_LOSS_CUTTING',
                                'symbol': symbol,
                                'direction': 'SELL',
                                'confidence': 0.85,
                                'urgency': 0.95,
                                'expected_profit': 1.5,  # Expected recovery
                                'sell_amount': sell_amount,
                                'reason': f"Dust-aware loss cut: {change_24h:.1f}%"
                            })
            
            # Sort by urgency and confidence
            opportunities.sort(key=lambda x: x['urgency'] * x['confidence'], reverse=True)
            
            return opportunities[:8]  # Return top 8 opportunities
            
        except Exception as e:
            logger.info(f"❌ Dust-aware opportunity finding error: {e}")
            return []
    
    async def execute_dust_aware_trade(self, opportunity: Dict) -> bool:
        """Execute trade with dust prevention"""
        try:
            symbol = opportunity['symbol']
            direction = opportunity['direction']
            
            logger.info(f"\n🔥 EXECUTING: {opportunity['type']}")
            logger.info(f"   Symbol: {symbol}")
            logger.info(f"   Direction: {direction}")
            logger.info(f"   Confidence: {opportunity['confidence']:.1%}")
            logger.info(f"   Urgency: {opportunity['urgency']:.1%}")
            logger.info(f"   Reason: {opportunity['reason']}")
            
            if direction == 'BUY':
                trade_size_usdt = opportunity.get('trade_size_usdt', 0)
                price = opportunity.get('price', 0)
                
                if trade_size_usdt >= self.min_trade_size and price > 0:
                    # Calculate exact quantity
                    quantity = trade_size_usdt / price
                    
                    # Check minimum quantity requirements
                    market = self.exchange.market(symbol)
                    min_amount = market.get('limits', {}).get('amount', {}).get('min', 0)
                    
                    if quantity >= min_amount:
                        # Execute BUY order
                        order = self.exchange.create_market_buy_order(symbol, quantity)
                        
                        actual_cost = order.get('cost', trade_size_usdt)
                        actual_quantity = order.get('amount', quantity)
                        
                        logger.info(f"   ✅ BOUGHT: {actual_quantity:.8f} {symbol.split('/')[0]}")
                        logger.info(f"   💰 Cost: ${actual_cost:.2f}")
                        logger.info(f"   📋 Order: {order['id']}")
                        
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
                        logger.info(f"   ❌ Quantity {quantity:.8f} below minimum {min_amount}")
                else:
                    logger.info(f"   ❌ Invalid trade parameters: size=${trade_size_usdt:.2f}, price=${price:.2f}")
            
            elif direction == 'SELL':
                sell_amount = opportunity.get('sell_amount', 0)
                
                if sell_amount > 0 and sell_amount is not None:
                    # Check minimum quantity
                    market = self.exchange.market(symbol)
                    min_amount = market.get('limits', {}).get('amount', {}).get('min', 0)
                    
                    if sell_amount >= min_amount:
                        order = self.exchange.create_market_sell_order(symbol, sell_amount)
                        
                        received_usdt = order.get('cost', 0)
                        
                        logger.info(f"   ✅ SOLD: {sell_amount:.8f} {symbol.split('/')[0]}")
                        logger.info(f"   💰 Received: ${received_usdt:.2f}")
                        logger.info(f"   📋 Order: {order['id']}")
                        
                        self.trades_executed += 1
                        self.total_profit += received_usdt * 0.01  # Conservative profit estimate
                        return True
                    else:
                        logger.info(f"   ❌ Sell amount {sell_amount:.8f} below minimum {min_amount}")
                else:
                    logger.info(f"   ❌ Invalid sell amount: {sell_amount} (must be > 0)")
            
        except ccxt.InsufficientFunds as e:
            logger.info(f"   ❌ Insufficient funds: {e}")
            self.failed_trades += 1
        except ccxt.InvalidOrder as e:
            logger.info(f"   ❌ Invalid order: {e}")
            self.failed_trades += 1
        except ccxt.NetworkError as e:
            logger.info(f"   ❌ Network error: {e}")
            self.failed_trades += 1
        except Exception as e:
            logger.info(f"   ❌ Trade failed: {e}")
            self.failed_trades += 1
        
        return False
    
    def monitor_ultra_aggressive_positions(self):
        """Monitor positions with dust-aware exit strategy"""
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
                
                # Time-based exit (60 seconds max)
                elif hold_time > 60:
                    should_exit = True
                    reason = f"TIME_EXIT ({pnl_pct:.2%})"
                
                # Momentum reversal detection
                elif pnl_pct > 0.003 and hold_time > 20:  # If >0.3% profit and held >20sec
                    change_short = ticker.get('percentage', 0)
                    if (position['type'] == 'long' and change_short is not None and change_short < -0.2) or \
                       (position['type'] == 'short' and change_short is not None and change_short > 0.2):
                        should_exit = True
                        reason = f"REVERSAL ({pnl_pct:.2%})"
                
                if should_exit:
                    # Exit position with dust awareness
                    base_asset = symbol.split('/')[0]
                    portfolio = self.get_dust_free_portfolio()
                    available = portfolio['assets'].get(base_asset, {}).get('amount', 0)
                    
                    if available > 0:
                        market = self.exchange.market(symbol)
                        min_amount = market.get('limits', {}).get('amount', {}).get('min', 0)
                        
                        if available >= min_amount:
                            order = self.exchange.create_market_sell_order(symbol, available)
                            
                            received_usdt = order.get('cost', 0)
                            profit_usd = received_usdt - position['entry_cost']
                            self.total_profit += profit_usd
                            
                            if profit_usd > 0:
                                self.successful_trades += 1
                            
                            logger.info(f"   🎯 EXITED {symbol}: {reason}")
                            logger.info(f"   💰 P&L: ${profit_usd:+.2f}")
                            
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
                            logger.info(f"   ⚠️ Cannot exit {symbol}: amount {available:.8f} below minimum {min_amount}")
                
            except Exception as e:
                logger.info(f"   ⚠️ Position monitoring error for {symbol}: {e}")
    
    async def run_dust_aware_session(self, duration_minutes: int = 5):
        """Run dust-aware ultra-aggressive session"""
        logger.info(f"\n🔥 STARTING DUST-AWARE ULTRA-AGGRESSIVE SESSION 🔥")
        logger.info(f"⏱️ DURATION: {duration_minutes} MINUTES")
        logger.info(f"🎯 TARGET: 100% PROFIT ({self.profit_target:.0%})")
        logger.info(f"💀 FULL WALLET CONTROL ACTIVATED")
        logger.info(f"💰 MIN TRADE SIZE: ${self.min_trade_size}")
        logger.info(f"🧹 DUST MANAGEMENT: ACTIVE")
        logger.info("🔥" * 60)
        
        # 🧹 PRE-SESSION DUST MANAGEMENT 🧹
        logger.info(f"\n🧹 PRE-SESSION DUST MANAGEMENT:")
        logger.info("-" * 50)
        dust_managed = self.dust_manager.auto_dust_management()
        if dust_managed:
            logger.info("✅ Dust management completed")
            time.sleep(2)  # Allow time for balance updates
        else:
            logger.info("⚠️ Dust management skipped or failed")
        
        self.session_start = time.time()
        portfolio = self.get_dust_free_portfolio()
        self.starting_portfolio_value = portfolio['total_value']
        self.peak_value = self.starting_portfolio_value
        self.running = True
        self.last_dust_check = time.time()
        
        logger.info(f"💰 Starting Portfolio: ${self.starting_portfolio_value:.2f}")
        logger.info(f"🎯 Target Value: ${self.starting_portfolio_value * (1 + self.profit_target):.2f}")
        
        session_duration = duration_minutes * 60
        last_trade_time = 0
        loop_count = 0
        
        # DUST-AWARE ULTRA-AGGRESSIVE MAIN LOOP
        while self.running and (time.time() - self.session_start) < session_duration:
            try:
                loop_count += 1
                elapsed = time.time() - self.session_start
                remaining = session_duration - elapsed
                
                # Periodic dust check
                if loop_count % 30 == 0:  # Every 30 loops
                    self.periodic_dust_check()
                
                # Update portfolio value every 15 loops
                if loop_count % 15 == 0:
                    current_portfolio = self.get_dust_free_portfolio()
                    self.current_portfolio_value = current_portfolio['total_value']
                    
                    # Calculate performance metrics
                    current_profit = self.current_portfolio_value - self.starting_portfolio_value
                    current_profit_pct = (current_profit / self.starting_portfolio_value) * 100
                    
                    # Update peak and drawdown
                    if self.current_portfolio_value > self.peak_value:
                        self.peak_value = self.current_portfolio_value
                    
                    drawdown = (self.peak_value - self.current_portfolio_value) / self.peak_value * 100 if self.peak_value > 0 else 0
                    if drawdown > self.max_drawdown:
                        self.max_drawdown = drawdown
                    
                    # Calculate trades per minute
                    self.trades_per_minute = self.trades_executed / max(elapsed / 60, 0.1)
                    
                    logger.info(f"\n⚡ Time: {remaining:.0f}s | Portfolio: ${self.current_portfolio_value:.2f} | "
                          f"Profit: {current_profit_pct:+.2f}% | Trades: {self.trades_executed} | "
                          f"Active: {len(self.active_positions)} | Success: {self.successful_trades} | 🧹 Clean")
                    
                    # Check if target achieved
                    if current_profit_pct >= self.profit_target * 100:
                        logger.info(f"\n🎯 TARGET ACHIEVED! {current_profit_pct:.2f}% PROFIT!")
                        break
                
                # Monitor existing positions
                if self.active_positions:
                    self.monitor_ultra_aggressive_positions()
                
                # Execute new trades
                if time.time() - last_trade_time >= self.trade_frequency:
                    if len(self.active_positions) < self.max_concurrent_trades:
                        opportunities = self.find_dust_aware_opportunities()
                        
                        if opportunities:
                            # Execute best opportunity
                            best_opp = opportunities[0]
                            if best_opp['urgency'] > 0.5 and best_opp['confidence'] > 0.3:
                                success = await self.execute_dust_aware_trade(best_opp)
                                if success:
                                    last_trade_time = time.time()
                                await asyncio.sleep(1)  # Brief pause after trade
                
                await asyncio.sleep(1.5)  # Main loop delay
                
            except KeyboardInterrupt:
                logger.info("\n🛑 MANUAL STOP REQUESTED")
                break
            except Exception as e:
                logger.info(f"⚠️ Trading loop error: {e}")
                await asyncio.sleep(3)
        
        # Session complete
        await self.close_dust_aware_session()
    
    async def close_dust_aware_session(self):
        """Close session with final dust management"""
        logger.info(f"\n🔚 CLOSING DUST-AWARE SESSION...")
        
        # Close all positions
        for symbol in list(self.active_positions.keys()):
            try:
                base_asset = symbol.split('/')[0]
                portfolio = self.get_dust_free_portfolio()
                available = portfolio['assets'].get(base_asset, {}).get('amount', 0)
                
                if available > 0:
                    market = self.exchange.market(symbol)
                    min_amount = market.get('limits', {}).get('amount', {}).get('min', 0)
                    
                    if available >= min_amount:
                        order = self.exchange.create_market_sell_order(symbol, available)
                        logger.info(f"   ✅ Closed {symbol}")
                    
            except Exception as e:
                logger.info(f"   ⚠️ Error closing {symbol}: {e}")
        
        # Final dust management
        logger.info(f"\n🧹 POST-SESSION DUST CLEANUP:")
        logger.info("-" * 50)
        final_dust_managed = self.dust_manager.auto_dust_management()
        if final_dust_managed:
            logger.info("✅ Final dust cleanup completed")
        
        # Final calculations
        final_portfolio = self.get_dust_free_portfolio()
        final_value = final_portfolio['total_value']
        total_profit = final_value - self.starting_portfolio_value
        total_profit_pct = (total_profit / self.starting_portfolio_value) * 100
        session_time = (time.time() - self.session_start) / 60
        
        # Generate comprehensive report
        logger.info("\n" + "🧹" * 60)
        logger.info("📊 KIMERA DUST-AWARE SESSION COMPLETE")
        logger.info("🧹" * 60)
        logger.info(f"⏱️ Session Duration: {session_time:.1f} minutes")
        logger.info(f"💰 Starting Value: ${self.starting_portfolio_value:.2f}")
        logger.info(f"💰 Final Value: ${final_value:.2f}")
        logger.info(f"📈 Total Profit: ${total_profit:+.2f}")
        logger.info(f"🎯 Profit Percentage: {total_profit_pct:+.2f}%")
        logger.info(f"📊 Peak Value: ${self.peak_value:.2f}")
        logger.info(f"📉 Max Drawdown: {self.max_drawdown:.2f}%")
        logger.info(f"🔄 Total Trades: {self.trades_executed}")
        logger.info(f"✅ Successful: {self.successful_trades}")
        logger.info(f"❌ Failed: {self.failed_trades}")
        logger.info(f"⚡ Trades/Minute: {self.trades_per_minute:.1f}")
        logger.info(f"🧹 Dust Management: ACTIVE")
        
        if self.trades_executed > 0:
            win_rate = (self.successful_trades / self.trades_executed) * 100
            logger.info(f"🎯 Win Rate: {win_rate:.1f}%")
        
        logger.info(f"\n🏆 PERFORMANCE RATING:")
        if total_profit_pct >= 100:
            logger.info("🔥🔥🔥 LEGENDARY! TARGET ACHIEVED! 🔥🔥🔥")
        elif total_profit_pct >= 50:
            logger.info("🔥🔥 EXCELLENT PERFORMANCE! 🔥🔥")
        elif total_profit_pct >= 20:
            logger.info("🔥 GOOD PERFORMANCE! 🔥")
        elif total_profit_pct >= 5:
            logger.info("✅ SOLID GAINS!")
        elif total_profit_pct >= 0:
            logger.info("📊 PROFITABLE SESSION")
        else:
            logger.info("📚 LEARNING EXPERIENCE")
        
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
            'dust_management': 'ACTIVE',
            'trade_history': self.trade_history,
            'final_portfolio': final_portfolio
        }
        
        # Save to file
        timestamp = int(time.time())
        filename = f"kimera_dust_aware_results_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"💾 Results saved to: {filename}")
        logger.info("🧹" * 60)

async def main():
    logger.info("🧹 INITIALIZING KIMERA DUST-AWARE ULTRA-AGGRESSIVE TRADER 🧹")
    
    logger.info("\n" + "⚠️" * 60)
    logger.info("🚨 DUST-AWARE ULTRA-AGGRESSIVE TRADING MODE")
    logger.info("🎯 TARGET: 100% PROFIT")
    logger.info("💀 FULL WALLET CONTROL")
    logger.info("⚡ MAXIMUM RISK - MAXIMUM REWARD")
    logger.info("🔥 REAL MONEY - REAL CONSEQUENCES")
    logger.info("🧹 AUTOMATIC DUST MANAGEMENT")
    logger.info("✅ PREVENTS DUST-RELATED TRADING FAILURES")
    logger.info("⚠️" * 60)
    
    response = input("\nActivate DUST-AWARE ULTRA-AGGRESSIVE mode? (yes/no): ")
    
    if response.lower() == 'yes':
        duration = input("Session duration in minutes (default 5): ")
        try:
            duration_minutes = int(duration) if duration else 5
        except Exception as e:
            logger.error(f"Error in kimera_dust_aware_trader.py: {e}", exc_info=True)
            raise  # Re-raise for proper error handling
            duration_minutes = 5
        
        trader = KimeraDustAwareTrader()
        await trader.run_dust_aware_session(duration_minutes)
    else:
        logger.info("🛑 Dust-aware ultra-aggressive mode cancelled")

if __name__ == "__main__":
    asyncio.run(main()) 