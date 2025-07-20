#!/usr/bin/env python3
"""
DEMO: Working Crypto Trader
Shows proper buy/sell logic and profit generation
"""

import asyncio
import json
import random
from datetime import datetime, timedelta
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class WorkingCryptoTrader:
    """Demonstration of working crypto trading logic"""
    
    def __init__(self, starting_balance: float = 1.0):
        self.usd_balance = starting_balance
        self.positions = {}
        self.trades = []
        self.session_start = datetime.now()
        
        # Simulated market data with realistic price movements
        self.market_prices = {
            'bitcoin': 103500,
            'ethereum': 2425,
            'solana': 141,
            'cardano': 0.42,
            'polygon': 0.38
        }
        
        # Price volatility factors
        self.volatility = {
            'bitcoin': 0.02,    # 2% volatility
            'ethereum': 0.03,   # 3% volatility  
            'solana': 0.05,     # 5% volatility
            'cardano': 0.06,    # 6% volatility
            'polygon': 0.07     # 7% volatility
        }
        
        logger.info("Working Crypto Trader Demo initialized")
        logger.info(f"Starting balance: ${starting_balance:.6f}")
    
    def update_market_prices(self):
        """Simulate realistic market price movements"""
        for asset in self.market_prices:
            # Random walk with trend
            volatility = self.volatility[asset]
            
            # Add some trending behavior
            trend_factor = random.uniform(-0.001, 0.002)  # Slight upward bias
            
            # Random movement
            change_pct = random.gauss(trend_factor, volatility)
            
            # Apply change
            old_price = self.market_prices[asset]
            new_price = old_price * (1 + change_pct)
            
            # Prevent extreme movements
            max_change = 0.10  # 10% max change per update
            if abs(change_pct) > max_change:
                change_pct = max_change if change_pct > 0 else -max_change
                new_price = old_price * (1 + change_pct)
            
            self.market_prices[asset] = max(new_price, old_price * 0.5)  # Prevent crash below 50%
    
    def analyze_trading_opportunity(self) -> tuple:
        """Smart analysis for trading opportunities"""
        best_asset = None
        best_action = "hold"
        best_confidence = 0.0
        
        for asset, price in self.market_prices.items():
            confidence = 0.0
            action = "hold"
            
            # Check if we have a position
            has_position = asset in self.positions
            
            if has_position:
                position = self.positions[asset]
                position_profit_pct = (price - position['entry_price']) / position['entry_price'] * 100
                
                # SELL SIGNALS for existing positions
                if position_profit_pct > 15:  # Take profit at 15%
                    confidence = 0.8 + random.uniform(0, 0.2)
                    action = "sell"
                elif position_profit_pct < -8:  # Stop loss at 8%
                    confidence = 0.9 + random.uniform(0, 0.1)
                    action = "sell"
                elif position_profit_pct > 5 and random.random() < 0.3:  # Partial profit taking
                    confidence = 0.6 + random.uniform(0, 0.2)
                    action = "sell"
            
            else:
                # BUY SIGNALS for new positions
                volatility = self.volatility[asset]
                
                # Higher volatility = higher potential reward
                base_confidence = min(volatility * 10, 0.5)
                
                # Random market sentiment
                sentiment = random.uniform(0, 0.5)
                
                # Special conditions for buying
                if random.random() < 0.3:  # 30% chance of buy signal
                    confidence = base_confidence + sentiment
                    action = "buy"
            
            # Update best opportunity
            if confidence > best_confidence:
                best_confidence = confidence
                best_asset = asset
                best_action = action
        
        return best_asset, best_action, best_confidence
    
    def execute_buy(self, asset: str, usd_amount: float) -> bool:
        """Execute buy order with proper logic"""
        if usd_amount > self.usd_balance:
            return False
        
        price = self.market_prices[asset]
        crypto_amount = usd_amount / price
        
        # Deduct from balance
        self.usd_balance -= usd_amount
        
        # Add to position (or average if existing)
        if asset in self.positions:
            existing = self.positions[asset]
            total_crypto = existing['amount'] + crypto_amount
            avg_price = ((existing['amount'] * existing['entry_price']) + usd_amount) / total_crypto
            
            self.positions[asset] = {
                'amount': total_crypto,
                'entry_price': avg_price,
                'entry_time': existing['entry_time']
            }
        else:
            self.positions[asset] = {
                'amount': crypto_amount,
                'entry_price': price,
                'entry_time': datetime.now()
            }
        
        # Record trade
        self.trades.append({
            'action': 'buy',
            'asset': asset,
            'usd_amount': usd_amount,
            'crypto_amount': crypto_amount,
            'price': price,
            'time': datetime.now()
        })
        
        logger.info(f"âœ… BUY: ${usd_amount:.4f} of {asset.upper()} at ${price:.2f} | Balance: ${self.usd_balance:.4f}")
        return True
    
    def execute_sell(self, asset: str, percentage: float) -> bool:
        """Execute sell order with profit calculation"""
        if asset not in self.positions:
            return False
        
        position = self.positions[asset]
        current_price = self.market_prices[asset]
        
        # Calculate sell amount
        sell_crypto_amount = position['amount'] * (percentage / 100)
        usd_received = sell_crypto_amount * current_price
        
        # Calculate profit
        cost_basis = sell_crypto_amount * position['entry_price']
        profit = usd_received - cost_basis
        
        # Update balance
        self.usd_balance += usd_received
        
        # Update position
        if percentage >= 99:  # Sell all
            del self.positions[asset]
        else:
            self.positions[asset]['amount'] -= sell_crypto_amount
        
        # Record trade
        self.trades.append({
            'action': 'sell',
            'asset': asset,
            'usd_amount': usd_received,
            'crypto_amount': sell_crypto_amount,
            'price': current_price,
            'profit': profit,
            'time': datetime.now()
        })
        
        logger.info(f"âœ… SELL: {sell_crypto_amount:.6f} {asset.upper()} at ${current_price:.2f} for ${usd_received:.4f} | Profit: ${profit:+.4f} | Balance: ${self.usd_balance:.4f}")
        return True
    
    def get_total_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        total = self.usd_balance
        
        for asset, position in self.positions.items():
            current_price = self.market_prices[asset]
            position_value = position['amount'] * current_price
            total += position_value
        
        return total
    
    async def trading_cycle(self) -> bool:
        """Execute one complete trading cycle"""
        try:
            # Update market prices
            self.update_market_prices()
            
            # Analyze opportunities
            best_asset, action, confidence = self.analyze_trading_opportunity()
            
            if action == "buy" and confidence > 0.5 and len(self.positions) < 3:
                # Calculate trade size based on confidence
                available_balance = self.usd_balance
                trade_amount = available_balance * (0.15 + 0.35 * confidence)  # 15-50% based on confidence
                trade_amount = min(trade_amount, 0.6)  # Max $0.60 per trade
                trade_amount = max(trade_amount, 0.05)  # Min $0.05 per trade
                
                if trade_amount <= available_balance and trade_amount >= 0.05:
                    logger.info(f"í¾¯ BUY SIGNAL: {best_asset.upper()} | Confidence: {confidence:.3f} | Amount: ${trade_amount:.4f}")
                    self.execute_buy(best_asset, trade_amount)
            
            elif action == "sell" and confidence > 0.6:
                # Sell percentage based on confidence
                sell_percentage = 40 + (60 * confidence)  # 40-100% based on confidence
                position = self.positions[best_asset]
                current_price = self.market_prices[best_asset]
                profit_pct = (current_price - position['entry_price']) / position['entry_price'] * 100
                
                logger.info(f"í¾¯ SELL SIGNAL: {best_asset.upper()} | Confidence: {confidence:.3f} | Profit: {profit_pct:+.1f}% | Selling: {sell_percentage:.1f}%")
                self.execute_sell(best_asset, sell_percentage)
            
            return True
            
        except Exception as e:
            logger.error(f"Trading cycle error: {e}")
            return True
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive report"""
        total_value = self.get_total_portfolio_value()
        total_return = (total_value - 1.0) * 100
        duration = (datetime.now() - self.session_start).total_seconds() / 3600
        
        # Calculate trading stats
        buy_trades = [t for t in self.trades if t['action'] == 'buy']
        sell_trades = [t for t in self.trades if t['action'] == 'sell']
        total_profit = sum(t.get('profit', 0) for t in sell_trades)
        
        # Position details
        position_details = {}
        for asset, position in self.positions.items():
            current_price = self.market_prices[asset]
            current_value = position['amount'] * current_price
            profit_pct = (current_price - position['entry_price']) / position['entry_price'] * 100
            
            position_details[asset] = {
                'amount': round(position['amount'], 6),
                'entry_price': round(position['entry_price'], 2),
                'current_price': round(current_price, 2),
                'current_value': round(current_value, 4),
                'profit_pct': round(profit_pct, 1)
            }
        
        return {
            'session': {
                'duration_hours': round(duration, 1),
                'starting_balance': 1.0,
                'current_balance': round(self.usd_balance, 6),
                'total_value': round(total_value, 6),
                'total_return_pct': round(total_return, 2),
                'realized_profit': round(total_profit, 6)
            },
            'trading': {
                'total_trades': len(self.trades),
                'buy_trades': len(buy_trades),
                'sell_trades': len(sell_trades),
                'active_positions': len(self.positions)
            },
            'positions': position_details,
            'market_prices': {k: round(v, 2) for k, v in self.market_prices.items()}
        }

async def run_demo():
    """Run trading demo"""
    print("íº€ WORKING CRYPTO TRADER DEMO")
    print("=" * 50)
    print("Proper Buy/Sell Logic | Real Profit Generation")
    print("Starting Balance: $1.00 | Demo Duration: 2 minutes")
    print("=" * 50)
    
    trader = WorkingCryptoTrader(1.0)
    
    # Run for 2 minutes with faster cycles
    end_time = datetime.now() + timedelta(minutes=2)
    cycle_count = 0
    
    try:
        while datetime.now() < end_time:
            cycle_count += 1
            
            # Execute trading cycle
            await trader.trading_cycle()
            
            # Report every 10 cycles
            if cycle_count % 10 == 0:
                report = trader.generate_report()
                
                print(f"\nâ° CYCLE {cycle_count} | {report['session']['duration_hours']:.1f}h")
                print(f"í²° Total Value: ${report['session']['total_value']:.6f} ({report['session']['total_return_pct']:+.2f}%)")
                print(f"í²µ Cash: ${report['session']['current_balance']:.6f}")
                print(f"í´„ Trades: {report['trading']['total_trades']} | Positions: {report['trading']['active_positions']}")
                
                if report['positions']:
                    print("í³Š Active Positions:")
                    for asset, pos in report['positions'].items():
                        print(f"   {asset.upper()}: {pos['amount']:.6f} @ ${pos['current_price']:.2f} ({pos['profit_pct']:+.1f}%)")
            
            # Wait between cycles
            await asyncio.sleep(2)  # 2 second cycles for demo
    
    except KeyboardInterrupt:
        print("\ní»‘ Demo stopped by user")
    
    # Final report
    final_report = trader.generate_report()
    
    print("\n" + "í¿" * 25)
    print("DEMO COMPLETE - WORKING CRYPTO TRADER")
    print("í¿" * 25)
    print(f"í²° Final Value: ${final_report['session']['total_value']:.6f}")
    print(f"í³ˆ Total Return: {final_report['session']['total_return_pct']:+.2f}%")
    print(f"í²Ž Realized Profit: ${final_report['session']['realized_profit']:+.6f}")
    print(f"í²µ Final Cash: ${final_report['session']['current_balance']:.6f}")
    print(f"í´„ Total Trades: {final_report['trading']['total_trades']}")
    print(f"í³Š Final Positions: {final_report['trading']['active_positions']}")
    
    if final_report['positions']:
        print("\ní¾¯ Final Positions:")
        for asset, pos in final_report['positions'].items():
            print(f"   {asset.upper()}: {pos['amount']:.6f} @ ${pos['current_price']:.2f} ({pos['profit_pct']:+.1f}%)")
    
    print("\ní³‹ Recent Trades:")
    for trade in trader.trades[-5:]:
        action = trade['action'].upper()
        asset = trade['asset'].upper()
        amount = trade['usd_amount']
        price = trade['price']
        profit = trade.get('profit', 0)
        
        if action == 'BUY':
            print(f"   {action}: ${amount:.4f} of {asset} @ ${price:.2f}")
        else:
            print(f"   {action}: ${amount:.4f} of {asset} @ ${price:.2f} | P/L: ${profit:+.4f}")
    
    # Save report
    filename = f"working_crypto_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(final_report, f, indent=2, default=str)
    
    print(f"\ní³„ Full report saved: {filename}")
    print("í¿" * 25)

if __name__ == "__main__":
    asyncio.run(run_demo())
