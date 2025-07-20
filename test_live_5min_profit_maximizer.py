#!/usr/bin/env python3
"""
KIMERA 5-MINUTE LIVE PROFIT MAXIMIZER
=====================================

🚨 WARNING: REAL MONEY TRADING 🚨
- Maximum position size: $10
- Trading duration: 5 minutes
- Target profit: $0.50-$2.00 (5-20% return)
- Strategy: High-frequency scalping

SAFETY FEATURES:
- Hard position limit: $10
- Maximum trades: 5
- Emergency stop: 4m50s
- Auto-exit on 2% total loss

Author: Kimera SWM Alpha
"""

import asyncio
import time
import logging
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

# Add backend to path
sys.path.append('backend')

from trading.api.binance_connector_hmac import BinanceConnector

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('profit_maximizer.log')
    ]
)

logger = logging.getLogger(__name__)

class ProfitMaximizer:
    def __init__(self):
        """Initialize the 5-minute profit maximizer."""
        self.max_position_size = 10.0  # $10 maximum
        self.trading_duration = 300  # 5 minutes in seconds
        self.emergency_exit_time = 290  # 4m50s emergency exit
        self.max_trades = 5
        self.target_profit_pct = 0.002  # 0.2% per trade
        self.stop_loss_pct = 0.005  # 0.5% stop loss
        self.max_total_loss_pct = 0.02  # 2% total loss limit
        
        self.connector = None
        self.start_time = None
        self.start_balance = None
        self.trades_executed = 0
        self.total_profit = 0.0
        self.open_positions = []
        self.trade_history = []
        self.running = False
        
        # Trading pair
        self.symbol = "BTCUSDT"
        
    async def initialize(self):
        """Initialize Binance connector and verify credentials."""
        try:
            logger.info("🚀 Initializing Kimera 5-Minute Profit Maximizer...")
            
            # Load credentials
            if not os.path.exists('kimera_binance_hmac.env'):
                raise FileNotFoundError("kimera_binance_hmac.env not found!")
                
            with open('kimera_binance_hmac.env', 'r') as f:
                for line in f:
                    if '=' in line and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value
                        
            # Get credentials from environment
            api_key = os.environ.get('BINANCE_API_KEY')
            secret_key = os.environ.get('BINANCE_SECRET_KEY')
            testnet = os.environ.get('BINANCE_USE_TESTNET', 'false').lower() == 'true'
            
            if not api_key or not secret_key:
                raise ValueError("API key and secret key are required")
                        
            # Initialize connector
            self.connector = BinanceConnector(
                api_key=api_key,
                secret_key=secret_key,
                testnet=testnet
            )
            
            # Test connection and get balance
            account_info = await self.connector.get_account_info()
            if not account_info:
                raise Exception("Failed to get account info")
                
            # Find USDT balance
            usdt_balance = 0.0
            for balance in account_info.get('balances', []):
                if balance['asset'] == 'USDT':
                    usdt_balance = float(balance['free'])
                    break
                    
            self.start_balance = usdt_balance
            logger.info(f"💰 Starting USDT balance: ${usdt_balance:.2f}")
            
            if usdt_balance < self.max_position_size:
                raise Exception(f"Insufficient balance: ${usdt_balance:.2f} < ${self.max_position_size}")
                
            # Get current market price
            ticker = await self.connector.get_ticker_price(self.symbol)
            current_price = float(ticker['price'])
            logger.info(f"📊 {self.symbol} current price: ${current_price:,.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            return False
            
    async def get_market_signal(self) -> Dict:
        """Get trading signal based on technical analysis."""
        try:
            # Get recent klines (1-minute candles)
            klines = await self.connector.get_klines(
                symbol=self.symbol,
                interval='1m',
                limit=20
            )
            
            if not klines:
                return {'action': 'HOLD', 'confidence': 0}
                
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Convert to numeric
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
                
            # Calculate indicators
            df['rsi'] = self.calculate_rsi(df['close'], 14)
            df['ma5'] = df['close'].rolling(5).mean()
            df['ma10'] = df['close'].rolling(10).mean()
            
            # Get latest values
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            current_price = latest['close']
            rsi = latest['rsi']
            ma5 = latest['ma5']
            ma10 = latest['ma10']
            volume = latest['volume']
            avg_volume = df['volume'].tail(10).mean()
            
            # Generate signal
            signal = {'action': 'HOLD', 'confidence': 0, 'price': current_price}
            
            # Buy conditions
            if (rsi < 35 and  # Oversold
                current_price > ma5 and  # Above short MA
                ma5 > ma10 and  # Upward momentum
                volume > avg_volume * 1.2):  # High volume
                signal = {
                    'action': 'BUY',
                    'confidence': min(0.8, (40 - rsi) / 10),
                    'price': current_price,
                    'rsi': rsi,
                    'ma_signal': 'bullish'
                }
                
            # Sell conditions  
            elif (rsi > 65 and  # Overbought
                  current_price < ma5 and  # Below short MA
                  ma5 < ma10):  # Downward momentum
                signal = {
                    'action': 'SELL',
                    'confidence': min(0.8, (rsi - 60) / 10),
                    'price': current_price,
                    'rsi': rsi,
                    'ma_signal': 'bearish'
                }
                
            return signal
            
        except Exception as e:
            logger.error(f"❌ Error getting market signal: {e}")
            return {'action': 'HOLD', 'confidence': 0}
            
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
        
    async def execute_buy_order(self, signal: Dict) -> bool:
        """Execute a buy order."""
        try:
            current_price = signal['price']
            
            # Calculate quantity for $10 position
            quantity = self.max_position_size / current_price
            
            # Round to appropriate precision (Binance requires specific lot sizes)
            quantity = round(quantity, 6)
            
            if quantity < 0.00001:  # Minimum quantity check
                logger.warning(f"⚠️ Quantity too small: {quantity}")
                return False
                
            logger.info(f"🟢 Executing BUY: {quantity} {self.symbol} at ${current_price:,.2f}")
            
            # Execute market buy order
            order = await self.connector.create_market_order(
                symbol=self.symbol,
                side='BUY',
                quantity=quantity
            )
            
            if order and order.get('status') == 'FILLED':
                filled_qty = float(order['executedQty'])
                filled_price = float(order['fills'][0]['price']) if order.get('fills') else current_price
                
                position = {
                    'symbol': self.symbol,
                    'side': 'BUY',
                    'quantity': filled_qty,
                    'entry_price': filled_price,
                    'entry_time': time.time(),
                    'target_price': filled_price * (1 + self.target_profit_pct),
                    'stop_price': filled_price * (1 - self.stop_loss_pct),
                    'order_id': order['orderId']
                }
                
                self.open_positions.append(position)
                self.trades_executed += 1
                
                logger.info(f"✅ BUY executed: {filled_qty} at ${filled_price:,.2f}")
                logger.info(f"🎯 Target: ${position['target_price']:,.2f} | Stop: ${position['stop_price']:,.2f}")
                
                return True
                
        except Exception as e:
            logger.error(f"❌ Buy order failed: {e}")
            
        return False
        
    async def execute_sell_order(self, position: Dict) -> bool:
        """Execute a sell order for a position."""
        try:
            current_price = await self.get_current_price()
            quantity = position['quantity']
            
            logger.info(f"🔴 Executing SELL: {quantity} {self.symbol} at ${current_price:,.2f}")
            
            # Execute market sell order
            order = await self.connector.create_market_order(
                symbol=self.symbol,
                side='SELL',
                quantity=quantity
            )
            
            if order and order.get('status') == 'FILLED':
                filled_price = float(order['fills'][0]['price']) if order.get('fills') else current_price
                
                # Calculate profit
                profit = (filled_price - position['entry_price']) * quantity
                profit_pct = (filled_price / position['entry_price'] - 1) * 100
                
                self.total_profit += profit
                
                # Record trade
                trade_record = {
                    'symbol': self.symbol,
                    'entry_price': position['entry_price'],
                    'exit_price': filled_price,
                    'quantity': quantity,
                    'profit': profit,
                    'profit_pct': profit_pct,
                    'duration': time.time() - position['entry_time'],
                    'timestamp': datetime.now().isoformat()
                }
                
                self.trade_history.append(trade_record)
                
                logger.info(f"✅ SELL executed: {quantity} at ${filled_price:,.2f}")
                logger.info(f"💰 Profit: ${profit:+.4f} ({profit_pct:+.2f}%)")
                logger.info(f"📈 Total profit: ${self.total_profit:+.4f}")
                
                # Remove from open positions
                self.open_positions.remove(position)
                
                return True
                
        except Exception as e:
            logger.error(f"❌ Sell order failed: {e}")
            
        return False
        
    async def get_current_price(self) -> float:
        """Get current market price."""
        try:
            ticker = await self.connector.get_ticker_price(self.symbol)
            return float(ticker['price'])
        except:
            return 0.0
            
    async def check_positions(self):
        """Check open positions for profit taking or stop loss."""
        if not self.open_positions:
            return
            
        current_price = await self.get_current_price()
        if current_price <= 0:
            return
            
        positions_to_close = []
        
        for position in self.open_positions:
            if position['side'] == 'BUY':
                # Check profit target
                if current_price >= position['target_price']:
                    logger.info(f"🎯 Profit target hit: ${current_price:,.2f} >= ${position['target_price']:,.2f}")
                    positions_to_close.append(position)
                    
                # Check stop loss
                elif current_price <= position['stop_price']:
                    logger.warning(f"🛑 Stop loss hit: ${current_price:,.2f} <= ${position['stop_price']:,.2f}")
                    positions_to_close.append(position)
                    
                # Check time-based exit (hold max 60 seconds)
                elif time.time() - position['entry_time'] > 60:
                    logger.info(f"⏰ Time-based exit: {time.time() - position['entry_time']:.1f}s")
                    positions_to_close.append(position)
                    
        # Close positions
        for position in positions_to_close:
            await self.execute_sell_order(position)
            
    async def emergency_exit(self):
        """Emergency exit all positions."""
        logger.warning("🚨 EMERGENCY EXIT - Closing all positions!")
        
        for position in self.open_positions.copy():
            await self.execute_sell_order(position)
            
    def check_safety_limits(self) -> bool:
        """Check if safety limits are breached."""
        elapsed_time = time.time() - self.start_time
        
        # Time limit check
        if elapsed_time >= self.emergency_exit_time:
            logger.warning(f"⏰ Emergency exit time reached: {elapsed_time:.1f}s")
            return False
            
        # Trade count limit
        if self.trades_executed >= self.max_trades:
            logger.warning(f"📊 Maximum trades reached: {self.trades_executed}")
            return False
            
        # Total loss limit
        total_loss_pct = abs(self.total_profit) / self.max_position_size
        if self.total_profit < 0 and total_loss_pct >= self.max_total_loss_pct:
            logger.warning(f"💸 Total loss limit reached: {total_loss_pct:.2%}")
            return False
            
        return True
        
    async def trading_loop(self):
        """Main trading loop."""
        logger.info("🔄 Starting 5-minute trading session...")
        self.start_time = time.time()
        self.running = True
        
        try:
            while self.running:
                # Check safety limits
                if not self.check_safety_limits():
                    await self.emergency_exit()
                    break
                    
                # Check existing positions
                await self.check_positions()
                
                # Get trading signal if we can still trade
                if (self.trades_executed < self.max_trades and 
                    len(self.open_positions) == 0):
                    
                    signal = await self.get_market_signal()
                    
                    if signal['action'] == 'BUY' and signal['confidence'] > 0.5:
                        await self.execute_buy_order(signal)
                        
                # Progress update
                elapsed = time.time() - self.start_time
                remaining = self.trading_duration - elapsed
                
                if elapsed % 30 < 1:  # Every 30 seconds
                    logger.info(f"⏱️ Time remaining: {remaining:.0f}s | "
                              f"Trades: {self.trades_executed}/{self.max_trades} | "
                              f"Profit: ${self.total_profit:+.4f}")
                              
                # Check if time is up
                if elapsed >= self.trading_duration:
                    logger.info("⏰ 5-minute session complete!")
                    await self.emergency_exit()
                    break
                    
                await asyncio.sleep(1)  # 1-second loop
                
        except KeyboardInterrupt:
            logger.warning("🛑 Manual stop - Emergency exit!")
            await self.emergency_exit()
            
        except Exception as e:
            logger.error(f"❌ Trading loop error: {e}")
            await self.emergency_exit()
            
        finally:
            self.running = False
            
    def generate_report(self):
        """Generate final trading report."""
        elapsed_time = time.time() - self.start_time
        
        # Get final balance
        final_balance = self.start_balance + self.total_profit
        total_return_pct = (self.total_profit / self.max_position_size) * 100
        
        report = {
            'session_summary': {
                'duration_seconds': elapsed_time,
                'duration_minutes': elapsed_time / 60,
                'start_balance': self.start_balance,
                'final_balance': final_balance,
                'total_profit': self.total_profit,
                'return_percentage': total_return_pct,
                'trades_executed': self.trades_executed,
                'max_position_size': self.max_position_size
            },
            'trade_history': self.trade_history,
            'performance_metrics': {
                'profit_per_minute': self.total_profit / (elapsed_time / 60),
                'trades_per_minute': self.trades_executed / (elapsed_time / 60),
                'win_rate': len([t for t in self.trade_history if t['profit'] > 0]) / max(len(self.trade_history), 1) * 100,
                'avg_profit_per_trade': self.total_profit / max(self.trades_executed, 1),
                'best_trade': max(self.trade_history, key=lambda x: x['profit']) if self.trade_history else None,
                'worst_trade': min(self.trade_history, key=lambda x: x['profit']) if self.trade_history else None
            }
        }
        
        # Print summary
        print("\n" + "="*60)
        print("🏁 KIMERA 5-MINUTE PROFIT MAXIMIZER RESULTS")
        print("="*60)
        print(f"⏱️  Duration: {elapsed_time/60:.2f} minutes")
        print(f"💰 Starting Balance: ${self.start_balance:.2f}")
        print(f"💰 Final Balance: ${final_balance:.2f}")
        print(f"📈 Total Profit: ${self.total_profit:+.4f}")
        print(f"📊 Return: {total_return_pct:+.2f}%")
        print(f"🔄 Trades Executed: {self.trades_executed}")
        
        if self.trade_history:
            winning_trades = [t for t in self.trade_history if t['profit'] > 0]
            print(f"🎯 Win Rate: {len(winning_trades)}/{len(self.trade_history)} ({len(winning_trades)/len(self.trade_history)*100:.1f}%)")
            print(f"💎 Best Trade: ${max(self.trade_history, key=lambda x: x['profit'])['profit']:+.4f}")
            print(f"💸 Worst Trade: ${min(self.trade_history, key=lambda x: x['profit'])['profit']:+.4f}")
            
        print("="*60)
        
        # Save report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"profit_maximizer_report_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        logger.info(f"📄 Report saved: {filename}")
        
        return report

async def main():
    """Main execution function."""
    maximizer = ProfitMaximizer()
    
    # Initialize
    if not await maximizer.initialize():
        logger.error("❌ Failed to initialize. Exiting.")
        return
        
    print("\n" + "="*60)
    print("🚀 KIMERA 5-MINUTE LIVE PROFIT MAXIMIZER")
    print("="*60)
    print("⚠️  WARNING: REAL MONEY TRADING")
    print(f"💰 Maximum position: ${maximizer.max_position_size}")
    print(f"⏱️  Duration: {maximizer.trading_duration/60} minutes")
    print(f"🎯 Target: 0.2% profit per trade")
    print(f"🛑 Stop loss: 0.5% per trade")
    print("="*60)
    
    # Countdown
    for i in range(5, 0, -1):
        print(f"🚀 Starting in {i}...")
        await asyncio.sleep(1)
        
    print("🔥 TRADING SESSION STARTED!")
    print("="*60)
    
    # Run trading session
    await maximizer.trading_loop()
    
    # Generate final report
    report = maximizer.generate_report()
    
    return report

if __name__ == "__main__":
    asyncio.run(main()) 