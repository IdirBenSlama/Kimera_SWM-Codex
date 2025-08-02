#!/usr/bin/env python3
"""
Kimera Aggressive 10-Minute Trader
=================================

EXTREME RISK WARNING: This system attempts to turn $50 into $300 in 10 minutes.
This is EXTREMELY HIGH RISK and you could lose all your money.

Strategy: Ultra-aggressive, high-frequency trading with maximum position sizing
Target: 500% return in 10 minutes
Risk Level: MAXIMUM

Only use with money you can afford to lose completely.
"""

import asyncio
import aiohttp
import hmac
import hashlib
import time
import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import random
import math
from urllib.parse import urlencode

# Configure aggressive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('aggressive_trader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AggressiveStrategy(Enum):
    SCALPING_FURY = "scalping_fury"
    MOMENTUM_BLAST = "momentum_blast"
    VOLATILITY_SURF = "volatility_surf"
    BREAKOUT_HUNT = "breakout_hunt"
    REVERSAL_SNIPE = "reversal_snipe"
    CHAOS_EXPLOIT = "chaos_exploit"

@dataclass
class AggressiveTrade:
    """High-frequency trade execution"""
    symbol: str
    side: str
    quantity: float
    price: float
    timestamp: datetime
    strategy: AggressiveStrategy
    confidence: float
    expected_return: float
    risk_score: float

class BinanceAggressiveTrader:
    """Ultra-aggressive Binance trader"""
    
    def __init__(self, api_key: str, secret_key: str):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = "https://api.binance.com"
        self.session = None
        
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
    
    async def get_account_balance(self) -> Dict[str, float]:
        """Get account balance"""
        url = f"{self.base_url}/api/v3/account"
        timestamp = int(time.time() * 1000)
        params = {"timestamp": timestamp}
        signature = self._generate_signature(params)
        params["signature"] = signature
        headers = {"X-MBX-APIKEY": self.api_key}
        
        async with self.session.get(url, params=params, headers=headers) as response:
            data = await response.json()
            balances = {}
            for balance in data.get('balances', []):
                asset = balance['asset']
                free = float(balance['free'])
                locked = float(balance['locked'])
                if free > 0 or locked > 0:
                    balances[asset] = {'free': free, 'locked': locked, 'total': free + locked}
            return balances
    
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get 24hr ticker statistics"""
        url = f"{self.base_url}/api/v3/ticker/24hr"
        params = {"symbol": symbol}
        
        async with self.session.get(url, params=params) as response:
            return await response.json()
    
    async def get_orderbook(self, symbol: str, limit: int = 10) -> Dict[str, Any]:
        """Get order book"""
        url = f"{self.base_url}/api/v3/depth"
        params = {"symbol": symbol, "limit": limit}
        
        async with self.session.get(url, params=params) as response:
            return await response.json()
    
    async def get_klines(self, symbol: str, interval: str = "1m", limit: int = 100) -> List[List]:
        """Get kline/candlestick data"""
        url = f"{self.base_url}/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        
        async with self.session.get(url, params=params) as response:
            return await response.json()
    
    async def place_market_order(self, symbol: str, side: str, quantity: float) -> Dict[str, Any]:
        """Place aggressive market order"""
        url = f"{self.base_url}/api/v3/order"
        timestamp = int(time.time() * 1000)
        
        # Format quantity properly
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
            return await response.json()

class AggressiveMarketAnalyzer:
    """Ultra-fast market analysis for aggressive trading"""
    
    def __init__(self, trader: BinanceAggressiveTrader):
        self.trader = trader
        self.price_history = {}
        self.volume_history = {}
        
    async def analyze_ultra_fast(self, symbol: str) -> Dict[str, Any]:
        """Ultra-fast market analysis"""
        try:
            # Get multiple data sources simultaneously
            ticker_task = self.trader.get_ticker(symbol)
            orderbook_task = self.trader.get_orderbook(symbol, 20)
            klines_task = self.trader.get_klines(symbol, "1m", 50)
            
            ticker, orderbook, klines = await asyncio.gather(
                ticker_task, orderbook_task, klines_task
            )
            
            # Calculate aggressive metrics
            analysis = self._calculate_aggressive_metrics(ticker, orderbook, klines)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Market analysis failed: {e}")
            return {}
    
    def _calculate_aggressive_metrics(self, ticker: Dict, orderbook: Dict, klines: List) -> Dict[str, Any]:
        """Calculate ultra-aggressive trading metrics"""
        
        # Price metrics
        price = float(ticker['lastPrice'])
        price_change_pct = float(ticker['priceChangePercent'])
        volume = float(ticker['volume'])
        
        # Volatility (key for aggressive trading)
        volatility = abs(price_change_pct)
        
        # Order book analysis
        bids = [[float(bid[0]), float(bid[1])] for bid in orderbook.get('bids', [])]
        asks = [[float(ask[0]), float(ask[1])] for ask in orderbook.get('asks', [])]
        
        spread = (asks[0][0] - bids[0][0]) / bids[0][0] * 100 if bids and asks else 0
        
        # Volume analysis
        volume_24h = float(ticker['volume'])
        volume_score = min(volume_24h / 1000000, 1.0)  # Normalize volume
        
        # Momentum analysis from klines
        if len(klines) >= 10:
            closes = [float(k[4]) for k in klines[-10:]]
            momentum = (closes[-1] - closes[0]) / closes[0] * 100
        else:
            momentum = 0
        
        # Aggressive opportunity score
        opportunity_score = (
            volatility * 0.4 +  # High volatility = high opportunity
            abs(momentum) * 0.3 +  # Strong momentum
            volume_score * 0.2 +  # High volume
            min(spread, 0.5) * 0.1  # Tight spread
        )
        
        # Trading signals
        signals = []
        if volatility > 5:
            signals.append("HIGH_VOLATILITY")
        if abs(momentum) > 2:
            signals.append("STRONG_MOMENTUM")
        if volume_score > 0.7:
            signals.append("HIGH_VOLUME")
        if spread < 0.1:
            signals.append("TIGHT_SPREAD")
        
        return {
            'symbol': ticker['symbol'],
            'price': price,
            'price_change_pct': price_change_pct,
            'volatility': volatility,
            'momentum': momentum,
            'volume_score': volume_score,
            'spread': spread,
            'opportunity_score': opportunity_score,
            'signals': signals,
            'timestamp': datetime.now()
        }

class AggressiveStrategyEngine:
    """Ultra-aggressive trading strategies"""
    
    def __init__(self):
        self.strategies = {
            AggressiveStrategy.SCALPING_FURY: self._scalping_fury,
            AggressiveStrategy.MOMENTUM_BLAST: self._momentum_blast,
            AggressiveStrategy.VOLATILITY_SURF: self._volatility_surf,
            AggressiveStrategy.BREAKOUT_HUNT: self._breakout_hunt,
            AggressiveStrategy.REVERSAL_SNIPE: self._reversal_snipe,
            AggressiveStrategy.CHAOS_EXPLOIT: self._chaos_exploit
        }
    
    def select_strategy(self, analysis: Dict[str, Any]) -> AggressiveStrategy:
        """Select the most aggressive strategy for current market conditions"""
        
        volatility = analysis.get('volatility', 0)
        momentum = analysis.get('momentum', 0)
        volume_score = analysis.get('volume_score', 0)
        signals = analysis.get('signals', [])
        
        # Ultra-aggressive strategy selection
        if volatility > 10 and "HIGH_VOLATILITY" in signals:
            return AggressiveStrategy.CHAOS_EXPLOIT
        elif abs(momentum) > 5 and "STRONG_MOMENTUM" in signals:
            return AggressiveStrategy.MOMENTUM_BLAST
        elif volatility > 5 and volume_score > 0.8:
            return AggressiveStrategy.VOLATILITY_SURF
        elif "TIGHT_SPREAD" in signals and volatility > 3:
            return AggressiveStrategy.SCALPING_FURY
        elif volatility > 7:
            return AggressiveStrategy.BREAKOUT_HUNT
        else:
            return AggressiveStrategy.REVERSAL_SNIPE
    
    def _scalping_fury(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Ultra-fast scalping strategy"""
        return {
            'action': 'buy' if analysis['momentum'] > 0 else 'sell',
            'confidence': min(analysis['opportunity_score'] * 1.5, 1.0),
            'position_size': 0.9,  # 90% of available capital
            'expected_return': 0.02,  # 2% per trade
            'max_hold_time': 30  # 30 seconds
        }
    
    def _momentum_blast(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Momentum following with maximum aggression"""
        return {
            'action': 'buy' if analysis['momentum'] > 0 else 'sell',
            'confidence': min(analysis['opportunity_score'] * 1.3, 1.0),
            'position_size': 0.95,  # 95% of available capital
            'expected_return': 0.05,  # 5% per trade
            'max_hold_time': 60  # 1 minute
        }
    
    def _volatility_surf(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Ride volatility waves"""
        return {
            'action': 'buy' if analysis['price_change_pct'] < -2 else 'sell',
            'confidence': min(analysis['volatility'] / 10, 1.0),
            'position_size': 0.8,  # 80% of available capital
            'expected_return': 0.03,  # 3% per trade
            'max_hold_time': 45  # 45 seconds
        }
    
    def _breakout_hunt(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Hunt for breakout opportunities"""
        return {
            'action': 'buy' if analysis['volatility'] > 5 else 'sell',
            'confidence': min(analysis['opportunity_score'] * 1.2, 1.0),
            'position_size': 0.85,  # 85% of available capital
            'expected_return': 0.04,  # 4% per trade
            'max_hold_time': 90  # 1.5 minutes
        }
    
    def _reversal_snipe(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Snipe reversal opportunities"""
        return {
            'action': 'buy' if analysis['price_change_pct'] < -5 else 'sell',
            'confidence': min(analysis['opportunity_score'] * 1.1, 1.0),
            'position_size': 0.75,  # 75% of available capital
            'expected_return': 0.06,  # 6% per trade
            'max_hold_time': 120  # 2 minutes
        }
    
    def _chaos_exploit(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Exploit extreme market chaos"""
        return {
            'action': 'buy' if random.random() > 0.5 else 'sell',  # Chaos trading
            'confidence': min(analysis['volatility'] / 5, 1.0),
            'position_size': 1.0,  # 100% of available capital - MAXIMUM RISK
            'expected_return': 0.10,  # 10% per trade - EXTREME TARGET
            'max_hold_time': 20  # 20 seconds - ULTRA FAST
        }

class KimeraAggressive10MinTrader:
    """Ultra-aggressive 10-minute trading system with auto-liquidation and force trade mode"""
    
    def __init__(self, starting_capital: float = 50.0, target_capital: float = 300.0, auto_liquidate: bool = True, force_trade: bool = True):
        self.starting_capital = starting_capital
        self.target_capital = target_capital
        self.current_capital = starting_capital
        self.target_return = (target_capital / starting_capital - 1) * 100  # 500%
        self.auto_liquidate = auto_liquidate
        self.force_trade = force_trade
        
        # Trading components
        self.trader = None
        self.analyzer = None
        self.strategy_engine = AggressiveStrategyEngine()
        
        # Trading state
        self.trades_executed = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.trade_history = []
        self.start_time = None
        self.is_running = False
        
        # Aggressive parameters
        self.trading_symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'SOLUSDT']
        self.trade_interval = 5  # 5 seconds between trades
        self.max_trades = 120  # Max trades in 10 minutes
        
        logger.info(f"🚨 AGGRESSIVE TRADER INITIALIZED")
        logger.info(f"💰 Starting Capital: ${starting_capital}")
        logger.info(f"🎯 Target Capital: ${target_capital}")
        logger.info(f"📈 Target Return: {self.target_return:.0f}%")
        logger.info(f"⏰ Time Limit: 10 minutes")
        logger.info(f"⚡ Strategy: MAXIMUM AGGRESSION")
    
    async def start_aggressive_trading(self):
        """Start ultra-aggressive 10-minute trading session with auto-liquidation and force trade mode"""
        try:
            # Initialize trader
            api_key = os.getenv('BINANCE_API_KEY')
            secret_key = os.getenv('BINANCE_SECRET_KEY')
            
            if not api_key or not secret_key:
                raise ValueError("API credentials not found")
            
            async with BinanceAggressiveTrader(api_key, secret_key) as trader:
                self.trader = trader
                self.analyzer = AggressiveMarketAnalyzer(trader)
                self.start_time = datetime.now()
                self.is_running = True
                
                logger.info("🚀 STARTING AGGRESSIVE TRADING SESSION")
                logger.info("=" * 50)
                
                # Auto-liquidate all non-USDT assets
                if self.auto_liquidate:
                    await self._auto_liquidate_all_assets()
                    await asyncio.sleep(2)  # Wait for balances to update
                
                # Verify starting balance
                balances = await trader.get_account_balance()
                usdt_balance = balances.get('USDT', {}).get('free', 0)
                
                if usdt_balance < self.starting_capital:
                    raise ValueError(f"Insufficient balance: ${usdt_balance:.2f} < ${self.starting_capital}")
                
                logger.info(f"✅ Starting balance verified: ${usdt_balance:.2f}")
                
                # Start trading loop
                await self._aggressive_trading_loop()
                
        except Exception as e:
            logger.error(f"Failed to start aggressive trading: {e}")
            raise

    async def _auto_liquidate_all_assets(self):
        """Sell all non-USDT assets for USDT at market price"""
        logger.info("🔄 AUTO-LIQUIDATING ALL NON-USDT ASSETS...")
        balances = await self.trader.get_account_balance()
        for asset, info in balances.items():
            if asset != 'USDT' and info['free'] > 0:
                symbol = f"{asset}USDT"
                try:
                    price = await self.trader.get_ticker(symbol)
                    if 'lastPrice' in price and float(price['lastPrice']) > 0:
                        quantity = info['free']
                        logger.info(f"💸 Selling {quantity} {asset} for USDT...")
                        result = await self.trader.place_market_order(symbol, 'SELL', quantity)
                        if 'orderId' in result:
                            logger.info(f"✅ Sold {quantity} {asset} for USDT (Order ID: {result['orderId']})")
                        else:
                            logger.warning(f"❌ Failed to sell {asset}: {result}")
                except Exception as e:
                    logger.warning(f"❌ Error selling {asset}: {e}")

    async def _aggressive_trading_loop(self):
        """Main aggressive trading loop (now uses real strategy engine and market analysis)"""
        while self.is_running:
            try:
                # Check time limit
                elapsed = datetime.now() - self.start_time
                if elapsed.total_seconds() > 600:  # 10 minutes
                    logger.info("⏰ 10-minute time limit reached")
                    break
                # Check max trades
                if self.trades_executed >= self.max_trades:
                    logger.info("📊 Maximum trades reached")
                    break
                # Execute one intelligent aggressive trading cycle
                await self._execute_aggressive_cycle()
                # Brief pause between cycles
                await asyncio.sleep(self.trade_interval)
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(1)
        # Final report
        await self._generate_final_report()

    async def _execute_aggressive_cycle(self):
        """Execute one aggressive trading cycle"""
        try:
            # Select most volatile symbol
            best_symbol = await self._select_best_symbol()
            
            # Ultra-fast analysis
            analysis = await self.analyzer.analyze_ultra_fast(best_symbol)
            
            if not analysis:
                return
            
            # Select aggressive strategy
            strategy = self.strategy_engine.select_strategy(analysis)
            
            # Get strategy parameters
            strategy_params = self.strategy_engine.strategies[strategy](analysis)
            
            # Check if trade is viable
            if strategy_params['confidence'] < 0.5:
                logger.info(f"📊 {best_symbol}: Low confidence ({strategy_params['confidence']:.2f})")
                return
            
            # Calculate position size
            current_balance = await self._get_current_balance()
            position_size_usd = current_balance * strategy_params['position_size']
            
            # Minimum position check
            if position_size_usd < 10:
                logger.info(f"⚠️ Position size too small: ${position_size_usd:.2f}")
                return
            
            # Execute trade
            await self._execute_aggressive_trade(best_symbol, strategy_params, position_size_usd)
            
        except Exception as e:
            logger.error(f"Error in aggressive cycle: {e}")
    
    async def _select_best_symbol(self) -> str:
        """Select the most volatile/profitable symbol"""
        try:
            # Get quick analysis for all symbols
            analyses = []
            for symbol in self.trading_symbols:
                try:
                    ticker = await self.trader.get_ticker(symbol)
                    volatility = abs(float(ticker['priceChangePercent']))
                    volume = float(ticker['volume'])
                    
                    # Score based on volatility and volume
                    score = volatility * 0.7 + min(volume / 1000000, 1.0) * 0.3
                    
                    analyses.append({
                        'symbol': symbol,
                        'volatility': volatility,
                        'volume': volume,
                        'score': score
                    })
                except Exception as e:
                    logger.error(f"Error in kimera_aggressive_10min_trader.py: {e}", exc_info=True)
                    raise  # Re-raise for proper error handling
                    continue
            
            # Select highest scoring symbol
            if analyses:
                best = max(analyses, key=lambda x: x['score'])
                logger.info(f"🎯 Selected {best['symbol']} (volatility: {best['volatility']:.2f}%, score: {best['score']:.2f})")
                return best['symbol']
            
            return 'BTCUSDT'  # Fallback
            
        except Exception as e:
            logger.error(f"Symbol selection failed: {e}")
            return 'BTCUSDT'
    
    async def _execute_aggressive_trade(self, symbol: str, strategy_params: Dict, position_size_usd: float):
        """Execute aggressive trade"""
        try:
            action = strategy_params['action']
            confidence = strategy_params['confidence']
            expected_return = strategy_params['expected_return']
            
            # Get current price
            ticker = await self.trader.get_ticker(symbol)
            current_price = float(ticker['lastPrice'])
            
            # Calculate quantity
            if action == 'buy':
                quantity = position_size_usd / current_price
            else:
                # For sell, check if we have position
                balances = await self.trader.get_account_balance()
                asset = symbol.replace('USDT', '')
                asset_balance = balances.get(asset, {}).get('free', 0)
                
                if asset_balance == 0:
                    logger.info(f"⚠️ No {asset} to sell")
                    return
                
                quantity = min(asset_balance * 0.95, position_size_usd / current_price)
            
            # Execute trade
            logger.info(f"🚀 EXECUTING AGGRESSIVE TRADE:")
            logger.info(f"   Symbol: {symbol}")
            logger.info(f"   Action: {action.upper()}")
            logger.info(f"   Quantity: {quantity:.6f}")
            logger.info(f"   Price: ${current_price:.2f}")
            logger.info(f"   Value: ${position_size_usd:.2f}")
            logger.info(f"   Confidence: {confidence:.2f}")
            logger.info(f"   Expected Return: {expected_return*100:.1f}%")
            
            # Place order
            order_result = await self.trader.place_market_order(symbol, action, quantity)
            
            if 'orderId' in order_result:
                self.trades_executed += 1
                
                # Record trade
                trade = AggressiveTrade(
                    symbol=symbol,
                    side=action,
                    quantity=quantity,
                    price=current_price,
                    timestamp=datetime.now(),
                    strategy=AggressiveStrategy.SCALPING_FURY,
                    confidence=confidence,
                    expected_return=expected_return,
                    risk_score=1.0
                )
                
                self.trade_history.append(trade)
                
                logger.info(f"✅ Trade #{self.trades_executed} executed successfully")
                logger.info(f"   Order ID: {order_result['orderId']}")
                
                # Update balance
                await asyncio.sleep(0.5)  # Brief pause for order to fill
                self.current_capital = await self._get_current_balance()
                
                # Calculate profit/loss
                profit = self.current_capital - self.starting_capital
                profit_pct = (profit / self.starting_capital) * 100
                
                logger.info(f"💰 Current Capital: ${self.current_capital:.2f}")
                logger.info(f"📈 Profit: ${profit:.2f} ({profit_pct:+.1f}%)")
                logger.info(f"🎯 Progress: {(self.current_capital/self.target_capital)*100:.1f}% to target")
                
            else:
                logger.error(f"❌ Trade failed: {order_result}")
                
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
    
    async def _get_current_balance(self) -> float:
        """Get current total balance in USDT"""
        try:
            balances = await self.trader.get_account_balance()
            total_usdt = balances.get('USDT', {}).get('free', 0)
            
            # Add value of other assets
            for asset, balance_info in balances.items():
                if asset != 'USDT' and balance_info['free'] > 0:
                    try:
                        symbol = f"{asset}USDT"
                        ticker = await self.trader.get_ticker(symbol)
                        price = float(ticker['lastPrice'])
                        total_usdt += balance_info['free'] * price
                    except Exception as e:
                        logger.error(f"Error in kimera_aggressive_10min_trader.py: {e}", exc_info=True)
                        raise  # Re-raise for proper error handling
                        continue
            
            return total_usdt
            
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return self.starting_capital
    
    async def _generate_final_report(self):
        """Generate final trading report"""
        try:
            final_balance = await self._get_current_balance()
            total_profit = final_balance - self.starting_capital
            total_return = (total_profit / self.starting_capital) * 100
            elapsed_time = (datetime.now() - self.start_time).total_seconds()
            
            # Calculate performance metrics
            win_rate = (self.winning_trades / max(1, self.trades_executed)) * 100
            avg_profit_per_trade = total_profit / max(1, self.trades_executed)
            
            logger.info("=" * 60)
            logger.info("🏁 AGGRESSIVE TRADING SESSION COMPLETE")
            logger.info("=" * 60)
            logger.info(f"⏰ Duration: {elapsed_time:.0f} seconds ({elapsed_time/60:.1f} minutes)")
            logger.info(f"💰 Starting Capital: ${self.starting_capital:.2f}")
            logger.info(f"💰 Final Capital: ${final_balance:.2f}")
            logger.info(f"📈 Total Profit: ${total_profit:.2f}")
            logger.info(f"📊 Total Return: {total_return:+.1f}%")
            logger.info(f"🎯 Target Achievement: {(final_balance/self.target_capital)*100:.1f}%")
            logger.info(f"🔥 Trades Executed: {self.trades_executed}")
            logger.info(f"🏆 Win Rate: {win_rate:.1f}%")
            logger.info(f"💵 Avg Profit/Trade: ${avg_profit_per_trade:.2f}")
            
            if final_balance >= self.target_capital:
                logger.info("🎉 TARGET ACHIEVED! MISSION SUCCESSFUL!")
            else:
                logger.info("⚠️ Target not reached, but valuable learning experience")
            
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")

async def main():
    """Main entry point for aggressive trading"""
    print("🚨 KIMERA AGGRESSIVE 10-MINUTE TRADER 🚨")
    print("=" * 50)
    print("⚠️  EXTREME RISK WARNING:")
    print("   • This attempts 500% return in 10 minutes")
    print("   • You could lose ALL your money")
    print("   • Only use money you can afford to lose")
    print("   • This is experimental and very risky")
    print("=" * 50)
    
    # Get user confirmation
    print("\n🤔 Do you want to proceed with EXTREME RISK trading?")
    print("   Type 'I UNDERSTAND THE RISKS' to continue")
    print("   Type anything else to cancel")
    
    response = input("\nYour response: ").strip()
    
    if response != "I UNDERSTAND THE RISKS":
        print("\n👋 Trading cancelled. Stay safe!")
        return
    
    # Final confirmation
    print("\n💰 Starting with $50 to target $300 in 10 minutes")
    print("🔥 This is MAXIMUM RISK trading")
    print("⚡ Press Enter to start, or Ctrl+C to cancel")
    
    try:
        input()
    except KeyboardInterrupt:
        print("\n🛑 Cancelled by user")
        return
    
    # Start aggressive trading
    trader = KimeraAggressive10MinTrader(starting_capital=50.0, target_capital=300.0)
    
    try:
        await trader.start_aggressive_trading()
    except KeyboardInterrupt:
        print("\n🛑 Trading stopped by user")
    except Exception as e:
        print(f"\n❌ Trading error: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 