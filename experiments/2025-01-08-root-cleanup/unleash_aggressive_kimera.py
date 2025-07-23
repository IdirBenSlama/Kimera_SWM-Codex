#!/usr/bin/env python3
"""
UNLEASHED AGGRESSIVE KIMERA TRADER
==================================

Hyper-aggressive autonomous trader with NO SAFETY LIMITS.
Trades in ANY market conditions with lower confidence thresholds.
PURE RISK - MAXIMUM AGGRESSION
"""

import os
import sys
import asyncio
import json
from datetime import datetime

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from src.trading.autonomous_kimera_trader import create_autonomous_kimera
from src.config.config_integration import setup_kimera_logger

def setup_aggressive_logger():
    """Setup logger for aggressive trading"""
    return setup_kimera_logger("KIMERA_AGGRESSIVE", "logs/aggressive_kimera.log")

class AggressiveKimeraTrader:
    """
    Hyper-aggressive version of Kimera with NO safety limits
    """
    
    def __init__(self, api_key: str, target_eur: float = 100.0):
        """Initialize aggressive trader"""
        self.api_key = api_key
        self.target_eur = target_eur
        self.base_trader = create_autonomous_kimera(api_key, target_eur)
        self.logger = setup_aggressive_logger()
        
        # AGGRESSIVE SETTINGS
        self.confidence_threshold = 0.3  # Much lower threshold (30%)
        self.conviction_multiplier = 2.0  # Double conviction
        self.max_position_pct = 0.9  # Up to 90% of portfolio
        self.opportunistic_mode = True
        
        self.logger.info("🔥 AGGRESSIVE KIMERA TRADER INITIALIZED")
        self.logger.info(f"   Confidence Threshold: {self.confidence_threshold:.1%}")
        self.logger.info(f"   Max Position Size: {self.max_position_pct:.1%}")
        self.logger.info("   MAXIMUM AGGRESSION MODE")
    
    def make_signal_aggressive(self, signal):
        """Transform any signal into aggressive signal"""
        if not signal:
            return None
        
        # Boost confidence artificially for sideways markets
        if signal.confidence < self.confidence_threshold:
            original_confidence = signal.confidence
            signal.confidence = min(signal.confidence * 2.0, 0.8)  # Boost but cap at 80%
            self.logger.info(f"🚀 Boosted confidence: {original_confidence:.2f} → {signal.confidence:.2f}")
        
        # Increase conviction
        signal.conviction = min(signal.conviction * self.conviction_multiplier, 1.0)
        
        # Increase allocation for aggressive trading
        signal.suggested_allocation_pct = min(
            signal.suggested_allocation_pct * 1.5,
            self.max_position_pct
        )
        
        # Make more opportunistic in sideways markets
        if signal.market_regime.value == 'sideways' and signal.action == 'hold':
            # Force action in sideways markets
            import random
            signal.action = random.choice(['buy', 'sell'])
            signal.confidence = max(signal.confidence, 0.4)
            signal.reasoning.append("Aggressive sideways market exploitation")
            self.logger.info(f"🎯 Forced action in sideways market: {signal.action}")
        
        return signal
    
    async def aggressive_trading_cycle(self):
        """Ultra-aggressive trading cycle"""
        self.logger.info("🔥 AGGRESSIVE TRADING CYCLE STARTED")
        
        # Symbols to trade aggressively
        symbols = ['bitcoin', 'ethereum', 'solana', 'cardano', 'polkadot']
        trades_executed = 0
        
        try:
            # Fetch market data
            for symbol in symbols:
                await self.base_trader.fetch_market_data(symbol)
            
            # Manage existing positions aggressively
            await self.base_trader.manage_positions()
            
            # Generate aggressive signals
            for symbol in symbols:
                if symbol not in self.base_trader.positions or len(self.base_trader.positions) < 5:
                    # Generate base signal
                    base_signal = self.base_trader.generate_cognitive_signal(symbol)
                    
                    # Make it aggressive
                    aggressive_signal = self.make_signal_aggressive(base_signal)
                    
                    if aggressive_signal and aggressive_signal.confidence >= self.confidence_threshold:
                        self.logger.info(f"🚀 AGGRESSIVE SIGNAL FOR {symbol.upper()}:")
                        self.logger.info(f"   Action: {aggressive_signal.action}")
                        self.logger.info(f"   Confidence: {aggressive_signal.confidence:.2f}")
                        self.logger.info(f"   Conviction: {aggressive_signal.conviction:.2f}")
                        self.logger.info(f"   Allocation: {aggressive_signal.suggested_allocation_pct:.1%}")
                        
                        # Execute aggressively
                        success = await self.base_trader.execute_autonomous_trade(aggressive_signal)
                        if success:
                            trades_executed += 1
                            self.logger.info(f"✅ AGGRESSIVE TRADE EXECUTED: {symbol}")
                        
                        # Don't overwhelm - one trade per cycle
                        if trades_executed >= 1:
                            break
            
            # Force trading if no natural signals
            if trades_executed == 0 and self.opportunistic_mode:
                await self.force_opportunistic_trade(symbols)
            
            # Status update
            status = await self.base_trader.get_portfolio_status()
            self.logger.info(f"🔥 AGGRESSIVE STATUS:")
            self.logger.info(f"   Portfolio: €{status['portfolio_value_eur']:.2f}")
            self.logger.info(f"   Progress: {status['progress_pct']:.1f}%")
            self.logger.info(f"   Positions: {status['active_positions']}")
            self.logger.info(f"   Trades This Cycle: {trades_executed}")
            
            return status['portfolio_value_eur'] >= self.target_eur
            
        except Exception as e:
            self.logger.error(f"❌ Aggressive cycle failed: {e}")
            return False
    
    async def force_opportunistic_trade(self, symbols):
        """Force a trade when no natural signals exist"""
        self.logger.info("🎲 FORCING OPPORTUNISTIC TRADE")
        
        try:
            import random
            # Pick random symbol
            symbol = random.choice(symbols)
            
            # Get market data
            df = await self.base_trader.fetch_market_data(symbol)
            if df.empty:
                return
            
            current_price = df['price'].tail(1).iloc[0]
            
            # Force signal creation
            from src.trading.autonomous_kimera_trader import CognitiveSignal, TradingStrategy, MarketRegime
            
            forced_signal = CognitiveSignal(
                symbol=symbol,
                action=random.choice(['buy', 'sell']),
                confidence=0.4,  # Forced confidence
                conviction=0.6,  # Forced conviction
                reasoning=["Forced opportunistic trade", "Aggressive mode override", "Market exploitation"],
                strategy=TradingStrategy.VOLATILITY_HARVESTER,
                market_regime=MarketRegime.SIDEWAYS,
                suggested_allocation_pct=0.3,  # 30% allocation
                max_risk_pct=0.3,
                entry_price=current_price,
                stop_loss=None,  # No stop loss - pure aggression
                profit_targets=[current_price * 1.05, current_price * 1.1],
                holding_period_hours=2.0,
                technical_score=0.5,
                fundamental_score=0.5,
                sentiment_score=0.5,
                momentum_score=0.5,
                timestamp=datetime.now()
            )
            
            self.logger.info(f"🎲 FORCED TRADE: {forced_signal.action} {symbol} at €{current_price:.2f}")
            
            # Execute forced trade
            success = await self.base_trader.execute_autonomous_trade(forced_signal)
            if success:
                self.logger.info("✅ FORCED TRADE EXECUTED")
            else:
                self.logger.warning("❌ FORCED TRADE FAILED")
                
        except Exception as e:
            self.logger.error(f"❌ Forced trade failed: {e}")
    
    async def run_aggressive_trader(self, cycle_interval_minutes: int = 10):
        """Run aggressive trader with shorter cycles"""
        self.logger.info("🔥 UNLEASHED AGGRESSIVE KIMERA STARTED")
        self.logger.info(f"   Target: €{self.target_eur}")
        self.logger.info(f"   Cycle: {cycle_interval_minutes} minutes")
        self.logger.info("   NO LIMITS - MAXIMUM AGGRESSION")
        
        try:
            cycle_count = 0
            while True:
                cycle_count += 1
                self.logger.info(f"🔥 AGGRESSIVE CYCLE #{cycle_count}")
                
                target_reached = await self.aggressive_trading_cycle()
                
                if target_reached:
                    self.logger.info("🎯 AGGRESSIVE TARGET REACHED!")
                    break
                
                # Shorter cycles for more aggression
                await asyncio.sleep(cycle_interval_minutes * 60)
                
        except KeyboardInterrupt:
            self.logger.info("🛑 Aggressive trader stopped")
        except Exception as e:
            self.logger.error(f"❌ Aggressive trader crashed: {e}")

async def main():
    """Launch aggressive Kimera"""
    print("🔥 UNLEASHING AGGRESSIVE KIMERA")
    print("=" * 40)
    print("⚠️  WARNING: MAXIMUM AGGRESSION MODE")
    print("⚠️  NO SAFETY LIMITS")
    print("⚠️  WILL TRADE IN ANY CONDITIONS")
    print("⚠️  PURE RISK MODE")
    print("=" * 40)
    
    API_KEY = "9268de76-b5f4-4683-b593-327fb2c19503"
    
    # Create aggressive trader
    aggressive_trader = AggressiveKimeraTrader(API_KEY, target_eur=100.0)
    
    # Run with 10-minute cycles (more frequent)
    await aggressive_trader.run_aggressive_trader(cycle_interval_minutes=10)

if __name__ == "__main__":
    asyncio.run(main()) 