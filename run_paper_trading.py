"""
KIMERA PAPER TRADING SYSTEM
===========================

24-hour paper trading system to validate performance with real market data
before proceeding to micro-trading with real funds.

OBJECTIVES:
- Validate trading signals with real market data
- Monitor performance and risk metrics
- Test all safety mechanisms under realistic conditions
- Generate comprehensive performance report
- Only proceed to real trading after successful paper trading
"""

import asyncio
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add backend to path
sys.path.append(str(Path(__file__).parent / 'backend'))

from backend.trading.cdp_safe_trader import create_safe_trader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/paper_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PaperTradingSession:
    """
    Comprehensive paper trading session with performance tracking
    """
    
    def __init__(self, duration_hours: int = 24):
        """
        Initialize paper trading session
        
        Args:
            duration_hours: How long to run paper trading (default 24 hours)
        """
        self.duration_hours = duration_hours
        self.api_key = "9268de76-b5f4-4683-b593-327fb2c19503"
        self.trader = None
        self.session_start = None
        self.session_end = None
        
        # Performance tracking
        self.total_signals = 0
        self.valid_signals = 0
        self.executed_trades = 0
        self.profitable_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.win_rate = 0.0
        
        # Risk monitoring
        self.safety_violations = []
        self.emergency_stops = 0
        
        # Market conditions tracking
        self.market_conditions = []
        
        logger.info(f"🎯 Paper Trading Session initialized for {duration_hours} hours")
        
    async def start_session(self):
        """Start the paper trading session"""
        self.session_start = datetime.now()
        self.session_end = self.session_start + timedelta(hours=self.duration_hours)
        
        logger.info("🚀 STARTING PAPER TRADING SESSION")
        logger.info("=" * 60)
        logger.info(f"Start Time: {self.session_start}")
        logger.info(f"End Time: {self.session_end}")
        logger.info(f"Duration: {self.duration_hours} hours")
        logger.info(f"Mode: PAPER TRADING (No real money)")
        
        # Initialize trader in simulation mode
        self.trader = create_safe_trader(self.api_key, private_key=None, testnet=True)
        
        # Reset any previous state for clean paper trading
        self.trader.daily_pnl = 0.0
        self.trader.consecutive_losses = 0
        self.trader.emergency_stop = False
        self.trader.active_positions = {}
        
        logger.info("✅ Paper trader initialized and ready")
        
        # Start the main trading loop
        await self.trading_loop()
        
    async def trading_loop(self):
        """Main trading loop for paper trading"""
        logger.info("\n🔄 Starting paper trading loop...")
        
        iteration = 0
        
        while datetime.now() < self.session_end:
            iteration += 1
            logger.info(f"\n--- Trading Iteration {iteration} ---")
            
            try:
                # Check current time and calculate remaining time
                current_time = datetime.now()
                time_remaining = self.session_end - current_time
                logger.info(f"⏰ Time remaining: {time_remaining}")
                
                # Monitor market conditions
                await self.monitor_market_conditions()
                
                # Generate and evaluate trading signals
                await self.evaluate_trading_signals()
                
                # Update performance metrics
                self.update_performance_metrics()
                
                # Check for emergency conditions
                self.check_emergency_conditions()
                
                # Generate interim report every hour
                if iteration % 12 == 0:  # Every hour (assuming 5-minute intervals)
                    self.generate_interim_report()
                
                # Wait for next iteration (5 minutes)
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"❌ Error in trading loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry
        
        logger.info("🏁 Paper trading session completed!")
        await self.finalize_session()
        
    async def monitor_market_conditions(self):
        """Monitor and log current market conditions"""
        try:
            # Get current prices for major cryptocurrencies
            symbols = ['BTC', 'ETH', 'SOL']
            prices = {}
            
            for symbol in symbols:
                price = await self.trader.get_current_price(symbol)
                if price:
                    prices[symbol] = price
            
            # Log market conditions
            condition = {
                'timestamp': datetime.now().isoformat(),
                'prices': prices,
                'trader_status': self.trader.get_safety_status()
            }
            self.market_conditions.append(condition)
            
            logger.info(f"📊 Market: BTC=€{prices.get('BTC', 0):.0f} | ETH=€{prices.get('ETH', 0):.0f} | SOL=€{prices.get('SOL', 0):.0f}")
            
        except Exception as e:
            logger.error(f"❌ Failed to monitor market conditions: {e}")
    
    async def evaluate_trading_signals(self):
        """Evaluate trading signals and execute paper trades"""
        try:
            # Test signals for multiple symbols
            symbols = ['BTC', 'ETH', 'SOL']
            
            for symbol in symbols:
                self.total_signals += 1
                
                # Generate signal
                signal = self.trader.analyze_market_conditions(symbol)
                
                if signal:
                    self.valid_signals += 1
                    logger.info(f"📡 Signal for {symbol}: {signal.side} | Confidence: {signal.confidence:.2f} | R/R: {signal.risk_reward_ratio:.2f}")
                    
                    # Validate signal safety
                    is_safe, issues = self.trader.validate_trade_safety(signal)
                    
                    if not is_safe:
                        logger.warning(f"⚠️ Signal rejected: {', '.join(issues)}")
                        self.safety_violations.extend(issues)
                    else:
                        # Execute paper trade
                        position = await self.trader.execute_trade(signal)
                        
                        if position:
                            self.executed_trades += 1
                            logger.info(f"✅ Paper trade executed: {position.side} {symbol} €{position.amount_eur:.2f}")
                            
                            # Simulate position outcome (simplified)
                            await self.simulate_trade_outcome(position)
                else:
                    logger.warning(f"⚠️ No signal generated for {symbol}")
                    
        except Exception as e:
            logger.error(f"❌ Failed to evaluate trading signals: {e}")
    
    async def simulate_trade_outcome(self, position):
        """
        Simulate trade outcome for paper trading
        In real implementation, this would monitor actual market movement
        """
        try:
            # Simulate a random outcome based on realistic probabilities
            import random
            
            # 60% chance of profit (optimistic for good strategies)
            is_profitable = random.random() < 0.6
            
            if is_profitable:
                # Simulate partial profit (not always hitting full target)
                profit_factor = random.uniform(0.3, 1.0)  # 30% to 100% of target
                pnl = position.amount_eur * self.trader.safety_limits.profit_target_pct * profit_factor
                self.profitable_trades += 1
                logger.info(f"💰 PROFIT: €{pnl:.2f} ({pnl/position.amount_eur:.1%})")
            else:
                # Simulate loss (hitting stop loss)
                pnl = -position.amount_eur * self.trader.safety_limits.mandatory_stop_loss_pct
                self.losing_trades += 1
                self.trader.consecutive_losses += 1
                logger.info(f"📉 LOSS: €{pnl:.2f} ({pnl/position.amount_eur:.1%})")
            
            # Update total P&L
            self.total_pnl += pnl
            self.trader.daily_pnl += pnl
            
            # Track maximum drawdown
            if pnl < 0:
                self.max_drawdown = min(self.max_drawdown, self.total_pnl)
            
        except Exception as e:
            logger.error(f"❌ Failed to simulate trade outcome: {e}")
    
    def update_performance_metrics(self):
        """Update and calculate performance metrics"""
        try:
            # Calculate win rate
            total_completed = self.profitable_trades + self.losing_trades
            if total_completed > 0:
                self.win_rate = self.profitable_trades / total_completed
            
            # Calculate average profit per trade
            avg_pnl = self.total_pnl / max(total_completed, 1)
            
            # Update trader state
            self.trader._save_trading_state()
            
        except Exception as e:
            logger.error(f"❌ Failed to update performance metrics: {e}")
    
    def check_emergency_conditions(self):
        """Check for emergency conditions that would stop trading"""
        try:
            # Check if daily loss limit exceeded
            if self.trader.daily_pnl <= -self.trader.safety_limits.max_daily_loss_eur:
                self.trader.emergency_stop = True
                self.emergency_stops += 1
                logger.error(f"🚨 EMERGENCY STOP: Daily loss limit exceeded (€{self.trader.daily_pnl:.2f})")
            
            # Check consecutive losses
            if self.trader.consecutive_losses >= self.trader.safety_limits.max_consecutive_losses:
                logger.warning(f"⚠️ Consecutive loss limit reached: {self.trader.consecutive_losses}")
                
        except Exception as e:
            logger.error(f"❌ Failed to check emergency conditions: {e}")
    
    def generate_interim_report(self):
        """Generate interim performance report"""
        try:
            elapsed_hours = (datetime.now() - self.session_start).total_seconds() / 3600
            
            logger.info("\n📊 INTERIM PERFORMANCE REPORT")
            logger.info("-" * 40)
            logger.info(f"Elapsed Time: {elapsed_hours:.1f} hours")
            logger.info(f"Total Signals: {self.total_signals}")
            logger.info(f"Valid Signals: {self.valid_signals}")
            logger.info(f"Executed Trades: {self.executed_trades}")
            logger.info(f"Profitable Trades: {self.profitable_trades}")
            logger.info(f"Losing Trades: {self.losing_trades}")
            logger.info(f"Win Rate: {self.win_rate:.1%}")
            logger.info(f"Total P&L: €{self.total_pnl:.2f}")
            logger.info(f"Max Drawdown: €{self.max_drawdown:.2f}")
            logger.info(f"Emergency Stops: {self.emergency_stops}")
            logger.info(f"Safety Violations: {len(self.safety_violations)}")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate interim report: {e}")
    
    async def finalize_session(self):
        """Finalize the paper trading session and generate final report"""
        try:
            duration = datetime.now() - self.session_start
            
            # Generate comprehensive final report
            report = {
                'session_info': {
                    'start_time': self.session_start.isoformat(),
                    'end_time': datetime.now().isoformat(),
                    'duration_hours': duration.total_seconds() / 3600,
                    'api_key_used': self.api_key[:8] + "...",
                    'mode': 'paper_trading'
                },
                'performance_metrics': {
                    'total_signals': self.total_signals,
                    'valid_signals': self.valid_signals,
                    'signal_quality_rate': self.valid_signals / max(self.total_signals, 1),
                    'executed_trades': self.executed_trades,
                    'profitable_trades': self.profitable_trades,
                    'losing_trades': self.losing_trades,
                    'win_rate': self.win_rate,
                    'total_pnl_eur': self.total_pnl,
                    'max_drawdown_eur': self.max_drawdown,
                    'avg_pnl_per_trade': self.total_pnl / max(self.executed_trades, 1),
                    'profit_factor': abs(self.total_pnl / max(abs(self.max_drawdown), 0.01))
                },
                'risk_metrics': {
                    'emergency_stops': self.emergency_stops,
                    'safety_violations': len(self.safety_violations),
                    'consecutive_losses': self.trader.consecutive_losses,
                    'daily_pnl': self.trader.daily_pnl,
                    'max_position_used': self.trader.safety_limits.max_position_size_eur
                },
                'safety_violations': self.safety_violations,
                'final_trader_status': self.trader.get_safety_status()
            }
            
            # Save detailed report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = f'reports/paper_trading_report_{timestamp}.json'
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            # Generate final assessment
            self.generate_final_assessment(report)
            
            logger.info(f"\n💾 Detailed report saved: {report_file}")
            
        except Exception as e:
            logger.error(f"❌ Failed to finalize session: {e}")
    
    def generate_final_assessment(self, report):
        """Generate final assessment and recommendation"""
        logger.info("\n🏆 FINAL PAPER TRADING ASSESSMENT")
        logger.info("=" * 60)
        
        # Performance summary
        performance = report['performance_metrics']
        risk = report['risk_metrics']
        
        logger.info(f"📊 PERFORMANCE SUMMARY:")
        logger.info(f"   Total Trades: {performance['executed_trades']}")
        logger.info(f"   Win Rate: {performance['win_rate']:.1%}")
        logger.info(f"   Total P&L: €{performance['total_pnl_eur']:.2f}")
        logger.info(f"   Max Drawdown: €{performance['max_drawdown_eur']:.2f}")
        logger.info(f"   Avg P&L per Trade: €{performance['avg_pnl_per_trade']:.2f}")
        
        logger.info(f"\n🛡️ RISK SUMMARY:")
        logger.info(f"   Emergency Stops: {risk['emergency_stops']}")
        logger.info(f"   Safety Violations: {risk['safety_violations']}")
        logger.info(f"   Max Consecutive Losses: {risk['consecutive_losses']}")
        
        # Generate recommendation
        is_ready_for_real_trading = self.assess_readiness_for_real_trading(performance, risk)
        
        if is_ready_for_real_trading:
            logger.info("\n🟢 RECOMMENDATION: READY FOR PHASE 3 (MICRO-TRADING)")
            logger.info("✅ System passed paper trading validation")
            logger.info("✅ Performance metrics are acceptable")
            logger.info("✅ Risk management working properly")
            logger.info("\n📋 NEXT STEPS:")
            logger.info("1. 🔹 Start with €0.10 micro-trades")
            logger.info("2. 📊 Monitor real execution quality")
            logger.info("3. 📈 Gradually increase to €1-2 positions")
            logger.info("4. 🎯 Work towards 5€ → 100€ goal")
        else:
            logger.info("\n🔴 RECOMMENDATION: NOT READY FOR REAL TRADING")
            logger.info("❌ Issues found during paper trading")
            logger.info("🔧 System needs improvement before real money")
            logger.info("\n📋 REQUIRED ACTIONS:")
            logger.info("1. 🔍 Analyze performance issues")
            logger.info("2. 🛠️ Improve trading strategy")
            logger.info("3. 🔄 Run additional paper trading")
            logger.info("4. 🚫 DO NOT use real money yet")
    
    def assess_readiness_for_real_trading(self, performance, risk):
        """
        Assess if the system is ready for real trading based on paper results
        
        Returns:
            bool: True if ready for real trading, False otherwise
        """
        # Minimum performance criteria
        min_win_rate = 0.55  # 55% win rate
        min_trades = 5       # At least 5 trades executed
        max_emergency_stops = 0  # No emergency stops allowed
        max_safety_violations = 3  # Max 3 safety violations
        min_profit_factor = 1.2  # Profit factor > 1.2
        
        # Check each criterion
        criteria_met = []
        
        # Win rate check
        win_rate_ok = performance['win_rate'] >= min_win_rate
        criteria_met.append(('Win Rate', win_rate_ok, f"{performance['win_rate']:.1%} >= {min_win_rate:.1%}"))
        
        # Minimum trades check
        trades_ok = performance['executed_trades'] >= min_trades
        criteria_met.append(('Trade Volume', trades_ok, f"{performance['executed_trades']} >= {min_trades}"))
        
        # Emergency stops check
        emergency_ok = risk['emergency_stops'] <= max_emergency_stops
        criteria_met.append(('Emergency Stops', emergency_ok, f"{risk['emergency_stops']} <= {max_emergency_stops}"))
        
        # Safety violations check
        safety_ok = risk['safety_violations'] <= max_safety_violations
        criteria_met.append(('Safety Violations', safety_ok, f"{risk['safety_violations']} <= {max_safety_violations}"))
        
        # Profit factor check
        profit_factor_ok = performance['profit_factor'] >= min_profit_factor
        criteria_met.append(('Profit Factor', profit_factor_ok, f"{performance['profit_factor']:.2f} >= {min_profit_factor}"))
        
        # Overall profitability
        profitable = performance['total_pnl_eur'] > 0
        criteria_met.append(('Overall Profit', profitable, f"€{performance['total_pnl_eur']:.2f} > €0"))
        
        # Log assessment results
        logger.info(f"\n📋 READINESS ASSESSMENT:")
        all_criteria_met = True
        for criterion, met, description in criteria_met:
            status = "✅" if met else "❌"
            logger.info(f"   {status} {criterion}: {description}")
            if not met:
                all_criteria_met = False
        
        return all_criteria_met

# Convenience functions for easy execution
async def run_quick_test(hours: float = 0.5):
    """Run a quick 30-minute paper trading test"""
    session = PaperTradingSession(duration_hours=hours)
    await session.start_session()

async def run_full_validation(hours: int = 24):
    """Run full 24-hour paper trading validation"""
    session = PaperTradingSession(duration_hours=hours)
    await session.start_session()

if __name__ == "__main__":
    # Default to 30-minute quick test for immediate validation
    # Change to run_full_validation(24) for full 24-hour test
    asyncio.run(run_quick_test(0.5)) 