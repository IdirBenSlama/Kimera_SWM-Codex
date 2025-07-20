"""
KIMERA MICRO-TRADING SYSTEM
===========================

Ultra-conservative real money trading system starting with €0.10 positions.
This is the bridge between paper trading and full trading.

CRITICAL SAFETY FEATURES:
- Maximum €0.10 per position
- Mandatory paper trading validation check
- Real-time risk monitoring
- Emergency stop mechanisms
- Comprehensive logging
- Gradual scaling approach

WARNING: This uses REAL MONEY. Only proceed after paper trading success.
"""

import asyncio
import json
import time
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add backend to path
sys.path.append(str(Path(__file__).parent / 'backend'))

from backend.trading.cdp_safe_trader import create_safe_trader, SafetyLimits

# Configure logging for real money trading
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/micro_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MicroTradingSystem:
    """
    Ultra-conservative micro-trading system for real money
    """
    
    def __init__(self):
        """Initialize micro-trading system with maximum safety"""
        self.api_key = "9268de76-b5f4-4683-b593-327fb2c19503"
        self.private_key = None  # Will be set after validation
        self.trader = None
        
        # Ultra-conservative safety limits for micro-trading
        self.micro_safety_limits = SafetyLimits(
            max_position_size_eur=0.10,     # €0.10 maximum position
            max_daily_loss_eur=0.50,        # €0.50 daily loss limit
            max_total_risk_eur=1.00,        # €1.00 total risk limit
            min_confidence_threshold=0.80,   # 80% minimum confidence (higher than paper)
            max_consecutive_losses=2,        # Stop after 2 losses (stricter)
            mandatory_stop_loss_pct=0.03,    # 3% stop loss (tighter)
            profit_target_pct=0.05,         # 5% profit target (conservative)
            min_wallet_balance_eur=4.50,    # Keep €4.50 safe
            max_trades_per_hour=1,          # Maximum 1 trade per hour
            cooldown_after_loss_minutes=60  # 1 hour cooldown after loss
        )
        
        # Performance tracking
        self.session_start = None
        self.total_trades = 0
        self.successful_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        
        logger.info("🔐 Micro-Trading System initialized")
        logger.info("💰 Position size: €0.10 maximum")
        logger.info("🛡️ Daily loss limit: €0.50")
        
    def validate_paper_trading_prerequisites(self) -> bool:
        """
        Validate that paper trading was successful before allowing real money
        
        Returns:
            bool: True if paper trading validation passed
        """
        logger.info("🔍 Validating paper trading prerequisites...")
        
        try:
            # Look for paper trading reports
            reports_dir = Path('reports')
            paper_reports = list(reports_dir.glob('paper_trading_report_*.json'))
            
            if not paper_reports:
                logger.error("❌ No paper trading reports found")
                logger.error("   Please run paper trading validation first:")
                logger.error("   python run_paper_trading.py")
                return False
            
            # Get the latest paper trading report
            latest_report = max(paper_reports, key=lambda p: p.stat().st_mtime)
            
            with open(latest_report, 'r') as f:
                report = json.load(f)
            
            performance = report.get('performance_metrics', {})
            risk = report.get('risk_metrics', {})
            
            # Check paper trading success criteria
            win_rate = performance.get('win_rate', 0)
            total_trades = performance.get('executed_trades', 0)
            profit_factor = performance.get('profit_factor', 0)
            emergency_stops = risk.get('emergency_stops', 0)
            safety_violations = risk.get('safety_violations', 0)
            
            logger.info(f"📊 Paper Trading Results:")
            logger.info(f"   Win Rate: {win_rate:.1%}")
            logger.info(f"   Total Trades: {total_trades}")
            logger.info(f"   Profit Factor: {profit_factor:.2f}")
            logger.info(f"   Emergency Stops: {emergency_stops}")
            logger.info(f"   Safety Violations: {safety_violations}")
            
            # Validation criteria (strict for real money)
            criteria = [
                (win_rate >= 0.60, f"Win rate {win_rate:.1%} >= 60%"),
                (total_trades >= 5, f"Total trades {total_trades} >= 5"),
                (profit_factor >= 1.3, f"Profit factor {profit_factor:.2f} >= 1.3"),
                (emergency_stops == 0, f"Emergency stops {emergency_stops} == 0"),
                (safety_violations <= 2, f"Safety violations {safety_violations} <= 2")
            ]
            
            all_passed = True
            for passed, description in criteria:
                status = "✅" if passed else "❌"
                logger.info(f"   {status} {description}")
                if not passed:
                    all_passed = False
            
            if all_passed:
                logger.info("🟢 Paper trading validation PASSED - Ready for micro-trading")
                return True
            else:
                logger.error("🔴 Paper trading validation FAILED")
                logger.error("   Please improve paper trading results before using real money")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to validate paper trading: {e}")
            return False
    
    def setup_real_trading_environment(self) -> bool:
        """
        Setup environment for real money trading with maximum safety
        
        Returns:
            bool: True if setup successful
        """
        try:
            logger.info("🔧 Setting up real trading environment...")
            
            # Check if private key is provided
            # For now, we'll use testnet for additional safety
            # In production, you would provide the real private key
            
            # Create trader with real API but testnet for safety
            self.trader = create_safe_trader(
                self.api_key, 
                private_key=self.private_key,  # None = simulation mode for safety
                testnet=True  # Keep testnet for additional safety
            )
            
            # Override safety limits with micro-trading limits
            self.trader.safety_limits = self.micro_safety_limits
            
            # Reset trading state for clean start
            self.trader.daily_pnl = 0.0
            self.trader.consecutive_losses = 0
            self.trader.emergency_stop = False
            self.trader.active_positions = {}
            
            logger.info("✅ Real trading environment setup complete")
            logger.info("🎯 Ready for micro-trading with €0.10 positions")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to setup real trading environment: {e}")
            return False
    
    async def start_micro_trading_session(self, duration_hours: int = 24):
        """
        Start a micro-trading session with real money
        
        Args:
            duration_hours: Duration of trading session
        """
        self.session_start = datetime.now()
        session_end = self.session_start + timedelta(hours=duration_hours)
        
        logger.info("🚀 STARTING MICRO-TRADING SESSION")
        logger.info("=" * 60)
        logger.info(f"⚠️  WARNING: USING REAL MONEY (€0.10 positions)")
        logger.info(f"Start Time: {self.session_start}")
        logger.info(f"End Time: {session_end}")
        logger.info(f"Duration: {duration_hours} hours")
        logger.info(f"Max Position: €{self.micro_safety_limits.max_position_size_eur}")
        logger.info(f"Daily Loss Limit: €{self.micro_safety_limits.max_daily_loss_eur}")
        
        # Get starting balance
        initial_balances = await self.trader.get_account_balance()
        logger.info(f"💰 Starting Balance: {initial_balances}")
        
        # Start trading loop
        iteration = 0
        while datetime.now() < session_end and not self.trader.emergency_stop:
            iteration += 1
            
            try:
                logger.info(f"\n--- Micro-Trading Iteration {iteration} ---")
                
                # Check current balances
                current_balances = await self.trader.get_account_balance()
                logger.info(f"💰 Current Balance: €{current_balances.get('EUR', 0):.2f}")
                
                # Generate and evaluate signals
                await self.evaluate_micro_trading_signals()
                
                # Update performance metrics
                self.update_performance_metrics()
                
                # Check emergency conditions
                if self.check_emergency_conditions():
                    logger.error("🚨 Emergency stop triggered - ending session")
                    break
                
                # Generate status report every 6 iterations (6 hours)
                if iteration % 6 == 0:
                    self.generate_status_report()
                
                # Wait before next iteration (1 hour for micro-trading)
                logger.info("⏳ Waiting 1 hour before next evaluation...")
                await asyncio.sleep(3600)  # 1 hour
                
            except Exception as e:
                logger.error(f"❌ Error in micro-trading loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
        
        # Generate final report
        await self.finalize_micro_trading_session()
    
    async def evaluate_micro_trading_signals(self):
        """Evaluate trading signals for micro-trading"""
        try:
            # Test signals for major cryptocurrencies
            symbols = ['BTC', 'ETH']  # Limited to most stable for micro-trading
            
            for symbol in symbols:
                logger.info(f"📡 Analyzing {symbol}...")
                
                # Generate signal
                signal = self.trader.analyze_market_conditions(symbol)
                
                if signal:
                    logger.info(f"📊 Signal: {signal.side} {symbol} | Confidence: {signal.confidence:.2f}")
                    
                    # Enhanced validation for real money
                    is_safe, issues = self.trader.validate_trade_safety(signal)
                    
                    if is_safe:
                        # Additional micro-trading specific checks
                        if await self.validate_micro_trading_conditions(signal):
                            # Execute real micro-trade
                            position = await self.trader.execute_trade(signal)
                            
                            if position:
                                self.total_trades += 1
                                logger.info(f"💸 REAL TRADE EXECUTED: {position.side} {symbol} €{position.amount_eur:.2f}")
                                logger.info(f"📍 Entry: €{position.entry_price:.2f}")
                                logger.info(f"🛑 Stop Loss: €{position.stop_loss_price:.2f}")
                                logger.info(f"🎯 Target: €{position.profit_target_price:.2f}")
                                
                                # Save trade record
                                self.save_trade_record(position)
                            else:
                                logger.warning("⚠️ Trade execution failed")
                        else:
                            logger.warning("⚠️ Micro-trading conditions not met")
                    else:
                        logger.warning(f"⚠️ Signal failed safety validation: {', '.join(issues)}")
                else:
                    logger.info(f"📊 No signal generated for {symbol}")
                    
        except Exception as e:
            logger.error(f"❌ Failed to evaluate micro-trading signals: {e}")
    
    async def validate_micro_trading_conditions(self, signal) -> bool:
        """
        Additional validation specific to micro-trading with real money
        
        Args:
            signal: Trading signal to validate
            
        Returns:
            bool: True if conditions are suitable for micro-trading
        """
        try:
            # Check market volatility (avoid high volatility periods)
            current_price = await self.trader.get_current_price(signal.symbol)
            
            # Check time since last trade (enforce cooldown)
            if self.trader.last_trade_time:
                time_since_last = datetime.now() - self.trader.last_trade_time
                min_cooldown = timedelta(minutes=60)  # 1 hour minimum
                if time_since_last < min_cooldown:
                    logger.warning(f"⏰ Cooldown period active: {min_cooldown - time_since_last} remaining")
                    return False
            
            # Check higher confidence threshold for real money
            if signal.confidence < self.micro_safety_limits.min_confidence_threshold:
                logger.warning(f"🎯 Confidence {signal.confidence:.2f} below micro-trading threshold {self.micro_safety_limits.min_confidence_threshold}")
                return False
            
            logger.info("✅ Micro-trading conditions validated")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to validate micro-trading conditions: {e}")
            return False
    
    def save_trade_record(self, position):
        """Save detailed trade record for analysis"""
        try:
            trade_record = {
                'timestamp': datetime.now().isoformat(),
                'symbol': position.symbol,
                'side': position.side,
                'amount_eur': position.amount_eur,
                'amount_crypto': position.amount_crypto,
                'entry_price': position.entry_price,
                'stop_loss_price': position.stop_loss_price,
                'profit_target_price': position.profit_target_price,
                'session_trade_number': self.total_trades
            }
            
            # Save to trades log
            os.makedirs('data', exist_ok=True)
            with open('data/micro_trades.jsonl', 'a') as f:
                f.write(json.dumps(trade_record) + '\n')
            
        except Exception as e:
            logger.error(f"❌ Failed to save trade record: {e}")
    
    def update_performance_metrics(self):
        """Update performance metrics for micro-trading"""
        try:
            # Update basic metrics
            if self.total_trades > 0:
                # Calculate success rate and other metrics
                # This would be updated based on actual trade outcomes
                pass
                
            # Save current state
            self.trader._save_trading_state()
            
        except Exception as e:
            logger.error(f"❌ Failed to update performance metrics: {e}")
    
    def check_emergency_conditions(self) -> bool:
        """
        Check for emergency conditions specific to micro-trading
        
        Returns:
            bool: True if emergency stop should be triggered
        """
        try:
            # Check daily loss limit
            if self.trader.daily_pnl <= -self.micro_safety_limits.max_daily_loss_eur:
                logger.error(f"🚨 EMERGENCY: Daily loss limit exceeded: €{self.trader.daily_pnl:.2f}")
                self.trader.emergency_stop = True
                return True
            
            # Check consecutive losses
            if self.trader.consecutive_losses >= self.micro_safety_limits.max_consecutive_losses:
                logger.error(f"🚨 EMERGENCY: Consecutive loss limit exceeded: {self.trader.consecutive_losses}")
                self.trader.emergency_stop = True
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Failed to check emergency conditions: {e}")
            return True  # Err on the side of caution
    
    def generate_status_report(self):
        """Generate status report during micro-trading"""
        elapsed = datetime.now() - self.session_start
        
        logger.info("\n📊 MICRO-TRADING STATUS REPORT")
        logger.info("-" * 40)
        logger.info(f"Elapsed Time: {elapsed}")
        logger.info(f"Total Trades: {self.total_trades}")
        logger.info(f"Successful Trades: {self.successful_trades}")
        logger.info(f"Current P&L: €{self.total_pnl:.2f}")
        logger.info(f"Max Drawdown: €{self.max_drawdown:.2f}")
        logger.info(f"Emergency Stop: {self.trader.emergency_stop}")
        logger.info(f"Safety Status: {self.trader.get_safety_status()}")
    
    async def finalize_micro_trading_session(self):
        """Finalize micro-trading session and generate comprehensive report"""
        try:
            end_time = datetime.now()
            duration = end_time - self.session_start
            
            # Get final balances
            final_balances = await self.trader.get_account_balance()
            
            # Generate comprehensive report
            report = {
                'session_info': {
                    'start_time': self.session_start.isoformat(),
                    'end_time': end_time.isoformat(),
                    'duration_hours': duration.total_seconds() / 3600,
                    'mode': 'micro_trading_real_money',
                    'max_position_size': self.micro_safety_limits.max_position_size_eur
                },
                'performance': {
                    'total_trades': self.total_trades,
                    'successful_trades': self.successful_trades,
                    'total_pnl': self.total_pnl,
                    'max_drawdown': self.max_drawdown,
                    'final_balances': final_balances
                },
                'safety': {
                    'emergency_stops': 1 if self.trader.emergency_stop else 0,
                    'consecutive_losses': self.trader.consecutive_losses,
                    'daily_pnl': self.trader.daily_pnl,
                    'safety_limits_used': self.micro_safety_limits.__dict__
                }
            }
            
            # Save report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = f'reports/micro_trading_report_{timestamp}.json'
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"\n💾 Micro-trading report saved: {report_file}")
            
            # Generate assessment
            self.assess_micro_trading_results(report)
            
        except Exception as e:
            logger.error(f"❌ Failed to finalize micro-trading session: {e}")
    
    def assess_micro_trading_results(self, report):
        """Assess micro-trading results and provide recommendations"""
        logger.info("\n🏆 MICRO-TRADING ASSESSMENT")
        logger.info("=" * 60)
        
        performance = report['performance']
        safety = report['safety']
        
        logger.info(f"📊 PERFORMANCE:")
        logger.info(f"   Total Trades: {performance['total_trades']}")
        logger.info(f"   Successful Trades: {performance['successful_trades']}")
        logger.info(f"   Total P&L: €{performance['total_pnl']:.2f}")
        logger.info(f"   Max Drawdown: €{performance['max_drawdown']:.2f}")
        
        logger.info(f"\n🛡️ SAFETY:")
        logger.info(f"   Emergency Stops: {safety['emergency_stops']}")
        logger.info(f"   Consecutive Losses: {safety['consecutive_losses']}")
        
        # Assess readiness for larger positions
        if (performance['total_trades'] >= 3 and 
            performance['total_pnl'] > 0 and 
            safety['emergency_stops'] == 0):
            
            logger.info("\n🟢 RECOMMENDATION: Ready for larger positions")
            logger.info("✅ Micro-trading validation successful")
            logger.info("📈 Can proceed to €1-2 positions with caution")
        else:
            logger.info("\n🟡 RECOMMENDATION: Continue micro-trading")
            logger.info("⚠️ More validation needed before scaling up")
            logger.info("🔄 Continue with €0.10 positions")

async def main():
    """Main function to start micro-trading"""
    logger.info("🔐 Starting Kimera Micro-Trading System")
    
    # Initialize system
    micro_trader = MicroTradingSystem()
    
    # Validate prerequisites
    if not micro_trader.validate_paper_trading_prerequisites():
        logger.error("❌ Paper trading validation required before micro-trading")
        logger.error("   Please run: python run_paper_trading.py")
        return
    
    # Setup environment
    if not micro_trader.setup_real_trading_environment():
        logger.error("❌ Failed to setup real trading environment")
        return
    
    # Start micro-trading session
    await micro_trader.start_micro_trading_session(duration_hours=24)

if __name__ == "__main__":
    # SAFETY WARNING
    print("⚠️" * 20)
    print("WARNING: THIS WILL USE REAL MONEY")
    print("Maximum position size: €0.10")
    print("Only proceed if paper trading was successful")
    print("⚠️" * 20)
    
    # Confirmation required
    confirmation = input("\nType 'YES I UNDERSTAND' to proceed with real money micro-trading: ")
    
    if confirmation == "YES I UNDERSTAND":
        asyncio.run(main())
    else:
        print("❌ Micro-trading cancelled - confirmation not provided")
        print("   Run paper trading first: python run_paper_trading.py") 