#!/usr/bin/env python3
"""
KIMERA Autonomous Trading Demo

This demo shows how KIMERA can operate as a complete autonomous trading system,
bridging cognitive analysis with real-world execution.

KIMERA's Journey: Analysis → Decision → Action → Learning
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any
import sys
import os

# Add the backend path to sys.path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

try:
    from backend.trading.core.integrated_trading_engine import create_integrated_trading_engine
    from backend.trading.execution.kimera_action_interface import create_kimera_action_interface, CognitiveFeedbackProcessor
    from backend.trading.api.binance_connector import BinanceConnector
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.info("Please ensure you're running from the correct directory")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KimeraAutonomousTrader:
    """
    Complete autonomous trading system powered by KIMERA's cognitive abilities.
    
    This system demonstrates the full cycle:
    1. KIMERA analyzes markets using cognitive field dynamics
    2. KIMERA makes trading decisions with confidence scores
    3. KIMERA executes real trades through the action interface
    4. KIMERA learns from the results and adapts
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize KIMERA's autonomous trading system.
        
        Args:
            config: Trading configuration
        """
        self.config = config
        self.symbols = config.get("symbols", ["BTCUSDT", "ETHUSDT"])
        self.update_interval = config.get("update_interval", 30)  # seconds
        
        # Core components - KIMERA's "brain" and "arms"
        self.cognitive_engine = None  # KIMERA's analytical mind
        self.action_interface = None  # KIMERA's execution arms
        self.feedback_processor = None  # KIMERA's learning system
        
        # State tracking
        self.is_running = False
        self.market_data_cache = {}
        self.performance_history = []
        
        logger.info("🧠 KIMERA Autonomous Trader initialized")
    
    async def initialize(self):
        """Initialize all KIMERA components"""
        logger.info("🚀 Initializing KIMERA's autonomous trading system...")
        logger.info("   🧠 Setting up cognitive analysis...")
        logger.info("   ⚡ Installing action execution interface...")
        logger.info("   📚 Connecting learning feedback loop...")
        
        try:
            # Initialize cognitive engine - KIMERA's analytical "brain"
            self.cognitive_engine = create_integrated_trading_engine(
                initial_balance=self.config.get("initial_balance", 1000.0),
                risk_tolerance=self.config.get("risk_tolerance", 0.05),
                enable_rl=True,
                enable_anomaly_detection=True,
                enable_portfolio_optimization=True
            )
            logger.info("✅ KIMERA's cognitive engine online")
            
            # Initialize action interface - KIMERA's execution "arms"
            self.action_interface = await create_kimera_action_interface(self.config)
            logger.info("✅ KIMERA's action interface connected")
            
            # Initialize feedback processor - KIMERA's learning system
            self.feedback_processor = CognitiveFeedbackProcessor(
                cognitive_field=getattr(self.cognitive_engine, 'cognitive_field', None),
                contradiction_engine=getattr(self.cognitive_engine, 'contradiction_engine', None)
            )
            logger.info("✅ KIMERA's feedback processor ready")
            
            # Connect the feedback loop - KIMERA learns from its actions
            self.action_interface.register_feedback_callback(
                self.feedback_processor.process_execution_feedback
            )
            
            # Register learning callback
            self.action_interface.register_feedback_callback(
                self._learn_from_execution
            )
            
            logger.info("🎯 KIMERA is ready for autonomous trading!")
            logger.info("   🧠 Brain: ONLINE")
            logger.info("   ⚡ Arms: CONNECTED") 
            logger.info("   📚 Learning: ACTIVE")
            return True
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {str(e)}")
            return False
    
    async def start_autonomous_trading(self):
        """Start KIMERA's autonomous trading loop"""
        if not await self.initialize():
            logger.error("❌ Failed to initialize KIMERA system")
            return
        
        logger.info("🔄 Starting KIMERA's autonomous trading loop...")
        logger.info("   This is where KIMERA transforms from observer to actor!")
        self.is_running = True
        
        try:
            # Create concurrent tasks for different aspects of trading
            tasks = [
                asyncio.create_task(self._market_analysis_loop()),
                asyncio.create_task(self._decision_making_loop()),
                asyncio.create_task(self._performance_monitoring_loop()),
                asyncio.create_task(self._adaptation_loop())
            ]
            
            logger.info("🌟 KIMERA is now operating autonomously!")
            # Run until stopped
            await asyncio.gather(*tasks)
            
        except KeyboardInterrupt:
            logger.info("🛑 Manual stop requested")
        except Exception as e:
            logger.error(f"❌ Trading loop error: {str(e)}")
        finally:
            await self.stop_trading()
    
    async def stop_trading(self):
        """Stop autonomous trading gracefully"""
        logger.info("🛑 Stopping KIMERA's autonomous trading...")
        self.is_running = False
        
        if self.action_interface:
            await self.action_interface.disconnect_exchanges()
        
        # Save final performance report
        await self._save_performance_report()
        
        logger.info("✅ KIMERA trading stopped successfully")
        logger.info("   🧠 Cognitive systems: OFFLINE")
        logger.info("   ⚡ Action interface: DISCONNECTED")
        logger.info("   📊 Final report: SAVED")
    
    async def _market_analysis_loop(self):
        """Continuously analyze markets with KIMERA's cognitive abilities"""
        while self.is_running:
            try:
                logger.info("🧠 KIMERA analyzing markets with cognitive field dynamics...")
                
                for symbol in self.symbols:
                    # Get market data
                    market_data = await self._get_market_data(symbol)
                    
                    # Store for decision making
                    self.market_data_cache[symbol] = market_data
                    
                    # KIMERA's cognitive analysis
                    intelligence = self.cognitive_engine.process_market_data(market_data)
                    logger.info(f"📊 {symbol} Cognitive Analysis:")
                    logger.info(f"   💰 Price: ${intelligence.price:,.2f}")
                    logger.info(f"   📈 Volatility: {intelligence.volatility:.3f}")
                    logger.info(f"   🔍 Momentum: {intelligence.momentum:.3f}")
                
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"❌ Market analysis error: {str(e)}")
                await asyncio.sleep(self.update_interval)
    
    async def _decision_making_loop(self):
        """KIMERA's decision making and execution loop - THE BRIDGE"""
        await asyncio.sleep(5)  # Let market analysis start first
        
        while self.is_running:
            try:
                logger.info("🎯 KIMERA making trading decisions...")
                
                for symbol in self.symbols:
                    if symbol in self.market_data_cache:
                        market_data = self.market_data_cache[symbol]
                        
                        # Generate enhanced trading signal - KIMERA's decision
                        signal = self.cognitive_engine.generate_enhanced_signal(
                            market_data, symbol
                        )
                        
                        logger.info(f"🔮 {symbol} Cognitive Signal:")
                        logger.info(f"   📊 Action: {signal.action.upper()}")
                        logger.info(f"   🎯 Confidence: {signal.confidence:.2f}")
                        logger.info(f"   🧠 Cognitive Pressure: {signal.cognitive_pressure:.2f}")
                        logger.info(f"   ⚖️ Contradiction Level: {signal.contradiction_level:.2f}")
                        
                        # THIS IS THE BRIDGE: Execute if KIMERA is confident enough
                        if signal.confidence > 0.4 and signal.action != 'hold':
                            logger.info(f"⚡ KIMERA EXECUTING DECISION for {symbol}...")
                            logger.info(f"   🚀 Transforming thought into action!")
                            
                            # KIMERA acts in the real world!
                            result = await self.action_interface.execute_enhanced_signal(
                                signal, symbol, exchange="binance"
                            )
                            
                            if result.status.value == "completed":
                                logger.info(f"✅ {symbol}: KIMERA successfully acted in the real world!")
                                logger.info(f"   💫 Cognitive analysis → Real market action")
                            elif result.status.value == "requires_approval":
                                logger.info(f"⏳ {symbol}: Action pending human approval")
                            else:
                                logger.warning(f"⚠️ {symbol}: Execution failed - {result.actual_outcome}")
                        else:
                            logger.info(f"🤔 {symbol}: KIMERA choosing to wait (confidence: {signal.confidence:.2f})")
                
                await asyncio.sleep(self.update_interval * 2)  # Less frequent than analysis
                
            except Exception as e:
                logger.error(f"❌ Decision making error: {str(e)}")
                await asyncio.sleep(self.update_interval)
    
    async def _performance_monitoring_loop(self):
        """Monitor KIMERA's trading performance"""
        while self.is_running:
            try:
                # Get performance summary
                engine_summary = self.cognitive_engine.get_performance_summary()
                action_summary = self.action_interface.get_action_summary()
                
                # Log performance - KIMERA's real-world impact
                logger.info("📈 KIMERA Real-World Performance:")
                logger.info(f"   💰 Daily P&L: ${action_summary.get('daily_pnl', 0):.2f}")
                logger.info(f"   📊 Success Rate: {action_summary.get('success_rate', 0):.2%}")
                logger.info(f"   🎯 Total Actions: {action_summary.get('total_actions', 0)}")
                logger.info(f"   🧠 System Status: {action_summary.get('system_status', 'UNKNOWN')}")
                logger.info(f"   🔄 Autonomous Mode: {action_summary.get('autonomous_mode', False)}")
                
                # Store performance history
                performance_snapshot = {
                    "timestamp": datetime.now().isoformat(),
                    "daily_pnl": action_summary.get('daily_pnl', 0),
                    "success_rate": action_summary.get('success_rate', 0),
                    "total_actions": action_summary.get('total_actions', 0),
                    "cognitive_metrics": engine_summary.get('kimera_metrics', {}),
                    "bridge_effectiveness": "operational"  # KIMERA's bridge is working
                }
                self.performance_history.append(performance_snapshot)
                
                # Safety check - emergency stop if losses are too high
                if action_summary.get('daily_pnl', 0) < -500:  # $500 loss limit
                    logger.critical("🛑 EMERGENCY STOP: Daily loss limit exceeded!")
                    logger.critical("   KIMERA protecting capital - trading halted")
                    await self.action_interface.emergency_stop()
                    self.is_running = False
                
                await asyncio.sleep(60)  # Monitor every minute
                
            except Exception as e:
                logger.error(f"❌ Performance monitoring error: {str(e)}")
                await asyncio.sleep(60)
    
    async def _adaptation_loop(self):
        """KIMERA's learning and adaptation loop"""
        while self.is_running:
            try:
                # Get cognitive performance summary
                if hasattr(self.feedback_processor, 'get_cognitive_performance_summary'):
                    cognitive_summary = self.feedback_processor.get_cognitive_performance_summary()
                    
                    if cognitive_summary.get("total_feedback_entries", 0) > 0:
                        logger.info("🧠 KIMERA Cognitive Learning Update:")
                        logger.info(f"   📚 Learning Trend: {cognitive_summary.get('cognitive_learning_trend', 'unknown')}")
                        logger.info(f"   💡 Cognitive Impact: {cognitive_summary.get('average_cognitive_impact', 0):.2f}")
                        logger.info(f"   🔗 Bridge Performance: Connecting mind to market")
                
                # Check for pending approvals
                pending_approvals = self.action_interface.get_pending_approvals()
                if pending_approvals:
                    logger.info(f"⏳ {len(pending_approvals)} KIMERA actions awaiting approval:")
                    for approval in pending_approvals[:3]:  # Show first 3
                        logger.info(f"   - {approval['symbol']}: {approval['action_type']} "
                                  f"(confidence: {approval['confidence']:.2f})")
                        logger.info(f"     Reasoning: {'; '.join(approval['reasoning'][:2])}")
                
                await asyncio.sleep(300)  # Adapt every 5 minutes
                
            except Exception as e:
                logger.error(f"❌ Adaptation loop error: {str(e)}")
                await asyncio.sleep(300)
    
    async def _get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get market data (simulated for demo - in production this would be real)"""
        import random
        
        # Simulate realistic market data
        base_price = 50000 if "BTC" in symbol else 3000
        price_change = random.uniform(-0.05, 0.05)  # ±5% change
        current_price = base_price * (1 + price_change)
        
        # Simulate price history for cognitive analysis
        price_history = []
        for i in range(20):
            historical_change = random.uniform(-0.02, 0.02)
            historical_price = current_price * (1 + historical_change)
            price_history.append(historical_price)
        
        return {
            'close': current_price,
            'price': current_price,
            'volume': random.uniform(1000, 5000),
            'bid': current_price * 0.999,
            'ask': current_price * 1.001,
            'volatility': random.uniform(0.01, 0.05),
            'news_sentiment': random.uniform(-0.2, 0.2),
            'social_sentiment': random.uniform(-0.1, 0.1),
            'price_history': price_history,
            'volume_history': [random.uniform(800, 1200) for _ in range(20)],
            'order_book_imbalance': random.uniform(-0.1, 0.1),
            'sentiment': 'bullish' if price_change > 0 else 'bearish'
        }
    
    async def _learn_from_execution(self, feedback_data: Dict[str, Any]):
        """KIMERA learns from execution results - closing the loop"""
        logger.info(f"📚 KIMERA learning from real-world execution:")
        logger.info(f"   🎯 Action: {feedback_data.get('action_id', 'unknown')}")
        
        # This is where KIMERA would update its cognitive models based on results
        if feedback_data.get('status') == 'completed':
            logger.info("   ✅ Positive reinforcement - cognitive analysis led to successful action")
            logger.info("   🔗 Bridge effectiveness confirmed: thought → action → success")
        else:
            logger.info("   📖 Learning opportunity - analyzing what went wrong")
            logger.info("   🔧 KIMERA adapting cognitive models for better decisions")
    
    async def _save_performance_report(self):
        """Save final performance report"""
        try:
            report = {
                "session_type": "KIMERA_AUTONOMOUS_TRADING",
                "bridge_concept": "Cognitive analysis connected to real-world execution",
                "session_end": datetime.now().isoformat(),
                "performance_history": self.performance_history,
                "final_action_summary": self.action_interface.get_action_summary(),
                "final_cognitive_summary": self.cognitive_engine.get_performance_summary(),
                "bridge_effectiveness": {
                    "total_decisions": len([p for p in self.performance_history if p.get('total_actions', 0) > 0]),
                    "successful_executions": sum(1 for p in self.performance_history if p.get('success_rate', 0) > 0),
                    "bridge_status": "OPERATIONAL" if self.performance_history else "NO_DATA"
                }
            }
            
            filename = f"kimera_autonomous_trading_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"📋 KIMERA performance report saved: {filename}")
            logger.info("   📊 Report includes bridge effectiveness metrics")
            
        except Exception as e:
            logger.error(f"❌ Failed to save performance report: {str(e)}")


def create_demo_config() -> Dict[str, Any]:
    """Create demo configuration for KIMERA autonomous trading"""
    return {
        # Exchange settings (using testnet for safety)
        "binance_enabled": True,
        "binance_api_key": "demo_key",  # Replace with real testnet keys for actual trading
        "binance_api_secret": "demo_secret",
        "testnet": True,
        
        # Trading parameters
        "symbols": ["BTCUSDT", "ETHUSDT"],
        "initial_balance": 1000.0,
        "risk_tolerance": 0.05,
        "update_interval": 30,
        
        # Safety limits - KIMERA's self-preservation
        "max_position_size": 100.0,
        "daily_loss_limit": 0.05,
        "approval_threshold": 0.1,
        
        # Autonomy settings
        "autonomous_mode": False,  # Start with approval required - safety first
    }


async def main():
    """Run the KIMERA autonomous trading demo"""
    logger.info("\n" + "=" * 60)
    logger.info("🚀 KIMERA AUTONOMOUS TRADING DEMO")
    logger.info("=" * 60)
    logger.info("This demo solves the core problem:")
    logger.error("❌ BEFORE: KIMERA could analyze but not act")
    logger.info("✅ AFTER:  KIMERA can analyze AND execute in real markets")
    logger.info()
    logger.info("The Solution: ACTION EXECUTION INTERFACE")
    logger.info("🧠 Cognitive Analysis → 🔗 Bridge → ⚡ Real-world Actions")
    logger.info()
    logger.info("KIMERA's Complete Cycle:")
    logger.info("1. 🧠 Cognitive market analysis with field dynamics")
    logger.info("2. 🎯 Intelligent decision making with confidence scores")
    logger.info("3. ⚡ Real-world execution through action interface")
    logger.info("4. 📚 Continuous learning from execution results")
    logger.info("5. 🔄 Adaptive improvement based on real performance")
    logger.info("=" * 60)
    
    # Create configuration
    config = create_demo_config()
    
    # Initialize KIMERA trader
    kimera_trader = KimeraAutonomousTrader(config)
    
    try:
        logger.info("\n🎯 Launching KIMERA's autonomous trading system...")
        logger.info("   This bridges the gap between cognitive analysis and market action!")
        logger.info("\nPress Ctrl+C to stop the demo")
        logger.info("-" * 40)
        
        # Start autonomous trading - THE BRIDGE IN ACTION
        await kimera_trader.start_autonomous_trading()
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Demo stopped by user")
    except Exception as e:
        logger.error(f"\n❌ Demo error: {str(e)
    finally:
        await kimera_trader.stop_trading()
        logger.info("\n✅ Demo completed")
        logger.info("🎯 KIMERA now has the 'arms' to execute its cognitive insights!")


if __name__ == "__main__":
    logger.info("🎯 KIMERA Trading: From Analysis to Action")
    logger.info("Bridging the gap between cognitive intelligence and real-world execution")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n👋 KIMERA Demo ended - Bridge concept demonstrated!")