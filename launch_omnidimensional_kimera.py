#!/usr/bin/env python3
"""
Kimera Omnidimensional Trading System Launcher
Integrated launch system for the Kimera SWM Omnidimensional Protocol Engine

This launcher now properly integrates with Kimera's backend architecture,
ethical governance, and hardware awareness systems.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime
from typing import Dict, Any

# Add project root to path for imports
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Kimera backend imports
from backend.engines.omnidimensional_protocol_engine import OmnidimensionalProtocolEngine
from backend.core.ethical_governor import EthicalGovernor
from backend.utils.kimera_logger import setup_kimera_logger
from backend.utils.config import get_api_settings

# Configure logging using Kimera's system
logger = setup_kimera_logger(__name__)

class KimeraOmnidimensionalLauncher:
    """
    Integrated launcher for Kimera's Omnidimensional Protocol Engine
    Fully compliant with Kimera's ethical governance and architecture
    """
    
    def __init__(self):
        self.ethical_governor = None
        self.omnidimensional_engine = None
        self.config = None
        
    async def initialize_systems(self) -> bool:
        """Initialize all required Kimera systems with ethical oversight"""
        try:
            logger.info("🚀 Initializing Kimera Omnidimensional Trading System...")
            
            # Load configuration
            self.config = get_api_settings()
            logger.info(f"✅ Configuration loaded (env: {self.config.environment})")
            
            # Initialize Ethical Governor
            logger.info("⚖️ Initializing Ethical Governor...")
            self.ethical_governor = EthicalGovernor(
                enable_enhanced_logging=True,
                enable_monitoring_integration=True
            )
            logger.info("✅ Ethical Governor initialized and operational")
            
            # Initialize Omnidimensional Engine with ethical oversight
            logger.info("🌟 Initializing Omnidimensional Protocol Engine...")
            self.omnidimensional_engine = OmnidimensionalProtocolEngine(self.ethical_governor)
            logger.info("✅ Omnidimensional Engine initialized with ethical compliance")
            
            # System health check
            logger.info("🔍 Performing system health check...")
            health_status = await self._perform_health_check()
            
            if health_status["status"] != "healthy":
                logger.error(f"❌ System health check failed: {health_status}")
                return False
            
            logger.info("✅ All systems initialized successfully")
            logger.info(f"   Hardware: {self.omnidimensional_engine.device_info['summary']}")
            logger.info(f"   Protocols: {len(self.omnidimensional_engine.registry.protocols)}")
            logger.info(f"   Ethical Oversight: ✅ Active")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ System initialization failed: {e}")
            return False
    
    async def _perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive system health check"""
        try:
            health_status = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "components": {}
            }
            
            # Check ethical governor
            if self.ethical_governor:
                health_status["components"]["ethical_governor"] = "operational"
            else:
                health_status["status"] = "unhealthy"
                health_status["components"]["ethical_governor"] = "failed"
            
            # Check omnidimensional engine
            if self.omnidimensional_engine:
                health_status["components"]["omnidimensional_engine"] = "operational"
                health_status["components"]["protocol_count"] = len(self.omnidimensional_engine.registry.protocols)
                health_status["components"]["hardware"] = self.omnidimensional_engine.device_info["summary"]
            else:
                health_status["status"] = "unhealthy"
                health_status["components"]["omnidimensional_engine"] = "failed"
            
            return health_status
            
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}
    
    async def run_demonstration_cycle(self) -> Dict[str, Any]:
        """Run a demonstration trading cycle with full ethical oversight"""
        try:
            if not self.omnidimensional_engine:
                raise RuntimeError("Omnidimensional engine not initialized")
            
            logger.info("🔄 Executing demonstration omnidimensional trading cycle...")
            
            # Execute trading cycle with ethical governance
            cycle_result = await self.omnidimensional_engine.execute_omnidimensional_cycle()
            
            # Log results using proper Kimera logging (NO PRINT STATEMENTS)
            logger.info("=" * 80)
            logger.info("🚀 KIMERA OMNIDIMENSIONAL DEMONSTRATION RESULTS")
            logger.info("=" * 80)
            logger.info(f"💰 Cycle Profit: ${cycle_result['total_profit']:.2f}")
            logger.info(f"⚡ Execution Time: {cycle_result['cycle_duration']:.2f}s")
            logger.info(f"📈 Trades Executed: {cycle_result['trades_executed']}")
            logger.info(f"💎 Portfolio Value: ${cycle_result['portfolio_value']:,.2f}")
            logger.info(f"🎯 Success Rate: {cycle_result['success_rate']:.1%}")
            logger.info(f"🔧 Hardware Used: {cycle_result['hardware_used']}")
            logger.info(f"⚖️ Ethical Compliance: {cycle_result['ethical_compliance']}")
            
            logger.info(f"📊 Strategies Executed: {len(cycle_result['strategies_executed'])}")
            for strategy in cycle_result['strategies_executed']:
                status = "✅" if strategy['status'] == 'success' else "❌"
                logger.info(f"  {status} {strategy['strategy']}: ${strategy['profit']:.2f} profit")
            
            return cycle_result
            
        except Exception as e:
            logger.error(f"❌ Demonstration cycle failed: {e}")
            raise
    
    async def run_continuous_trading(self, duration_hours: float = 24.0):
        """Run continuous trading with ethical oversight"""
        try:
            if not self.omnidimensional_engine:
                raise RuntimeError("Omnidimensional engine not initialized")
            
            logger.info(f"🚀 Starting continuous omnidimensional trading...")
            logger.info(f"   Duration: {duration_hours} hours")
            logger.info(f"   Ethical Oversight: ✅ Active")
            logger.info(f"   Hardware: {self.omnidimensional_engine.device_info['summary']}")
            
            # Start continuous trading with ethical governance
            await self.omnidimensional_engine.run_continuous_trading(duration_hours)
            
        except Exception as e:
            logger.error(f"❌ Continuous trading failed: {e}")
            raise
    
    async def analyze_market_sentiment(self, protocols: list = None) -> Dict[str, Any]:
        """Analyze market sentiment for specified protocols"""
        try:
            if not self.omnidimensional_engine:
                raise RuntimeError("Omnidimensional engine not initialized")
            
            if protocols is None:
                protocols = ["uniswap_v4", "curve_finance", "gmx_v2", "convex_finance"]
            
            logger.info(f"🔮 Analyzing sentiment for {len(protocols)} protocols...")
            
            sentiment_results = {}
            for protocol in protocols:
                try:
                    sentiment = await self.omnidimensional_engine.sentiment_analyzer.analyze_protocol_sentiment(protocol)
                    sentiment_results[protocol] = sentiment
                    logger.info(f"  • {protocol}: {sentiment['composite_score']:.3f} ({sentiment['recommendation']})")
                except Exception as e:
                    logger.warning(f"  ⚠️ {protocol}: Sentiment analysis failed - {e}")
                    sentiment_results[protocol] = {"error": str(e)}
            
            logger.info("✅ Sentiment analysis complete")
            return sentiment_results
            
        except Exception as e:
            logger.error(f"❌ Sentiment analysis failed: {e}")
            raise
    
    async def shutdown(self):
        """Gracefully shutdown all systems"""
        try:
            logger.info("🛑 Shutting down Kimera Omnidimensional Trading System...")
            
            # Shutdown components
            if hasattr(self.omnidimensional_engine, 'shutdown'):
                await self.omnidimensional_engine.shutdown()
            
            logger.info("✅ Shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ Shutdown error: {e}")

async def show_main_menu():
    """Display main menu and handle user selection"""
    logger.info("🌟 Kimera Omnidimensional Trading System")
    logger.info("=" * 50)
    logger.info("1. Run Demonstration Trading Cycle")
    logger.info("2. Start Continuous Trading (24 hours)")
    logger.info("3. Start Continuous Trading (Custom Duration)")
    logger.info("4. Analyze Market Sentiment")
    logger.info("5. System Health Check")
    logger.info("6. Exit")
    logger.info("=" * 50)

async def main():
    """Main launcher function with comprehensive error handling"""
    launcher = None
    
    try:
        # Initialize launcher
        launcher = KimeraOmnidimensionalLauncher()
        
        # Initialize all systems
        logger.info("🚀 Starting Kimera Omnidimensional Trading System...")
        
        if not await launcher.initialize_systems():
            logger.error("❌ System initialization failed")
            return 1
        
        # Main menu loop
        while True:
            await show_main_menu()
            
            try:
                choice = input("\nSelect option (1-6): ").strip()
            except KeyboardInterrupt:
                logger.info("\n🛑 User interrupted")
                choice = "6"
            
            try:
                if choice == "1":
                    # Demonstration cycle
                    result = await launcher.run_demonstration_cycle()
                    logger.info(f"🎯 Demonstration complete: ${result['total_profit']:.2f} profit")
                    
                elif choice == "2":
                    # 24-hour continuous trading
                    await launcher.run_continuous_trading(24.0)
                    
                elif choice == "3":
                    # Custom duration trading
                    try:
                        hours = float(input("Enter duration in hours: "))
                        if hours <= 0 or hours > 168:
                            logger.error("❌ Duration must be between 0 and 168 hours")
                            continue
                        await launcher.run_continuous_trading(hours)
                    except ValueError:
                        logger.error("❌ Invalid duration entered")
                        continue
                    
                elif choice == "4":
                    # Sentiment analysis
                    sentiment_results = await launcher.analyze_market_sentiment()
                    logger.info(f"📊 Analyzed sentiment for {len(sentiment_results)} protocols")
                    
                elif choice == "5":
                    # Health check
                    health = await launcher._perform_health_check()
                    logger.info(f"🔍 System Health: {health['status']}")
                    for component, status in health.get('components', {}).items():
                        logger.info(f"  • {component}: {status}")
                    
                elif choice == "6":
                    # Exit
                    logger.info("👋 Goodbye!")
                    break
                    
                else:
                    logger.warning("❌ Invalid selection. Please choose 1-6.")
                    
            except KeyboardInterrupt:
                logger.info("\n🛑 Operation interrupted by user")
                choice = input("Return to menu? (y/n): ").strip().lower()
                if choice != 'y':
                    break
            except Exception as e:
                logger.error(f"❌ Operation failed: {e}")
                input("Press Enter to continue...")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Critical error in main launcher: {e}")
        return 1
        
    finally:
        # Cleanup
        if launcher:
            try:
                await launcher.shutdown()
            except Exception as e:
                logger.error(f"❌ Shutdown error: {e}")

if __name__ == "__main__":
    try:
        # Run the main launcher
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        logger.info("\n🛑 System interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1) 