#!/usr/bin/env python3
"""
KIMERA AUTONOMOUS TRADER - SIMPLE RUNNER
========================================

Simplified autonomous trader execution
"""

import os
import sys
import asyncio

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from src.trading.autonomous_kimera_trader import create_autonomous_kimera

async def run_autonomous():
    """Run autonomous trader"""
    logger.info("🧠 STARTING KIMERA AUTONOMOUS TRADER")
    logger.info("   Target: EUR 5 → EUR 100")
    logger.info("   Mode: FULLY AUTONOMOUS")
    logger.info("   Safety Limits: NONE")
    logger.info()
    
    # Create trader
    API_KEY = os.getenv("CDP_API_KEY_NAME", "")
    trader = create_autonomous_kimera(API_KEY, target_eur=100.0)
    
    # Show status
    status = await trader.get_portfolio_status()
    logger.info(f"Portfolio Value: EUR {status['portfolio_value_eur']:.2f}")
    logger.info(f"Target: EUR {status['target_eur']}")
    logger.info(f"Progress: {status['progress_pct']:.1f}%")
    logger.info()
    
    logger.info("🚀 LAUNCHING AUTONOMOUS TRADING...")
    
    # Start autonomous trading
    await trader.run_autonomous_trader(cycle_interval_minutes=15)

if __name__ == "__main__":
    try:
        asyncio.run(run_autonomous())
    except KeyboardInterrupt:
        logger.info("\n🛑 Autonomous trading stopped")
    except Exception as e:
        logger.info(f"\n❌ Error: {e}")
        import traceback
import logging
logger = logging.getLogger(__name__)
        traceback.print_exc() 