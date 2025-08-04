import asyncio
from kimera_trading.core.engine import CognitiveThermodynamicTradingEngine
import signal
import logging
logger = logging.getLogger(__name__)

async def main():
    engine = CognitiveThermodynamicTradingEngine()
    await engine.initialize()

    loop = asyncio.get_running_loop()
    for signame in ('SIGINT', 'SIGTERM'):
        loop.add_signal_handler(
            getattr(signal, signame),
            lambda: asyncio.create_task(engine.stop())
        )

    await engine.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("KIMERA Trading System shutting down.")
