"""
Async Shutdown Handler for KIMERA SWM
====================================

Provides utilities for handling async shutdown operations properly.
"""

import asyncio
import logging
from typing import Any, Callable, List

logger = logging.getLogger(__name__)
class AsyncShutdownHandler:
    """Auto-generated class."""
    pass
    """Handles async shutdown operations safely"""

    def __init__(self):
        self.shutdown_tasks: List[Callable] = []
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    def register_async_shutdown(self, component: Any) -> None:
        """Register a component for async shutdown"""
        if hasattr(component, "shutdown") and asyncio.iscoroutinefunction(
            component.shutdown
        ):
            self.shutdown_tasks.append(component.shutdown)
        elif hasattr(component, "shutdown"):
            # Wrap sync shutdown in async
            async def sync_wrapper():
                component.shutdown()

            self.shutdown_tasks.append(sync_wrapper)

    async def shutdown_all_async(self) -> None:
        """Shutdown all registered components asynchronously"""
        if not self.shutdown_tasks:
            return

        logger.info(f"Shutting down {len(self.shutdown_tasks)} async components...")

        # Run all shutdowns concurrently
        try:
            await asyncio.gather(
                *[task() for task in self.shutdown_tasks], return_exceptions=True
            )
        except Exception as e:
            logger.error(f"Error during async shutdown: {e}")

        logger.info("Async shutdown complete")

    def shutdown_all_sync(self) -> None:
        """Shutdown all components from synchronous context"""
        if not self.shutdown_tasks:
            return

        logger.info(
            f"Shutting down {len(self.shutdown_tasks)} components synchronously..."
        )

        # Create or use existing event loop
        try:
            loop = asyncio.get_running_loop()
            # If we're in a running loop, schedule the shutdown
            loop.create_task(self.shutdown_all_async())
        except RuntimeError:
            # No running loop, create a new one
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.shutdown_all_async())
            finally:
                loop.close()

        logger.info("Synchronous shutdown complete")


def safe_async_shutdown(component: Any) -> None:
    """Safely shutdown a component that might have async shutdown"""
    try:
        if hasattr(component, "shutdown"):
            if asyncio.iscoroutinefunction(component.shutdown):
                # Handle async shutdown
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(component.shutdown())
                except RuntimeError:
                    # No running loop, create new one
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(component.shutdown())
                    finally:
                        loop.close()
            else:
                # Sync shutdown
                component.shutdown()

        logger.debug(f"Successfully shutdown {type(component).__name__}")

    except Exception as e:
        logger.warning(f"Error shutting down {type(component).__name__}: {e}")


# Global shutdown handler instance
global_shutdown_handler = AsyncShutdownHandler()
