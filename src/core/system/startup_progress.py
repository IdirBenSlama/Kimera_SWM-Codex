"""
Startup Progress Tracking for KIMERA System
Provides real-time feedback during system initialization
Phase 3, Week 8: Performance Optimization
"""

import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ProgressStage(Enum):
    """Startup progress stages"""

    INITIALIZING = "initializing"
    LOADING_CONFIG = "loading_config"
    CONNECTING_DB = "connecting_db"
    LOADING_MODELS = "loading_models"
    STARTING_SERVICES = "starting_services"
    READY = "ready"
    FAILED = "failed"


@dataclass
class ProgressStep:
    """Auto-generated class."""
    pass
    """Individual progress step"""

    name: str
    stage: ProgressStage
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    progress: float = 0.0  # 0.0 to 1.0
    message: str = ""
    error: Optional[str] = None

    @property
    def duration(self) -> Optional[float]:
        if self.end_time:
            return self.end_time - self.start_time
        return None

    @property
    def is_complete(self) -> bool:
        return self.end_time is not None

    def complete(self, message: str = ""):
        """Mark step as complete"""
        self.end_time = time.time()
        self.progress = 1.0
        if message:
            self.message = message

    def fail(self, error: str):
        """Mark step as failed"""
        self.end_time = time.time()
        self.error = error

    def update_progress(self, progress: float, message: str = ""):
        """Update step progress"""
        self.progress = max(0.0, min(1.0, progress))
        if message:
            self.message = message
class StartupProgressTracker:
    """Auto-generated class."""
    pass
    """
    Tracks and reports startup progress with real-time updates
    """

    def __init__(self):
        self.steps: List[ProgressStep] = []
        self.current_step: Optional[ProgressStep] = None
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.callbacks: List[Callable[[ProgressStep], None]] = []
        self._lock = asyncio.Lock()

        # WebSocket connections for real-time updates
        self.websocket_connections: List[Any] = []

        logger.info("Startup progress tracker initialized")

    def add_callback(self, callback: Callable[[ProgressStep], None]):
        """Add callback for progress updates"""
        self.callbacks.append(callback)

    def add_websocket(self, websocket):
        """Add WebSocket connection for real-time updates"""
        self.websocket_connections.append(websocket)

    def remove_websocket(self, websocket):
        """Remove WebSocket connection"""
        if websocket in self.websocket_connections:
            self.websocket_connections.remove(websocket)

    async def start_step(
        self, name: str, stage: ProgressStage, message: str = ""
    ) -> ProgressStep:
        """Start a new progress step"""
        async with self._lock:
            # Complete previous step if not already done
            if self.current_step and not self.current_step.is_complete:
                self.current_step.complete()

            # Create new step
            step = ProgressStep(name=name, stage=stage, message=message)

            self.steps.append(step)
            self.current_step = step

            logger.info(f"Starting step: {name} ({stage.value})")

            # Notify callbacks
            await self._notify_callbacks(step)

            return step

    async def update_current_step(self, progress: float, message: str = ""):
        """Update current step progress"""
        if self.current_step:
            async with self._lock:
                self.current_step.update_progress(progress, message)
                await self._notify_callbacks(self.current_step)

    async def complete_current_step(self, message: str = ""):
        """Complete current step"""
        if self.current_step:
            async with self._lock:
                self.current_step.complete(message)
                logger.info(
                    f"Completed step: {self.current_step.name} in {self.current_step.duration:.2f}s"
                )
                await self._notify_callbacks(self.current_step)

    async def fail_current_step(self, error: str):
        """Fail current step"""
        if self.current_step:
            async with self._lock:
                self.current_step.fail(error)
                logger.error(f"Failed step: {self.current_step.name} - {error}")
                await self._notify_callbacks(self.current_step)

    async def complete_startup(self):
        """Mark startup as complete"""
        async with self._lock:
            if self.current_step and not self.current_step.is_complete:
                await self.complete_current_step("Startup complete")

            self.end_time = time.time()
            total_time = self.end_time - self.start_time

            logger.info(f"Startup completed in {total_time:.2f}s")

            # Send final update
            await self._broadcast_websocket(
                {
                    "type": "startup_complete",
                    "total_time": total_time
                    "summary": self.get_summary(),
                }
            )

    async def _notify_callbacks(self, step: ProgressStep):
        """Notify all callbacks of progress update"""
        for callback in self.callbacks:
            try:
                callback(step)
            except Exception as e:
                logger.error(f"Progress callback error: {e}")

        # Send WebSocket update
        await self._broadcast_websocket(
            {
                "type": "progress_update",
                "step": {
                    "name": step.name
                    "stage": step.stage.value
                    "progress": step.progress
                    "message": step.message
                    "duration": step.duration
                    "error": step.error
                },
                "overall_progress": self.get_overall_progress(),
            }
        )

    async def _broadcast_websocket(self, data: Dict[str, Any]):
        """Broadcast data to all WebSocket connections"""
        if not self.websocket_connections:
            return

        message = json.dumps(data)
        disconnected = []

        for ws in self.websocket_connections:
            try:
                await ws.send_text(message)
            except Exception:
                disconnected.append(ws)

        # Remove disconnected WebSockets
        for ws in disconnected:
            self.websocket_connections.remove(ws)

    def get_overall_progress(self) -> float:
        """Calculate overall progress percentage"""
        if not self.steps:
            return 0.0

        total_progress = sum(step.progress for step in self.steps)
        return total_progress / len(self.steps)

    def get_summary(self) -> Dict[str, Any]:
        """Get startup summary"""
        total_time = (self.end_time or time.time()) - self.start_time
        completed_steps = [s for s in self.steps if s.is_complete and not s.error]
        failed_steps = [s for s in self.steps if s.error]

        return {
            "total_time": total_time
            "total_steps": len(self.steps),
            "completed_steps": len(completed_steps),
            "failed_steps": len(failed_steps),
            "overall_progress": self.get_overall_progress(),
            "is_complete": self.end_time is not None
            "steps": [
                {
                    "name": step.name
                    "stage": step.stage.value
                    "duration": step.duration
                    "error": step.error
                }
                for step in self.steps
            ],
        }

    def get_detailed_report(self) -> Dict[str, Any]:
        """Get detailed startup report"""
        return {
            "summary": self.get_summary(),
            "timeline": [
                {
                    "name": step.name
                    "stage": step.stage.value
                    "start_time": datetime.fromtimestamp(step.start_time).isoformat(),
                    "end_time": (
                        datetime.fromtimestamp(step.end_time).isoformat()
                        if step.end_time
                        else None
                    ),
                    "duration": step.duration
                    "progress": step.progress
                    "message": step.message
                    "error": step.error
                }
                for step in self.steps
            ],
            "performance_metrics": {
                "fastest_step": min(
                    (s for s in self.steps if s.duration),
                    key=lambda x: x.duration
                    default=None
                ),
                "slowest_step": max(
                    (s for s in self.steps if s.duration),
                    key=lambda x: x.duration
                    default=None
                ),
                "average_step_time": (
                    sum(s.duration for s in self.steps if s.duration)
                    / len([s for s in self.steps if s.duration])
                    if self.steps
                    else 0
                ),
            },
        }


# Global progress tracker
_progress_tracker: Optional[StartupProgressTracker] = None
_tracker_lock = threading.Lock()


def get_progress_tracker() -> StartupProgressTracker:
    """
    Get global progress tracker instance (thread-safe).

    This uses a double-checked locking pattern to ensure thread-safe
    instantiation of the singleton. The instance itself uses an
    asyncio.Lock for its async operations.
    """
    global _progress_tracker
    if _progress_tracker is None:
        with _tracker_lock:
            if _progress_tracker is None:
                _progress_tracker = StartupProgressTracker()
    return _progress_tracker


# Context manager for progress steps
class ProgressStepContext:
    """Auto-generated class."""
    pass
    """Context manager for tracking progress steps"""

    def __init__(self, name: str, stage: ProgressStage, message: str = ""):
        self.name = name
        self.stage = stage
        self.message = message
        self.tracker = get_progress_tracker()
        self.step: Optional[ProgressStep] = None

    async def __aenter__(self):
        self.step = await self.tracker.start_step(self.name, self.stage, self.message)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            await self.tracker.fail_current_step(str(exc_val))
        else:
            await self.tracker.complete_current_step()

    async def update(self, progress: float, message: str = ""):
        """Update progress within the step"""
        await self.tracker.update_current_step(progress, message)


# Convenience functions
def track_startup_step(name: str, stage: ProgressStage, message: str = ""):
    """Context manager for tracking a startup step"""
    return ProgressStepContext(name, stage, message)


def log_progress_callback(step: ProgressStep):
    """Default callback that logs progress to console"""
    if step.error:
        logger.error(f"FAILED {step.name}: {step.error}")
    elif step.is_complete:
        logger.info(f"COMPLETED {step.name} ({step.duration:.2f}s)")
    else:
        logger.info(f"PROGRESS {step.name}: {step.progress:.0%} - {step.message}")


# Initialize default logging callback
get_progress_tracker().add_callback(log_progress_callback)
