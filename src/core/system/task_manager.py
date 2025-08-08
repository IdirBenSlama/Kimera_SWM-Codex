"""
Task Manager for KIMERA System
Implements proper async/await patterns with lifecycle management
Phase 2, Week 5: Async/Await Patterns Implementation
"""

import asyncio
import logging
import traceback
import weakref
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, Optional, Set

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskInfo:
    """Auto-generated class."""
    pass
    """Information about a managed task"""

    name: str
    status: TaskStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    result: Any = None
    task: Optional[asyncio.Task] = field(default=None, repr=False)
    cleanup_callback: Optional[Callable] = field(default=None, repr=False)
class TaskManager:
    """Auto-generated class."""
    pass
    """
    Centralized task management system for KIMERA
    Handles async task lifecycle, monitoring, and cleanup
    """

    def __init__(self, max_concurrent_tasks: int = 100):
        self.tasks: Dict[str, TaskInfo] = {}
        self.task_results: Dict[str, Any] = {}
        self.max_concurrent_tasks = max_concurrent_tasks
        self._lock = asyncio.Lock()
        self._shutdown = False
        self._background_tasks: Set[asyncio.Task] = set()

        # Weak references to track task completion
        self._task_refs: weakref.WeakValueDictionary = weakref.WeakValueDictionary()

        logger.info(
            f"TaskManager initialized with max_concurrent_tasks={max_concurrent_tasks}"
        )

    async def create_managed_task(
        self
        name: str
        coro: Coroutine
        cleanup: Optional[Callable] = None
        replace_existing: bool = True
    ) -> asyncio.Task:
        """
        Create and manage an async task with proper lifecycle management

        Args:
            name: Unique identifier for the task
            coro: Coroutine to execute
            cleanup: Optional cleanup function to call on completion
            replace_existing: Whether to cancel existing task with same name

        Returns:
            The created asyncio.Task

        Raises:
            RuntimeError: If max concurrent tasks exceeded
        """
        async with self._lock:
            if self._shutdown:
                raise RuntimeError("TaskManager is shutting down")

            # Check concurrent task limit
            running_tasks = sum(
                1 for task in self.tasks.values() if task.status == TaskStatus.RUNNING
            )
            if running_tasks >= self.max_concurrent_tasks:
                raise RuntimeError(
                    f"Maximum concurrent tasks ({self.max_concurrent_tasks}) exceeded"
                )

            # Handle existing task
            if name in self.tasks:
                if replace_existing:
                    await self._cancel_task(name)
                else:
                    raise ValueError(f"Task '{name}' already exists")

            # Create task info
            task_info = TaskInfo(
                name=name
                status=TaskStatus.PENDING
                created_at=datetime.now(),
                cleanup_callback=cleanup
            )

            # Create the actual task
            task = asyncio.create_task(self._execute_task(name, coro))
            task_info.task = task

            # Store task info
            self.tasks[name] = task_info
            self._task_refs[id(task)] = task

            # Add completion callback
            task.add_done_callback(
                lambda t: asyncio.create_task(self._task_done(name, t))
            )

            logger.info(f"Created managed task: {name}")
            return task

    async def _execute_task(self, name: str, coro: Coroutine) -> Any:
        """Execute a task with proper status tracking"""
        task_info = self.tasks[name]

        try:
            # Update status to running
            task_info.status = TaskStatus.RUNNING
            task_info.started_at = datetime.now()

            logger.debug(f"Task '{name}' started execution")

            # Execute the coroutine
            result = await coro

            # Store result
            task_info.result = result
            self.task_results[name] = result

            return result

        except asyncio.CancelledError:
            logger.info(f"Task '{name}' was cancelled")
            raise

        except Exception as e:
            logger.error(f"Task '{name}' failed with error: {e}", exc_info=True)
            task_info.error = str(e)
            raise

    async def _task_done(self, name: str, task: asyncio.Task) -> None:
        """Handle task completion"""
        async with self._lock:
            if name not in self.tasks:
                return

            task_info = self.tasks[name]
            task_info.completed_at = datetime.now()

            # Update status based on task result
            if task.cancelled():
                task_info.status = TaskStatus.CANCELLED
            elif task.exception():
                task_info.status = TaskStatus.FAILED
                task_info.error = str(task.exception())
            else:
                task_info.status = TaskStatus.COMPLETED

            # Run cleanup callback if provided
            if task_info.cleanup_callback:
                try:
                    if asyncio.iscoroutinefunction(task_info.cleanup_callback):
                        await task_info.cleanup_callback()
                    else:
                        task_info.cleanup_callback()
                except Exception as e:
                    logger.error(f"Cleanup callback for task '{name}' failed: {e}")

            # Log task completion
            duration = (task_info.completed_at - task_info.started_at).total_seconds()
            logger.info(
                f"Task '{name}' completed with status {task_info.status} "
                f"after {duration:.2f} seconds"
            )

    async def _cancel_task(self, name: str) -> None:
        """Cancel a running task"""
        if name in self.tasks:
            task_info = self.tasks[name]
            if task_info.task and not task_info.task.done():
                task_info.task.cancel()
                try:
                    await task_info.task
                except asyncio.CancelledError:
                    pass
                logger.info(f"Cancelled task: {name}")

    async def cancel_task(self, name: str) -> bool:
        """
        Cancel a specific task by name

        Args:
            name: Task identifier

        Returns:
            True if task was cancelled, False if not found or already completed
        """
        async with self._lock:
            if name not in self.tasks:
                return False

            task_info = self.tasks[name]
            if task_info.status in [
                TaskStatus.COMPLETED
                TaskStatus.FAILED
                TaskStatus.CANCELLED
            ]:
                return False

            await self._cancel_task(name)
            return True

    async def wait_for_task(self, name: str, timeout: Optional[float] = None) -> Any:
        """
        Wait for a task to complete and return its result

        Args:
            name: Task identifier
            timeout: Maximum time to wait in seconds

        Returns:
            Task result

        Raises:
            KeyError: If task not found
            asyncio.TimeoutError: If timeout exceeded
            Exception: If task failed
        """
        if name not in self.tasks:
            raise KeyError(f"Task '{name}' not found")

        task_info = self.tasks[name]
        if task_info.task:
            try:
                result = await asyncio.wait_for(task_info.task, timeout=timeout)
                return result
            except asyncio.TimeoutError:
                logger.warning(f"Timeout waiting for task '{name}'")
                raise

    def get_task_status(self, name: str) -> Optional[TaskStatus]:
        """Get the current status of a task"""
        if name in self.tasks:
            return self.tasks[name].status
        return None

    def get_task_info(self, name: str) -> Optional[TaskInfo]:
        """Get detailed information about a task"""
        return self.tasks.get(name)

    def get_all_tasks(self) -> Dict[str, TaskInfo]:
        """Get information about all tasks"""
        return self.tasks.copy()

    def get_running_tasks(self) -> Dict[str, TaskInfo]:
        """Get all currently running tasks"""
        return {
            name: info
            for name, info in self.tasks.items()
            if info.status == TaskStatus.RUNNING
        }

    async def cancel_all_tasks(self) -> None:
        """Cancel all running tasks"""
        async with self._lock:
            tasks_to_cancel = [
                (name, info)
                for name, info in self.tasks.items()
                if info.status == TaskStatus.RUNNING
            ]

            for name, _ in tasks_to_cancel:
                await self._cancel_task(name)

            logger.info(f"Cancelled {len(tasks_to_cancel)} tasks")

    async def shutdown(self, timeout: float = 30.0) -> None:
        """
        Gracefully shutdown the task manager

        Args:
            timeout: Maximum time to wait for tasks to complete
        """
        logger.info("TaskManager shutdown initiated")

        async with self._lock:
            self._shutdown = True

        # Cancel all running tasks
        await self.cancel_all_tasks()

        # Wait for all tasks to complete
        running_tasks = [
            info.task
            for info in self.tasks.values()
            if info.task and not info.task.done()
        ]

        if running_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*running_tasks, return_exceptions=True),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                logger.warning(
                    f"Shutdown timeout exceeded, {len(running_tasks)} tasks still running"
                )

        logger.info("TaskManager shutdown complete")

    def create_background_task(self, coro: Coroutine) -> asyncio.Task:
        """
        Create a background task that's tracked but not managed
        Useful for fire-and-forget operations that still need cleanup
        """
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup"""
        await self.shutdown()


# --- Thread-Safe Singleton Access ---
_task_manager: Optional[TaskManager] = None
_manager_lock = asyncio.Lock()


def get_task_manager() -> TaskManager:
    """
    Get the global task manager instance.

    Raises:
        RuntimeError: If the manager has not been initialized via
                      `initialize_task_manager()`.
    """
    if _task_manager is None:
        raise RuntimeError(
            "TaskManager not initialized. " "Call initialize_task_manager() first."
        )
    return _task_manager


async def initialize_task_manager(max_concurrent_tasks: int = 100) -> TaskManager:
    """
    Initializes the global task manager in a thread-safe/async-safe way.
    """
    global _task_manager
    if _task_manager is None:
        async with _manager_lock:
            if _task_manager is None:
                _task_manager = TaskManager(max_concurrent_tasks)
    return _task_manager
