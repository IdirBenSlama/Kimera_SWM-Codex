from __future__ import annotations

"""backend.utils.threading_utils
--------------------------------
Utility helpers for running background tasks in a safe, logged manner.

This module abstracts the creation of background threads so that modules can
kick-off asynchronous initialisation or long-running maintenance work without
blocking the main application startup.

Key features
============
1. Automatic daemon threads – background tasks will not prevent interpreter
   shutdown.
2. Robust error handling – any exception raised inside the task is caught and
   logged via the standard Kimera logger.
3. Coroutine awareness – if the supplied *func* is a coroutine function, it is
   executed in its own event-loop via ``asyncio.run`` so that regular and async
   call-sites share the same helper.
4. Type-hints and documentation to comply with the Zero-Debugging Constraint.

Example
-------
>>> from backend.utils.threading_utils import start_background_task
>>> def long_running_job():
...     ...
>>> start_background_task(long_running_job)
"""

from typing import Any, Callable
import asyncio
import logging
import threading

logger = logging.getLogger(__name__)


def _run_sync(func: Callable[..., Any], *args: Any, **kwargs: Any) -> None:  # noqa: D401
    """Run a synchronous function, catching and logging any exception."""
    try:
        func(*args, **kwargs)
    except Exception as exc:  # pragma: no cover – we only log here
        logger.exception("Background task %s raised an exception: %s", func.__name__, exc)


def _run_async(coro_func: Callable[..., Any], *args: Any, **kwargs: Any) -> None:  # noqa: D401
    """Run an *async* coroutine function inside its own event-loop."""
    try:
        asyncio.run(coro_func(*args, **kwargs))
    except Exception as exc:  # pragma: no cover
        logger.exception("Async background task %s raised an exception: %s", coro_func.__name__, exc)


def start_background_task(func: Callable[..., Any], *args: Any, **kwargs: Any) -> threading.Thread:  # noqa: D401
    """Start *func* in a background daemon thread.

    Parameters
    ----------
    func:
        The callable (sync or async) to run.
    *args, **kwargs:
        Arguments forwarded to *func*.

    Returns
    -------
    threading.Thread
        The ``Thread`` object so the caller may inspect or join it if needed.
    """
    # Decide whether *func* is async or sync.
    target: Callable[..., Any]
    if asyncio.iscoroutinefunction(func):
        target = _run_async  # type: ignore[assignment]
    else:
        target = _run_sync  # type: ignore[assignment]

    thread = threading.Thread(
        target=target,
        args=(func, *args),
        kwargs=kwargs,
        name=f"BackgroundTask:{func.__name__}",
        daemon=True,
    )

    thread.start()
    logger.debug("Started background task '%s' (thread ident=%s)", func.__name__, thread.ident)
    return thread 