# -*- coding: utf-8 -*-
"""
Hybrid Logging System for Kimera
================================
Provides high-performance logging with optional debug capabilities.

Features:
- Async logging to minimize performance impact
- Ring buffer for recent logs
- Structured logging with performance metrics
- Dynamic log level adjustment
- Per-module configuration
"""

import asyncio
import logging
import time
import json
from collections import deque
from datetime import datetime
from typing import Dict, List, Any, Optional
from threading import Lock
import os


class RingBufferHandler(logging.Handler):
    """In-memory ring buffer for recent logs"""
    
    def __init__(self, capacity: int = 1000):
        super().__init__()
        self.buffer = deque(maxlen=capacity)
        self.lock = Lock()
    
    def emit(self, record):
        """Store log record in ring buffer"""
        with self.lock:
            self.buffer.append({
                'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                'level': record.levelname,
                'logger': record.name,
                'message': self.format(record),
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno,
                'thread': record.thread,
                'process': record.process
            })
    
    def get_logs(self, limit: Optional[int] = None, level: Optional[str] = None) -> List[Dict]:
        """Retrieve logs from buffer with optional filtering"""
        with self.lock:
            logs = list(self.buffer)
            
        if level:
            logs = [log for log in logs if log['level'] == level.upper()]
        
        if limit:
            logs = logs[-limit:]
            
        return logs
    
    def clear(self):
        """Clear the ring buffer"""
        with self.lock:
            self.buffer.clear()


class AsyncLoggingHandler(logging.Handler):
    """Asynchronous logging handler for non-blocking operations"""
    
    def __init__(self, base_handler: logging.Handler):
        super().__init__()
        self.base_handler = base_handler
        self.queue = asyncio.Queue(maxsize=10000)
        self.worker_task = None
        self._running = False
    
    def start(self):
        """Start the async worker"""
        if not self._running:
            self._running = True
            loop = asyncio.get_event_loop()
            self.worker_task = loop.create_task(self._worker())
    
    def stop(self):
        """Stop the async worker"""
        self._running = False
        if self.worker_task:
            self.worker_task.cancel()
    
    async def _worker(self):
        """Process log records asynchronously"""
        while self._running:
            try:
                record = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                self.base_handler.emit(record)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Async logging error: {e}")
    
    def emit(self, record):
        """Queue log record for async processing"""
        try:
            self.queue.put_nowait(record)
        except asyncio.QueueFull:
            # Fallback to synchronous logging if queue is full
            self.base_handler.emit(record)


class PerformanceLogger:
    """Logger with built-in performance tracking"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.metrics = {
            'total_logs': 0,
            'logs_per_level': {},
            'logs_per_module': {},
            'average_log_time_ms': 0
        }
        self._log_times = deque(maxlen=1000)
    
    def log_with_metrics(self, level: int, msg: str, *args, **kwargs):
        """Log with performance tracking"""
        start_time = time.perf_counter()
        
        # Add performance data to extra
        extra = kwargs.get('extra', {})
        extra['perf_timestamp'] = time.time()
        kwargs['extra'] = extra
        
        # Log the message
        self.logger.log(level, msg, *args, **kwargs)
        
        # Track metrics
        log_time = (time.perf_counter() - start_time) * 1000
        self._log_times.append(log_time)
        self.metrics['total_logs'] += 1
        
        level_name = logging.getLevelName(level)
        self.metrics['logs_per_level'][level_name] = \
            self.metrics['logs_per_level'].get(level_name, 0) + 1
        
        if self._log_times:
            self.metrics['average_log_time_ms'] = \
                sum(self._log_times) / len(self._log_times)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get logging performance metrics"""
        return self.metrics.copy()


class HybridLogger:
    """Main hybrid logging system"""
    
    def __init__(self, name: str = "kimera"):
        self.name = name
        self.logger = logging.getLogger(name)
        self.ring_buffer = RingBufferHandler(capacity=5000)
        self.performance_logger = PerformanceLogger(self.logger)
        self.async_handlers = []
        
        # Configuration
        self.debug_mode = os.getenv('KIMERA_DEBUG', 'false').lower() == 'true'
        self.performance_mode = os.getenv('KIMERA_PERFORMANCE', 'true').lower() == 'true'
        
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure logging based on mode"""
        # Base configuration
        if self.debug_mode:
            self.logger.setLevel(logging.DEBUG)
            format_str = '%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s.%(funcName)s:%(lineno)d - %(message)s'
        else:
            self.logger.setLevel(logging.WARNING)
            format_str = '%(asctime)s [%(levelname)s] %(name)s - %(message)s'
        
        formatter = logging.Formatter(format_str, datefmt='%Y-%m-%d %H:%M:%S')
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        if self.performance_mode:
            # Wrap in async handler for non-blocking logging
            async_console = AsyncLoggingHandler(console_handler)
            async_console.start()
            self.async_handlers.append(async_console)
            self.logger.addHandler(async_console)
        else:
            self.logger.addHandler(console_handler)
        
        # Always add ring buffer for debug access
        self.ring_buffer.setFormatter(formatter)
        self.logger.addHandler(self.ring_buffer)
        
        # File handler for debug mode
        if self.debug_mode:
            file_handler = logging.FileHandler('kimera_debug.log')
            file_handler.setFormatter(formatter)
            
            if self.performance_mode:
                async_file = AsyncLoggingHandler(file_handler)
                async_file.start()
                self.async_handlers.append(async_file)
                self.logger.addHandler(async_file)
            else:
                self.logger.addHandler(file_handler)
    
    def set_level(self, level: str, module: Optional[str] = None):
        """Dynamically set log level"""
        numeric_level = getattr(logging, level.upper(), logging.WARNING)
        
        if module:
            module_logger = logging.getLogger(f"{self.name}.{module}")
            module_logger.setLevel(numeric_level)
        else:
            self.logger.setLevel(numeric_level)
    
    def get_recent_logs(self, limit: int = 100, level: Optional[str] = None) -> List[Dict]:
        """Get recent logs from ring buffer"""
        return self.ring_buffer.get_logs(limit=limit, level=level)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get logging performance metrics"""
        return {
            'logger_metrics': self.performance_logger.get_metrics(),
            'buffer_size': len(self.ring_buffer.buffer),
            'debug_mode': self.debug_mode,
            'performance_mode': self.performance_mode,
            'current_level': logging.getLevelName(self.logger.level)
        }
    
    def log_with_context(self, level: str, message: str, **context):
        """Log with additional context for debugging"""
        extra = {
            'context': context,
            'timestamp_ns': time.time_ns()
        }
        
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        
        if self.performance_mode:
            # Use performance logger for metrics tracking
            self.performance_logger.log_with_metrics(
                numeric_level, message, extra=extra
            )
        else:
            self.logger.log(numeric_level, message, extra=extra)
    
    def create_child_logger(self, module: str) -> 'HybridLogger':
        """Create a child logger for a specific module"""
        child = HybridLogger(f"{self.name}.{module}")
        child.debug_mode = self.debug_mode
        child.performance_mode = self.performance_mode
        return child
    
    def shutdown(self):
        """Clean shutdown of async handlers"""
        for handler in self.async_handlers:
            handler.stop()


# Global instance
hybrid_logger = HybridLogger()


# Convenience functions
def get_logger(module: str) -> HybridLogger:
    """Get a logger instance for a module"""
    return hybrid_logger.create_child_logger(module)


def log_performance(operation: str, duration_ms: float, **metadata):
    """Log performance metrics"""
    hybrid_logger.log_with_context(
        'INFO',
        f"Performance: {operation} completed in {duration_ms:.2f}ms",
        operation=operation,
        duration_ms=duration_ms,
        **metadata
    )


def enable_debug_mode():
    """Enable debug mode at runtime"""
    hybrid_logger.debug_mode = True
    hybrid_logger.set_level('DEBUG')


def disable_debug_mode():
    """Disable debug mode at runtime"""
    hybrid_logger.debug_mode = False
    hybrid_logger.set_level('WARNING')