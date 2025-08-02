#!/usr/bin/env python3
"""
Kimera SWM Pipeline Optimization System
======================================

Advanced pipeline optimization for parallel processing, dynamic load balancing,
and intelligent resource allocation in Kimera SWM cognitive operations.

This module delivers:
- Async component orchestration with dependency management
- Dynamic load balancing across processing units
- Intelligent resource allocation and scheduling
- Performance profiling and bottleneck detection
- Adaptive pipeline optimization

Author: Kimera SWM Development Team
Date: January 30, 2025
Version: 5.2.0
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Union, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from collections import defaultdict, deque
import heapq

import torch
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    MEDIUM = 3
    HIGH = 5
    CRITICAL = 7
    URGENT = 10


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PipelineTask:
    """Individual pipeline task"""
    task_id: str
    function: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.MEDIUM
    dependencies: Set[str] = field(default_factory=set)
    estimated_duration: float = 1.0  # seconds
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    
    # Execution state
    status: TaskStatus = TaskStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    result: Any = None
    error: Optional[Exception] = None
    
    # Metrics
    actual_duration: Optional[float] = None
    resource_usage: Dict[str, float] = field(default_factory=dict)
    
    def __lt__(self, other):
        """Comparison for priority queue"""
        return self.priority.value > other.priority.value


@dataclass
class ResourcePool:
    """Computing resource pool"""
    cpu_cores: int = 4
    memory_gb: float = 8.0
    gpu_memory_gb: float = 0.0
    max_concurrent_tasks: int = 10
    
    # Current usage
    allocated_cpu: float = 0.0
    allocated_memory: float = 0.0
    allocated_gpu_memory: float = 0.0
    active_tasks: int = 0
    
    def can_allocate(self, requirements: Dict[str, float]) -> bool:
        """Check if resources can be allocated"""
        cpu_needed = requirements.get('cpu', 0.1)
        memory_needed = requirements.get('memory', 0.1)
        gpu_memory_needed = requirements.get('gpu_memory', 0.0)
        
        return (
            self.allocated_cpu + cpu_needed <= self.cpu_cores and
            self.allocated_memory + memory_needed <= self.memory_gb and
            self.allocated_gpu_memory + gpu_memory_needed <= self.gpu_memory_gb and
            self.active_tasks < self.max_concurrent_tasks
        )
    
    def allocate(self, requirements: Dict[str, float]) -> bool:
        """Allocate resources"""
        if not self.can_allocate(requirements):
            return False
        
        self.allocated_cpu += requirements.get('cpu', 0.1)
        self.allocated_memory += requirements.get('memory', 0.1)
        self.allocated_gpu_memory += requirements.get('gpu_memory', 0.0)
        self.active_tasks += 1
        return True
    
    def deallocate(self, requirements: Dict[str, float]):
        """Deallocate resources"""
        self.allocated_cpu -= requirements.get('cpu', 0.1)
        self.allocated_memory -= requirements.get('memory', 0.1)
        self.allocated_gpu_memory -= requirements.get('gpu_memory', 0.0)
        self.active_tasks -= 1
        
        # Ensure non-negative values
        self.allocated_cpu = max(0, self.allocated_cpu)
        self.allocated_memory = max(0, self.allocated_memory)
        self.allocated_gpu_memory = max(0, self.allocated_gpu_memory)
        self.active_tasks = max(0, self.active_tasks)


@dataclass
class PipelineMetrics:
    """Pipeline performance metrics"""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    
    total_execution_time: float = 0.0
    avg_task_duration: float = 0.0
    throughput_tasks_per_second: float = 0.0
    
    resource_efficiency: float = 0.0
    parallelization_factor: float = 1.0
    
    bottleneck_stages: List[str] = field(default_factory=list)
    performance_score: float = 0.0
    
    last_updated: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))


class DependencyGraph:
    """Task dependency management"""
    
    def __init__(self):
        self.graph: Dict[str, Set[str]] = defaultdict(set)  # task_id -> dependencies
        self.reverse_graph: Dict[str, Set[str]] = defaultdict(set)  # task_id -> dependents
        self.completed_tasks: Set[str] = set()
    
    def add_dependency(self, task_id: str, dependency_id: str):
        """Add dependency relationship"""
        self.graph[task_id].add(dependency_id)
        self.reverse_graph[dependency_id].add(task_id)
    
    def mark_completed(self, task_id: str):
        """Mark task as completed"""
        self.completed_tasks.add(task_id)
    
    def get_ready_tasks(self, all_tasks: Set[str]) -> Set[str]:
        """Get tasks that are ready to run (all dependencies completed)"""
        ready_tasks = set()
        
        for task_id in all_tasks:
            if task_id in self.completed_tasks:
                continue
            
            dependencies = self.graph.get(task_id, set())
            if dependencies.issubset(self.completed_tasks):
                ready_tasks.add(task_id)
        
        return ready_tasks
    
    def get_dependent_tasks(self, task_id: str) -> Set[str]:
        """Get tasks that depend on the given task"""
        return self.reverse_graph.get(task_id, set())
    
    def validate_dependencies(self, all_tasks: Set[str]) -> bool:
        """Validate that all dependencies are valid and no cycles exist"""
        # Check for invalid dependencies
        for task_id in all_tasks:
            dependencies = self.graph.get(task_id, set())
            if not dependencies.issubset(all_tasks):
                logger.error(f"Task {task_id} has invalid dependencies: {dependencies - all_tasks}")
                return False
        
        # Check for cycles using DFS
        visited = set()
        recursion_stack = set()
        
        def has_cycle(task_id: str) -> bool:
            if task_id in recursion_stack:
                return True
            if task_id in visited:
                return False
            
            visited.add(task_id)
            recursion_stack.add(task_id)
            
            for dependency in self.graph.get(task_id, set()):
                if has_cycle(dependency):
                    return True
            
            recursion_stack.remove(task_id)
            return False
        
        for task_id in all_tasks:
            if task_id not in visited:
                if has_cycle(task_id):
                    logger.error(f"Cycle detected involving task {task_id}")
                    return False
        
        return True


class LoadBalancer:
    """Dynamic load balancing for task distribution"""
    
    def __init__(self, resource_pools: List[ResourcePool]):
        self.resource_pools = resource_pools
        self.pool_metrics = [{"tasks_completed": 0, "avg_duration": 0.0} for _ in resource_pools]
        self._lock = threading.Lock()
    
    def select_optimal_pool(self, task: PipelineTask) -> Optional[int]:
        """Select optimal resource pool for task execution"""
        with self._lock:
            best_pool = None
            best_score = float('-inf')
            
            for i, pool in enumerate(self.resource_pools):
                if not pool.can_allocate(task.resource_requirements):
                    continue
                
                # Calculate pool score based on multiple factors
                utilization = pool.active_tasks / pool.max_concurrent_tasks
                efficiency = self.pool_metrics[i]["tasks_completed"] / max(1, pool.active_tasks)
                
                # Prefer less utilized pools with good efficiency
                score = (1 - utilization) * 0.6 + efficiency * 0.4
                
                if score > best_score:
                    best_score = score
                    best_pool = i
            
            return best_pool
    
    def update_pool_metrics(self, pool_index: int, task_duration: float):
        """Update pool performance metrics"""
        with self._lock:
            metrics = self.pool_metrics[pool_index]
            metrics["tasks_completed"] += 1
            
            # Exponential moving average for duration
            alpha = 0.2
            metrics["avg_duration"] = (
                alpha * task_duration + 
                (1 - alpha) * metrics["avg_duration"]
            )
    
    def get_pool_status(self) -> List[Dict[str, Any]]:
        """Get status of all resource pools"""
        status = []
        for i, pool in enumerate(self.resource_pools):
            status.append({
                "pool_id": i,
                "cpu_utilization": pool.allocated_cpu / pool.cpu_cores,
                "memory_utilization": pool.allocated_memory / pool.memory_gb,
                "gpu_utilization": pool.allocated_gpu_memory / max(0.1, pool.gpu_memory_gb),
                "active_tasks": pool.active_tasks,
                "max_tasks": pool.max_concurrent_tasks,
                "tasks_completed": self.pool_metrics[i]["tasks_completed"],
                "avg_duration": self.pool_metrics[i]["avg_duration"]
            })
        return status


class PipelineOptimizer:
    """Main pipeline optimization engine"""
    
    def __init__(self, resource_pools: Optional[List[ResourcePool]] = None):
        # Resource management
        self.resource_pools = resource_pools or [ResourcePool()]
        self.load_balancer = LoadBalancer(self.resource_pools)
        
        # Task management
        self.tasks: Dict[str, PipelineTask] = {}
        self.task_queue = []  # Priority queue
        self.dependency_graph = DependencyGraph()
        
        # Execution
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.running_tasks: Dict[str, asyncio.Task] = {}
        
        # Monitoring
        self.metrics = PipelineMetrics()
        self.performance_history = deque(maxlen=100)
        
        # State
        self.is_running = False
        self._shutdown_event = asyncio.Event()
    
    def add_task(self, task: PipelineTask) -> bool:
        """Add task to pipeline"""
        try:
            # Validate task
            if task.task_id in self.tasks:
                logger.warning(f"Task {task.task_id} already exists")
                return False
            
            # Add to tasks
            self.tasks[task.task_id] = task
            
            # Add dependencies
            for dep_id in task.dependencies:
                self.dependency_graph.add_dependency(task.task_id, dep_id)
            
            # Add to queue if ready
            if not task.dependencies:
                heapq.heappush(self.task_queue, task)
                task.status = TaskStatus.QUEUED
            
            self.metrics.total_tasks += 1
            logger.debug(f"Added task {task.task_id} to pipeline")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add task {task.task_id}: {e}")
            return False
    
    def remove_task(self, task_id: str) -> bool:
        """Remove task from pipeline"""
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        
        # Cancel if running
        if task_id in self.running_tasks:
            self.running_tasks[task_id].cancel()
            del self.running_tasks[task_id]
        
        # Remove from queue
        self.task_queue = [t for t in self.task_queue if t.task_id != task_id]
        heapq.heapify(self.task_queue)
        
        # Mark as cancelled
        task.status = TaskStatus.CANCELLED
        
        # Remove from tasks
        del self.tasks[task_id]
        
        return True
    
    async def execute_task(self, task: PipelineTask, pool_index: int) -> Any:
        """Execute individual task"""
        pool = self.resource_pools[pool_index]
        
        try:
            # Allocate resources
            if not pool.allocate(task.resource_requirements):
                raise RuntimeError(f"Failed to allocate resources for task {task.task_id}")
            
            # Update task status
            task.status = TaskStatus.RUNNING
            task.start_time = time.time()
            
            # Execute function
            if asyncio.iscoroutinefunction(task.function):
                result = await task.function(*task.args, **task.kwargs)
            else:
                # Run in thread pool for CPU-bound tasks
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.executor, task.function, *task.args, **task.kwargs
                )
            
            # Update task completion
            task.end_time = time.time()
            task.actual_duration = task.end_time - task.start_time
            task.result = result
            task.status = TaskStatus.COMPLETED
            
            # Update metrics
            self.metrics.completed_tasks += 1
            self.metrics.total_execution_time += task.actual_duration
            self.load_balancer.update_pool_metrics(pool_index, task.actual_duration)
            
            logger.debug(f"Task {task.task_id} completed in {task.actual_duration:.3f}s")
            return result
            
        except Exception as e:
            task.end_time = time.time()
            task.error = e
            task.status = TaskStatus.FAILED
            self.metrics.failed_tasks += 1
            
            logger.error(f"Task {task.task_id} failed: {e}")
            raise
            
        finally:
            # Deallocate resources
            pool.deallocate(task.resource_requirements)
            
            # Mark dependency as completed
            self.dependency_graph.mark_completed(task.task_id)
            
            # Check for newly ready tasks
            await self._check_ready_tasks()
    
    async def _check_ready_tasks(self):
        """Check for tasks that became ready after dependency completion"""
        all_task_ids = set(self.tasks.keys())
        ready_task_ids = self.dependency_graph.get_ready_tasks(all_task_ids)
        
        for task_id in ready_task_ids:
            task = self.tasks[task_id]
            if task.status == TaskStatus.PENDING:
                heapq.heappush(self.task_queue, task)
                task.status = TaskStatus.QUEUED
                logger.debug(f"Task {task_id} is now ready")
    
    async def _process_task_queue(self):
        """Main task processing loop"""
        while self.is_running and not self._shutdown_event.is_set():
            try:
                # Get next task from queue
                if not self.task_queue:
                    await asyncio.sleep(0.1)
                    continue
                
                task = heapq.heappop(self.task_queue)
                
                # Select optimal resource pool
                pool_index = self.load_balancer.select_optimal_pool(task)
                
                if pool_index is None:
                    # No resources available, put back in queue
                    heapq.heappush(self.task_queue, task)
                    await asyncio.sleep(0.1)
                    continue
                
                # Execute task asynchronously
                task_coroutine = self.execute_task(task, pool_index)
                async_task = asyncio.create_task(task_coroutine)
                self.running_tasks[task.task_id] = async_task
                
                # Clean up completed tasks
                await self._cleanup_completed_tasks()
                
            except Exception as e:
                logger.error(f"Error in task processing loop: {e}")
                await asyncio.sleep(1.0)
    
    async def _cleanup_completed_tasks(self):
        """Clean up completed async tasks"""
        completed_task_ids = []
        
        for task_id, async_task in self.running_tasks.items():
            if async_task.done():
                completed_task_ids.append(task_id)
                
                try:
                    # Get result or exception
                    await async_task
                except Exception as e:
                    logger.error(f"Task {task_id} raised exception: {e}")
        
        # Remove completed tasks
        for task_id in completed_task_ids:
            del self.running_tasks[task_id]
    
    async def start_pipeline(self):
        """Start the pipeline processing"""
        if self.is_running:
            logger.warning("Pipeline is already running")
            return
        
        # Validate dependencies
        all_task_ids = set(self.tasks.keys())
        if not self.dependency_graph.validate_dependencies(all_task_ids):
            raise ValueError("Invalid task dependencies detected")
        
        self.is_running = True
        self._shutdown_event.clear()
        
        logger.info("Starting pipeline optimization engine")
        
        # Start processing loop
        await self._process_task_queue()
    
    async def stop_pipeline(self):
        """Stop the pipeline processing"""
        logger.info("Stopping pipeline optimization engine")
        
        self.is_running = False
        self._shutdown_event.set()
        
        # Cancel running tasks
        for async_task in self.running_tasks.values():
            async_task.cancel()
        
        # Wait for tasks to complete
        if self.running_tasks:
            await asyncio.gather(*self.running_tasks.values(), return_exceptions=True)
        
        self.running_tasks.clear()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("Pipeline stopped")
    
    def get_pipeline_metrics(self) -> PipelineMetrics:
        """Get comprehensive pipeline metrics"""
        # Update derived metrics
        if self.metrics.completed_tasks > 0:
            self.metrics.avg_task_duration = (
                self.metrics.total_execution_time / self.metrics.completed_tasks
            )
        
        # Calculate throughput
        if self.metrics.total_execution_time > 0:
            self.metrics.throughput_tasks_per_second = (
                self.metrics.completed_tasks / self.metrics.total_execution_time
            )
        
        # Resource efficiency
        total_capacity = sum(pool.max_concurrent_tasks for pool in self.resource_pools)
        active_tasks = sum(pool.active_tasks for pool in self.resource_pools)
        self.metrics.resource_efficiency = active_tasks / total_capacity if total_capacity > 0 else 0
        
        # Parallelization factor
        self.metrics.parallelization_factor = len(self.running_tasks)
        
        # Performance score (combination of throughput, efficiency, and completion rate)
        completion_rate = (
            self.metrics.completed_tasks / self.metrics.total_tasks 
            if self.metrics.total_tasks > 0 else 0
        )
        self.metrics.performance_score = (
            self.metrics.throughput_tasks_per_second * 0.4 +
            self.metrics.resource_efficiency * 0.3 +
            completion_rate * 0.3
        )
        
        return self.metrics
    
    def get_task_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all tasks"""
        status = {}
        
        for task_id, task in self.tasks.items():
            status[task_id] = {
                "status": task.status.value,
                "priority": task.priority.value,
                "dependencies": list(task.dependencies),
                "estimated_duration": task.estimated_duration,
                "actual_duration": task.actual_duration,
                "start_time": task.start_time,
                "end_time": task.end_time,
                "error": str(task.error) if task.error else None
            }
        
        return status


# Global pipeline optimizer
pipeline_optimizer = PipelineOptimizer()

# Convenience functions
def add_pipeline_task(task_id: str, function: Callable, *args, 
                     priority: TaskPriority = TaskPriority.MEDIUM,
                     dependencies: Optional[Set[str]] = None,
                     estimated_duration: float = 1.0,
                     resource_requirements: Optional[Dict[str, float]] = None,
                     **kwargs) -> bool:
    """Add task to pipeline"""
    global pipeline_optimizer
    
    task = PipelineTask(
        task_id=task_id,
        function=function,
        args=args,
        kwargs=kwargs,
        priority=priority,
        dependencies=dependencies or set(),
        estimated_duration=estimated_duration,
        resource_requirements=resource_requirements or {}
    )
    
    return pipeline_optimizer.add_task(task)

async def start_pipeline():
    """Start pipeline processing"""
    global pipeline_optimizer
    await pipeline_optimizer.start_pipeline()

async def stop_pipeline():
    """Stop pipeline processing"""
    global pipeline_optimizer
    await pipeline_optimizer.stop_pipeline()

def get_pipeline_metrics() -> PipelineMetrics:
    """Get pipeline metrics"""
    global pipeline_optimizer
    return pipeline_optimizer.get_pipeline_metrics()


if __name__ == "__main__":
    # Test pipeline optimization
    async def test_pipeline():
        print("âš¡ Testing Kimera SWM Pipeline Optimization")
        print("=" * 45)
        
        # Test function
        async def test_task(task_name: str, duration: float = 1.0):
            await asyncio.sleep(duration)
            return f"Result from {task_name}"
        
        # Add test tasks
        add_pipeline_task("task1", test_task, "Task 1", 0.5, priority=TaskPriority.HIGH)
        add_pipeline_task("task2", test_task, "Task 2", 1.0, dependencies={"task1"})
        add_pipeline_task("task3", test_task, "Task 3", 0.3, priority=TaskPriority.MEDIUM)
        
        print("âœ… Added test tasks to pipeline")
        
        # Start pipeline
        start_task = asyncio.create_task(start_pipeline())
        
        # Wait a bit for processing
        await asyncio.sleep(3)
        
        # Stop pipeline
        await stop_pipeline()
        
        # Get metrics
        metrics = get_pipeline_metrics()
        print(f"Tasks completed: {metrics.completed_tasks}")
        print(f"Average duration: {metrics.avg_task_duration:.3f}s")
        print(f"Performance score: {metrics.performance_score:.3f}")
        
        print("\nðŸŽ¯ Pipeline Optimization System Ready!")
    
    asyncio.run(test_pipeline())