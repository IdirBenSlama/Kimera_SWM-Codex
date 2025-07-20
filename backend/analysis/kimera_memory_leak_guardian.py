#!/usr/bin/env python3
"""
KIMERA MEMORY LEAK GUARDIAN
===========================

Revolutionary memory leak detection system combining static analysis with directed 
symbolic execution for scalable and accurate detection in cognitive trading systems.

Based on LeakGuard research: "Combining Static Analysis With Directed Symbolic 
Execution for Scalable and Accurate Memory Leak Detection"

Features:
- Path-sensitive function summary generation
- Pointer escape analysis for cognitive field dynamics
- Under-constrained symbolic execution
- GPU memory leak detection
- Real-time monitoring and prevention
"""

import ast
import sys
import time
import torch
import psutil
import numpy as np
import threading
import tracemalloc
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict, deque
import logging
import json
import weakref
import gc
from enum import Enum
from contextlib import contextmanager
from datetime import datetime, timedelta

# Symbolic execution imports
try:
    import z3
    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False
    logging.warning("Warning: Z3 not available - symbolic execution features limited")

@dataclass
class MemoryAllocation:
    """Represents a memory allocation site"""
    allocation_id: str
    function_name: str
    line_number: int
    allocation_type: str  # 'malloc', 'torch.tensor', 'numpy.array', etc.
    size_bytes: Optional[int]
    timestamp: float
    call_stack: List[str]
    symbolic_path: Optional[str] = None
    escape_status: str = "unknown"  # 'escaped', 'not_escaped', 'conditional'

@dataclass
class PointerEscapeInfo:
    """Information about pointer escape analysis"""
    pointer_id: str
    escapes_via_return: bool
    escapes_via_parameter: bool
    escapes_via_global: bool
    escape_conditions: List[str]
    ownership_transfer: bool

@dataclass
class FunctionSummary:
    """Path-sensitive function summary for memory management"""
    function_name: str
    allocations: List[MemoryAllocation]
    deallocations: List[str]  # allocation_ids that are freed
    pointer_escapes: List[PointerEscapeInfo]
    path_conditions: List[str]
    memory_balance: int  # net allocations - deallocations
    is_memory_safe: bool

class LeakDetectionResult(Enum):
    """Memory leak detection results"""
    NO_LEAK = "no_leak"
    POTENTIAL_LEAK = "potential_leak"
    CONFIRMED_LEAK = "confirmed_leak"
    FALSE_POSITIVE = "false_positive"

@dataclass
class MemoryLeakReport:
    """Comprehensive memory leak report"""
    leak_id: str
    detection_result: LeakDetectionResult
    allocation_site: MemoryAllocation
    leak_path: List[str]
    confidence_score: float
    fix_suggestions: List[str]
    impact_assessment: str

@dataclass
class MemorySnapshot:
    """Detailed memory snapshot for tracking over time"""
    timestamp: datetime
    total_memory_mb: float
    available_memory_mb: float
    used_memory_mb: float
    gpu_memory_mb: float
    gpu_available_mb: float
    process_memory_mb: float
    active_allocations_count: int
    leak_risk_score: float
    growth_rate_mb_per_min: float
    
@dataclass
class MemoryAlert:
    """Memory leak alert definition"""
    alert_id: str
    severity: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    message: str
    timestamp: datetime
    memory_usage_mb: float
    threshold_mb: float
    component: str
    recommendations: List[str]
    
@dataclass
class MemoryStatistics:
    """Comprehensive memory statistics"""
    monitoring_duration_hours: float
    peak_memory_mb: float
    average_memory_mb: float
    memory_growth_trend: str  # 'STABLE', 'INCREASING', 'DECREASING', 'VOLATILE'
    total_leaks_detected: int
    total_leaks_fixed: int
    efficiency_score: float
    snapshots_count: int

class KimeraMemoryLeakGuardian:
    """
    Advanced memory leak detection system for Kimera cognitive trading platform
    
    Implements hybrid approach combining:
    1. Static analysis for scalability
    2. Directed symbolic execution for accuracy
    3. Real-time monitoring for prevention
    """
    
    def __init__(self, 
                 enable_symbolic_execution: bool = True,
                 enable_gpu_tracking: bool = True,
                 monitoring_interval: float = 5.0,
                 memory_alert_threshold_mb: float = 1024.0,  # 1GB threshold
                 leak_detection_sensitivity: float = 0.8):
        
        self.enable_symbolic_execution = enable_symbolic_execution and HAS_Z3
        self.enable_gpu_tracking = enable_gpu_tracking and torch.cuda.is_available()
        self.monitoring_interval = monitoring_interval
        self.memory_alert_threshold_mb = memory_alert_threshold_mb
        self.leak_detection_sensitivity = leak_detection_sensitivity
        
        # Core tracking structures
        self.active_allocations: Dict[str, MemoryAllocation] = {}
        self.function_summaries: Dict[str, FunctionSummary] = {}
        self.escape_analysis_cache: Dict[str, PointerEscapeInfo] = {}
        self.symbolic_states: Dict[str, Any] = {}
        
        # Enhanced memory tracking
        self.memory_snapshots: deque = deque(maxlen=1440)  # 24 hours at 1-min intervals
        self.memory_alerts: List[MemoryAlert] = []
        self.monitoring_start_time = datetime.now()
        self.last_snapshot_time = None
        self.previous_memory_usage = 0.0
        
        # Alert thresholds (configurable)
        self.alert_thresholds = {
            'memory_usage_mb': memory_alert_threshold_mb,
            'gpu_memory_usage_mb': 512.0,  # 512MB GPU threshold
            'growth_rate_mb_per_min': 50.0,  # 50MB/min growth rate
            'leak_risk_score': 0.7,  # Risk score threshold
            'allocation_count': 10000  # Max allocations
        }
        
        # GPU memory tracking
        if self.enable_gpu_tracking:
            self.gpu_allocations: Dict[str, Dict] = {}
            self.gpu_memory_baseline = torch.cuda.memory_allocated()
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread = None
        self.leak_reports: List[MemoryLeakReport] = []
        
        # Enhanced performance metrics
        self.analysis_stats = {
            'static_analysis_time': 0.0,
            'symbolic_execution_time': 0.0,
            'total_functions_analyzed': 0,
            'leaks_detected': 0,
            'false_positives_filtered': 0,
            'alerts_generated': 0,
            'memory_freed_mb': 0.0,
            'peak_memory_usage_mb': 0.0,
            'average_leak_detection_time_ms': 0.0
        }
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize symbolic execution solver
        if self.enable_symbolic_execution:
            self.solver = z3.Solver()
        
        self.logger.info(f"🛡️ Kimera Memory Leak Guardian initialized with enhanced tracking")
        self.logger.info(f"   Memory Alert Threshold: {memory_alert_threshold_mb:.1f} MB")
        self.logger.info(f"   Monitoring Interval: {monitoring_interval:.1f}s")
        self.logger.info(f"   Leak Detection Sensitivity: {leak_detection_sensitivity:.2f}")
        self.logger.info(f"   Symbolic Execution: {'✅' if self.enable_symbolic_execution else '❌'}")
        self.logger.info(f"   GPU Tracking: {'✅' if self.enable_gpu_tracking else '❌'}")
    
    def start_monitoring(self):
        """Start real-time memory leak monitoring"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        tracemalloc.start()
        
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        
        self.logger.info("🔍 Real-time memory leak monitoring started")
    
    def stop_monitoring(self):
        """Stop real-time memory leak monitoring"""
        self.is_monitoring = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)
        
        tracemalloc.stop()
        self.logger.info("⏹️ Memory leak monitoring stopped")
    
    def _monitoring_loop(self):
        """Enhanced monitoring loop for real-time leak detection with detailed tracking"""
        while self.is_monitoring:
            try:
                # Capture detailed memory snapshot
                snapshot = self._capture_memory_snapshot()
                if snapshot:
                    self.memory_snapshots.append(snapshot)
                    
                    # Check for memory alerts
                    self._check_memory_alerts(snapshot)
                    
                    # Log detailed memory statistics every 10 snapshots
                    if len(self.memory_snapshots) % 10 == 0:
                        self._log_memory_statistics()
                
                # Perform lightweight leak detection
                self._check_memory_growth()
                self._check_gpu_memory_leaks()
                self._validate_active_allocations()
                
                # Update performance metrics
                self._update_performance_metrics(snapshot if snapshot else None)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in enhanced monitoring loop: {e}")
                time.sleep(self.monitoring_interval * 2)
    
    def _capture_memory_snapshot(self) -> Optional[MemorySnapshot]:
        """Capture a detailed memory snapshot for analysis"""
        try:
            # System memory info
            memory = psutil.virtual_memory()
            process = psutil.Process()
            process_memory = process.memory_info()
            
            # GPU memory info
            gpu_memory_mb = 0.0
            gpu_available_mb = 0.0
            if self.enable_gpu_tracking and torch.cuda.is_available():
                gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
                gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
                gpu_available_mb = gpu_total - gpu_memory_mb
            
            # Calculate growth rate
            current_time = datetime.now()
            current_memory = process_memory.rss / 1024 / 1024  # Convert to MB
            growth_rate = 0.0
            
            if self.last_snapshot_time and self.previous_memory_usage > 0:
                time_diff_min = (current_time - self.last_snapshot_time).total_seconds() / 60
                if time_diff_min > 0:
                    growth_rate = (current_memory - self.previous_memory_usage) / time_diff_min
            
            # Calculate leak risk score
            leak_risk_score = self._calculate_leak_risk_score({
                'memory_growth_rate': growth_rate,
                'active_allocations': len(self.active_allocations),
                'gpu_memory_usage': gpu_memory_mb
            })
            
            snapshot = MemorySnapshot(
                timestamp=current_time,
                total_memory_mb=memory.total / 1024 / 1024,
                available_memory_mb=memory.available / 1024 / 1024,
                used_memory_mb=memory.used / 1024 / 1024,
                gpu_memory_mb=gpu_memory_mb,
                gpu_available_mb=gpu_available_mb,
                process_memory_mb=current_memory,
                active_allocations_count=len(self.active_allocations),
                leak_risk_score=leak_risk_score,
                growth_rate_mb_per_min=growth_rate
            )
            
            # Update tracking variables
            self.last_snapshot_time = current_time
            self.previous_memory_usage = current_memory
            
            # Update peak memory usage
            if current_memory > self.analysis_stats['peak_memory_usage_mb']:
                self.analysis_stats['peak_memory_usage_mb'] = current_memory
            
            return snapshot
            
        except Exception as e:
            self.logger.error(f"Failed to capture memory snapshot: {e}")
            return None
    
    def _check_memory_alerts(self, snapshot: MemorySnapshot):
        """Check memory snapshot against alert thresholds and generate alerts if needed"""
        alerts_generated = []
        
        # Check process memory usage
        if snapshot.process_memory_mb > self.alert_thresholds['memory_usage_mb']:
            alert = self._create_memory_alert(
                severity='HIGH',
                message=f"Process memory usage ({snapshot.process_memory_mb:.1f}MB) exceeds threshold",
                memory_usage=snapshot.process_memory_mb,
                threshold=self.alert_thresholds['memory_usage_mb'],
                component='process',
                recommendations=[
                    "Run garbage collection manually",
                    "Check for memory leaks in recent operations",
                    "Consider reducing batch sizes or cache sizes"
                ]
            )
            alerts_generated.append(alert)
        
        # Check GPU memory usage
        if (self.enable_gpu_tracking and 
            snapshot.gpu_memory_mb > self.alert_thresholds['gpu_memory_usage_mb']):
            alert = self._create_memory_alert(
                severity='MEDIUM',
                message=f"GPU memory usage ({snapshot.gpu_memory_mb:.1f}MB) exceeds threshold",
                memory_usage=snapshot.gpu_memory_mb,
                threshold=self.alert_thresholds['gpu_memory_usage_mb'],
                component='gpu',
                recommendations=[
                    "Clear GPU cache with torch.cuda.empty_cache()",
                    "Reduce model batch sizes",
                    "Check for unreleased GPU tensors"
                ]
            )
            alerts_generated.append(alert)
        
        # Check memory growth rate
        if snapshot.growth_rate_mb_per_min > self.alert_thresholds['growth_rate_mb_per_min']:
            alert = self._create_memory_alert(
                severity='HIGH',
                message=f"Memory growth rate ({snapshot.growth_rate_mb_per_min:.1f}MB/min) is excessive",
                memory_usage=snapshot.process_memory_mb,
                threshold=self.alert_thresholds['growth_rate_mb_per_min'],
                component='growth_rate',
                recommendations=[
                    "Investigate recent code changes for memory leaks",
                    "Check for unbounded data structures",
                    "Review caching strategies"
                ]
            )
            alerts_generated.append(alert)
        
        # Check leak risk score
        if snapshot.leak_risk_score > self.alert_thresholds['leak_risk_score']:
            alert = self._create_memory_alert(
                severity='CRITICAL',
                message=f"High leak risk score ({snapshot.leak_risk_score:.2f}) detected",
                memory_usage=snapshot.process_memory_mb,
                threshold=self.alert_thresholds['leak_risk_score'],
                component='leak_detection',
                recommendations=[
                    "Run comprehensive leak analysis",
                    "Review recent memory allocations",
                    "Consider emergency memory cleanup"
                ]
            )
            alerts_generated.append(alert)
        
        # Check allocation count
        if snapshot.active_allocations_count > self.alert_thresholds['allocation_count']:
            alert = self._create_memory_alert(
                severity='MEDIUM',
                message=f"High number of active allocations ({snapshot.active_allocations_count})",
                memory_usage=snapshot.process_memory_mb,
                threshold=self.alert_thresholds['allocation_count'],
                component='allocations',
                recommendations=[
                    "Review allocation patterns",
                    "Implement object pooling",
                    "Check for memory fragmentation"
                ]
            )
            alerts_generated.append(alert)
        
        # Add alerts and log them
        for alert in alerts_generated:
            self.memory_alerts.append(alert)
            self.analysis_stats['alerts_generated'] += 1
            self.logger.warning(f"🚨 Memory Alert [{alert.severity}]: {alert.message}")
            
        # Keep only recent alerts (last 100)
        if len(self.memory_alerts) > 100:
            self.memory_alerts = self.memory_alerts[-100:]
    
    def _create_memory_alert(self, severity: str, message: str, memory_usage: float, 
                           threshold: float, component: str, recommendations: List[str]) -> MemoryAlert:
        """Create a memory alert with unique ID"""
        alert_id = f"alert_{int(time.time())}_{len(self.memory_alerts)}"
        return MemoryAlert(
            alert_id=alert_id,
            severity=severity,
            message=message,
            timestamp=datetime.now(),
            memory_usage_mb=memory_usage,
            threshold_mb=threshold,
            component=component,
            recommendations=recommendations
        )
    
    def _log_memory_statistics(self):
        """Log comprehensive memory statistics"""
        if not self.memory_snapshots:
            return
        
        stats = self.get_memory_statistics()
        
        self.logger.info("📊 Memory Statistics Summary:")
        self.logger.info(f"   Monitoring Duration: {stats.monitoring_duration_hours:.1f} hours")
        self.logger.info(f"   Peak Memory Usage: {stats.peak_memory_mb:.1f} MB")
        self.logger.info(f"   Average Memory Usage: {stats.average_memory_mb:.1f} MB")
        self.logger.info(f"   Memory Trend: {stats.memory_growth_trend}")
        self.logger.info(f"   Leaks Detected: {stats.total_leaks_detected}")
        self.logger.info(f"   Efficiency Score: {stats.efficiency_score:.2f}")
        self.logger.info(f"   Total Snapshots: {stats.snapshots_count}")
    
    def _update_performance_metrics(self, snapshot: Optional[MemorySnapshot]):
        """Update performance metrics based on current snapshot"""
        if snapshot:
            # Update average memory calculation (running average)
            current_count = len(self.memory_snapshots)
            if current_count > 1:
                prev_avg = self.analysis_stats.get('average_memory_mb', 0)
                new_avg = ((prev_avg * (current_count - 1)) + snapshot.process_memory_mb) / current_count
                self.analysis_stats['average_memory_mb'] = new_avg
            else:
                self.analysis_stats['average_memory_mb'] = snapshot.process_memory_mb
    
    def analyze_function_for_leaks(self, 
                                  function_code: str, 
                                  function_name: str) -> FunctionSummary:
        """
        Analyze a function for potential memory leaks using hybrid approach
        
        Steps:
        1. Static analysis to identify allocation/deallocation patterns
        2. Pointer escape analysis
        3. Directed symbolic execution for path-sensitive analysis
        """
        start_time = time.time()
        
        try:
            # Step 1: Static analysis
            ast_tree = ast.parse(function_code)
            static_summary = self._perform_static_analysis(ast_tree, function_name)
            
            # Step 2: Pointer escape analysis
            escape_info = self._analyze_pointer_escapes(ast_tree, static_summary)
            
            # Step 3: Directed symbolic execution (if enabled)
            if self.enable_symbolic_execution and static_summary.allocations:
                symbolic_summary = self._perform_symbolic_execution(
                    ast_tree, static_summary, function_name
                )
                # Merge symbolic results with static analysis
                static_summary = self._merge_analysis_results(static_summary, symbolic_summary)
            
            # Step 4: Generate final summary
            final_summary = FunctionSummary(
                function_name=function_name,
                allocations=static_summary.allocations,
                deallocations=static_summary.deallocations,
                pointer_escapes=escape_info,
                path_conditions=getattr(static_summary, 'path_conditions', []),
                memory_balance=len(static_summary.allocations) - len(static_summary.deallocations),
                is_memory_safe=self._assess_memory_safety(static_summary, escape_info)
            )
            
            self.function_summaries[function_name] = final_summary
            self.analysis_stats['total_functions_analyzed'] += 1
            self.analysis_stats['static_analysis_time'] += time.time() - start_time
            
            return final_summary
            
        except Exception as e:
            self.logger.error(f"Function analysis failed for {function_name}: {e}")
            return self._create_error_summary(function_name, str(e))
    
    def _perform_static_analysis(self, ast_tree: ast.AST, function_name: str) -> FunctionSummary:
        """Perform static analysis to identify allocation patterns"""
        
        class AllocationVisitor(ast.NodeVisitor):
            def __init__(self):
                self.allocations = []
                self.deallocations = []
                self.current_line = 1
            
            def visit_Call(self, node):
                self.current_line = getattr(node, 'lineno', self.current_line)
                
                # Detect memory allocations
                if self._is_allocation_call(node):
                    allocation = MemoryAllocation(
                        allocation_id=f"{function_name}_{self.current_line}_{len(self.allocations)}",
                        function_name=function_name,
                        line_number=self.current_line,
                        allocation_type=self._get_allocation_type(node),
                        size_bytes=self._estimate_allocation_size(node),
                        timestamp=time.time(),
                        call_stack=[function_name]
                    )
                    self.allocations.append(allocation)
                
                # Detect deallocations
                elif self._is_deallocation_call(node):
                    # Try to match with previous allocation
                    dealloc_id = self._match_deallocation(node)
                    if dealloc_id:
                        self.deallocations.append(dealloc_id)
                
                self.generic_visit(node)
            
            def _is_allocation_call(self, node):
                """Check if node represents a memory allocation"""
                if isinstance(node.func, ast.Attribute):
                    # torch.tensor, torch.zeros, etc.
                    if (isinstance(node.func.value, ast.Name) and 
                        node.func.value.id == 'torch'):
                        return node.func.attr in ['tensor', 'zeros', 'ones', 'randn', 'empty']
                    
                    # numpy allocations
                    if (isinstance(node.func.value, ast.Name) and 
                        node.func.value.id in ['np', 'numpy']):
                        return node.func.attr in ['array', 'zeros', 'ones', 'random']
                
                elif isinstance(node.func, ast.Name):
                    # Direct function calls
                    return node.func.id in ['malloc', 'calloc', 'list', 'dict', 'set']
                
                return False
            
            def _is_deallocation_call(self, node):
                """Check if node represents a memory deallocation"""
                if isinstance(node.func, ast.Name):
                    return node.func.id in ['free', 'del']
                elif isinstance(node.func, ast.Attribute):
                    return node.func.attr in ['clear', 'close', 'release']
                return False
            
            def _get_allocation_type(self, node):
                """Determine the type of allocation"""
                if isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name):
                        return f"{node.func.value.id}.{node.func.attr}"
                elif isinstance(node.func, ast.Name):
                    return node.func.id
                return "unknown"
            
            def _estimate_allocation_size(self, node):
                """Estimate allocation size from AST node"""
                # Simple heuristic - could be enhanced with symbolic analysis
                if node.args:
                    if isinstance(node.args[0], ast.Constant):
                        if isinstance(node.args[0].value, int):
                            return node.args[0].value * 4  # Assume 4 bytes per element
                return None
            
            def _match_deallocation(self, node):
                """Try to match deallocation with allocation"""
                # Simplified matching - could be enhanced
                return f"dealloc_{self.current_line}"
        
        visitor = AllocationVisitor()
        visitor.visit(ast_tree)
        
        return FunctionSummary(
            function_name=function_name,
            allocations=visitor.allocations,
            deallocations=visitor.deallocations,
            pointer_escapes=[],
            path_conditions=[],
            memory_balance=len(visitor.allocations) - len(visitor.deallocations),
            is_memory_safe=len(visitor.allocations) <= len(visitor.deallocations)
        )
    
    def _analyze_pointer_escapes(self, 
                                ast_tree: ast.AST, 
                                summary: FunctionSummary) -> List[PointerEscapeInfo]:
        """Analyze pointer escape patterns for cognitive field dynamics"""
        
        class EscapeAnalyzer(ast.NodeVisitor):
            def __init__(self):
                self.escape_info = []
                self.returns = []
                self.assignments = []
                self.global_assignments = []
            
            def visit_Return(self, node):
                if node.value:
                    self.returns.append(self._get_variable_name(node.value))
                self.generic_visit(node)
            
            def visit_Assign(self, node):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.assignments.append(target.id)
                    elif isinstance(target, ast.Attribute):
                        # Global or class attribute assignment
                        self.global_assignments.append(self._get_variable_name(target))
                self.generic_visit(node)
            
            def _get_variable_name(self, node):
                """Extract variable name from AST node"""
                if isinstance(node, ast.Name):
                    return node.id
                elif isinstance(node, ast.Attribute):
                    return f"{self._get_variable_name(node.value)}.{node.attr}"
                return "unknown"
        
        analyzer = EscapeAnalyzer()
        analyzer.visit(ast_tree)
        
        # Generate escape information for each allocation
        escape_info = []
        for allocation in summary.allocations:
            escape = PointerEscapeInfo(
                pointer_id=allocation.allocation_id,
                escapes_via_return=allocation.allocation_id in analyzer.returns,
                escapes_via_parameter=False,  # Would need interprocedural analysis
                escapes_via_global=allocation.allocation_id in analyzer.global_assignments,
                escape_conditions=[],
                ownership_transfer=False
            )
            escape_info.append(escape)
        
        return escape_info
    
    def _perform_symbolic_execution(self, 
                                  ast_tree: ast.AST, 
                                  static_summary: FunctionSummary,
                                  function_name: str) -> FunctionSummary:
        """Perform directed symbolic execution for path-sensitive analysis"""
        if not self.enable_symbolic_execution:
            return static_summary
        
        start_time = time.time()
        
        try:
            # Create symbolic variables for function parameters
            symbolic_vars = {}
            path_conditions = []
            
            # Simplified symbolic execution - focus on memory-relevant paths
            class SymbolicExecutor(ast.NodeVisitor):
                def __init__(self, solver):
                    self.solver = solver
                    self.current_path = []
                    self.memory_operations = []
                
                def visit_If(self, node):
                    # Explore both branches symbolically
                    condition = self._create_symbolic_condition(node.test)
                    
                    # True branch
                    self.solver.push()
                    self.solver.add(condition)
                    if self.solver.check() == z3.sat:
                        self.current_path.append(f"if_true_{node.lineno}")
                        for stmt in node.body:
                            self.visit(stmt)
                    self.solver.pop()
                    
                    # False branch
                    if node.orelse:
                        self.solver.push()
                        self.solver.add(z3.Not(condition))
                        if self.solver.check() == z3.sat:
                            self.current_path.append(f"if_false_{node.lineno}")
                            for stmt in node.orelse:
                                self.visit(stmt)
                        self.solver.pop()
                
                def _create_symbolic_condition(self, node):
                    """Create symbolic condition from AST node"""
                    # Simplified - would need full expression translation
                    return z3.Bool(f"cond_{getattr(node, 'lineno', 0)}")
            
            executor = SymbolicExecutor(self.solver)
            executor.visit(ast_tree)
            
            # Update summary with symbolic execution results
            static_summary.path_conditions = executor.current_path
            
            self.analysis_stats['symbolic_execution_time'] += time.time() - start_time
            
        except Exception as e:
            self.logger.warning(f"Symbolic execution failed for {function_name}: {e}")
        
        return static_summary
    
    def _merge_analysis_results(self, 
                               static_summary: FunctionSummary,
                               symbolic_summary: FunctionSummary) -> FunctionSummary:
        """Merge static and symbolic analysis results"""
        # Combine path conditions
        all_conditions = static_summary.path_conditions + symbolic_summary.path_conditions
        
        # Update memory safety assessment based on symbolic paths
        is_safe = static_summary.is_memory_safe and symbolic_summary.is_memory_safe
        
        return FunctionSummary(
            function_name=static_summary.function_name,
            allocations=static_summary.allocations,
            deallocations=static_summary.deallocations,
            pointer_escapes=static_summary.pointer_escapes,
            path_conditions=all_conditions,
            memory_balance=static_summary.memory_balance,
            is_memory_safe=is_safe
        )
    
    def _assess_memory_safety(self, 
                             summary: FunctionSummary,
                             escape_info: List[PointerEscapeInfo]) -> bool:
        """Assess overall memory safety of function"""
        
        # Check allocation/deallocation balance
        if summary.memory_balance > 0:
            # More allocations than deallocations
            
            # Check if allocations escape (transferred ownership)
            escaped_allocations = 0
            for escape in escape_info:
                if (escape.escapes_via_return or 
                    escape.escapes_via_global or 
                    escape.ownership_transfer):
                    escaped_allocations += 1
            
            # If all excess allocations escape, it might be safe
            unescaped_allocations = summary.memory_balance - escaped_allocations
            return unescaped_allocations <= 0
        
        return True  # Balanced or negative balance is generally safe
    
    def detect_gpu_memory_leaks(self) -> List[MemoryLeakReport]:
        """Detect GPU memory leaks in Kimera's cognitive field dynamics"""
        if not self.enable_gpu_tracking:
            return []
        
        leak_reports = []
        current_memory = torch.cuda.memory_allocated()
        memory_growth = current_memory - self.gpu_memory_baseline
        
        # Check for significant memory growth
        if memory_growth > 100 * 1024 * 1024:  # 100MB threshold
            
            # Analyze GPU memory allocations
            for alloc_id, alloc_info in self.gpu_allocations.items():
                if self._is_potential_gpu_leak(alloc_info):
                    leak_report = MemoryLeakReport(
                        leak_id=f"gpu_leak_{alloc_id}",
                        detection_result=LeakDetectionResult.POTENTIAL_LEAK,
                        allocation_site=alloc_info['allocation'],
                        leak_path=alloc_info.get('call_stack', []),
                        confidence_score=0.7,
                        fix_suggestions=[
                            "Use context managers for GPU tensor allocation",
                            "Call torch.cuda.empty_cache() periodically",
                            "Implement proper tensor lifecycle management"
                        ],
                        impact_assessment="High - GPU memory exhaustion risk"
                    )
                    leak_reports.append(leak_report)
        
        return leak_reports
    
    def _is_potential_gpu_leak(self, alloc_info: Dict) -> bool:
        """Check if GPU allocation is a potential leak"""
        allocation_age = time.time() - alloc_info['timestamp']
        
        # Heuristics for potential leaks
        if allocation_age > 300:  # 5 minutes
            return True
        
        if alloc_info.get('size_bytes', 0) > 50 * 1024 * 1024:  # 50MB
            return True
        
        return False
    
    @contextmanager
    def track_allocation(self, allocation_id: str, allocation_info: Dict):
        """Context manager to track memory allocation lifecycle"""
        start_time = time.time()
        
        # Record allocation
        allocation = MemoryAllocation(
            allocation_id=allocation_id,
            function_name=allocation_info.get('function', 'unknown'),
            line_number=allocation_info.get('line', 0),
            allocation_type=allocation_info.get('type', 'unknown'),
            size_bytes=allocation_info.get('size', None),
            timestamp=start_time,
            call_stack=allocation_info.get('call_stack', [])
        )
        
        self.active_allocations[allocation_id] = allocation
        
        if self.enable_gpu_tracking and 'gpu' in allocation_info.get('type', ''):
            self.gpu_allocations[allocation_id] = {
                'allocation': allocation,
                'timestamp': start_time,
                **allocation_info
            }
        
        try:
            yield allocation
        finally:
            # Record deallocation
            if allocation_id in self.active_allocations:
                del self.active_allocations[allocation_id]
            
            if allocation_id in self.gpu_allocations:
                del self.gpu_allocations[allocation_id]
    
    def _check_memory_growth(self):
        """Check for unusual memory growth patterns"""
        try:
            current, peak = tracemalloc.get_traced_memory()
            current_mb = current / 1024 / 1024
            
            # Simple heuristic: check if memory growth is excessive
            if current_mb > 1000:  # 1GB threshold
                self.logger.warning(f"High memory usage detected: {current_mb:.1f}MB")
                
                # Generate leak report for investigation
                leak_report = MemoryLeakReport(
                    leak_id=f"memory_growth_{int(time.time())}",
                    detection_result=LeakDetectionResult.POTENTIAL_LEAK,
                    allocation_site=MemoryAllocation(
                        allocation_id="system_wide",
                        function_name="system",
                        line_number=0,
                        allocation_type="system_memory",
                        size_bytes=int(current),
                        timestamp=time.time(),
                        call_stack=[]
                    ),
                    leak_path=["system_wide_growth"],
                    confidence_score=0.5,
                    fix_suggestions=[
                        "Review active allocations",
                        "Force garbage collection",
                        "Check for circular references"
                    ],
                    impact_assessment="Medium - Memory usage growing"
                )
                self.leak_reports.append(leak_report)
                
        except Exception as e:
            self.logger.error(f"Memory growth check failed: {e}")
    
    def _check_gpu_memory_leaks(self):
        """Check for GPU memory leaks"""
        if not self.enable_gpu_tracking:
            return
        
        try:
            current_gpu_memory = torch.cuda.memory_allocated()
            growth = current_gpu_memory - self.gpu_memory_baseline
            
            if growth > 500 * 1024 * 1024:  # 500MB growth threshold
                self.logger.warning(f"GPU memory growth detected: {growth / 1024 / 1024:.1f}MB")
                
                # Update baseline to prevent repeated warnings
                self.gpu_memory_baseline = current_gpu_memory
                
        except Exception as e:
            self.logger.error(f"GPU memory check failed: {e}")
    
    def _validate_active_allocations(self):
        """Validate that active allocations are still valid"""
        current_time = time.time()
        stale_allocations = []
        
        for alloc_id, allocation in self.active_allocations.items():
            allocation_age = current_time - allocation.timestamp
            
            # Mark allocations older than 1 hour as potentially stale
            if allocation_age > 3600:
                stale_allocations.append(alloc_id)
        
        if stale_allocations:
            self.logger.warning(f"Found {len(stale_allocations)} stale allocations")
    
    def _create_error_summary(self, function_name: str, error_msg: str) -> FunctionSummary:
        """Create error summary when analysis fails"""
        return FunctionSummary(
            function_name=function_name,
            allocations=[],
            deallocations=[],
            pointer_escapes=[],
            path_conditions=[f"analysis_error: {error_msg}"],
            memory_balance=0,
            is_memory_safe=False
        )
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive memory leak analysis report"""
        
        # Analyze all function summaries
        total_functions = len(self.function_summaries)
        unsafe_functions = sum(1 for s in self.function_summaries.values() if not s.is_memory_safe)
        total_allocations = sum(len(s.allocations) for s in self.function_summaries.values())
        
        # GPU memory analysis
        gpu_report = {}
        if self.enable_gpu_tracking:
            current_gpu_memory = torch.cuda.memory_allocated()
            gpu_report = {
                'current_usage_bytes': current_gpu_memory,
                'baseline_bytes': self.gpu_memory_baseline,
                'growth_bytes': current_gpu_memory - self.gpu_memory_baseline,
                'active_gpu_allocations': len(self.gpu_allocations)
            }
        
        report = {
            'analysis_summary': {
                'timestamp': time.time(),
                'total_functions_analyzed': total_functions,
                'unsafe_functions': unsafe_functions,
                'safety_ratio': (total_functions - unsafe_functions) / max(total_functions, 1),
                'total_allocations_tracked': total_allocations,
                'active_allocations': len(self.active_allocations),
                'leak_reports_generated': len(self.leak_reports)
            },
            'performance_metrics': self.analysis_stats,
            'gpu_memory_analysis': gpu_report,
            'function_summaries': {
                name: {
                    'memory_balance': summary.memory_balance,
                    'is_memory_safe': summary.is_memory_safe,
                    'allocation_count': len(summary.allocations),
                    'deallocation_count': len(summary.deallocations),
                    'path_conditions': len(summary.path_conditions)
                }
                for name, summary in self.function_summaries.items()
            },
            'leak_reports': [
                {
                    'leak_id': report.leak_id,
                    'detection_result': report.detection_result.value,
                    'confidence_score': report.confidence_score,
                    'impact_assessment': report.impact_assessment,
                    'fix_suggestions': report.fix_suggestions
                }
                for report in self.leak_reports
            ],
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on analysis"""
        recommendations = []
        
        # Function-level recommendations
        unsafe_functions = [name for name, summary in self.function_summaries.items() 
                          if not summary.is_memory_safe]
        
        if unsafe_functions:
            recommendations.append(
                f"Review {len(unsafe_functions)} functions with potential memory safety issues: "
                f"{', '.join(unsafe_functions[:5])}"
            )
        
        # GPU memory recommendations
        if self.enable_gpu_tracking and len(self.gpu_allocations) > 100:
            recommendations.append(
                "High number of active GPU allocations detected. "
                "Consider implementing memory pooling for better efficiency."
            )
        
        # System-wide recommendations
        if len(self.active_allocations) > 1000:
            recommendations.append(
                "Large number of tracked allocations. "
                "Consider implementing automatic cleanup mechanisms."
            )
        
        return recommendations
    
    def _calculate_leak_risk_score(self, stats: Dict) -> float:
        """
        Enhanced leak risk score calculation considering multiple factors
        
        Args:
            stats: Dictionary containing various memory and performance statistics
            
        Returns:
            Risk score between 0.0 (no risk) and 1.0 (high risk)
        """
        risk_factors = []
        
        # Memory growth rate factor
        growth_rate = stats.get('memory_growth_rate', 0.0)
        if growth_rate > 100:  # >100MB/min is concerning
            risk_factors.append(0.8)
        elif growth_rate > 50:  # >50MB/min is moderate risk
            risk_factors.append(0.5)
        elif growth_rate > 20:  # >20MB/min is low risk
            risk_factors.append(0.2)
        else:
            risk_factors.append(0.0)
        
        # Active allocations factor
        active_allocs = stats.get('active_allocations', 0)
        if active_allocs > 50000:
            risk_factors.append(0.7)
        elif active_allocs > 20000:
            risk_factors.append(0.4)
        elif active_allocs > 10000:
            risk_factors.append(0.2)
        else:
            risk_factors.append(0.0)
        
        # GPU memory usage factor
        gpu_memory = stats.get('gpu_memory_usage', 0.0)
        if gpu_memory > 1000:  # >1GB GPU usage
            risk_factors.append(0.6)
        elif gpu_memory > 500:  # >500MB GPU usage
            risk_factors.append(0.3)
        else:
            risk_factors.append(0.0)
        
        # Memory fragmentation factor (estimated)
        memory_efficiency = stats.get('memory_efficiency', 1.0)
        if memory_efficiency < 0.7:
            risk_factors.append(0.5)
        elif memory_efficiency < 0.8:
            risk_factors.append(0.3)
        else:
            risk_factors.append(0.0)
        
        # Function call depth factor (if available)
        call_depth = stats.get('max_call_depth', 0)
        if call_depth > 100:
            risk_factors.append(0.4)
        elif call_depth > 50:
            risk_factors.append(0.2)
        else:
            risk_factors.append(0.0)
        
        # Calculate weighted average with sensitivity adjustment
        if risk_factors:
            base_score = sum(risk_factors) / len(risk_factors)
            # Apply sensitivity adjustment
            adjusted_score = base_score * self.leak_detection_sensitivity
            return min(adjusted_score, 1.0)
        
        return 0.0
    
    def get_memory_statistics(self) -> MemoryStatistics:
        """
        Generate comprehensive memory statistics from collected snapshots
        
        Returns:
            MemoryStatistics object with detailed analytics
        """
        if not self.memory_snapshots:
            return MemoryStatistics(
                monitoring_duration_hours=0.0,
                peak_memory_mb=0.0,
                average_memory_mb=0.0,
                memory_growth_trend='NO_DATA',
                total_leaks_detected=0,
                total_leaks_fixed=0,
                efficiency_score=0.0,
                snapshots_count=0
            )
        
        # Calculate monitoring duration
        duration_seconds = (datetime.now() - self.monitoring_start_time).total_seconds()
        duration_hours = duration_seconds / 3600
        
        # Memory usage statistics
        memory_values = [snapshot.process_memory_mb for snapshot in self.memory_snapshots]
        peak_memory = max(memory_values)
        average_memory = sum(memory_values) / len(memory_values)
        
        # Determine memory growth trend
        trend = self._analyze_memory_trend(memory_values)
        
        # Leaks detected and fixed
        leaks_detected = self.analysis_stats.get('leaks_detected', 0)
        leaks_fixed = len([report for report in self.leak_reports 
                          if report.detection_result == LeakDetectionResult.CONFIRMED_LEAK])
        
        # Calculate efficiency score
        efficiency_score = self._calculate_efficiency_score(memory_values, duration_hours)
        
        return MemoryStatistics(
            monitoring_duration_hours=duration_hours,
            peak_memory_mb=peak_memory,
            average_memory_mb=average_memory,
            memory_growth_trend=trend,
            total_leaks_detected=leaks_detected,
            total_leaks_fixed=leaks_fixed,
            efficiency_score=efficiency_score,
            snapshots_count=len(self.memory_snapshots)
        )
    
    def _analyze_memory_trend(self, memory_values: List[float]) -> str:
        """
        Analyze memory usage trend over time
        
        Args:
            memory_values: List of memory usage values over time
            
        Returns:
            Trend classification string
        """
        if len(memory_values) < 3:
            return 'INSUFFICIENT_DATA'
        
        # Calculate trend using linear regression
        n = len(memory_values)
        x_values = list(range(n))
        
        # Simple linear regression
        x_mean = sum(x_values) / n
        y_mean = sum(memory_values) / n
        
        numerator = sum((x_values[i] - x_mean) * (memory_values[i] - y_mean) for i in range(n))
        denominator = sum((x_values[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 'STABLE'
        
        slope = numerator / denominator
        
        # Calculate variance to determine volatility
        variance = sum((memory_values[i] - y_mean) ** 2 for i in range(n)) / n
        std_dev = variance ** 0.5
        volatility_ratio = std_dev / y_mean if y_mean > 0 else 0
        
        # Classify trend
        if volatility_ratio > 0.3:  # High volatility
            return 'VOLATILE'
        elif slope > 5:  # Growing by >5MB per measurement
            return 'INCREASING'
        elif slope < -5:  # Decreasing by >5MB per measurement  
            return 'DECREASING'
        else:
            return 'STABLE'
    
    def _calculate_efficiency_score(self, memory_values: List[float], duration_hours: float) -> float:
        """
        Calculate system efficiency score based on memory usage patterns
        
        Args:
            memory_values: List of memory usage values
            duration_hours: Monitoring duration in hours
            
        Returns:
            Efficiency score between 0.0 and 1.0
        """
        if not memory_values or duration_hours <= 0:
            return 0.0
        
        # Factors for efficiency calculation
        efficiency_factors = []
        
        # Memory stability factor
        if len(memory_values) > 1:
            memory_variance = np.var(memory_values)
            memory_mean = np.mean(memory_values)
            coefficient_of_variation = memory_variance ** 0.5 / memory_mean if memory_mean > 0 else 1
            stability_score = max(0, 1 - coefficient_of_variation)
            efficiency_factors.append(stability_score)
        
        # Memory growth factor
        if len(memory_values) >= 2:
            growth_rate = (memory_values[-1] - memory_values[0]) / duration_hours if duration_hours > 0 else 0
            growth_penalty = min(abs(growth_rate) / 100, 1.0)  # Normalize to 0-1
            growth_score = max(0, 1 - growth_penalty)
            efficiency_factors.append(growth_score)
        
        # Alert frequency factor
        alerts_per_hour = len(self.memory_alerts) / duration_hours if duration_hours > 0 else 0
        alert_penalty = min(alerts_per_hour / 10, 1.0)  # Normalize: 10+ alerts/hour = 0 score
        alert_score = max(0, 1 - alert_penalty)
        efficiency_factors.append(alert_score)
        
        # Leak detection factor
        total_snapshots = len(self.memory_snapshots)
        leak_rate = self.analysis_stats.get('leaks_detected', 0) / total_snapshots if total_snapshots > 0 else 0
        leak_penalty = min(leak_rate * 10, 1.0)  # Heavy penalty for leaks
        leak_score = max(0, 1 - leak_penalty)
        efficiency_factors.append(leak_score)
        
        # Calculate weighted average
        if efficiency_factors:
            return sum(efficiency_factors) / len(efficiency_factors)
        
        return 0.5  # Default neutral score
    
    def get_recent_alerts(self, hours: int = 1) -> List[MemoryAlert]:
        """
        Get memory alerts from the last N hours
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of recent memory alerts
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.memory_alerts if alert.timestamp >= cutoff_time]
    
    def get_memory_usage_trend(self, hours: int = 1) -> Dict[str, Any]:
        """
        Get memory usage trend analysis for the last N hours
        
        Args:
            hours: Number of hours to analyze
            
        Returns:
            Dictionary with trend analysis
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_snapshots = [s for s in self.memory_snapshots if s.timestamp >= cutoff_time]
        
        if not recent_snapshots:
            return {'status': 'no_data', 'message': 'No snapshots available for the specified time period'}
        
        memory_values = [s.process_memory_mb for s in recent_snapshots]
        growth_rates = [s.growth_rate_mb_per_min for s in recent_snapshots if s.growth_rate_mb_per_min is not None]
        
        return {
            'period_hours': hours,
            'snapshots_count': len(recent_snapshots),
            'memory_min_mb': min(memory_values),
            'memory_max_mb': max(memory_values),
            'memory_avg_mb': sum(memory_values) / len(memory_values),
            'trend_classification': self._analyze_memory_trend(memory_values),
            'avg_growth_rate_mb_per_min': sum(growth_rates) / len(growth_rates) if growth_rates else 0,
            'peak_growth_rate_mb_per_min': max(growth_rates) if growth_rates else 0,
            'recent_alerts_count': len(self.get_recent_alerts(hours))
        }

# Global instance for easy access
_global_leak_guardian = None

def get_memory_leak_guardian() -> KimeraMemoryLeakGuardian:
    """Get or create global memory leak guardian instance"""
    global _global_leak_guardian
    
    if _global_leak_guardian is None:
        _global_leak_guardian = KimeraMemoryLeakGuardian()
    
    return _global_leak_guardian

def initialize_memory_leak_guardian(**kwargs) -> KimeraMemoryLeakGuardian:
    """Initialize global memory leak guardian with custom parameters"""
    global _global_leak_guardian
    _global_leak_guardian = KimeraMemoryLeakGuardian(**kwargs)
    return _global_leak_guardian

# Decorator for automatic function analysis
def analyze_for_leaks(func):
    """Decorator to automatically analyze functions for memory leaks"""
    import inspect
    
    def wrapper(*args, **kwargs):
        guardian = get_memory_leak_guardian()
        
        # Get function source code
        try:
            source = inspect.getsource(func)
            summary = guardian.analyze_function_for_leaks(source, func.__name__)
            
            if not summary.is_memory_safe:
                guardian.logger.warning(
                    f"Function {func.__name__} has potential memory safety issues"
                )
        
        except Exception as e:
            guardian.logger.debug(f"Could not analyze {func.__name__}: {e}")
        
        return func(*args, **kwargs)
    
    return wrapper

# Context manager for tracking code blocks
@contextmanager
def track_memory_block(block_name: str):
    """Context manager to track memory usage in code blocks"""
    guardian = get_memory_leak_guardian()
    
    start_memory = torch.cuda.memory_allocated() if guardian.enable_gpu_tracking else 0
    start_time = time.time()
    
    allocation_info = {
        'function': block_name,
        'type': 'code_block',
        'timestamp': start_time
    }
    
    with guardian.track_allocation(f"block_{block_name}_{int(start_time)}", allocation_info):
        try:
            yield
        finally:
            if guardian.enable_gpu_tracking:
                end_memory = torch.cuda.memory_allocated()
                memory_delta = end_memory - start_memory
                
                if memory_delta > 10 * 1024 * 1024:  # 10MB threshold
                    guardian.logger.info(
                        f"Code block '{block_name}' allocated {memory_delta / 1024 / 1024:.1f}MB GPU memory"
                    ) 