# PERFORMANCE OPTIMIZATION PROMPT
## System Performance Enhancement Task

---

### OBJECTIVE
Optimize system performance following Kimera SWM Protocol v3.0 transdisciplinary best practices.

### PROCEDURE

#### 1. Core Optimization Protocol
```markdown
Optimize system performance:
1. Profile all critical paths
2. Identify bottlenecks with data
3. Generate multiple solution approaches:
   - Algorithmic (better big-O)
   - Implementation (better constants)
   - Architectural (better design)
   - Hardware (better utilization)
4. Test each approach in isolation
5. Document trade-offs and rationale
```

#### 2. Profiling Strategy

**CPU Profiling**
```python
import cProfile
import pstats
from memory_profiler import profile
import time

class PerformanceProfiler:
    def __init__(self):
        self.cpu_profiler = cProfile.Profile()
        self.timing_results = {}
        
    def profile_function(self, func, *args, **kwargs):
        """Profile a single function execution"""
        # CPU profiling
        self.cpu_profiler.enable()
        start_time = time.perf_counter()
        
        result = func(*args, **kwargs)
        
        end_time = time.perf_counter()
        self.cpu_profiler.disable()
        
        # Store results
        self.timing_results[func.__name__] = {
            'execution_time': end_time - start_time,
            'cpu_stats': pstats.Stats(self.cpu_profiler),
            'memory_usage': get_memory_usage()
        }
        
        return result
```

**GPU Profiling**
```python
import torch
import numpy as np

def profile_gpu_operation(operation):
    """Profile GPU memory and computation"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        result = operation()
        end_event.record()
        
        torch.cuda.synchronize()
        
        return {
            'gpu_time_ms': start_event.elapsed_time(end_event),
            'peak_memory_mb': torch.cuda.max_memory_allocated() / 1024**2,
            'result': result
        }
```

#### 3. Optimization Approaches

**Algorithmic Optimization**
```yaml
strategies:
  - Replace O(n²) with O(n log n) algorithms
  - Use dynamic programming for overlapping subproblems
  - Implement memoization for expensive computations
  - Apply divide-and-conquer for parallelizable tasks
  
example:
  before: "Nested loops checking all pairs"
  after: "Hash table for O(1) lookups"
  improvement: "100x speedup for n=10000"
```

**Implementation Optimization**
```python
# NumPy vectorization example
# Before: Python loops
def slow_computation(data):
    result = []
    for i in range(len(data)):
        result.append(data[i] ** 2 + 2 * data[i] + 1)
    return result

# After: Vectorized operations
def fast_computation(data):
    data = np.array(data)
    return data ** 2 + 2 * data + 1

# 50-100x speedup for large arrays
```

**Architectural Optimization**
```yaml
patterns:
  - Implement caching layers
  - Use connection pooling
  - Apply lazy loading
  - Implement batch processing
  
example:
  component: "Database queries"
  before: "Individual queries per request"
  after: "Batched queries with Redis cache"
  improvement: "10x throughput increase"
```

**Hardware Optimization**
```python
# CUDA kernel for parallel processing
@cuda.jit
def parallel_computation(input_array, output_array):
    """GPU-accelerated computation"""
    idx = cuda.grid(1)
    if idx < input_array.size:
        # Complex computation here
        output_array[idx] = expensive_function(input_array[idx])

# CPU parallel processing
from multiprocessing import Pool
import os

def optimize_cpu_usage():
    """Use all available CPU cores"""
    num_cores = os.cpu_count()
    with Pool(processes=num_cores) as pool:
        results = pool.map(process_chunk, data_chunks)
    return combine_results(results)
```

#### 4. Benchmark Suite

```python
class KimeraBenchmarkSuite:
    """Comprehensive performance benchmarking"""
    
    def __init__(self):
        self.benchmarks = {}
        
    def add_benchmark(self, name, setup, test, teardown=None):
        """Add a new benchmark to the suite"""
        self.benchmarks[name] = {
            'setup': setup,
            'test': test,
            'teardown': teardown or (lambda: None)
        }
    
    def run_all(self, iterations=100):
        """Run all benchmarks and generate report"""
        results = {}
        
        for name, benchmark in self.benchmarks.items():
            # Setup
            context = benchmark['setup']()
            
            # Warmup
            for _ in range(10):
                benchmark['test'](context)
            
            # Actual benchmark
            times = []
            for _ in range(iterations):
                start = time.perf_counter()
                benchmark['test'](context)
                end = time.perf_counter()
                times.append(end - start)
            
            # Teardown
            benchmark['teardown']()
            
            # Statistics
            results[name] = {
                'mean': np.mean(times),
                'std': np.std(times),
                'min': np.min(times),
                'max': np.max(times),
                'p50': np.percentile(times, 50),
                'p95': np.percentile(times, 95),
                'p99': np.percentile(times, 99)
            }
        
        return results
```

#### 5. Trade-off Analysis Framework

```yaml
optimization_matrix:
  - approach: "Algorithm Change"
    pros:
      - "Fundamental improvement"
      - "Scales better"
    cons:
      - "Development time"
      - "Testing overhead"
    when_to_use:
      - "Current algorithm is bottleneck"
      - "Large data scales expected"
      
  - approach: "Caching"
    pros:
      - "Quick wins"
      - "Minimal code changes"
    cons:
      - "Memory overhead"
      - "Cache invalidation complexity"
    when_to_use:
      - "Repeated computations"
      - "Read-heavy workloads"
      
  - approach: "Parallelization"
    pros:
      - "Linear speedup potential"
      - "Hardware utilization"
    cons:
      - "Synchronization overhead"
      - "Debugging complexity"
    when_to_use:
      - "Independent computations"
      - "CPU/GPU underutilized"
```

#### 6. Performance Report Template

```markdown
# Performance Optimization Report
Date: {date}
Engineer: {name}
System: Kimera SWM v{version}

## Executive Summary
- Overall improvement: {X}% faster
- Memory reduction: {Y}% less
- Cost savings: ${Z}/month

## Bottleneck Analysis
1. **{Component A}**: {time}ms (45% of total)
   - Root cause: {reason}
   - Solution: {approach}
   - Result: {improvement}

## Optimizations Applied
### 1. {Optimization Name}
- **Approach**: {description}
- **Implementation**: {code_reference}
- **Benchmark Results**:
  - Before: {metric}
  - After: {metric}
  - Improvement: {percentage}%
  
## Resource Utilization
- CPU: {before}% → {after}%
- Memory: {before}GB → {after}GB
- GPU: {before}% → {after}%

## Trade-offs Accepted
1. {Trade-off 1}: Chose {A} over {B} because {reason}
2. {Trade-off 2}: Accepted {downside} for {benefit}

## Future Recommendations
1. Consider {optimization} when {condition}
2. Monitor {metric} for degradation
3. Revisit {component} if scale increases

## Appendix: Detailed Metrics
{detailed_benchmark_data}
```

### VALIDATION CRITERIA

Performance optimizations must meet ALL criteria:

1. **Correctness**: All tests still pass
2. **Measurable**: >10% improvement in target metric
3. **Sustainable**: No technical debt introduced
4. **Documented**: Clear explanation of changes
5. **Reversible**: Can rollback if needed

### MONITORING SETUP

```python
# Add performance monitoring
def add_performance_monitoring():
    """Setup continuous performance tracking"""
    metrics = {
        'response_time_p95': Histogram(),
        'throughput': Counter(),
        'error_rate': Gauge(),
        'cpu_usage': Gauge(),
        'memory_usage': Gauge()
    }
    
    # Export to Prometheus/Grafana
    start_http_server(8000)
    
    return metrics
```

---

*Remember: Premature optimization is the root of all evil, but measured optimization is the path to excellence.* 