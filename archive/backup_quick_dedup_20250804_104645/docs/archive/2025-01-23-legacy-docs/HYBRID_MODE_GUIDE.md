# Kimera Hybrid Mode Guide

## Overview

The Kimera Hybrid Mode combines the best of both worlds: **sub-millisecond performance** with **comprehensive debugging capabilities**. This guide explains how to use the hybrid system effectively.

## Quick Start

### Basic Usage

```bash
# High performance mode (default)
python kimera_hybrid.py

# Debug mode with performance optimizations
python kimera_hybrid.py --debug

# Full debug mode (comprehensive logging, slower)
python kimera_hybrid.py --debug --no-performance

# Custom port and host
python kimera_hybrid.py --port 8080 --host localhost
```

### Environment Variables

```bash
# Enable debug mode
export KIMERA_DEBUG=true

# Disable performance optimizations
export KIMERA_PERFORMANCE=false

# Run with environment configuration
python kimera_hybrid.py
```

## Operating Modes

### 1. Pure Performance Mode (Default)
- **Use Case**: Production environments requiring maximum speed
- **Features**:
  - Sub-millisecond response times
  - Minimal logging (WARNING level)
  - Response caching enabled
  - Async operations optimized
  - Debug API still accessible

```bash
python kimera_hybrid.py
```

### 2. High Performance with Debug
- **Use Case**: Production with monitoring capabilities
- **Features**:
  - Maintains high performance
  - Request tracing enabled
  - Logs stored in ring buffer
  - Debug endpoints active
  - Performance profiling available

```bash
python kimera_hybrid.py --debug
# or
KIMERA_DEBUG=true python kimera_hybrid.py
```

### 3. Full Debug Mode
- **Use Case**: Development and troubleshooting
- **Features**:
  - Comprehensive logging to console and file
  - All debug features enabled
  - Synchronous logging (may impact performance)
  - Detailed request/response tracking

```bash
python kimera_hybrid.py --debug --no-performance
# or
KIMERA_DEBUG=true KIMERA_PERFORMANCE=false python kimera_hybrid.py
```

## Debug API Endpoints

### 1. Debug Information Dashboard
```http
GET /debug/info?include_logs=true&include_traces=true&include_profiles=true
```

Returns comprehensive debug information including:
- Recent logs from ring buffer
- Request traces and statistics
- Performance profiles
- Current system configuration

### 2. Runtime Mode Control
```http
# Enable/disable debug mode
POST /debug/mode?enabled=true

# Enable/disable performance profiling
POST /debug/profiling?enabled=true
```

### 3. Log Management
```http
# Get recent logs
GET /debug/logs?limit=100&level=ERROR

# Change log level dynamically
POST /debug/log-level?level=DEBUG&module=kimera.core
```

### 4. Request Tracing
```http
# Get request traces
GET /debug/traces?limit=50&path_filter=/kimera

# Response includes:
{
  "traces": [...],
  "statistics": {
    "endpoints": {
      "/health": {
        "count": 1000,
        "avg_time_ms": 0.5,
        "min_time_ms": 0.1,
        "max_time_ms": 2.3
      }
    }
  }
}
```

### 5. Performance Statistics
```http
GET /performance/stats
```

## Key Features

### 1. Hybrid Logging System
- **Ring Buffer**: Stores recent logs in memory (no disk I/O)
- **Async Handlers**: Non-blocking log processing
- **Structured Logging**: JSON-formatted logs with context
- **Per-Module Control**: Adjust log levels for specific components

### 2. Request Tracing
- **Automatic Tracking**: All requests traced with minimal overhead
- **Performance Metrics**: Response times, error rates, throughput
- **Request Correlation**: Unique request IDs for debugging
- **Slow Request Detection**: Automatic logging of slow requests

### 3. Performance Profiling
- **Optional Profiling**: Enable/disable at runtime
- **Memory Tracking**: Monitor memory usage per operation
- **Function Timing**: Detailed timing for critical functions
- **Zero Overhead**: No impact when disabled

### 4. Smart Caching
- **Cache Headers**: Shows hit/miss status and age
- **Debug Bypass**: `?bypass_cache=true` for testing
- **TTL Management**: Configurable cache durations
- **Cache Statistics**: Monitor cache effectiveness

## Performance Comparison

| Mode | Response Time | Logging | Memory Usage | CPU Usage |
|------|--------------|---------|--------------|-----------|
| Pure Performance | <1ms | Minimal | Low | Low |
| Performance + Debug | <2ms | Ring Buffer | Medium | Low-Medium |
| Full Debug | 5-10ms | Full | High | Medium |

## Best Practices

### 1. Production Deployment
```bash
# Use performance mode with debug capabilities
KIMERA_DEBUG=true python kimera_hybrid.py

# Monitor via debug endpoints without impacting users
curl http://localhost:8001/debug/info
```

### 2. Development
```bash
# Full debug for troubleshooting
python kimera_hybrid.py --debug --no-performance --reload

# Watch logs in real-time
curl http://localhost:8001/debug/logs?limit=50
```

### 3. Performance Testing
```bash
# Enable profiling temporarily
curl -X POST http://localhost:8001/debug/profiling?enabled=true

# Run tests
# ...

# Get performance data
curl http://localhost:8001/performance/stats

# Disable profiling
curl -X POST http://localhost:8001/debug/profiling?enabled=false
```

### 4. Troubleshooting
```python
# Dynamic log level adjustment
import requests

# Enable debug logging for specific module
requests.post("http://localhost:8001/debug/log-level", 
              params={"level": "DEBUG", "module": "kimera.core"})

# Get recent errors
response = requests.get("http://localhost:8001/debug/logs",
                       params={"level": "ERROR", "limit": 20})
```

## Architecture Benefits

1. **Zero-Cost Abstractions**: Debug features have minimal impact when disabled
2. **Lazy Loading**: Components loaded only when needed
3. **Async Everything**: Non-blocking operations throughout
4. **Memory Efficiency**: Ring buffers prevent memory bloat
5. **Granular Control**: Fine-tune performance vs debugging per component

## Migration Guide

### From `kimera.py`:
```bash
# Replace
python kimera.py

# With
python kimera_hybrid.py --no-performance
```

### From `kimera_optimized.py`:
```bash
# Replace
python kimera_optimized.py

# With
python kimera_hybrid.py
```

## Monitoring Dashboard Example

Create a simple monitoring script:

```python
import requests
import time

def monitor_kimera():
    while True:
        # Get performance stats
        stats = requests.get("http://localhost:8001/performance/stats").json()
        
        # Get recent errors
        errors = requests.get("http://localhost:8001/debug/logs", 
                            params={"level": "ERROR", "limit": 5}).json()
        
        # Display dashboard
        print("\033[2J\033[H")  # Clear screen
        print("=== Kimera Monitor ===")
        print(f"Active Requests: {stats['request_statistics']['active_requests']}")
        print(f"Total Requests: {stats['request_statistics']['total_requests']}")
        print(f"Cache Size: {stats['response_cache_size']}")
        print(f"Recent Errors: {errors['count']}")
        
        time.sleep(1)

if __name__ == "__main__":
    monitor_kimera()
```

## Conclusion

The Kimera Hybrid Mode provides a powerful solution for maintaining high performance while retaining full debugging capabilities. Use it to:

- Run production systems with confidence
- Debug issues without impacting performance
- Monitor system health in real-time
- Profile and optimize specific operations
- Maintain comprehensive logs without disk I/O overhead

Choose the appropriate mode based on your needs, and leverage the debug API to gain insights into your system's behavior without sacrificing performance.