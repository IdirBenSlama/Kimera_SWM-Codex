# Kimera SWM Performance Testing Suite

This comprehensive performance testing suite is designed to push Kimera to its peak performance and gather real-world metrics without any mocks or simulations.

## 🚀 Quick Start

### Prerequisites

1. Ensure Kimera server is running on `http://localhost:8000`
2. Install required dependencies:
```bash
pip install -r performance_test_requirements.txt
```

### Running the Tests

#### 1. Comprehensive Performance Test
This is the main test that pushes Kimera to its limits:

```bash
python performance_test_kimera.py
```

This test will:
- Test all major endpoints with varying loads
- Perform concurrent request testing (up to 100 concurrent requests)
- Measure response times, throughput, and error rates
- Capture system metrics (CPU, memory, disk, network)
- Generate detailed performance reports and visualizations
- Run a peak load test with 1000 requests

**Expected Duration**: ~5-10 minutes

#### 2. Real-time Performance Monitor
For live monitoring during tests or regular operation:

```bash
python monitor_kimera_realtime.py
```

Features:
- Live dashboard with real-time metrics
- Response time trends
- CPU and memory usage graphs
- Request rate and error tracking
- Press 'q' to quit, 'r' to reset statistics

#### 3. Performance Results Analysis
After running the performance test:

```bash
python analyze_performance_results.py
```

This will:
- Load the latest test results
- Identify performance bottlenecks
- Analyze scalability and reliability
- Generate detailed visualizations
- Create a comprehensive analysis report

## 📊 Metrics Collected

### Request Metrics
- Response time (average, min, max, median, P95, P99)
- Throughput (requests per second)
- Success/failure rates
- Response size

### System Metrics
- CPU usage (average, peak)
- Memory usage (average, peak)
- Disk I/O
- Network I/O
- GPU utilization (if available)

### Kimera-specific Metrics
- Geoid creation performance
- SCAR processing speed
- Embedding generation time
- Thermodynamic analysis performance
- Cognitive cycle execution time

## 📈 Test Scenarios

The performance test includes:

1. **Basic Health Checks**
   - Root endpoint: 500 requests @ 50 concurrent
   - Health endpoint: 1000 requests @ 100 concurrent
   - Metrics endpoint: 500 requests @ 50 concurrent

2. **Core Functionality**
   - System status: 200 requests @ 20 concurrent
   - Cognitive cycles: 100 requests @ 10 concurrent

3. **Heavy Load Tests**
   - Geoid creation: 200 requests @ 20 concurrent
   - SCAR creation: 300 requests @ 30 concurrent
   - Embedding generation: 250 requests @ 25 concurrent

4. **Peak Load Test**
   - Geoid creation: 1000 requests @ 100 concurrent

## 📁 Output Files

All results are saved in the `performance_results/` directory:

- `kimera_performance_report_TIMESTAMP.json` - Raw test data
- `kimera_performance_charts_TIMESTAMP.png` - Performance visualizations
- `kimera_detailed_analysis_TIMESTAMP.png` - Detailed analysis charts
- `kimera_analysis_report_TIMESTAMP.txt` - Human-readable analysis report

## 🎯 Performance Targets

Based on the tests, Kimera should aim for:

- **Response Time**: < 200ms average for most endpoints
- **Success Rate**: > 99% under normal load
- **Throughput**: > 100 requests/second for read operations
- **CPU Usage**: < 80% under peak load
- **Memory Usage**: Stable, < 85% of available

## 🔧 Customization

### Adjusting Test Parameters

Edit `performance_test_kimera.py` to modify:

```python
test_scenarios = [
    {
        "endpoint": "/your/endpoint",
        "method": "POST",
        "data": {"your": "data"},
        "concurrent": 50,  # Number of concurrent requests
        "total": 500       # Total requests to make
    }
]
```

### Adding New Metrics

To collect additional metrics, modify the `capture_system_metrics` method in `KimeraPerformanceTester`.

## ⚠️ Important Notes

1. **Resource Usage**: These tests will consume significant CPU and memory. Ensure your system has adequate resources.

2. **Database Impact**: Tests that create geoids/SCARs will add data to your database. Consider using a test database.

3. **Network**: Ensure stable network connection if testing remote servers.

4. **Monitoring**: Run the real-time monitor in a separate terminal to watch system behavior during tests.

## 🐛 Troubleshooting

### Connection Refused
- Verify Kimera server is running: `curl http://localhost:8000/health`
- Check the port number in the test scripts

### High Failure Rate
- Check server logs for errors
- Reduce concurrent request count
- Verify database connections

### Out of Memory
- Reduce total request count
- Run tests in smaller batches
- Monitor with `monitor_kimera_realtime.py`

## 📊 Interpreting Results

### Response Time
- **< 100ms**: Excellent
- **100-200ms**: Good
- **200-500ms**: Acceptable
- **> 500ms**: Needs optimization

### Success Rate
- **> 99.9%**: Excellent
- **99-99.9%**: Good
- **95-99%**: Acceptable
- **< 95%**: Critical issue

### CPU Usage
- **< 50%**: Plenty of headroom
- **50-70%**: Healthy utilization
- **70-85%**: Monitor closely
- **> 85%**: Consider scaling

## 🚀 Next Steps

After running the performance tests:

1. Review the analysis report for bottlenecks
2. Implement recommended optimizations
3. Re-run tests to measure improvements
4. Set up continuous performance monitoring
5. Establish performance regression tests

For production environments, consider:
- Load balancing for horizontal scaling
- Caching strategies for frequently accessed data
- Database query optimization
- Asynchronous processing for heavy operations
- Connection pooling optimization