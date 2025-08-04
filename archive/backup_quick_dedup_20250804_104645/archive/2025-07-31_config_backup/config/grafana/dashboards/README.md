# Kimera SWM Grafana Dashboards

This directory contains comprehensive Grafana dashboard templates for monitoring the Kimera Spherical Word Methodology (SWM) system. These dashboards provide real-time insights into cognitive architecture, system performance, and revolutionary intelligence metrics.

## 🚀 Quick Start

### Access Grafana
- **URL**: http://localhost:3000
- **Username**: `admin`
- **Password**: `kimera_grafana_2025`

### Prerequisites
1. Ensure Docker containers are running:
   ```bash
   docker-compose up -d
   ```
2. Verify Prometheus is collecting metrics at http://localhost:9090
3. Ensure Kimera SWM is running and exposing metrics

## 📊 Available Dashboards

### 1. **Kimera SWM - Comprehensive Revolutionary Intelligence Dashboard**
**File**: `kimera-comprehensive-dashboard.json`  
**UID**: `kimera-comprehensive`

**Overview**: The main dashboard providing a complete view of the Kimera system.

**Sections**:
- 🧠 **Revolutionary Intelligence Overview**: Key metrics (Geoids, SCARs, Insights, Coherence)
- 🔥 **GPU & AI Workload Performance**: RTX 4090 utilization, temperature, power
- 🌊 **Cognitive Field Dynamics**: Geoid creation, SCAR formation rates
- ⚡ **Performance & Latency**: API response times, embedding generation
- 🖥️ **System Resources**: CPU, Memory, Network I/O
- 🧬 **Selective Feedback**: Operations and accuracy metrics
- 🔥 **Contradiction Engine**: Event rates and thermodynamic entropy
- 🚨 **Alerts & Error Monitoring**: System errors by component

**Key Metrics**:
- `kimera_geoids_total` - Total number of geoids
- `kimera_scars_total` - Total SCARs in system
- `kimera_revolutionary_insights_total` - Breakthrough insights generated
- `kimera_cognitive_coherence` - System coherence score
- `kimera_gpu_utilization_percent` - GPU performance
- `kimera_embedding_duration_seconds` - AI model latency

### 2. **Kimera SWM - Cognitive Field Dynamics Dashboard**
**File**: `kimera-cognitive-field-dashboard.json`  
**UID**: `kimera-cognitive-field`

**Overview**: Specialized dashboard for deep cognitive field analysis.

**Sections**:
- 🌊 **Wave Propagation Dynamics**: Active waves, field evolution, resonance events
- 🎯 **Field Evolution & Resonance**: Field strength, frequency distribution, pattern heatmaps
- ⚡ **Performance Metrics**: Evolution time, wave propagation, memory usage
- 🔬 **Advanced Analytics**: Phase coherence, energy transfer, clustering analysis

**Key Metrics**:
- `kimera_cognitive_field_active_waves` - Current active semantic waves
- `kimera_cognitive_field_total_fields` - Total semantic fields
- `kimera_cognitive_field_resonance_events` - Field resonance interactions
- `kimera_cognitive_field_wave_amplitude` - Wave strength evolution
- `kimera_cognitive_field_phase_coherence` - System synchronization

### 3. **Kimera SWM - Critical Alerts & Health Dashboard**
**File**: `kimera-alerts-dashboard.json`  
**UID**: `kimera-alerts`

**Overview**: Focused on system health and critical alerting.

**Sections**:
- 🚨 **Critical System Health**: Service status, resource usage, GPU temperature
- ⚠️ **Error Rates & Anomalies**: Component errors, HTTP failures
- 🧠 **Cognitive System Health**: Coherence, contradictions, insights
- 📊 **Performance Thresholds**: Latency monitoring with alert thresholds

**Alert Thresholds**:
- CPU Usage: >90% (Critical), >70% (Warning)
- Memory Usage: >95% (Critical), >80% (Warning)
- GPU Temperature: >85°C (Critical), >75°C (Warning)
- Request Latency: >1s (Critical), >0.5s (Warning)
- Cognitive Coherence: <0.6 (Critical), <0.8 (Warning)

## 🛠️ Configuration

### Template Variables
- **Time Range**: Adjustable time windows (1m, 5m, 15m, 1h)
- **Field ID**: Filter by specific cognitive field IDs
- **Component**: Filter metrics by system component

### Refresh Rates
- **Comprehensive Dashboard**: 5 seconds
- **Cognitive Field Dashboard**: 5 seconds  
- **Alerts Dashboard**: 5 seconds

### Data Sources
All dashboards use Prometheus as the primary data source:
- **Type**: Prometheus
- **URL**: http://prometheus:9090
- **UID**: `prometheus`

## 📈 Metric Categories

### System Metrics
```
kimera_system_cpu_percent          # CPU usage percentage
kimera_system_memory_percent       # Memory usage percentage
kimera_system_disk_percent         # Disk usage percentage
kimera_system_network_bytes_total  # Network I/O bytes
```

### GPU Metrics
```
kimera_gpu_utilization_percent     # GPU utilization
kimera_gpu_memory_used_bytes       # GPU memory usage
kimera_gpu_temperature_celsius     # GPU temperature
kimera_gpu_power_watts            # GPU power consumption
```

### Cognitive Architecture Metrics
```
kimera_geoids_total                # Total geoids
kimera_scars_total                 # Total SCARs
kimera_contradictions_total        # Contradiction events
kimera_revolutionary_insights_total # Revolutionary insights
kimera_cognitive_coherence         # Cognitive coherence score
```

### Performance Metrics
```
kimera_request_duration_seconds    # API request latency
kimera_embedding_duration_seconds  # Embedding generation time
kimera_requests_total             # Total API requests
kimera_errors_total              # Error count by type
```

### Cognitive Field Dynamics
```
kimera_cognitive_field_active_waves      # Active semantic waves
kimera_cognitive_field_total_fields      # Total semantic fields
kimera_cognitive_field_resonance_events  # Resonance interactions
kimera_cognitive_field_wave_amplitude    # Wave strength
kimera_cognitive_field_evolution_duration_ms # Evolution performance
```

## 🎯 Usage Guidelines

### For System Administrators
1. **Start with Alerts Dashboard** - Monitor critical health metrics
2. **Use Comprehensive Dashboard** - Get overall system overview
3. **Set up alerting rules** based on threshold panels

### For Researchers & Developers
1. **Cognitive Field Dashboard** - Analyze semantic wave dynamics
2. **Performance sections** - Optimize AI model latency
3. **Custom time ranges** - Study specific experimental periods

### For Operations Teams
1. **Resource monitoring** - Track CPU, Memory, GPU usage
2. **Error tracking** - Monitor component failures
3. **Capacity planning** - Use historical data for scaling

## 🔧 Customization

### Adding New Panels
1. Edit dashboard JSON files
2. Add new panel configuration
3. Define Prometheus queries
4. Set appropriate thresholds and styling

### Creating Custom Dashboards
1. Use existing dashboards as templates
2. Follow Kimera naming conventions
3. Include appropriate tags for organization
4. Document new metrics in this README

### Modifying Alert Thresholds
Update threshold values in panel configurations:
```json
"thresholds": {
  "mode": "absolute",
  "steps": [
    {"color": "green", "value": null},
    {"color": "yellow", "value": 70},
    {"color": "red", "value": 90}
  ]
}
```

## 🚨 Troubleshooting

### Dashboard Not Loading
1. Check Grafana container status: `docker ps`
2. Verify Prometheus data source connection
3. Ensure Kimera metrics are being exposed

### Missing Data
1. Verify Kimera SWM is running: `curl http://localhost:8000/metrics`
2. Check Prometheus targets: http://localhost:9090/targets
3. Confirm metric names match dashboard queries

### Performance Issues
1. Reduce refresh rate for complex dashboards
2. Limit time range for heavy queries
3. Use template variables to filter data

## 📚 Additional Resources

- **Kimera API Documentation**: http://localhost:8000/docs
- **Prometheus Metrics**: http://localhost:9090/metrics
- **Grafana Documentation**: https://grafana.com/docs/
- **Prometheus Query Language**: https://prometheus.io/docs/prometheus/latest/querying/

## 🏷️ Tags
- `kimera` - All Kimera-related dashboards
- `swm` - Spherical Word Methodology specific
- `cognitive` - Cognitive architecture metrics
- `ai` - AI/ML performance monitoring
- `alerts` - Critical alerting dashboards
- `health` - System health monitoring

---

**Note**: These dashboards are designed for the Kimera SWM Alpha Prototype V0.1. Metric names and structures may evolve with system updates. 