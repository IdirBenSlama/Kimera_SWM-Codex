# KIMERA SWM Configuration Consolidation Report
**Generated**: 2025-07-31T23:17:21.069914
**Configuration Files Processed**: 40
**Categories**: 10
**Conflicts Found**: 28

## Configuration Categories

### Development
**Files**: 7

- `development.yaml` (1460 bytes)
- `tcse_config.yaml` (734 bytes)
- `grafana\provisioning\dashboards\dashboards.yml` (255 bytes)
- `optimized_settings.json` (462 bytes)
- `grafana\dashboards\kimera-alerts-dashboard.json` (23029 bytes)
- `grafana\dashboards\kimera-cognitive-field-dashboard.json` (10316 bytes)
- `grafana\dashboards\kimera-comprehensive-dashboard.json` (41537 bytes)

---

### Production
**Files**: 7

- `production.yaml` (1527 bytes)
- `tcse_config.yaml` (734 bytes)
- `grafana\provisioning\dashboards\dashboards.yml` (255 bytes)
- `optimized_settings.json` (462 bytes)
- `grafana\dashboards\kimera-alerts-dashboard.json` (23029 bytes)
- `grafana\dashboards\kimera-cognitive-field-dashboard.json` (10316 bytes)
- `grafana\dashboards\kimera-comprehensive-dashboard.json` (41537 bytes)

---

### General
**Files**: 6

- `tcse_config.yaml` (734 bytes)
- `grafana\provisioning\dashboards\dashboards.yml` (255 bytes)
- `optimized_settings.json` (462 bytes)
- `grafana\dashboards\kimera-alerts-dashboard.json` (23029 bytes)
- `grafana\dashboards\kimera-cognitive-field-dashboard.json` (10316 bytes)
- `grafana\dashboards\kimera-comprehensive-dashboard.json` (41537 bytes)

---

### Monitoring
**Files**: 2

- `prometheus.yml` (1502 bytes)
- `grafana\provisioning\datasources\prometheus.yml` (174 bytes)

---

### Docker
**Files**: 2

- `docker\docker-compose-databases.yml` (3225 bytes)
- `docker\docker-compose.yml` (3142 bytes)

---

### Testing
**Files**: 7

- `ai_test_suite_config.json` (9843 bytes)
- `tcse_config.yaml` (734 bytes)
- `grafana\provisioning\dashboards\dashboards.yml` (255 bytes)
- `optimized_settings.json` (462 bytes)
- `grafana\dashboards\kimera-alerts-dashboard.json` (23029 bytes)
- `grafana\dashboards\kimera-cognitive-field-dashboard.json` (10316 bytes)
- `grafana\dashboards\kimera-comprehensive-dashboard.json` (41537 bytes)

---

### Ai_Ml
**Files**: 3

- `fine_tuning_config.json` (846 bytes)
- `kimera_fine_tuning_integration.json` (1150 bytes)
- `scientific_fine_tuning_config.json` (692 bytes)

---

### Trading
**Files**: 4

- `trading_config.json` (925 bytes)
- `kimera_binance_hmac.env` (846 bytes)
- `kimera_cdp_config.env` (1033 bytes)
- `kimera_cdp_live.env` (716 bytes)

---

### Environment
**Files**: 1

- `kimera_max_profit_config.env` (1845 bytes)

---

### Database
**Files**: 1

- `redis_sample.env` (1527 bytes)

---

## Configuration Conflicts

### database
**Conflicting values:**

- `development`: `{'echo': True, 'pool_size': 5}` (from `development.yaml`)
- `production`: `{'echo': False, 'pool_size': 20}` (from `production.yaml`)
- `general`: `{'connection_pool_size': 20, 'query_timeout': 30, 'batch_size': 100, 'cache_size': 10000}` (from `optimized_settings.json`)

---

### environment
**Conflicting values:**

- `development`: `development` (from `development.yaml`)
- `production`: `production` (from `production.yaml`)

---

### logging
**Conflicting values:**

- `development`: `{'level': 'DEBUG', 'structured': True}` (from `development.yaml`)
- `production`: `{'level': 'INFO', 'structured': False}` (from `production.yaml`)

---

### monitoring
**Conflicting values:**

- `development`: `{'enabled': True, 'health_check_interval': 15, 'memory_tracking': True, 'performance_tracking': True, 'thresholds': {'cpu_critical': 95.0, 'cpu_warning': 80.0, 'disk_critical': 95.0, 'disk_warning': 85.0, 'gpu_memory_critical': 95.0, 'gpu_memory_warning': 85.0, 'memory_critical': 95.0, 'memory_warning': 80.0, 'response_time_critical': 5.0, 'response_time_warning': 2.0}}` (from `development.yaml`)
- `production`: `{'detailed_metrics': True, 'enabled': True, 'health_check_interval': 30, 'memory_tracking': True, 'performance_tracking': True, 'thresholds': {'cpu_critical': 95.0, 'cpu_warning': 80.0, 'disk_critical': 95.0, 'disk_warning': 85.0, 'gpu_memory_critical': 95.0, 'gpu_memory_warning': 85.0, 'memory_critical': 95.0, 'memory_warning': 80.0, 'response_time_critical': 5.0, 'response_time_warning': 2.0}}` (from `production.yaml`)

---

### security
**Conflicting values:**

- `development`: `{'rate_limit_enabled': False}` (from `development.yaml`)
- `production`: `{'cors_enabled': True, 'https_only': True, 'rate_limit_enabled': True}` (from `production.yaml`)

---

### server
**Conflicting values:**

- `development`: `{'host': '127.0.0.1', 'port': 8000, 'reload': True}` (from `development.yaml`)
- `production`: `{'host': '0.0.0.0', 'port': 8000, 'reload': False}` (from `production.yaml`)

---

### annotations
**Conflicting values:**

- `general`: `{'list': [{'builtIn': 1, 'datasource': {'type': 'grafana', 'uid': '-- Grafana --'}, 'enable': True, 'hide': True, 'iconColor': 'rgba(0, 211, 255, 1)', 'name': 'Annotations & Alerts', 'target': {'limit': 100, 'matchAny': False, 'tags': [], 'type': 'dashboard'}, 'type': 'dashboard'}]}` (from `kimera-alerts-dashboard.json`)
- `general`: `{'list': [{'builtIn': 1, 'datasource': {'type': 'grafana', 'uid': '-- Grafana --'}, 'enable': True, 'hide': True, 'iconColor': 'rgba(0, 211, 255, 1)', 'name': 'Annotations & Alerts', 'type': 'dashboard'}]}` (from `kimera-cognitive-field-dashboard.json`)
- `general`: `{'list': [{'builtIn': 1, 'datasource': {'type': 'grafana', 'uid': '-- Grafana --'}, 'enable': True, 'hide': True, 'iconColor': 'rgba(0, 211, 255, 1)', 'name': 'Annotations & Alerts', 'target': {'limit': 100, 'matchAny': False, 'tags': [], 'type': 'dashboard'}, 'type': 'dashboard'}]}` (from `kimera-comprehensive-dashboard.json`)

---

### description
**Conflicting values:**

- `general`: `Kimera SWM Critical Alerts and System Health Monitoring Dashboard` (from `kimera-alerts-dashboard.json`)
- `general`: `Comprehensive Kimera SWM Monitoring Dashboard - Cognitive Architecture, System Resources, GPU Performance, and Revolutionary Intelligence Metrics` (from `kimera-comprehensive-dashboard.json`)

---

### graphTooltip
**Conflicting values:**

- `general`: `1` (from `kimera-alerts-dashboard.json`)
- `general`: `0` (from `kimera-cognitive-field-dashboard.json`)
- `general`: `1` (from `kimera-comprehensive-dashboard.json`)

---

### id
**Conflicting values:**

- `general`: `None` (from `kimera-alerts-dashboard.json`)
- `general`: `2` (from `kimera-cognitive-field-dashboard.json`)
- `general`: `None` (from `kimera-comprehensive-dashboard.json`)

---

### links
**Conflicting values:**

- `general`: `[]` (from `kimera-alerts-dashboard.json`)
- `general`: `[]` (from `kimera-cognitive-field-dashboard.json`)
- `general`: `[{'asDropdown': False, 'icon': 'external link', 'includeVars': False, 'keepTime': False, 'tags': [], 'targetBlank': True, 'title': 'Kimera API Documentation', 'tooltip': 'Open Kimera API docs', 'type': 'link', 'url': 'http://localhost:8000/docs'}]` (from `kimera-comprehensive-dashboard.json`)

---

### panels
**Conflicting values:**

- `general`: `[{'collapsed': False, 'gridPos': {'h': 1, 'w': 24, 'x': 0, 'y': 0}, 'id': 1, 'panels': [], 'title': '🚨 Critical System Health', 'type': 'row'}, {'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'fieldConfig': {'defaults': {'color': {'mode': 'thresholds'}, 'mappings': [{'options': {'0': {'color': 'red', 'index': 0, 'text': 'DOWN'}, '1': {'color': 'green', 'index': 1, 'text': 'UP'}}, 'type': 'value'}], 'thresholds': {'mode': 'absolute', 'steps': [{'color': 'red', 'value': None}, {'color': 'green', 'value': 1}]}, 'unit': 'short'}, 'overrides': []}, 'gridPos': {'h': 4, 'w': 6, 'x': 0, 'y': 1}, 'id': 2, 'options': {'colorMode': 'background', 'graphMode': 'none', 'justifyMode': 'center', 'orientation': 'auto', 'reduceOptions': {'values': False, 'calcs': ['lastNotNull'], 'fields': ''}, 'textMode': 'auto'}, 'pluginVersion': '10.0.0', 'targets': [{'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'expr': 'up{job="kimera"}', 'refId': 'A'}], 'title': 'Kimera Service Status', 'type': 'stat'}, {'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'fieldConfig': {'defaults': {'color': {'mode': 'thresholds'}, 'mappings': [], 'max': 100, 'min': 0, 'thresholds': {'mode': 'absolute', 'steps': [{'color': 'green', 'value': None}, {'color': 'yellow', 'value': 70}, {'color': 'red', 'value': 90}]}, 'unit': 'percent'}, 'overrides': []}, 'gridPos': {'h': 4, 'w': 6, 'x': 6, 'y': 1}, 'id': 3, 'options': {'colorMode': 'value', 'graphMode': 'area', 'justifyMode': 'center', 'orientation': 'auto', 'reduceOptions': {'values': False, 'calcs': ['lastNotNull'], 'fields': ''}, 'textMode': 'auto'}, 'pluginVersion': '10.0.0', 'targets': [{'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'expr': 'kimera_system_cpu_percent', 'refId': 'A'}], 'title': 'CPU Usage', 'type': 'stat'}, {'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'fieldConfig': {'defaults': {'color': {'mode': 'thresholds'}, 'mappings': [], 'max': 100, 'min': 0, 'thresholds': {'mode': 'absolute', 'steps': [{'color': 'green', 'value': None}, {'color': 'yellow', 'value': 80}, {'color': 'red', 'value': 95}]}, 'unit': 'percent'}, 'overrides': []}, 'gridPos': {'h': 4, 'w': 6, 'x': 12, 'y': 1}, 'id': 4, 'options': {'colorMode': 'value', 'graphMode': 'area', 'justifyMode': 'center', 'orientation': 'auto', 'reduceOptions': {'values': False, 'calcs': ['lastNotNull'], 'fields': ''}, 'textMode': 'auto'}, 'pluginVersion': '10.0.0', 'targets': [{'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'expr': 'kimera_system_memory_percent', 'refId': 'A'}], 'title': 'Memory Usage', 'type': 'stat'}, {'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'fieldConfig': {'defaults': {'color': {'mode': 'thresholds'}, 'mappings': [], 'thresholds': {'mode': 'absolute', 'steps': [{'color': 'green', 'value': None}, {'color': 'yellow', 'value': 75}, {'color': 'red', 'value': 85}]}, 'unit': 'celsius'}, 'overrides': []}, 'gridPos': {'h': 4, 'w': 6, 'x': 18, 'y': 1}, 'id': 5, 'options': {'colorMode': 'value', 'graphMode': 'area', 'justifyMode': 'center', 'orientation': 'auto', 'reduceOptions': {'values': False, 'calcs': ['lastNotNull'], 'fields': ''}, 'textMode': 'auto'}, 'pluginVersion': '10.0.0', 'targets': [{'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'expr': 'kimera_gpu_temperature_celsius', 'refId': 'A'}], 'title': 'GPU Temperature', 'type': 'stat'}, {'collapsed': False, 'gridPos': {'h': 1, 'w': 24, 'x': 0, 'y': 5}, 'id': 6, 'panels': [], 'title': '⚠️ Error Rates & Anomalies', 'type': 'row'}, {'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'fieldConfig': {'defaults': {'color': {'mode': 'palette-classic'}, 'custom': {'axisLabel': '', 'axisPlacement': 'auto', 'barAlignment': 0, 'drawStyle': 'line', 'fillOpacity': 10, 'gradientMode': 'none', 'hideFrom': {'legend': False, 'tooltip': False, 'vis': False}, 'lineInterpolation': 'linear', 'lineWidth': 2, 'pointSize': 5, 'scaleDistribution': {'type': 'linear'}, 'showPoints': 'never', 'spanNulls': False, 'stacking': {'group': 'A', 'mode': 'none'}, 'thresholdsStyle': {'mode': 'off'}}, 'mappings': [], 'thresholds': {'mode': 'absolute', 'steps': [{'color': 'green', 'value': None}, {'color': 'red', 'value': 80}]}, 'unit': 'short'}, 'overrides': [{'matcher': {'id': 'byName', 'options': 'Error Rate'}, 'properties': [{'id': 'color', 'value': {'mode': 'fixed', 'fixedColor': 'red'}}]}]}, 'gridPos': {'h': 8, 'w': 12, 'x': 0, 'y': 6}, 'id': 7, 'options': {'legend': {'calcs': [], 'displayMode': 'list', 'placement': 'bottom'}, 'tooltip': {'mode': 'single', 'sort': 'none'}}, 'targets': [{'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'expr': 'rate(kimera_errors_total[5m])', 'legendFormat': '{{error_type}} - {{component}}', 'refId': 'A'}], 'title': 'Error Rate by Component', 'type': 'timeseries'}, {'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'fieldConfig': {'defaults': {'color': {'mode': 'palette-classic'}, 'custom': {'axisLabel': '', 'axisPlacement': 'auto', 'barAlignment': 0, 'drawStyle': 'line', 'fillOpacity': 10, 'gradientMode': 'none', 'hideFrom': {'legend': False, 'tooltip': False, 'vis': False}, 'lineInterpolation': 'linear', 'lineWidth': 2, 'pointSize': 5, 'scaleDistribution': {'type': 'linear'}, 'showPoints': 'never', 'spanNulls': False, 'stacking': {'group': 'A', 'mode': 'none'}, 'thresholdsStyle': {'mode': 'off'}}, 'mappings': [], 'thresholds': {'mode': 'absolute', 'steps': [{'color': 'green', 'value': None}, {'color': 'red', 'value': 80}]}, 'unit': 'reqps'}, 'overrides': []}, 'gridPos': {'h': 8, 'w': 12, 'x': 12, 'y': 6}, 'id': 8, 'options': {'legend': {'calcs': [], 'displayMode': 'list', 'placement': 'bottom'}, 'tooltip': {'mode': 'single', 'sort': 'none'}}, 'targets': [{'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'expr': 'rate(kimera_requests_total{status=~"4..|5.."}[5m])', 'legendFormat': '{{method}} {{endpoint}} - {{status}}', 'refId': 'A'}], 'title': 'HTTP Error Requests', 'type': 'timeseries'}, {'collapsed': False, 'gridPos': {'h': 1, 'w': 24, 'x': 0, 'y': 14}, 'id': 9, 'panels': [], 'title': '🧠 Cognitive System Health', 'type': 'row'}, {'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'fieldConfig': {'defaults': {'color': {'mode': 'thresholds'}, 'mappings': [], 'max': 1, 'min': 0, 'thresholds': {'mode': 'absolute', 'steps': [{'color': 'red', 'value': None}, {'color': 'yellow', 'value': 0.6}, {'color': 'green', 'value': 0.8}]}, 'unit': 'percentunit'}, 'overrides': []}, 'gridPos': {'h': 4, 'w': 8, 'x': 0, 'y': 15}, 'id': 10, 'options': {'colorMode': 'value', 'graphMode': 'area', 'justifyMode': 'center', 'orientation': 'auto', 'reduceOptions': {'values': False, 'calcs': ['lastNotNull'], 'fields': ''}, 'textMode': 'auto'}, 'pluginVersion': '10.0.0', 'targets': [{'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'expr': 'kimera_cognitive_coherence{component="overall"}', 'refId': 'A'}], 'title': 'Cognitive Coherence', 'type': 'stat'}, {'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'fieldConfig': {'defaults': {'color': {'mode': 'thresholds'}, 'mappings': [], 'thresholds': {'mode': 'absolute', 'steps': [{'color': 'green', 'value': None}, {'color': 'yellow', 'value': 10}, {'color': 'red', 'value': 20}]}, 'unit': 'short'}, 'overrides': []}, 'gridPos': {'h': 4, 'w': 8, 'x': 8, 'y': 15}, 'id': 11, 'options': {'colorMode': 'value', 'graphMode': 'area', 'justifyMode': 'center', 'orientation': 'auto', 'reduceOptions': {'values': False, 'calcs': ['lastNotNull'], 'fields': ''}, 'textMode': 'auto'}, 'pluginVersion': '10.0.0', 'targets': [{'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'expr': 'rate(kimera_contradictions_total[5m]) * 60', 'refId': 'A'}], 'title': 'Contradictions/min', 'type': 'stat'}, {'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'fieldConfig': {'defaults': {'color': {'mode': 'thresholds'}, 'mappings': [], 'thresholds': {'mode': 'absolute', 'steps': [{'color': 'red', 'value': None}, {'color': 'yellow', 'value': 1}, {'color': 'green', 'value': 5}]}, 'unit': 'short'}, 'overrides': []}, 'gridPos': {'h': 4, 'w': 8, 'x': 16, 'y': 15}, 'id': 12, 'options': {'colorMode': 'value', 'graphMode': 'area', 'justifyMode': 'center', 'orientation': 'auto', 'reduceOptions': {'values': False, 'calcs': ['lastNotNull'], 'fields': ''}, 'textMode': 'auto'}, 'pluginVersion': '10.0.0', 'targets': [{'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'expr': 'rate(kimera_revolutionary_insights_total[5m]) * 60', 'refId': 'A'}], 'title': 'Insights/min', 'type': 'stat'}, {'collapsed': False, 'gridPos': {'h': 1, 'w': 24, 'x': 0, 'y': 19}, 'id': 13, 'panels': [], 'title': '📊 Performance Thresholds', 'type': 'row'}, {'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'fieldConfig': {'defaults': {'color': {'mode': 'palette-classic'}, 'custom': {'axisLabel': '', 'axisPlacement': 'auto', 'barAlignment': 0, 'drawStyle': 'line', 'fillOpacity': 10, 'gradientMode': 'none', 'hideFrom': {'legend': False, 'tooltip': False, 'vis': False}, 'lineInterpolation': 'linear', 'lineWidth': 2, 'pointSize': 5, 'scaleDistribution': {'type': 'linear'}, 'showPoints': 'never', 'spanNulls': False, 'stacking': {'group': 'A', 'mode': 'none'}, 'thresholdsStyle': {'mode': 'line'}}, 'mappings': [], 'thresholds': {'mode': 'absolute', 'steps': [{'color': 'green', 'value': None}, {'color': 'red', 'value': 1}]}, 'unit': 's'}, 'overrides': []}, 'gridPos': {'h': 8, 'w': 12, 'x': 0, 'y': 20}, 'id': 14, 'options': {'legend': {'calcs': [], 'displayMode': 'list', 'placement': 'bottom'}, 'tooltip': {'mode': 'single', 'sort': 'none'}}, 'targets': [{'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'expr': 'histogram_quantile(0.95, rate(kimera_request_duration_seconds_bucket[5m]))', 'legendFormat': '95th percentile', 'refId': 'A'}, {'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'expr': 'histogram_quantile(0.99, rate(kimera_request_duration_seconds_bucket[5m]))', 'legendFormat': '99th percentile', 'refId': 'B'}], 'title': 'Request Latency (with 1s threshold)', 'type': 'timeseries'}, {'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'fieldConfig': {'defaults': {'color': {'mode': 'palette-classic'}, 'custom': {'axisLabel': '', 'axisPlacement': 'auto', 'barAlignment': 0, 'drawStyle': 'line', 'fillOpacity': 10, 'gradientMode': 'none', 'hideFrom': {'legend': False, 'tooltip': False, 'vis': False}, 'lineInterpolation': 'linear', 'lineWidth': 2, 'pointSize': 5, 'scaleDistribution': {'type': 'linear'}, 'showPoints': 'never', 'spanNulls': False, 'stacking': {'group': 'A', 'mode': 'none'}, 'thresholdsStyle': {'mode': 'line'}}, 'mappings': [], 'thresholds': {'mode': 'absolute', 'steps': [{'color': 'green', 'value': None}, {'color': 'yellow', 'value': 0.5}, {'color': 'red', 'value': 2}]}, 'unit': 's'}, 'overrides': []}, 'gridPos': {'h': 8, 'w': 12, 'x': 12, 'y': 20}, 'id': 15, 'options': {'legend': {'calcs': [], 'displayMode': 'list', 'placement': 'bottom'}, 'tooltip': {'mode': 'single', 'sort': 'none'}}, 'targets': [{'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'expr': 'histogram_quantile(0.95, rate(kimera_embedding_duration_seconds_bucket[5m]))', 'legendFormat': 'Embedding Latency 95th', 'refId': 'A'}], 'title': 'Embedding Generation Latency (with thresholds)', 'type': 'timeseries'}]` (from `kimera-alerts-dashboard.json`)
- `general`: `[{'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'fieldConfig': {'defaults': {'color': {'mode': 'palette-classic'}, 'custom': {'axisLabel': '', 'axisPlacement': 'auto', 'barAlignment': 0, 'drawStyle': 'line', 'fillOpacity': 0, 'gradientMode': 'none', 'hideFrom': {'legend': False, 'tooltip': False, 'vis': False}, 'lineInterpolation': 'linear', 'lineWidth': 1, 'pointSize': 5, 'scaleDistribution': {'type': 'linear'}, 'showPoints': 'auto', 'spanNulls': False, 'stacking': {'group': 'A', 'mode': 'none'}, 'thresholdsStyle': {'mode': 'off'}}, 'mappings': [], 'thresholds': {'mode': 'absolute', 'steps': [{'color': 'green', 'value': None}, {'color': 'red', 'value': 80}]}}, 'overrides': []}, 'gridPos': {'h': 8, 'w': 12, 'x': 0, 'y': 0}, 'id': 1, 'options': {'legend': {'calcs': [], 'displayMode': 'list', 'placement': 'bottom'}, 'tooltip': {'mode': 'single', 'sort': 'none'}}, 'targets': [{'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'expr': 'kimera_system_cpu_percent', 'interval': '', 'legendFormat': 'CPU Usage %', 'refId': 'A'}], 'title': 'Cognitive Field Wave Propagation', 'type': 'timeseries'}, {'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'fieldConfig': {'defaults': {'color': {'mode': 'thresholds'}, 'mappings': [], 'thresholds': {'mode': 'absolute', 'steps': [{'color': 'green', 'value': None}, {'color': 'red', 'value': 80}]}}, 'overrides': []}, 'gridPos': {'h': 8, 'w': 12, 'x': 12, 'y': 0}, 'id': 2, 'options': {'orientation': 'auto', 'reduceOptions': {'values': False, 'calcs': ['lastNotNull'], 'fields': ''}, 'showThresholdLabels': False, 'showThresholdMarkers': True}, 'pluginVersion': '8.0.0', 'targets': [{'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'expr': 'kimera_system_memory_percent', 'interval': '', 'legendFormat': 'Memory Usage %', 'refId': 'A'}], 'title': 'Field Resonance Amplitude', 'type': 'gauge'}, {'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'fieldConfig': {'defaults': {'color': {'mode': 'palette-classic'}, 'custom': {'axisLabel': '', 'axisPlacement': 'auto', 'barAlignment': 0, 'drawStyle': 'line', 'fillOpacity': 10, 'gradientMode': 'none', 'hideFrom': {'legend': False, 'tooltip': False, 'vis': False}, 'lineInterpolation': 'linear', 'lineWidth': 1, 'pointSize': 5, 'scaleDistribution': {'type': 'linear'}, 'showPoints': 'never', 'spanNulls': False, 'stacking': {'group': 'A', 'mode': 'none'}, 'thresholdsStyle': {'mode': 'off'}}, 'mappings': [], 'thresholds': {'mode': 'absolute', 'steps': [{'color': 'green', 'value': None}, {'color': 'red', 'value': 80}]}, 'unit': 'Hz'}, 'overrides': []}, 'gridPos': {'h': 8, 'w': 24, 'x': 0, 'y': 8}, 'id': 3, 'options': {'legend': {'calcs': [], 'displayMode': 'list', 'placement': 'bottom'}, 'tooltip': {'mode': 'single', 'sort': 'none'}}, 'targets': [{'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'expr': 'kimera_gpu_utilization_percent', 'interval': '', 'legendFormat': 'GPU Utilization %', 'refId': 'A'}], 'title': 'Cognitive Field Frequency Spectrum', 'type': 'timeseries'}, {'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'fieldConfig': {'defaults': {'color': {'mode': 'palette-classic'}, 'custom': {'hideFrom': {'legend': False, 'tooltip': False, 'vis': False}}, 'mappings': []}, 'overrides': []}, 'gridPos': {'h': 8, 'w': 12, 'x': 0, 'y': 16}, 'id': 4, 'options': {'legend': {'displayMode': 'list', 'placement': 'bottom'}, 'pieType': 'pie', 'reduceOptions': {'values': False, 'calcs': ['lastNotNull'], 'fields': ''}, 'tooltip': {'mode': 'single', 'sort': 'none'}}, 'targets': [{'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'expr': 'kimera_revolutionary_insights_total', 'interval': '', 'legendFormat': '{{pattern_type}}', 'refId': 'A'}], 'title': 'Field Interference Patterns', 'type': 'piechart'}, {'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'fieldConfig': {'defaults': {'color': {'mode': 'continuous-GrYlRd'}, 'custom': {'hideFrom': {'legend': False, 'tooltip': False, 'vis': False}}, 'mappings': [], 'thresholds': {'mode': 'absolute', 'steps': [{'color': 'green', 'value': None}, {'color': 'red', 'value': 80}]}}, 'overrides': []}, 'gridPos': {'h': 8, 'w': 12, 'x': 12, 'y': 16}, 'id': 5, 'options': {'cellGap': 1, 'cellValues': {'unit': 'short'}, 'color': {'exponent': 0.5, 'fill': 'dark-orange', 'mode': 'spectrum', 'reverse': False, 'scale': 'exponential', 'scheme': 'Spectral', 'steps': 64}, 'exemplars': {'color': 'rgba(255,0,255,0.7)'}, 'filterValues': {'le': 1e-09}, 'legend': {'show': False}, 'rowsFrame': {'layout': 'auto'}, 'tooltip': {'show': True, 'yHistogram': False}, 'yAxis': {'axisPlacement': 'left', 'reverse': False, 'unit': 'short'}}, 'pluginVersion': '8.0.0', 'targets': [{'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'expr': 'kimera_cognitive_cycles_total', 'format': 'heatmap', 'interval': '', 'legendFormat': '{{le}}', 'refId': 'A'}], 'title': 'Cognitive Field Spatial Distribution', 'type': 'heatmap'}]` (from `kimera-cognitive-field-dashboard.json`)
- `general`: `[{'collapsed': False, 'gridPos': {'h': 1, 'w': 24, 'x': 0, 'y': 0}, 'id': 1, 'panels': [], 'title': '🧠 Kimera Revolutionary Intelligence Overview', 'type': 'row'}, {'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'fieldConfig': {'defaults': {'color': {'mode': 'thresholds'}, 'mappings': [], 'thresholds': {'mode': 'absolute', 'steps': [{'color': 'green', 'value': None}, {'color': 'red', 'value': 80}]}, 'unit': 'short'}, 'overrides': []}, 'gridPos': {'h': 4, 'w': 6, 'x': 0, 'y': 1}, 'id': 2, 'options': {'colorMode': 'value', 'graphMode': 'area', 'justifyMode': 'center', 'orientation': 'auto', 'reduceOptions': {'values': False, 'calcs': ['lastNotNull'], 'fields': ''}, 'textMode': 'auto'}, 'pluginVersion': '10.0.0', 'targets': [{'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'expr': 'kimera_geoids_total', 'refId': 'A'}], 'title': 'Total Geoids', 'type': 'stat'}, {'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'fieldConfig': {'defaults': {'color': {'mode': 'thresholds'}, 'mappings': [], 'thresholds': {'mode': 'absolute', 'steps': [{'color': 'green', 'value': None}, {'color': 'red', 'value': 80}]}, 'unit': 'short'}, 'overrides': []}, 'gridPos': {'h': 4, 'w': 6, 'x': 6, 'y': 1}, 'id': 3, 'options': {'colorMode': 'value', 'graphMode': 'area', 'justifyMode': 'center', 'orientation': 'auto', 'reduceOptions': {'values': False, 'calcs': ['lastNotNull'], 'fields': ''}, 'textMode': 'auto'}, 'pluginVersion': '10.0.0', 'targets': [{'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'expr': 'kimera_scars_total', 'refId': 'A'}], 'title': 'Total SCARs', 'type': 'stat'}, {'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'fieldConfig': {'defaults': {'color': {'mode': 'thresholds'}, 'mappings': [], 'thresholds': {'mode': 'absolute', 'steps': [{'color': 'green', 'value': None}, {'color': 'yellow', 'value': 5}, {'color': 'red', 'value': 10}]}, 'unit': 'short'}, 'overrides': []}, 'gridPos': {'h': 4, 'w': 6, 'x': 12, 'y': 1}, 'id': 4, 'options': {'colorMode': 'value', 'graphMode': 'area', 'justifyMode': 'center', 'orientation': 'auto', 'reduceOptions': {'values': False, 'calcs': ['lastNotNull'], 'fields': ''}, 'textMode': 'auto'}, 'pluginVersion': '10.0.0', 'targets': [{'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'expr': 'rate(kimera_revolutionary_insights_total[5m]) * 60', 'refId': 'A'}], 'title': 'Revolutionary Insights/min', 'type': 'stat'}, {'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'fieldConfig': {'defaults': {'color': {'mode': 'thresholds'}, 'mappings': [], 'max': 1, 'min': 0, 'thresholds': {'mode': 'absolute', 'steps': [{'color': 'red', 'value': None}, {'color': 'yellow', 'value': 0.6}, {'color': 'green', 'value': 0.8}]}, 'unit': 'percentunit'}, 'overrides': []}, 'gridPos': {'h': 4, 'w': 6, 'x': 18, 'y': 1}, 'id': 5, 'options': {'colorMode': 'value', 'graphMode': 'area', 'justifyMode': 'center', 'orientation': 'auto', 'reduceOptions': {'values': False, 'calcs': ['lastNotNull'], 'fields': ''}, 'textMode': 'auto'}, 'pluginVersion': '10.0.0', 'targets': [{'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'expr': 'kimera_cognitive_cycles_total', 'refId': 'A'}], 'title': 'Cognitive Coherence', 'type': 'stat'}, {'collapsed': False, 'gridPos': {'h': 1, 'w': 24, 'x': 0, 'y': 5}, 'id': 6, 'panels': [], 'title': '🔥 GPU & AI Workload Performance', 'type': 'row'}, {'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'fieldConfig': {'defaults': {'color': {'mode': 'palette-classic'}, 'custom': {'axisLabel': '', 'axisPlacement': 'auto', 'barAlignment': 0, 'drawStyle': 'line', 'fillOpacity': 10, 'gradientMode': 'none', 'hideFrom': {'legend': False, 'tooltip': False, 'vis': False}, 'lineInterpolation': 'linear', 'lineWidth': 2, 'pointSize': 5, 'scaleDistribution': {'type': 'linear'}, 'showPoints': 'never', 'spanNulls': False, 'stacking': {'group': 'A', 'mode': 'none'}, 'thresholdsStyle': {'mode': 'off'}}, 'mappings': [], 'max': 100, 'min': 0, 'thresholds': {'mode': 'absolute', 'steps': [{'color': 'green', 'value': None}, {'color': 'red', 'value': 80}]}, 'unit': 'percent'}, 'overrides': []}, 'gridPos': {'h': 8, 'w': 12, 'x': 0, 'y': 6}, 'id': 7, 'options': {'legend': {'calcs': [], 'displayMode': 'list', 'placement': 'bottom'}, 'tooltip': {'mode': 'single', 'sort': 'none'}}, 'targets': [{'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'expr': 'kimera_gpu_utilization_percent', 'legendFormat': 'GPU {{gpu_id}} Utilization', 'refId': 'A'}], 'title': 'GPU Utilization', 'type': 'timeseries'}, {'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'fieldConfig': {'defaults': {'color': {'mode': 'palette-classic'}, 'custom': {'axisLabel': '', 'axisPlacement': 'auto', 'barAlignment': 0, 'drawStyle': 'line', 'fillOpacity': 10, 'gradientMode': 'none', 'hideFrom': {'legend': False, 'tooltip': False, 'vis': False}, 'lineInterpolation': 'linear', 'lineWidth': 2, 'pointSize': 5, 'scaleDistribution': {'type': 'linear'}, 'showPoints': 'never', 'spanNulls': False, 'stacking': {'group': 'A', 'mode': 'none'}, 'thresholdsStyle': {'mode': 'off'}}, 'mappings': [], 'thresholds': {'mode': 'absolute', 'steps': [{'color': 'green', 'value': None}, {'color': 'red', 'value': 80}]}, 'unit': 'celsius'}, 'overrides': []}, 'gridPos': {'h': 8, 'w': 12, 'x': 12, 'y': 6}, 'id': 8, 'options': {'legend': {'calcs': [], 'displayMode': 'list', 'placement': 'bottom'}, 'tooltip': {'mode': 'single', 'sort': 'none'}}, 'targets': [{'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'expr': 'kimera_gpu_temperature_celsius', 'legendFormat': 'GPU {{gpu_id}} Temperature', 'refId': 'A'}], 'title': 'GPU Temperature', 'type': 'timeseries'}, {'collapsed': False, 'gridPos': {'h': 1, 'w': 24, 'x': 0, 'y': 14}, 'id': 9, 'panels': [], 'title': '🌊 Cognitive Field Dynamics', 'type': 'row'}, {'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'fieldConfig': {'defaults': {'color': {'mode': 'palette-classic'}, 'custom': {'axisLabel': '', 'axisPlacement': 'auto', 'barAlignment': 0, 'drawStyle': 'line', 'fillOpacity': 10, 'gradientMode': 'none', 'hideFrom': {'legend': False, 'tooltip': False, 'vis': False}, 'lineInterpolation': 'linear', 'lineWidth': 2, 'pointSize': 5, 'scaleDistribution': {'type': 'linear'}, 'showPoints': 'never', 'spanNulls': False, 'stacking': {'group': 'A', 'mode': 'none'}, 'thresholdsStyle': {'mode': 'off'}}, 'mappings': [], 'thresholds': {'mode': 'absolute', 'steps': [{'color': 'green', 'value': None}, {'color': 'red', 'value': 80}]}, 'unit': 'ops'}, 'overrides': []}, 'gridPos': {'h': 8, 'w': 12, 'x': 0, 'y': 15}, 'id': 10, 'options': {'legend': {'calcs': [], 'displayMode': 'list', 'placement': 'bottom'}, 'tooltip': {'mode': 'single', 'sort': 'none'}}, 'targets': [{'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'expr': 'kimera_geoids_total', 'legendFormat': 'Geoid Creation Rate - {{vault}}', 'refId': 'A'}], 'title': 'Geoid Creation Rate', 'type': 'timeseries'}, {'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'fieldConfig': {'defaults': {'color': {'mode': 'palette-classic'}, 'custom': {'axisLabel': '', 'axisPlacement': 'auto', 'barAlignment': 0, 'drawStyle': 'line', 'fillOpacity': 10, 'gradientMode': 'none', 'hideFrom': {'legend': False, 'tooltip': False, 'vis': False}, 'lineInterpolation': 'linear', 'lineWidth': 2, 'pointSize': 5, 'scaleDistribution': {'type': 'linear'}, 'showPoints': 'never', 'spanNulls': False, 'stacking': {'group': 'A', 'mode': 'none'}, 'thresholdsStyle': {'mode': 'off'}}, 'mappings': [], 'thresholds': {'mode': 'absolute', 'steps': [{'color': 'green', 'value': None}, {'color': 'red', 'value': 80}]}, 'unit': 'ops'}, 'overrides': []}, 'gridPos': {'h': 8, 'w': 12, 'x': 12, 'y': 15}, 'id': 11, 'options': {'legend': {'calcs': [], 'displayMode': 'list', 'placement': 'bottom'}, 'tooltip': {'mode': 'single', 'sort': 'none'}}, 'targets': [{'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'expr': 'kimera_scars_total', 'legendFormat': 'SCAR Formation - {{type}}', 'refId': 'A'}], 'title': 'SCAR Formation Rate', 'type': 'timeseries'}, {'collapsed': False, 'gridPos': {'h': 1, 'w': 24, 'x': 0, 'y': 23}, 'id': 12, 'panels': [], 'title': '⚡ Performance & Latency Metrics', 'type': 'row'}, {'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'fieldConfig': {'defaults': {'color': {'mode': 'palette-classic'}, 'custom': {'axisLabel': '', 'axisPlacement': 'auto', 'barAlignment': 0, 'drawStyle': 'line', 'fillOpacity': 10, 'gradientMode': 'none', 'hideFrom': {'legend': False, 'tooltip': False, 'vis': False}, 'lineInterpolation': 'linear', 'lineWidth': 2, 'pointSize': 5, 'scaleDistribution': {'type': 'linear'}, 'showPoints': 'never', 'spanNulls': False, 'stacking': {'group': 'A', 'mode': 'none'}, 'thresholdsStyle': {'mode': 'off'}}, 'mappings': [], 'thresholds': {'mode': 'absolute', 'steps': [{'color': 'green', 'value': None}, {'color': 'red', 'value': 80}]}, 'unit': 's'}, 'overrides': []}, 'gridPos': {'h': 8, 'w': 12, 'x': 0, 'y': 24}, 'id': 13, 'options': {'legend': {'calcs': [], 'displayMode': 'list', 'placement': 'bottom'}, 'tooltip': {'mode': 'single', 'sort': 'none'}}, 'targets': [{'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'expr': 'histogram_quantile(0.95, rate(kimera_embedding_duration_seconds_bucket[5m]))', 'legendFormat': '95th percentile', 'refId': 'A'}, {'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'expr': 'histogram_quantile(0.50, rate(kimera_embedding_duration_seconds_bucket[5m]))', 'legendFormat': '50th percentile', 'refId': 'B'}], 'title': 'Embedding Generation Latency', 'type': 'timeseries'}, {'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'fieldConfig': {'defaults': {'color': {'mode': 'palette-classic'}, 'custom': {'axisLabel': '', 'axisPlacement': 'auto', 'barAlignment': 0, 'drawStyle': 'line', 'fillOpacity': 10, 'gradientMode': 'none', 'hideFrom': {'legend': False, 'tooltip': False, 'vis': False}, 'lineInterpolation': 'linear', 'lineWidth': 2, 'pointSize': 5, 'scaleDistribution': {'type': 'linear'}, 'showPoints': 'never', 'spanNulls': False, 'stacking': {'group': 'A', 'mode': 'none'}, 'thresholdsStyle': {'mode': 'off'}}, 'mappings': [], 'thresholds': {'mode': 'absolute', 'steps': [{'color': 'green', 'value': None}, {'color': 'red', 'value': 80}]}, 'unit': 'reqps'}, 'overrides': []}, 'gridPos': {'h': 8, 'w': 12, 'x': 12, 'y': 24}, 'id': 14, 'options': {'legend': {'calcs': [], 'displayMode': 'list', 'placement': 'bottom'}, 'tooltip': {'mode': 'single', 'sort': 'none'}}, 'targets': [{'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'expr': 'rate(kimera_requests_total[5m])', 'legendFormat': '{{method}} {{endpoint}} - {{status}}', 'refId': 'A'}], 'title': 'API Request Rate', 'type': 'timeseries'}, {'collapsed': False, 'gridPos': {'h': 1, 'w': 24, 'x': 0, 'y': 32}, 'id': 15, 'panels': [], 'title': '🖥️ System Resources', 'type': 'row'}, {'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'fieldConfig': {'defaults': {'color': {'mode': 'palette-classic'}, 'custom': {'axisLabel': '', 'axisPlacement': 'auto', 'barAlignment': 0, 'drawStyle': 'line', 'fillOpacity': 10, 'gradientMode': 'none', 'hideFrom': {'legend': False, 'tooltip': False, 'vis': False}, 'lineInterpolation': 'linear', 'lineWidth': 2, 'pointSize': 5, 'scaleDistribution': {'type': 'linear'}, 'showPoints': 'never', 'spanNulls': False, 'stacking': {'group': 'A', 'mode': 'none'}, 'thresholdsStyle': {'mode': 'off'}}, 'mappings': [], 'max': 100, 'min': 0, 'thresholds': {'mode': 'absolute', 'steps': [{'color': 'green', 'value': None}, {'color': 'red', 'value': 80}]}, 'unit': 'percent'}, 'overrides': []}, 'gridPos': {'h': 8, 'w': 8, 'x': 0, 'y': 33}, 'id': 16, 'options': {'legend': {'calcs': [], 'displayMode': 'list', 'placement': 'bottom'}, 'tooltip': {'mode': 'single', 'sort': 'none'}}, 'targets': [{'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'expr': 'kimera_system_cpu_percent', 'legendFormat': 'CPU Usage', 'refId': 'A'}], 'title': 'CPU Usage', 'type': 'timeseries'}, {'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'fieldConfig': {'defaults': {'color': {'mode': 'palette-classic'}, 'custom': {'axisLabel': '', 'axisPlacement': 'auto', 'barAlignment': 0, 'drawStyle': 'line', 'fillOpacity': 10, 'gradientMode': 'none', 'hideFrom': {'legend': False, 'tooltip': False, 'vis': False}, 'lineInterpolation': 'linear', 'lineWidth': 2, 'pointSize': 5, 'scaleDistribution': {'type': 'linear'}, 'showPoints': 'never', 'spanNulls': False, 'stacking': {'group': 'A', 'mode': 'none'}, 'thresholdsStyle': {'mode': 'off'}}, 'mappings': [], 'max': 100, 'min': 0, 'thresholds': {'mode': 'absolute', 'steps': [{'color': 'green', 'value': None}, {'color': 'red', 'value': 80}]}, 'unit': 'percent'}, 'overrides': []}, 'gridPos': {'h': 8, 'w': 8, 'x': 8, 'y': 33}, 'id': 17, 'options': {'legend': {'calcs': [], 'displayMode': 'list', 'placement': 'bottom'}, 'tooltip': {'mode': 'single', 'sort': 'none'}}, 'targets': [{'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'expr': 'kimera_system_memory_percent', 'legendFormat': 'Memory Usage', 'refId': 'A'}], 'title': 'Memory Usage', 'type': 'timeseries'}, {'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'fieldConfig': {'defaults': {'color': {'mode': 'palette-classic'}, 'custom': {'axisLabel': '', 'axisPlacement': 'auto', 'barAlignment': 0, 'drawStyle': 'line', 'fillOpacity': 10, 'gradientMode': 'none', 'hideFrom': {'legend': False, 'tooltip': False, 'vis': False}, 'lineInterpolation': 'linear', 'lineWidth': 2, 'pointSize': 5, 'scaleDistribution': {'type': 'linear'}, 'showPoints': 'never', 'spanNulls': False, 'stacking': {'group': 'A', 'mode': 'none'}, 'thresholdsStyle': {'mode': 'off'}}, 'mappings': [], 'thresholds': {'mode': 'absolute', 'steps': [{'color': 'green', 'value': None}, {'color': 'red', 'value': 80}]}, 'unit': 'binBps'}, 'overrides': []}, 'gridPos': {'h': 8, 'w': 8, 'x': 16, 'y': 33}, 'id': 18, 'options': {'legend': {'calcs': [], 'displayMode': 'list', 'placement': 'bottom'}, 'tooltip': {'mode': 'single', 'sort': 'none'}}, 'targets': [{'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'expr': 'rate(kimera_system_network_bytes_total[5m])', 'legendFormat': '{{direction}}', 'refId': 'A'}], 'title': 'Network I/O', 'type': 'timeseries'}, {'collapsed': False, 'gridPos': {'h': 1, 'w': 24, 'x': 0, 'y': 41}, 'id': 19, 'panels': [], 'title': '🧬 Selective Feedback & Revolutionary Intelligence', 'type': 'row'}, {'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'fieldConfig': {'defaults': {'color': {'mode': 'palette-classic'}, 'custom': {'axisLabel': '', 'axisPlacement': 'auto', 'barAlignment': 0, 'drawStyle': 'line', 'fillOpacity': 10, 'gradientMode': 'none', 'hideFrom': {'legend': False, 'tooltip': False, 'vis': False}, 'lineInterpolation': 'linear', 'lineWidth': 2, 'pointSize': 5, 'scaleDistribution': {'type': 'linear'}, 'showPoints': 'never', 'spanNulls': False, 'stacking': {'group': 'A', 'mode': 'none'}, 'thresholdsStyle': {'mode': 'off'}}, 'mappings': [], 'thresholds': {'mode': 'absolute', 'steps': [{'color': 'green', 'value': None}, {'color': 'red', 'value': 80}]}, 'unit': 'ops'}, 'overrides': []}, 'gridPos': {'h': 8, 'w': 12, 'x': 0, 'y': 42}, 'id': 20, 'options': {'legend': {'calcs': [], 'displayMode': 'list', 'placement': 'bottom'}, 'tooltip': {'mode': 'single', 'sort': 'none'}}, 'targets': [{'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'expr': 'rate(kimera_selective_feedback_ops_total[5m])', 'legendFormat': '{{domain}}', 'refId': 'A'}], 'title': 'Selective Feedback Operations', 'type': 'timeseries'}, {'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'fieldConfig': {'defaults': {'color': {'mode': 'palette-classic'}, 'custom': {'axisLabel': '', 'axisPlacement': 'auto', 'barAlignment': 0, 'drawStyle': 'line', 'fillOpacity': 10, 'gradientMode': 'none', 'hideFrom': {'legend': False, 'tooltip': False, 'vis': False}, 'lineInterpolation': 'linear', 'lineWidth': 2, 'pointSize': 5, 'scaleDistribution': {'type': 'linear'}, 'showPoints': 'never', 'spanNulls': False, 'stacking': {'group': 'A', 'mode': 'none'}, 'thresholdsStyle': {'mode': 'off'}}, 'mappings': [], 'max': 1, 'min': 0, 'thresholds': {'mode': 'absolute', 'steps': [{'color': 'green', 'value': None}, {'color': 'red', 'value': 80}]}, 'unit': 'percentunit'}, 'overrides': []}, 'gridPos': {'h': 8, 'w': 12, 'x': 12, 'y': 42}, 'id': 21, 'options': {'legend': {'calcs': [], 'displayMode': 'list', 'placement': 'bottom'}, 'tooltip': {'mode': 'single', 'sort': 'none'}}, 'targets': [{'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'expr': 'kimera_selective_feedback_accuracy', 'legendFormat': '{{domain}}', 'refId': 'A'}], 'title': 'Selective Feedback Accuracy', 'type': 'timeseries'}, {'collapsed': False, 'gridPos': {'h': 1, 'w': 24, 'x': 0, 'y': 50}, 'id': 22, 'panels': [], 'title': '🔥 Contradiction Engine & Thermodynamics', 'type': 'row'}, {'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'fieldConfig': {'defaults': {'color': {'mode': 'palette-classic'}, 'custom': {'axisLabel': '', 'axisPlacement': 'auto', 'barAlignment': 0, 'drawStyle': 'line', 'fillOpacity': 10, 'gradientMode': 'none', 'hideFrom': {'legend': False, 'tooltip': False, 'vis': False}, 'lineInterpolation': 'linear', 'lineWidth': 2, 'pointSize': 5, 'scaleDistribution': {'type': 'linear'}, 'showPoints': 'never', 'spanNulls': False, 'stacking': {'group': 'A', 'mode': 'none'}, 'thresholdsStyle': {'mode': 'off'}}, 'mappings': [], 'thresholds': {'mode': 'absolute', 'steps': [{'color': 'green', 'value': None}, {'color': 'red', 'value': 80}]}, 'unit': 'short'}, 'overrides': []}, 'gridPos': {'h': 8, 'w': 12, 'x': 0, 'y': 51}, 'id': 23, 'options': {'legend': {'calcs': [], 'displayMode': 'list', 'placement': 'bottom'}, 'tooltip': {'mode': 'single', 'sort': 'none'}}, 'targets': [{'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'expr': 'rate(kimera_contradictions_total[5m])', 'legendFormat': '{{type}}', 'refId': 'A'}], 'title': 'Contradiction Events', 'type': 'timeseries'}, {'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'fieldConfig': {'defaults': {'color': {'mode': 'palette-classic'}, 'custom': {'axisLabel': '', 'axisPlacement': 'auto', 'barAlignment': 0, 'drawStyle': 'line', 'fillOpacity': 10, 'gradientMode': 'none', 'hideFrom': {'legend': False, 'tooltip': False, 'vis': False}, 'lineInterpolation': 'linear', 'lineWidth': 2, 'pointSize': 5, 'scaleDistribution': {'type': 'linear'}, 'showPoints': 'never', 'spanNulls': False, 'stacking': {'group': 'A', 'mode': 'none'}, 'thresholdsStyle': {'mode': 'off'}}, 'mappings': [], 'thresholds': {'mode': 'absolute', 'steps': [{'color': 'green', 'value': None}, {'color': 'red', 'value': 80}]}, 'unit': 'short'}, 'overrides': []}, 'gridPos': {'h': 8, 'w': 12, 'x': 12, 'y': 51}, 'id': 24, 'options': {'legend': {'calcs': [], 'displayMode': 'list', 'placement': 'bottom'}, 'tooltip': {'mode': 'single', 'sort': 'none'}}, 'targets': [{'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'expr': 'kimera_thermodynamic_entropy', 'legendFormat': 'System Entropy', 'refId': 'A'}], 'title': 'Thermodynamic Entropy', 'type': 'timeseries'}, {'collapsed': False, 'gridPos': {'h': 1, 'w': 24, 'x': 0, 'y': 59}, 'id': 25, 'panels': [], 'title': '🚨 Alerts & Error Monitoring', 'type': 'row'}, {'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'fieldConfig': {'defaults': {'color': {'mode': 'palette-classic'}, 'custom': {'axisLabel': '', 'axisPlacement': 'auto', 'barAlignment': 0, 'drawStyle': 'line', 'fillOpacity': 10, 'gradientMode': 'none', 'hideFrom': {'legend': False, 'tooltip': False, 'vis': False}, 'lineInterpolation': 'linear', 'lineWidth': 2, 'pointSize': 5, 'scaleDistribution': {'type': 'linear'}, 'showPoints': 'never', 'spanNulls': False, 'stacking': {'group': 'A', 'mode': 'none'}, 'thresholdsStyle': {'mode': 'off'}}, 'mappings': [], 'thresholds': {'mode': 'absolute', 'steps': [{'color': 'green', 'value': None}, {'color': 'red', 'value': 80}]}, 'unit': 'short'}, 'overrides': []}, 'gridPos': {'h': 8, 'w': 24, 'x': 0, 'y': 60}, 'id': 26, 'options': {'legend': {'calcs': [], 'displayMode': 'list', 'placement': 'bottom'}, 'tooltip': {'mode': 'single', 'sort': 'none'}}, 'targets': [{'datasource': {'type': 'prometheus', 'uid': 'PBFA97CFB590B2093'}, 'expr': 'rate(kimera_errors_total[5m])', 'legendFormat': '{{error_type}} - {{component}}', 'refId': 'A'}], 'title': 'Error Rate by Component', 'type': 'timeseries'}]` (from `kimera-comprehensive-dashboard.json`)

---

### schemaVersion
**Conflicting values:**

- `general`: `37` (from `kimera-alerts-dashboard.json`)
- `general`: `27` (from `kimera-cognitive-field-dashboard.json`)
- `general`: `37` (from `kimera-comprehensive-dashboard.json`)

---

### tags
**Conflicting values:**

- `general`: `['kimera', 'alerts', 'health', 'monitoring']` (from `kimera-alerts-dashboard.json`)
- `general`: `['kimera', 'cognitive-field', 'dynamics']` (from `kimera-cognitive-field-dashboard.json`)
- `general`: `['kimera', 'swm', 'cognitive', 'ai', 'monitoring']` (from `kimera-comprehensive-dashboard.json`)

---

### templating
**Conflicting values:**

- `general`: `{'list': []}` (from `kimera-alerts-dashboard.json`)
- `general`: `{'list': []}` (from `kimera-cognitive-field-dashboard.json`)
- `general`: `{'list': [{'current': {'selected': False, 'text': '5m', 'value': '5m'}, 'hide': 0, 'includeAll': False, 'label': 'Time Range', 'multi': False, 'name': 'range', 'options': [{'selected': False, 'text': '1m', 'value': '1m'}, {'selected': True, 'text': '5m', 'value': '5m'}, {'selected': False, 'text': '15m', 'value': '15m'}, {'selected': False, 'text': '1h', 'value': '1h'}], 'query': '1m,5m,15m,1h', 'queryValue': '', 'skipUrlSync': False, 'type': 'custom'}]}` (from `kimera-comprehensive-dashboard.json`)

---

### time
**Conflicting values:**

- `general`: `{'from': 'now-15m', 'to': 'now'}` (from `kimera-alerts-dashboard.json`)
- `general`: `{'from': 'now-5m', 'to': 'now'}` (from `kimera-cognitive-field-dashboard.json`)
- `general`: `{'from': 'now-1h', 'to': 'now'}` (from `kimera-comprehensive-dashboard.json`)

---

### title
**Conflicting values:**

- `general`: `Kimera SWM - Critical Alerts & Health Dashboard` (from `kimera-alerts-dashboard.json`)
- `general`: `Kimera SWM - Cognitive Field Dynamics Dashboard` (from `kimera-cognitive-field-dashboard.json`)
- `general`: `Kimera SWM - Comprehensive Revolutionary Intelligence Dashboard` (from `kimera-comprehensive-dashboard.json`)

---

### uid
**Conflicting values:**

- `general`: `kimera-alerts` (from `kimera-alerts-dashboard.json`)
- `general`: `kimera-cognitive-field` (from `kimera-cognitive-field-dashboard.json`)
- `general`: `kimera-comprehensive` (from `kimera-comprehensive-dashboard.json`)

---

### version
**Conflicting values:**

- `general`: `1` (from `kimera-alerts-dashboard.json`)
- `general`: `1` (from `kimera-cognitive-field-dashboard.json`)
- `general`: `1` (from `kimera-comprehensive-dashboard.json`)
- `docker`: `3.8` (from `docker-compose-databases.yml`)
- `docker`: `3.8` (from `docker-compose.yml`)

---

### services
**Conflicting values:**

- `docker`: `{'postgres': {'image': 'pgvector/pgvector:pg16', 'container_name': 'kimera_postgres', 'environment': {'POSTGRES_USER': 'kimera', 'POSTGRES_PASSWORD': 'kimera_secure_pass_2025', 'POSTGRES_DB': 'kimera_swm', 'POSTGRES_INITDB_ARGS': '--encoding=UTF-8 --lc-collate=C --lc-ctype=C', 'POSTGRES_HOST_AUTH_METHOD': 'md5'}, 'ports': ['5432:5432'], 'volumes': ['kimera_postgres_data:/var/lib/postgresql/data', './init_db_fixed.sql:/docker-entrypoint-initdb.d/init.sql:ro'], 'healthcheck': {'test': ['CMD-SHELL', 'pg_isready -U kimera -d kimera_swm'], 'interval': '10s', 'timeout': '5s', 'retries': 5}, 'networks': ['kimera_network'], 'restart': 'unless-stopped'}, 'neo4j': {'image': 'neo4j:5.15-community', 'container_name': 'kimera_neo4j', 'environment': {'NEO4J_AUTH': 'neo4j/kimera_graph_pass_2025', 'NEO4J_dbms_default__database': 'kimera_graph', 'NEO4J_dbms_security_procedures_unrestricted': 'gds.*,apoc.*', 'NEO4J_dbms_security_procedures_allowlist': 'gds.*,apoc.*', 'NEO4J_PLUGINS': '["apoc", "graph-data-science"]', 'NEO4J_dbms_memory_heap_initial__size': '512M', 'NEO4J_dbms_memory_heap_max__size': '2G', 'NEO4J_dbms_memory_pagecache_size': '1G', 'NEO4J_dbms_jvm_additional': '-XX:+ExitOnOutOfMemoryError'}, 'ports': ['7474:7474', '7687:7687'], 'volumes': ['kimera_neo4j_data:/data', 'kimera_neo4j_logs:/logs', 'kimera_neo4j_import:/var/lib/neo4j/import', 'kimera_neo4j_plugins:/plugins'], 'healthcheck': {'test': ['CMD', 'cypher-shell', '-u', 'neo4j', '-p', 'kimera_graph_pass_2025', 'RETURN 1'], 'interval': '30s', 'timeout': '10s', 'retries': 5, 'start_period': '30s'}, 'networks': ['kimera_network'], 'restart': 'unless-stopped'}, 'redis': {'image': 'redis:7-alpine', 'container_name': 'kimera_redis', 'environment': {'REDIS_PASSWORD': 'kimera_cache_pass_2025'}, 'ports': ['6379:6379'], 'volumes': ['kimera_redis_data:/data'], 'command': 'redis-server --appendonly yes --requirepass kimera_cache_pass_2025', 'healthcheck': {'test': ['CMD', 'redis-cli', '--no-auth-warning', '-a', 'kimera_cache_pass_2025', 'ping'], 'interval': '10s', 'timeout': '5s', 'retries': 5}, 'networks': ['kimera_network'], 'restart': 'unless-stopped'}, 'adminer': {'image': 'adminer', 'container_name': 'kimera_adminer', 'ports': ['8080:8080'], 'environment': {'ADMINER_DEFAULT_SERVER': 'postgres'}, 'networks': ['kimera_network'], 'restart': 'unless-stopped', 'depends_on': ['postgres']}}` (from `docker-compose-databases.yml`)
- `docker`: `{'postgres': {'image': 'pgvector/pgvector:pg16', 'container_name': 'kimera_postgres', 'environment': {'POSTGRES_USER': 'kimera', 'POSTGRES_PASSWORD': 'kimera_secure_pass_2025', 'POSTGRES_DB': 'kimera_swm'}, 'ports': ['5432:5432'], 'volumes': ['kimera_postgres_data:/var/lib/postgresql/data', './init_db.sql:/docker-entrypoint-initdb.d/init.sql:ro'], 'healthcheck': {'test': ['CMD-SHELL', 'pg_isready -U kimera -d kimera_swm'], 'interval': '10s', 'timeout': '5s', 'retries': 5}, 'networks': ['kimera_network']}, 'redis': {'image': 'redis:7-alpine', 'container_name': 'kimera_redis', 'ports': ['6379:6379'], 'volumes': ['kimera_redis_data:/data'], 'command': 'redis-server --appendonly yes', 'healthcheck': {'test': ['CMD', 'redis-cli', 'ping'], 'interval': '10s', 'timeout': '5s', 'retries': 5}, 'networks': ['kimera_network']}, 'kimera_app': {'build': {'context': '../../', 'dockerfile': 'config/docker/Dockerfile'}, 'container_name': 'kimera_app', 'environment': {'KIMERA_DATABASE_URL': 'postgresql://kimera:kimera_secure_pass_2025@postgres:5432/kimera_swm', 'KIMERA_DB_POOL_SIZE': 20, 'REDIS_URL': 'redis://redis:6379', 'KIMERA_ENV': 'docker', 'KIMERA_SECRET_KEY': 'IJIhXeOxdqEbF_YdePBV8bg8JKegIujEpKs5av-29AA', 'KIMERA_LOG_LEVEL': 'INFO', 'KIMERA_HOST': '0.0.0.0', 'KIMERA_PORT': 8000, 'KIMERA_CORS_ORIGINS': '*'}, 'ports': ['8000:8000', '8001:8001', '8002:8002'], 'volumes': ['kimera_data:/app/data', 'kimera_logs:/app/logs', 'kimera_models:/app/models'], 'depends_on': {'postgres': {'condition': 'service_healthy'}, 'redis': {'condition': 'service_healthy'}}, 'healthcheck': {'test': ['CMD', 'curl', '-f', 'http://localhost:8000/health'], 'interval': '30s', 'timeout': '10s', 'retries': 3, 'start_period': '40s'}, 'networks': ['kimera_network'], 'restart': 'unless-stopped'}, 'prometheus': {'image': 'prom/prometheus:latest', 'container_name': 'kimera_prometheus', 'ports': ['9090:9090'], 'volumes': ['../../config/prometheus.yml:/etc/prometheus/prometheus.yml:ro', 'kimera_prometheus_data:/prometheus'], 'command': ['--config.file=/etc/prometheus/prometheus.yml', '--storage.tsdb.path=/prometheus', '--web.console.libraries=/etc/prometheus/console_libraries', '--web.console.templates=/etc/prometheus/consoles', '--storage.tsdb.retention.time=200h', '--web.enable-lifecycle'], 'networks': ['kimera_network']}}` (from `docker-compose.yml`)

---

### volumes
**Conflicting values:**

- `docker`: `{'kimera_postgres_data': {'name': 'kimera_postgres_data'}, 'kimera_neo4j_data': {'name': 'kimera_neo4j_data'}, 'kimera_neo4j_logs': {'name': 'kimera_neo4j_logs'}, 'kimera_neo4j_import': {'name': 'kimera_neo4j_import'}, 'kimera_neo4j_plugins': {'name': 'kimera_neo4j_plugins'}, 'kimera_redis_data': {'name': 'kimera_redis_data'}}` (from `docker-compose-databases.yml`)
- `docker`: `{'kimera_postgres_data': None, 'kimera_redis_data': None, 'kimera_data': None, 'kimera_logs': None, 'kimera_models': None, 'kimera_prometheus_data': None}` (from `docker-compose.yml`)

---

### networks
**Conflicting values:**

- `docker`: `{'kimera_network': {'name': 'kimera_network', 'driver': 'bridge'}}` (from `docker-compose-databases.yml`)
- `docker`: `{'kimera_network': {'driver': 'bridge'}}` (from `docker-compose.yml`)

---

### CDP_API_KEY_PRIVATE_KEY
**Conflicting values:**

- `trading`: `your_cdp_private_key_here` (from `kimera_cdp_config.env`)
- `trading`: `9268de76-b5f4-4683-b593-327fb2c19503` (from `kimera_cdp_live.env`)
- `environment`: `9268de76-b5f4-4683-b593-327fb2c19503` (from `kimera_max_profit_config.env`)

---

### CDP_NETWORK_ID
**Conflicting values:**

- `trading`: `base-sepolia  # Testnet for initial testing` (from `kimera_cdp_config.env`)
- `trading`: `base-sepolia` (from `kimera_cdp_live.env`)
- `environment`: `base-sepolia` (from `kimera_max_profit_config.env`)

---

### KIMERA_CDP_MAX_POSITION_SIZE
**Conflicting values:**

- `trading`: `0.1  # 10% max position` (from `kimera_cdp_config.env`)
- `trading`: `0.1` (from `kimera_cdp_live.env`)

---

### KIMERA_CDP_MIN_CONFIDENCE
**Conflicting values:**

- `trading`: `0.6     # Minimum confidence for trades` (from `kimera_cdp_config.env`)
- `trading`: `0.7` (from `kimera_cdp_live.env`)

---

### KIMERA_CDP_GAS_LIMIT
**Conflicting values:**

- `trading`: `200000       # Gas limit for transactions` (from `kimera_cdp_config.env`)
- `trading`: `200000` (from `kimera_cdp_live.env`)

---

### KIMERA_CDP_MAX_SLIPPAGE
**Conflicting values:**

- `trading`: `0.02      # 2% max slippage` (from `kimera_cdp_config.env`)
- `trading`: `0.02` (from `kimera_cdp_live.env`)

---

## Consolidated Structure

```
configs_consolidated/
├── development/
│   ├── config.yaml
│   ├── trading.yaml
│   ├── monitoring.yaml
│   └── database.yaml
├── production/
│   ├── config.yaml
│   ├── trading.yaml
│   ├── monitoring.yaml
│   └── database.yaml
├── testing/
│   └── config.yaml
├── shared/
│   ├── docker/
│   └── monitoring/
└── env/
    ├── development.env
    ├── production.env
    └── testing.env
```

## Usage

### Loading Configuration
```python
from src.config.unified_config import load_config

# Load environment-specific config
config = load_config('development')

# Load category-specific config
trading_config = load_config('development', 'trading')
```

### Environment Variables
```bash
# Load environment variables
source configs_consolidated/env/development.env
```

## Migration Notes

1. **Update import statements** to use unified config loader
2. **Test all configuration loading** in different environments
3. **Verify environment variable resolution**
4. **Update deployment scripts** to use consolidated configs
5. **Remove old configuration files** after verification

## Backup Location
Original configurations backed up to: `archive\2025-07-31_config_backup`