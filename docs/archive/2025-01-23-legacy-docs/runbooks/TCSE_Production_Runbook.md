# Kimera TCSE Production Runbook

**Version: 1.0**
**Last Updated:** July 2025
**System Owner:** Kimera Core Development Team

---

## 1. Overview

This document provides operational procedures for monitoring, managing, and troubleshooting the Thermodynamic Cognitive Signal Evolution (TCSE) subsystem within the Kimera SWM platform. This system is responsible for the real-time evolution of cognitive signals based on thermodynamic principles and is a critical component of Kimera's advanced intelligence architecture.

## 2. System Architecture & Key Components

- **`tcse_system_integration`**: The main pipeline orchestrating all TCSE engines.
- **`thermodynamic_signal_evolution`**: The core engine performing the evolution calculations.
- **`cognitive_gpu_kernels`**: The CUDA kernels that execute the computations on the GPU.
- **`automated_tcse_tuner`**: The self-optimization engine that adjusts parameters in real-time.
- **`tcse_monitoring`**: The dashboard providing real-time metrics.
- **`tcse_config.yaml`**: The master configuration file for the entire TCSE subsystem.

## 3. Monitoring Procedures

The primary tool for monitoring is the `TCSignalMonitoringDashboard`. It provides a real-time snapshot of system health across three key areas.

### 3.1. Interpreting Dashboard Metrics

| Section         | Metric                             | Normal Range | Critical Threshold | Meaning & Action                                                                                             |
|-----------------|------------------------------------|--------------|--------------------|--------------------------------------------------------------------------------------------------------------|
| **Performance** | `gpu_utilization_percent`          | 80-95%       | < 70%              | If low, system is underutilized. The auto-tuner should increase `batch_size`. If persistently low, check data pipeline. |
|                 | `thermal_budget_remaining_percent` | > 20%        | < 10%              | If critical, system is overheating. Auto-tuner will decrease `evolution_rate`. If it persists, check physical cooling. |
|                 | `memory_usage_gb`                  | < 90% of total| > 95% of total    | High memory usage can lead to instability. The system should be restarted if this persists.                    |
| **Signal Evo**  | `thermodynamic_compliance_percent` | > 99.0%      | < 95.0%            | A drop indicates potential bugs or data corruption. Escalate to the development team immediately.            |
|                 | `signals_processed_per_second`     | > 100        | < 50               | Low throughput indicates a performance bottleneck. Check GPU utilization and logs for errors.                  |
| **Consciousness**| `average_consciousness_score`      | 0.6 - 0.9    | < 0.5              | A sustained drop may indicate a degradation in cognitive function. Monitor and report to the research team. |
|                 | `consciousness_events_detected`    | > 0          | -                  | This is an informational metric for tracking emergent events.                                                 |

## 4. Configuration Management

All TCSE parameters are managed via `config/tcse_config.yaml`. Changes to this file are picked up by the `AutomatedTCSEHyperparameterTuner` and applied to the running system. **Manual edits should be made with extreme caution.**

### 4.1. Key Configuration Parameters

| Parameter                           | Section               | Description                                                                    | Default | Tuning Impact                                                                |
|-------------------------------------|-----------------------|--------------------------------------------------------------------------------|---------|------------------------------------------------------------------------------|
| `enabled`                           | `tcse`                | Master switch to enable or disable the entire TCSE system.                     | `true`  | Disabling stops all TCSE processing.                                         |
| `mode`                              | `tcse`                | "conservative", "balanced", "aggressive". Influences auto-tuner behavior.      | `balanced`| Changes the risk profile of the auto-tuner.                                    |
| `batch_size`                        | `signal_evolution`    | Number of signals processed in a single batch on the GPU.                      | `32`    | **Increased** by tuner when GPU is underutilized. Higher = more throughput.   |
| `evolution_rate`                    | `signal_evolution`    | A factor controlling the speed of signal evolution.                            | `0.8`   | **Decreased** by tuner when thermal budget is low. Higher = more CPU/GPU load. |
| `entropy_accuracy_threshold`        | `thermodynamic_constraints` | Minimum accuracy for entropy calculations during validation.               | `0.985` | Do not change without consulting development. Lowering may hide errors.      |
| `consciousness_threshold`           | `consciousness_detection` | The threshold for detecting a "consciousness event".                           | `0.7`   | Informational. Adjust based on research requirements.                        |

## 5. Alerting & Escalation

The system generates alerts via standard logging. These should be ingested by a centralized logging platform (e.g., Splunk, ELK).

### 5.1. Critical Alerts

| Alert Message                      | Log Level  | Meaning                                               | **Immediate Action Required**                                                                  |
|------------------------------------|------------|-------------------------------------------------------|------------------------------------------------------------------------------------------------|
| `ALERT: Thermal budget critical!`  | `CRITICAL` | GPU is close to overheating. Performance is throttled.  | 1. **Verify physical cooling** of the server hardware. <br> 2. Monitor the dashboard to ensure the auto-tuner reduces load. <br> 3. If it persists for >15 mins, consider a graceful restart. |
| `ALERT: Thermodynamic compliance low!` | `ERROR`    | Signal evolutions are failing validation checks.      | 1. **Escalate immediately** to the L2/L3 development team. <br> 2. This may indicate a serious bug or data corruption. <br> 3. Do not restart the system unless instructed. |
| `Failed to allocate new GPU memory`| `CRITICAL` | The GPU has run out of memory.                        | 1. The service will likely become unstable. <br> 2. Perform a **graceful restart** of the application. <br> 3. If this reoccurs frequently, file a bug report to investigate memory leaks. |

## 6. Common Troubleshooting

- **Issue:** Low `signals_processed_per_second` and low `gpu_utilization_percent`.
  - **Cause:** The data pipeline feeding signals into the TCSE system may be a bottleneck.
  - **Action:** Check the status of upstream services and data queues.

- **Issue:** `AutomatedTCSEHyperparameterTuner` is not making adjustments.
  - **Cause:** The tuner only runs periodically (default: 5 minutes). It may also determine that no changes are necessary.
  - **Action:** Check the logs for messages like "No tuning adjustments needed". If there are no logs from the tuner, it may have failed to initialize. Restart the service. 