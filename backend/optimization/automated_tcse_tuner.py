"""
Automated TCSE Hyperparameter Tuner
===================================

This module implements a sophisticated automated tuning engine for the TCSE
system. It monitors real-time performance and adjusts system parameters
dynamically to meet specified performance goals (e.g., maximizing throughput
while respecting a thermal budget).
"""

from __future__ import annotations
import yaml
import logging
from typing import Dict, Any, Optional
from backend.layer_2_governance.monitoring.tcse_monitoring import TCSignalMonitoringDashboard
import time
import os

logger = logging.getLogger(__name__)

class AutomatedTCSEHyperparameterTuner:
    """
    Dynamically tunes TCSE hyperparameters based on real-time monitoring data.
    """
    def __init__(self, 
                 config_path: str = 'config/tcse_config.yaml',
                 monitoring_dashboard: Optional[TCSignalMonitoringDashboard] = None,
                 tuning_interval_seconds: int = 300):
        """
        Initializes the automated tuner.

        Args:
            config_path: Path to the TCSE configuration file.
            monitoring_dashboard: An instance of the monitoring dashboard.
            tuning_interval_seconds: How often to run the tuning logic.
        """
        self.config_path = config_path
        self.dashboard = monitoring_dashboard or TCSignalMonitoringDashboard()
        self.tuning_interval = tuning_interval_seconds
        self.last_tune_time = 0
        self.config_cache = self._load_config()
        
        if not self.config_cache:
            raise ValueError(f"TCSE configuration could not be loaded from {config_path}")
            
        logger.info(f"🤖 Automated TCSE Hyperparameter Tuner initialized. Tuning interval: {self.tuning_interval}s.")

    def _load_config(self) -> Optional[Dict[str, Any]]:
        """Loads the YAML configuration file."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Configuration file not found at {self.config_path}")
            return None
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration at {self.config_path}: {e}")
            return None

    def _save_config(self, config: Dict[str, Any]):
        """Saves the configuration dictionary back to the YAML file."""
        try:
            with open(self.config_path, 'w') as f:
                yaml.dump(config, f, indent=2, default_flow_style=False)
            logger.info(f"Successfully saved updated configuration to {self.config_path}")
        except IOError as e:
            logger.error(f"Failed to write configuration to {self.config_path}: {e}")

    def _apply_tuning_logic(self, metrics: Dict[str, Any]):
        """
        The core tuning logic that adjusts parameters based on metrics.
        This is a simplified rule-based approach. More advanced versions could use
        Bayesian optimization, reinforcement learning, etc.
        """
        # Ensure we have the latest config
        current_config = self._load_config()
        if not current_config:
            logger.error("Cannot apply tuning logic without a valid configuration.")
            return

        perf_metrics = metrics.get("performance", {})
        evo_metrics = metrics.get("signal_evolution", {})
        
        # Goal: Maintain high throughput while respecting thermal limits.
        gpu_util = perf_metrics.get("gpu_utilization_percent", 90)
        thermal_budget = perf_metrics.get("thermal_budget_remaining_percent", 50)
        
        # --- Tuning Rule 1: Thermal Throttling ---
        if thermal_budget < 15.0:
            logger.warning(f"Thermal budget critical ({thermal_budget:.1f}%). Reducing evolution rate.")
            current_config['tcse']['signal_evolution']['evolution_rate'] = max(0.1, current_config['tcse']['signal_evolution']['evolution_rate'] * 0.8)
        
        # --- Tuning Rule 2: Performance Scaling ---
        elif gpu_util < 70.0 and thermal_budget > 40.0:
            logger.info(f"GPU underutilized ({gpu_util:.1f}%). Increasing batch size.")
            current_config['tcse']['signal_evolution']['batch_size'] = min(512, current_config['tcse']['signal_evolution']['batch_size'] * 2)

        # --- Tuning Rule 3: Efficiency Recovery ---
        elif thermal_budget > 50.0 and current_config['tcse']['signal_evolution']['evolution_rate'] < 0.8:
             logger.info(f"Thermal budget healthy ({thermal_budget:.1f}%). Restoring evolution rate.")
             current_config['tcse']['signal_evolution']['evolution_rate'] = min(0.8, current_config['tcse']['signal_evolution']['evolution_rate'] * 1.1)

        # Save if changes were made
        if current_config != self.config_cache:
            self._save_config(current_config)
            self.config_cache = current_config
        else:
            logger.info("No tuning adjustments needed at this time.")

    def run_tuning_cycle(self):
        """
        Executes a single cycle of monitoring and tuning if the interval has passed.
        """
        current_time = time.time()
        if current_time - self.last_tune_time > self.tuning_interval:
            logger.info("--- Starting new tuning cycle ---")
            latest_metrics = self.dashboard.get_real_time_signal_metrics()
            self._apply_tuning_logic(latest_metrics)
            self.last_tune_time = current_time
            logger.info("--- Tuning cycle complete ---")

# Example of how this might be used in a main application loop
if __name__ == '__main__':
    # Ensure a dummy config exists for testing
    if not os.path.exists('config'):
        os.makedirs('config')
    with open('config/tcse_config.yaml', 'w') as f:
        yaml.dump({
            'tcse': {
                'signal_evolution': {
                    'batch_size': 32,
                    'evolution_rate': 0.8
                }
            }
        }, f)
        
    tuner = AutomatedTCSEHyperparameterTuner()
    
    # Simulate running in a service loop
    for i in range(5):
        print(f"\nMain loop iteration {i+1}")
        tuner.run_tuning_cycle()
        # In a real app, we'd just call tuner.run_tuning_cycle() periodically
        time.sleep(1) # Short sleep for demo; real interval is much longer
    
    # Manually change metrics to trigger different logic for demo
    class MockDashboard(TCSignalMonitoringDashboard):
        def get_real_time_signal_metrics(self):
            metrics = super().get_real_time_signal_metrics()
            metrics['performance']['gpu_utilization_percent'] = 50 
            metrics['performance']['thermal_budget_remaining_percent'] = 60
            print("--- Injecting low GPU utilization metrics ---")
            return metrics
            
    tuner.dashboard = MockDashboard()
    tuner.last_tune_time = 0 # Force re-run
    tuner.run_tuning_cycle()
    
    class MockDashboardCritical(TCSignalMonitoringDashboard):
        def get_real_time_signal_metrics(self):
            metrics = super().get_real_time_signal_metrics()
            metrics['performance']['thermal_budget_remaining_percent'] = 5
            print("--- Injecting critical thermal metrics ---")
            return metrics
            
    tuner.dashboard = MockDashboardCritical()
    tuner.last_tune_time = 0 # Force re-run
    tuner.run_tuning_cycle() 