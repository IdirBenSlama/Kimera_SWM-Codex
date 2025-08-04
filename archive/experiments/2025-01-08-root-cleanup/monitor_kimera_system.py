#!/usr/bin/env python3
"""
Kimera System Real-Time Monitor Client
=====================================

This client continuously monitors the running Kimera system and provides
real-time updates on system health, vault activity, and cognitive processes.
"""

import time
import requests
import json
import sys
from datetime import datetime
from typing import Dict, Any, Optional

class KimeraSystemMonitor:
    """Real-time monitor for the Kimera system"""
    
    def __init__(self, base_url: str = "http://localhost:8000", update_interval: int = 5):
        self.base_url = base_url
        self.update_interval = update_interval
        self.last_activity_count = 0
        self.last_geoid_count = 0
        self.last_scar_count = 0
        self.last_insight_count = 0
        
    def wait_for_system_ready(self, timeout: int = 120):
        """Wait for the Kimera system to be ready"""
        logger.info("🔍 Waiting for Kimera system to start...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.base_url}/health", timeout=5)
                if response.status_code == 200:
                    logger.info("✅ Kimera system is ready!")
                    return True
            except Exception:
                pass
            
            logger.info("⏳ System starting...", end="\r")
            time.sleep(2)
        
        logger.info(f"\n❌ System did not start within {timeout} seconds")
        return False
    
    def get_system_health(self) -> Optional[Dict[str, Any]]:
        """Get system health information"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.info(f"❌ Failed to get system health: {e}")
        return None
    
    def get_monitoring_health(self) -> Optional[Dict[str, Any]]:
        """Get monitoring system health"""
        try:
            response = requests.get(f"{self.base_url}/monitoring/health", timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.info(f"❌ Failed to get monitoring health: {e}")
        return None
    
    def get_vault_health(self) -> Optional[Dict[str, Any]]:
        """Get vault health information"""
        try:
            response = requests.get(f"{self.base_url}/kimera/vault/monitoring/health", timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.info(f"❌ Failed to get vault health: {e}")
        return None
    
    def start_vault_monitoring(self) -> bool:
        """Start vault monitoring if not already running"""
        try:
            # Check if monitoring is already running
            status_response = requests.get(f"{self.base_url}/kimera/vault/monitoring/status", timeout=5)
            if status_response.status_code == 200:
                status = status_response.json()
                if status.get('is_monitoring', False):
                    logger.info("📊 Vault monitoring already active")
                    return True
            
            # Start monitoring
            response = requests.post(f"{self.base_url}/kimera/vault/monitoring/start", timeout=10)
            if response.status_code == 200:
                result = response.json()
                logger.info("✅ Vault monitoring started successfully")
                logger.info(f"   📈 Monitoring interval: {result.get('monitoring_interval', 'unknown')}s")
                logger.info(f"   🏥 Health check interval: {result.get('health_check_interval', 'unknown')}s")
                return True
            else:
                logger.info(f"⚠️ Failed to start vault monitoring: {response.text}")
                return False
        except Exception as e:
            logger.info(f"❌ Error starting vault monitoring: {e}")
            return False
    
    def get_recent_activities(self) -> Optional[Dict[str, Any]]:
        """Get recent vault activities"""
        try:
            response = requests.get(f"{self.base_url}/kimera/vault/monitoring/activities?limit=10", timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.info(f"❌ Failed to get recent activities: {e}")
        return None
    
    def display_system_overview(self):
        """Display comprehensive system overview"""
        logger.info("\n" + "=" * 80)
        logger.info("🌟 KIMERA SWM SYSTEM OVERVIEW")
        logger.info("=" * 80)
        
        # System Health
        health = self.get_system_health()
        if health:
            logger.info(f"🏥 System Status: {health.get('status', 'unknown')}")
            logger.info(f"🚀 GPU Available: {'✅' if health.get('gpu_available') else '❌'}")
            if health.get('gpu_name'):
                logger.info(f"   GPU: {health.get('gpu_name')}")
        
        # Monitoring Health
        monitoring = self.get_monitoring_health()
        if monitoring:
            logger.info(f"📊 Monitoring: {monitoring.get('status', 'unknown')}")
            components = monitoring.get('components', {})
            
            # Engine status
            engine_components = {k: v for k, v in components.items() if k.startswith('engine_')}
            operational = sum(1 for v in engine_components.values() if v == 'healthy')
            total = len(engine_components)
            logger.info(f"⚙️ Engines: {operational}/{total} operational")
        
        # Vault Health
        vault_health = self.get_vault_health()
        if vault_health and vault_health.get('status') == 'health_data_available':
            metrics = vault_health.get('health_metrics', {})
            logger.info("🧠 Vault Cognitive State:")
            logger.info(f"   State: {metrics.get('cognitive_state', 'unknown')}")
            logger.info(f"   Geoids: {metrics.get('total_geoids', 0)}")
            logger.info(f"   Scars: {metrics.get('total_scars', 0)}")
            logger.info(f"   Insights: {metrics.get('total_insights', 0)}")
            logger.info(f"   Activity Rate: {metrics.get('recent_activity_rate', 0):.2f}/min")
            logger.info(f"   DB Connected: {'✅' if metrics.get('database_connected') else '❌'}")
            logger.info(f"   DB Latency: {metrics.get('database_latency_ms', 0):.1f}ms")
            
            # Store current counts for comparison
            self.last_geoid_count = metrics.get('total_geoids', 0)
            self.last_scar_count = metrics.get('total_scars', 0)
            self.last_insight_count = metrics.get('total_insights', 0)
        else:
            logger.info("🧠 Vault: Health data not available")
        
        logger.info("=" * 80)
    
    def monitor_real_time_activity(self):
        """Monitor real-time vault activity"""
        logger.info(f"📊 Starting real-time monitoring (updates every {self.update_interval}s)")
        logger.info("📝 Press Ctrl+C to stop monitoring\n")
        
        try:
            while True:
                # Get current vault health
                vault_health = self.get_vault_health()
                
                if vault_health and vault_health.get('status') == 'health_data_available':
                    metrics = vault_health.get('health_metrics', {})
                    
                    current_geoid_count = metrics.get('total_geoids', 0)
                    current_scar_count = metrics.get('total_scars', 0)
                    current_insight_count = metrics.get('total_insights', 0)
                    activity_rate = metrics.get('recent_activity_rate', 0)
                    cognitive_state = metrics.get('cognitive_state', 'unknown')
                    
                    # Check for changes
                    changes_detected = False
                    
                    if current_geoid_count > self.last_geoid_count:
                        delta = current_geoid_count - self.last_geoid_count
                        logger.info(f"🧠 NEW GEOID FORMATIONS: +{delta} (total: {current_geoid_count})")
                        self.last_geoid_count = current_geoid_count
                        changes_detected = True
                    
                    if current_scar_count > self.last_scar_count:
                        delta = current_scar_count - self.last_scar_count
                        logger.info(f"⚡ NEW SCAR FORMATIONS: +{delta} (total: {current_scar_count})")
                        self.last_scar_count = current_scar_count
                        changes_detected = True
                    
                    if current_insight_count > self.last_insight_count:
                        delta = current_insight_count - self.last_insight_count
                        logger.info(f"💡 NEW INSIGHTS GENERATED: +{delta} (total: {current_insight_count})")
                        self.last_insight_count = current_insight_count
                        changes_detected = True
                    
                    # Show periodic status if no changes
                    if not changes_detected and activity_rate > 0:
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        logger.info(f"[{timestamp}] 📊 {cognitive_state} | Activity: {activity_rate:.2f}/min | "
                              f"G:{current_geoid_count} S:{current_scar_count} I:{current_insight_count}")
                    
                    # Get recent activities
                    activities = self.get_recent_activities()
                    if activities and activities.get('count', 0) > self.last_activity_count:
                        recent_activities = activities.get('activities', [])
                        new_activities = recent_activities[:activities.get('count', 0) - self.last_activity_count]
                        
                        for activity in new_activities:
                            activity_type = activity.get('activity_type', 'unknown')
                            metadata = activity.get('metadata', {})
                            
                            if 'formation' in activity_type:
                                logger.info(f"   🔄 {activity_type}: {metadata}")
                        
                        self.last_activity_count = activities.get('count', 0)
                
                else:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    logger.info(f"[{timestamp}] ⏳ Waiting for vault health data...")
                
                time.sleep(self.update_interval)
                
        except KeyboardInterrupt:
            logger.info("\n🛑 Monitoring stopped by user")
        except Exception as e:
            logger.info(f"\n❌ Monitoring error: {e}")
    
    def run_monitor(self):
        """Run the complete monitoring process"""
        # Wait for system to be ready
        if not self.wait_for_system_ready():
            sys.exit(1)
        
        # Start vault monitoring
        self.start_vault_monitoring()
        
        # Display initial overview
        time.sleep(5)  # Give monitoring time to initialize
        self.display_system_overview()
        
        # Start real-time monitoring
        self.monitor_real_time_activity()

def main():
    """Main entry point"""
    import argparse
import logging
logger = logging.getLogger(__name__)
    
    parser = argparse.ArgumentParser(description="Kimera System Real-Time Monitor")
    parser.add_argument("--url", default="http://localhost:8000", help="Kimera system URL")
    parser.add_argument("--interval", type=int, default=5, help="Update interval in seconds")
    
    args = parser.parse_args()
    
    monitor = KimeraSystemMonitor(base_url=args.url, update_interval=args.interval)
    monitor.run_monitor()

if __name__ == "__main__":
    main() 