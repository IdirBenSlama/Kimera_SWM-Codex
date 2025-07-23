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
        print("🔍 Waiting for Kimera system to start...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.base_url}/health", timeout=5)
                if response.status_code == 200:
                    print("✅ Kimera system is ready!")
                    return True
            except Exception:
                pass
            
            print("⏳ System starting...", end="\r")
            time.sleep(2)
        
        print(f"\n❌ System did not start within {timeout} seconds")
        return False
    
    def get_system_health(self) -> Optional[Dict[str, Any]]:
        """Get system health information"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"❌ Failed to get system health: {e}")
        return None
    
    def get_monitoring_health(self) -> Optional[Dict[str, Any]]:
        """Get monitoring system health"""
        try:
            response = requests.get(f"{self.base_url}/monitoring/health", timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"❌ Failed to get monitoring health: {e}")
        return None
    
    def get_vault_health(self) -> Optional[Dict[str, Any]]:
        """Get vault health information"""
        try:
            response = requests.get(f"{self.base_url}/kimera/vault/monitoring/health", timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"❌ Failed to get vault health: {e}")
        return None
    
    def start_vault_monitoring(self) -> bool:
        """Start vault monitoring if not already running"""
        try:
            # Check if monitoring is already running
            status_response = requests.get(f"{self.base_url}/kimera/vault/monitoring/status", timeout=5)
            if status_response.status_code == 200:
                status = status_response.json()
                if status.get('is_monitoring', False):
                    print("📊 Vault monitoring already active")
                    return True
            
            # Start monitoring
            response = requests.post(f"{self.base_url}/kimera/vault/monitoring/start", timeout=10)
            if response.status_code == 200:
                result = response.json()
                print("✅ Vault monitoring started successfully")
                print(f"   📈 Monitoring interval: {result.get('monitoring_interval', 'unknown')}s")
                print(f"   🏥 Health check interval: {result.get('health_check_interval', 'unknown')}s")
                return True
            else:
                print(f"⚠️ Failed to start vault monitoring: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Error starting vault monitoring: {e}")
            return False
    
    def get_recent_activities(self) -> Optional[Dict[str, Any]]:
        """Get recent vault activities"""
        try:
            response = requests.get(f"{self.base_url}/kimera/vault/monitoring/activities?limit=10", timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"❌ Failed to get recent activities: {e}")
        return None
    
    def display_system_overview(self):
        """Display comprehensive system overview"""
        print("\n" + "=" * 80)
        print("🌟 KIMERA SWM SYSTEM OVERVIEW")
        print("=" * 80)
        
        # System Health
        health = self.get_system_health()
        if health:
            print(f"🏥 System Status: {health.get('status', 'unknown')}")
            print(f"🚀 GPU Available: {'✅' if health.get('gpu_available') else '❌'}")
            if health.get('gpu_name'):
                print(f"   GPU: {health.get('gpu_name')}")
        
        # Monitoring Health
        monitoring = self.get_monitoring_health()
        if monitoring:
            print(f"📊 Monitoring: {monitoring.get('status', 'unknown')}")
            components = monitoring.get('components', {})
            
            # Engine status
            engine_components = {k: v for k, v in components.items() if k.startswith('engine_')}
            operational = sum(1 for v in engine_components.values() if v == 'healthy')
            total = len(engine_components)
            print(f"⚙️ Engines: {operational}/{total} operational")
        
        # Vault Health
        vault_health = self.get_vault_health()
        if vault_health and vault_health.get('status') == 'health_data_available':
            metrics = vault_health.get('health_metrics', {})
            print("🧠 Vault Cognitive State:")
            print(f"   State: {metrics.get('cognitive_state', 'unknown')}")
            print(f"   Geoids: {metrics.get('total_geoids', 0)}")
            print(f"   Scars: {metrics.get('total_scars', 0)}")
            print(f"   Insights: {metrics.get('total_insights', 0)}")
            print(f"   Activity Rate: {metrics.get('recent_activity_rate', 0):.2f}/min")
            print(f"   DB Connected: {'✅' if metrics.get('database_connected') else '❌'}")
            print(f"   DB Latency: {metrics.get('database_latency_ms', 0):.1f}ms")
            
            # Store current counts for comparison
            self.last_geoid_count = metrics.get('total_geoids', 0)
            self.last_scar_count = metrics.get('total_scars', 0)
            self.last_insight_count = metrics.get('total_insights', 0)
        else:
            print("🧠 Vault: Health data not available")
        
        print("=" * 80)
    
    def monitor_real_time_activity(self):
        """Monitor real-time vault activity"""
        print(f"📊 Starting real-time monitoring (updates every {self.update_interval}s)")
        print("📝 Press Ctrl+C to stop monitoring\n")
        
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
                        print(f"🧠 NEW GEOID FORMATIONS: +{delta} (total: {current_geoid_count})")
                        self.last_geoid_count = current_geoid_count
                        changes_detected = True
                    
                    if current_scar_count > self.last_scar_count:
                        delta = current_scar_count - self.last_scar_count
                        print(f"⚡ NEW SCAR FORMATIONS: +{delta} (total: {current_scar_count})")
                        self.last_scar_count = current_scar_count
                        changes_detected = True
                    
                    if current_insight_count > self.last_insight_count:
                        delta = current_insight_count - self.last_insight_count
                        print(f"💡 NEW INSIGHTS GENERATED: +{delta} (total: {current_insight_count})")
                        self.last_insight_count = current_insight_count
                        changes_detected = True
                    
                    # Show periodic status if no changes
                    if not changes_detected and activity_rate > 0:
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        print(f"[{timestamp}] 📊 {cognitive_state} | Activity: {activity_rate:.2f}/min | "
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
                                print(f"   🔄 {activity_type}: {metadata}")
                        
                        self.last_activity_count = activities.get('count', 0)
                
                else:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(f"[{timestamp}] ⏳ Waiting for vault health data...")
                
                time.sleep(self.update_interval)
                
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
        except Exception as e:
            print(f"\n❌ Monitoring error: {e}")
    
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
    
    parser = argparse.ArgumentParser(description="Kimera System Real-Time Monitor")
    parser.add_argument("--url", default="http://localhost:8000", help="Kimera system URL")
    parser.add_argument("--interval", type=int, default=5, help="Update interval in seconds")
    
    args = parser.parse_args()
    
    monitor = KimeraSystemMonitor(base_url=args.url, update_interval=args.interval)
    monitor.run_monitor()

if __name__ == "__main__":
    main() 