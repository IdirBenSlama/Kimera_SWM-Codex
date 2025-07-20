#!/usr/bin/env python3
"""
Trigger Kimera Cognitive Activity
=================================

This script triggers real cognitive activity using actual working endpoints
to generate insights and process contradictions, which should be visible
in the vault monitoring system.
"""

import requests
import time
import json
from datetime import datetime

class CognitiveActivityTrigger:
    """Triggers actual cognitive activity in Kimera"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def check_system_ready(self) -> bool:
        """Check if system is ready"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def get_baseline_metrics(self) -> dict:
        """Get baseline vault metrics"""
        metrics = {}
        try:
            # Get geoid count
            response = requests.get(f"{self.base_url}/kimera/vault/geoids/count", timeout=5)
            if response.status_code == 200:
                metrics['geoids'] = response.json().get('geoid_count', 0)
            
            # Get scar count
            response = requests.get(f"{self.base_url}/kimera/vault/scars/count", timeout=5)
            if response.status_code == 200:
                metrics['scars'] = response.json().get('scar_count', 0)
            
            # Get vault health
            response = requests.get(f"{self.base_url}/kimera/vault/monitoring/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                if health_data.get('status') == 'health_data_available':
                    vault_metrics = health_data.get('health_metrics', {})
                    metrics['cognitive_state'] = vault_metrics.get('cognitive_state', 'unknown')
                    metrics['activity_rate'] = vault_metrics.get('recent_activity_rate', 0)
                    metrics['insights'] = vault_metrics.get('total_insights', 0)
        except Exception as e:
            print(f"⚠️ Error getting baseline metrics: {e}")
        
        return metrics
    
    def trigger_insight_generation(self) -> bool:
        """Trigger insight generation"""
        print("💡 Triggering insight generation...")
        
        try:
            # Try auto-generate insights first
            response = requests.post(f"{self.base_url}/kimera/insights/auto_generate", timeout=15)
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Auto-generated insights: {result}")
                return True
            else:
                print(f"   ⚠️ Auto-generate failed: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Auto-generate error: {e}")
        
        try:
            # Try regular insight generation
            insight_request = {
                "topic": "consciousness_emergence",
                "context": "exploring the nature of self-awareness and cognitive emergence",
                "depth": "deep"
            }
            
            response = requests.post(
                f"{self.base_url}/kimera/insights/generate",
                json=insight_request,
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Generated insight: {result}")
                return True
            else:
                print(f"   ⚠️ Insight generation failed: {response.status_code}")
                try:
                    error_detail = response.json()
                    print(f"      Error: {error_detail}")
                except:
                    print(f"      Response: {response.text[:200]}")
        except Exception as e:
            print(f"   ❌ Insight generation error: {e}")
        
        return False
    
    def trigger_contradiction_processing(self) -> bool:
        """Trigger contradiction processing"""
        print("🔍 Triggering contradiction processing...")
        
        try:
            # Create a contradiction scenario
            contradiction_data = {
                "statement_a": "All knowledge is certain and absolute",
                "statement_b": "All knowledge is provisional and subject to revision",
                "context": "epistemological_paradox",
                "urgency": "high"
            }
            
            response = requests.post(
                f"{self.base_url}/kimera/contradiction/process/contradictions",
                json=contradiction_data,
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Processed contradiction: {result}")
                return True
            else:
                print(f"   ⚠️ Contradiction processing failed: {response.status_code}")
                try:
                    error_detail = response.json()
                    print(f"      Error: {error_detail}")
                except:
                    print(f"      Response: {response.text[:200]}")
        except Exception as e:
            print(f"   ❌ Contradiction processing error: {e}")
        
        return False
    
    def trigger_statistical_analysis(self) -> bool:
        """Trigger statistical analysis"""
        print("📊 Triggering statistical analysis...")
        
        try:
            response = requests.post(f"{self.base_url}/kimera/statistics/analyze", timeout=15)
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Statistical analysis: {result}")
                return True
            else:
                print(f"   ⚠️ Statistical analysis failed: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Statistical analysis error: {e}")
        
        return False
    
    def check_activity_changes(self, baseline: dict) -> dict:
        """Check for changes in activity"""
        print("📈 Checking for activity changes...")
        
        current_metrics = self.get_baseline_metrics()
        changes = {}
        
        for key, baseline_value in baseline.items():
            current_value = current_metrics.get(key, baseline_value)
            if current_value != baseline_value:
                changes[key] = {
                    'before': baseline_value,
                    'after': current_value,
                    'delta': current_value - baseline_value if isinstance(current_value, (int, float)) else 'changed'
                }
        
        if changes:
            print("   🎉 Activity changes detected:")
            for key, change in changes.items():
                print(f"      {key}: {change['before']} → {change['after']}")
        else:
            print("   📝 No activity changes detected yet")
        
        return changes
    
    def run_cognitive_activation(self):
        """Run comprehensive cognitive activation"""
        print("=" * 80)
        print("🧠 KIMERA COGNITIVE ACTIVITY ACTIVATION")
        print("=" * 80)
        
        if not self.check_system_ready():
            print("❌ System not ready")
            return False
        
        print("✅ System ready. Starting cognitive activation...\n")
        
        # Get baseline metrics
        print("📊 Getting baseline metrics...")
        baseline = self.get_baseline_metrics()
        print(f"   Baseline: {baseline}")
        print()
        
        # Trigger various cognitive activities
        activities_attempted = 0
        activities_successful = 0
        
        # 1. Insight Generation
        if self.trigger_insight_generation():
            activities_successful += 1
        activities_attempted += 1
        time.sleep(3)
        
        # 2. Contradiction Processing
        if self.trigger_contradiction_processing():
            activities_successful += 1
        activities_attempted += 1
        time.sleep(3)
        
        # 3. Statistical Analysis
        if self.trigger_statistical_analysis():
            activities_successful += 1
        activities_attempted += 1
        time.sleep(3)
        
        # Check for changes
        print("\n📈 RESULTS ANALYSIS")
        changes = self.check_activity_changes(baseline)
        
        # Summary
        print("\n" + "=" * 80)
        if activities_successful > 0:
            print(f"🎉 COGNITIVE ACTIVATION SUCCESSFUL!")
            print(f"   Successfully triggered {activities_successful}/{activities_attempted} activities")
            if changes:
                print(f"   Detected {len(changes)} metric changes")
            else:
                print("   Monitor may show delayed activity - check real-time monitor")
        else:
            print("⚠️ COGNITIVE ACTIVATION PARTIAL")
            print("   No activities successfully triggered")
            print("   System may need initialization or different approach")
        
        print("\n💡 Monitor the real-time vault monitor for:")
        print("   - Geoid formations from insight generation")
        print("   - Scar formations from contradiction processing")
        print("   - Cognitive state transitions")
        print("   - Activity rate increases")
        print("=" * 80)
        
        return activities_successful > 0

def main():
    """Main entry point"""
    trigger = CognitiveActivityTrigger()
    trigger.run_cognitive_activation()

if __name__ == "__main__":
    main() 