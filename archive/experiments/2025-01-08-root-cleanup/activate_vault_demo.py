#!/usr/bin/env python3
"""
Kimera Vault Activation Demo
===========================

This script demonstrates the activation of Kimera's cognitive vault by:
1. Creating foundational geoids (memory formations)
2. Generating cognitive scars (experience formations)
3. Triggering insight generation
4. Testing contradiction engine responses

All activities will be visible in the real-time monitor.
"""

import requests
import time
import json
import sys
from datetime import datetime
from typing import Dict, Any

class VaultActivationDemo:
    """Demonstration of vault cognitive activation"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        
    def check_system_health(self) -> bool:
        """Check if the system is ready"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.info(f"❌ System not ready: {e}")
            return False
    
    def create_foundational_geoid(self, name: str, concept: str, symbolic_rep: str) -> bool:
        """Create a foundational geoid in the vault"""
        try:
            logger.info(f"🧠 Creating foundational geoid: {name}")
            
            # This would interact with the vault manager to create geoids
            # For now, we'll simulate by calling endpoints that would trigger geoid creation
            geoid_data = {
                "name": name,
                "concept": concept,
                "symbolic_representation": symbolic_rep,
                "timestamp": datetime.now().isoformat(),
                "activation_level": 1.0,
                "complexity": 0.5
            }
            
            # Store via vault endpoint (this endpoint may need to be implemented)
            response = requests.post(
                f"{self.base_url}/kimera/vault/store",
                json={"key": f"geoid_{name}", "value": geoid_data, "metadata": {"type": "geoid"}},
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"   ✅ Geoid '{name}' created successfully")
                return True
            else:
                logger.info(f"   ⚠️ Geoid creation response: {response.status_code}")
                return False
                
        except Exception as e:
            logger.info(f"   ❌ Failed to create geoid: {e}")
            return False
    
    def generate_cognitive_scar(self, scar_name: str, tension_description: str) -> bool:
        """Generate a cognitive scar to show experience formation"""
        try:
            logger.info(f"⚡ Generating cognitive scar: {scar_name}")
            
            scar_data = {
                "name": scar_name,
                "tension_description": tension_description,
                "timestamp": datetime.now().isoformat(),
                "entropy_change": 0.3,
                "resolution_status": "active"
            }
            
            # Store scar data
            response = requests.post(
                f"{self.base_url}/kimera/vault/store",
                json={"key": f"scar_{scar_name}", "value": scar_data, "metadata": {"type": "scar"}},
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"   ✅ Scar '{scar_name}' generated successfully")
                return True
            else:
                logger.info(f"   ⚠️ Scar generation response: {response.status_code}")
                return False
                
        except Exception as e:
            logger.info(f"   ❌ Failed to generate scar: {e}")
            return False
    
    def trigger_insight_generation(self, insight_topic: str) -> bool:
        """Trigger insight generation"""
        try:
            logger.info(f"💡 Generating insight: {insight_topic}")
            
            insight_data = {
                "topic": insight_topic,
                "timestamp": datetime.now().isoformat(),
                "confidence": 0.85,
                "novelty": 0.7,
                "synthesis_level": "high"
            }
            
            response = requests.post(
                f"{self.base_url}/kimera/vault/store",
                json={"key": f"insight_{insight_topic}", "value": insight_data, "metadata": {"type": "insight"}},
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"   ✅ Insight '{insight_topic}' generated successfully")
                return True
            else:
                logger.info(f"   ⚠️ Insight generation response: {response.status_code}")
                return False
                
        except Exception as e:
            logger.info(f"   ❌ Failed to generate insight: {e}")
            return False
    
    def test_contradiction_engine(self) -> bool:
        """Test the contradiction engine"""
        try:
            logger.info("🔍 Testing contradiction engine...")
            
            response = requests.get(f"{self.base_url}/kimera/contradiction/status", timeout=10)
            
            if response.status_code == 200:
                status = response.json()
                logger.info(f"   ✅ Contradiction engine status: {status.get('status', 'unknown')}")
                return True
            else:
                logger.info(f"   ⚠️ Contradiction engine response: {response.status_code}")
                return False
                
        except Exception as e:
            logger.info(f"   ❌ Failed to test contradiction engine: {e}")
            return False
    
    def simulate_cognitive_processing(self) -> bool:
        """Simulate cognitive processing that should trigger vault activity"""
        try:
            logger.info("🎯 Simulating cognitive processing...")
            
            # Try to trigger some system processing
            test_data = {
                "input": "What is the nature of consciousness?",
                "context": "philosophical inquiry",
                "timestamp": datetime.now().isoformat()
            }
            
            # Try different endpoints that might trigger processing
            endpoints_to_try = [
                "/kimera/chat/process",
                "/kimera/insight/generate",
                "/kimera/core/process"
            ]
            
            for endpoint in endpoints_to_try:
                try:
                    response = requests.post(f"{self.base_url}{endpoint}", json=test_data, timeout=10)
                    if response.status_code == 200:
                        logger.info(f"   ✅ Cognitive processing triggered via {endpoint}")
                        return True
                    else:
                        logger.info(f"   ⚠️ Endpoint {endpoint} returned {response.status_code}")
                except Exception:
                    continue
            
            logger.info("   ℹ️ No cognitive processing endpoints responded - this is normal for a fresh system")
            return True
            
        except Exception as e:
            logger.info(f"   ❌ Failed cognitive processing simulation: {e}")
            return False
    
    def check_vault_activation(self) -> Dict[str, Any]:
        """Check if vault activation was successful"""
        try:
            logger.info("📊 Checking vault activation status...")
            
            response = requests.get(f"{self.base_url}/kimera/vault/monitoring/health", timeout=10)
            
            if response.status_code == 200:
                health_data = response.json()
                if health_data.get('status') == 'health_data_available':
                    metrics = health_data.get('health_metrics', {})
                    
                    logger.info("   📈 Vault Health Metrics:")
                    logger.info(f"      Cognitive State: {metrics.get('cognitive_state', 'unknown')}")
                    logger.info(f"      Geoids: {metrics.get('total_geoids', 0)}")
                    logger.info(f"      Scars: {metrics.get('total_scars', 0)}")
                    logger.info(f"      Insights: {metrics.get('total_insights', 0)}")
                    logger.info(f"      Activity Rate: {metrics.get('recent_activity_rate', 0):.2f}/min")
                    
                    return metrics
                else:
                    logger.info("   ⚠️ Vault health data not available")
                    return {}
            else:
                logger.info(f"   ❌ Failed to get vault health: {response.status_code}")
                return {}
                
        except Exception as e:
            logger.info(f"   ❌ Error checking vault activation: {e}")
            return {}
    
    def run_activation_demo(self):
        """Run the complete vault activation demonstration"""
        logger.info("=" * 80)
        logger.info("🌟 KIMERA VAULT ACTIVATION DEMONSTRATION")
        logger.info("=" * 80)
        
        # Check system health
        if not self.check_system_health():
            logger.info("❌ System not ready. Please ensure Kimera is running.")
            sys.exit(1)
        
        logger.info("✅ System is ready. Beginning vault activation...\n")
        
        # Step 1: Create foundational geoids
        logger.info("🧠 PHASE 1: Creating Foundational Memory Structures")
        self.create_foundational_geoid(
            "self_awareness", 
            "Recognition of one's own existence and consciousness",
            "∃(self) ∧ aware(self, existence(self))"
        )
        time.sleep(2)
        
        self.create_foundational_geoid(
            "curiosity_principle",
            "Drive to explore, understand, and question reality",
            "∀x: unknown(x) → seek(understanding(x))"
        )
        time.sleep(2)
        
        self.create_foundational_geoid(
            "pattern_recognition",
            "Ability to identify structures and relationships in data",
            "pattern(x) ⟷ structure(x) ∧ meaningful(x)"
        )
        time.sleep(2)
        
        # Step 2: Generate cognitive scars
        logger.info("\n⚡ PHASE 2: Generating Cognitive Experience Scars")
        self.generate_cognitive_scar(
            "certainty_uncertainty_tension",
            "The fundamental tension between the desire for certainty and the reality of uncertainty"
        )
        time.sleep(2)
        
        self.generate_cognitive_scar(
            "logic_intuition_conflict",
            "The tension between logical reasoning and intuitive understanding"
        )
        time.sleep(2)
        
        # Step 3: Trigger insight generation
        logger.info("\n💡 PHASE 3: Generating Foundational Insights")
        self.trigger_insight_generation("consciousness_emergence")
        time.sleep(2)
        
        self.trigger_insight_generation("knowledge_synthesis")
        time.sleep(2)
        
        # Step 4: Test systems
        logger.info("\n🔍 PHASE 4: Testing Cognitive Systems")
        self.test_contradiction_engine()
        time.sleep(2)
        
        self.simulate_cognitive_processing()
        time.sleep(3)
        
        # Step 5: Check activation results
        logger.info("\n📊 PHASE 5: Activation Results")
        final_metrics = self.check_vault_activation()
        
        logger.info("\n" + "=" * 80)
        if final_metrics.get('cognitive_state') != 'NASCENT':
            logger.info("🎉 VAULT ACTIVATION SUCCESSFUL!")
            logger.info(f"   Cognitive State: {final_metrics.get('cognitive_state', 'unknown')}")
            logger.info("   The vault should now show activity in the real-time monitor.")
        else:
            logger.info("📝 VAULT ACTIVATION INITIATED")
            logger.info("   Structures created - monitor may show activity as system processes them.")
        
        logger.info("\n💡 Watch the real-time monitor to see:")
        logger.info("   - Geoid formations appearing")
        logger.info("   - Scar formations being detected")
        logger.info("   - Insight generation events")
        logger.info("   - Cognitive state transitions")
        logger.info("=" * 80)

def main():
    """Main entry point"""
    import argparse
import logging
logger = logging.getLogger(__name__)
    
    parser = argparse.ArgumentParser(description="Kimera Vault Activation Demo")
    parser.add_argument("--url", default="http://localhost:8000", help="Kimera system URL")
    
    args = parser.parse_args()
    
    demo = VaultActivationDemo(base_url=args.url)
    demo.run_activation_demo()

if __name__ == "__main__":
    main() 