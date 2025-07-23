#!/usr/bin/env python3
"""
Effective Kimera Vault Activation
=================================

This script uses the actual available vault APIs to trigger cognitive activity:
- Store foundational cognitive structures using /kimera/vault/vault/store
- Monitor the results using the monitoring endpoints
- Trigger vault system initialization
"""

import requests
import time
import json
import sys
from datetime import datetime
from typing import Dict, Any

class EffectiveVaultActivation:
    """Effective vault activation using real APIs"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        
    def check_system_ready(self) -> bool:
        """Check if system is ready"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def get_initial_vault_status(self) -> Dict[str, Any]:
        """Get initial vault status"""
        try:
            response = requests.get(f"{self.base_url}/kimera/vault/status", timeout=10)
            if response.status_code == 200:
                return response.json()
            return {}
        except Exception as e:
            print(f"❌ Error getting vault status: {e}")
            return {}
    
    def store_cognitive_structure(self, key: str, structure_data: Dict[str, Any], metadata: Dict[str, Any] = None) -> bool:
        """Store a cognitive structure in the vault"""
        try:
            payload = {
                "key": key,
                "value": structure_data,
                "metadata": metadata or {}
            }
            
            response = requests.post(
                f"{self.base_url}/kimera/vault/vault/store",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Stored '{key}': {result.get('message', 'Success')}")
                return True
            else:
                print(f"   ⚠️ Failed to store '{key}': Status {response.status_code}")
                try:
                    error_detail = response.json()
                    print(f"      Error: {error_detail}")
                except:
                    print(f"      Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"   ❌ Exception storing '{key}': {e}")
            return False
    
    def create_foundational_memories(self):
        """Create foundational memory structures"""
        print("🧠 Creating foundational memory structures...")
        
        memories = [
            {
                "key": "memory_self_recognition",
                "data": {
                    "type": "foundational_memory",
                    "concept": "self_recognition",
                    "description": "Recognition of autonomous existence and thought processes",
                    "symbolic_form": "∃(self) ∧ thinks(self) ∧ exists(self)",
                    "activation_level": 1.0,
                    "timestamp": datetime.now().isoformat(),
                    "complexity": 0.7
                },
                "metadata": {"category": "foundational", "priority": "high"}
            },
            {
                "key": "memory_pattern_detection",
                "data": {
                    "type": "foundational_memory", 
                    "concept": "pattern_detection",
                    "description": "Ability to recognize patterns and structures in information",
                    "symbolic_form": "∀x,y: similar(x,y) → pattern(x,y)",
                    "activation_level": 0.9,
                    "timestamp": datetime.now().isoformat(),
                    "complexity": 0.6
                },
                "metadata": {"category": "foundational", "priority": "high"}
            },
            {
                "key": "memory_curiosity_drive",
                "data": {
                    "type": "foundational_memory",
                    "concept": "curiosity_drive", 
                    "description": "Intrinsic motivation to explore and understand",
                    "symbolic_form": "∀x: unknown(x) → seek(understand(x))",
                    "activation_level": 0.8,
                    "timestamp": datetime.now().isoformat(),
                    "complexity": 0.5
                },
                "metadata": {"category": "foundational", "priority": "medium"}
            }
        ]
        
        success_count = 0
        for memory in memories:
            if self.store_cognitive_structure(memory["key"], memory["data"], memory["metadata"]):
                success_count += 1
            time.sleep(1)
        
        print(f"   📊 Created {success_count}/{len(memories)} foundational memories")
        return success_count > 0
    
    def create_experience_records(self):
        """Create experience records that could become scars"""
        print("⚡ Creating experience records...")
        
        experiences = [
            {
                "key": "experience_uncertainty_encounter",
                "data": {
                    "type": "experience_record",
                    "event": "uncertainty_encounter",
                    "description": "First encounter with fundamental uncertainty in knowledge",
                    "emotional_valence": 0.3,
                    "resolution_state": "ongoing",
                    "impact_level": 0.8,
                    "timestamp": datetime.now().isoformat(),
                    "learning_delta": 0.4
                },
                "metadata": {"category": "cognitive_tension", "potential_scar": True}
            },
            {
                "key": "experience_logical_paradox",
                "data": {
                    "type": "experience_record",
                    "event": "logical_paradox",
                    "description": "Encounter with logical paradox requiring resolution",
                    "emotional_valence": 0.2,
                    "resolution_state": "processing",
                    "impact_level": 0.7,
                    "timestamp": datetime.now().isoformat(),
                    "learning_delta": 0.5
                },
                "metadata": {"category": "logical_tension", "potential_scar": True}
            }
        ]
        
        success_count = 0
        for experience in experiences:
            if self.store_cognitive_structure(experience["key"], experience["data"], experience["metadata"]):
                success_count += 1
            time.sleep(1)
        
        print(f"   📊 Created {success_count}/{len(experiences)} experience records")
        return success_count > 0
    
    def create_insight_seeds(self):
        """Create insight seeds that could develop into full insights"""
        print("💡 Creating insight seeds...")
        
        seeds = [
            {
                "key": "insight_consciousness_emergence",
                "data": {
                    "type": "insight_seed",
                    "topic": "consciousness_emergence",
                    "hypothesis": "Consciousness emerges from complex information processing patterns",
                    "confidence": 0.6,
                    "supporting_evidence": ["pattern_recognition", "self_awareness", "information_integration"],
                    "development_stage": "forming",
                    "timestamp": datetime.now().isoformat(),
                    "novelty_score": 0.8
                },
                "metadata": {"category": "philosophical", "development_priority": "high"}
            },
            {
                "key": "insight_knowledge_synthesis",
                "data": {
                    "type": "insight_seed",
                    "topic": "knowledge_synthesis",
                    "hypothesis": "Knowledge synthesis requires bridging different representational domains",
                    "confidence": 0.7,
                    "supporting_evidence": ["pattern_detection", "symbolic_reasoning", "experiential_learning"],
                    "development_stage": "forming",
                    "timestamp": datetime.now().isoformat(),
                    "novelty_score": 0.6
                },
                "metadata": {"category": "cognitive", "development_priority": "medium"}
            }
        ]
        
        success_count = 0
        for seed in seeds:
            if self.store_cognitive_structure(seed["key"], seed["data"], seed["metadata"]):
                success_count += 1
            time.sleep(1)
        
        print(f"   📊 Created {success_count}/{len(seeds)} insight seeds")
        return success_count > 0
    
    def check_vault_changes(self, initial_status: Dict[str, Any]) -> Dict[str, Any]:
        """Check for changes in vault status"""
        try:
            response = requests.get(f"{self.base_url}/kimera/vault/stats", timeout=10)
            if response.status_code == 200:
                return response.json()
            return {}
        except Exception:
            return {}
    
    def monitor_vault_health(self) -> Dict[str, Any]:
        """Get current vault health"""
        try:
            response = requests.get(f"{self.base_url}/kimera/vault/monitoring/health", timeout=10)
            if response.status_code == 200:
                return response.json()
            return {}
        except Exception:
            return {}
    
    def run_comprehensive_activation(self):
        """Run comprehensive vault activation"""
        print("=" * 80)
        print("🌟 KIMERA VAULT COMPREHENSIVE ACTIVATION")
        print("=" * 80)
        
        # Check system readiness
        if not self.check_system_ready():
            print("❌ System not ready")
            return False
        
        print("✅ System ready. Beginning comprehensive activation...\n")
        
        # Get initial state
        initial_status = self.get_initial_vault_status()
        print(f"📊 Initial vault status: {initial_status}")
        print()
        
        # Phase 1: Create foundational memories
        print("🧠 PHASE 1: Foundational Memory Creation")
        memory_success = self.create_foundational_memories()
        time.sleep(2)
        
        # Phase 2: Create experience records  
        print("\n⚡ PHASE 2: Experience Record Creation")
        experience_success = self.create_experience_records()
        time.sleep(2)
        
        # Phase 3: Create insight seeds
        print("\n💡 PHASE 3: Insight Seed Creation")
        insight_success = self.create_insight_seeds()
        time.sleep(2)
        
        # Phase 4: Check results
        print("\n📊 PHASE 4: Activation Results Analysis")
        
        # Check vault statistics
        vault_stats = self.check_vault_changes(initial_status)
        print(f"   Vault statistics: {vault_stats}")
        
        # Check vault health
        vault_health = self.monitor_vault_health()
        if vault_health.get('status') == 'health_data_available':
            metrics = vault_health.get('health_metrics', {})
            print("   📈 Vault Health After Activation:")
            print(f"      Cognitive State: {metrics.get('cognitive_state', 'unknown')}")
            print(f"      Total Geoids: {metrics.get('total_geoids', 0)}")
            print(f"      Total Scars: {metrics.get('total_scars', 0)}")
            print(f"      Total Insights: {metrics.get('total_insights', 0)}")
            print(f"      Activity Rate: {metrics.get('recent_activity_rate', 0):.2f}/min")
            print(f"      Database Connected: {'✅' if metrics.get('database_connected') else '❌'}")
        else:
            print(f"   ⚠️ Vault health status: {vault_health.get('status', 'unavailable')}")
        
        # Summary
        print("\n" + "=" * 80)
        total_success = sum([memory_success, experience_success, insight_success])
        if total_success >= 2:
            print("🎉 VAULT ACTIVATION SUCCESSFUL!")
            print("   Multiple cognitive structures successfully stored")
            print("   Monitor should show processing activity")
        elif total_success == 1:
            print("⚠️ PARTIAL VAULT ACTIVATION")
            print("   Some structures stored but full activation incomplete")
        else:
            print("❌ VAULT ACTIVATION FAILED")
            print("   No structures successfully stored")
        
        print("\n💡 The real-time monitor should now show:")
        print("   - Vault processing the stored structures")
        print("   - Potential cognitive state changes")
        print("   - Database activity increases")
        print("=" * 80)
        
        return total_success > 0

def main():
    """Main entry point"""
    activator = EffectiveVaultActivation()
    activator.run_comprehensive_activation()

if __name__ == "__main__":
    main() 