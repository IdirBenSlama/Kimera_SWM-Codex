#!/usr/bin/env python3
"""
Comprehensive System Verification Script
Tests ALL components and routes in the Kimera system
"""

import requests
import json
from datetime import datetime
from typing import Dict, List, Tuple

BASE_URL = "http://localhost:8000"

class ComponentVerifier:
    def __init__(self):
        self.results = []
        self.passed = 0
        self.failed = 0
        
    def check_endpoint(self, method: str, path: str, data=None, expected_status=200) -> Tuple[bool, str]:
        """Check a single endpoint and return (success, message)"""
        url = f"{BASE_URL}{path}"
        try:
            if method == "GET":
                response = requests.get(url)
            elif method == "POST":
                response = requests.post(url, json=data)
            else:
                return False, f"Unsupported method: {method}"
            
            success = response.status_code == expected_status
            if success:
                self.passed += 1
                return True, f"✅ {method} {path} - Status: {response.status_code}"
            else:
                self.failed += 1
                return False, f"❌ {method} {path} - Status: {response.status_code} (Expected: {expected_status})"
        except Exception as e:
            self.failed += 1
            return False, f"❌ {method} {path} - Error: {str(e)}"
    
    def verify_component(self, name: str, checks: List[Tuple[str, str, dict, int]]):
        """Verify a component with multiple endpoint checks"""
        print(f"\n{'='*60}")
        print(f"🔍 {name}")
        print(f"{'='*60}")
        
        for method, path, data, expected in checks:
            success, message = self.check_endpoint(method, path, data, expected)
            print(message)
            self.results.append({
                "component": name,
                "endpoint": f"{method} {path}",
                "success": success,
                "message": message
            })

def main():
    verifier = ComponentVerifier()
    
    print("=" * 80)
    print("KIMERA COMPREHENSIVE SYSTEM VERIFICATION")
    print(f"Time: {datetime.now().isoformat()}")
    print("=" * 80)
    
    # Core System
    verifier.verify_component("CORE SYSTEM", [
        ("GET", "/", None, 200),
        ("GET", "/kimera/status", None, 200),
        ("GET", "/kimera/system/health", None, 200),
        ("GET", "/kimera/system/health/detailed", None, 200),
        ("GET", "/kimera/system/stability", None, 200),
        ("GET", "/kimera/system/utilization_stats", None, 200),
    ])
    
    # GPU Foundation
    verifier.verify_component("GPU FOUNDATION", [
        ("GET", "/kimera/system/gpu_foundation", None, 200),
    ])
    
    # Embedding & Vector Operations
    verifier.verify_component("EMBEDDING & VECTORS", [
        ("POST", "/kimera/embed", {"text": "Test embedding"}, 200),
        ("POST", "/kimera/semantic_features", {"text": "Test semantic extraction"}, 200),
    ])
    
    # Geoid Operations
    verifier.verify_component("GEOID OPERATIONS", [
        ("POST", "/kimera/geoids", {
            "semantic_features": {"test": 1.0, "verification": 0.8},
            "symbolic_content": {"description": "Test geoid"},
            "metadata": {"type": "test"}
        }, 200),
        ("GET", "/kimera/geoids/search?query=test&limit=5", None, 200),
    ])
    
    # SCAR Operations
    verifier.verify_component("SCAR OPERATIONS", [
        ("GET", "/kimera/scars/search?query=test&limit=3", None, 200),
    ])
    
    # Vault Manager
    verifier.verify_component("VAULT MANAGER", [
        ("GET", "/kimera/vault/stats", None, 200),
        ("GET", "/kimera/vault/geoids/recent?limit=5", None, 200),
        ("GET", "/kimera/vault/scars/recent?limit=5", None, 200),
    ])
    
    # Statistical Engine
    verifier.verify_component("STATISTICAL ENGINE", [
        ("GET", "/kimera/statistics/capabilities", None, 200),
        ("POST", "/kimera/statistics/analyze", {
            "data": [1, 2, 3, 4, 5],
            "analysis_type": "basic"
        }, 200),
    ])
    
    # Thermodynamic Engine
    verifier.verify_component("THERMODYNAMIC ENGINE", [
        ("GET", "/kimera/thermodynamic_engine", None, 200),
        ("POST", "/kimera/thermodynamics/analyze", {
            "geoid_ids": ["test_id"],
            "analysis_type": "temperature"
        }, 200),
    ])
    
    # Contradiction Engine
    verifier.verify_component("CONTRADICTION ENGINE", [
        ("GET", "/kimera/contradiction_engine", None, 200),
        ("POST", "/kimera/contradiction/detect", {
            "geoid_pairs": [["id1", "id2"]]
        }, 200),
    ])
    
    # Insight Engine
    verifier.verify_component("INSIGHT ENGINE", [
        ("GET", "/kimera/insights/status", None, 200),
        ("POST", "/kimera/insights/generate_simple", {
            "context": {"topic": "test"},
            "depth": "shallow"
        }, 200),
    ])
    
    # Cognitive Control
    verifier.verify_component("COGNITIVE CONTROL", [
        ("GET", "/kimera/cognitive-control/health", None, 200),
        ("GET", "/kimera/cognitive-control/system/status", None, 200),
        ("GET", "/kimera/cognitive-control/context/status", None, 200),
        ("GET", "/kimera/cognitive-control/profiler/status", None, 200),
        ("GET", "/kimera/cognitive-control/security/status", None, 200),
        ("POST", "/kimera/cognitive-control/context/configure", {
            "processing_level": "standard"
        }, 200),
        ("GET", "/kimera/cognitive-control/context/presets/standard", None, 200),
    ])
    
    # Advanced Routes (if available)
    print("\n" + "="*80)
    print("🚀 CHECKING ADVANCED COMPONENTS")
    print("="*80)
    
    # Monitoring Routes
    verifier.verify_component("MONITORING SYSTEM", [
        ("GET", "/kimera/monitoring/status", None, 200),
        ("GET", "/kimera/monitoring/integration/status", None, 200),
        ("GET", "/kimera/monitoring/engines/status", None, 200),
    ])
    
    # Revolutionary Intelligence
    verifier.verify_component("REVOLUTIONARY INTELLIGENCE", [
        ("GET", "/kimera/revolutionary/status/complete", None, 200),
    ])
    
    # Law Enforcement
    verifier.verify_component("LAW ENFORCEMENT", [
        ("GET", "/kimera/law_enforcement/system_status", None, 200),
    ])
    
    # Cognitive Pharmaceutical
    verifier.verify_component("COGNITIVE PHARMACEUTICAL", [
        ("GET", "/kimera/cognitive-pharmaceutical/system/status", None, 200),
    ])
    
    # Foundational Thermodynamics
    verifier.verify_component("FOUNDATIONAL THERMODYNAMICS", [
        ("GET", "/kimera/thermodynamics/status/system", None, 200),
    ])
    
    # Output Analysis
    verifier.verify_component("OUTPUT ANALYSIS", [
        ("POST", "/kimera/output/analyze", {
            "content": "Test output analysis",
            "context": {"type": "test"}
        }, 200),
    ])
    
    # Core Actions
    verifier.verify_component("CORE ACTIONS", [
        ("POST", "/kimera/action/execute", {
            "action_type": "analyze",
            "parameters": {"text": "test"}
        }, 200),
    ])
    
    # Chat with Diffusion Model
    verifier.verify_component("CHAT (DIFFUSION MODEL)", [
        ("POST", "/kimera/chat/", {
            "message": "Hello, test the diffusion model",
            "mode": "natural_language"
        }, 200),
        ("GET", "/kimera/chat/capabilities", None, 200),
        ("POST", "/kimera/chat/modes/test", None, 200),
    ])
    
    # Print Summary
    print("\n" + "="*80)
    print("📊 VERIFICATION SUMMARY")
    print("="*80)
    print(f"Total Checks: {verifier.passed + verifier.failed}")
    print(f"✅ Passed: {verifier.passed}")
    print(f"❌ Failed: {verifier.failed}")
    print(f"Success Rate: {(verifier.passed / (verifier.passed + verifier.failed) * 100):.1f}%")
    
    # Component Summary
    component_summary = {}
    for result in verifier.results:
        component = result["component"]
        if component not in component_summary:
            component_summary[component] = {"passed": 0, "failed": 0}
        if result["success"]:
            component_summary[component]["passed"] += 1
        else:
            component_summary[component]["failed"] += 1
    
    print("\n📋 COMPONENT BREAKDOWN:")
    for component, stats in component_summary.items():
        total = stats["passed"] + stats["failed"]
        status = "✅" if stats["failed"] == 0 else "⚠️" if stats["passed"] > 0 else "❌"
        print(f"{status} {component}: {stats['passed']}/{total} passed")
    
    # Failed Endpoints Detail
    failed_endpoints = [r for r in verifier.results if not r["success"]]
    if failed_endpoints:
        print("\n⚠️ FAILED ENDPOINTS:")
        for failure in failed_endpoints:
            print(f"  - {failure['endpoint']} ({failure['component']})")
    
    print("\n" + "="*80)
    print("VERIFICATION COMPLETE")
    print("="*80)
    
    # Save detailed report
    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_checks": verifier.passed + verifier.failed,
            "passed": verifier.passed,
            "failed": verifier.failed,
            "success_rate": verifier.passed / (verifier.passed + verifier.failed) * 100
        },
        "component_summary": component_summary,
        "detailed_results": verifier.results
    }
    
    with open("verification_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\n📄 Detailed report saved to verification_report.json")

if __name__ == "__main__":
    main()