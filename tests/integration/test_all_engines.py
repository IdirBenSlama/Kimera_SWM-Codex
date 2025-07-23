#!/usr/bin/env python3
"""
Comprehensive Kimera Engines Test Suite
=======================================

Tests all 8 Kimera engines to verify they are operational.
"""

import requests
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Tuple

# Configuration
BASE_URL = "http://localhost:8000"
TIMEOUT = 30  # seconds
RETRY_COUNT = 3

def test_endpoint(name: str, url: str, description: str) -> Tuple[bool, str, Dict[str, Any]]:
    """Test a single endpoint with retries."""
    for attempt in range(RETRY_COUNT):
        try:
            start_time = time.time()
            response = requests.get(url, timeout=TIMEOUT)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                status = data.get("status", "unknown")
                
                result = {
                    "success": True,
                    "status": status,
                    "response_time": f"{response_time:.3f}s",
                    "data": data
                }
                
                if status == "operational":
                    return True, "✅ PASS", result
                elif status == "not_available":
                    return False, "❌ FAIL", result
                else:
                    return None, "⚠️ WARN", result
            else:
                return False, "❌ FAIL", {"error": f"HTTP {response.status_code}", "response_time": f"{response_time:.3f}s"}
                
        except requests.exceptions.ConnectionError:
            if attempt == RETRY_COUNT - 1:
                return False, "❌ FAIL", {"error": "Connection error - server not running?"}
            time.sleep(1)
        except requests.exceptions.Timeout:
            if attempt == RETRY_COUNT - 1:
                return False, "❌ FAIL", {"error": "Request timeout"}
            time.sleep(1)
        except Exception as e:
            if attempt == RETRY_COUNT - 1:
                return False, "❌ FAIL", {"error": str(e)}
            time.sleep(1)
    
    return False, "❌ FAIL", {"error": "Max retries exceeded"}

def main():
    """Run comprehensive engine tests."""
    print("🚀 Starting Comprehensive Kimera Engines Test")
    print("=" * 60)
    print(f"Testing server at: {BASE_URL}")
    print(f"Test started at: {datetime.now().isoformat()}")
    print()
    
    # Test configurations: (name, endpoint, description)
    tests = [
        ("System Health Check", f"{BASE_URL}/kimera/monitoring/health", "Basic system health and availability"),
        ("Engines Status Overview", f"{BASE_URL}/kimera/monitoring/engines/status", "Overall engines status summary"),
        ("Contradiction Engine", f"{BASE_URL}/kimera/monitoring/engines/contradiction", "Logical contradiction detection and resolution"),
        ("Thermodynamics Engine", f"{BASE_URL}/kimera/monitoring/engines/thermodynamics", "Physics-compliant thermodynamic processing"),
        ("SPDE Engine", f"{BASE_URL}/kimera/monitoring/engines/spde", "Stochastic Partial Differential Equation solver"),
        ("Cognitive Cycle Engine", f"{BASE_URL}/kimera/monitoring/engines/cognitive_cycle", "Iterative cognitive processing and attention management"),
        ("Meta Insight Engine", f"{BASE_URL}/kimera/monitoring/engines/meta_insight", "Higher-order insight generation and meta-cognition"),
        ("Proactive Detector", f"{BASE_URL}/kimera/monitoring/engines/proactive_detector", "Predictive analysis and early warning systems"),
        ("Revolutionary Intelligence", f"{BASE_URL}/kimera/monitoring/engines/revolutionary", "Revolutionary intelligence and breakthrough detection")
    ]
    
    results = []
    passed = 0
    failed = 0
    warnings = 0
    
    for i, (name, url, description) in enumerate(tests, 1):
        print(f"Test {i:2d}/{len(tests)}: {name}")
        print(f"         {description}")
        
        success, status, result = test_endpoint(name, url, description)
        results.append((name, status, result))
        
        print(f"         Result: {status}")
        
        if success is True:
            passed += 1
            print(f"         Response Time: {result.get('response_time', 'N/A')}")
            print(f"         Engine Status: {result.get('status', 'unknown')}")
        elif success is False:
            failed += 1
            error = result.get('error', 'Unknown error')
            print(f"         Error: {error}")
        else:  # warning
            warnings += 1
            print(f"         Response Time: {result.get('response_time', 'N/A')}")
            print(f"         Engine Status: {result.get('status', 'unknown')}")
        
        print()
    
    # Summary
    print("=" * 60)
    print("🎯 TEST SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {len(tests)}")
    print(f"✅ Passed: {passed}")
    print(f"⚠️ Warnings: {warnings}")
    print(f"❌ Failed: {failed}")
    print(f"Success Rate: {(passed / len(tests) * 100):.1f}%")
    print()
    
    # Engine Status Overview
    print("🔧 ENGINE STATUS OVERVIEW")
    print("=" * 60)
    
    operational_engines = []
    unavailable_engines = []
    error_engines = []
    
    for name, status, result in results:
        if name in ["System Health Check", "Engines Status Overview"]:
            continue  # Skip system tests
            
        engine_status = result.get('status', 'unknown')
        if engine_status == 'operational':
            operational_engines.append(name)
        elif engine_status == 'not_available':
            unavailable_engines.append(name)
        else:
            error_engines.append(name)
    
    print(f"✅ Operational Engines ({len(operational_engines)}):")
    for engine in operational_engines:
        print(f"   • {engine}")
    
    if unavailable_engines:
        print(f"\n❌ Unavailable Engines ({len(unavailable_engines)}):")
        for engine in unavailable_engines:
            print(f"   • {engine}")
    
    if error_engines:
        print(f"\n⚠️ Engines with Issues ({len(error_engines)}):")
        for engine in error_engines:
            print(f"   • {engine}")
    
    print()
    
    # Revolutionary Capabilities Summary
    if operational_engines:
        print("🚀 REVOLUTIONARY CAPABILITIES ACHIEVED")
        print("=" * 60)
        
        capabilities = []
        if "SPDE Engine" in operational_engines:
            capabilities.append("• Advanced Mathematical Modeling (Stochastic PDEs)")
        if "Cognitive Cycle Engine" in operational_engines:
            capabilities.append("• Iterative Cognitive Processing with Attention Management")
        if "Meta Insight Engine" in operational_engines:
            capabilities.append("• Higher-Order Meta-Cognitive Intelligence")
        if "Proactive Detector" in operational_engines:
            capabilities.append("• Predictive Analysis and Early Warning Systems")
        if "Revolutionary Intelligence" in operational_engines:
            capabilities.append("• Revolutionary Intelligence and Breakthrough Detection")
        
        for capability in capabilities:
            print(capability)
        
        print()
    
    # Final Assessment
    total_engines = len([name for name, _, _ in results if name not in ["System Health Check", "Engines Status Overview"]])
    operational_count = len(operational_engines)
    
    print("🎖️ FINAL ASSESSMENT")
    print("=" * 60)
    print(f"Kimera System Status: {operational_count}/{total_engines} engines operational")
    
    if operational_count == total_engines:
        print("🏆 PERFECT SCORE! All engines are operational!")
        print("🌟 Kimera SWM is ready for revolutionary AI operations!")
    elif operational_count >= total_engines * 0.8:
        print("🎯 EXCELLENT! Most engines are operational.")
        print("🔧 Minor fixes needed for remaining engines.")
    elif operational_count >= total_engines * 0.6:
        print("⚠️ GOOD! Majority of engines are operational.")
        print("🔧 Some fixes needed for optimal performance.")
    else:
        print("❌ NEEDS ATTENTION! Several engines require fixes.")
        print("🔧 System initialization or configuration issues detected.")
    
    print(f"\nTest completed at: {datetime.now().isoformat()}")
    
    # Return exit code based on results
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    exit(main()) 