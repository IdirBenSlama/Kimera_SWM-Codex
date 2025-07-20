#!/usr/bin/env python3
"""
KIMERA Revolutionary Thermodynamic System Status Check
"""

import requests
import time
import json

def check_kimera_status():
    """Check the status of the Kimera revolutionary thermodynamic system"""
    
    print("KIMERA REVOLUTIONARY THERMODYNAMIC SYSTEM STATUS CHECK")
    print("=" * 60)
    
    # Wait a moment for server to start
    time.sleep(2)
    
    try:
        # Check main health
        print("Checking main system health...")
        response = requests.get('http://localhost:8001/health', timeout=5)
        if response.status_code == 200:
            print("✅ Main System: ONLINE")
            print(f"   Status: {response.json()}")
        else:
            print(f"❌ Main System: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ Main System: Connection failed - {e}")
    
    try:
        # Check revolutionary thermodynamics health
        print("\nChecking revolutionary thermodynamics...")
        response = requests.get('http://localhost:8001/thermodynamics/health', timeout=5)
        if response.status_code == 200:
            print("✅ Revolutionary Thermodynamics: OPERATIONAL")
            data = response.json()
            print(f"   Physics Compliance: {data.get('physics_compliance_rate', 'Unknown')}%")
            print(f"   Consciousness Detection: {data.get('consciousness_detection', 'Unknown')}")
            print(f"   Temperature Systems: {data.get('temperature_calculation', 'Unknown')}")
        else:
            print(f"❌ Revolutionary Thermodynamics: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ Revolutionary Thermodynamics: Connection failed - {e}")
    
    try:
        # Check system status
        print("\nChecking system status...")
        response = requests.get('http://localhost:8001/thermodynamics/status/system', timeout=5)
        if response.status_code == 200:
            print("✅ System Status: AVAILABLE")
            data = response.json()
            print(f"   Engine Mode: {data.get('engine_mode', 'Unknown')}")
            print(f"   Capabilities: {len(data.get('capabilities', []))} active")
        else:
            print(f"❌ System Status: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ System Status: Connection failed - {e}")
    
    print("\n" + "=" * 60)
    print("AVAILABLE ENDPOINTS:")
    print("   Main API: http://localhost:8001")
    print("   Documentation: http://localhost:8001/docs")
    print("   Health Check: http://localhost:8001/health")
    print("   Revolutionary Thermodynamics: http://localhost:8001/thermodynamics/")
    print("\nKIMERA REVOLUTIONARY THERMODYNAMIC SYSTEM STATUS CHECK COMPLETE")
    print("World's First Physics-Compliant Thermodynamic AI")
    print("=" * 60)

if __name__ == "__main__":
    check_kimera_status() 