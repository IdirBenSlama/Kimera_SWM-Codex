#!/usr/bin/env python3
"""
🔍 KIMERA STATUS CHECKER
========================

Quick script to check if KIMERA is running and ready.
Use this while KIMERA is initializing to see current status.

Author: KIMERA AI System
Version: 1.0.0
"""

import requests
import time
import subprocess
import sys

def check_process():
    """Check if KIMERA Python process is running"""
    try:
        if sys.platform == "win32":
            result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe'], 
                                  capture_output=True, text=True)
            return 'python.exe' in result.stdout
        else:
            result = subprocess.run(['pgrep', '-f', 'uvicorn'], 
                                  capture_output=True, text=True)
            return bool(result.stdout.strip())
    except:
        return False

def check_server_health():
    """Check if KIMERA server is responding"""
    try:
        response = requests.get("http://localhost:8001/system/health", timeout=5)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except requests.exceptions.ConnectionError:
        return False, "Connection refused - server not ready"
    except requests.exceptions.Timeout:
        return False, "Timeout - server may be initializing"
    except Exception as e:
        return False, f"Error: {e}"

def main():
    print("🔍 KIMERA STATUS CHECK")
    print("=" * 30)
    
    # Check if process is running
    process_running = check_process()
    print(f"🔧 KIMERA Process: {'✅ Running' if process_running else '❌ Not Found'}")
    
    if not process_running:
        print("\n💡 KIMERA is not running. Start it with:")
        print("   python start_kimera_patient.py")
        print("   or")
        print("   python start_kimera.py")
        return
    
    # Check server health
    print("🌐 Checking server status...")
    is_ready, status = check_server_health()
    
    if is_ready:
        print("✅ KIMERA is ready and responding!")
        print("\n🌐 Available endpoints:")
        print("   • Main API: http://localhost:8001")
        print("   • API Docs: http://localhost:8001/docs")
        print("   • Health Check: http://localhost:8001/system/health")
        print("   • System Status: http://localhost:8001/system/status")
        
        if isinstance(status, dict):
            print(f"\n📊 Health Status: {status}")
    else:
        print("⏳ KIMERA is still initializing...")
        print(f"   Status: {status}")
        print("\n💡 This is normal during startup (5-10 minutes expected)")
        print("   Check again in a few minutes or wait for the patient startup script")

if __name__ == "__main__":
    main() 