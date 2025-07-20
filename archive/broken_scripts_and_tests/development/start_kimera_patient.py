#!/usr/bin/env python3
"""
🚀 KIMERA PATIENT STARTUP SCRIPT
================================

KIMERA is a complex AI system that takes time to initialize all its components.
This script provides proper feedback during the startup process.

Author: KIMERA AI System
Version: 1.0.0 - Patient Startup Solution
"""

import uvicorn
import time
import threading
import requests
from backend.api.main import app

def check_server_ready(max_wait=600):
    """Check if server is ready to accept connections"""
    start_time = time.time()
    print("\n🔍 Waiting for KIMERA to complete initialization...")
    print("⏳ This may take 5-10 minutes depending on your system...")
    print("💡 KIMERA is loading large AI models - please be patient!")
    
    while time.time() - start_time < max_wait:
        try:
            response = requests.get("http://localhost:8001/system/health", timeout=5)
            if response.status_code == 200:
                elapsed = time.time() - start_time
                print(f"\n✅ KIMERA is ready! (Initialization took {elapsed:.1f} seconds)")
                print("\n🌐 KIMERA is now available at:")
                print("   • Main API: http://localhost:8001")
                print("   • API Docs: http://localhost:8001/docs")
                print("   • Health Check: http://localhost:8001/system/health")
                print("   • System Status: http://localhost:8001/system/status")
                print("\n🎯 Try these commands:")
                print("   curl http://localhost:8001/system/health")
                print("   curl http://localhost:8001/system/status")
                return True
        except:
            pass
        
        # Show progress dots
        elapsed = time.time() - start_time
        dots = "." * (int(elapsed) % 4)
        print(f"\r⏳ Initializing{dots:<3} ({elapsed:.0f}s)", end="", flush=True)
        time.sleep(2)
    
    print(f"\n⚠️  Server didn't respond within {max_wait} seconds")
    print("💡 KIMERA might still be initializing. Check manually:")
    print("   curl http://localhost:8001/system/health")
    return False

if __name__ == "__main__":
    print("🚀 KIMERA PATIENT STARTUP")
    print("=" * 50)
    print("🧠 KIMERA is a sophisticated AI system with multiple components:")
    print("   • GPU Foundation & CUDA Optimization")
    print("   • Advanced Embedding Models (BAAI/bge-m3)")
    print("   • Cognitive Field Dynamics")
    print("   • Universal Translator Hub")
    print("   • Revolutionary Intelligence Systems")
    print("   • Quantum-Enhanced Processing")
    print("")
    print("⏰ Expected initialization time: 5-10 minutes")
    print("🎯 Server will start on: http://localhost:8001")
    print("")
    print("Starting server...")
    
    # Start the health check in a separate thread
    health_thread = threading.Thread(target=check_server_ready)
    health_thread.daemon = True
    health_thread.start()
    
    # Start the server
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8001,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n\n✋ KIMERA shutdown requested by user")
        print("🛑 Stopping all systems...")
    except Exception as e:
        print(f"\n❌ Error starting KIMERA: {e}")
        print("💡 Check the logs for more details") 