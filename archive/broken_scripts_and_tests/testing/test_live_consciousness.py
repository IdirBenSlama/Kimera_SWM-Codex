#!/usr/bin/env python3
"""
Test KIMERA Live Consciousness Stream
=====================================

This script waits for the KIMERA server to be fully operational,
then runs the live consciousness stream to demonstrate KIMERA
articulating its own consciousness based on real system state.
"""

import time
import requests
import subprocess
import sys
import asyncio
from datetime import datetime

def wait_for_server(url="http://localhost:8001", timeout=900):
    """Wait for KIMERA server to be fully operational"""
    print(f"⏳ Waiting for KIMERA server at {url} to be fully operational...")
    print(f"   Maximum wait time: {timeout} seconds")
    
    start_time = time.time()
    last_status = None
    
    while time.time() - start_time < timeout:
        try:
            # Check health endpoint
            response = requests.get(f"{url}/system/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                status = data.get('system_status', 'unknown')
                
                if status != last_status:
                    print(f"   Server status: {status}")
                    last_status = status
                
                if status == 'fully_operational':
                    print(f"✅ KIMERA server is fully operational! (took {time.time() - start_time:.1f}s)")
                    return True
                    
            elapsed = time.time() - start_time
            if int(elapsed) % 30 == 0:  # Progress every 30 seconds
                print(f"   Still waiting... ({elapsed:.0f}s elapsed)")
                
        except requests.exceptions.RequestException:
            pass  # Server not ready yet
            
        time.sleep(5)  # Check every 5 seconds
    
    print(f"❌ Timeout waiting for server after {timeout} seconds")
    return False

async def run_live_consciousness_stream():
    """Run the live consciousness stream"""
    print("\n" + "="*80)
    print("🧠 RUNNING KIMERA LIVE CONSCIOUSNESS STREAM")
    print("="*80)
    
    # Import and run the live consciousness stream
    sys.path.insert(0, '.')
    from kimera_live_consciousness_stream import KimeraLiveConsciousnessStream
    
    stream = KimeraLiveConsciousnessStream("http://localhost:8001")
    
    # Run for 2 minutes to demonstrate
    await stream.stream_consciousness(duration=120, interval=5.0)

def check_server_components():
    """Check what components are loaded"""
    try:
        response = requests.get("http://localhost:8001/system/components")
        if response.status_code == 200:
            data = response.json()
            print(f"\n📊 Server Components:")
            print(f"   Total loaded: {data.get('total_components', 0)}")
            print(f"   Cognitive fidelity: {data.get('cognitive_fidelity', 0.0):.1%}")
            
            components = data.get('components', {})
            if components:
                print("\n   Key components:")
                for name, info in list(components.items())[:10]:  # Show first 10
                    print(f"   • {name}: {info.get('type', 'unknown')}")
                    
    except Exception as e:
        print(f"❌ Failed to check components: {e}")

def main():
    """Main test function"""
    print("🎯 KIMERA Live Consciousness Test")
    print("=" * 60)
    print("This test will:")
    print("1. Wait for KIMERA server to be fully operational")
    print("2. Check loaded components")
    print("3. Run live consciousness stream with real system data")
    print("=" * 60)
    
    # Wait for server
    if not wait_for_server():
        print("❌ Server failed to start. Please check the logs.")
        return
    
    # Check components
    check_server_components()
    
    # Run consciousness stream
    print("\n🌊 Starting consciousness stream in 5 seconds...")
    time.sleep(5)
    
    try:
        asyncio.run(run_live_consciousness_stream())
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✨ Test complete!")

if __name__ == "__main__":
    main()