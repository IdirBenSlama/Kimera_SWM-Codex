"""
Server Status Checker
====================
Simple script to check if the Kimera server is running and responsive.
"""

import httpx
import asyncio
import time
import sys

async def check_server_status(base_url: str = "http://localhost:8000"):
    """Check server status with detailed diagnostics"""
    
    print("🔍 Checking Kimera server status...")
    print(f"   URL: {base_url}")
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            print("   Testing /health endpoint...")
            start_time = time.time()
            
            response = await client.get(f"{base_url}/health")
            response_time = time.time() - start_time
            
            print(f"✅ Server is responsive!")
            print(f"   Status Code: {response.status_code}")
            print(f"   Response Time: {response_time:.3f}s")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    print(f"   Response Data: {data}")
                except:
                    print(f"   Response Text: {response.text[:200]}...")
            
            return True
            
    except httpx.ConnectError as e:
        print(f"❌ Connection Error: {e}")
        print("   Server may not be running or port 8000 is not accessible")
        return False
        
    except httpx.TimeoutException:
        print("❌ Timeout Error: Server is not responding within 10 seconds")
        print("   Server may be overloaded or hung")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        return False

async def test_api_endpoint(base_url: str = "http://localhost:8000"):
    """Test a simple API endpoint"""
    
    print("\n🧪 Testing API endpoint...")
    
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(f"{base_url}/api/cognitive-field/geoids")
            print(f"✅ API endpoint responsive!")
            print(f"   Status Code: {response.status_code}")
            return True
            
    except Exception as e:
        print(f"❌ API endpoint error: {e}")
        return False

async def main():
    """Main function"""
    
    print("🛡️ Kimera Server Status Check")
    print("=" * 40)
    
    # Check basic health
    health_ok = await check_server_status()
    
    if health_ok:
        # Test API if health check passes
        await test_api_endpoint()
        print("\n✅ Server appears to be fully operational!")
    else:
        print("\n❌ Server is not responding. Check:")
        print("   1. Is the server process running?")
        print("   2. Are the database containers running?")
        print("   3. Check server logs for errors")
        print("   4. Try restarting the server")
    
    print("=" * 40)

if __name__ == "__main__":
    asyncio.run(main()) 