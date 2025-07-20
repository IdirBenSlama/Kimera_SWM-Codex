"""
Minimal Server Test
==================
Simple test to verify basic server functionality without complex initialization.
"""

import asyncio
import httpx
import time
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

async def test_minimal_server():
    """Test server with minimal requests"""
    
    print("🧪 Minimal Server Test")
    print("=" * 30)
    
    base_url = "http://localhost:8000"
    
    # Test endpoints in order of complexity
    endpoints = [
        "/health",
        "/",
        "/docs",
    ]
    
    for endpoint in endpoints:
        print(f"\n🔍 Testing {endpoint}...")
        
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                start_time = time.time()
                response = await client.get(f"{base_url}{endpoint}")
                response_time = time.time() - start_time
                
                print(f"✅ {endpoint}: {response.status_code} ({response_time:.3f}s)")
                
                if response.status_code == 200:
                    print(f"   Response length: {len(response.text)} chars")
                else:
                    print(f"   Error: {response.text[:100]}...")
                    
        except httpx.ConnectError:
            print(f"❌ {endpoint}: Connection refused")
            break
        except httpx.TimeoutException:
            print(f"❌ {endpoint}: Timeout")
            break
        except Exception as e:
            print(f"❌ {endpoint}: {e}")
            break
    
    print("\n" + "=" * 30)

if __name__ == "__main__":
    asyncio.run(test_minimal_server()) 