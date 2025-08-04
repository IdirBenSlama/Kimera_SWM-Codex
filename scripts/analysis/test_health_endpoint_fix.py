#!/usr/bin/env python3
"""
Health Endpoint Fix Verification
===============================

Tests the health endpoint fix to ensure it's working correctly.
"""

import requests
import json
import time
from datetime import datetime

def test_health_endpoint():
    """Test the health endpoint after the fix"""

    print("🔍 TESTING HEALTH ENDPOINT FIX")
    print("=" * 50)

    # Test endpoints
    test_urls = [
        "http://127.0.0.1:8000/health",
        "http://127.0.0.1:8001/health",
        "http://127.0.0.1:8002/health"
    ]

    results = {}

    for url in test_urls:
        print(f"\n🌐 Testing: {url}")
        try:
            start_time = time.time()
            response = requests.get(url, timeout=10)
            response_time = time.time() - start_time

            print(f"   Status: {response.status_code}")
            print(f"   Response Time: {response_time:.3f}s")

            if response.status_code == 200:
                try:
                    data = response.json()
                    print(f"   Service: {data.get('service', 'Unknown')}")
                    print(f"   Status: {data.get('status', 'Unknown')}")
                    print(f"   ✅ SUCCESS")

                    results[url] = {
                        "status": "success",
                        "code": response.status_code,
                        "response_time": response_time,
                        "data": data
                    }
                except Exception as e:
                    print(f"   ⚠️  JSON parsing failed: {e}")
                    results[url] = {
                        "status": "partial_success",
                        "code": response.status_code,
                        "response_time": response_time,
                        "error": str(e)
                    }
            else:
                print(f"   ❌ FAILED - HTTP {response.status_code}")
                print(f"   Response: {response.text[:100]}")
                results[url] = {
                    "status": "failed",
                    "code": response.status_code,
                    "response_time": response_time,
                    "error": response.text[:200]
                }

        except Exception as e:
            print(f"   ❌ ERROR: {str(e)}")
            results[url] = {
                "status": "error",
                "error": str(e)
            }

    # Summary
    print(f"\n📊 SUMMARY")
    print("=" * 30)

    successful = [url for url, result in results.items() if result["status"] == "success"]
    failed = [url for url, result in results.items() if result["status"] in ["failed", "error"]]

    print(f"✅ Successful: {len(successful)}")
    print(f"❌ Failed: {len(failed)}")
    print(f"📈 Success Rate: {len(successful)/len(test_urls)*100:.1f}%")

    if len(successful) > 0:
        print(f"\n🎉 Health endpoint fix is working!")
        print(f"   Working endpoints: {successful}")

    if len(failed) > 0:
        print(f"\n⚠️  Still need attention: {failed}")

    return results

if __name__ == "__main__":
    test_health_endpoint()
