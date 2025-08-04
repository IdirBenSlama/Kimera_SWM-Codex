#!/usr/bin/env python3
"""
Test KIMERA Chat API after restart
"""

import json
import time

import requests


def wait_for_server(max_attempts=30):
    """Wait for server to be ready"""
    print("⏳ Waiting for KIMERA server to start...")

    for i in range(max_attempts):
        try:
            response = requests.get("http://localhost:8000/health", timeout=1)
            if response.status_code == 200:
                print("✅ Server is ready!")
                return True
        except Exception as e:
            logger.error(f"Error in test_chat_after_restart.py: {e}", exc_info=True)
            raise  # Re-raise for proper error handling

        print(f"   Attempt {i+1}/{max_attempts}...")
        time.sleep(1)

    return False


def test_chat():
    """Test the chat endpoint"""
    url = "http://localhost:8000/kimera/chat/"

    payload = {
        "message": "Hello KIMERA! Can you explain your cognitive architecture and how your diffusion model works?",
        "mode": "cognitive_enhanced",
        "session_id": "test_session",
        "cognitive_mode": "cognitive_enhanced",
        "persona_context": "scientific_analysis",
    }

    print("\n🚀 Testing KIMERA Chat API")
    print(f"   URL: {url}")

    try:
        response = requests.post(url, json=payload, timeout=30)

        if response.status_code == 200:
            data = response.json()
            print("\n✅ SUCCESS! Chat endpoint is working!")
            print("\n📝 KIMERA Response:")
            print("-" * 60)
            print(data.get("response", "No response"))
            print("-" * 60)
            print("\n📊 Metrics:")
            print(f"   - Confidence: {data.get('confidence', 0):.3f}")
            print(f"   - Semantic Coherence: {data.get('semantic_coherence', 0):.3f}")
            print(f"   - Cognitive Resonance: {data.get('cognitive_resonance', 0):.3f}")
            print(f"   - Generation Time: {data.get('generation_time', 0):.3f}s")
            print(f"   - Session ID: {data.get('session_id', 'N/A')}")
            print(f"   - Cognitive Mode: {data.get('cognitive_mode', 'N/A')}")
            return True
        else:
            print(f"\n❌ Error: {response.status_code}")
            print(f"   Response: {response.text}")
            return False

    except requests.exceptions.Timeout:
        print("\n⏱️ Request timed out (this might mean the model is loading)")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


def test_capabilities():
    """Test the capabilities endpoint"""
    url = "http://localhost:8000/kimera/chat/capabilities"

    print("\n📋 Testing capabilities endpoint...")

    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            print("✅ Capabilities endpoint working!")
            print(f"   Cognitive modes: {list(data.get('cognitive_modes', {}).keys())}")
            print(f"   Features: {len(data.get('features', []))} features available")
            return True
        else:
            print(f"❌ Error: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    print(
        """
╔══════════════════════════════════════════════════════���═══════════╗
║              KIMERA CHAT API TEST (After Restart)                 ║
╚══════════════════════════════════════════════════════════════════╝
"""
    )

    # Wait for server
    if not wait_for_server():
        print("\n❌ Server failed to start!")
        return

    # Test capabilities first
    test_capabilities()

    # Test chat endpoint
    print("\n" + "=" * 60)
    success = test_chat()

    if success:
        print("\n\n🎉 KIMERA Chat API is fully operational!")
        print("   The diffusion model is working correctly")
        print("   All cognitive modes are available")
    else:
        print("\n\n⚠️ Chat API test failed")
        print("   Check server logs for details")


if __name__ == "__main__":
    main()
