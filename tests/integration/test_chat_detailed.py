#!/usr/bin/env python3
"""
Detailed Chat Testing to Diagnose Issues
"""

import json
import time

import requests


def test_chat_detailed():
    """Test chat with detailed output"""
    url = "http://localhost:8000/kimera/chat/"

    test_cases = [
        {
            "name": "Basic Test",
            "payload": {
                "message": "Hello, can you hear me?",
                "mode": "standard",
                "session_id": "test_basic",
            },
        },
        {
            "name": "Architecture Question",
            "payload": {
                "message": "Explain your cognitive architecture",
                "mode": "cognitive_enhanced",
                "session_id": "test_arch",
                "cognitive_mode": "cognitive_enhanced",
            },
        },
        {
            "name": "Simple Math",
            "payload": {
                "message": "What is 2 + 2?",
                "mode": "standard",
                "session_id": "test_math",
            },
        },
    ]

    for test in test_cases:
        print(f"\n{'='*60}")
        print(f"TEST: {test['name']}")
        print(f"{'='*60}")
        print(f"Payload: {json.dumps(test['payload'], indent=2)}")

        try:
            start_time = time.time()
            response = requests.post(url, json=test["payload"], timeout=180)
            elapsed = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                print(f"\n✅ Success in {elapsed:.2f}s")
                print(f"\nFull Response Data:")
                print(json.dumps(data, indent=2))

                # Analyze the response
                response_text = data.get("response", "")
                print(f"\n📝 Response Text ({len(response_text)} chars):")
                print(f"'{response_text}'")

                # Check for common issues
                if len(response_text) < 50:
                    print("\n⚠️ WARNING: Response is unusually short!")

                if response_text.startswith("I'm processing"):
                    print("\n⚠️ WARNING: Response appears to be a fallback message!")

                coherence = data.get("semantic_coherence", 0)
                if coherence < 0.1:
                    print(f"\n⚠️ WARNING: Extremely low semantic coherence: {coherence}")

            else:
                print(f"\n❌ Error: {response.status_code}")
                print(response.text)

        except Exception as e:
            print(f"\n❌ Exception: {e}")


def check_model_status():
    """Check if models are loaded properly"""
    print("\n" + "=" * 60)
    print("CHECKING MODEL STATUS")
    print("=" * 60)

    # Check translator hub
    try:
        response = requests.get("http://localhost:8000/translator/status")
        if response.status_code == 200:
            print("✅ Translator Hub Status:")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"❌ Translator Hub Error: {response.status_code}")
    except Exception as e:
        print(f"❌ Translator Hub Exception: {e}")

    # Check GPU status
    try:
        response = requests.get(
            "http://localhost:8000/kimera/system/gpu_foundation", timeout=10
        )
        if response.status_code == 200:
            print("\n✅ GPU Foundation Status:")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"\n❌ GPU Foundation Error: {response.status_code}")
    except Exception as e:
        print(f"\n❌ GPU Foundation Exception: {e}")


def test_diffusion_modes():
    """Test different diffusion modes"""
    print("\n" + "=" * 60)
    print("TESTING DIFFUSION MODES")
    print("=" * 60)

    url = "http://localhost:8000/kimera/chat/modes/test"

    try:
        print("Testing all cognitive modes...")
        response = requests.post(url, timeout=300)

        if response.status_code == 200:
            data = response.json()
            print("\n✅ Mode Test Results:")

            for mode, result in data.get("results", {}).items():
                print(f"\n{mode.upper()}:")
                print(f"  Response: {result.get('response', 'N/A')}")
                print(f"  Confidence: {result.get('confidence', 0):.3f}")
                print(f"  Coherence: {result.get('semantic_coherence', 0):.3f}")
                print(f"  Engine: {result.get('engine_used', 'N/A')}")
        else:
            print(f"❌ Error: {response.status_code}")

    except Exception as e:
        print(f"❌ Exception: {e}")


def main():
    print(
        """
╔═════════════���════════════════════════════════════════════════════╗
║                 KIMERA CHAT DETAILED DIAGNOSTICS                  ║
╚══════════════════════════════════════════════════════════════════╝
"""
    )

    # Check model status first
    check_model_status()

    # Test chat functionality
    test_chat_detailed()

    # Test diffusion modes
    test_diffusion_modes()


if __name__ == "__main__":
    main()
