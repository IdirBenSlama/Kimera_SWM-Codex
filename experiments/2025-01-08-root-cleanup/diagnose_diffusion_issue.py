#!/usr/bin/env python3
"""
Diagnose the Diffusion Model Issue
==================================

The chat is returning 39-character responses with 0.005 semantic coherence.
This script will help identify where the text generation pipeline is failing.
"""

import requests
import json
import time
import sys

def test_raw_chat():
    """Test the raw chat endpoint to see exact response"""
    print("\n" + "="*60)
    print("TESTING RAW CHAT RESPONSE")
    print("="*60)
    
    url = "http://localhost:8000/kimera/chat/"
    payload = {
        "message": "What is 2 + 2?",
        "mode": "standard",
        "session_id": "debug_test"
    }
    
    print(f"Sending: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(url, json=payload, timeout=180)
        
        print(f"\nStatus Code: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            
            # Print the complete response
            print("\nComplete Response:")
            print(json.dumps(data, indent=2))
            
            # Analyze the response text
            response_text = data.get('response', '')
            print(f"\n📝 Response Text Analysis:")
            print(f"   Length: {len(response_text)} characters")
            print(f"   Content: '{response_text}'")
            print(f"   First 20 chars: '{response_text[:20]}'")
            print(f"   Last 20 chars: '{response_text[-20:]}'")
            
            # Check if it's a fallback response
            fallback_patterns = [
                "I'm processing",
                "I'm experiencing", 
                "I'm working with",
                "semantic patterns",
                "magnitude"
            ]
            
            for pattern in fallback_patterns:
                if pattern in response_text:
                    print(f"\n⚠️ DETECTED FALLBACK PATTERN: '{pattern}'")
                    print("   This indicates the diffusion model failed and used fallback generation!")
            
            return data
        else:
            print(f"\nError Response: {response.text}")
            
    except Exception as e:
        print(f"\nException: {e}")
        import traceback
        traceback.print_exc()

def check_translator_hub():
    """Check the translator hub configuration"""
    print("\n" + "="*60)
    print("CHECKING TRANSLATOR HUB")
    print("="*60)
    
    # This endpoint doesn't exist in the current setup, but let's check what we can
    endpoints = [
        "/kimera/chat/capabilities",
        "/kimera/chat/integration/status",
        "/kimera/system/gpu_foundation"
    ]
    
    for endpoint in endpoints:
        url = f"http://localhost:8000{endpoint}"
        print(f"\nChecking {endpoint}...")
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                print(f"✅ Success:")
                data = response.json()
                print(json.dumps(data, indent=2)[:500] + "..." if len(json.dumps(data)) > 500 else json.dumps(data, indent=2))
            else:
                print(f"❌ Error {response.status_code}: {response.text[:100]}")
        except requests.exceptions.Timeout:
            print("⏱️ Timeout")
        except Exception as e:
            print(f"❌ Exception: {e}")

def test_different_messages():
    """Test with different message types to see if all fail the same way"""
    print("\n" + "="*60)
    print("TESTING DIFFERENT MESSAGE TYPES")
    print("="*60)
    
    test_messages = [
        ("Simple", "Hi"),
        ("Question", "What is your name?"),
        ("Math", "Calculate 5 + 3"),
        ("Complex", "Explain quantum mechanics in simple terms"),
        ("Code", "Write a Python hello world program")
    ]
    
    url = "http://localhost:8000/kimera/chat/"
    
    for msg_type, message in test_messages:
        print(f"\n--- Testing {msg_type}: '{message}' ---")
        
        payload = {
            "message": message,
            "mode": "standard",
            "session_id": f"test_{msg_type.lower()}"
        }
        
        try:
            start = time.time()
            response = requests.post(url, json=payload, timeout=180)
            elapsed = time.time() - start
            
            if response.status_code == 200:
                data = response.json()
                resp_text = data.get('response', '')
                coherence = data.get('semantic_coherence', 0)
                
                print(f"✅ Response ({elapsed:.1f}s): '{resp_text}'")
                print(f"   Length: {len(resp_text)} chars, Coherence: {coherence:.6f}")
                
                # Check if all responses are similar
                if "magnitude" in resp_text.lower() or "processing" in resp_text.lower():
                    print("   ⚠️ FALLBACK RESPONSE DETECTED")
            else:
                print(f"❌ Error: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Exception: {e}")

def check_model_loading():
    """Try to understand if the model is loading correctly"""
    print("\n" + "="*60)
    print("CHECKING MODEL LOADING STATUS")
    print("="*60)
    
    # Check system status
    try:
        response = requests.get("http://localhost:8000/kimera/status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("System Status:")
            print(json.dumps(data, indent=2))
    except:
        print("Could not get system status")
    
    # Check if we can get any info about loaded models
    print("\nLooking for model information...")
    
    # Try the monitoring endpoint
    try:
        response = requests.get("http://localhost:8000/kimera/monitoring/engines/status", timeout=5)
        if response.status_code == 200:
            print("Engine Status:")
            print(json.dumps(response.json(), indent=2))
    except:
        pass

def main():
    print("""
╔══════════════════════════════════════════════════════��═══════════╗
║              KIMERA DIFFUSION MODEL DIAGNOSTICS                   ║
║                                                                  ║
║  Investigating why responses are:                                ║
║  • Only 39 characters long                                       ║
║  • Have 0.005 semantic coherence                                ║
║  • Take 119 seconds to generate                                 ║
╚══════════════════════════════════════════════════════════════════╝
""")
    
    # Run diagnostics
    test_raw_chat()
    check_translator_hub()
    test_different_messages()
    check_model_loading()
    
    print("\n\n" + "="*60)
    print("DIAGNOSIS SUMMARY")
    print("="*60)
    print("""
Based on the symptoms:
1. All responses are ~39 characters (fallback responses)
2. Semantic coherence is near zero (0.005)
3. Generation takes 119+ seconds
4. Responses mention "magnitude" or "processing"

CONCLUSION: The diffusion model is failing completely and falling back
to a hardcoded response generator. This is likely because:

1. The language model (phi-2) failed to load properly
2. The embedding model is not working correctly
3. The diffusion process is producing invalid embeddings
4. GPU is not being used (running on CPU)

The system needs the models to be properly loaded and GPU acceleration
to be enabled for functional text generation.
""")

if __name__ == "__main__":
    main()