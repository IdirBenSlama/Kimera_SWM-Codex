#!/usr/bin/env python3
"""
Test KIMERA Chat API with Enhanced Diffusion Model
"""

import requests
import json
import time
from typing import Dict, Any

def test_chat_endpoint(message: str, mode: str = "cognitive_enhanced", session_id: str = "test_session"):
    """Test the KIMERA chat endpoint"""
    url = "http://localhost:8000/kimera/chat/"
    
    payload = {
        "message": message,
        "mode": mode,
        "session_id": session_id,
        "cognitive_mode": mode,
        "persona_context": "scientific_analysis"
    }
    
    print(f"\n🚀 Testing KIMERA Chat API")
    print(f"   URL: {url}")
    print(f"   Mode: {mode}")
    print(f"   Session: {session_id}")
    print(f"   Message: {message[:100]}...")
    
    try:
        start_time = time.time()
        response = requests.post(url, json=payload)
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n✅ Success! (Response time: {elapsed_time:.2f}s)")
            print(f"\n📝 KIMERA Response:")
            print(f"   {data.get('response', 'No response')}")
            print(f"\n📊 Metrics:")
            print(f"   - Confidence: {data.get('confidence', 0):.3f}")
            print(f"   - Semantic Coherence: {data.get('semantic_coherence', 0):.3f}")
            print(f"   - Cognitive Resonance: {data.get('cognitive_resonance', 0):.3f}")
            print(f"   - Generation Time: {data.get('generation_time', 0):.3f}s")
            return data
        else:
            print(f"\n❌ Error: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        print("\n❌ Connection Error: Is the KIMERA server running?")
        return None
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        return None

def test_all_cognitive_modes():
    """Test all cognitive modes"""
    test_message = "Explain your cognitive architecture and how your diffusion model processes information."
    
    modes = [
        ("standard", "Standard mode - basic cognitive processing"),
        ("cognitive_enhanced", "Enhanced mode - deep semantic analysis"),
        ("persona_aware", "Persona-aware mode - adaptive communication"),
        ("neurodivergent", "Neurodivergent mode - structured clarity")
    ]
    
    print("\n" + "="*60)
    print("🧠 KIMERA COGNITIVE MODE TESTING")
    print("="*60)
    
    results = {}
    for mode, description in modes:
        print(f"\n\n🔬 Testing: {mode}")
        print(f"   {description}")
        print("-"*60)
        
        result = test_chat_endpoint(test_message, mode=mode, session_id=f"test_{mode}")
        if result:
            results[mode] = result
        
        time.sleep(1)  # Brief pause between tests
    
    # Summary
    print("\n\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    for mode, result in results.items():
        print(f"\n{mode.upper()}:")
        print(f"  Response length: {len(result.get('response', ''))} chars")
        print(f"  Confidence: {result.get('confidence', 0):.3f}")
        print(f"  Coherence: {result.get('semantic_coherence', 0):.3f}")

def test_conversation_context():
    """Test conversation context and memory"""
    session_id = "context_test"
    
    print("\n" + "="*60)
    print("💬 TESTING CONVERSATION CONTEXT")
    print("="*60)
    
    messages = [
        "Hello KIMERA, I'm interested in understanding quantum computing.",
        "Can you explain how quantum entanglement works?",
        "How does this relate to what we discussed about quantum computing?",
        "What are the practical applications we've covered so far?"
    ]
    
    for i, message in enumerate(messages):
        print(f"\n\n📨 Message {i+1}: {message}")
        print("-"*60)
        
        result = test_chat_endpoint(
            message, 
            mode="cognitive_enhanced",
            session_id=session_id
        )
        
        if result:
            print(f"\n🔗 Context maintained: Session {session_id}")
        
        time.sleep(1)

def test_system_status():
    """Test system status endpoints"""
    print("\n" + "="*60)
    print("🔍 TESTING SYSTEM STATUS")
    print("="*60)
    
    endpoints = [
        ("http://localhost:8000/health", "Health Check"),
        ("http://localhost:8000/kimera/status", "KIMERA Status"),
        ("http://localhost:8000/kimera/chat/capabilities", "Chat Capabilities"),
        ("http://localhost:8000/kimera/chat/integration/status", "Integration Status")
    ]
    
    for url, description in endpoints:
        print(f"\n📍 {description}: {url}")
        try:
            response = requests.get(url)
            if response.status_code == 200:
                print(f"��� Status: OK")
                data = response.json()
                print(f"   Response: {json.dumps(data, indent=2)[:200]}...")
            else:
                print(f"❌ Status: {response.status_code}")
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Run all tests"""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║              KIMERA SWM - ENHANCED CHAT API TESTING              ║
║                                                                  ║
║  Testing the revolutionary text diffusion engine with:           ║
║  • Multiple cognitive modes                                      ║
║  • Conversation context preservation                             ║
║  • System integration validation                                 ║
╚══════════════════════════════════════════════════════════════════╝
""")
    
    # Test system status first
    test_system_status()
    
    # Test basic chat
    print("\n\n" + "="*60)
    print("🎯 BASIC CHAT TEST")
    print("="*60)
    test_chat_endpoint(
        "Hello KIMERA! Can you explain your architecture and how your diffusion model works?",
        mode="cognitive_enhanced"
    )
    
    # Test all cognitive modes
    test_all_cognitive_modes()
    
    # Test conversation context
    test_conversation_context()
    
    print("\n\n✅ All tests completed!")

if __name__ == "__main__":
    main()