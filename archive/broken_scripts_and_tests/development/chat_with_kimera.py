#!/usr/bin/env python3
"""
Interactive Chat with KIMERA
============================

Simple script to demonstrate KIMERA's enhanced conversation capabilities.
Run this while the KIMERA server is running to experience the different cognitive modes.
"""

import requests
import json
import time
from typing import List, Dict

class KimeraChat:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session_id = f"interactive_{int(time.time())}"
        self.conversation_history = []
        
    def check_connection(self) -> bool:
        """Check if KIMERA server is running."""
        try:
            response = requests.get(f"{self.base_url}/api/chat/capabilities", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def chat(self, message: str, cognitive_mode: str = "standard") -> Dict:
        """Send a message to KIMERA and get response."""
        try:
            payload = {
                "message": message,
                "cognitive_mode": cognitive_mode,
                "session_id": self.session_id,
                "conversation_history": self.conversation_history[-5:]  # Last 5 messages
            }
            
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Update conversation history
                self.conversation_history.append({
                    "role": "user",
                    "content": message,
                    "timestamp": time.time()
                })
                self.conversation_history.append({
                    "role": "assistant", 
                    "content": result["response"],
                    "timestamp": time.time()
                })
                
                return result
            else:
                return {"error": f"HTTP {response.status_code}: {response.text}"}
                
        except Exception as e:
            return {"error": str(e)}

def main():
    """Interactive chat session with KIMERA."""
    print("🧠 KIMERA Enhanced Chat Demo")
    print("=" * 50)
    
    chat = KimeraChat()
    
    # Check connection
    print("Checking connection to KIMERA server...")
    if not chat.check_connection():
        print("❌ Cannot connect to KIMERA server!")
        print("Please make sure the server is running:")
        print("python -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8000")
        return
    
    print("✅ Connected to KIMERA server!")
    print()
    
    # Available modes
    modes = {
        "1": ("standard", "Standard conversation mode"),
        "2": ("cognitive_enhanced", "Enhanced cognitive processing"),
        "3": ("persona_aware", "Persona-aware responses"),
        "4": ("neurodivergent", "Neurodivergent-friendly communication")
    }
    
    print("Available Cognitive Modes:")
    for key, (mode, desc) in modes.items():
        print(f"  {key}. {mode}: {desc}")
    print()
    
    current_mode = "standard"
    
    # Demo conversations
    demo_messages = [
        "Hello KIMERA! Tell me about yourself.",
        "What makes your cognitive architecture unique?",
        "How do you process information differently from other AI systems?",
        "Can you explain consciousness in AI systems?"
    ]
    
    print("🎯 Demo Mode - KIMERA will respond to sample questions")
    print("Press Enter to continue through each demo, or type 'interactive' for manual chat")
    print()
    
    user_input = input("Ready to start? (Press Enter or type 'interactive'): ").strip().lower()
    
    if user_input == 'interactive':
        # Interactive mode
        print("\n🗣️  Interactive Chat Mode")
        print("Commands: 'mode <1-4>' to change cognitive mode, 'quit' to exit")
        print("-" * 50)
        
        while True:
            try:
                user_message = input(f"\nYou [{current_mode}]: ").strip()
                
                if user_message.lower() == 'quit':
                    break
                elif user_message.startswith('mode '):
                    mode_choice = user_message.split()[1]
                    if mode_choice in modes:
                        current_mode = modes[mode_choice][0]
                        print(f"✅ Switched to {current_mode} mode")
                        continue
                    else:
                        print("❌ Invalid mode. Use 'mode 1', 'mode 2', 'mode 3', or 'mode 4'")
                        continue
                elif not user_message:
                    continue
                
                print("🧠 KIMERA is thinking...")
                result = chat.chat(user_message, current_mode)
                
                if "error" in result:
                    print(f"❌ Error: {result['error']}")
                else:
                    print(f"\n🤖 KIMERA [{current_mode}]: {result['response']}")
                    print(f"📊 Metrics - Confidence: {result.get('confidence', 0):.2f}, "
                          f"Coherence: {result.get('semantic_coherence', 0):.2f}, "
                          f"Time: {result.get('generation_time', 0):.2f}s")
                
            except KeyboardInterrupt:
                break
    else:
        # Demo mode
        for i, message in enumerate(demo_messages, 1):
            print(f"\n--- Demo {i}/4 ---")
            print(f"User: {message}")
            
            # Cycle through different modes for demo
            mode_cycle = ["standard", "cognitive_enhanced", "persona_aware", "neurodivergent"]
            current_mode = mode_cycle[(i-1) % len(mode_cycle)]
            
            print(f"🧠 KIMERA [{current_mode}] is thinking...")
            result = chat.chat(message, current_mode)
            
            if "error" in result:
                print(f"❌ Error: {result['error']}")
            else:
                print(f"\n🤖 KIMERA: {result['response']}")
                print(f"📊 Metrics - Confidence: {result.get('confidence', 0):.2f}, "
                      f"Coherence: {result.get('semantic_coherence', 0):.2f}, "
                      f"Resonance: {result.get('cognitive_resonance', 0):.2f}")
            
            if i < len(demo_messages):
                input("\nPress Enter to continue to next demo...")
    
    print("\n🎉 Chat session completed!")
    print("Thank you for experiencing KIMERA's enhanced conversation capabilities!")

if __name__ == "__main__":
    main() 