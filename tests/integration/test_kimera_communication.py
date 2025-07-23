"""
Test Kimera Communication with Fixes
===================================

This script demonstrates how to interact with Kimera using the fixed
communication system that produces natural, direct responses.

Run this after starting Kimera:
    python kimera.py  # In one terminal
    python test_kimera_communication.py  # In another terminal
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Kimera API endpoint
KIMERA_URL = "http://localhost:8000"


async def test_chat(message: str, mode: str = "natural_language", 
                   cognitive_mode: str = None) -> Dict[str, Any]:
    """Send a chat message to Kimera and get response."""
    async with aiohttp.ClientSession() as session:
        payload = {
            "message": message,
            "mode": mode,
            "session_id": "test_session_001"
        }
        
        if cognitive_mode:
            payload["cognitive_mode"] = cognitive_mode
        
        try:
            async with session.post(
                f"{KIMERA_URL}/kimera/api/chat/",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(f"Error {response.status}: {error_text}")
                    return {"error": error_text}
        except aiohttp.ClientError as e:
            logger.error(f"Connection error: {e}")
            return {"error": str(e)}


async def test_kimera_modes():
    """Test different Kimera communication modes."""
    
    print("\n" + "="*60)
    print("🧠 KIMERA COMMUNICATION TEST")
    print("="*60)
    
    # Test 1: Simple greeting
    print("\n1️⃣ TEST: Simple Greeting")
    print("-" * 40)
    response = await test_chat("Hello Kimera, how are you today?")
    if "response" in response:
        print(f"KIMERA: {response['response']}")
        print(f"Confidence: {response.get('confidence', 0):.2%}")
    else:
        print(f"Error: {response.get('error', 'Unknown error')}")
    
    # Test 2: Complex philosophical question
    print("\n2️⃣ TEST: Philosophical Question")
    print("-" * 40)
    response = await test_chat(
        "What is consciousness and how do you experience it?",
        cognitive_mode="cognitive_enhanced"
    )
    if "response" in response:
        print(f"KIMERA: {response['response']}")
        print(f"Semantic Coherence: {response.get('semantic_coherence', 0):.2%}")
    else:
        print(f"Error: {response.get('error', 'Unknown error')}")
    
    # Test 3: Technical question
    print("\n3️⃣ TEST: Technical Question")
    print("-" * 40)
    response = await test_chat(
        "Can you explain how your cognitive field dynamics work?"
    )
    if "response" in response:
        print(f"KIMERA: {response['response']}")
    else:
        print(f"Error: {response.get('error', 'Unknown error')}")
    
    # Test 4: Emotional/Empathetic
    print("\n4️⃣ TEST: Emotional Connection")
    print("-" * 40)
    response = await test_chat(
        "I'm feeling overwhelmed by the complexity of understanding AI consciousness.",
        cognitive_mode="persona_aware"
    )
    if "response" in response:
        print(f"KIMERA: {response['response']}")
        print(f"Cognitive Resonance: {response.get('cognitive_resonance', 0):.2%}")
    else:
        print(f"Error: {response.get('error', 'Unknown error')}")
    
    # Test 5: Practical question
    print("\n5️⃣ TEST: Practical Application")
    print("-" * 40)
    response = await test_chat(
        "How can I use your capabilities for cryptocurrency trading?"
    )
    if "response" in response:
        print(f"KIMERA: {response['response']}")
    else:
        print(f"Error: {response.get('error', 'Unknown error')}")


async def interactive_chat():
    """Interactive chat session with Kimera."""
    print("\n" + "="*60)
    print("💬 INTERACTIVE CHAT WITH KIMERA")
    print("="*60)
    print("Type 'exit' to quit, 'mode:<mode>' to change cognitive mode")
    print("Available modes: standard, cognitive_enhanced, persona_aware, neurodivergent")
    print("-"*60)
    
    cognitive_mode = "standard"
    session_id = f"interactive_{int(asyncio.get_event_loop().time())}"
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'exit':
                print("Goodbye!")
                break
            
            if user_input.lower().startswith('mode:'):
                cognitive_mode = user_input.split(':')[1].strip()
                print(f"Switched to {cognitive_mode} mode")
                continue
            
            if not user_input:
                continue
            
            # Send to Kimera
            response = await test_chat(user_input, cognitive_mode=cognitive_mode)
            
            if "response" in response:
                print(f"\nKIMERA: {response['response']}")
                
                # Show metrics in subtle way
                confidence = response.get('confidence', 0)
                if confidence > 0:
                    print(f"[Confidence: {confidence:.0%}]", end="")
                    
            else:
                print(f"Error: {response.get('error', 'Unknown error')}")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


async def test_trading_analysis():
    """Test Kimera's trading analysis capabilities."""
    print("\n" + "="*60)
    print("📈 KIMERA TRADING ANALYSIS TEST")
    print("="*60)
    
    # Ask about market analysis
    response = await test_chat(
        "Analyze the current Bitcoin market conditions and suggest a trading strategy for a small account of $342.",
        cognitive_mode="cognitive_enhanced"
    )
    
    if "response" in response:
        print(f"KIMERA Trading Analysis:\n{response['response']}")
    else:
        print(f"Error: {response.get('error', 'Unknown error')}")


async def main():
    """Main test function."""
    print("""
    🧠 KIMERA COMMUNICATION TEST SUITE
    ==================================
    
    This will test Kimera's communication capabilities
    with the fixes applied for natural conversation.
    
    Make sure Kimera is running on http://localhost:8000
    """)
    
    # Check if Kimera is running
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{KIMERA_URL}/health") as response:
                if response.status == 200:
                    health = await response.json()
                    print(f"✅ Kimera is {health.get('status', 'unknown')}")
                else:
                    print("❌ Kimera is not responding. Please start it first.")
                    return
    except:
        print("❌ Cannot connect to Kimera. Please ensure it's running.")
        return
    
    while True:
        print("\nChoose a test:")
        print("1. Run all communication tests")
        print("2. Interactive chat with Kimera")
        print("3. Test trading analysis")
        print("4. Exit")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            await test_kimera_modes()
        elif choice == "2":
            await interactive_chat()
        elif choice == "3":
            await test_trading_analysis()
        elif choice == "4":
            break
        else:
            print("Invalid choice")


if __name__ == "__main__":
    asyncio.run(main())