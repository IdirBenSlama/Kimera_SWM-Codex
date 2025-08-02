#!/usr/bin/env python3
"""
Demo of Intelligent Conversation System
Shows real context awareness and meaningful responses
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

from intelligent_conversation_system import IntelligentConversationSystem

async def demo_conversation():
    """Demonstrate intelligent conversation capabilities"""
    system = IntelligentConversationSystem()
    
    print("=" * 80)
    print("DEMO: Intelligent Conversation System")
    print("Watch how it maintains context and builds on the conversation!")
    print("=" * 80)
    
    await system.initialize()
    
    # Demo conversation showing context awareness
    test_messages = [
        "Hi there!",
        "I've been thinking about AI and consciousness lately",
        "Do you think machines can truly understand things?",
        "That's interesting. I work in neuroscience actually",
        "Yes, studying how the brain processes information",
        "What parallels do you see between brains and AI systems?"
    ]
    
    print("\nStarting demo conversation...\n")
    
    for i, message in enumerate(test_messages):
        print(f"User: {message}")
        
        result = await system.process_message(message)
        
        print(f"AI: {result['response']}")
        print(f"   [Understanding: {result['understanding_depth']:.1%} | "
              f"Confidence: {result['confidence']:.1%} | "
              f"Relationship: {result['relationship_depth']:.1%}]")
        
        if result['current_topics']:
            print(f"   [Topics tracked: {', '.join(result['current_topics'])}]")
            
        print()
        
        # Show how context builds
        if i == 2:
            print(">>> Notice how the AI remembers we're discussing AI and consciousness")
        elif i == 4:
            print(">>> The AI now knows you work in neuroscience and adjusts responses")
            
        await asyncio.sleep(0.5)  # Brief pause for readability
        
    print("\n" + "=" * 80)
    print("Key differences from the template-based system:")
    print("1. Remembers and builds on previous topics (AI, consciousness, neuroscience)")
    print("2. Responses are contextual, not random phrases")
    print("3. Relationship depth increases with meaningful exchanges")
    print("4. Tracks conversation topics and user interests")
    print("5. Each response is unique and relevant to the conversation flow")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(demo_conversation()) 