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
import logging
logger = logging.getLogger(__name__)

async def demo_conversation():
    """Demonstrate intelligent conversation capabilities"""
    system = IntelligentConversationSystem()
    
    logger.info("=" * 80)
    logger.info("DEMO: Intelligent Conversation System")
    logger.info("Watch how it maintains context and builds on the conversation!")
    logger.info("=" * 80)
    
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
    
    logger.info("\nStarting demo conversation...\n")
    
    for i, message in enumerate(test_messages):
        logger.info(f"User: {message}")
        
        result = await system.process_message(message)
        
        logger.info(f"AI: {result['response']}")
        logger.info(f"   [Understanding: {result['understanding_depth']:.1%} | "
              f"Confidence: {result['confidence']:.1%} | "
              f"Relationship: {result['relationship_depth']:.1%}]")
        
        if result['current_topics']:
            logger.info(f"   [Topics tracked: {', '.join(result['current_topics'])}]")
            
        logger.info()
        
        # Show how context builds
        if i == 2:
            logger.info(">>> Notice how the AI remembers we're discussing AI and consciousness")
        elif i == 4:
            logger.info(">>> The AI now knows you work in neuroscience and adjusts responses")
            
        await asyncio.sleep(0.5)  # Brief pause for readability
        
    logger.info("\n" + "=" * 80)
    logger.info("Key differences from the template-based system:")
    logger.info("1. Remembers and builds on previous topics (AI, consciousness, neuroscience)")
    logger.info("2. Responses are contextual, not random phrases")
    logger.info("3. Relationship depth increases with meaningful exchanges")
    logger.info("4. Tracks conversation topics and user interests")
    logger.info("5. Each response is unique and relevant to the conversation flow")
    logger.info("=" * 80)

if __name__ == "__main__":
    asyncio.run(demo_conversation()) 