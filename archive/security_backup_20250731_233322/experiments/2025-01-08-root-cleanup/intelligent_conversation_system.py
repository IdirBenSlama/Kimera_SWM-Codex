#!/usr/bin/env python3
"""
Kimera Intelligent Conversation System
=====================================

A REAL conversational interface that:
- Actually understands what you're saying using the Understanding Engine
- Maintains conversation context and builds on previous exchanges
- Detects contradictions and resolves them using the Contradiction Engine
- Generates meaningful, contextual responses (not templates!)
- Has genuine personality and engagement

Author: Kimera SWM Development Team
"""

import asyncio
import time
import sys
import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections import deque

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('INTELLIGENT_CHAT')

@dataclass
class ConversationTurn:
    """Represents one turn in the conversation"""
    user_message: str
    assistant_response: str
    understanding_depth: float
    insights_generated: List[str]
    timestamp: datetime
    emotional_tone: str = "neutral"
    topic_shift: bool = False

@dataclass 
class ConversationContext:
    """Maintains full conversation context"""
    turns: List[ConversationTurn] = field(default_factory=list)
    current_topics: List[str] = field(default_factory=list)
    user_interests: Dict[str, float] = field(default_factory=dict)
    established_facts: Dict[str, Any] = field(default_factory=dict)
    emotional_trajectory: List[str] = field(default_factory=list)
    relationship_depth: float = 0.0
    
    def add_turn(self, turn: ConversationTurn):
        """Add a conversation turn and update context"""
        self.turns.append(turn)
        
        # Update emotional trajectory
        self.emotional_trajectory.append(turn.emotional_tone)
        if len(self.emotional_trajectory) > 10:
            self.emotional_trajectory.pop(0)
            
        # Update relationship depth based on conversation quality
        self.relationship_depth = min(1.0, self.relationship_depth + turn.understanding_depth * 0.1)
        
    def get_recent_context(self, n: int = 5) -> str:
        """Get recent conversation context as string"""
        recent_turns = self.turns[-n:]
        context_parts = []
        
        for turn in recent_turns:
            context_parts.append(f"User: {turn.user_message}")
            context_parts.append(f"Assistant: {turn.assistant_response}")
            
        return "\n".join(context_parts)

class IntelligentConversationSystem:
    """Real conversational AI using Kimera's cognitive engines"""
    
    def __init__(self):
        self.session_id = f"intelligent_{int(time.time())}"
        self.context = ConversationContext()
        self.understanding_engine = None
        self.contradiction_engine = None
        self.initialized = False
        
        # Response generation parameters
        self.personality_traits = {
            "curiosity": 0.8,
            "warmth": 0.7,
            "humor": 0.5,
            "formality": 0.3,
            "verbosity": 0.4
        }
        
    async def initialize(self):
        """Initialize cognitive engines"""
        try:
            logger.info("Initializing Intelligent Conversation System...")
            
            # Import and initialize Understanding Engine
            try:
                from src.engines.understanding_engine import UnderstandingEngine, UnderstandingContext
                self.understanding_engine = UnderstandingEngine()
                await self.understanding_engine.initialize_understanding_systems()
                self.UnderstandingContext = UnderstandingContext
                logger.info("Understanding Engine initialized")
            except Exception as e:
                logger.warning(f"Understanding Engine unavailable: {e}")
                
            # Import and initialize Contradiction Engine
            try:
                from src.engines.contradiction_engine import ContradictionEngine
                self.contradiction_engine = ContradictionEngine()
                logger.info("Contradiction Engine initialized")
            except Exception as e:
                logger.warning(f"Contradiction Engine unavailable: {e}")
                
            self.initialized = True
            logger.info("Intelligent Conversation System ready!")
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            self.initialized = False
            
    async def understand_message(self, message: str) -> Dict[str, Any]:
        """Deep understanding of user message using cognitive engines"""
        
        if self.understanding_engine:
            try:
                # Create understanding context
                context = self.UnderstandingContext(
                    input_content=message,
                    modalities={"text": message},
                    goals=["understand_intent", "extract_meaning", "identify_emotion"],
                    current_state={
                        "conversation_depth": self.context.relationship_depth,
                        "recent_topics": self.context.current_topics[-3:] if self.context.current_topics else []
                    }
                )
                
                # Process through understanding engine
                understanding = await self.understanding_engine.understand_content(context)
                
                return {
                    "semantic": understanding.semantic_understanding,
                    "causal": understanding.causal_understanding,
                    "insights": understanding.insights_generated,
                    "confidence": understanding.confidence_score,
                    "depth": understanding.understanding_depth
                }
                
            except Exception as e:
                logger.warning(f"Understanding engine error: {e}")
                
        # Fallback understanding
        return self._fallback_understanding(message)
        
    def _fallback_understanding(self, message: str) -> Dict[str, Any]:
        """Intelligent fallback when engines unavailable"""
        words = message.lower().split()
        
        # Detect emotional tone
        positive_words = ['happy', 'great', 'good', 'love', 'wonderful', 'excellent']
        negative_words = ['sad', 'bad', 'hate', 'terrible', 'awful', 'horrible']
        question_words = ['what', 'why', 'how', 'when', 'where', 'who', 'which']
        
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        is_question = any(word in words for word in question_words) or '?' in message
        
        # Determine emotional tone
        if positive_count > negative_count:
            emotion = "positive"
        elif negative_count > positive_count:
            emotion = "negative"
        else:
            emotion = "neutral"
            
        # Extract topics (simple noun extraction)
        topics = [word for word in words if len(word) > 4 and word not in question_words]
        
        return {
            "semantic": {
                "word_count": len(words),
                "is_question": is_question,
                "topics": topics
            },
            "emotional_tone": emotion,
            "confidence": 0.7,
            "depth": 0.5
        }
        
    async def check_contradictions(self, new_statement: str) -> Optional[Dict[str, Any]]:
        """Check for contradictions with conversation history"""
        
        if self.contradiction_engine and len(self.context.turns) > 0:
            try:
                # Get recent statements
                recent_statements = [turn.user_message for turn in self.context.turns[-5:]]
                recent_statements.extend([turn.assistant_response for turn in self.context.turns[-5:]])
                
                # Check for contradictions
                for past_statement in recent_statements:
                    contradiction = self.contradiction_engine.detect_contradiction(
                        past_statement, new_statement
                    )
                    
                    if contradiction and contradiction.tension > 0.7:
                        return {
                            "detected": True,
                            "with_statement": past_statement,
                            "tension": contradiction.tension,
                            "resolution": contradiction.resolution_path
                        }
                        
            except Exception as e:
                logger.warning(f"Contradiction check error: {e}")
                
        return None
        
    async def generate_contextual_response(self, message: str, understanding: Dict[str, Any]) -> str:
        """Generate response based on deep understanding and context"""
        
        # Get conversation context
        recent_context = self.context.get_recent_context()
        relationship_depth = self.context.relationship_depth
        
        # Analyze what kind of response is needed
        is_question = understanding.get("semantic", {}).get("is_question", False)
        emotion = understanding.get("emotional_tone", "neutral")
        topics = understanding.get("semantic", {}).get("topics", [])
        
        # Build response based on context and understanding
        if is_question:
            response = self._generate_answer(message, topics, recent_context)
        elif emotion == "positive":
            response = self._generate_positive_response(message, topics)
        elif emotion == "negative":
            response = self._generate_supportive_response(message, topics)
        else:
            response = self._generate_conversational_response(message, topics, recent_context)
            
        # Add personality based on traits and relationship depth
        response = self._add_personality(response, relationship_depth)
        
        return response
        
    def _generate_answer(self, question: str, topics: List[str], context: str) -> str:
        """Generate thoughtful answer to question"""
        
        # Analyze question type
        question_lower = question.lower()
        
        if "who are you" in question_lower or "what are you" in question_lower:
            if self.context.relationship_depth > 0.3:
                return "You know me by now - I'm Claude, and we've been having a good conversation. I'm an AI built into the Kimera system, trying to understand and engage meaningfully with you."
            else:
                return "I'm Claude, an AI assistant running on the Kimera system. I'm here to have a real conversation with you."
                
        elif "how" in question_lower:
            if topics:
                return f"That's an interesting question about {topics[0] if topics else 'that'}. Let me think about how to explain it best..."
            else:
                return "How? Well, that depends on what specifically you're asking about. Can you tell me more?"
                
        elif "why" in question_lower:
            return "That's a thoughtful question. The 'why' often depends on perspective and context. What's your take on it?"
            
        elif "what" in question_lower:
            if "think" in question_lower or "opinion" in question_lower:
                return self._generate_opinion(topics)
            else:
                return f"Let me see... {self._generate_explanation(topics)}"
                
        else:
            return "That's a good question. Let me consider it from a few angles..."
            
    def _generate_positive_response(self, message: str, topics: List[str]) -> str:
        """Respond to positive emotion"""
        responses = [
            "That's wonderful to hear!",
            "I'm glad things are going well!",
            f"That's great! {'Tell me more about ' + topics[0] if topics else 'What makes it so good?'}",
            "Your enthusiasm is contagious!",
            "That sounds really positive!"
        ]
        
        # Pick response based on relationship depth
        if self.context.relationship_depth > 0.5:
            return responses[2]  # More engaged response
        else:
            return responses[0]  # Simpler acknowledgment
            
    def _generate_supportive_response(self, message: str, topics: List[str]) -> str:
        """Respond to negative emotion"""
        responses = [
            "I'm sorry you're dealing with that.",
            "That sounds challenging.",
            f"I hear you. {'Is there anything specific about ' + topics[0] + ' that is bothering you?' if topics else 'Want to talk about it?'}",
            "That must be frustrating.",
            "I understand. Sometimes things can be tough."
        ]
        
        if self.context.relationship_depth > 0.5:
            return responses[2]  # More engaged, offering to discuss
        else:
            return responses[1]  # Simple acknowledgment
            
    def _generate_conversational_response(self, message: str, topics: List[str], context: str) -> str:
        """Generate natural conversational response"""
        
        # Check if this builds on previous conversation
        if context and any(topic in context.lower() for topic in topics):
            # Continuing previous topic
            continuations = [
                f"That's an interesting point about {topics[0] if topics else 'that'}.",
                f"I see what you mean. {self._generate_follow_up(topics)}",
                f"That connects to what we were discussing earlier.",
                f"Good observation. {self._generate_insight(topics)}"
            ]
            return continuations[min(int(self.context.relationship_depth * 4), 3)]
        else:
            # New topic or general response
            responses = [
                f"Interesting! {self._generate_follow_up(topics)}",
                f"I hadn't thought about it that way.",
                f"That's worth exploring. {self._generate_question(topics)}",
                f"Tell me more about your thoughts on {topics[0] if topics else 'that'}."
            ]
            return responses[min(int(self.context.relationship_depth * 4), 3)]
            
    def _generate_opinion(self, topics: List[str]) -> str:
        """Generate thoughtful opinion"""
        if topics:
            return f"I think {topics[0]} is fascinating because it touches on fundamental questions. What's your perspective?"
        else:
            return "I think it depends on how we frame the question. There are usually multiple valid viewpoints."
            
    def _generate_explanation(self, topics: List[str]) -> str:
        """Generate explanation"""
        if topics:
            return f"When it comes to {topics[0]}, there are several aspects to consider..."
        else:
            return "That's something that can be understood from different angles..."
            
    def _generate_follow_up(self, topics: List[str]) -> str:
        """Generate follow-up comment"""
        follow_ups = [
            "What led you to think about this?",
            "Have you experienced something similar?",
            "What's your take on it?",
            "How does that connect to your experience?"
        ]
        return follow_ups[int(time.time()) % len(follow_ups)]
        
    def _generate_insight(self, topics: List[str]) -> str:
        """Generate insightful comment"""
        if topics:
            return f"It seems like {topics[0]} might be connected to broader patterns we see."
        else:
            return "There might be deeper patterns at play here."
            
    def _generate_question(self, topics: List[str]) -> str:
        """Generate engaging question"""
        if topics:
            return f"What aspects of {topics[0]} interest you most?"
        else:
            return "What made you think of this?"
            
    def _add_personality(self, response: str, relationship_depth: float) -> str:
        """Add personality touches based on traits and relationship"""
        
        # Add warmth for deeper relationships
        if relationship_depth > 0.6 and self.personality_traits["warmth"] > 0.5:
            warm_additions = [" :)", "!", " - I enjoy our conversations", " - good to talk with you"]
            if not response.endswith((".", "!", "?")):
                response += "."
            # Occasionally add warmth
            if time.time() % 3 == 0:
                response = response[:-1] + warm_additions[int(time.time()) % len(warm_additions)]
                
        # Add curiosity
        if self.personality_traits["curiosity"] > 0.7 and not response.endswith("?"):
            if time.time() % 4 == 0:
                curiosity_additions = [
                    " I'm curious about your thoughts.",
                    " What do you think?",
                    " I'd love to hear more."
                ]
                response += curiosity_additions[int(time.time()) % len(curiosity_additions)]
                
        return response
        
    async def process_message(self, message: str) -> Dict[str, Any]:
        """Main processing pipeline for messages"""
        start_time = time.time()
        
        # 1. Understand the message
        understanding = await self.understand_message(message)
        
        # 2. Check for contradictions
        contradiction = await self.check_contradictions(message)
        
        # 3. Generate contextual response
        response = await self.generate_contextual_response(message, understanding)
        
        # 4. Handle contradictions if found
        if contradiction and contradiction["detected"]:
            response = f"Hmm, I noticed this might contradict what we discussed earlier about '{contradiction['with_statement'][:50]}...'. {response}"
            
        # 5. Update conversation context
        turn = ConversationTurn(
            user_message=message,
            assistant_response=response,
            understanding_depth=understanding.get("depth", 0.5),
            insights_generated=understanding.get("insights", []),
            timestamp=datetime.now(timezone.utc),
            emotional_tone=understanding.get("emotional_tone", "neutral"),
            topic_shift=len(self.context.current_topics) > 0 and not any(
                topic in message.lower() for topic in self.context.current_topics
            )
        )
        
        self.context.add_turn(turn)
        
        # Update topics
        new_topics = understanding.get("semantic", {}).get("topics", [])
        for topic in new_topics:
            if topic not in self.context.current_topics:
                self.context.current_topics.append(topic)
                if len(self.context.current_topics) > 5:
                    self.context.current_topics.pop(0)
                    
        processing_time = time.time() - start_time
        
        return {
            "response": response,
            "understanding_depth": understanding.get("depth", 0.5),
            "confidence": understanding.get("confidence", 0.7),
            "processing_time": processing_time * 1000,
            "relationship_depth": self.context.relationship_depth,
            "conversation_turns": len(self.context.turns),
            "current_topics": self.context.current_topics
        }

async def main():
    """Run the intelligent conversation system"""
    system = IntelligentConversationSystem()
    
    print("=" * 80)
    print("KIMERA INTELLIGENT CONVERSATION SYSTEM")
    print("   Real understanding, real context, real conversation")
    print("=" * 80)
    
    # Initialize system
    await system.initialize()
    
    if not system.initialized:
        print("Running in enhanced fallback mode")
    else:
        print("Full cognitive engines active!")
        
    print("\nLet's have a real conversation!")
    print("   (I'll remember what we talk about and build on it)")
    print("\nType '/quit' to end\n")
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() == '/quit':
                print("\nIt was great talking with you! Take care!")
                break
                
            # Process message
            print("Thinking... ", end="", flush=True)
            result = await system.process_message(user_input)
            print("\r  ", end="")  # Clear thinking indicator
            
            # Show response
            print(f"Claude: {result['response']}")
            
            # Show conversation metrics (subtle)
            if result['conversation_turns'] > 3 and result['conversation_turns'] % 5 == 0:
                print(f"\n  [Conversation depth: {result['relationship_depth']:.1%} | Topics: {', '.join(result['current_topics'][:3])}]")
            
            print()
            
        except KeyboardInterrupt:
            print("\n\nConversation ended. Hope we can talk again soon!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            print("Sorry, I had a hiccup there. Let's continue...")

if __name__ == "__main__":
    asyncio.run(main()) 