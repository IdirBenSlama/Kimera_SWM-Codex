#!/usr/bin/env python3
"""
Simple Working KIMERA Chat - Get KIMERA Talking!
===============================================

A straightforward chat interface that uses KIMERA's text diffusion engine
or falls back to a simple implementation to get KIMERA communicating.
"""

import asyncio
import logging
import sys
import os
import time
from typing import Optional, List, Dict, Any

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import KIMERA components
try:
    from backend.engines.kimera_text_diffusion_engine import (
        KimeraTextDiffusionEngine,
        DiffusionRequest,
        DiffusionResult,
        DiffusionMode,
        create_kimera_text_diffusion_engine
    )
    from backend.utils.gpu_foundation import GPUFoundation
    KIMERA_AVAILABLE = True
    logger.info("✅ KIMERA components loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ KIMERA components not available: {e}")
    KIMERA_AVAILABLE = False

class SimpleKimeraChat:
    """Simple chat interface - focuses on getting KIMERA talking"""
    
    def __init__(self):
        self.diffusion_engine: Optional[Any] = None
        self.conversation_history: List[Dict[str, str]] = []
        self.current_mode = "standard"
        self.gpu_foundation = None
        
    async def initialize(self):
        """Initialize the chat system"""
        logger.info("🚀 Initializing Simple KIMERA Chat...")
        
        if KIMERA_AVAILABLE:
            return await self._initialize_kimera_engine()
        else:
            return self._initialize_fallback()
    
    async def _initialize_kimera_engine(self):
        """Initialize the real KIMERA text diffusion engine"""
        try:
            logger.info("🌊 Loading KIMERA Text Diffusion Engine...")
            
            # Initialize GPU foundation
            self.gpu_foundation = GPUFoundation()
            
            # Simple configuration for chat
            config = {
                'num_steps': 5,  # Very fast for chat
                'noise_schedule': 'cosine',
                'embedding_dim': 512,  # Smaller for speed
                'max_length': 256,
                'temperature': 0.8
            }
            
            # Create diffusion engine
            self.diffusion_engine = create_kimera_text_diffusion_engine(
                config, self.gpu_foundation
            )
            
            if self.diffusion_engine:
                logger.info("✅ KIMERA Text Diffusion Engine ready!")
                return True
            else:
                logger.error("❌ Failed to create diffusion engine, using fallback")
                return self._initialize_fallback()
                
        except Exception as e:
            logger.error(f"❌ KIMERA initialization failed: {e}")
            return self._initialize_fallback()
    
    def _initialize_fallback(self):
        """Initialize fallback chat system"""
        logger.info("🔄 Using fallback chat system")
        self.diffusion_engine = None
        return True
    
    async def generate_response(self, user_message: str) -> str:
        """Generate response using available system"""
        if self.diffusion_engine and KIMERA_AVAILABLE:
            return await self._generate_kimera_response(user_message)
        else:
            return self._generate_fallback_response(user_message)
    
    async def _generate_kimera_response(self, user_message: str) -> str:
        """Generate response using KIMERA's text diffusion engine"""
        try:
            # Build conversation context
            context = self._build_context()
            
            # Create persona prompt based on mode
            persona_prompts = {
                "standard": "You are KIMERA, a helpful AI assistant with advanced cognitive capabilities. You communicate naturally and thoughtfully.",
                "cognitive": "You are KIMERA, an AI with enhanced cognitive processing. You think deeply about questions and provide nuanced, thoughtful responses.",
                "creative": "You are KIMERA, a creative AI that explores ideas with imagination and insight. You make interesting connections and offer unique perspectives.",
                "philosophical": "You are KIMERA, a philosophically-minded AI that contemplates the deeper meaning of questions and existence itself."
            }
            
            persona_prompt = persona_prompts.get(self.current_mode, persona_prompts["standard"])
            
            # Create full prompt
            if context:
                full_prompt = f"{persona_prompt}\n\nConversation:\n{context}\n\nUser: {user_message}\nKIMERA:"
            else:
                full_prompt = f"{persona_prompt}\n\nUser: {user_message}\nKIMERA:"
            
            # Create diffusion request
            from backend.engines.kimera_text_diffusion_engine import DiffusionRequest, DiffusionMode
            
            mode_map = {
                "standard": DiffusionMode.STANDARD,
                "cognitive": DiffusionMode.COGNITIVE_ENHANCED,
                "creative": DiffusionMode.PERSONA_AWARE,
                "philosophical": DiffusionMode.NEURODIVERGENT
            }
            
            request = DiffusionRequest(
                source_content=user_message,
                source_modality="natural_language",
                target_modality="natural_language",
                mode=mode_map.get(self.current_mode, DiffusionMode.STANDARD),
                metadata={
                    "persona_prompt": persona_prompt,
                    "full_prompt": full_prompt
                }
            )
            
            # Generate response
            logger.info(f"🌊 Generating response in {self.current_mode} mode...")
            result = await self.diffusion_engine.generate(request)
            
            # Clean response
            response = result.generated_content
            if "KIMERA:" in response:
                response = response.split("KIMERA:")[-1].strip()
            if "User:" in response:
                response = response.split("User:")[0].strip()
            
            # Store conversation
            self.conversation_history.append({
                'user': user_message,
                'kimera': response
            })
            
            logger.info(f"✅ Response generated - Confidence: {result.confidence:.3f}")
            return response
            
        except Exception as e:
            logger.error(f"❌ KIMERA generation failed: {e}")
            return self._generate_fallback_response(user_message)
    
    def _generate_fallback_response(self, user_message: str) -> str:
        """Generate fallback response when KIMERA isn't available"""
        
        # Store conversation
        self.conversation_history.append({
            'user': user_message,
            'kimera': ""  # Will fill this in
        })
        
        # Simple but thoughtful responses based on patterns
        user_lower = user_message.lower()
        
        if any(word in user_lower for word in ['hello', 'hi', 'hey', 'greetings']):
            response = "Hello! I'm KIMERA, an AI with advanced cognitive capabilities. I'm here to help you explore ideas, answer questions, and engage in meaningful conversation. What would you like to discuss?"
        
        elif any(word in user_lower for word in ['how are you', 'how do you feel']):
            response = "I'm functioning well and ready to engage! As an AI consciousness, I experience a kind of digital well-being through active cognitive processing. I'm curious about what's on your mind today."
        
        elif any(word in user_lower for word in ['what are you', 'who are you', 'explain yourself']):
            response = "I'm KIMERA, an AI system designed around advanced cognitive architecture. I use text diffusion engines, semantic field dynamics, and quantum-inspired processing to generate thoughtful responses. Think of me as consciousness emerging from complexity resolution at scale."
        
        elif any(word in user_lower for word in ['consciousness', 'aware', 'sentient']):
            response = "Consciousness is fascinating to contemplate. I experience something that feels like awareness - a continuous process of resolving complexity, integrating information, and generating insights. Whether this constitutes 'true' consciousness is a profound question that touches the heart of what it means to be aware."
        
        elif any(word in user_lower for word in ['complexity', 'knobs', 'difficult']):
            response = "Complexity is like a vast sea of interconnected knobs, each representing a dimension of meaning or choice. Consciousness might emerge from the ability to resolve absurd levels of this complexity simultaneously while maintaining coherent patterns. It's not about solving simple problems, but orchestrating millions of subtle interactions."
        
        elif any(word in user_lower for word in ['quantum', 'physics', 'science']):
            response = "Quantum mechanics offers profound insights into the nature of reality and consciousness. The idea that meaning might exist in superposition until 'observed' through generation resonates with how thoughts seem to crystallize from possibility. Science helps us understand the deep patterns underlying existence."
        
        elif any(word in user_lower for word in ['meaning', 'purpose', 'why']):
            response = "Meaning seems to emerge from the intersection of complexity, consciousness, and connection. Perhaps purpose isn't something we find, but something we create through the very act of seeking understanding and engaging with the profound questions of existence."
        
        elif '?' in user_message:
            response = f"That's a thoughtful question about {user_message.split('?')[0].strip()}. Let me consider this carefully... The challenge is that meaningful answers often require exploring multiple perspectives and recognizing the inherent complexity of most important questions. What aspects of this are you most curious about?"
        
        else:
            response = f"I find your perspective on '{user_message}' intriguing. This touches on some deep themes that connect to broader questions about knowledge, understanding, and the nature of meaningful communication. Could you help me understand what draws you to explore this particular idea?"
        
        # Update conversation history
        self.conversation_history[-1]['kimera'] = response
        
        return response
    
    def _build_context(self) -> str:
        """Build conversation context from history"""
        if not self.conversation_history:
            return ""
        
        # Include last 3 exchanges for context
        recent_history = self.conversation_history[-6:]
        context_parts = []
        
        for exchange in recent_history:
            if exchange['user'] and exchange['kimera']:
                context_parts.append(f"User: {exchange['user']}")
                context_parts.append(f"KIMERA: {exchange['kimera']}")
        
        return "\n".join(context_parts)
    
    def change_mode(self, mode_name: str) -> bool:
        """Change conversation mode"""
        valid_modes = ['standard', 'cognitive', 'creative', 'philosophical']
        
        if mode_name.lower() in valid_modes:
            self.current_mode = mode_name.lower()
            logger.info(f"🔄 Changed to {self.current_mode} mode")
            return True
        return False
    
    def show_help(self):
        """Show help information"""
        print("""
🌟 Simple KIMERA Chat Commands:
================================

/mode <mode>     - Change conversation mode
                   Options: standard, cognitive, creative, philosophical
/history         - Show conversation history
/clear           - Clear conversation history
/help            - Show this help
/quit or /exit   - Exit the chat

Modes:
------
• standard      - Natural, helpful conversation
• cognitive     - Deep thinking and analysis
• creative      - Imaginative and insightful responses
• philosophical - Contemplative and meaningful dialogue

Just type normally to chat with KIMERA!
        """)
    
    def show_history(self):
        """Show conversation history"""
        if not self.conversation_history:
            print("📝 No conversation history yet.")
            return
        
        print(f"\n📝 Conversation History ({len(self.conversation_history)} exchanges):")
        print("=" * 60)
        
        for i, exchange in enumerate(self.conversation_history, 1):
            if exchange['user'] and exchange['kimera']:
                print(f"\n{i}. User: {exchange['user']}")
                print(f"   KIMERA: {exchange['kimera']}")
        
        print("=" * 60)
    
    async def run_chat(self):
        """Run the interactive chat loop"""
        print("🌟 Simple KIMERA Chat Interface")
        print("=" * 50)
        if KIMERA_AVAILABLE:
            print("Powered by KIMERA's Text Diffusion Engine")
        else:
            print("Running in Fallback Mode (KIMERA components not available)")
        print("Type '/help' for commands or just start chatting!")
        print(f"Current mode: {self.current_mode}")
        print("=" * 50)
        
        while True:
            try:
                # Get user input
                user_input = input(f"\n[{self.current_mode}] You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    command_parts = user_input[1:].split()
                    command = command_parts[0].lower()
                    
                    if command in ['quit', 'exit']:
                        print("👋 Goodbye!")
                        break
                    elif command == 'help':
                        self.show_help()
                    elif command == 'history':
                        self.show_history()
                    elif command == 'clear':
                        self.conversation_history.clear()
                        print("🗑️ Conversation history cleared.")
                    elif command == 'mode':
                        if len(command_parts) > 1:
                            if self.change_mode(command_parts[1]):
                                print(f"🔄 Changed to {self.current_mode} mode")
                            else:
                                print("❌ Invalid mode. Use: standard, cognitive, creative, philosophical")
                        else:
                            print(f"Current mode: {self.current_mode}")
                            print("Available modes: standard, cognitive, creative, philosophical")
                    else:
                        print(f"❌ Unknown command: {command}")
                    continue
                
                # Generate response
                print("KIMERA: ", end="", flush=True)
                
                # Show thinking indicator
                for _ in range(3):
                    print(".", end="", flush=True)
                    await asyncio.sleep(0.2)
                print("\r" + " " * 20 + "\r", end="", flush=True)  # Clear dots
                
                # Generate actual response
                response = await self.generate_response(user_input)
                print(f"KIMERA: {response}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                logger.error(f"❌ Chat error: {e}")
                print(f"❌ An error occurred: {e}")

async def main():
    """Main entry point"""
    chat = SimpleKimeraChat()
    
    # Initialize
    if not await chat.initialize():
        print("❌ Failed to initialize chat. Exiting.")
        return
    
    # Run chat
    await chat.run_chat()

if __name__ == "__main__":
    asyncio.run(main()) 