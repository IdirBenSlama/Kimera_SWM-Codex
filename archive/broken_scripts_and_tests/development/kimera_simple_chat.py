#!/usr/bin/env python3
"""
Simple KIMERA Chat Interface using Text Diffusion Engine
========================================================

A straightforward chat interface that lets you talk to KIMERA using its
text diffusion engine - like a normal chatbot but powered by KIMERA's
advanced text generation capabilities.
"""

import asyncio
import logging
import sys
import os
from typing import Optional, List, Dict, Any

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import KIMERA components
try:
    from backend.engines.kimera_text_diffusion_engine import (
        KimeraTextDiffusionEngine,
        DiffusionRequest,
        DiffusionResult,
        DiffusionMode,
        create_kimera_text_diffusion_engine
    )
    from backend.utils.gpu_foundation import GPUFoundation
    from backend.utils.kimera_logger import get_logger, LogCategory
    KIMERA_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ KIMERA components not available: {e}")
    KIMERA_AVAILABLE = False

# Setup logging
if KIMERA_AVAILABLE:
    logger = get_logger(__name__, LogCategory.SYSTEM)
else:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

class KimeraSimpleChat:
    """Simple chat interface using KIMERA's text diffusion engine"""
    
    def __init__(self):
        self.diffusion_engine: Optional[KimeraTextDiffusionEngine] = None
        self.conversation_history: List[Dict[str, str]] = []
        self.current_mode = DiffusionMode.STANDARD
        self.gpu_foundation = None
        
    async def initialize(self):
        """Initialize the text diffusion engine"""
        if not KIMERA_AVAILABLE:
            logger.error("❌ KIMERA components not available")
            return False
            
        try:
            if hasattr(logger, 'info'):
                logger.info("🌊 Initializing KIMERA Text Diffusion Engine...")
            else:
                logger.info("🌊 Initializing KIMERA Text Diffusion Engine...")
            
            # Initialize GPU foundation
            self.gpu_foundation = GPUFoundation()
            
            # Configuration for chat
            config = {
                'num_steps': 10,  # Fast generation for chat
                'noise_schedule': 'cosine',
                'embedding_dim': 1024,
                'max_length': 512,
                'temperature': 0.8,
                'top_k': 50,
                'top_p': 0.9
            }
            
            # Create diffusion engine
            self.diffusion_engine = create_kimera_text_diffusion_engine(
                config, self.gpu_foundation
            )
            
            if self.diffusion_engine:
                if hasattr(logger, 'info'):
                    logger.info("✅ KIMERA Text Diffusion Engine initialized successfully")
                else:
                    logger.info("✅ KIMERA Text Diffusion Engine initialized successfully")
                return True
            else:
                if hasattr(logger, 'error'):
                    logger.error("❌ Failed to create diffusion engine")
                else:
                    logger.error("❌ Failed to create diffusion engine")
                return False
                
        except Exception as e:
            if hasattr(logger, 'error'):
                logger.error(f"❌ Failed to initialize: {e}")
            else:
                logger.error(f"❌ Failed to initialize: {e}")
            return False
    
    def get_persona_prompt(self, mode: DiffusionMode) -> str:
        """Get persona prompt based on mode"""
        prompts = {
            DiffusionMode.STANDARD: "You are KIMERA, a helpful AI assistant with advanced text generation capabilities. You communicate naturally and helpfully.",
            
            DiffusionMode.COGNITIVE_ENHANCED: "You are KIMERA, an AI with enhanced cognitive capabilities. You think deeply about questions, consider multiple perspectives, and provide thoughtful, nuanced responses.",
            
            DiffusionMode.PERSONA_AWARE: "You are KIMERA, an adaptive AI that mirrors the user's communication style while maintaining your helpful nature. You adjust your complexity and tone to match the user's needs.",
            
            DiffusionMode.NEURODIVERGENT: "You are KIMERA, an AI designed to communicate clearly and systematically. You provide structured responses, explain connections between concepts, and celebrate different ways of thinking."
        }
        return prompts.get(mode, prompts[DiffusionMode.STANDARD])
    
    def build_conversation_context(self) -> str:
        """Build conversation context from history"""
        if not self.conversation_history:
            return ""
        
        # Include last few exchanges for context
        recent_history = self.conversation_history[-6:]  # Last 3 exchanges
        context_parts = []
        
        for exchange in recent_history:
            context_parts.append(f"User: {exchange['user']}")
            context_parts.append(f"KIMERA: {exchange['kimera']}")
        
        return "\n".join(context_parts)
    
    async def generate_response(self, user_message: str) -> str:
        """Generate response using text diffusion engine"""
        if not self.diffusion_engine:
            return "❌ Text diffusion engine not available. Please restart the chat."
        
        try:
            # Build conversation context
            conversation_context = self.build_conversation_context()
            persona_prompt = self.get_persona_prompt(self.current_mode)
            
            # Create full prompt with context
            if conversation_context:
                full_prompt = f"{persona_prompt}\n\nConversation so far:\n{conversation_context}\n\nUser: {user_message}\nKIMERA:"
            else:
                full_prompt = f"{persona_prompt}\n\nUser: {user_message}\nKIMERA:"
            
            # Create diffusion request
            request = DiffusionRequest(
                source_content=user_message,
                source_modality="natural_language",
                target_modality="natural_language",
                mode=self.current_mode,
                metadata={
                    "persona_prompt": persona_prompt,
                    "conversation_context": conversation_context,
                    "full_prompt": full_prompt
                }
            )
            
            # Generate response
            logger.info(f"🌊 Generating response in {self.current_mode.value} mode...")
            result = await self.diffusion_engine.generate(request)
            
            # Extract and clean response
            response = result.generated_content
            
            # Clean up response (remove any prompt artifacts)
            if "KIMERA:" in response:
                response = response.split("KIMERA:")[-1].strip()
            if "User:" in response:
                response = response.split("User:")[0].strip()
            
            # Store in conversation history
            self.conversation_history.append({
                'user': user_message,
                'kimera': response
            })
            
            # Log generation metrics
            logger.info(f"✅ Response generated - Confidence: {result.confidence:.3f}, "
                       f"Coherence: {result.semantic_coherence:.3f}, "
                       f"Time: {result.generation_time:.2f}s")
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Failed to generate response: {e}")
            return f"I apologize, I'm experiencing some difficulty generating a response: {e}"
    
    def change_mode(self, mode_name: str) -> bool:
        """Change conversation mode"""
        mode_map = {
            'standard': DiffusionMode.STANDARD,
            'cognitive': DiffusionMode.COGNITIVE_ENHANCED,
            'persona': DiffusionMode.PERSONA_AWARE,
            'neurodivergent': DiffusionMode.NEURODIVERGENT
        }
        
        if mode_name.lower() in mode_map:
            self.current_mode = mode_map[mode_name.lower()]
            logger.info(f"🔄 Changed to {self.current_mode.value} mode")
            return True
        return False
    
    def show_help(self):
        """Show help information"""
        print("""
🌟 KIMERA Simple Chat Commands:
================================

/mode <mode>     - Change conversation mode
                   Options: standard, cognitive, persona, neurodivergent
/history         - Show conversation history
/clear           - Clear conversation history
/help            - Show this help
/quit or /exit   - Exit the chat

Modes:
------
• standard      - Natural, helpful conversation
• cognitive     - Deep thinking and analysis
• persona       - Adapts to your communication style
• neurodivergent - Clear, structured responses

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
            print(f"\n{i}. User: {exchange['user']}")
            print(f"   KIMERA: {exchange['kimera']}")
        
        print("=" * 60)
    
    async def run_chat(self):
        """Run the interactive chat loop"""
        print("🌟 KIMERA Simple Chat Interface")
        print("=" * 50)
        print("Powered by KIMERA's Text Diffusion Engine")
        print("Type '/help' for commands or just start chatting!")
        print(f"Current mode: {self.current_mode.value}")
        print("=" * 50)
        
        while True:
            try:
                # Get user input
                user_input = input(f"\n[{self.current_mode.value}] You: ").strip()
                
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
                                print(f"🔄 Changed to {self.current_mode.value} mode")
                            else:
                                print("❌ Invalid mode. Use: standard, cognitive, persona, neurodivergent")
                        else:
                            print(f"Current mode: {self.current_mode.value}")
                            print("Available modes: standard, cognitive, persona, neurodivergent")
                    else:
                        print(f"❌ Unknown command: {command}")
                    continue
                
                # Generate response
                print("KIMERA: ", end="", flush=True)
                
                # Show typing indicator
                for _ in range(3):
                    print(".", end="", flush=True)
                    await asyncio.sleep(0.3)
                print("\r", end="", flush=True)  # Clear dots
                
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
    chat = KimeraSimpleChat()
    
    # Initialize
    if not await chat.initialize():
        print("❌ Failed to initialize KIMERA. Exiting.")
        return
    
    # Run chat
    await chat.run_chat()

if __name__ == "__main__":
    asyncio.run(main()) 