"""
Meta Commentary Integration - Core Integration Wrapper
=====================================================

Integrates the Meta Commentary Eliminator engine into the core Kimera system
to fix communication issues by eliminating meta-analysis responses.

This fixes the critical communication problem where Kimera generates responses like:
"The diffusion model reveals..." instead of directly answering questions.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

# Import the actual engine from engines directory
try:
    from ...engines.meta_commentary_eliminator import MetaCommentaryEliminator, CommentaryType
    from ...engines.human_interface import KimeraHumanInterface, ResponseMode, HumanResponse
    ENGINE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Meta commentary engine not available: {e}")
    ENGINE_AVAILABLE = False
    
    # Fallback classes
    class MetaCommentaryEliminator:
        def __init__(self): pass
        async def eliminate_meta_commentary(self, text): return text
    
    class KimeraHumanInterface:
        def __init__(self): pass
        def format_response(self, text, **kwargs): 
            return type('obj', (object,), {'content': text, 'confidence': 1.0})()

logger = logging.getLogger(__name__)

@dataclass
class CommunicationResult:
    """Result of communication processing"""
    original_text: str
    processed_text: str
    meta_commentary_removed: bool
    human_formatted: bool
    confidence: float
    processing_time: float
    timestamp: datetime

class MetaCommentaryIntegration:
    """
    Core integration wrapper for Meta Commentary Eliminator
    
    This class provides a clean interface for the core system to access
    the meta commentary elimination capabilities.
    """
    
    def __init__(self):
        """Initialize the meta commentary integration"""
        self.engine_available = ENGINE_AVAILABLE
        self.meta_eliminator = None
        self.human_interface = None
        self.total_processed = 0
        self.successful_eliminations = 0
        
        if self.engine_available:
            self.meta_eliminator = MetaCommentaryEliminator()
            self.human_interface = KimeraHumanInterface()
            logger.info("💬 Meta Commentary Integration initialized successfully")
        else:
            logger.warning("💬 Meta Commentary Integration using fallback mode")
    
    async def process_response(self, 
                             text: str,
                             eliminate_meta: bool = True,
                             human_format: bool = True,
                             thinking_summary: Optional[str] = None) -> CommunicationResult:
        """
        Process a response to eliminate meta-commentary and format for humans
        
        Args:
            text: The original response text
            eliminate_meta: Whether to eliminate meta-commentary
            human_format: Whether to format for human readability
            thinking_summary: Optional summary of thinking process
            
        Returns:
            CommunicationResult with processed text and metadata
        """
        start_time = asyncio.get_event_loop().time()
        processed_text = text
        meta_removed = False
        human_formatted = False
        confidence = 1.0
        
        try:
            # Step 1: Eliminate meta-commentary if requested and available
            if eliminate_meta and self.meta_eliminator:
                processed_text = await self.meta_eliminator.eliminate_meta_commentary(processed_text)
                meta_removed = True
                self.successful_eliminations += 1
                logger.debug(f"Meta-commentary eliminated from response")
            
            # Step 2: Format for human readability if requested
            if human_format and self.human_interface:
                formatted_response = self.human_interface.format_response(
                    processed_text,
                    thinking_summary=thinking_summary,
                    confidence=confidence
                )
                processed_text = formatted_response.content
                confidence = formatted_response.confidence
                human_formatted = True
                logger.debug(f"Response formatted for human readability")
            
            self.total_processed += 1
            processing_time = asyncio.get_event_loop().time() - start_time
            
            return CommunicationResult(
                original_text=text,
                processed_text=processed_text,
                meta_commentary_removed=meta_removed,
                human_formatted=human_formatted,
                confidence=confidence,
                processing_time=processing_time,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error processing response: {e}")
            processing_time = asyncio.get_event_loop().time() - start_time
            
            # Return original text if processing fails
            return CommunicationResult(
                original_text=text,
                processed_text=text,
                meta_commentary_removed=False,
                human_formatted=False,
                confidence=0.5,
                processing_time=processing_time,
                timestamp=datetime.now()
            )
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get statistics about communication processing"""
        success_rate = (self.successful_eliminations / max(self.total_processed, 1)) * 100
        
        return {
            "engine_available": self.engine_available,
            "total_processed": self.total_processed,
            "successful_eliminations": self.successful_eliminations,
            "success_rate": success_rate,
            "status": "operational" if self.engine_available else "fallback"
        }
    
    async def test_communication_fix(self) -> bool:
        """Test if the communication fix is working"""
        test_input = "The diffusion model reveals that your question demonstrates semantic complexity."
        
        result = await self.process_response(test_input)
        
        # Check if meta-commentary was removed
        meta_phrases = ["diffusion model", "analysis shows", "processing reveals"]
        has_meta = any(phrase in result.processed_text.lower() for phrase in meta_phrases)
        
        logger.info(f"Communication fix test: {'PASSED' if not has_meta else 'NEEDS_WORK'}")
        return not has_meta