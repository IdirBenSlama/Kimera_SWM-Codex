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
        self.total_processed = 0
        self.successful_eliminations = 0
        logger.info("ðŸ’¬ Meta Commentary Integration initialized (fallback mode)")
    
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
        confidence = 1.0
        
        try:
            # Basic meta-commentary elimination (fallback implementation)
            if eliminate_meta:
                meta_patterns = [
                    "the diffusion model reveals",
                    "the analysis shows", 
                    "processing reveals",
                    "demonstrates how",
                    "the model processes",
                    "analysis indicates"
                ]
                
                original_text = processed_text
                for pattern in meta_patterns:
                    if pattern in processed_text.lower():
                        # Replace meta-commentary with direct response
                        processed_text = processed_text.replace(pattern, "I understand that")
                        meta_removed = True
                
                if meta_removed:
                    self.successful_eliminations += 1
                    logger.debug("Meta-commentary eliminated from response")
            
            # Basic human formatting
            if human_format and thinking_summary:
                processed_text = f"[{thinking_summary[:50]}...]\n\n{processed_text}"
            
            self.total_processed += 1
            processing_time = asyncio.get_event_loop().time() - start_time
            
            return CommunicationResult(
                original_text=text,
                processed_text=processed_text,
                meta_commentary_removed=meta_removed,
                human_formatted=human_format,
                confidence=confidence,
                processing_time=processing_time,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error processing response: {e}")
            processing_time = asyncio.get_event_loop().time() - start_time
            
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
            "engine_available": True,  # Fallback is always available
            "total_processed": self.total_processed,
            "successful_eliminations": self.successful_eliminations,
            "success_rate": success_rate,
            "status": "fallback"
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