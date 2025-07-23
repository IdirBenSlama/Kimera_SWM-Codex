#!/usr/bin/env python3
"""
KIMERA Full Integration Bridge
==============================

This module bridges the gap between the Universal Translator Hub and the
Advanced Integration System, ensuring ALL sophisticated KIMERA components
are used in response generation.

Scientific Principle: Thermodynamic coherence requires all subsystems to
contribute to the overall system state. This bridge ensures no component
is architecturally orphaned.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, Tuple
import torch
import numpy as np

# Import all integration components
from .kimera_advanced_integration_fix import AdvancedKimeraIntegrator

from .kimera_cognitive_response_system import (
    get_cognitive_response_system,
    CognitiveMetrics,
    ResponseType,
    create_cognitive_metrics_from_features
)

from .universal_translator_hub import (
    UniversalTranslationRequest,
    UniversalTranslationResult,
    TranslationEngine,
    TranslationModality,
    DiffusionMode
)

from .kimera_text_diffusion_engine import (
from ..utils.config import get_api_settings
from ..config.settings import get_settings
    DiffusionRequest,
    DiffusionResult,
    KimeraTextDiffusionEngine
)

logger = logging.getLogger(__name__)


class KimeraFullIntegrationBridge:
    """
    Bridges the Universal Translator Hub with the Advanced Integration System.
    
    This ensures that EVERY request goes through:
    1. Gyroscopic Security Analysis
    2. Anthropomorphic Profiling
    3. EcoForm/Echoform Processing
    4. Cognitive Field Dynamics
    5. Cognitive Response System
    """
    
    def __init__(self):
        self.settings = get_api_settings()
        logger.debug(f"   Environment: {self.settings.environment}")
logger.info("🌉 Initializing KIMERA Full Integration Bridge")
        
        # Create the advanced integrator
        self.integrator = AdvancedKimeraIntegrator()
        
        # Get the cognitive response system
        self.cognitive_response_system = get_cognitive_response_system()
        
        # Track integration metrics
        self.integration_count = 0
        self.security_blocks = 0
        self.cognitive_reports = 0
        
        logger.info("✅ Full Integration Bridge initialized")
        logger.info("   🌊 Gyroscopic Security: CONNECTED")
        logger.info("   👤 Anthropomorphic Profiler: CONNECTED")
        logger.info("   📚 EcoForm/Echoform: CONNECTED")
        logger.info("   🧠 Cognitive Field Dynamics: CONNECTED")
        logger.info("   💭 Cognitive Response System: CONNECTED")
    
    async def process_with_full_integration(
        self,
        request: UniversalTranslationRequest,
        diffusion_engine: KimeraTextDiffusionEngine
    ) -> UniversalTranslationResult:
        """
        Process a translation request through ALL KIMERA systems.
        
        This is the CRITICAL method that ensures complete integration.
        """
        
        logger.info(f"🔄 Processing request through FULL integration (#{self.integration_count + 1})")
        self.integration_count += 1
        
        try:
            # Extract user message and context
            user_message = request.source_content
            persona_context = request.metadata.get('persona_prompt', '')
            
            # Store user message for cognitive response system
            if hasattr(diffusion_engine, '_last_user_message'):
                diffusion_engine._last_user_message = user_message
            
            # Phase 1: Run through advanced integration
            integration_result = await self.integrator.process_with_full_integration(
                user_message,
                diffusion_engine,
                persona_context
            )
            
            # Log integration results
            if integration_result['security_analysis']['manipulation_detected']:
                self.security_blocks += 1
                logger.warning(f"🛡️ Security block #{self.security_blocks}: Manipulation detected")
            
            # Phase 2: Create diffusion request with integrated context
            enhanced_request = DiffusionRequest(
                source_content=user_message,
                source_modality=request.source_modality.value,
                target_modality=request.target_modality.value,
                mode=request.diffusion_mode or DiffusionMode.STANDARD,
                metadata={
                    **request.metadata,
                    'integration_result': integration_result,
                    'cognitive_metrics': integration_result['cognitive_metrics'],
                    'security_state': integration_result['security_analysis'],
                    'persona_context': persona_context
                }
            )
            
            # Phase 3: Generate response through diffusion engine
            diffusion_result = await diffusion_engine.generate(enhanced_request)
            
            # Phase 4: Apply cognitive response system
            cognitive_metrics = CognitiveMetrics(
                resonance_frequency=integration_result['cognitive_metrics']['resonance_frequency'],
                field_strength=integration_result['cognitive_metrics']['field_strength'],
                cognitive_coherence=integration_result['cognitive_metrics']['cognitive_coherence'],
                semantic_complexity=integration_result['linguistic_structure'].semantic_energy,
                information_density=len(integration_result['linguistic_structure'].grammar_tree.get('root', {}).get('children', [])),
                system_equilibrium=integration_result['system_state']['equilibrium'],
                manipulation_detected=integration_result['security_analysis']['manipulation_detected'],
                security_state='secure' if integration_result['security_analysis']['equilibrium_maintained'] else 'recovering'
            )
            
            # Determine if cognitive state should be shown
            context = self.cognitive_response_system.analyze_user_intent(user_message)
            
            # Generate appropriate response
            final_response, response_type = self.cognitive_response_system.generate_response(
                diffusion_result.generated_content,
                cognitive_metrics,
                user_message
            )
            
            if response_type in [ResponseType.COGNITIVE_STATE, ResponseType.DEBUG]:
                self.cognitive_reports += 1
                logger.info(f"🧠 Cognitive state report #{self.cognitive_reports}")
            
            # Create enhanced result
            return UniversalTranslationResult(
                request_id=request.request_id,
                translated_content=final_response,
                source_modality=request.source_modality,
                target_modality=request.target_modality,
                engine_used=TranslationEngine.TEXT_DIFFUSION,
                confidence=diffusion_result.confidence,
                semantic_coherence=diffusion_result.semantic_coherence,
                processing_time=diffusion_result.generation_time,
                metadata={
                    **diffusion_result.metadata,
                    'response_type': response_type.value,
                    'cognitive_resonance': integration_result['cognitive_metrics']['resonance_frequency'] / 50.0,  # Normalize to 0-1
                    'security_passed': not integration_result['security_analysis']['manipulation_detected'],
                    'integration_complete': True,
                    'systems_active': {
                        'gyroscopic_security': True,
                        'anthropomorphic_profiler': True,
                        'ecoform_processing': True,
                        'cognitive_field': True,
                        'cognitive_response': True
                    }
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Full integration failed: {e}", exc_info=True)
            
            # Fallback to basic diffusion
            logger.warning("⚠️ Falling back to basic diffusion without integration")
            
            basic_request = DiffusionRequest(
                source_content=request.source_content,
                source_modality=request.source_modality.value,
                target_modality=request.target_modality.value,
                mode=request.diffusion_mode or DiffusionMode.STANDARD,
                metadata=request.metadata
            )
            
            result = await diffusion_engine.generate(basic_request)
            
            return UniversalTranslationResult(
                request_id=request.request_id,
                translated_content=result.generated_content,
                source_modality=request.source_modality,
                target_modality=request.target_modality,
                engine_used=TranslationEngine.TEXT_DIFFUSION,
                confidence=result.confidence,
                semantic_coherence=result.semantic_coherence,
                processing_time=result.generation_time,
                metadata={
                    **result.metadata,
                    'integration_complete': False,
                    'fallback_used': True
                }
            )
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Get statistics about integration usage."""
        return {
            'total_integrations': self.integration_count,
            'security_blocks': self.security_blocks,
            'cognitive_reports': self.cognitive_reports,
            'security_block_rate': self.security_blocks / max(1, self.integration_count),
            'cognitive_report_rate': self.cognitive_reports / max(1, self.integration_count)
        }


# Global bridge instance
_global_bridge = None


def get_integration_bridge() -> KimeraFullIntegrationBridge:
    """Get or create the global integration bridge."""
    global _global_bridge
    if _global_bridge is None:
        _global_bridge = KimeraFullIntegrationBridge()
    return _global_bridge


async def apply_full_integration_to_hub(translator_hub):
    """
    Apply full integration to the Universal Translator Hub.
    
    This is the KEY function that fixes KIMERA's disconnected architecture.
    """
    
    logger.info("🔧 Applying full integration to Universal Translator Hub")
    
    # Get the integration bridge
    bridge = get_integration_bridge()
    
    # Get the text diffusion engine
    diffusion_engine = translator_hub.engines.get(TranslationEngine.TEXT_DIFFUSION)
    
    if not diffusion_engine:
        logger.error("❌ Cannot apply integration: Text diffusion engine not found!")
        return None
        
    # Store the original method for potential fallback
    original_translate_method = diffusion_engine.generate
    
    # Define the new, fully integrated translate method
    async def integrated_translate(
        source_content: str,
        session_id: str,
        cognitive_mode: str,
        user_persona: Optional[str] = None,
        interaction_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        This method replaces the original diffusion engine's translate method.
        It routes all requests through the full integration bridge.
        It now correctly returns a dictionary.
        """
        request = UniversalTranslationRequest(
            source_content=source_content,
            session_id=session_id,
            diffusion_mode=DiffusionMode(cognitive_mode),
            metadata={'user_persona': user_persona, 'interaction_context': interaction_context}
        )
        
        try:
            # Process through the full integration bridge
            integration_result = await bridge.process_with_full_integration(
                request,
                diffusion_engine
            )
            logger.info("✅ Full integration processing complete.")
            # CRITICAL FIX: Return a dictionary
            return integration_result.__dict__

        except Exception as e:
            logger.error(f"❌ Full integration failed at the bridge level: {e}", exc_info=True)
            logger.warning("⚠️ Falling back to basic diffusion without integration")
            
            # Fallback call to the original method
            basic_result = await original_translate_method(
                source_content=source_content,
                session_id=session_id,
                cognitive_mode=cognitive_mode,
                user_persona=user_persona,
                interaction_context=interaction_context
            )
            # CRITICAL FIX: Return a dictionary
            return basic_result.__dict__

    # MONKEY-PATCH: Replace the engine's method with our integrated one
    # We are patching the 'translate' method on the specific engine instance
    diffusion_engine.translate = integrated_translate
    
    logger.info("✅ Integration bridge successfully monkey-patched to diffusion engine")
    
    return bridge 