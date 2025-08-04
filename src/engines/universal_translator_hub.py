"""
KIMERA Universal Translator Hub
==============================

The central orchestrator for multidimensional polyglot translation.
Acts as the hub/distributor/gate that coordinates multiple translation engines
including the text diffusion engine.
"""

import asyncio
import json
import logging
import math
import threading
import time
import uuid
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

# Scientific computing
from scipy import linalg as la
from scipy.stats import entropy

# Make this a primary import, not a conditional one
from src.core.context_imposer import get_context_imposer
from src.utils.gpu_foundation import GPUFoundation

# KIMERA Core Integration
try:
    from src.engines.kimera_text_diffusion_engine import (
        KimeraTextDiffusionEngine, 
        DiffusionRequest, 
        DiffusionResult,
        DiffusionMode,
        DiffusionConfig,
        create_kimera_text_diffusion_engine
    )
    from src.engines.cognitive_field_dynamics import CognitiveFieldDynamics
    from src.core.embedding_utils import encode_text
    from src.monitoring.kimera_prometheus_metrics import KimeraPrometheusMetrics

from ..config.settings import get_settings
from ..utils.config import get_api_settings

    # from src.engines.gyroscopic_universal_translator import GyroscopicUniversalTranslator # Assuming this might not exist yet
    KIMERA_CORE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"KIMERA core not available: {e}")
    KIMERA_CORE_AVAILABLE = False

logger = logging.getLogger(__name__)

# Enums and Dataclasses
class TranslationModality(Enum):
    NATURAL_LANGUAGE = "natural_language"
    MATHEMATICAL = "mathematical"
    ECHOFORM = "echoform"
    COGNITIVE_ENHANCED = "cognitive_enhanced"
    PERSONA_AWARE = "persona_aware"
    NEURODIVERGENT = "neurodivergent"

class TranslationEngine(Enum):
    TEXT_DIFFUSION = "text_diffusion"
    DIRECT_SEMANTIC = "direct_semantic"

class TranslationPriority(Enum):
    NORMAL = 2

@dataclass
class UniversalTranslationRequest:
    source_content: Any
    source_modality: TranslationModality
    target_modality: TranslationModality
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: TranslationPriority = TranslationPriority.NORMAL
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    diffusion_mode: Optional[DiffusionMode] = None
    conversation_context: Optional[List[Dict[str, str]]] = None
    session_id: Optional[str] = None

class TranslateApiRequest(BaseModel):
    engine: Optional[str] = None
    source_content: str
    source_modality: str = "natural_language"
    target_modality: str = "natural_language"
    metadata: Dict[str, Any] = {}

@dataclass
class UniversalTranslationResult:
    request_id: str
    translated_content: Any
    source_modality: TranslationModality
    target_modality: TranslationModality
    engine_used: TranslationEngine
    confidence: float
    semantic_coherence: float
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EngineCapability:
    supported_modalities: Set[TranslationModality]
    quality_score: float = 0.9
    speed_score: float = 0.7
    reliability_score: float = 0.8
    resource_cost: float = 0.5

class SemanticRouter:
    def __init__(self, logger_instance):
        self.settings = get_api_settings()
        logger.debug(f"   Environment: {self.settings.environment}")
        self.logger = logger_instance
        self.engine_capabilities = {
            TranslationEngine.TEXT_DIFFUSION: EngineCapability(
                supported_modalities={
                    TranslationModality.NATURAL_LANGUAGE,
                    TranslationModality.COGNITIVE_ENHANCED,
                    TranslationModality.PERSONA_AWARE,
                    TranslationModality.NEURODIVERGENT
                }
            ),
            TranslationEngine.DIRECT_SEMANTIC: EngineCapability(
                supported_modalities=set(TranslationModality)
            ),
        }

    def route_request(self, request: UniversalTranslationRequest) -> TranslationEngine:
        # Enhanced routing: prioritize diffusion engine for advanced modalities
        diffusion_caps = self.engine_capabilities[TranslationEngine.TEXT_DIFFUSION]
        
        # Check if this is an advanced cognitive request
        is_advanced_request = (
            request.source_modality in [
                TranslationModality.COGNITIVE_ENHANCED,
                TranslationModality.PERSONA_AWARE,
                TranslationModality.NEURODIVERGENT
            ] or
            request.target_modality in [
                TranslationModality.COGNITIVE_ENHANCED,
                TranslationModality.PERSONA_AWARE,
                TranslationModality.NEURODIVERGENT
            ] or
            request.diffusion_mode is not None
        )
        
        if is_advanced_request or (
            request.source_modality in diffusion_caps.supported_modalities and 
            request.target_modality in diffusion_caps.supported_modalities
        ):
            self.logger.info(f"🎯 Routed to {TranslationEngine.TEXT_DIFFUSION.name} for advanced processing")
            return TranslationEngine.TEXT_DIFFUSION
        
        self.logger.info(f"🎯 Routed to fallback {TranslationEngine.DIRECT_SEMANTIC.name}")
        return TranslationEngine.DIRECT_SEMANTIC

class ConversationMemoryManager:
    """Thread-safe conversation memory manager with LRU eviction and size limits."""
    
    def __init__(self, max_sessions: int = 1000, max_history_per_session: int = 10, 
                 session_timeout_hours: int = 24):
        self.settings = get_api_settings()
        logger.debug(f"   Environment: {self.settings.environment}")
        self.max_sessions = max_sessions
        self.max_history_per_session = max_history_per_session
        self.session_timeout = timedelta(hours=session_timeout_hours)
        
        # Use OrderedDict for LRU behavior
        self._sessions: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()
        
        # Track memory usage
        self._total_interactions = 0
        self._evicted_sessions = 0
        
    def get_session(self, session_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a session (thread-safe)."""
        with self._lock:
            if session_id in self._sessions:
                # Move to end (most recently used)
                self._sessions.move_to_end(session_id)
                session = self._sessions[session_id]
                
                # Check if session has expired
                if datetime.now() - session['last_access'] > self.session_timeout:
                    del self._sessions[session_id]
                    self._evicted_sessions += 1
                    return []
                
                session['last_access'] = datetime.now()
                return session['history']
            return []
    
    def add_interaction(self, session_id: str, user_input: str, assistant_response: str):
        """Add an interaction to session history (thread-safe)."""
        with self._lock:
            # Ensure we don't exceed max sessions
            if session_id not in self._sessions and len(self._sessions) >= self.max_sessions:
                # Evict least recently used session
                oldest_session = next(iter(self._sessions))
                del self._sessions[oldest_session]
                self._evicted_sessions += 1
            
            # Initialize session if needed
            if session_id not in self._sessions:
                self._sessions[session_id] = {
                    'history': [],
                    'created': datetime.now(),
                    'last_access': datetime.now()
                }
            
            # Add interaction
            session = self._sessions[session_id]
            session['history'].append({
                'user': user_input,
                'assistant': assistant_response,
                'timestamp': time.time()
            })
            
            # Trim history to max size
            if len(session['history']) > self.max_history_per_session:
                session['history'] = session['history'][-self.max_history_per_session:]
            
            session['last_access'] = datetime.now()
            self._sessions.move_to_end(session_id)
            self._total_interactions += 1
    
    def update_context(self, session_id: str, context: List[Dict[str, str]]):
        """Update session with new context (thread-safe)."""
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = {
                    'history': [],
                    'created': datetime.now(),
                    'last_access': datetime.now()
                }
            
            session = self._sessions[session_id]
            for item in context:
                session['history'].append(item)
            
            # Trim to max size
            if len(session['history']) > self.max_history_per_session:
                session['history'] = session['history'][-self.max_history_per_session:]
            
            session['last_access'] = datetime.now()
            self._sessions.move_to_end(session_id)
    
    def clear_session(self, session_id: str):
        """Clear a specific session (thread-safe)."""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions (thread-safe)."""
        with self._lock:
            current_time = datetime.now()
            expired_sessions = []
            
            for session_id, session_data in self._sessions.items():
                if current_time - session_data['last_access'] > self.session_timeout:
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                del self._sessions[session_id]
                self._evicted_sessions += 1
            
            return len(expired_sessions)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory manager statistics (thread-safe)."""
        with self._lock:
            total_memory_items = sum(len(s['history']) for s in self._sessions.values())
            return {
                'active_sessions': len(self._sessions),
                'total_interactions': self._total_interactions,
                'evicted_sessions': self._evicted_sessions,
                'total_memory_items': total_memory_items,
                'max_sessions': self.max_sessions,
                'max_history_per_session': self.max_history_per_session
            }


class UniversalTranslatorHub:
    def __init__(self, config: Dict[str, Any], gpu_foundation: Optional[GPUFoundation] = None):
        self.settings = get_api_settings()
        logger.debug(f"   Environment: {self.settings.environment}")
        self.config = config
        self.router = SemanticRouter(logger)
        self.context_imposer = get_context_imposer()
        self.gpu_foundation = gpu_foundation
        self.engines = {}
        
        # Initialize thread-safe conversation memory manager
        self.memory_manager = ConversationMemoryManager(
            max_sessions=config.get('max_sessions', 1000),
            max_history_per_session=config.get('max_conversation_history', 10),
            session_timeout_hours=config.get('session_timeout_hours', 24)
        )
        
        # Start periodic cleanup task
        self._cleanup_task = None
        self._start_cleanup_task()
        
        self._initialize_engines()
        
        logger.info(f"🌍 KIMERA Universal Translator Hub initialized with engines: {list(self.engines.keys())}")
        logger.info(f"   Memory limits: {self.memory_manager.max_sessions} sessions, "
                   f"{self.memory_manager.max_history_per_session} interactions per session")

    def _initialize_engines(self):
        # Initialize Enhanced Text Diffusion Engine
        if KIMERA_CORE_AVAILABLE:
            # Enhanced configuration for better conversation capabilities
            diffusion_config = {
                'num_steps': 15,  # Optimized for conversation speed
                'noise_schedule': 'cosine',
                'embedding_dim': 1024,
                'max_length': 512,
                'temperature': 0.8,
                'top_k': 50,
                'top_p': 0.9
            }
            
            engine = create_kimera_text_diffusion_engine(
                diffusion_config, self.gpu_foundation
            )
            if engine:
                self.engines[TranslationEngine.TEXT_DIFFUSION] = engine
                logger.info("✅ Enhanced Text Diffusion Engine initialized")
                logger.info("   Features: Cognitive Enhancement, Persona Awareness, Neurodivergent Modeling")
            else:
                 logger.error("❌ Enhanced Text Diffusion Engine failed to initialize.")
        
        # Enhanced Fallback Engine
        self.engines[TranslationEngine.DIRECT_SEMANTIC] = DirectSemanticEngine()
        logger.info("✅ Direct Semantic Engine initialized")
    
    def _start_cleanup_task(self):
        """Start periodic cleanup of expired sessions."""
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(3600)  # Run every hour
                    expired_count = self.memory_manager.cleanup_expired_sessions()
                    if expired_count > 0:
                        logger.info(f"🧹 Cleaned up {expired_count} expired sessions")
                        
                    # Log memory stats periodically
                    stats = self.memory_manager.get_stats()
                    logger.info(f"📊 Memory stats: {stats['active_sessions']} active sessions, "
                               f"{stats['total_memory_items']} total items")
                except Exception as e:
                    logger.error(f"Error in cleanup task: {e}")
        
        # Create task but don't await it
        self._cleanup_task = asyncio.create_task(cleanup_loop())

    async def translate(self, request: UniversalTranslationRequest) -> UniversalTranslationResult:
        start_time = time.time()
        try:
            # Update conversation context if provided
            if request.conversation_context:
                self.memory_manager.update_context(request.request_id, request.conversation_context)
            
            selected_engine_type = self.router.route_request(request)
            engine = self.engines.get(selected_engine_type)

            if not engine:
                raise RuntimeError(f"Selected engine {selected_engine_type.name} not available")

            if selected_engine_type == TranslationEngine.TEXT_DIFFUSION:
                result_dict = await self._translate_with_enhanced_diffusion(engine, request)
            else:
                result_dict = await self._translate_with_direct(engine, request)

            processing_time = time.time() - start_time
            return UniversalTranslationResult(
                request_id=request.request_id,
                translated_content=result_dict.get('generated_content', 'Translation failed'),
                source_modality=request.source_modality,
                target_modality=request.target_modality,
                engine_used=selected_engine_type,
                confidence=result_dict.get('confidence', 0.0),
                semantic_coherence=result_dict.get('semantic_coherence', 0.0),
                processing_time=processing_time,
                metadata=result_dict.get('metadata', {})
            )
        except Exception as e:
            logger.error(f"❌ Universal translation failed: {e}", exc_info=True)
            return UniversalTranslationResult(
                request_id=request.request_id,
                translated_content=f"Translation failed: {e}",
                source_modality=request.source_modality,
                target_modality=request.target_modality,
                engine_used=TranslationEngine.DIRECT_SEMANTIC,
                confidence=0.0,
                semantic_coherence=0.0,
                processing_time=time.time() - start_time,
            )

    async def _translate_with_enhanced_diffusion(
        self, 
        engine: KimeraTextDiffusionEngine, 
        request: UniversalTranslationRequest
    ) -> Dict[str, Any]:
        """Enhanced diffusion translation with cognitive awareness."""
        
        # Determine diffusion mode based on target modality or explicit request
        diffusion_mode = request.diffusion_mode
        if not diffusion_mode:
            if request.target_modality == TranslationModality.COGNITIVE_ENHANCED:
                diffusion_mode = DiffusionMode.COGNITIVE_ENHANCED
            elif request.target_modality == TranslationModality.PERSONA_AWARE:
                diffusion_mode = DiffusionMode.PERSONA_AWARE
            elif request.target_modality == TranslationModality.NEURODIVERGENT:
                diffusion_mode = DiffusionMode.NEURODIVERGENT
            else:
                diffusion_mode = DiffusionMode.STANDARD
        
        # Enhanced persona prompt with conversation context
        persona_prompt = self._create_enhanced_persona_prompt(request)
        
        # Create enhanced diffusion request
        diffusion_request = DiffusionRequest(
            source_content=request.source_content,
            source_modality=request.source_modality.value,
            target_modality=request.target_modality.value,
            mode=diffusion_mode,
            metadata={
                "persona_prompt": persona_prompt,
                "conversation_context": request.conversation_context,
                "cognitive_mode": diffusion_mode.value,
                "session_id": request.request_id
            }
        )
        
        result = await engine.generate(diffusion_request)
        
        # Store successful interaction in conversation memory
        if result.confidence > 0.5:
            self.memory_manager.add_interaction(request.request_id, request.source_content, result.generated_content)
        
        return result.__dict__

    def _create_enhanced_persona_prompt(self, request: UniversalTranslationRequest) -> str:
        """Create enhanced persona prompt with conversation context."""
        base_persona = self.context_imposer.get_persona_prompt(request.metadata.get("context", {}))
        
        # Add conversation context if available
        conversation_context = ""
        session_history = self.memory_manager.get_session(request.request_id)
        if session_history:
            recent_interactions = session_history[-3:]  # Last 3 interactions
            if recent_interactions:
                conversation_context = "\n\nRecent conversation context:\n"
                for i, interaction in enumerate(recent_interactions):
                    conversation_context += f"User: {interaction['user']}\nKIMERA: {interaction['assistant']}\n"
        
        # Add cognitive mode specific instructions
        mode_instructions = ""
        if request.target_modality == TranslationModality.COGNITIVE_ENHANCED:
            mode_instructions = "\n\nUse enhanced cognitive processing with deep semantic analysis and multi-layered reasoning."
        elif request.target_modality == TranslationModality.PERSONA_AWARE:
            mode_instructions = "\n\nAdapt your communication style to match the user's preferences and maintain consistent persona throughout the conversation."
        elif request.target_modality == TranslationModality.NEURODIVERGENT:
            mode_instructions = "\n\nUse neurodivergent-friendly communication patterns: clear structure, detailed explanations, and acknowledge different processing styles."
        
        return f"{base_persona}{conversation_context}{mode_instructions}"

    def get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a session."""
        return self.memory_manager.get_session(session_id)

    def clear_conversation_history(self, session_id: str):
        """Clear conversation history for a session."""
        self.memory_manager.clear_session(session_id)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        return self.memory_manager.get_stats()

    def add_interaction(self, session_id: str, user_input: str, assistant_response: str):
        """Proxy method to add an interaction to the conversation memory."""
        self.memory_manager.add_interaction(session_id, user_input, assistant_response)

    async def _translate_with_direct(self, engine, request: UniversalTranslationRequest) -> Dict[str, Any]:
        # A simple pass-through for now
        # In a real scenario, this would involve a different model or API
        return await engine.translate(request.source_content, request.source_modality.value, request.target_modality.value)

    def get_available_engines(self) -> List[str]:
        return [engine.value for engine in self.engines.keys()]

class DirectSemanticEngine:
    """
    A direct semantic translation engine.
    This is a placeholder for a non-diffusion-based translation model.
    It performs a simple echo translation for demonstration purposes.
    """
    async def translate(self, content: Any, source_modality: str, target_modality: str) -> Dict[str, Any]:
        logger.info(f"DirectSemanticEngine translating from {source_modality} to {target_modality}")
        return {
            "translated_content": f"Direct translation of '{content}'",
            "confidence": 0.95,
            "semantic_coherence": 0.98,
            "metadata": {}
        }

def create_universal_translator_hub(
    config: Optional[Dict[str, Any]] = None,
    gpu_foundation: Optional[GPUFoundation] = None
) -> UniversalTranslatorHub:
    if config is None:
        config = {}
    return UniversalTranslatorHub(config, gpu_foundation=gpu_foundation)

# --- FastAPI Router ---
router = APIRouter(prefix="/translator", tags=["Universal Translator"])

@router.get("/status")
async def get_translator_status(request: Request):
    """Returns the status and available engines of the Universal Translator Hub."""
    hub: UniversalTranslatorHub = request.app.state.translator_hub
    if not hub:
        raise HTTPException(status_code=503, detail="Translator Hub not available")
    return {"available_engines": hub.get_available_engines()}

@router.post("/translate", response_model=UniversalTranslationResult)
async def translate(request: Request, api_request: TranslateApiRequest):
    """Translate content using the Universal Translator Hub."""
    try:
        translator_hub = request.app.state.translator_hub
        if not translator_hub:
            raise HTTPException(status_code=503, detail="Universal Translator Hub is not available.")
        
        # This endpoint is now more for general translation, chat has its own.
        translation_request = UniversalTranslationRequest(
            source_content=api_request.source_content,
            source_modality=TranslationModality(api_request.source_modality),
            target_modality=TranslationModality(api_request.target_modality),
            metadata=api_request.metadata
        )

        result = await translator_hub.translate(translation_request)
        return result
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid modality specified.")
    except Exception as e:
        logger.error(f"Error in translation endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during translation.") 