"""
Selective Feedback Interpreter
=============================

Implements selective feedback loop ONLY for the interpreter profiler,
keeping behavior profiles completely isolated to prevent bias contamination.

The interpreter profiler learns "HOW TO ANALYZE" while behavior profiles
remain static to maintain ethical consistency and prevent manipulation.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
from collections import deque
import numpy as np
import json

from .anthropomorphic_profiler import AnthropomorphicProfiler, InteractionAnalysis
from .context_field_selector import ContextFieldSelector

logger = logging.getLogger(__name__)


class AnalysisEffectiveness(Enum):
    """Measures of analysis effectiveness for feedback learning"""
    ACCURACY = "accuracy"           # How accurate was the analysis
    EFFICIENCY = "efficiency"       # How fast was the analysis
    COMPLETENESS = "completeness"   # How thorough was the analysis
    RELEVANCE = "relevance"         # How relevant to context
    SOPHISTICATION = "sophistication"  # How nuanced was the analysis


@dataclass
class AnalysisFeedback:
    """Feedback data for improving analysis capabilities ONLY"""
    
    timestamp: datetime
    context_type: str
    analysis_quality: float
    successful_patterns: List[str]
    missed_patterns: List[str]
    optimal_settings: Dict[str, Any]


@dataclass
class InterpreterLearningState:
    """Current learning state of the interpreter profiler"""
    
    # Learning progress
    learning_iterations: int = 0
    total_analyses: int = 0
    successful_analyses: int = 0
    
    # Context-specific learning
    context_expertise: Dict[str, float] = field(default_factory=dict)  # Domain expertise levels
    pattern_effectiveness: Dict[str, List[float]] = field(default_factory=dict)  # Pattern success rates
    optimal_configurations: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # Best settings per context
    
    # Adaptation metrics
    accuracy_trend: deque = field(default_factory=lambda: deque(maxlen=100))
    efficiency_trend: deque = field(default_factory=lambda: deque(maxlen=100))
    learning_rate: float = 0.1
    
    # Safety constraints
    max_pattern_modifications: int = 5   # Limit pattern changes per iteration
    stability_threshold: float = 0.95    # Minimum stability before changes
    rollback_threshold: float = 0.8      # Performance threshold for rollback


class SelectiveFeedbackInterpreter:
    """
    Selective feedback system - learns ONLY analysis patterns,
    keeps behavior profiles completely isolated.
    """
    
    def __init__(self, base_profiler: AnthropomorphicProfiler):
        self.base_profiler = base_profiler
        self.learned_analysis_patterns = {}
        self.context_expertise = {}
        
        logger.info("🧠🔄 Selective Feedback Interpreter - Analysis learning enabled, behaviors protected")
    
    def analyze_with_learning(self, message: str, context: Dict[str, Any]) -> InteractionAnalysis:
        """Perform analysis with learning capability"""
        
        # Apply learned analysis improvements
        enhanced_profiler = self._enhance_analysis_only(context)
        
        # Perform analysis (behavior remains unchanged)
        result = enhanced_profiler.analyze_interaction(message)
        
        return result
    
    def _enhance_analysis_only(self, context: Dict[str, Any]) -> AnthropomorphicProfiler:
        """Enhance ONLY analysis patterns, never behavior"""
        
        enhanced_profiler = AnthropomorphicProfiler(self.base_profiler.baseline_profile)
        
        # Apply learned pattern recognition improvements (ANALYSIS ONLY)
        context_type = context.get('type', 'general')
        if context_type in self.learned_analysis_patterns:
            # Enhance detection patterns only - behavior profile KNOWS but is NOT INFLUENCED
            learned_patterns = self.learned_analysis_patterns[context_type]
            enhanced_profiler.technical_patterns['high'].extend(learned_patterns.get('technical', []))
        
        # CRITICAL: Behavior profile receives analysis information but maintains static response
        # It KNOWS the communication style needed but is NOT INFLUENCED by feedback
        enhanced_profiler._analysis_context = context  # Informs behavior without changing it
        
        return enhanced_profiler


def create_selective_feedback_interpreter(domain_focus: str = 'balanced') -> SelectiveFeedbackInterpreter:
    """Create a selective feedback interpreter optimized for specific domain"""
    
    from .anthropomorphic_profiler import create_default_profiler
    from .context_field_selector import create_domain_selector
    
    base_profiler = create_default_profiler()
    context_selector = create_domain_selector(domain_focus) if domain_focus != 'balanced' else None
    
    return SelectiveFeedbackInterpreter(base_profiler) 