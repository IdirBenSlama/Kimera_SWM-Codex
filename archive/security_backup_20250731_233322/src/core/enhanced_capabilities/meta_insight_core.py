"""
Meta Insight Core - Higher-Order Insight Generation
=================================================

Implements meta-cognitive processing with:
- Higher-order insight generation
- Meta-cognition and self-reflection
- Pattern recognition across cognitive domains
- Insight synthesis and breakthrough detection
- Meta-learning and adaptation

This core generates insights about insights and provides meta-cognitive
awareness of the cognitive processing itself.
"""

import logging
import time
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum

import torch
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)


class InsightType(Enum):
    """Types of insights that can be generated"""
    PATTERN_INSIGHT = "pattern_insight"       # Pattern recognition insights
    CAUSAL_INSIGHT = "causal_insight"         # Causal relationship insights
    STRUCTURAL_INSIGHT = "structural_insight" # Structural understanding insights
    META_INSIGHT = "meta_insight"             # Insights about insights
    BREAKTHROUGH_INSIGHT = "breakthrough_insight"  # Revolutionary insights
    SYNTHESIS_INSIGHT = "synthesis_insight"   # Synthesis across domains
    PREDICTIVE_INSIGHT = "predictive_insight" # Predictive insights


class MetaCognitionLevel(Enum):
    """Levels of meta-cognitive processing"""
    OBJECT_LEVEL = "object_level"             # Direct cognitive processing
    META_LEVEL_1 = "meta_level_1"            # Thinking about thinking
    META_LEVEL_2 = "meta_level_2"            # Thinking about thinking about thinking
    META_LEVEL_3 = "meta_level_3"            # Higher-order meta-cognition
    RECURSIVE_META = "recursive_meta"         # Recursive meta-cognitive loops


class InsightQuality(Enum):
    """Quality levels of generated insights"""
    TRIVIAL = "trivial"                       # Trivial or obvious insights
    BASIC = "basic"                           # Basic but useful insights
    SIGNIFICANT = "significant"               # Significant insights
    PROFOUND = "profound"                     # Profound insights
    REVOLUTIONARY = "revolutionary"           # Revolutionary breakthroughs


@dataclass
class PatternInsight:
    """Pattern recognition insight"""
    pattern_id: str
    pattern_type: str
    pattern_description: str
    pattern_strength: float
    supporting_evidence: List[Any]
    confidence: float
    generalizability: float
    novelty_score: float


@dataclass
class MetaInsightResult:
    """Result from meta-insight processing"""
    insight_id: str
    insight_type: InsightType
    insight_quality: InsightQuality
    meta_cognition_level: MetaCognitionLevel
    
    # Core insight content
    insight_content: str
    insight_description: str
    supporting_patterns: List[PatternInsight]
    
    # Meta-cognitive analysis
    meta_cognitive_assessment: Dict[str, Any]
    self_reflection_analysis: Dict[str, Any]
    cognitive_monitoring: Dict[str, Any]
    
    # Insight metrics
    insight_strength: float              # Strength of the insight
    novelty_score: float                # How novel/original the insight is
    significance_score: float           # How significant the insight is
    confidence_score: float             # Confidence in the insight
    generalizability: float             # How generalizable the insight is
    
    # Processing information
    processing_time: float
    computational_cost: float
    breakthrough_potential: float       # Potential for breakthrough
    
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    success: bool = True
    error_log: List[str] = field(default_factory=list)


class HigherOrderProcessor:
    """Higher-order cognitive processing system"""
    
    def __init__(self, max_meta_depth: int = 3, recursion_limit: int = 5):
        self.max_meta_depth = max_meta_depth
        self.recursion_limit = recursion_limit
        
        # Processing state
        self.current_meta_level = MetaCognitionLevel.OBJECT_LEVEL
        self.meta_processing_stack = []
        self.recursion_count = 0
        
        # Pattern tracking
        self.discovered_patterns = []
        self.meta_patterns = []
        
        logger.debug("Higher-order processor initialized")
    
    async def process_higher_order(self, 
                                 cognitive_input: torch.Tensor,
                                 target_meta_level: MetaCognitionLevel,
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """Process at higher-order meta-cognitive levels"""
        try:
            self.current_meta_level = target_meta_level
            
            if target_meta_level == MetaCognitionLevel.OBJECT_LEVEL:
                result = await self._process_object_level(cognitive_input, context)
            elif target_meta_level == MetaCognitionLevel.META_LEVEL_1:
                result = await self._process_meta_level_1(cognitive_input, context)
            elif target_meta_level == MetaCognitionLevel.META_LEVEL_2:
                result = await self._process_meta_level_2(cognitive_input, context)
            elif target_meta_level == MetaCognitionLevel.META_LEVEL_3:
                result = await self._process_meta_level_3(cognitive_input, context)
            elif target_meta_level == MetaCognitionLevel.RECURSIVE_META:
                result = await self._process_recursive_meta(cognitive_input, context)
            else:
                raise ValueError(f"Unknown meta-cognition level: {target_meta_level}")
            
            return result
            
        except Exception as e:
            logger.error(f"Higher-order processing failed: {e}")
            return {
                'error': str(e),
                'meta_level': target_meta_level.value,
                'processing_quality': 0.0
            }
    
    async def _process_object_level(self, input_tensor: torch.Tensor, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process at object level (direct cognitive processing)"""
        # Direct processing of cognitive input
        processed_state = F.tanh(input_tensor * 1.1)  # Simple nonlinear processing
        
        # Extract basic features
        activation_level = torch.mean(torch.abs(processed_state)).item()
        coherence = 1.0 - torch.std(processed_state).item() / (torch.mean(torch.abs(processed_state)).item() + 1e-8)
        complexity = torch.sum(processed_state != 0).item() / len(processed_state)
        
        return {
            'processing_level': 'object_level',
            'processed_state': processed_state,
            'activation_level': activation_level,
            'coherence': max(0.0, min(1.0, coherence)),
            'complexity': complexity,
            'processing_quality': (activation_level + coherence + complexity) / 3.0
        }
    
    async def _process_meta_level_1(self, input_tensor: torch.Tensor, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process at meta-level 1 (thinking about thinking)"""
        # First process at object level
        object_result = await self._process_object_level(input_tensor, context)
        
        # Then reflect on the object-level processing
        meta_reflection = self._reflect_on_processing(object_result, context)
        
        # Generate meta-level insights
        meta_insights = self._generate_meta_level_insights(object_result, meta_reflection)
        
        return {
            'processing_level': 'meta_level_1',
            'object_level_result': object_result,
            'meta_reflection': meta_reflection,
            'meta_insights': meta_insights,
            'processing_quality': (
                object_result.get('processing_quality', 0.0) +
                meta_reflection.get('reflection_quality', 0.0)
            ) / 2.0
        }
    
    async def _process_meta_level_2(self, input_tensor: torch.Tensor, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process at meta-level 2 (thinking about thinking about thinking)"""
        # Process at meta-level 1
        meta_1_result = await self._process_meta_level_1(input_tensor, context)
        
        # Reflect on meta-level 1 processing
        meta_meta_reflection = self._reflect_on_meta_processing(meta_1_result, context)
        
        # Generate higher-order meta-insights
        higher_order_insights = self._generate_higher_order_insights(meta_1_result, meta_meta_reflection)
        
        return {
            'processing_level': 'meta_level_2',
            'meta_level_1_result': meta_1_result,
            'meta_meta_reflection': meta_meta_reflection,
            'higher_order_insights': higher_order_insights,
            'processing_quality': min(1.0, (
                meta_1_result.get('processing_quality', 0.0) +
                meta_meta_reflection.get('meta_reflection_quality', 0.0) +
                higher_order_insights.get('insight_quality', 0.0)
            ) / 3.0)
        }
    
    async def _process_meta_level_3(self, input_tensor: torch.Tensor, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process at meta-level 3 (highest explicit meta-cognition)"""
        # Process at meta-level 2
        meta_2_result = await self._process_meta_level_2(input_tensor, context)
        
        # Ultra-meta reflection
        ultra_meta_reflection = self._ultra_meta_reflection(meta_2_result, context)
        
        # Generate transcendent insights
        transcendent_insights = self._generate_transcendent_insights(meta_2_result, ultra_meta_reflection)
        
        return {
            'processing_level': 'meta_level_3',
            'meta_level_2_result': meta_2_result,
            'ultra_meta_reflection': ultra_meta_reflection,
            'transcendent_insights': transcendent_insights,
            'processing_quality': min(1.0, (
                meta_2_result.get('processing_quality', 0.0) +
                ultra_meta_reflection.get('ultra_reflection_quality', 0.0) +
                transcendent_insights.get('transcendence_quality', 0.0)
            ) / 3.0)
        }
    
    async def _process_recursive_meta(self, input_tensor: torch.Tensor, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process with recursive meta-cognition"""
        if self.recursion_count >= self.recursion_limit:
            return {'error': 'Recursion limit reached', 'processing_quality': 0.0}
        
        self.recursion_count += 1
        
        try:
            # Start with meta-level 3
            current_result = await self._process_meta_level_3(input_tensor, context)
            
            # Recursively apply meta-cognition to the result
            recursive_input = self._extract_recursive_input(current_result)
            recursive_result = await self._process_recursive_meta(recursive_input, context)
            
            # Integrate recursive insights
            integrated_result = self._integrate_recursive_insights(current_result, recursive_result)
            
            return integrated_result
            
        finally:
            self.recursion_count -= 1
    
    def _reflect_on_processing(self, processing_result: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Reflect on cognitive processing results"""
        activation = processing_result.get('activation_level', 0.0)
        coherence = processing_result.get('coherence', 0.0)
        complexity = processing_result.get('complexity', 0.0)
        
        # Meta-cognitive assessment
        reflection = {
            'processing_effectiveness': (activation + coherence) / 2.0,
            'cognitive_load_assessment': complexity,
            'processing_coherence_evaluation': coherence,
            'attention_allocation_analysis': activation,
            'cognitive_resource_utilization': (activation * coherence + complexity) / 2.0,
            'reflection_quality': (activation + coherence + (1.0 - complexity)) / 3.0
        }
        
        # Add contextual reflections
        if context.get('task_difficulty', 0.5) > 0.7:
            reflection['difficulty_adaptation'] = min(1.0, activation * 1.2)
        
        return reflection
    
    def _generate_meta_level_insights(self, object_result: Dict[str, Any], reflection: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights from meta-level 1 processing"""
        insights = {
            'cognitive_efficiency_insight': 
                f"Processing efficiency: {reflection.get('processing_effectiveness', 0.0):.2f}",
            'attention_insight': 
                f"Attention allocation shows {object_result.get('activation_level', 0.0):.2f} focus level",
            'coherence_insight': 
                f"Cognitive coherence at {object_result.get('coherence', 0.0):.2f} indicates {'high' if object_result.get('coherence', 0.0) > 0.6 else 'moderate'} integration",
            'insight_quality': (
                reflection.get('processing_effectiveness', 0.0) +
                reflection.get('reflection_quality', 0.0)
            ) / 2.0
        }
        
        return insights
    
    def _reflect_on_meta_processing(self, meta_1_result: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Reflect on meta-level 1 processing"""
        object_quality = meta_1_result.get('object_level_result', {}).get('processing_quality', 0.0)
        meta_quality = meta_1_result.get('processing_quality', 0.0)
        
        meta_reflection = {
            'meta_processing_effectiveness': meta_quality,
            'object_meta_integration': (object_quality + meta_quality) / 2.0,
            'meta_cognitive_coherence': meta_1_result.get('meta_reflection', {}).get('reflection_quality', 0.0),
            'recursive_thinking_quality': min(1.0, meta_quality * 1.1),
            'meta_reflection_quality': (meta_quality + object_quality) / 2.0
        }
        
        return meta_reflection
    
    def _generate_higher_order_insights(self, meta_1_result: Dict[str, Any], meta_meta_reflection: Dict[str, Any]) -> Dict[str, Any]:
        """Generate higher-order insights from meta-level 2"""
        insights = {
            'meta_cognitive_pattern_insight': 
                f"Meta-cognitive processing shows {meta_meta_reflection.get('meta_processing_effectiveness', 0.0):.2f} effectiveness",
            'recursive_thinking_insight': 
                f"Recursive thinking quality: {meta_meta_reflection.get('recursive_thinking_quality', 0.0):.2f}",
            'integration_insight': 
                f"Object-meta integration: {meta_meta_reflection.get('object_meta_integration', 0.0):.2f}",
            'insight_quality': meta_meta_reflection.get('meta_reflection_quality', 0.0)
        }
        
        return insights
    
    def _ultra_meta_reflection(self, meta_2_result: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform ultra-meta reflection (meta-level 3)"""
        processing_quality = meta_2_result.get('processing_quality', 0.0)
        
        ultra_reflection = {
            'transcendent_processing_assessment': processing_quality,
            'meta_meta_cognitive_coherence': min(1.0, processing_quality * 1.15),
            'consciousness_of_consciousness': processing_quality ** 2,  # Squared for consciousness depth
            'recursive_depth_analysis': min(1.0, len(str(meta_2_result)) / 1000.0),  # Complexity proxy
            'ultra_reflection_quality': processing_quality
        }
        
        return ultra_reflection
    
    def _generate_transcendent_insights(self, meta_2_result: Dict[str, Any], ultra_reflection: Dict[str, Any]) -> Dict[str, Any]:
        """Generate transcendent insights from meta-level 3"""
        consciousness_level = ultra_reflection.get('consciousness_of_consciousness', 0.0)
        
        insights = {
            'consciousness_insight': 
                f"Consciousness of consciousness level: {consciousness_level:.3f}",
            'transcendent_pattern_insight': 
                f"Transcendent processing quality: {ultra_reflection.get('transcendent_processing_assessment', 0.0):.3f}",
            'recursive_depth_insight': 
                f"Recursive cognitive depth: {ultra_reflection.get('recursive_depth_analysis', 0.0):.3f}",
            'meta_meta_coherence_insight': 
                f"Meta-meta-cognitive coherence: {ultra_reflection.get('meta_meta_cognitive_coherence', 0.0):.3f}",
            'transcendence_quality': consciousness_level
        }
        
        return insights
    
    def _extract_recursive_input(self, current_result: Dict[str, Any]) -> torch.Tensor:
        """Extract input for recursive processing"""
        # Convert processing quality to tensor for recursive input
        quality = current_result.get('processing_quality', 0.0)
        transcendence = current_result.get('transcendent_insights', {}).get('transcendence_quality', 0.0)
        
        # Create recursive input based on processing results
        recursive_input = torch.tensor([quality, transcendence] * 32, dtype=torch.float32)  # 64-dim
        
        return recursive_input
    
    def _integrate_recursive_insights(self, current_result: Dict[str, Any], recursive_result: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate insights from recursive processing"""
        if 'error' in recursive_result:
            return current_result
        
        # Combine insights
        integrated_quality = (
            current_result.get('processing_quality', 0.0) +
            recursive_result.get('processing_quality', 0.0)
        ) / 2.0
        
        return {
            'processing_level': 'recursive_meta',
            'current_result': current_result,
            'recursive_result': recursive_result,
            'integrated_quality': integrated_quality,
            'recursion_depth': self.recursion_count,
            'processing_quality': integrated_quality
        }


class MetaCognitionEngine:
    """Meta-cognition and self-reflection engine"""
    
    def __init__(self, awareness_threshold: float = 0.6):
        self.awareness_threshold = awareness_threshold
        self.metacognitive_state = torch.zeros(256)  # Meta-cognitive state vector
        self.self_model = {}
        self.cognitive_monitoring_data = []
        
        logger.debug("Meta-cognition engine initialized")
    
    async def monitor_cognition(self, 
                              cognitive_state: torch.Tensor,
                              processing_context: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor ongoing cognitive processing"""
        try:
            # Update meta-cognitive state
            self.metacognitive_state = 0.9 * self.metacognitive_state + 0.1 * cognitive_state[:256]
            
            # Analyze cognitive processing
            cognitive_analysis = self._analyze_cognitive_processing(cognitive_state, processing_context)
            
            # Monitor cognitive resources
            resource_monitoring = self._monitor_cognitive_resources(cognitive_state)
            
            # Assess cognitive strategies
            strategy_assessment = self._assess_cognitive_strategies(cognitive_state, processing_context)
            
            # Evaluate meta-cognitive awareness
            awareness_level = self._evaluate_metacognitive_awareness(cognitive_analysis)
            
            monitoring_result = {
                'cognitive_analysis': cognitive_analysis,
                'resource_monitoring': resource_monitoring,
                'strategy_assessment': strategy_assessment,
                'awareness_level': awareness_level,
                'monitoring_quality': (
                    cognitive_analysis.get('analysis_quality', 0.0) +
                    resource_monitoring.get('monitoring_accuracy', 0.0) +
                    strategy_assessment.get('assessment_quality', 0.0) +
                    awareness_level
                ) / 4.0
            }
            
            # Record monitoring data
            self.cognitive_monitoring_data.append({
                'timestamp': time.time(),
                'monitoring_result': monitoring_result
            })
            
            # Keep monitoring history manageable
            if len(self.cognitive_monitoring_data) > 100:
                self.cognitive_monitoring_data = self.cognitive_monitoring_data[-50:]
            
            return monitoring_result
            
        except Exception as e:
            logger.error(f"Cognitive monitoring failed: {e}")
            return {
                'error': str(e),
                'monitoring_quality': 0.0,
                'awareness_level': 0.0
            }
    
    def _analyze_cognitive_processing(self, state: torch.Tensor, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current cognitive processing"""
        # Processing intensity
        processing_intensity = torch.mean(torch.abs(state)).item()
        
        # Processing coherence
        coherence = 1.0 - torch.std(state).item() / (torch.mean(torch.abs(state)).item() + 1e-8)
        
        # Processing complexity
        complexity = torch.sum(state != 0).item() / len(state)
        
        # Processing efficiency
        efficiency = processing_intensity * coherence / (complexity + 1e-8)
        
        return {
            'processing_intensity': processing_intensity,
            'processing_coherence': max(0.0, min(1.0, coherence)),
            'processing_complexity': complexity,
            'processing_efficiency': min(1.0, efficiency),
            'analysis_quality': (processing_intensity + coherence + (1.0 - complexity)) / 3.0
        }
    
    def _monitor_cognitive_resources(self, state: torch.Tensor) -> Dict[str, Any]:
        """Monitor cognitive resource utilization"""
        # Attention resources
        attention_usage = torch.mean(torch.abs(state[:64])).item() if len(state) > 64 else 0.0
        
        # Memory resources
        memory_usage = torch.mean(torch.abs(state[64:128])).item() if len(state) > 128 else 0.0
        
        # Processing resources
        processing_usage = torch.mean(torch.abs(state[128:192])).item() if len(state) > 192 else 0.0
        
        # Executive resources
        executive_usage = torch.mean(torch.abs(state[192:])).item() if len(state) > 192 else 0.0
        
        # Overall resource utilization
        total_utilization = (attention_usage + memory_usage + processing_usage + executive_usage) / 4.0
        
        return {
            'attention_usage': attention_usage,
            'memory_usage': memory_usage,
            'processing_usage': processing_usage,
            'executive_usage': executive_usage,
            'total_utilization': total_utilization,
            'resource_balance': 1.0 - np.std([attention_usage, memory_usage, processing_usage, executive_usage]),
            'monitoring_accuracy': min(1.0, total_utilization + 0.2)
        }
    
    def _assess_cognitive_strategies(self, state: torch.Tensor, context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess effectiveness of current cognitive strategies"""
        # Strategy effectiveness based on state patterns
        strategy_coherence = torch.cosine_similarity(
            state[:len(state)//2].unsqueeze(0),
            state[len(state)//2:].unsqueeze(0),
            dim=1
        ).item() if len(state) > 1 else 0.0
        
        # Adaptive strategy assessment
        task_complexity = context.get('complexity', 0.5)
        strategy_appropriateness = 1.0 - abs(strategy_coherence - task_complexity)
        
        # Strategy persistence
        persistence = min(1.0, torch.mean(torch.abs(state)).item() * 2.0)
        
        return {
            'strategy_coherence': abs(strategy_coherence),
            'strategy_appropriateness': strategy_appropriateness,
            'strategy_persistence': persistence,
            'strategy_effectiveness': (abs(strategy_coherence) + strategy_appropriateness + persistence) / 3.0,
            'assessment_quality': min(1.0, (abs(strategy_coherence) + strategy_appropriateness) / 2.0)
        }
    
    def _evaluate_metacognitive_awareness(self, cognitive_analysis: Dict[str, Any]) -> float:
        """Evaluate level of meta-cognitive awareness"""
        analysis_quality = cognitive_analysis.get('analysis_quality', 0.0)
        processing_coherence = cognitive_analysis.get('processing_coherence', 0.0)
        processing_efficiency = cognitive_analysis.get('processing_efficiency', 0.0)
        
        # Meta-cognitive awareness from processing quality
        awareness = (analysis_quality + processing_coherence + processing_efficiency) / 3.0
        
        # Apply awareness threshold
        if awareness > self.awareness_threshold:
            awareness = awareness * 1.2  # Boost above threshold
        
        return max(0.0, min(1.0, awareness))


class PatternRecognitionSystem:
    """Advanced pattern recognition for insights"""
    
    def __init__(self, pattern_threshold: float = 0.5, max_patterns: int = 50):
        self.pattern_threshold = pattern_threshold
        self.max_patterns = max_patterns
        self.discovered_patterns = []
        self.pattern_templates = self._initialize_pattern_templates()
        
        logger.debug("Pattern recognition system initialized")
    
    def _initialize_pattern_templates(self) -> List[Dict[str, Any]]:
        """Initialize pattern recognition templates"""
        templates = [
            {
                'name': 'periodic_pattern',
                'detector': self._detect_periodic_pattern,
                'significance': 0.7
            },
            {
                'name': 'hierarchical_pattern',
                'detector': self._detect_hierarchical_pattern,
                'significance': 0.8
            },
            {
                'name': 'emergent_pattern',
                'detector': self._detect_emergent_pattern,
                'significance': 0.9
            },
            {
                'name': 'causal_pattern',
                'detector': self._detect_causal_pattern,
                'significance': 0.75
            },
            {
                'name': 'recursive_pattern',
                'detector': self._detect_recursive_pattern,
                'significance': 0.85
            }
        ]
        return templates
    
    async def recognize_patterns(self, 
                               data: torch.Tensor,
                               context: Dict[str, Any]) -> List[PatternInsight]:
        """Recognize patterns in cognitive data"""
        try:
            recognized_patterns = []
            
            for template in self.pattern_templates:
                pattern_result = await template['detector'](data, context)
                
                if pattern_result and pattern_result.get('strength', 0.0) > self.pattern_threshold:
                    pattern_insight = PatternInsight(
                        pattern_id=f"pattern_{uuid.uuid4().hex[:8]}",
                        pattern_type=template['name'],
                        pattern_description=pattern_result.get('description', ''),
                        pattern_strength=pattern_result.get('strength', 0.0),
                        supporting_evidence=pattern_result.get('evidence', []),
                        confidence=pattern_result.get('confidence', 0.0),
                        generalizability=pattern_result.get('generalizability', 0.0),
                        novelty_score=self._calculate_novelty(template['name'])
                    )
                    
                    recognized_patterns.append(pattern_insight)
            
            # Sort by strength and keep top patterns
            recognized_patterns.sort(key=lambda p: p.pattern_strength, reverse=True)
            recognized_patterns = recognized_patterns[:self.max_patterns]
            
            # Update discovered patterns
            self.discovered_patterns.extend(recognized_patterns)
            if len(self.discovered_patterns) > self.max_patterns * 2:
                self.discovered_patterns = self.discovered_patterns[-self.max_patterns:]
            
            return recognized_patterns
            
        except Exception as e:
            logger.error(f"Pattern recognition failed: {e}")
            return []
    
    async def _detect_periodic_pattern(self, data: torch.Tensor, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Detect periodic patterns in data"""
        try:
            # Use FFT to detect periodicities
            try:
                fft_result = torch.fft.fft(data.float())
            except AttributeError:
                # Fallback for older PyTorch versions
                fft_result = torch.rfft(data.float(), 1)
                fft_result = torch.sqrt(fft_result[:, 0]**2 + fft_result[:, 1]**2)
            magnitudes = torch.abs(fft_result)
            
            # Find dominant frequencies
            max_magnitude = torch.max(magnitudes).item()
            max_index = torch.argmax(magnitudes).item()
            
            if max_magnitude > 0.5 * torch.sum(magnitudes).item() / len(magnitudes):
                # Strong periodic component found
                period = len(data) / (max_index + 1) if max_index > 0 else len(data)
                
                return {
                    'strength': min(1.0, max_magnitude / (torch.mean(magnitudes).item() + 1e-8)),
                    'description': f"Periodic pattern with period ~{period:.1f}",
                    'evidence': [{'frequency_index': max_index, 'magnitude': max_magnitude}],
                    'confidence': min(1.0, max_magnitude),
                    'generalizability': 0.7
                }
            
            return None
            
        except Exception:
            return None
    
    async def _detect_hierarchical_pattern(self, data: torch.Tensor, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Detect hierarchical patterns in data"""
        try:
            # Multi-scale analysis for hierarchical patterns
            scales = [1, 2, 4, 8]
            hierarchical_scores = []
            
            for scale in scales:
                if len(data) >= scale * 2:
                    # Downsample data
                    downsampled = data[::scale]
                    
                    # Calculate structure at this scale
                    structure_score = 1.0 - torch.std(downsampled).item() / (torch.mean(torch.abs(downsampled)).item() + 1e-8)
                    hierarchical_scores.append(max(0.0, structure_score))
            
            if hierarchical_scores:
                # Hierarchical pattern strength from multi-scale consistency
                hierarchy_strength = np.mean(hierarchical_scores)
                
                if hierarchy_strength > 0.4:
                    return {
                        'strength': hierarchy_strength,
                        'description': f"Hierarchical pattern across {len(scales)} scales",
                        'evidence': [{'scale': s, 'score': score} for s, score in zip(scales, hierarchical_scores)],
                        'confidence': hierarchy_strength,
                        'generalizability': 0.8
                    }
            
            return None
            
        except Exception:
            return None
    
    async def _detect_emergent_pattern(self, data: torch.Tensor, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Detect emergent patterns in data"""
        try:
            # Look for emergent properties through nonlinear combinations
            data_squared = data ** 2
            data_interactions = data[:len(data)//2] * data[len(data)//2:] if len(data) > 1 else torch.zeros(1)
            
            # Emergent signal from nonlinear combinations
            emergent_signal = torch.mean(data_squared).item() - torch.mean(torch.abs(data)).item() ** 2
            interaction_strength = torch.mean(torch.abs(data_interactions)).item()
            
            emergence_strength = abs(emergent_signal) + interaction_strength
            
            if emergence_strength > 0.3:
                return {
                    'strength': min(1.0, emergence_strength),
                    'description': f"Emergent pattern with interaction strength {interaction_strength:.3f}",
                    'evidence': [{'emergent_signal': emergent_signal, 'interactions': interaction_strength}],
                    'confidence': min(1.0, emergence_strength),
                    'generalizability': 0.6
                }
            
            return None
            
        except Exception:
            return None
    
    async def _detect_causal_pattern(self, data: torch.Tensor, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Detect causal patterns in data"""
        try:
            # Simple causal pattern detection through temporal relationships
            if len(data) < 3:
                return None
            
            # Calculate temporal correlations
            data_lag1 = data[1:]
            data_lag0 = data[:-1]
            
            # Causal relationship strength
            correlation = torch.cosine_similarity(data_lag0.unsqueeze(0), data_lag1.unsqueeze(0), dim=1).item()
            causal_strength = abs(correlation)
            
            if causal_strength > 0.5:
                return {
                    'strength': causal_strength,
                    'description': f"Temporal causal pattern with correlation {correlation:.3f}",
                    'evidence': [{'temporal_correlation': correlation}],
                    'confidence': causal_strength,
                    'generalizability': 0.75
                }
            
            return None
            
        except Exception:
            return None
    
    async def _detect_recursive_pattern(self, data: torch.Tensor, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Detect recursive patterns in data"""
        try:
            # Look for self-similar patterns at different scales
            if len(data) < 4:
                return None
            
            # Self-similarity analysis
            half_length = len(data) // 2
            first_half = data[:half_length]
            second_half = data[half_length:half_length + len(first_half)]
            
            # Calculate self-similarity
            similarity = torch.cosine_similarity(first_half.unsqueeze(0), second_half.unsqueeze(0), dim=1).item()
            recursive_strength = abs(similarity)
            
            if recursive_strength > 0.6:
                return {
                    'strength': recursive_strength,
                    'description': f"Recursive self-similar pattern with similarity {similarity:.3f}",
                    'evidence': [{'self_similarity': similarity}],
                    'confidence': recursive_strength,
                    'generalizability': 0.85
                }
            
            return None
            
        except Exception:
            return None
    
    def _calculate_novelty(self, pattern_type: str) -> float:
        """Calculate novelty score for a pattern type"""
        # Count existing patterns of this type
        existing_count = sum(1 for p in self.discovered_patterns if p.pattern_type == pattern_type)
        
        # Novelty decreases with frequency
        novelty = 1.0 / (1.0 + existing_count * 0.1)
        
        return max(0.1, min(1.0, novelty))


class InsightGenerationEngine:
    """Core insight generation and synthesis engine"""
    
    def __init__(self, breakthrough_threshold: float = 0.8, synthesis_threshold: float = 0.7):
        self.breakthrough_threshold = breakthrough_threshold
        self.synthesis_threshold = synthesis_threshold
        self.generated_insights = []
        self.insight_quality_history = []
        
        logger.debug("Insight generation engine initialized")
    
    async def generate_insight(self,
                             patterns: List[PatternInsight],
                             meta_cognitive_analysis: Dict[str, Any],
                             higher_order_processing: Dict[str, Any],
                             context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights from patterns and meta-cognitive analysis"""
        try:
            # Synthesize patterns into insights
            pattern_synthesis = self._synthesize_patterns(patterns)
            
            # Generate meta-cognitive insights
            meta_insights = self._generate_meta_cognitive_insights(meta_cognitive_analysis)
            
            # Create higher-order insights
            higher_order_insights = self._create_higher_order_insights(higher_order_processing)
            
            # Combine all insights
            combined_insights = self._combine_insights(pattern_synthesis, meta_insights, higher_order_insights)
            
            # Evaluate insight quality
            insight_quality = self._evaluate_insight_quality(combined_insights, patterns)
            
            # Determine insight type and significance
            insight_type = self._determine_insight_type(combined_insights, insight_quality)
            insight_significance = self._assess_insight_significance(insight_quality, patterns)
            
            # Generate insight description
            insight_description = self._generate_insight_description(
                combined_insights, pattern_synthesis, insight_type
            )
            
            insight_result = {
                'insight_content': combined_insights,
                'insight_description': insight_description,
                'insight_type': insight_type,
                'insight_quality': insight_quality,
                'insight_significance': insight_significance,
                'pattern_synthesis': pattern_synthesis,
                'meta_insights': meta_insights,
                'higher_order_insights': higher_order_insights,
                'breakthrough_potential': self._assess_breakthrough_potential(insight_quality, insight_significance)
            }
            
            # Record insight
            self.generated_insights.append(insight_result)
            self.insight_quality_history.append(insight_quality)
            
            # Keep history manageable
            if len(self.generated_insights) > 100:
                self.generated_insights = self.generated_insights[-50:]
            if len(self.insight_quality_history) > 200:
                self.insight_quality_history = self.insight_quality_history[-100:]
            
            return insight_result
            
        except Exception as e:
            logger.error(f"Insight generation failed: {e}")
            return {
                'insight_content': '',
                'insight_description': f'Insight generation failed: {e}',
                'insight_quality': 0.0,
                'breakthrough_potential': 0.0
            }
    
    def _synthesize_patterns(self, patterns: List[PatternInsight]) -> Dict[str, Any]:
        """Synthesize multiple patterns into coherent insights"""
        if not patterns:
            return {'synthesis_quality': 0.0, 'pattern_connections': []}
        
        # Find connections between patterns
        connections = []
        for i, pattern1 in enumerate(patterns):
            for j, pattern2 in enumerate(patterns[i+1:], i+1):
                connection_strength = self._calculate_pattern_connection(pattern1, pattern2)
                if connection_strength > 0.3:
                    connections.append({
                        'pattern1': pattern1.pattern_id,
                        'pattern2': pattern2.pattern_id,
                        'connection_strength': connection_strength,
                        'connection_type': self._determine_connection_type(pattern1, pattern2)
                    })
        
        # Calculate synthesis quality
        total_strength = sum(p.pattern_strength for p in patterns)
        avg_confidence = sum(p.confidence for p in patterns) / len(patterns)
        connection_density = len(connections) / max(len(patterns) * (len(patterns) - 1) / 2, 1)
        
        synthesis_quality = (total_strength / len(patterns) + avg_confidence + connection_density) / 3.0
        
        return {
            'synthesis_quality': min(1.0, synthesis_quality),
            'pattern_connections': connections,
            'dominant_patterns': sorted(patterns, key=lambda p: p.pattern_strength, reverse=True)[:3],
            'synthesis_coherence': avg_confidence
        }
    
    def _generate_meta_cognitive_insights(self, meta_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights from meta-cognitive analysis"""
        processing_quality = meta_analysis.get('monitoring_quality', 0.0)
        awareness_level = meta_analysis.get('awareness_level', 0.0)
        
        insights = {
            'meta_cognitive_effectiveness': processing_quality,
            'self_awareness_insight': f"Self-awareness level: {awareness_level:.2f}",
            'cognitive_monitoring_insight': f"Cognitive monitoring quality: {processing_quality:.2f}",
            'meta_insight_quality': (processing_quality + awareness_level) / 2.0
        }
        
        # Add specific insights based on analysis
        resource_monitoring = meta_analysis.get('resource_monitoring', {})
        if resource_monitoring:
            insights['resource_utilization_insight'] = (
                f"Resource utilization: {resource_monitoring.get('total_utilization', 0.0):.2f}"
            )
        
        return insights
    
    def _create_higher_order_insights(self, higher_order_processing: Dict[str, Any]) -> Dict[str, Any]:
        """Create insights from higher-order processing"""
        processing_quality = higher_order_processing.get('processing_quality', 0.0)
        processing_level = higher_order_processing.get('processing_level', 'unknown')
        
        insights = {
            'higher_order_quality': processing_quality,
            'processing_level_insight': f"Achieved {processing_level} processing",
            'meta_level_effectiveness': processing_quality
        }
        
        # Add level-specific insights
        if processing_level == 'recursive_meta':
            insights['recursive_depth_insight'] = f"Recursive depth: {higher_order_processing.get('recursion_depth', 0)}"
        elif 'transcendent_insights' in higher_order_processing:
            transcendent = higher_order_processing['transcendent_insights']
            insights['transcendence_insight'] = f"Transcendence quality: {transcendent.get('transcendence_quality', 0.0):.3f}"
        
        return insights
    
    def _combine_insights(self, pattern_synthesis: Dict[str, Any], 
                         meta_insights: Dict[str, Any], 
                         higher_order_insights: Dict[str, Any]) -> str:
        """Combine all insights into coherent content"""
        # Extract key insights
        synthesis_quality = pattern_synthesis.get('synthesis_quality', 0.0)
        meta_quality = meta_insights.get('meta_insight_quality', 0.0)
        higher_order_quality = higher_order_insights.get('higher_order_quality', 0.0)
        
        # Create combined insight content
        if synthesis_quality > 0.7 and meta_quality > 0.6 and higher_order_quality > 0.6:
            insight_content = (
                f"High-quality insight synthesis achieved (quality: {synthesis_quality:.2f}). "
                f"Meta-cognitive analysis reveals {meta_quality:.2f} effectiveness. "
                f"Higher-order processing demonstrates {higher_order_quality:.2f} quality. "
                f"Pattern connections show {len(pattern_synthesis.get('pattern_connections', []))} significant relationships."
            )
        elif synthesis_quality > 0.5:
            insight_content = (
                f"Moderate insight synthesis with {synthesis_quality:.2f} quality. "
                f"Meta-cognitive processing at {meta_quality:.2f} level. "
                f"Identified {len(pattern_synthesis.get('pattern_connections', []))} pattern relationships."
            )
        else:
            insight_content = (
                f"Basic insight generation with limited pattern synthesis. "
                f"Processing quality: {max(synthesis_quality, meta_quality, higher_order_quality):.2f}."
            )
        
        return insight_content
    
    def _calculate_pattern_connection(self, pattern1: PatternInsight, pattern2: PatternInsight) -> float:
        """Calculate connection strength between two patterns"""
        # Connection based on type compatibility and strength
        type_compatibility = 0.8 if pattern1.pattern_type == pattern2.pattern_type else 0.4
        strength_compatibility = 1.0 - abs(pattern1.pattern_strength - pattern2.pattern_strength)
        confidence_compatibility = 1.0 - abs(pattern1.confidence - pattern2.confidence)
        
        connection_strength = (type_compatibility + strength_compatibility + confidence_compatibility) / 3.0
        
        return max(0.0, min(1.0, connection_strength))
    
    def _determine_connection_type(self, pattern1: PatternInsight, pattern2: PatternInsight) -> str:
        """Determine the type of connection between patterns"""
        if pattern1.pattern_type == pattern2.pattern_type:
            return "reinforcing"
        elif abs(pattern1.pattern_strength - pattern2.pattern_strength) < 0.2:
            return "complementary"
        else:
            return "contrasting"
    
    def _evaluate_insight_quality(self, insight_content: str, patterns: List[PatternInsight]) -> float:
        """Evaluate the quality of generated insights"""
        # Quality factors
        content_length_factor = min(1.0, len(insight_content) / 100.0)  # Normalize by expected length
        pattern_support_factor = sum(p.pattern_strength for p in patterns) / max(len(patterns), 1)
        pattern_diversity_factor = len(set(p.pattern_type for p in patterns)) / max(len(patterns), 1)
        
        insight_quality = (content_length_factor + pattern_support_factor + pattern_diversity_factor) / 3.0
        
        return max(0.0, min(1.0, insight_quality))
    
    def _determine_insight_type(self, insight_content: str, insight_quality: float) -> InsightType:
        """Determine the type of insight generated"""
        if insight_quality > 0.9:
            return InsightType.BREAKTHROUGH_INSIGHT
        elif insight_quality > 0.8:
            return InsightType.META_INSIGHT
        elif insight_quality > 0.7:
            return InsightType.SYNTHESIS_INSIGHT
        elif insight_quality > 0.6:
            return InsightType.STRUCTURAL_INSIGHT
        elif insight_quality > 0.4:
            return InsightType.CAUSAL_INSIGHT
        else:
            return InsightType.PATTERN_INSIGHT
    
    def _assess_insight_significance(self, insight_quality: float, patterns: List[PatternInsight]) -> InsightQuality:
        """Assess the significance level of the insight"""
        # Calculate significance from quality and novelty
        avg_novelty = sum(p.novelty_score for p in patterns) / max(len(patterns), 1)
        significance_score = (insight_quality + avg_novelty) / 2.0
        
        if significance_score > 0.9:
            return InsightQuality.REVOLUTIONARY
        elif significance_score > 0.8:
            return InsightQuality.PROFOUND
        elif significance_score > 0.6:
            return InsightQuality.SIGNIFICANT
        elif significance_score > 0.4:
            return InsightQuality.BASIC
        else:
            return InsightQuality.TRIVIAL
    
    def _generate_insight_description(self, insight_content: str, 
                                    pattern_synthesis: Dict[str, Any], 
                                    insight_type: InsightType) -> str:
        """Generate a descriptive explanation of the insight"""
        synthesis_quality = pattern_synthesis.get('synthesis_quality', 0.0)
        num_connections = len(pattern_synthesis.get('pattern_connections', []))
        
        description = (
            f"{insight_type.value.replace('_', ' ').title()} generated with "
            f"{synthesis_quality:.2f} synthesis quality. "
            f"Integrated {num_connections} pattern connections. "
            f"Content: {insight_content[:100]}..."
        )
        
        return description
    
    def _assess_breakthrough_potential(self, insight_quality: float, insight_significance: InsightQuality) -> float:
        """Assess the breakthrough potential of the insight"""
        significance_scores = {
            InsightQuality.TRIVIAL: 0.1,
            InsightQuality.BASIC: 0.3,
            InsightQuality.SIGNIFICANT: 0.6,
            InsightQuality.PROFOUND: 0.8,
            InsightQuality.REVOLUTIONARY: 1.0
        }
        
        significance_score = significance_scores.get(insight_significance, 0.5)
        breakthrough_potential = (insight_quality + significance_score) / 2.0
        
        # Boost if above breakthrough threshold
        if breakthrough_potential > self.breakthrough_threshold:
            breakthrough_potential = min(1.0, breakthrough_potential * 1.2)
        
        return breakthrough_potential


class MetaInsightCore:
    """Main Meta Insight Core system integrating all meta-cognitive capabilities"""
    
    def __init__(self,
                 default_meta_level: MetaCognitionLevel = MetaCognitionLevel.META_LEVEL_1,
                 insight_threshold: float = 0.6,
                 device: str = "cpu"):
        
        self.default_meta_level = default_meta_level
        self.insight_threshold = insight_threshold
        self.device = device
        
        # Initialize meta-cognitive components
        self.higher_order_processor = HigherOrderProcessor()
        self.metacognition_engine = MetaCognitionEngine()
        self.pattern_recognition_system = PatternRecognitionSystem()
        self.insight_generation_engine = InsightGenerationEngine()
        
        # Performance tracking
        self.total_insight_requests = 0
        self.successful_insights = 0
        self.meta_insight_history = []
        
        # Integration with foundational systems
        self.foundational_systems = {}
        
        logger.info("ðŸ§  Meta Insight Core initialized")
        logger.info(f"   Default meta-level: {default_meta_level.value}")
        logger.info(f"   Insight threshold: {insight_threshold}")
        logger.info(f"   Device: {device}")
    
    def register_foundational_systems(self, **systems):
        """Register foundational systems for integration"""
        self.foundational_systems.update(systems)
        logger.info("âœ… Meta Insight Core foundational systems registered")
    
    async def generate_meta_insight(self,
                                  cognitive_state: torch.Tensor,
                                  meta_level: Optional[MetaCognitionLevel] = None,
                                  context: Optional[Dict[str, Any]] = None) -> MetaInsightResult:
        """Main meta-insight generation method"""
        
        insight_id = f"META_{uuid.uuid4().hex[:8]}"
        processing_start = time.time()
        meta_level = meta_level or self.default_meta_level
        context = context or {}
        
        logger.debug(f"Processing meta-insight generation {insight_id}")
        
        try:
            self.total_insight_requests += 1
            
            # Phase 1: Higher-Order Processing
            higher_order_result = await self.higher_order_processor.process_higher_order(
                cognitive_state, meta_level, context
            )
            
            # Phase 2: Meta-Cognitive Monitoring
            meta_cognitive_analysis = await self.metacognition_engine.monitor_cognition(
                cognitive_state, context
            )
            
            # Phase 3: Pattern Recognition
            recognized_patterns = await self.pattern_recognition_system.recognize_patterns(
                cognitive_state, context
            )
            
            # Phase 4: Insight Generation
            insight_result = await self.insight_generation_engine.generate_insight(
                recognized_patterns, meta_cognitive_analysis, higher_order_result, context
            )
            
            # Phase 5: Integration and Assessment
            final_assessment = self._integrate_and_assess(
                insight_result, higher_order_result, meta_cognitive_analysis, recognized_patterns
            )
            
            # Calculate final metrics
            processing_time = time.time() - processing_start
            
            # Determine insight quality and type
            insight_quality = self._determine_insight_quality(final_assessment)
            insight_type = insight_result.get('insight_type', InsightType.PATTERN_INSIGHT)
            
            # Create meta-insight result
            result = MetaInsightResult(
                insight_id=insight_id,
                insight_type=insight_type,
                insight_quality=self._map_quality_to_enum(insight_quality),
                meta_cognition_level=meta_level,
                insight_content=insight_result.get('insight_content', ''),
                insight_description=insight_result.get('insight_description', ''),
                supporting_patterns=recognized_patterns,
                meta_cognitive_assessment=meta_cognitive_analysis,
                self_reflection_analysis=higher_order_result,
                cognitive_monitoring=meta_cognitive_analysis,
                insight_strength=final_assessment.get('insight_strength', 0.0),
                novelty_score=final_assessment.get('novelty_score', 0.0),
                significance_score=final_assessment.get('significance_score', 0.0),
                confidence_score=final_assessment.get('confidence_score', 0.0),
                generalizability=final_assessment.get('generalizability', 0.0),
                processing_time=processing_time,
                computational_cost=self._calculate_computational_cost(processing_time, meta_level),
                breakthrough_potential=insight_result.get('breakthrough_potential', 0.0)
            )
            
            # Update success tracking
            if final_assessment.get('insight_strength', 0.0) > self.insight_threshold:
                self.successful_insights += 1
            
            # Record in history
            self.meta_insight_history.append(result)
            if len(self.meta_insight_history) > 100:
                self.meta_insight_history = self.meta_insight_history[-50:]
            
            logger.debug(f"âœ… Meta-insight {insight_id} completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Meta-insight generation failed: {e}")
            error_result = MetaInsightResult(
                insight_id=insight_id,
                insight_type=InsightType.PATTERN_INSIGHT,
                insight_quality=InsightQuality.TRIVIAL,
                meta_cognition_level=meta_level,
                insight_content='',
                insight_description=f'Meta-insight generation failed: {e}',
                supporting_patterns=[],
                meta_cognitive_assessment={},
                self_reflection_analysis={},
                cognitive_monitoring={},
                insight_strength=0.0,
                novelty_score=0.0,
                significance_score=0.0,
                confidence_score=0.0,
                generalizability=0.0,
                processing_time=time.time() - processing_start,
                computational_cost=0.0,
                breakthrough_potential=0.0,
                success=False,
                error_log=[str(e)]
            )
            
            return error_result
    
    def _integrate_and_assess(self,
                             insight_result: Dict[str, Any],
                             higher_order_result: Dict[str, Any],
                             meta_cognitive_analysis: Dict[str, Any],
                             patterns: List[PatternInsight]) -> Dict[str, Any]:
        """Integrate all results and assess final insight quality"""
        
        # Extract component qualities
        insight_quality = insight_result.get('insight_quality', 0.0)
        higher_order_quality = higher_order_result.get('processing_quality', 0.0)
        meta_quality = meta_cognitive_analysis.get('monitoring_quality', 0.0)
        
        # Calculate pattern metrics
        if patterns:
            avg_pattern_strength = sum(p.pattern_strength for p in patterns) / len(patterns)
            avg_pattern_novelty = sum(p.novelty_score for p in patterns) / len(patterns)
            avg_pattern_confidence = sum(p.confidence for p in patterns) / len(patterns)
        else:
            avg_pattern_strength = 0.0
            avg_pattern_novelty = 0.0
            avg_pattern_confidence = 0.0
        
        # Integrate insight strength
        insight_strength = (
            0.4 * insight_quality +
            0.2 * higher_order_quality +
            0.2 * meta_quality +
            0.2 * avg_pattern_strength
        )
        
        # Calculate novelty score
        novelty_score = (
            0.6 * avg_pattern_novelty +
            0.4 * higher_order_quality  # Higher-order processing contributes to novelty
        )
        
        # Calculate significance score
        significance_score = (
            0.5 * insight_strength +
            0.3 * novelty_score +
            0.2 * meta_quality
        )
        
        # Calculate confidence score
        confidence_score = (
            0.3 * avg_pattern_confidence +
            0.3 * meta_quality +
            0.4 * min(insight_strength, significance_score)
        )
        
        # Calculate generalizability
        pattern_diversity = len(set(p.pattern_type for p in patterns)) / max(len(patterns), 1) if patterns else 0.0
        generalizability = (
            0.6 * pattern_diversity +
            0.4 * higher_order_quality
        )
        
        return {
            'insight_strength': max(0.0, min(1.0, insight_strength)),
            'novelty_score': max(0.0, min(1.0, novelty_score)),
            'significance_score': max(0.0, min(1.0, significance_score)),
            'confidence_score': max(0.0, min(1.0, confidence_score)),
            'generalizability': max(0.0, min(1.0, generalizability))
        }
    
    def _determine_insight_quality(self, assessment: Dict[str, Any]) -> float:
        """Determine overall insight quality from assessment"""
        insight_strength = assessment.get('insight_strength', 0.0)
        significance_score = assessment.get('significance_score', 0.0)
        confidence_score = assessment.get('confidence_score', 0.0)
        
        # Weighted combination
        quality = (
            0.5 * insight_strength +
            0.3 * significance_score +
            0.2 * confidence_score
        )
        
        return max(0.0, min(1.0, quality))
    
    def _map_quality_to_enum(self, quality_score: float) -> InsightQuality:
        """Map quality score to InsightQuality enum"""
        if quality_score > 0.9:
            return InsightQuality.REVOLUTIONARY
        elif quality_score > 0.8:
            return InsightQuality.PROFOUND
        elif quality_score > 0.6:
            return InsightQuality.SIGNIFICANT
        elif quality_score > 0.4:
            return InsightQuality.BASIC
        else:
            return InsightQuality.TRIVIAL
    
    def _calculate_computational_cost(self, processing_time: float, meta_level: MetaCognitionLevel) -> float:
        """Calculate computational cost of meta-insight processing"""
        base_cost = processing_time * 3.0  # 3 units per second (higher than understanding)
        
        # Meta-level specific costs
        level_costs = {
            MetaCognitionLevel.OBJECT_LEVEL: 0.5,
            MetaCognitionLevel.META_LEVEL_1: 1.0,
            MetaCognitionLevel.META_LEVEL_2: 2.0,
            MetaCognitionLevel.META_LEVEL_3: 4.0,
            MetaCognitionLevel.RECURSIVE_META: 8.0
        }
        
        level_cost = level_costs.get(meta_level, 1.0)
        
        return base_cost + level_cost
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive meta-insight core system status"""
        
        success_rate = self.successful_insights / max(self.total_insight_requests, 1)
        
        recent_performance = {}
        if self.meta_insight_history:
            recent_insights = self.meta_insight_history[-10:]
            recent_performance = {
                'avg_insight_strength': sum(i.insight_strength for i in recent_insights) / len(recent_insights),
                'avg_novelty_score': sum(i.novelty_score for i in recent_insights) / len(recent_insights),
                'avg_significance_score': sum(i.significance_score for i in recent_insights) / len(recent_insights),
                'avg_confidence_score': sum(i.confidence_score for i in recent_insights) / len(recent_insights),
                'avg_processing_time': sum(i.processing_time for i in recent_insights) / len(recent_insights),
                'insight_type_distribution': {
                    itype.value: sum(1 for i in recent_insights if i.insight_type == itype)
                    for itype in InsightType
                },
                'insight_quality_distribution': {
                    quality.value: sum(1 for i in recent_insights if i.insight_quality == quality)
                    for quality in InsightQuality
                }
            }
        
        return {
            'meta_insight_core_status': 'operational',
            'total_insight_requests': self.total_insight_requests,
            'successful_insights': self.successful_insights,
            'success_rate': success_rate,
            'insight_threshold': self.insight_threshold,
            'default_meta_level': self.default_meta_level.value,
            'recent_performance': recent_performance,
            'components': {
                'higher_order_processor': 'operational',
                'metacognition_engine': len(self.metacognition_engine.cognitive_monitoring_data),
                'pattern_recognition_system': len(self.pattern_recognition_system.discovered_patterns),
                'insight_generation_engine': len(self.insight_generation_engine.generated_insights)
            },
            'foundational_systems': {
                system: system in self.foundational_systems
                for system in ['spde_core', 'barenholtz_core', 'cognitive_cycle_core', 'understanding_core']
            }
        }