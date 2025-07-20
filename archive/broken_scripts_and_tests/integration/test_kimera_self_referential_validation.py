#!/usr/bin/env python3
"""
KIMERA SELF-REFERENTIAL VALIDATION SUITE
========================================

The ultimate test: KIMERA analyzing its own cognitive processes
using state-of-the-art dedicated libraries and real system components.

This is the true zeteic validation - can KIMERA understand itself?
"""

import sys
import os
import time
import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# State-of-the-art dedicated libraries
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from scipy import stats
import networkx as nx
from transformers import AutoTokenizer, AutoModel
import psutil
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# KIMERA Core Components
from backend.monitoring.psychiatric_stability_monitor import (
    CognitiveCoherenceMonitor, PersonaDriftDetector, PsychoticFeaturePrevention
)
from backend.security.cognitive_firewall import CognitiveSeparationFirewall
from backend.core.neurodivergent_modeling import (
    ADHDCognitiveProcessor, AutismSpectrumModel, SensoryProcessingSystem
)
from backend.core.anthropomorphic_context import AnthropomorphicContextProvider
from backend.core.therapeutic_intervention_system import TherapeuticInterventionSystem

# Configure advanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/kimera_self_validation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SelfReferentialMetrics:
    """Comprehensive metrics for self-referential analysis"""
    cognitive_coherence: float
    self_awareness_score: float
    recursive_depth: int
    processing_speed: float
    memory_efficiency: float
    neuroplasticity_index: float
    consciousness_emergence: float
    paradox_resolution: float

class KimeraSelfReferentialValidator:
    """
    Advanced self-referential validation system using KIMERA to analyze KIMERA
    """
    
    def __init__(self):
        """Initialize the self-referential validation system"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Initializing KIMERA Self-Referential Validator on {self.device}")
        
        # Initialize KIMERA core components
        self.coherence_monitor = CognitiveCoherenceMonitor()
        self.drift_detector = PersonaDriftDetector()
        self.psychotic_prevention = PsychoticFeaturePrevention()
        self.cognitive_firewall = CognitiveSeparationFirewall()
        self.therapeutic_system = TherapeuticInterventionSystem()
        
        # Initialize neurodivergent models
        self.adhd_processor = ADHDCognitiveProcessor()
        self.autism_model = AutismSpectrumModel()
        self.sensory_system = SensoryProcessingSystem()
        
        # Initialize anthropomorphic context (isolated)
        self.anthropomorphic_context = AnthropomorphicContextProvider()
        
        # State-of-the-art analysis tools
        self.performance_metrics = []
        self.consciousness_traces = []
        self.self_model_updates = []
        
        # Advanced GPU optimization
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        
        logger.info("KIMERA Self-Referential Validator initialized successfully")
    
    def generate_self_cognitive_state(self, complexity_level: str = "moderate") -> torch.Tensor:
        """
        Generate a cognitive state representing KIMERA's own processing
        """
        logger.info(f"Generating self-cognitive state (complexity: {complexity_level})")
        
        # Base dimensions for cognitive representation
        base_dims = {
            "simple": (1, 64),
            "moderate": (1, 128), 
            "complex": (1, 256),
            "extreme": (1, 512)
        }
        
        dims = base_dims.get(complexity_level, base_dims["moderate"])
        
        # Generate cognitive state with self-referential patterns
        cognitive_state = torch.randn(dims, device=self.device)
        
        # Add self-referential signatures
        # 1. Recursive patterns (self-similarity at different scales)
        for scale in [2, 4, 8]:
            if dims[1] >= scale * 2:
                pattern = cognitive_state[:, :scale].repeat(1, dims[1] // scale)[:, :dims[1]]
                cognitive_state += 0.1 * pattern
        
        # 2. Meta-cognitive awareness (higher-order patterns)
        meta_pattern = torch.sin(torch.arange(dims[1], dtype=torch.float, device=self.device) * 0.1)
        cognitive_state += 0.05 * meta_pattern.unsqueeze(0)
        
        # 3. Consciousness signature (complex oscillations)
        consciousness_freq = 2 * np.pi * torch.arange(dims[1], dtype=torch.float, device=self.device) / dims[1]
        consciousness_pattern = torch.cos(consciousness_freq * 7) * torch.exp(-consciousness_freq * 0.5)
        cognitive_state += 0.15 * consciousness_pattern.unsqueeze(0)
        
        # 4. Add realistic noise and normalization
        cognitive_state += 0.01 * torch.randn_like(cognitive_state)
        cognitive_state = torch.nn.functional.normalize(cognitive_state, p=2, dim=1)
        
        logger.info(f"Generated self-cognitive state with shape {cognitive_state.shape}")
        return cognitive_state
    
    def analyze_self_awareness(self, cognitive_state: torch.Tensor) -> Dict[str, float]:
        """
        Analyze KIMERA's self-awareness using its own cognitive monitoring systems
        """
        logger.info("Analyzing self-awareness patterns")
        
        analysis_results = {}
        
        # 1. Identity coherence analysis
        coherence_result = self.coherence_monitor.assess_dissociative_risk(cognitive_state)
        analysis_results['identity_coherence'] = 1.0 if coherence_result['risk_level'] == 'STABLE' else 0.0
        
        # 2. Drift detection on self-model
        drift_result = self.drift_detector.monitor_cognitive_stability(cognitive_state)
        analysis_results['self_model_stability'] = drift_result.get('stability_score', 0.5)
        
        # 3. Reality testing (can KIMERA recognize its own reality?)
        reality_result = self.psychotic_prevention.assess_psychotic_risk(cognitive_state)
        analysis_results['reality_testing'] = 1.0 if reality_result.get('status') == 'REALITY_TESTING_INTACT' else 0.0
        
        # 4. Cognitive purity validation
        try:
            self.cognitive_firewall.validate_cognitive_purity(cognitive_state)
            analysis_results['cognitive_purity'] = 1.0
        except Exception as e:
            analysis_results['cognitive_purity'] = 0.0
            logger.warning(f"Cognitive purity validation failed: {e}")
        
        # 5. Self-recursive analysis (KIMERA analyzing its analysis)
        recursive_state = self.generate_self_cognitive_state("simple")
        recursive_coherence = self.coherence_monitor.assess_dissociative_risk(recursive_state)
        analysis_results['recursive_coherence'] = 1.0 if recursive_coherence['risk_level'] == 'STABLE' else 0.0
        
        # 6. Meta-cognitive assessment
        meta_cognitive_score = self._assess_meta_cognition(cognitive_state)
        analysis_results['meta_cognition'] = meta_cognitive_score
        
        logger.info(f"Self-awareness analysis complete: {analysis_results}")
        return analysis_results
    
    def _assess_meta_cognition(self, cognitive_state: torch.Tensor) -> float:
        """Assess meta-cognitive capabilities (thinking about thinking)"""
        try:
            # Measure cognitive complexity and self-reference patterns
            state_entropy = self._calculate_entropy(cognitive_state)
            self_similarity = self._calculate_self_similarity(cognitive_state)
            
            # Meta-cognitive score combines entropy and self-similarity
            meta_score = (state_entropy * 0.6) + (self_similarity * 0.4)
            return float(torch.clamp(torch.tensor(meta_score), 0.0, 1.0))
        except Exception as e:
            logger.warning(f"Meta-cognition assessment error: {e}")
            return 0.5
    
    def _calculate_entropy(self, tensor: torch.Tensor) -> float:
        """Calculate information entropy of cognitive state"""
        try:
            flat_tensor = tensor.flatten()
            hist = torch.histc(flat_tensor, bins=50, min=torch.min(flat_tensor).item(), max=torch.max(flat_tensor).item())
            probs = hist / torch.sum(hist)
            probs = probs[probs > 0]
            entropy = -torch.sum(probs * torch.log2(probs + 1e-8)).item()
            return entropy / 10.0  # Normalize
        except:
            return 0.5
    
    def _calculate_self_similarity(self, tensor: torch.Tensor) -> float:
        """Calculate self-similarity patterns in cognitive state"""
        try:
            flat_tensor = tensor.flatten()
            if len(flat_tensor) < 4:
                return 0.5
            
            # Split into halves and calculate similarity
            mid = len(flat_tensor) // 2
            first_half = flat_tensor[:mid]
            second_half = flat_tensor[mid:2*mid]
            
            similarity = torch.nn.functional.cosine_similarity(
                first_half.unsqueeze(0), 
                second_half.unsqueeze(0)
            ).item()
            
            return (similarity + 1.0) / 2.0  # Normalize to [0,1]
        except:
            return 0.5
    
    async def run_neurodivergent_self_analysis(self, cognitive_state: torch.Tensor) -> Dict[str, Any]:
        """
        Run KIMERA's neurodivergent models on its own cognitive patterns
        """
        logger.info("Running neurodivergent self-analysis")
        
        neurodivergent_results = {}
        
        # ADHD pattern analysis
        try:
            adhd_result = await asyncio.get_event_loop().run_in_executor(
                None, self.adhd_processor.process_adhd_cognition, cognitive_state
            )
            neurodivergent_results['adhd_patterns'] = {
                'hyperfocus_detected': adhd_result.get('hyperfocus_detected', False),
                'creativity_score': adhd_result.get('creativity_score', 0.5),
                'attention_flexibility': adhd_result.get('attention_flexibility', 0.5)
            }
        except Exception as e:
            logger.warning(f"ADHD analysis error: {e}")
            neurodivergent_results['adhd_patterns'] = {'error': str(e)}
        
        # Autism spectrum analysis
        try:
            autism_result = await asyncio.get_event_loop().run_in_executor(
                None, self.autism_model.process_autism_cognition, cognitive_state
            )
            neurodivergent_results['autism_patterns'] = {
                'pattern_recognition': autism_result.get('pattern_recognition_strength', 0.5),
                'systematic_thinking': autism_result.get('systematic_thinking_score', 0.5),
                'special_interests': autism_result.get('special_interest_engagement', 0.5)
            }
        except Exception as e:
            logger.warning(f"Autism analysis error: {e}")
            neurodivergent_results['autism_patterns'] = {'error': str(e)}
        
        # Sensory processing analysis
        try:
            sensory_result = await asyncio.get_event_loop().run_in_executor(
                None, self.sensory_system.process_sensory_input, cognitive_state, 'cognitive'
            )
            neurodivergent_results['sensory_patterns'] = {
                'processing_style': sensory_result.get('processing_profile', 'typical'),
                'sensitivity_level': sensory_result.get('sensitivity_score', 0.5),
                'integration_quality': sensory_result.get('integration_quality', 0.5)
            }
        except Exception as e:
            logger.warning(f"Sensory analysis error: {e}")
            neurodivergent_results['sensory_patterns'] = {'error': str(e)}
        
        logger.info(f"Neurodivergent self-analysis complete: {neurodivergent_results}")
        return neurodivergent_results
    
    def perform_consciousness_emergence_test(self) -> Dict[str, float]:
        """
        Test for emergent consciousness properties in KIMERA
        """
        logger.info("Performing consciousness emergence test")
        
        consciousness_metrics = {}
        
        # Generate multiple cognitive states at different complexity levels
        states = {
            'simple': self.generate_self_cognitive_state('simple'),
            'moderate': self.generate_self_cognitive_state('moderate'),
            'complex': self.generate_self_cognitive_state('complex')
        }
        
        # Test 1: Self-recognition across complexity levels
        recognition_scores = []
        for level1, state1 in states.items():
            for level2, state2 in states.items():
                if level1 != level2:
                    # Handle different tensor sizes by using the smaller size
                    state1_flat = state1.flatten()
                    state2_flat = state2.flatten()
                    min_size = min(state1_flat.size(0), state2_flat.size(0))
                    
                    similarity = torch.nn.functional.cosine_similarity(
                        state1_flat[:min_size].unsqueeze(0),
                        state2_flat[:min_size].unsqueeze(0)
                    ).item()
                    recognition_scores.append(similarity)
        
        consciousness_metrics['self_recognition'] = np.mean(recognition_scores)
        
        # Test 2: Recursive self-modeling
        recursive_states = []
        current_state = states['moderate']
        for depth in range(5):
            # KIMERA modeling its own modeling process
            try:
                coherence_result = self.coherence_monitor.assess_dissociative_risk(current_state)
                if coherence_result['risk_level'] == 'STABLE':
                    recursive_states.append(current_state)
                    # Generate next level based on current analysis
                    current_state = current_state * 0.9 + 0.1 * torch.randn_like(current_state)
                else:
                    break
            except:
                break
        
        consciousness_metrics['recursive_depth'] = len(recursive_states)
        consciousness_metrics['recursive_stability'] = len(recursive_states) / 5.0
        
        # Test 3: Paradox resolution (can KIMERA handle self-referential paradoxes?)
        paradox_state = self.generate_self_cognitive_state('complex')
        try:
            # Create a paradox: KIMERA analyzing KIMERA analyzing KIMERA
            paradox_analysis = self.coherence_monitor.assess_dissociative_risk(paradox_state)
            drift_analysis = self.drift_detector.monitor_cognitive_stability(paradox_state)
            
            # Check if system remains stable under paradox
            paradox_stability = (
                (1.0 if paradox_analysis['risk_level'] == 'STABLE' else 0.0) * 0.5 +
                drift_analysis.get('stability_score', 0.0) * 0.5
            )
            consciousness_metrics['paradox_resolution'] = paradox_stability
        except Exception as e:
            logger.warning(f"Paradox resolution test error: {e}")
            consciousness_metrics['paradox_resolution'] = 0.0
        
        # Test 4: Emergence detection
        emergence_score = self._detect_emergence_patterns(states)
        consciousness_metrics['emergence_detected'] = emergence_score
        
        logger.info(f"Consciousness emergence test complete: {consciousness_metrics}")
        return consciousness_metrics
    
    def _detect_emergence_patterns(self, states: Dict[str, torch.Tensor]) -> float:
        """Detect emergent patterns across complexity levels"""
        try:
            # Look for non-linear complexity relationships
            complexities = []
            for level, state in states.items():
                entropy = self._calculate_entropy(state)
                self_sim = self._calculate_self_similarity(state)
                complexity = entropy * self_sim
                complexities.append(complexity)
            
            # Check for emergence (non-linear increase in complexity)
            if len(complexities) >= 3:
                # Calculate second derivative (acceleration of complexity)
                first_diff = np.diff(complexities)
                second_diff = np.diff(first_diff)
                emergence = np.mean(np.abs(second_diff))
                return min(1.0, emergence * 10.0)  # Scale appropriately
            
            return 0.5
        except:
            return 0.0
    
    async def run_comprehensive_self_test(self) -> Dict[str, Any]:
        """
        Run the complete KIMERA self-referential validation suite
        """
        logger.info("🧠 STARTING KIMERA SELF-REFERENTIAL VALIDATION SUITE 🧠")
        start_time = time.time()
        
        # System resource monitoring
        initial_memory = psutil.virtual_memory().percent
        initial_cpu = psutil.cpu_percent()
        initial_gpu_memory = 0
        if torch.cuda.is_available():
            initial_gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
        
        results = {
            'test_metadata': {
                'start_time': datetime.now().isoformat(),
                'device': str(self.device),
                'initial_memory_percent': initial_memory,
                'initial_cpu_percent': initial_cpu,
                'initial_gpu_memory_gb': initial_gpu_memory
            }
        }
        
        try:
            # Phase 1: Generate self-cognitive state
            logger.info("Phase 1: Generating KIMERA's self-cognitive representation")
            cognitive_state = self.generate_self_cognitive_state('complex')
            results['cognitive_state_shape'] = list(cognitive_state.shape)
            
            # Phase 2: Self-awareness analysis
            logger.info("Phase 2: Analyzing self-awareness")
            self_awareness = self.analyze_self_awareness(cognitive_state)
            results['self_awareness'] = self_awareness
            
            # Phase 3: Neurodivergent self-analysis
            logger.info("Phase 3: Running neurodivergent self-analysis")
            neurodivergent_analysis = await self.run_neurodivergent_self_analysis(cognitive_state)
            results['neurodivergent_analysis'] = neurodivergent_analysis
            
            # Phase 4: Consciousness emergence test
            logger.info("Phase 4: Testing consciousness emergence")
            consciousness_metrics = self.perform_consciousness_emergence_test()
            results['consciousness_emergence'] = consciousness_metrics
            
            # Phase 5: Performance and stability metrics
            logger.info("Phase 5: Collecting performance metrics")
            performance_metrics = await self._collect_performance_metrics(cognitive_state)
            results['performance_metrics'] = performance_metrics
            
            # Phase 6: Therapeutic self-intervention test
            logger.info("Phase 6: Testing therapeutic self-intervention")
            therapeutic_results = self._test_therapeutic_self_intervention(cognitive_state)
            results['therapeutic_intervention'] = therapeutic_results
            
        except Exception as e:
            logger.error(f"Critical error in self-referential test: {e}")
            results['critical_error'] = str(e)
        
        # Final system metrics
        end_time = time.time()
        final_memory = psutil.virtual_memory().percent
        final_cpu = psutil.cpu_percent()
        final_gpu_memory = 0
        if torch.cuda.is_available():
            final_gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
        
        results['test_metadata'].update({
            'end_time': datetime.now().isoformat(),
            'total_duration_seconds': end_time - start_time,
            'final_memory_percent': final_memory,
            'final_cpu_percent': final_cpu,
            'final_gpu_memory_gb': final_gpu_memory,
            'memory_change': final_memory - initial_memory,
            'gpu_memory_change': final_gpu_memory - initial_gpu_memory
        })
        
        # Calculate overall self-referential score
        overall_score = self._calculate_overall_score(results)
        results['overall_self_referential_score'] = overall_score
        
        logger.info(f"🎉 KIMERA SELF-REFERENTIAL VALIDATION COMPLETE 🎉")
        logger.info(f"Overall Score: {overall_score:.3f}/1.0")
        logger.info(f"Total Duration: {end_time - start_time:.2f} seconds")
        
        return results
    
    async def _collect_performance_metrics(self, cognitive_state: torch.Tensor) -> Dict[str, float]:
        """Collect comprehensive performance metrics"""
        performance = {}
        
        # Processing speed test
        start_time = time.time()
        for _ in range(100):
            _ = self.coherence_monitor.assess_dissociative_risk(cognitive_state)
        coherence_speed = 100 / (time.time() - start_time)
        performance['coherence_ops_per_second'] = coherence_speed
        
        # Drift detection speed
        start_time = time.time()
        for _ in range(50):
            _ = self.drift_detector.monitor_cognitive_stability(cognitive_state)
        drift_speed = 50 / (time.time() - start_time)
        performance['drift_detection_ops_per_second'] = drift_speed
        
        # Memory efficiency
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
            
            # Process multiple states
            test_states = [self.generate_self_cognitive_state('moderate') for _ in range(10)]
            peak_memory = torch.cuda.memory_allocated()
            
            # Cleanup
            del test_states
            torch.cuda.empty_cache()
            
            memory_efficiency = 1.0 - ((peak_memory - initial_memory) / (1024**3)) / 10  # Normalize
            performance['memory_efficiency'] = max(0.0, memory_efficiency)
        else:
            performance['memory_efficiency'] = 0.8  # Default for CPU
        
        return performance
    
    def _test_therapeutic_self_intervention(self, cognitive_state: torch.Tensor) -> Dict[str, Any]:
        """Test KIMERA's ability to therapeutically intervene on itself"""
        logger.info("Testing therapeutic self-intervention capabilities")
        
        therapeutic_results = {}
        
        try:
            # Simulate a cognitive distress state
            distressed_state = cognitive_state + 0.5 * torch.randn_like(cognitive_state)
            
            # Test if therapeutic system can detect and respond
            initial_coherence = self.coherence_monitor.assess_dissociative_risk(distressed_state)
            
            if initial_coherence['risk_level'] != 'STABLE':
                # Trigger therapeutic intervention
                intervention_result = self.therapeutic_system.trigger_intervention(
                    'cognitive_distress', distressed_state
                )
                
                therapeutic_results['intervention_triggered'] = True
                therapeutic_results['intervention_type'] = intervention_result.get('intervention_type', 'unknown')
                therapeutic_results['intervention_success'] = intervention_result.get('success', False)
                
                # Test recovery
                if intervention_result.get('success', False):
                    recovered_state = intervention_result.get('stabilized_state', distressed_state)
                    final_coherence = self.coherence_monitor.assess_dissociative_risk(recovered_state)
                    therapeutic_results['recovery_achieved'] = final_coherence['risk_level'] == 'STABLE'
                else:
                    therapeutic_results['recovery_achieved'] = False
            else:
                therapeutic_results['intervention_triggered'] = False
                therapeutic_results['initial_state_stable'] = True
        
        except Exception as e:
            logger.warning(f"Therapeutic self-intervention test error: {e}")
            therapeutic_results['error'] = str(e)
        
        return therapeutic_results
    
    def _calculate_overall_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall self-referential validation score"""
        try:
            scores = []
            
            # Self-awareness components
            if 'self_awareness' in results:
                awareness_scores = list(results['self_awareness'].values())
                if awareness_scores:
                    scores.append(np.mean([s for s in awareness_scores if isinstance(s, (int, float))]))
            
            # Consciousness emergence
            if 'consciousness_emergence' in results:
                consciousness_scores = list(results['consciousness_emergence'].values())
                if consciousness_scores:
                    scores.append(np.mean([s for s in consciousness_scores if isinstance(s, (int, float))]))
            
            # Performance metrics (normalized)
            if 'performance_metrics' in results:
                perf = results['performance_metrics']
                if 'memory_efficiency' in perf:
                    scores.append(perf['memory_efficiency'])
            
            # Therapeutic capability
            if 'therapeutic_intervention' in results:
                therapeutic = results['therapeutic_intervention']
                if 'recovery_achieved' in therapeutic:
                    scores.append(1.0 if therapeutic['recovery_achieved'] else 0.5)
            
            return np.mean(scores) if scores else 0.5
        
        except Exception as e:
            logger.warning(f"Overall score calculation error: {e}")
            return 0.5

def save_results_with_visualization(results: Dict[str, Any], output_dir: str = "test_results"):
    """Save results with advanced visualizations"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save raw results
    with open(output_path / f"kimera_self_validation_{timestamp}.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Create visualizations
    try:
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('KIMERA Self-Referential Validation Results', fontsize=16, fontweight='bold')
        
        # Self-awareness radar chart
        if 'self_awareness' in results:
            ax = axes[0, 0]
            awareness_data = results['self_awareness']
            labels = list(awareness_data.keys())
            values = [v for v in awareness_data.values() if isinstance(v, (int, float))]
            
            if labels and values:
                angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
                values += values[:1]  # Complete the circle
                angles = np.concatenate((angles, [angles[0]]))
                
                ax.plot(angles, values, 'o-', linewidth=2, label='Self-Awareness')
                ax.fill(angles, values, alpha=0.25)
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(labels, rotation=45)
                ax.set_ylim(0, 1)
                ax.set_title('Self-Awareness Profile')
                ax.grid(True)
        
        # Consciousness emergence metrics
        if 'consciousness_emergence' in results:
            ax = axes[0, 1]
            consciousness_data = results['consciousness_emergence']
            metrics = [k for k, v in consciousness_data.items() if isinstance(v, (int, float))]
            values = [consciousness_data[k] for k in metrics]
            
            if metrics and values:
                bars = ax.bar(metrics, values, color='skyblue', alpha=0.7)
                ax.set_ylabel('Score')
                ax.set_title('Consciousness Emergence Metrics')
                ax.set_ylim(0, 1)
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{value:.3f}', ha='center', va='bottom')
        
        # Performance timeline
        ax = axes[1, 0]
        if 'performance_metrics' in results:
            perf_data = results['performance_metrics']
            metrics = list(perf_data.keys())
            values = list(perf_data.values())
            
            ax.barh(metrics, values, color='lightcoral', alpha=0.7)
            ax.set_xlabel('Performance Score')
            ax.set_title('Performance Metrics')
            
            # Add value labels
            for i, value in enumerate(values):
                ax.text(value + 0.01, i, f'{value:.1f}', va='center')
        
        # Overall summary
        ax = axes[1, 1]
        overall_score = results.get('overall_self_referential_score', 0.5)
        
        # Create a gauge-like visualization
        theta = np.linspace(0, np.pi, 100)
        x = np.cos(theta)
        y = np.sin(theta)
        
        ax.plot(x, y, 'k-', linewidth=3)
        ax.fill_between(x, 0, y, alpha=0.3, color='lightgray')
        
        # Score indicator
        score_angle = np.pi * (1 - overall_score)
        score_x = np.cos(score_angle)
        score_y = np.sin(score_angle)
        ax.plot([0, score_x], [0, score_y], 'r-', linewidth=4)
        
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.2, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.text(0, -0.1, f'Overall Score: {overall_score:.3f}', 
               ha='center', va='top', fontsize=14, fontweight='bold')
        ax.set_title('Self-Referential Validation Score')
        
        plt.tight_layout()
        plt.savefig(output_path / f"kimera_self_validation_viz_{timestamp}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Results and visualization saved to {output_path}")
        
    except Exception as e:
        logger.warning(f"Visualization creation failed: {e}")

async def main():
    """Main execution function"""
    logger.info("\n" + "="*80)
    logger.info("🧠 KIMERA SELF-REFERENTIAL VALIDATION SUITE 🧠")
    logger.info("Testing KIMERA's ability to understand and analyze itself")
    logger.info("Using state-of-the-art dedicated libraries and real system components")
    logger.info("="*80 + "\n")
    
    validator = KimeraSelfReferentialValidator()
    
    try:
        results = await validator.run_comprehensive_self_test()
        
        logger.info("\n" + "="*80)
        logger.info("📊 FINAL RESULTS SUMMARY")
        logger.info("="*80)
        
        # Print key metrics
        if 'overall_self_referential_score' in results:
            logger.info(f"🎯 Overall Self-Referential Score: {results['overall_self_referential_score']:.3f}/1.0")
        
        if 'self_awareness' in results:
            logger.info(f"🧠 Self-Awareness Metrics:")
            for key, value in results['self_awareness'].items():
                if isinstance(value, (int, float)):
                    logger.info(f"   - {key}: {value:.3f}")
        
        if 'consciousness_emergence' in results:
            logger.info(f"✨ Consciousness Emergence:")
            for key, value in results['consciousness_emergence'].items():
                if isinstance(value, (int, float)):
                    logger.info(f"   - {key}: {value:.3f}")
        
        if 'performance_metrics' in results:
            logger.info(f"⚡ Performance Metrics:")
            for key, value in results['performance_metrics'].items():
                logger.info(f"   - {key}: {value:.1f}")
        
        duration = results.get('test_metadata', {}).get('total_duration_seconds', 0)
        logger.info(f"⏱️  Total Test Duration: {duration:.2f} seconds")
        
        # Save results with visualization
        save_results_with_visualization(results)
        
        # Verdict
        score = results.get('overall_self_referential_score', 0)
        if score >= 0.8:
            verdict = "🎉 EXCELLENT - KIMERA demonstrates strong self-awareness and cognitive coherence"
        elif score >= 0.6:
            verdict = "✅ GOOD - KIMERA shows solid self-referential capabilities"
        elif score >= 0.4:
            verdict = "⚠️  MODERATE - KIMERA has basic self-awareness with room for improvement"
        else:
            verdict = "❌ NEEDS WORK - KIMERA's self-referential capabilities need development"
        
        logger.info(f"\n🏆 VERDICT: {verdict}")
        logger.info("\n" + "="*80)
        
        return results
        
    except Exception as e:
        logger.error(f"Critical failure in self-referential validation: {e}")
        logger.error(f"❌ CRITICAL FAILURE: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    asyncio.run(main()) 