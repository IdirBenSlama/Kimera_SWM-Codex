"""
Meta Insight Engine - Higher-Order Insight Generation System
============================================================

Advanced meta-cognitive engine that generates insights about insights,
performs higher-order cognitive processing, and manages meta-level
understanding of cognitive processes.

Key Features:
- Meta-cognitive insight generation
- Higher-order pattern recognition
- Insight quality assessment
- Meta-level reasoning and reflection
- Cognitive process optimization
- Insight network analysis
- Emergent understanding detection

Scientific Foundation:
- Metacognition Theory
- Higher-Order Thought Theory
- Insight Problem Solving
- Meta-Learning Frameworks
- Cognitive Architecture Theory
- Network Analysis of Cognition
"""

import numpy as np
import torch
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import time
from datetime import datetime, timedelta
import asyncio
from collections import deque, defaultdict
import networkx as nx
import threading
from abc import ABC, abstractmethod
from ..utils.config import get_api_settings
from ..config.settings import get_settings

logger = logging.getLogger(__name__)

class InsightType(Enum):
    """Types of insights that can be generated"""
    PATTERN_RECOGNITION = "pattern_recognition"
    CAUSAL_RELATIONSHIP = "causal_relationship"
    CONCEPTUAL_BRIDGE = "conceptual_bridge"
    EMERGENT_PROPERTY = "emergent_property"
    CONTRADICTION_RESOLUTION = "contradiction_resolution"
    SYSTEM_OPTIMIZATION = "system_optimization"
    META_INSIGHT = "meta_insight"
    BREAKTHROUGH = "breakthrough"

class InsightQuality(Enum):
    """Quality levels for insights"""
    TRIVIAL = "trivial"
    INTERESTING = "interesting"
    SIGNIFICANT = "significant"
    PROFOUND = "profound"
    REVOLUTIONARY = "revolutionary"

class MetaCognitiveProcess(Enum):
    """Types of meta-cognitive processes"""
    MONITORING = "monitoring"
    EVALUATION = "evaluation"
    PLANNING = "planning"
    REGULATION = "regulation"
    REFLECTION = "reflection"
    OPTIMIZATION = "optimization"

@dataclass
class Insight:
    """Represents a cognitive insight"""
    insight_id: str
    content: str
    insight_type: InsightType
    quality: InsightQuality
    confidence: float
    source_data: torch.Tensor
    context: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    dependencies: List[str] = field(default_factory=list)
    implications: List[str] = field(default_factory=list)
    verification_status: str = "unverified"
    novelty_score: float = 0.0
    coherence_score: float = 0.0
    utility_score: float = 0.0

@dataclass
class MetaInsight:
    """Represents a meta-level insight about other insights"""
    meta_insight_id: str
    target_insights: List[str]
    meta_content: str
    meta_type: str
    emergence_pattern: str
    meta_confidence: float
    synthesis_method: str
    timestamp: datetime = field(default_factory=datetime.now)
    higher_order_implications: List[str] = field(default_factory=list)

@dataclass
class InsightNetwork:
    """Network representation of insights and their relationships"""
    graph: nx.Graph = field(default_factory=nx.Graph)
    insight_nodes: Dict[str, Insight] = field(default_factory=dict)
    relationship_weights: Dict[Tuple[str, str], float] = field(default_factory=dict)
    clusters: List[Set[str]] = field(default_factory=list)
    centrality_scores: Dict[str, float] = field(default_factory=dict)

class InsightQualityAssessor:
    """
    Assesses the quality of generated insights
    
    Uses multiple criteria to evaluate insight quality
    """
    
    def __init__(self, device: str = "cpu"):
        self.settings = get_api_settings()
        logger.debug(f"   Environment: {self.settings.environment}")
        self.device = torch.device(device)
        self.quality_criteria = {
            "novelty": 0.3,
            "coherence": 0.25,
            "utility": 0.25,
            "verification": 0.2
        }
        self.quality_history = deque(maxlen=1000)
        
    def assess_insight_quality(self, insight: Insight) -> InsightQuality:
        """
        Assess the quality of an insight
        
        Args:
            insight: Insight to assess
            
        Returns:
            Quality level of the insight
        """
        
        # Compute quality scores
        novelty_score = self._compute_novelty_score(insight)
        coherence_score = self._compute_coherence_score(insight)
        utility_score = self._compute_utility_score(insight)
        verification_score = self._compute_verification_score(insight)
        
        # Update insight scores
        insight.novelty_score = novelty_score
        insight.coherence_score = coherence_score
        insight.utility_score = utility_score
        
        # Weighted overall quality
        overall_quality = (
            self.quality_criteria["novelty"] * novelty_score +
            self.quality_criteria["coherence"] * coherence_score +
            self.quality_criteria["utility"] * utility_score +
            self.quality_criteria["verification"] * verification_score
        )
        
        # Map to quality levels
        if overall_quality >= 0.9:
            quality = InsightQuality.REVOLUTIONARY
        elif overall_quality >= 0.75:
            quality = InsightQuality.PROFOUND
        elif overall_quality >= 0.6:
            quality = InsightQuality.SIGNIFICANT
        elif overall_quality >= 0.4:
            quality = InsightQuality.INTERESTING
        else:
            quality = InsightQuality.TRIVIAL
        
        # Store assessment
        self.quality_history.append({
            "insight_id": insight.insight_id,
            "quality": quality,
            "overall_score": overall_quality,
            "novelty": novelty_score,
            "coherence": coherence_score,
            "utility": utility_score,
            "verification": verification_score,
            "timestamp": datetime.now()
        })
        
        return quality
    
    def _compute_novelty_score(self, insight: Insight) -> float:
        """Compute novelty score for an insight"""
        # Check against historical insights
        novelty = 1.0
        
        # Compare with recent insights
        for assessment in list(self.quality_history)[-50:]:
            if assessment["insight_id"] != insight.insight_id:
                # Simple content similarity (would be better with semantic similarity)
                content_similarity = self._compute_content_similarity(
                    insight.content, assessment.get("content", "")
                )
                novelty *= (1.0 - content_similarity * 0.1)
        
        return max(0.0, min(1.0, novelty))
    
    def _compute_coherence_score(self, insight: Insight) -> float:
        """Compute coherence score for an insight"""
        # Coherence based on consistency with context and dependencies
        coherence = 0.8  # Base coherence
        
        # Check dependency consistency
        if insight.dependencies:
            # Would check if dependencies are logically consistent
            coherence += 0.1
        
        # Check context consistency
        if insight.context:
            # Would check if context supports the insight
            coherence += 0.1
        
        return max(0.0, min(1.0, coherence))
    
    def _compute_utility_score(self, insight: Insight) -> float:
        """Compute utility score for an insight"""
        # Utility based on potential implications and applications
        utility = 0.5  # Base utility
        
        # More implications = higher utility
        if insight.implications:
            utility += min(0.3, len(insight.implications) * 0.1)
        
        # Higher confidence = higher utility
        utility += insight.confidence * 0.2
        
        return max(0.0, min(1.0, utility))
    
    def _compute_verification_score(self, insight: Insight) -> float:
        """Compute verification score for an insight"""
        # Verification based on verifiability and current status
        if insight.verification_status == "verified":
            return 1.0
        elif insight.verification_status == "partially_verified":
            return 0.7
        elif insight.verification_status == "verifiable":
            return 0.5
        else:
            return 0.3
    
    def _compute_content_similarity(self, content1: str, content2: str) -> float:
        """Compute similarity between two content strings"""
        # Simple word-based similarity (would use semantic embeddings in practice)
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0

class MetaCognitiveProcessor:
    """
    Performs meta-cognitive processing on insights
    
    Monitors, evaluates, and optimizes cognitive processes
    """
    
    def __init__(self, device: str = "cpu"):
        self.settings = get_api_settings()
        logger.debug(f"   Environment: {self.settings.environment}")
        self.device = torch.device(device)
        self.process_history = deque(maxlen=500)
        self.optimization_strategies = {}
        
    def process_metacognition(self, 
                            insights: List[Insight],
                            process_type: MetaCognitiveProcess) -> Dict[str, Any]:
        """
        Perform meta-cognitive processing
        
        Args:
            insights: List of insights to process
            process_type: Type of meta-cognitive process
            
        Returns:
            Results of meta-cognitive processing
        """
        
        start_time = time.time()
        
        # Route to appropriate processor
        if process_type == MetaCognitiveProcess.MONITORING:
            result = self._monitor_cognitive_processes(insights)
        elif process_type == MetaCognitiveProcess.EVALUATION:
            result = self._evaluate_cognitive_performance(insights)
        elif process_type == MetaCognitiveProcess.PLANNING:
            result = self._plan_cognitive_strategies(insights)
        elif process_type == MetaCognitiveProcess.REGULATION:
            result = self._regulate_cognitive_processes(insights)
        elif process_type == MetaCognitiveProcess.REFLECTION:
            result = self._reflect_on_cognitive_processes(insights)
        elif process_type == MetaCognitiveProcess.OPTIMIZATION:
            result = self._optimize_cognitive_processes(insights)
        else:
            result = {"error": f"Unknown process type: {process_type}"}
        
        # Record processing
        processing_time = time.time() - start_time
        self.process_history.append({
            "process_type": process_type.value,
            "insights_processed": len(insights),
            "processing_time": processing_time,
            "result_keys": list(result.keys()),
            "timestamp": datetime.now()
        })
        
        result["processing_time"] = processing_time
        result["process_type"] = process_type.value
        
        return result
    
    def _monitor_cognitive_processes(self, insights: List[Insight]) -> Dict[str, Any]:
        """Monitor ongoing cognitive processes"""
        monitoring_result = {
            "total_insights": len(insights),
            "insight_types": {},
            "quality_distribution": {},
            "confidence_stats": {},
            "temporal_patterns": {}
        }
        
        # Analyze insight types
        for insight in insights:
            insight_type = insight.insight_type.value
            monitoring_result["insight_types"][insight_type] = \
                monitoring_result["insight_types"].get(insight_type, 0) + 1
        
        # Analyze quality distribution
        for insight in insights:
            quality = insight.quality.value
            monitoring_result["quality_distribution"][quality] = \
                monitoring_result["quality_distribution"].get(quality, 0) + 1
        
        # Confidence statistics
        if insights:
            confidences = [insight.confidence for insight in insights]
            monitoring_result["confidence_stats"] = {
                "mean": np.mean(confidences),
                "std": np.std(confidences),
                "min": np.min(confidences),
                "max": np.max(confidences)
            }
        
        return monitoring_result
    
    def _evaluate_cognitive_performance(self, insights: List[Insight]) -> Dict[str, Any]:
        """Evaluate cognitive performance"""
        evaluation_result = {
            "performance_score": 0.0,
            "efficiency_metrics": {},
            "effectiveness_metrics": {},
            "improvement_areas": []
        }
        
        if not insights:
            return evaluation_result
        
        # Performance based on quality and quantity
        quality_scores = {
            InsightQuality.TRIVIAL: 0.1,
            InsightQuality.INTERESTING: 0.3,
            InsightQuality.SIGNIFICANT: 0.6,
            InsightQuality.PROFOUND: 0.8,
            InsightQuality.REVOLUTIONARY: 1.0
        }
        
        total_quality = sum(quality_scores.get(insight.quality, 0.0) for insight in insights)
        evaluation_result["performance_score"] = total_quality / len(insights)
        
        # Efficiency metrics
        processing_times = [
            (insight.timestamp - insights[0].timestamp).total_seconds() 
            for insight in insights[1:]
        ]
        if processing_times:
            evaluation_result["efficiency_metrics"] = {
                "avg_processing_time": np.mean(processing_times),
                "processing_consistency": 1.0 / (1.0 + np.std(processing_times))
            }
        
        # Effectiveness metrics
        high_quality_insights = [
            insight for insight in insights 
            if insight.quality in [InsightQuality.SIGNIFICANT, InsightQuality.PROFOUND, InsightQuality.REVOLUTIONARY]
        ]
        evaluation_result["effectiveness_metrics"] = {
            "high_quality_ratio": len(high_quality_insights) / len(insights),
            "average_confidence": np.mean([insight.confidence for insight in insights])
        }
        
        # Improvement areas
        if evaluation_result["performance_score"] < 0.5:
            evaluation_result["improvement_areas"].append("overall_quality")
        if evaluation_result["effectiveness_metrics"]["high_quality_ratio"] < 0.3:
            evaluation_result["improvement_areas"].append("insight_quality")
        
        return evaluation_result
    
    def _plan_cognitive_strategies(self, insights: List[Insight]) -> Dict[str, Any]:
        """Plan cognitive strategies"""
        planning_result = {
            "recommended_strategies": [],
            "resource_allocation": {},
            "priority_areas": [],
            "expected_outcomes": {}
        }
        
        # Analyze current state
        if not insights:
            planning_result["recommended_strategies"].append("increase_exploration")
            return planning_result
        
        # Strategy based on insight patterns
        insight_types = [insight.insight_type for insight in insights]
        type_counts = {t: insight_types.count(t) for t in set(insight_types)}
        
        # Recommend strategies based on gaps
        all_types = set(InsightType)
        missing_types = all_types - set(type_counts.keys())
        
        for missing_type in missing_types:
            planning_result["recommended_strategies"].append(f"explore_{missing_type.value}")
        
        # Priority areas based on quality
        low_quality_types = [
            insight.insight_type for insight in insights 
            if insight.quality in [InsightQuality.TRIVIAL, InsightQuality.INTERESTING]
        ]
        
        for insight_type in set(low_quality_types):
            planning_result["priority_areas"].append(f"improve_{insight_type.value}")
        
        return planning_result
    
    def _regulate_cognitive_processes(self, insights: List[Insight]) -> Dict[str, Any]:
        """Regulate cognitive processes"""
        regulation_result = {
            "adjustments_made": [],
            "process_modifications": {},
            "resource_reallocation": {},
            "performance_targets": {}
        }
        
        # Analyze current performance
        if not insights:
            return regulation_result
        
        # Regulate based on quality distribution
        quality_counts = {}
        for insight in insights:
            quality = insight.quality.value
            quality_counts[quality] = quality_counts.get(quality, 0) + 1
        
        # If too many trivial insights, adjust threshold
        total_insights = len(insights)
        trivial_ratio = quality_counts.get("trivial", 0) / total_insights
        
        if trivial_ratio > 0.5:
            regulation_result["adjustments_made"].append("increase_quality_threshold")
            regulation_result["process_modifications"]["quality_threshold"] = 0.6
        
        # If too few high-quality insights, increase exploration
        high_quality_ratio = (
            quality_counts.get("significant", 0) + 
            quality_counts.get("profound", 0) + 
            quality_counts.get("revolutionary", 0)
        ) / total_insights
        
        if high_quality_ratio < 0.2:
            regulation_result["adjustments_made"].append("increase_exploration")
            regulation_result["resource_reallocation"]["exploration"] = 0.7
        
        return regulation_result
    
    def _reflect_on_cognitive_processes(self, insights: List[Insight]) -> Dict[str, Any]:
        """Reflect on cognitive processes"""
        reflection_result = {
            "process_insights": [],
            "learning_outcomes": [],
            "pattern_discoveries": [],
            "future_directions": []
        }
        
        if not insights:
            return reflection_result
        
        # Reflect on temporal patterns
        if len(insights) > 1:
            time_diffs = [
                (insights[i].timestamp - insights[i-1].timestamp).total_seconds()
                for i in range(1, len(insights))
            ]
            
            if time_diffs:
                avg_time = np.mean(time_diffs)
                reflection_result["process_insights"].append(
                    f"Average insight generation time: {avg_time:.2f} seconds"
                )
        
        # Reflect on quality progression
        quality_progression = [insight.quality.value for insight in insights]
        if len(set(quality_progression)) > 1:
            reflection_result["learning_outcomes"].append(
                "Quality diversity observed in insight generation"
            )
        
        # Reflect on type diversity
        type_diversity = len(set(insight.insight_type for insight in insights))
        reflection_result["pattern_discoveries"].append(
            f"Generated {type_diversity} different types of insights"
        )
        
        return reflection_result
    
    def _optimize_cognitive_processes(self, insights: List[Insight]) -> Dict[str, Any]:
        """Optimize cognitive processes"""
        optimization_result = {
            "optimization_strategies": [],
            "parameter_adjustments": {},
            "efficiency_improvements": [],
            "quality_enhancements": []
        }
        
        if not insights:
            return optimization_result
        
        # Optimize based on performance analysis
        high_confidence_insights = [i for i in insights if i.confidence > 0.8]
        if len(high_confidence_insights) / len(insights) > 0.7:
            optimization_result["optimization_strategies"].append("increase_confidence_threshold")
        
        # Optimize based on quality distribution
        quality_scores = [self._get_quality_score(insight.quality) for insight in insights]
        avg_quality = np.mean(quality_scores)
        
        if avg_quality < 0.5:
            optimization_result["quality_enhancements"].append("improve_assessment_criteria")
            optimization_result["parameter_adjustments"]["quality_weight"] = 0.8
        
        return optimization_result
    
    def _get_quality_score(self, quality: InsightQuality) -> float:
        """Get numeric score for quality level"""
        scores = {
            InsightQuality.TRIVIAL: 0.1,
            InsightQuality.INTERESTING: 0.3,
            InsightQuality.SIGNIFICANT: 0.6,
            InsightQuality.PROFOUND: 0.8,
            InsightQuality.REVOLUTIONARY: 1.0
        }
        return scores.get(quality, 0.0)

class InsightNetworkAnalyzer:
    """
    Analyzes networks of insights to find patterns and connections
    """
    
    def __init__(self):
        self.settings = get_api_settings()
        logger.debug(f"   Environment: {self.settings.environment}")
        self.insight_networks = {}
        self.analysis_cache = {}
        
    def build_insight_network(self, insights: List[Insight]) -> InsightNetwork:
        """
        Build a network of insights based on their relationships
        
        Args:
            insights: List of insights to network
            
        Returns:
            InsightNetwork with connections and analysis
        """
        
        network = InsightNetwork()
        
        # Add insights as nodes
        for insight in insights:
            network.graph.add_node(insight.insight_id)
            network.insight_nodes[insight.insight_id] = insight
        
        # Add edges based on relationships
        for insight in insights:
            # Connect based on dependencies
            for dep_id in insight.dependencies:
                if dep_id in network.insight_nodes:
                    weight = 0.8  # High weight for dependencies
                    network.graph.add_edge(insight.insight_id, dep_id, weight=weight)
                    network.relationship_weights[(insight.insight_id, dep_id)] = weight
            
            # Connect based on content similarity
            for other_insight in insights:
                if other_insight.insight_id != insight.insight_id:
                    similarity = self._compute_insight_similarity(insight, other_insight)
                    if similarity > 0.3:  # Threshold for connection
                        network.graph.add_edge(
                            insight.insight_id, 
                            other_insight.insight_id, 
                            weight=similarity
                        )
                        network.relationship_weights[(insight.insight_id, other_insight.insight_id)] = similarity
        
        # Compute network properties
        network.centrality_scores = nx.betweenness_centrality(network.graph)
        
        # Find clusters
        if network.graph.number_of_nodes() > 2:
            try:
                communities = nx.community.greedy_modularity_communities(network.graph)
                network.clusters = [set(community) for community in communities]
            except Exception as e:
                logger.error(f"Error in meta_insight_engine.py: {e}", exc_info=True)
                raise  # Re-raise for proper error handling
                network.clusters = []
        
        return network
    
    def analyze_insight_emergence(self, network: InsightNetwork) -> Dict[str, Any]:
        """
        Analyze how insights emerge from the network
        
        Args:
            network: InsightNetwork to analyze
            
        Returns:
            Analysis of insight emergence patterns
        """
        
        analysis = {
            "emergence_patterns": [],
            "key_insights": [],
            "network_metrics": {},
            "cluster_analysis": {},
            "temporal_evolution": {}
        }
        
        # Network metrics
        if network.graph.number_of_nodes() > 0:
            analysis["network_metrics"] = {
                "node_count": network.graph.number_of_nodes(),
                "edge_count": network.graph.number_of_edges(),
                "density": nx.density(network.graph),
                "average_clustering": nx.average_clustering(network.graph),
                "diameter": nx.diameter(network.graph) if nx.is_connected(network.graph) else "disconnected"
            }
        
        # Key insights based on centrality
        sorted_centrality = sorted(
            network.centrality_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        for insight_id, centrality in sorted_centrality[:5]:
            if insight_id in network.insight_nodes:
                insight = network.insight_nodes[insight_id]
                analysis["key_insights"].append({
                    "insight_id": insight_id,
                    "centrality": centrality,
                    "quality": insight.quality.value,
                    "type": insight.insight_type.value
                })
        
        # Cluster analysis
        if network.clusters:
            analysis["cluster_analysis"] = {
                "cluster_count": len(network.clusters),
                "cluster_sizes": [len(cluster) for cluster in network.clusters],
                "largest_cluster_size": max(len(cluster) for cluster in network.clusters),
                "cluster_modularity": self._compute_modularity(network)
            }
        
        # Emergence patterns
        analysis["emergence_patterns"] = self._identify_emergence_patterns(network)
        
        return analysis
    
    def _compute_insight_similarity(self, insight1: Insight, insight2: Insight) -> float:
        """Compute similarity between two insights"""
        # Simple similarity based on type and content
        type_similarity = 1.0 if insight1.insight_type == insight2.insight_type else 0.0
        
        # Content similarity (simplified)
        content_similarity = self._compute_content_similarity(insight1.content, insight2.content)
        
        # Context similarity
        context_similarity = self._compute_context_similarity(insight1.context, insight2.context)
        
        # Weighted combination
        similarity = (0.4 * type_similarity + 0.4 * content_similarity + 0.2 * context_similarity)
        
        return similarity
    
    def _compute_content_similarity(self, content1: str, content2: str) -> float:
        """Compute content similarity between two strings"""
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _compute_context_similarity(self, context1: Dict[str, Any], context2: Dict[str, Any]) -> float:
        """Compute context similarity between two contexts"""
        keys1 = set(context1.keys())
        keys2 = set(context2.keys())
        
        if not keys1 or not keys2:
            return 0.0
        
        common_keys = keys1.intersection(keys2)
        if not common_keys:
            return 0.0
        
        # Simple similarity based on common keys
        return len(common_keys) / len(keys1.union(keys2))
    
    def _compute_modularity(self, network: InsightNetwork) -> float:
        """Compute modularity of the network clusters"""
        if not network.clusters or not network.graph.edges():
            return 0.0
        
        try:
            # Convert clusters to proper format for networkx
            partition = {}
            for i, cluster in enumerate(network.clusters):
                for node in cluster:
                    partition[node] = i
            
            return nx.community.modularity(network.graph, network.clusters)
        except Exception as e:
            logger.error(f"Error in meta_insight_engine.py: {e}", exc_info=True)
            raise  # Re-raise for proper error handling
            return 0.0
    
    def _identify_emergence_patterns(self, network: InsightNetwork) -> List[str]:
        """Identify patterns of insight emergence"""
        patterns = []
        
        # Check for hub patterns
        if network.centrality_scores:
            max_centrality = max(network.centrality_scores.values())
            if max_centrality > 0.5:
                patterns.append("hub_emergence")
        
        # Check for cluster patterns
        if len(network.clusters) > 1:
            patterns.append("cluster_formation")
        
        # Check for chain patterns
        if network.graph.number_of_edges() > network.graph.number_of_nodes():
            patterns.append("chain_development")
        
        return patterns

class MetaInsightEngine:
    """
    Main Meta Insight Engine
    
    Orchestrates meta-cognitive processing, insight quality assessment,
    and higher-order insight generation.
    """
    
    def __init__(self, device: str = "cpu"):
        self.settings = get_api_settings()
        logger.debug(f"   Environment: {self.settings.environment}")
        self.device = torch.device(device)
        
        # Core components
        self.quality_assessor = InsightQualityAssessor(device=device)
        self.metacognitive_processor = MetaCognitiveProcessor(device=device)
        self.network_analyzer = InsightNetworkAnalyzer()
        
        # State management
        self.insights_repository = {}
        self.meta_insights_repository = {}
        self.processing_history = deque(maxlen=1000)
        
        # Metrics
        self.total_insights_processed = 0
        self.total_meta_insights_generated = 0
        self.average_processing_time = 0.0
        
        # Threading
        self.processing_lock = threading.Lock()
        
        logger.info(f"Meta Insight Engine initialized on device: {device}")
    
    def process_insights(self, 
                        input_insights: List[Insight],
                        generate_meta_insights: bool = True) -> Dict[str, Any]:
        """
        Process insights through meta-cognitive analysis
        
        Args:
            input_insights: List of insights to process
            generate_meta_insights: Whether to generate meta-insights
            
        Returns:
            Processing results with enhanced insights and meta-insights
        """
        
        with self.processing_lock:
            start_time = time.time()
            
            # Store insights
            for insight in input_insights:
                self.insights_repository[insight.insight_id] = insight
            
            # Assess insight quality
            enhanced_insights = []
            for insight in input_insights:
                quality = self.quality_assessor.assess_insight_quality(insight)
                insight.quality = quality
                enhanced_insights.append(insight)
            
            # Perform meta-cognitive processing
            metacognitive_results = {}
            for process_type in MetaCognitiveProcess:
                result = self.metacognitive_processor.process_metacognition(
                    enhanced_insights, process_type
                )
                metacognitive_results[process_type.value] = result
            
            # Build insight network
            insight_network = self.network_analyzer.build_insight_network(enhanced_insights)
            
            # Analyze emergence patterns
            emergence_analysis = self.network_analyzer.analyze_insight_emergence(insight_network)
            
            # Generate meta-insights if requested
            meta_insights = []
            if generate_meta_insights:
                meta_insights = self._generate_meta_insights(
                    enhanced_insights, 
                    metacognitive_results, 
                    emergence_analysis
                )
            
            # Update metrics
            processing_time = time.time() - start_time
            self.total_insights_processed += len(input_insights)
            self.total_meta_insights_generated += len(meta_insights)
            
            if self.total_insights_processed > 0:
                self.average_processing_time = (
                    (self.average_processing_time * (self.total_insights_processed - len(input_insights)) + 
                     processing_time) / self.total_insights_processed
                )
            
            # Store processing record
            processing_record = {
                "timestamp": datetime.now(),
                "insights_processed": len(input_insights),
                "meta_insights_generated": len(meta_insights),
                "processing_time": processing_time,
                "quality_distribution": self._get_quality_distribution(enhanced_insights),
                "metacognitive_summary": self._summarize_metacognitive_results(metacognitive_results)
            }
            
            self.processing_history.append(processing_record)
            
            return {
                "enhanced_insights": enhanced_insights,
                "meta_insights": meta_insights,
                "metacognitive_results": metacognitive_results,
                "network_analysis": emergence_analysis,
                "processing_metrics": {
                    "processing_time": processing_time,
                    "insights_processed": len(input_insights),
                    "meta_insights_generated": len(meta_insights),
                    "quality_improvements": self._count_quality_improvements(enhanced_insights)
                }
            }
    
    def _generate_meta_insights(self, 
                               insights: List[Insight],
                               metacognitive_results: Dict[str, Any],
                               emergence_analysis: Dict[str, Any]) -> List[MetaInsight]:
        """Generate meta-insights from processed insights"""
        
        meta_insights = []
        
        # Meta-insight from quality patterns
        quality_meta_insight = self._generate_quality_meta_insight(insights)
        if quality_meta_insight:
            meta_insights.append(quality_meta_insight)
        
        # Meta-insight from emergence patterns
        emergence_meta_insight = self._generate_emergence_meta_insight(emergence_analysis)
        if emergence_meta_insight:
            meta_insights.append(emergence_meta_insight)
        
        # Meta-insight from cognitive performance
        performance_meta_insight = self._generate_performance_meta_insight(metacognitive_results)
        if performance_meta_insight:
            meta_insights.append(performance_meta_insight)
        
        # Store meta-insights
        for meta_insight in meta_insights:
            self.meta_insights_repository[meta_insight.meta_insight_id] = meta_insight
        
        return meta_insights
    
    def _generate_quality_meta_insight(self, insights: List[Insight]) -> Optional[MetaInsight]:
        """Generate meta-insight about insight quality patterns"""
        
        if not insights:
            return None
        
        quality_distribution = self._get_quality_distribution(insights)
        
        # Find dominant quality pattern
        dominant_quality = max(quality_distribution.items(), key=lambda x: x[1])
        
        meta_insight = MetaInsight(
            meta_insight_id=f"quality_meta_{int(time.time())}",
            target_insights=[i.insight_id for i in insights],
            meta_content=f"Quality pattern analysis reveals {dominant_quality[0]} insights dominate with {dominant_quality[1]} occurrences",
            meta_type="quality_pattern",
            emergence_pattern="quality_distribution_analysis",
            meta_confidence=0.8,
            synthesis_method="statistical_analysis",
            higher_order_implications=[
                "Quality patterns indicate cognitive processing effectiveness",
                "Dominant quality levels suggest optimization opportunities"
            ]
        )
        
        return meta_insight
    
    def _generate_emergence_meta_insight(self, emergence_analysis: Dict[str, Any]) -> Optional[MetaInsight]:
        """Generate meta-insight about emergence patterns"""
        
        if not emergence_analysis.get("emergence_patterns"):
            return None
        
        patterns = emergence_analysis["emergence_patterns"]
        
        meta_insight = MetaInsight(
            meta_insight_id=f"emergence_meta_{int(time.time())}",
            target_insights=[],  # Applies to network as a whole
            meta_content=f"Emergence analysis reveals patterns: {', '.join(patterns)}",
            meta_type="emergence_pattern",
            emergence_pattern="network_topology_analysis",
            meta_confidence=0.7,
            synthesis_method="network_analysis",
            higher_order_implications=[
                "Emergence patterns indicate cognitive network dynamics",
                "Network topology suggests information flow patterns"
            ]
        )
        
        return meta_insight
    
    def _generate_performance_meta_insight(self, metacognitive_results: Dict[str, Any]) -> Optional[MetaInsight]:
        """Generate meta-insight about cognitive performance"""
        
        evaluation_result = metacognitive_results.get("evaluation", {})
        performance_score = evaluation_result.get("performance_score", 0.0)
        
        if performance_score == 0.0:
            return None
        
        meta_insight = MetaInsight(
            meta_insight_id=f"performance_meta_{int(time.time())}",
            target_insights=[],  # Applies to cognitive system as a whole
            meta_content=f"Cognitive performance analysis shows score of {performance_score:.2f}",
            meta_type="performance_analysis",
            emergence_pattern="metacognitive_evaluation",
            meta_confidence=0.9,
            synthesis_method="performance_evaluation",
            higher_order_implications=[
                "Performance metrics indicate cognitive system effectiveness",
                "Performance patterns suggest optimization strategies"
            ]
        )
        
        return meta_insight
    
    def _get_quality_distribution(self, insights: List[Insight]) -> Dict[str, int]:
        """Get distribution of insight qualities"""
        distribution = {}
        for insight in insights:
            quality = insight.quality.value
            distribution[quality] = distribution.get(quality, 0) + 1
        return distribution
    
    def _summarize_metacognitive_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize metacognitive processing results"""
        summary = {}
        
        for process_type, result in results.items():
            if isinstance(result, dict):
                summary[process_type] = {
                    "status": "completed",
                    "key_metrics": list(result.keys())[:3],  # Top 3 metrics
                    "processing_time": result.get("processing_time", 0.0)
                }
            else:
                summary[process_type] = {"status": "error", "result": str(result)}
        
        return summary
    
    def _count_quality_improvements(self, insights: List[Insight]) -> int:
        """Count insights that had quality improvements"""
        # This would compare against previous quality assessments
        # For now, return a placeholder
        return len([i for i in insights if i.quality != InsightQuality.TRIVIAL])
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get current engine status and metrics"""
        return {
            "status": "operational",
            "device": str(self.device),
            "total_insights_processed": self.total_insights_processed,
            "total_meta_insights_generated": self.total_meta_insights_generated,
            "average_processing_time": self.average_processing_time,
            "insights_in_repository": len(self.insights_repository),
            "meta_insights_in_repository": len(self.meta_insights_repository),
            "processing_history_size": len(self.processing_history),
            "quality_assessor_history": len(self.quality_assessor.quality_history),
            "metacognitive_processor_history": len(self.metacognitive_processor.process_history),
            "last_updated": datetime.now().isoformat()
        }
    
    def get_insights_summary(self) -> Dict[str, Any]:
        """Get summary of processed insights"""
        if not self.insights_repository:
            return {
                "total_insights": 0,
                "quality_distribution": {},
                "type_distribution": {},
                "recent_insights": []
            }
        
        insights = list(self.insights_repository.values())
        
        # Quality distribution
        quality_dist = self._get_quality_distribution(insights)
        
        # Type distribution
        type_dist = {}
        for insight in insights:
            insight_type = insight.insight_type.value
            type_dist[insight_type] = type_dist.get(insight_type, 0) + 1
        
        # Recent insights
        recent_insights = sorted(insights, key=lambda x: x.timestamp, reverse=True)[:10]
        
        return {
            "total_insights": len(insights),
            "quality_distribution": quality_dist,
            "type_distribution": type_dist,
            "recent_insights": [
                {
                    "insight_id": insight.insight_id,
                    "quality": insight.quality.value,
                    "type": insight.insight_type.value,
                    "confidence": insight.confidence,
                    "timestamp": insight.timestamp.isoformat()
                }
                for insight in recent_insights
            ]
        }
    
    def get_meta_insights_summary(self) -> Dict[str, Any]:
        """Get summary of generated meta-insights"""
        if not self.meta_insights_repository:
            return {
                "total_meta_insights": 0,
                "meta_types": {},
                "recent_meta_insights": []
            }
        
        meta_insights = list(self.meta_insights_repository.values())
        
        # Meta-type distribution
        meta_type_dist = {}
        for meta_insight in meta_insights:
            meta_type = meta_insight.meta_type
            meta_type_dist[meta_type] = meta_type_dist.get(meta_type, 0) + 1
        
        # Recent meta-insights
        recent_meta_insights = sorted(meta_insights, key=lambda x: x.timestamp, reverse=True)[:5]
        
        return {
            "total_meta_insights": len(meta_insights),
            "meta_types": meta_type_dist,
            "recent_meta_insights": [
                {
                    "meta_insight_id": meta_insight.meta_insight_id,
                    "meta_type": meta_insight.meta_type,
                    "meta_confidence": meta_insight.meta_confidence,
                    "target_insights_count": len(meta_insight.target_insights),
                    "timestamp": meta_insight.timestamp.isoformat()
                }
                for meta_insight in recent_meta_insights
            ]
        }
    
    def reset_engine(self):
        """Reset engine state"""
        self.insights_repository.clear()
        self.meta_insights_repository.clear()
        self.processing_history.clear()
        self.quality_assessor.quality_history.clear()
        self.metacognitive_processor.process_history.clear()
        
        self.total_insights_processed = 0
        self.total_meta_insights_generated = 0
        self.average_processing_time = 0.0
        
        logger.info("Meta Insight Engine reset")

# Factory function for easy instantiation
def create_meta_insight_engine(device: str = "cpu") -> MetaInsightEngine:
    """
    Create and initialize Meta Insight Engine
    
    Args:
        device: Computing device ("cpu" or "cuda")
        
    Returns:
        Initialized Meta Insight Engine
    """
    return MetaInsightEngine(device=device) 