"""
Cognitive Ensemble Trading Engine for Kimera

Revolutionary multi-model ensemble that combines:
- Contradiction detection models
- Thermodynamic optimization models  
- Pattern recognition models
- Sentiment analysis models
- Macro economic models

Each model contributes to trading decisions with dynamic weight adjustment
based on performance and market conditions.
"""

import asyncio
import logging
import time
import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum

# Kimera cognitive components
from src.engines.cognitive_field_dynamics import CognitiveFieldDynamics
from src.engines.contradiction_engine import ContradictionEngine
from src.engines.thermodynamics import SemanticThermodynamicsEngine

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Types of cognitive models in the ensemble"""
    CONTRADICTION_DETECTOR = "contradiction_detector"
    THERMODYNAMIC_OPTIMIZER = "thermodynamic_optimizer"
    PATTERN_RECOGNIZER = "pattern_recognizer"
    SENTIMENT_ANALYZER = "sentiment_analyzer"
    MACRO_ANALYZER = "macro_analyzer"
    VOLATILITY_PREDICTOR = "volatility_predictor"
    MOMENTUM_TRACKER = "momentum_tracker"

@dataclass
class ModelSignal:
    """Signal from an individual model"""
    model_type: ModelType
    action: str  # 'buy', 'sell', 'hold'
    confidence: float  # 0-1
    strength: float  # Signal strength
    reasoning: str
    timestamp: datetime
    market_context: Dict[str, Any]

@dataclass
class EnsembleDecision:
    """Final ensemble decision"""
    action: str
    confidence: float
    position_size: float
    expected_return: float
    risk_score: float
    contributing_models: List[ModelType]
    model_weights: Dict[ModelType, float]
    reasoning: str
    timestamp: datetime

class ContradictionTradingModel:
    """Model that detects market contradictions for trading opportunities"""
    
    def __init__(self):
        self.contradiction_engine = ContradictionEngine()
        self.cognitive_field = CognitiveFieldDynamics(dimension=256)
        self.performance_history = deque(maxlen=1000)
        
    async def analyze(self, market_data: Dict[str, Any]) -> Tuple[float, float]:
        """Analyze market for contradiction-based trading signals"""
        try:
            # Create market embedding
            price = market_data.get('price', 0)
            volume = market_data.get('volume', 0)
            volatility = market_data.get('volatility', 0)
            
            market_embedding = torch.tensor([
                price / 100000,  # Normalize price
                volume / 1000000,  # Normalize volume
                volatility * 10  # Amplify volatility signal
            ], dtype=torch.float32)
            
            # Add to cognitive field
            field = self.cognitive_field.add_geoid(
                f"contradiction_analysis_{time.time()}",
                market_embedding
            )
            
            if not field:
                return 0.0, 0.5  # Neutral signal, medium confidence
            
            # Detect semantic contradictions
            field_strength = field.field_strength
            
            # Generate trading signal based on contradictions
            if field_strength > 0.8:
                signal = 1.0  # Strong buy signal
                confidence = 0.9
                reasoning = f"High contradiction detected (strength: {field_strength:.3f})"
            elif field_strength > 0.6:
                signal = 0.5  # Moderate buy signal
                confidence = 0.7
                reasoning = f"Moderate contradiction detected (strength: {field_strength:.3f})"
            elif field_strength < 0.3:
                signal = -0.5  # Moderate sell signal
                confidence = 0.6
                reasoning = f"Low contradiction, market efficient (strength: {field_strength:.3f})"
            else:
                signal = 0.0  # Hold signal
                confidence = 0.5
                reasoning = f"Neutral contradiction level (strength: {field_strength:.3f})"
            
            return signal, confidence
            
        except Exception as e:
            logger.error(f"❌ Contradiction model error: {e}")
            return 0.0, 0.1  # Neutral with low confidence on error

class ThermodynamicTradingModel:
    """Model that uses thermodynamic principles for trading optimization"""
    
    def __init__(self):
        self.thermodynamics_engine = SemanticThermodynamicsEngine()
        self.temperature_history = deque(maxlen=100)
        self.entropy_history = deque(maxlen=100)
        
    async def analyze(self, market_data: Dict[str, Any]) -> Tuple[float, float]:
        """Analyze market using thermodynamic principles"""
        try:
            # Calculate market temperature (volatility-based)
            volatility = market_data.get('volatility', 0)
            volume = market_data.get('volume', 0)
            
            # Market temperature calculation
            temperature = volatility * np.log(1 + volume / 1000000)
            self.temperature_history.append(temperature)
            
            # Calculate entropy (market disorder)
            price_changes = market_data.get('price_changes', [0])
            if len(price_changes) > 1:
                entropy = -sum(p * np.log(p + 1e-10) for p in price_changes if p > 0)
            else:
                entropy = 0.5
            self.entropy_history.append(entropy)
            
            # Thermodynamic trading logic
            avg_temperature = np.mean(list(self.temperature_history)[-10:])
            avg_entropy = np.mean(list(self.entropy_history)[-10:])
            
            # Generate signals based on thermodynamic state
            if temperature > avg_temperature * 1.5 and entropy > avg_entropy * 1.2:
                # High temperature, high entropy = market overheating
                signal = -0.8  # Strong sell signal
                confidence = 0.85
                reasoning = f"Market overheating (T={temperature:.3f}, S={entropy:.3f})"
            elif temperature < avg_temperature * 0.7 and entropy < avg_entropy * 0.8:
                # Low temperature, low entropy = market cooling
                signal = 0.8  # Strong buy signal
                confidence = 0.85
                reasoning = f"Market cooling down (T={temperature:.3f}, S={entropy:.3f})"
            elif entropy > avg_entropy * 1.5:
                # High entropy = uncertainty, reduce positions
                signal = -0.3
                confidence = 0.6
                reasoning = f"High market entropy (S={entropy:.3f})"
            else:
                signal = 0.0
                confidence = 0.5
                reasoning = f"Thermodynamic equilibrium (T={temperature:.3f}, S={entropy:.3f})"
            
            return signal, confidence
            
        except Exception as e:
            logger.error(f"❌ Thermodynamic model error: {e}")
            return 0.0, 0.1

class PatternRecognitionModel:
    """Model that recognizes complex market patterns"""
    
    def __init__(self):
        self.cognitive_field = CognitiveFieldDynamics(dimension=512)
        self.pattern_memory = {}
        self.pattern_performance = defaultdict(list)
        
    async def analyze(self, market_data: Dict[str, Any]) -> Tuple[float, float]:
        """Analyze market patterns using cognitive fields"""
        try:
            # Extract pattern features
            price = market_data.get('price', 0)
            volume = market_data.get('volume', 0)
            volatility = market_data.get('volatility', 0)
            trend = market_data.get('trend', 0)
            
            # Create pattern embedding
            pattern_features = torch.tensor([
                price / 100000,
                volume / 1000000,
                volatility,
                trend,
                np.sin(2 * np.pi * time.time() / 86400),  # Daily cycle
                np.sin(2 * np.pi * time.time() / 604800)  # Weekly cycle
            ], dtype=torch.float32)
            
            # Add to cognitive field for pattern analysis
            field = self.cognitive_field.add_geoid(
                f"pattern_{time.time()}",
                pattern_features
            )
            
            if not field:
                return 0.0, 0.3
            
            # Find similar patterns in memory
            neighbors = self.cognitive_field.find_semantic_neighbors(
                field.geoid_id, energy_threshold=0.1
            )
            
            # Analyze pattern strength and direction
            pattern_strength = field.field_strength
            resonance = field.resonance_frequency
            
            # Generate signal based on pattern analysis
            if pattern_strength > 0.7 and resonance > 1.5:
                signal = 0.6  # Bullish pattern
                confidence = 0.8
                reasoning = f"Strong bullish pattern (strength={pattern_strength:.3f})"
            elif pattern_strength > 0.7 and resonance < 0.5:
                signal = -0.6  # Bearish pattern
                confidence = 0.8
                reasoning = f"Strong bearish pattern (strength={pattern_strength:.3f})"
            elif len(neighbors) > 3:
                # Similar patterns found
                signal = 0.3 if pattern_strength > 0.5 else -0.3
                confidence = 0.6
                reasoning = f"Similar patterns detected ({len(neighbors)} neighbors)"
            else:
                signal = 0.0
                confidence = 0.4
                reasoning = "No clear patterns detected"
            
            return signal, confidence
            
        except Exception as e:
            logger.error(f"❌ Pattern recognition error: {e}")
            return 0.0, 0.1

class SentimentAnalysisModel:
    """Model that analyzes market sentiment"""
    
    def __init__(self):
        self.sentiment_history = deque(maxlen=50)
        self.fear_greed_index = 0.5
        
    async def analyze(self, market_data: Dict[str, Any]) -> Tuple[float, float]:
        """Analyze market sentiment"""
        try:
            # Extract sentiment indicators
            volume_spike = market_data.get('volume_spike', 0)
            price_momentum = market_data.get('momentum', 0)
            volatility = market_data.get('volatility', 0)
            
            # Calculate sentiment score
            sentiment = (price_momentum * 0.4 + 
                        (1 - volatility) * 0.3 + 
                        volume_spike * 0.3)
            
            self.sentiment_history.append(sentiment)
            
            # Update fear/greed index
            avg_sentiment = np.mean(list(self.sentiment_history))
            self.fear_greed_index = max(0, min(1, avg_sentiment))
            
            # Generate signals based on sentiment
            if self.fear_greed_index < 0.2:
                # Extreme fear = buying opportunity
                signal = 0.7
                confidence = 0.8
                reasoning = f"Extreme fear detected (F&G: {self.fear_greed_index:.3f})"
            elif self.fear_greed_index > 0.8:
                # Extreme greed = selling opportunity
                signal = -0.7
                confidence = 0.8
                reasoning = f"Extreme greed detected (F&G: {self.fear_greed_index:.3f})"
            elif sentiment > avg_sentiment * 1.3:
                # Sentiment improving
                signal = 0.4
                confidence = 0.6
                reasoning = f"Improving sentiment (current: {sentiment:.3f})"
            elif sentiment < avg_sentiment * 0.7:
                # Sentiment deteriorating
                signal = -0.4
                confidence = 0.6
                reasoning = f"Deteriorating sentiment (current: {sentiment:.3f})"
            else:
                signal = 0.0
                confidence = 0.4
                reasoning = f"Neutral sentiment (F&G: {self.fear_greed_index:.3f})"
            
            return signal, confidence
            
        except Exception as e:
            logger.error(f"❌ Sentiment analysis error: {e}")
            return 0.0, 0.1

class MacroEconomicModel:
    """Model that analyzes macro economic factors"""
    
    def __init__(self):
        self.macro_indicators = {}
        self.correlation_matrix = {}
        
    async def analyze(self, market_data: Dict[str, Any]) -> Tuple[float, float]:
        """Analyze macro economic factors"""
        try:
            # Extract macro indicators
            interest_rates = market_data.get('interest_rates', 0.05)
            inflation = market_data.get('inflation', 0.02)
            gdp_growth = market_data.get('gdp_growth', 0.03)
            dollar_index = market_data.get('dollar_index', 100)
            
            # Calculate macro score
            # Lower interest rates = bullish for crypto
            # Higher inflation = bullish for crypto (store of value)
            # Strong dollar = bearish for crypto
            
            macro_score = (
                (0.05 - interest_rates) * 10 +  # Lower rates = positive
                (inflation - 0.02) * 5 +        # Higher inflation = positive
                gdp_growth * 3 +                # Growth = positive
                (100 - dollar_index) * 0.01     # Weaker dollar = positive
            )
            
            # Generate signal based on macro conditions
            if macro_score > 0.5:
                signal = min(0.8, macro_score)
                confidence = 0.7
                reasoning = f"Favorable macro conditions (score: {macro_score:.3f})"
            elif macro_score < -0.5:
                signal = max(-0.8, macro_score)
                confidence = 0.7
                reasoning = f"Unfavorable macro conditions (score: {macro_score:.3f})"
            else:
                signal = 0.0
                confidence = 0.5
                reasoning = f"Neutral macro conditions (score: {macro_score:.3f})"
            
            return signal, confidence
            
        except Exception as e:
            logger.error(f"❌ Macro analysis error: {e}")
            return 0.0, 0.1

class CognitiveEnsembleEngine:
    """
    Revolutionary ensemble engine that combines multiple cognitive models
    for superior trading performance with dynamic weight adjustment.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize all cognitive models
        self.models = {
            ModelType.CONTRADICTION_DETECTOR: ContradictionTradingModel(),
            ModelType.THERMODYNAMIC_OPTIMIZER: ThermodynamicTradingModel(),
            ModelType.PATTERN_RECOGNIZER: PatternRecognitionModel(),
            ModelType.SENTIMENT_ANALYZER: SentimentAnalysisModel(),
            ModelType.MACRO_ANALYZER: MacroEconomicModel()
        }
        
        # Initial ensemble weights (will be dynamically adjusted)
        self.ensemble_weights = {
            ModelType.CONTRADICTION_DETECTOR: 0.30,
            ModelType.THERMODYNAMIC_OPTIMIZER: 0.25,
            ModelType.PATTERN_RECOGNIZER: 0.20,
            ModelType.SENTIMENT_ANALYZER: 0.15,
            ModelType.MACRO_ANALYZER: 0.10
        }
        
        # Performance tracking for weight adjustment
        self.model_performance = defaultdict(list)
        self.ensemble_history = deque(maxlen=1000)
        self.weight_adjustment_interval = 100  # Adjust weights every 100 decisions
        self.decision_count = 0
        
        logger.info("🧠 Cognitive Ensemble Engine initialized")
        logger.info(f"   Models: {len(self.models)}")
        logger.info(f"   Initial weights: {dict(self.ensemble_weights)}")
    
    async def generate_ensemble_signal(self, market_data: Dict[str, Any]) -> EnsembleDecision:
        """Generate trading signal from ensemble of cognitive models"""
        start_time = time.time()
        
        # Collect signals from all models
        model_signals = {}
        model_confidences = {}
        model_reasoning = {}
        
        for model_type, model in self.models.items():
            try:
                signal, confidence = await model.analyze(market_data)
                model_signals[model_type] = signal
                model_confidences[model_type] = confidence
                model_reasoning[model_type] = f"{model_type.value}: {signal:.3f} (conf: {confidence:.3f})"
                
            except Exception as e:
                logger.error(f"❌ Model {model_type.value} failed: {e}")
                model_signals[model_type] = 0.0
                model_confidences[model_type] = 0.1
                model_reasoning[model_type] = f"{model_type.value}: ERROR"
        
        # Calculate weighted ensemble signal
        weighted_signal = 0.0
        total_weight = 0.0
        contributing_models = []
        
        for model_type, signal in model_signals.items():
            # Weight by both ensemble weight and model confidence
            effective_weight = (self.ensemble_weights[model_type] * 
                              model_confidences[model_type])
            
            weighted_signal += signal * effective_weight
            total_weight += effective_weight
            
            if abs(signal) > 0.1:  # Model is contributing
                contributing_models.append(model_type)
        
        # Normalize signal
        final_signal = weighted_signal / total_weight if total_weight > 0 else 0.0
        ensemble_confidence = total_weight / len(self.models)
        
        # Determine action
        if final_signal > 0.3:
            action = "buy"
        elif final_signal < -0.3:
            action = "sell"
        else:
            action = "hold"
        
        # Calculate position size based on confidence and signal strength
        position_size = min(abs(final_signal) * ensemble_confidence * 0.2, 0.1)  # Max 10%
        
        # Estimate expected return and risk
        expected_return = abs(final_signal) * 0.05  # 5% max expected return
        risk_score = 1.0 - ensemble_confidence
        
        # Create reasoning summary
        reasoning_parts = list(model_reasoning.values())
        reasoning = f"Ensemble decision (signal: {final_signal:.3f}, conf: {ensemble_confidence:.3f})\n"
        reasoning += "\n".join(reasoning_parts)
        
        # Create ensemble decision
        decision = EnsembleDecision(
            action=action,
            confidence=ensemble_confidence,
            position_size=position_size,
            expected_return=expected_return,
            risk_score=risk_score,
            contributing_models=contributing_models,
            model_weights=dict(self.ensemble_weights),
            reasoning=reasoning,
            timestamp=datetime.now()
        )
        
        # Record for performance tracking
        self.ensemble_history.append({
            'decision': decision,
            'model_signals': model_signals,
            'model_confidences': model_confidences,
            'processing_time': time.time() - start_time
        })
        
        self.decision_count += 1
        
        # Adjust weights periodically
        if self.decision_count % self.weight_adjustment_interval == 0:
            await self.adjust_ensemble_weights()
        
        return decision
    
    async def adjust_ensemble_weights(self):
        """Dynamically adjust ensemble weights based on model performance"""
        logger.info("🔄 Adjusting ensemble weights based on performance...")
        
        # Calculate performance metrics for each model
        if len(self.ensemble_history) < 50:
            return  # Need more data
        
        recent_history = list(self.ensemble_history)[-50:]  # Last 50 decisions
        
        model_performance_scores = {}
        
        for model_type in self.models.keys():
            # Calculate average confidence and signal quality
            confidences = []
            signal_qualities = []
            
            for record in recent_history:
                conf = record['model_confidences'].get(model_type, 0)
                signal = record['model_signals'].get(model_type, 0)
                
                confidences.append(conf)
                signal_qualities.append(abs(signal))  # Signal strength
            
            avg_confidence = np.mean(confidences)
            avg_signal_quality = np.mean(signal_qualities)
            
            # Combined performance score
            performance_score = (avg_confidence * 0.6 + avg_signal_quality * 0.4)
            model_performance_scores[model_type] = performance_score
        
        # Normalize performance scores to create new weights
        total_performance = sum(model_performance_scores.values())
        if total_performance > 0:
            new_weights = {}
            for model_type, score in model_performance_scores.items():
                new_weights[model_type] = score / total_performance
            
            # Smooth transition (blend with current weights)
            alpha = 0.3  # Learning rate
            for model_type in self.ensemble_weights.keys():
                old_weight = self.ensemble_weights[model_type]
                new_weight = new_weights.get(model_type, old_weight)
                self.ensemble_weights[model_type] = (
                    (1 - alpha) * old_weight + alpha * new_weight
                )
            
            logger.info(f"✅ Updated ensemble weights: {dict(self.ensemble_weights)}")
        
        # Log performance summary
        for model_type, score in model_performance_scores.items():
            logger.info(f"   {model_type.value}: {score:.3f} (weight: {self.ensemble_weights[model_type]:.3f})")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        if not self.ensemble_history:
            return {}
        
        recent_decisions = list(self.ensemble_history)[-100:]
        
        # Calculate ensemble metrics
        avg_confidence = np.mean([r['decision'].confidence for r in recent_decisions])
        avg_processing_time = np.mean([r['processing_time'] for r in recent_decisions])
        
        # Model contribution analysis
        model_contributions = defaultdict(int)
        for record in recent_decisions:
            for model_type in record['decision'].contributing_models:
                model_contributions[model_type] += 1
        
        return {
            'total_decisions': self.decision_count,
            'recent_decisions': len(recent_decisions),
            'average_confidence': avg_confidence,
            'average_processing_time_ms': avg_processing_time * 1000,
            'current_weights': dict(self.ensemble_weights),
            'model_contributions': dict(model_contributions),
            'ensemble_history_size': len(self.ensemble_history)
        }

# Factory function
def create_cognitive_ensemble_engine(config: Dict[str, Any]) -> CognitiveEnsembleEngine:
    """Create and initialize cognitive ensemble engine"""
    return CognitiveEnsembleEngine(config) 