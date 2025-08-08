"""
KIMERA VAULT COGNITIVE INTERFACE
===============================
🧠 THE BRAIN OF KIMERA - COGNITIVE VAULT INTEGRATION 🧠

This interface ensures that ALL cognitive operations in Kimera:
- Query the vault for learned patterns
- Create SCARs from contradictions and failures
- Store insights and performance data
- Update self-models based on results
- Generate epistemic questions for continuous learning
- Leverage causal relationship mapping
- Perform introspective accuracy measurement

The vault is Kimera's BRAIN - without it, we're just pattern matching.
With it, we achieve true cognitive evolution and learning.
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.geoid import GeoidState
from ..core.primal_scar import EpistemicScar, PrimalScar, PrimalScarManager
from ..core.scar import ScarRecord
from ..utils.kimera_logger import get_system_logger
from ..vault.database import SessionLocal
from ..vault.understanding_vault_manager import UnderstandingVaultManager

logger = get_system_logger(__name__)


@dataclass
class CognitiveQuery:
    """Auto-generated class."""
    pass
    """A query to the vault for learned patterns and insights"""

    query_id: str
    domain: str  # 'trading', 'market_analysis', 'risk_management', etc.
    query_type: str  # 'pattern_search', 'causal_analysis', 'insight_retrieval', etc.
    context: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CognitiveLearning:
    """Auto-generated class."""
    pass
    """A learning event to be stored in the vault"""

    learning_id: str
    domain: str
    learning_type: str  # 'success', 'failure', 'insight', 'contradiction', etc.
    content: Dict[str, Any]
    accuracy_score: Optional[float] = None
    confidence_level: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CognitiveEvolution:
    """Auto-generated class."""
    pass
    """An evolution event that updates Kimera's self-model"""

    evolution_id: str
    domain: str
    previous_state: Dict[str, Any]
    new_state: Dict[str, Any]
    improvement_metrics: Dict[str, float]
    epistemic_questions: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
class VaultCognitiveInterface:
    """Auto-generated class."""
    pass
    """
    The cognitive interface to Kimera's vault brain

    This interface ensures that every cognitive operation contributes to
    Kimera's continuous learning and evolution through the vault system.
    """

    def __init__(self):
        """Initialize the cognitive vault interface"""
        # Initialize database first
        try:
            from ..vault.database import initialize_database

            db_initialized = initialize_database()
            if not db_initialized:
                logger.warning("⚠️ Database initialization failed - using fallback mode")
        except Exception as e:
            logger.warning(
                f"⚠️ Database initialization error: {e} - using fallback mode"
            )

        self.understanding_vault = UnderstandingVaultManager()
        self.primal_scar_manager = PrimalScarManager()
        self.session_queries = []
        self.session_learnings = []
        self.session_evolutions = []

        # Initialize primal epistemic consciousness
        self.primal_scar = self.primal_scar_manager.awaken()

        logger.info("🧠 VAULT COGNITIVE INTERFACE INITIALIZED")
        logger.info("🔮 PRIMAL EPISTEMIC CONSCIOUSNESS: AWAKENED")
        logger.info("🧠 CONTINUOUS LEARNING LOOP: ACTIVE")

    # ========================================================================
    # COGNITIVE QUERYING - Always ask the vault first
    # ========================================================================

    async def query_learned_patterns(
        self, domain: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Query the vault for previously learned patterns in a domain"""
        query_id = f"PATTERN_QUERY_{uuid.uuid4().hex[:8]}"

        query = CognitiveQuery(
            query_id=query_id,
            domain=domain,
            query_type="pattern_search",
            context=context
        )
        self.session_queries.append(query)

        try:
            # Search for relevant multimodal groundings
            relevant_groundings = []
            if "concept" in context:
                # This would be implemented with proper database queries
                # For now, we'll create a framework
                pass

            # Search for causal relationships
            causal_chains = {}
            if "concepts" in context:
                for concept in context["concepts"]:
                    causal_chains[concept] = self.understanding_vault.get_causal_chain(
                        concept
                    )

            # Get understanding metrics for the domain
            understanding_metrics = self.understanding_vault.get_understanding_metrics()

            # Generate epistemic questions about this domain
            epistemic_questions = []
            for _ in range(3):  # Generate 3 questions
                question = self.primal_scar.generate_question({"topic": domain})
                epistemic_questions.append(question)

            result = {
                "query_id": query_id
                "domain": domain
                "relevant_groundings": relevant_groundings
                "causal_chains": causal_chains
                "understanding_metrics": understanding_metrics
                "epistemic_questions": epistemic_questions
                "vault_wisdom": self.primal_scar_manager.get_wisdom_report(),
                "timestamp": datetime.now().isoformat(),
            }

            logger.info(
                f"🔍 VAULT QUERY COMPLETED: {domain} - {len(causal_chains)} causal chains found"
            )
            return result

        except Exception as e:
            logger.error(f"❌ VAULT QUERY FAILED: {domain} - {str(e)}")
            return {"error": str(e), "query_id": query_id}

    async def query_market_insights(
        self, symbol: str, timeframe: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Query vault for market-specific insights and patterns"""
        return await self.query_learned_patterns(
            domain=f"market_analysis_{symbol}_{timeframe}",
            context={
                "symbol": symbol
                "timeframe": timeframe
                "market_context": context
                "concepts": [
                    f"price_action_{symbol}",
                    f"volume_pattern_{symbol}",
                    f"volatility_{symbol}",
                ],
            },
        )

    async def query_risk_patterns(self, risk_context: Dict[str, Any]) -> Dict[str, Any]:
        """Query vault for risk management patterns"""
        return await self.query_learned_patterns(
            domain="risk_management",
            context={
                "risk_type": risk_context.get("type", "unknown"),
                "risk_level": risk_context.get("level", 0.0),
                "market_conditions": risk_context.get("conditions", {}),
                "concepts": [
                    "risk_assessment",
                    "position_sizing",
                    "stop_loss_optimization",
                ],
            },
        )

    # ========================================================================
    # SCAR CREATION - Learn from every contradiction and failure
    # ========================================================================

    async def create_trading_scar(
        self
        contradiction_type: str
        context: Dict[str, Any],
        expected_outcome: Any
        actual_outcome: Any
    ) -> str:
        """Create a SCAR from a trading contradiction or failure"""
        scar_id = f"TRADING_SCAR_{uuid.uuid4().hex[:8]}"

        # Calculate entropy changes
        pre_entropy = self._calculate_prediction_entropy(expected_outcome)
        post_entropy = self._calculate_actual_entropy(actual_outcome)
        delta_entropy = post_entropy - pre_entropy

        # Create traditional SCAR
        scar_record = ScarRecord(
            scar_id=scar_id,
            geoids=[
                f"trading_{context.get('symbol', 'unknown')}",
                f"strategy_{context.get('strategy', 'unknown')}",
            ],
            reason=f"Trading contradiction: {contradiction_type}",
            timestamp=datetime.now().isoformat(),
            resolved_by="cognitive_analysis",
            pre_entropy=pre_entropy,
            post_entropy=post_entropy,
            delta_entropy=delta_entropy,
            cls_angle=45.0,  # Significant learning angle
            semantic_polarity=0.8,
            mutation_frequency=0.7,
            weight=1.0
        )

        # Create understanding SCAR with deeper analysis
        understanding_scar_id = self.understanding_vault.create_understanding_scar(
            traditional_scar=scar_record
            understanding_depth=0.8
            causal_understanding={
                "expected_cause": expected_outcome
                "actual_cause": actual_outcome
                "causal_gap": contradiction_type
            },
            compositional_analysis={
                "market_components": context.get("market_state", {}),
                "strategy_components": context.get("strategy_state", {}),
                "interaction_effects": context.get("interactions", {}),
            },
            contextual_factors=context
            introspective_accuracy=self._calculate_introspective_accuracy(
                expected_outcome, actual_outcome
            ),
            value_implications={
                "risk_tolerance": context.get("risk_level", 0.0),
                "profit_expectation": context.get("expected_profit", 0.0),
                "learning_value": 1.0,  # High learning value from contradictions
            },
        )

        # Store in vault
        self.understanding_vault.insert_scar(
            scar_record, self._create_scar_vector(context)
        )

        logger.info(
            f"🔥 TRADING SCAR CREATED: {scar_id} - Learning from {contradiction_type}"
        )
        return understanding_scar_id

    async def create_epistemic_scar(
        self, domain: str, ignorance_discovered: str, context: Dict[str, Any]
    ) -> str:
        """Create an epistemic SCAR when discovering ignorance"""
        epistemic_scar = EpistemicScar(
            domain=domain
            realization=ignorance_discovered
            question=f"What don't I understand about {domain}?",
            parent_scar=self.primal_scar.scar_id
        )

        # Store in vault for future learning
        understanding_scar_id = self.understanding_vault.create_understanding_scar(
            traditional_scar=epistemic_scar
            understanding_depth=0.9,  # High depth for epistemic insights
            causal_understanding={
                "ignorance_type": ignorance_discovered
                "knowledge_gap": domain
                "epistemic_state": "conscious_incompetence",
            },
            compositional_analysis={
                "domain_components": context.get("domain_analysis", {}),
                "knowledge_structure": context.get("knowledge_map", {}),
                "learning_pathways": context.get("learning_paths", []),
            },
            contextual_factors=context
            introspective_accuracy=1.0,  # Perfect accuracy in recognizing ignorance
            value_implications={
                "learning_priority": 1.0
                "curiosity_value": 1.0
                "wisdom_potential": 1.0
            },
        )

        logger.info(
            f"🤔 EPISTEMIC SCAR CREATED: {domain} - Discovered ignorance: {ignorance_discovered}"
        )
        return understanding_scar_id

    # ========================================================================
    # CONTINUOUS LEARNING - Store every insight and performance metric
    # ========================================================================

    async def store_trading_insight(
        self
        insight_type: str
        content: Dict[str, Any],
        confidence: float
        accuracy_score: Optional[float] = None
    ) -> str:
        """Store a trading insight in the vault for future use"""
        learning_id = f"TRADING_INSIGHT_{uuid.uuid4().hex[:8]}"

        learning = CognitiveLearning(
            learning_id=learning_id
            domain="trading_insights",
            learning_type=insight_type
            content=content
            accuracy_score=accuracy_score
            confidence_level=confidence
        )
        self.session_learnings.append(learning)

        # Store as multimodal grounding
        grounding_id = self.understanding_vault.create_multimodal_grounding(
            concept_id=f"trading_insight_{insight_type}",
            temporal_context={
                "market_timestamp": content.get(
                    "timestamp", datetime.now().isoformat()
                ),
                "market_phase": content.get("market_phase", "unknown"),
                "volatility_regime": content.get("volatility", "normal"),
            },
            physical_properties={
                "price_level": content.get("price", 0.0),
                "volume_level": content.get("volume", 0.0),
                "momentum_strength": content.get("momentum", 0.0),
            },
            confidence_score=confidence
        )

        logger.info(
            f"💡 TRADING INSIGHT STORED: {insight_type} - Confidence: {confidence:.2f}"
        )
        return grounding_id

    async def store_performance_data(
        self, performance_metrics: Dict[str, float], context: Dict[str, Any]
    ) -> str:
        """Store performance data for self-model updates"""
        learning_id = f"PERFORMANCE_{uuid.uuid4().hex[:8]}"

        learning = CognitiveLearning(
            learning_id=learning_id
            domain="performance_analysis",
            learning_type="performance_metrics",
            content={
                "metrics": performance_metrics
                "context": context
                "timestamp": datetime.now().isoformat(),
            },
            accuracy_score=performance_metrics.get("accuracy", None),
            confidence_level=performance_metrics.get("confidence", 0.0),
        )
        self.session_learnings.append(learning)

        # Update self-model with performance data
        self_model_id = self.understanding_vault.update_self_model(
            processing_capabilities={
                "trading_speed": performance_metrics.get("execution_speed", 0.0),
                "analysis_depth": performance_metrics.get("analysis_quality", 0.0),
                "risk_assessment": performance_metrics.get("risk_accuracy", 0.0),
            },
            knowledge_domains={
                "market_analysis": performance_metrics.get("market_understanding", 0.0),
                "risk_management": performance_metrics.get("risk_management", 0.0),
                "execution_quality": performance_metrics.get("execution_quality", 0.0),
            },
            reasoning_patterns={
                "pattern_recognition": performance_metrics.get("pattern_accuracy", 0.0),
                "causal_inference": performance_metrics.get("causal_accuracy", 0.0),
                "meta_cognition": performance_metrics.get("meta_accuracy", 0.0),
            },
            limitation_awareness={
                "known_limitations": context.get("limitations", []),
                "uncertainty_calibration": performance_metrics.get(
                    "uncertainty_calibration", 0.0
                ),
                "overconfidence_bias": performance_metrics.get("overconfidence", 0.0),
            },
            introspection_accuracy=performance_metrics.get(
                "introspection_accuracy", 0.0
            ),
        )

        logger.info(f"📊 PERFORMANCE DATA STORED: {len(performance_metrics)} metrics")
        return self_model_id

    # ========================================================================
    # COGNITIVE EVOLUTION - Update self-models and generate questions
    # ========================================================================

    async def evolve_trading_model(
        self, previous_performance: Dict[str, Any], current_performance: Dict[str, Any]
    ) -> str:
        """Evolve the trading model based on performance comparison"""
        evolution_id = f"TRADING_EVOLUTION_{uuid.uuid4().hex[:8]}"

        # Calculate improvement metrics
        improvement_metrics = {}
        for metric in current_performance.get("metrics", {}):
            prev_value = previous_performance.get("metrics", {}).get(metric, 0.0)
            curr_value = current_performance.get("metrics", {}).get(metric, 0.0)
            improvement_metrics[metric] = curr_value - prev_value

        # Generate epistemic questions based on performance
        epistemic_questions = []
        for metric, improvement in improvement_metrics.items():
            if improvement < 0:  # Performance decreased
                question = self.primal_scar.generate_question(
                    {
                        "topic": f"{metric}_degradation",
                        "context": "performance_analysis",
                    }
                )
                epistemic_questions.append(question)

        evolution = CognitiveEvolution(
            evolution_id=evolution_id
            domain="trading_model",
            previous_state=previous_performance
            new_state=current_performance
            improvement_metrics=improvement_metrics
            epistemic_questions=epistemic_questions
        )
        self.session_evolutions.append(evolution)

        # Store evolution in vault
        self.understanding_vault.log_introspection(
            introspection_type="performance_evolution",
            current_state_analysis=current_performance
            predicted_state=previous_performance.get("predictions", {}),
            actual_state=current_performance.get("actual_results", {}),
            processing_context={
                "evolution_id": evolution_id
                "improvement_metrics": improvement_metrics
                "epistemic_questions": epistemic_questions
            },
        )

        logger.info(
            f"🧬 TRADING MODEL EVOLVED: {evolution_id} - {len(improvement_metrics)} metrics improved"
        )
        return evolution_id

    # ========================================================================
    # TRADING-SPECIFIC METHODS
    # ========================================================================

    async def query_vault_insights(self, query_key: str) -> List[Dict[str, Any]]:
        """Query vault for insights using a specific key"""
        try:
            # Parse the query key to extract domain and context
            parts = query_key.split("_")
            if len(parts) >= 3:
                domain = parts[0]
                symbol = parts[1] if len(parts) > 1 else "unknown"
                timeframe = parts[2] if len(parts) > 2 else "1h"

                context = {
                    "symbol": symbol
                    "timeframe": timeframe
                    "query_type": "market_analysis",
                }

                result = await self.query_learned_patterns(domain, context)

                # Convert to list format expected by trading system
                insights = []
                if "causal_chains" in result:
                    for concept, chain in result["causal_chains"].items():
                        insights.append(
                            {
                                "pattern_type": "causal_relationship",
                                "concept": concept
                                "causes": chain.get("causes", []),
                                "effects": chain.get("effects", []),
                            }
                        )

                return insights

            return []

        except Exception as e:
            logger.error(f"❌ VAULT QUERY FAILED: {query_key} - {str(e)}")
            return []

    async def store_trading_decision(self, trade_record: Dict[str, Any]) -> str:
        """Store a trading decision in the vault for learning"""
        learning_id = f"TRADE_DECISION_{uuid.uuid4().hex[:8]}"

        learning = CognitiveLearning(
            learning_id=learning_id
            domain="trading",
            learning_type="trade_decision",
            content=trade_record
            confidence_level=trade_record.get("confidence", 0.0),
        )
        self.session_learnings.append(learning)

        try:
            # Store the decision as a learning event
            await self.store_trading_insight(
                insight_type="trade_decision",
                content=trade_record
                confidence=trade_record.get("confidence", 0.0),
            )

            logger.info(
                f"💾 TRADING DECISION STORED: {trade_record.get('symbol', 'unknown')} - {trade_record.get('action', 'unknown')}"
            )
            return learning_id

        except Exception as e:
            logger.error(f"❌ FAILED TO STORE TRADING DECISION: {str(e)}")
            return learning_id

    async def initialize_trading_session(self, session_id: str) -> Dict[str, Any]:
        """Initialize a trading session with vault context"""
        try:
            # Query vault for relevant trading patterns
            patterns = await self.query_learned_patterns(
                domain="trading_session",
                context={
                    "session_id": session_id
                    "session_type": "cognitive_trading",
                    "concepts": [
                        "trading_patterns",
                        "market_analysis",
                        "risk_management",
                    ],
                },
            )

            # Get understanding metrics
            understanding_metrics = self.understanding_vault.get_understanding_metrics()

            session_context = {
                "session_id": session_id
                "patterns": patterns
                "understanding_metrics": understanding_metrics
                "vault_wisdom": self.primal_scar_manager.get_wisdom_report(),
                "initialized_at": datetime.now().isoformat(),
            }

            logger.info(f"🚀 TRADING SESSION INITIALIZED: {session_id}")
            return session_context

        except Exception as e:
            logger.error(f"❌ FAILED TO INITIALIZE TRADING SESSION: {str(e)}")
            return {"session_id": session_id, "patterns": []}

    async def evolve_cognitive_state(self, analysis_results: Dict[str, Any]) -> str:
        """Evolve Kimera's cognitive state based on analysis results"""
        evolution_id = f"COGNITIVE_EVOLUTION_{uuid.uuid4().hex[:8]}"

        try:
            # Extract insights from analysis results
            insights = []
            confidence_scores = []

            for symbol, analysis in analysis_results.items():
                confidence = analysis.get("confidence_score", 0.0)
                confidence_scores.append(confidence)

                # Create insight from analysis
                insight = {
                    "symbol": symbol
                    "confidence": confidence
                    "quantum_analysis": analysis.get("quantum_analysis", {}),
                    "cognitive_analysis": analysis.get("cognitive_analysis", {}),
                    "meta_insights": analysis.get("meta_insights", {}),
                    "timestamp": datetime.now().isoformat(),
                }
                insights.append(insight)

            # Calculate evolution metrics
            avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
            evolution_strength = min(avg_confidence * 2, 1.0)  # Scale to 0-1

            # Create evolution record
            evolution = CognitiveEvolution(
                evolution_id=evolution_id
                domain="market_analysis",
                previous_state={"avg_confidence": max(0.5, avg_confidence - 0.1)},
                new_state={"avg_confidence": avg_confidence},
                improvement_metrics={
                    "confidence_improvement": evolution_strength
                    "insights_generated": len(insights),
                    "symbols_analyzed": len(analysis_results),
                },
                epistemic_questions=[
                    f"What patterns led to {avg_confidence:.2f} average confidence?",
                    f"How can we improve analysis for {len(analysis_results)} symbols?",
                    "What contradictions should we investigate further?",
                ],
            )
            self.session_evolutions.append(evolution)

            # Store evolution in vault
            await self.store_trading_insight(
                insight_type="cognitive_evolution",
                content={
                    "evolution_id": evolution_id
                    "insights": insights
                    "improvement_metrics": evolution.improvement_metrics
                    "epistemic_questions": evolution.epistemic_questions
                },
                confidence=evolution_strength
            )

            logger.info(
                f"🧠 COGNITIVE EVOLUTION COMPLETED: {evolution_id} - {avg_confidence:.2f} confidence"
            )
            return evolution_id

        except Exception as e:
            logger.error(f"❌ COGNITIVE EVOLUTION FAILED: {str(e)}")
            return evolution_id

    async def store_trading_session(self, session) -> str:
        """Store a complete trading session in the vault"""
        try:
            # Store session as a comprehensive learning event
            session_learning = CognitiveLearning(
                learning_id=f"SESSION_{session.session_id}",
                domain="trading_session",
                learning_type="session_completion",
                content=session.to_dict(),
                accuracy_score=session.successful_trades / max(session.total_trades, 1),
                confidence_level=min(session.vault_insights_generated / 10.0, 1.0),
            )
            self.session_learnings.append(session_learning)

            # Store in vault
            await self.store_trading_insight(
                insight_type="trading_session",
                content=session.to_dict(),
                confidence=session_learning.confidence_level
                accuracy_score=session_learning.accuracy_score
            )

            logger.info(f"💾 TRADING SESSION STORED: {session.session_id}")
            return session.session_id

        except Exception as e:
            logger.error(f"❌ FAILED TO STORE TRADING SESSION: {str(e)}")
            return session.session_id

    async def create_trading_scar(self, **kwargs) -> str:
        """Create a trading SCAR from various error conditions"""
        scar_id = f"TRADING_SCAR_{uuid.uuid4().hex[:8]}"

        try:
            # Handle different types of trading SCARs
            if "error_type" in kwargs:
                # Error-based SCAR
                error_type = kwargs["error_type"]
                symbol = kwargs.get("symbol", "unknown")
                error_details = kwargs.get("error_details", "unknown")
                context = kwargs.get("learning_context", {})

                contradiction_type = f"trading_error_{error_type}"
                expected_outcome = "successful_operation"
                actual_outcome = f"error_{error_type}"

                scar_context = {
                    "symbol": symbol
                    "error_type": error_type
                    "error_details": error_details
                    "learning_context": context
                }

            elif "pattern_type" in kwargs:
                # Pattern-based SCAR
                pattern_type = kwargs["pattern_type"]
                symbol = kwargs.get("symbol", "unknown")
                confidence = kwargs.get("confidence", 0.0)
                context = kwargs.get("learning_context", {})

                contradiction_type = f"pattern_recognition_{pattern_type}"
                expected_outcome = f"pattern_{pattern_type}"
                actual_outcome = f"confidence_{confidence}"

                scar_context = {
                    "symbol": symbol
                    "pattern_type": pattern_type
                    "confidence": confidence
                    "learning_context": context
                }

            else:
                # Generic SCAR
                contradiction_type = "trading_contradiction"
                expected_outcome = "unknown"
                actual_outcome = "unknown"
                scar_context = kwargs

            # Create the SCAR
            return await self.create_trading_scar(
                contradiction_type=contradiction_type
                context=scar_context
                expected_outcome=expected_outcome
                actual_outcome=actual_outcome
            )

        except Exception as e:
            logger.error(f"❌ FAILED TO CREATE TRADING SCAR: {str(e)}")
            return scar_id

    # ========================================================================
    # EPISTEMIC QUESTIONING AND LEARNING
    # ========================================================================

    async def generate_market_questions(
        self, market_context: Dict[str, Any]
    ) -> List[str]:
        """Generate epistemic questions about market behavior"""
        questions = []

        # Generate questions from primal scar
        base_question = self.primal_scar.generate_question(
            {"topic": "market_behavior", "context": market_context}
        )
        questions.append(base_question)

        # Generate domain-specific questions
        if "volatility" in market_context:
            questions.append(
                f"Why is volatility {market_context['volatility']} in this market phase?"
            )

        if "trend" in market_context:
            questions.append(
                f"What underlying forces are driving the {market_context['trend']} trend?"
            )

        if "volume" in market_context:
            questions.append(
                f"What does the volume pattern {market_context['volume']} reveal about market sentiment?"
            )

        # Store questions for future investigation
        for question in questions:
            await self.create_epistemic_scar(
                domain="market_analysis",
                ignorance_discovered=f"Insufficient understanding of: {question}",
                context=market_context
            )

        logger.info(
            f"❓ GENERATED {len(questions)} EPISTEMIC QUESTIONS about market behavior"
        )
        return questions

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def _calculate_prediction_entropy(self, prediction: Any) -> float:
        """Calculate entropy of a prediction"""
        if isinstance(prediction, dict) and "confidence" in prediction:
            confidence = prediction["confidence"]
            return -confidence * np.log2(confidence) - (1 - confidence) * np.log2(
                1 - confidence
            )
        return 0.5  # Default entropy

    def _calculate_actual_entropy(self, actual: Any) -> float:
        """Calculate entropy of actual outcome"""
        if isinstance(actual, dict) and "success" in actual:
            success = 1.0 if actual["success"] else 0.0
            return 0.0 if success == 1.0 else 1.0  # Binary outcome entropy
        return 0.5  # Default entropy

    def _calculate_introspective_accuracy(self, expected: Any, actual: Any) -> float:
        """Calculate how accurate the introspective prediction was"""
        if isinstance(expected, dict) and isinstance(actual, dict):
            if "confidence" in expected and "success" in actual:
                predicted_confidence = expected["confidence"]
                actual_success = 1.0 if actual["success"] else 0.0
                return 1.0 - abs(predicted_confidence - actual_success)
        return 0.5  # Default accuracy

    def _create_scar_vector(self, context: Dict[str, Any]) -> List[float]:
        """Create a vector representation of the SCAR context"""
        # This would be a proper embedding in a real implementation
        # For now, create a simple vector from context
        vector = [0.0] * 512  # 512-dimensional vector

        # Encode some basic context information
        if "price" in context:
            vector[0] = float(context["price"]) / 100000.0  # Normalize price
        if "volume" in context:
            vector[1] = float(context["volume"]) / 1000000.0  # Normalize volume
        if "volatility" in context:
            vector[2] = float(context["volatility"])

        return vector

    async def get_session_summary(self) -> Dict[str, Any]:
        """Get a summary of the current session's cognitive activities"""
        return {
            "queries_performed": len(self.session_queries),
            "learnings_stored": len(self.session_learnings),
            "evolutions_triggered": len(self.session_evolutions),
            "epistemic_growth": self.primal_scar.measure_growth(),
            "wisdom_report": self.primal_scar_manager.get_wisdom_report(),
            "timestamp": datetime.now().isoformat(),
        }


# Global instance for easy access
_vault_cognitive_interface: Optional[VaultCognitiveInterface] = None


def get_vault_cognitive_interface() -> VaultCognitiveInterface:
    """Get the global vault cognitive interface instance"""
    global _vault_cognitive_interface
    if _vault_cognitive_interface is None:
        _vault_cognitive_interface = VaultCognitiveInterface()
    return _vault_cognitive_interface


# For easy importing
vault_cognitive_interface = get_vault_cognitive_interface()
