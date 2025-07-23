"""
Smart Order Routing (SOR) System for Kimera SWM

AI-powered order routing that optimizes execution across multiple venues
while minimizing market impact and maximizing execution quality.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict, deque
import random

# Machine learning imports
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logging.warning("ML libraries not available. Using rule-based routing.")

# Local imports
from src.core.geoid import GeoidState as Geoid
from src.engines.cognitive_field_dynamics import CognitiveFieldDynamics as CognitiveFieldDynamicsEngine
from src.engines.thermodynamic_engine import ThermodynamicEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Venue:
    """Trading venue information"""
    venue_id: str
    venue_type: str  # 'exchange', 'dark_pool', 'ecn', 'market_maker'
    name: str
    average_latency: float  # microseconds
    liquidity_score: float  # 0-1
    fee_structure: Dict[str, float]
    supported_order_types: List[str]
    is_active: bool = True
    
    
@dataclass
class RoutingDecision:
    """Smart order routing decision"""
    order_id: str
    selected_venue: str
    venue_allocations: Dict[str, float]  # venue_id -> percentage
    expected_slippage: float
    expected_latency: float
    expected_cost: float
    routing_strategy: str
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    
@dataclass
class ExecutionMetrics:
    """Post-execution metrics for learning"""
    order_id: str
    venue_id: str
    actual_slippage: float
    actual_latency: float
    actual_cost: float
    fill_rate: float
    timestamp: datetime


class SmartOrderRouter:
    """
    Smart Order Routing System
    
    Features:
    - ML-based venue selection
    - Multi-venue order splitting
    - Real-time latency monitoring
    - Dark pool integration
    - Adaptive routing strategies
    """
    
    def __init__(self):
        """Initialize the smart order router"""
        self.venues = {}
        self.performance_metrics = {}
        self.routing_decisions = []
        
        # Defer async task creation to avoid event loop issues
        self._background_tasks_started = False
        
        logger.info("Smart Order Router initialized")
    
    async def start_background_tasks(self):
        """Start background monitoring tasks"""
        if not self._background_tasks_started:
            asyncio.create_task(self._monitor_venue_performance())
            self._background_tasks_started = True
            logger.info("Background tasks started")
        
    def _initialize_venues(self) -> Dict[str, Venue]:
        """Initialize available trading venues"""
        venues = {
            'binance': Venue(
                venue_id='binance',
                venue_type='exchange',
                name='Binance',
                average_latency=50.0,
                liquidity_score=0.95,
                fee_structure={'maker': 0.001, 'taker': 0.001},
                supported_order_types=['limit', 'market', 'stop']
            ),
            'coinbase': Venue(
                venue_id='coinbase',
                venue_type='exchange',
                name='Coinbase Pro',
                average_latency=75.0,
                liquidity_score=0.85,
                fee_structure={'maker': 0.005, 'taker': 0.005},
                supported_order_types=['limit', 'market']
            ),
            'kraken': Venue(
                venue_id='kraken',
                venue_type='exchange',
                name='Kraken',
                average_latency=100.0,
                liquidity_score=0.80,
                fee_structure={'maker': 0.0016, 'taker': 0.0026},
                supported_order_types=['limit', 'market', 'stop']
            ),
            'dark_pool_1': Venue(
                venue_id='dark_pool_1',
                venue_type='dark_pool',
                name='Anonymous Dark Pool',
                average_latency=30.0,
                liquidity_score=0.60,
                fee_structure={'maker': 0.0005, 'taker': 0.0008},
                supported_order_types=['limit', 'iceberg']
            ),
            'mm_1': Venue(
                venue_id='mm_1',
                venue_type='market_maker',
                name='Market Maker 1',
                average_latency=10.0,
                liquidity_score=0.70,
                fee_structure={'maker': -0.0001, 'taker': 0.0015},
                supported_order_types=['limit', 'market']
            )
        }
        
        return venues
        
    def _initialize_ml_models(self):
        """Initialize ML models for venue selection"""
        # Model for each venue type
        for venue_type in ['exchange', 'dark_pool', 'market_maker']:
            self.venue_models[venue_type] = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            self.feature_scalers[venue_type] = StandardScaler()
            
    async def route_order(self, order: Dict[str, Any]) -> RoutingDecision:
        """
        Route order to optimal venue(s)
        
        Args:
            order: Order details including symbol, side, quantity, order_type
            
        Returns:
            Routing decision with venue selection
        """
        try:
            # Extract order features
            features = self._extract_order_features(order)
            
            # Select routing strategy
            strategy = self._select_routing_strategy(order, features)
            
            # Execute routing strategy
            routing_func = self.routing_strategies.get(strategy, self._smart_routing)
            decision = await routing_func(order, features)
            
            # Integrate with cognitive field
            if self.cognitive_field:
                await self._cognitive_routing_adjustment(decision, order)
                
            # Track routing
            self.orders_routed += 1
            
            return decision
            
        except Exception as e:
            logger.error(f"Order routing error: {e}")
            # Fallback to simple routing
            return await self._fallback_routing(order)
            
    def _extract_order_features(self, order: Dict[str, Any]) -> Dict[str, float]:
        """Extract features for ML routing"""
        features = {
            'order_size': order.get('quantity', 0),
            'is_buy': 1.0 if order.get('side') == 'buy' else 0.0,
            'is_market_order': 1.0 if order.get('order_type') == 'market' else 0.0,
            'urgency': order.get('urgency', 0.5),
            'hour_of_day': datetime.now().hour,
            'day_of_week': datetime.now().weekday()
        }
        
        # Add market conditions if available
        if 'market_conditions' in order:
            features.update({
                'volatility': order['market_conditions'].get('volatility', 0.01),
                'spread': order['market_conditions'].get('spread', 0.0001),
                'volume': order['market_conditions'].get('volume', 1000)
            })
            
        return features
        
    def _select_routing_strategy(self, order: Dict[str, Any], features: Dict[str, float]) -> str:
        """Select appropriate routing strategy"""
        # Large orders use iceberg
        if features['order_size'] > 1000:
            return 'iceberg'
            
        # High urgency uses aggressive
        if features.get('urgency', 0.5) > 0.8:
            return 'aggressive'
            
        # Low urgency can be passive
        if features.get('urgency', 0.5) < 0.3:
            return 'passive'
            
        # Check for dark pool preference
        if order.get('prefer_dark', False):
            return 'dark_seeking'
            
        # Default to smart routing
        return 'smart'
        
    async def _smart_routing(self, order: Dict[str, Any], features: Dict[str, float]) -> RoutingDecision:
        """Smart routing using ML predictions"""
        venue_scores = {}
        
        # Score each venue
        for venue_id, venue in self.venues.items():
            if not venue.is_active:
                continue
                
            # Check if venue supports order type
            if order.get('order_type', 'limit') not in venue.supported_order_types:
                continue
                
            # Calculate base score
            score = self._calculate_venue_score(venue, order, features)
            
            # ML adjustment if available
            if ML_AVAILABLE and venue.venue_type in self.venue_models:
                ml_score = await self._get_ml_venue_score(venue, features)
                score = 0.7 * score + 0.3 * ml_score
                
            venue_scores[venue_id] = score
            
        # Select best venue(s)
        if not venue_scores:
            return await self._fallback_routing(order)
            
        # Sort venues by score
        sorted_venues = sorted(venue_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Single venue or split order
        if order.get('quantity', 0) < 100 or len(sorted_venues) == 1:
            # Single venue
            best_venue = sorted_venues[0][0]
            allocations = {best_venue: 1.0}
        else:
            # Split across top venues
            allocations = self._calculate_venue_allocations(sorted_venues, order)
            
        # Calculate expected metrics
        expected_slippage = self._estimate_slippage(allocations, order)
        expected_latency = self._estimate_latency(allocations)
        expected_cost = self._estimate_cost(allocations, order)
        
        return RoutingDecision(
            order_id=order.get('order_id', f"order_{datetime.now().timestamp()}"),
            selected_venue=sorted_venues[0][0],
            venue_allocations=allocations,
            expected_slippage=expected_slippage,
            expected_latency=expected_latency,
            expected_cost=expected_cost,
            routing_strategy='smart',
            confidence=sorted_venues[0][1]
        )
        
    def _calculate_venue_score(self, venue: Venue, order: Dict[str, Any], features: Dict[str, float]) -> float:
        """Calculate venue score based on multiple factors"""
        # Liquidity score (40% weight)
        liquidity_score = venue.liquidity_score
        
        # Latency score (30% weight) - inverse relationship
        latency_score = 1.0 / (1.0 + venue.average_latency / 100.0)
        
        # Cost score (20% weight) - inverse relationship
        fee = venue.fee_structure.get('taker' if features.get('is_market_order') else 'maker', 0.001)
        cost_score = 1.0 / (1.0 + fee * 100)
        
        # Venue type preference (10% weight)
        type_score = 0.5  # Default
        if venue.venue_type == 'exchange' and not order.get('prefer_dark', False):
            type_score = 1.0
        elif venue.venue_type == 'dark_pool' and order.get('prefer_dark', False):
            type_score = 1.0
            
        # Weighted score
        total_score = (
            0.4 * liquidity_score +
            0.3 * latency_score +
            0.2 * cost_score +
            0.1 * type_score
        )
        
        return total_score
        
    async def _get_ml_venue_score(self, venue: Venue, features: Dict[str, float]) -> float:
        """Get ML-based venue score"""
        if venue.venue_type not in self.venue_models:
            return 0.5
            
        try:
            # Prepare features
            feature_vector = np.array([list(features.values())])
            
            # Scale features
            if venue.venue_type in self.feature_scalers:
                feature_vector = self.feature_scalers[venue.venue_type].transform(feature_vector)
                
            # Predict performance
            model = self.venue_models[venue.venue_type]
            if hasattr(model, 'predict'):
                score = model.predict(feature_vector)[0]
                return np.clip(score, 0, 1)
                
        except Exception as e:
            logger.error(f"ML scoring error: {e}")
            
        return 0.5
        
    def _calculate_venue_allocations(self, sorted_venues: List[Tuple[str, float]], order: Dict[str, Any]) -> Dict[str, float]:
        """Calculate order allocation across venues"""
        allocations = {}
        
        # Use top 3 venues
        top_venues = sorted_venues[:3]
        total_score = sum(score for _, score in top_venues)
        
        if total_score == 0:
            # Equal allocation
            for venue_id, _ in top_venues:
                allocations[venue_id] = 1.0 / len(top_venues)
        else:
            # Score-weighted allocation
            for venue_id, score in top_venues:
                allocations[venue_id] = score / total_score
                
        return allocations
        
    async def _aggressive_routing(self, order: Dict[str, Any], features: Dict[str, float]) -> RoutingDecision:
        """Aggressive routing for immediate execution"""
        # Prioritize low latency and high liquidity
        best_venue = None
        best_score = -1
        
        for venue_id, venue in self.venues.items():
            if not venue.is_active:
                continue
                
            # Score based on latency and liquidity only
            score = venue.liquidity_score / (1 + venue.average_latency / 50)
            
            if score > best_score:
                best_score = score
                best_venue = venue_id
                
        if not best_venue:
            return await self._fallback_routing(order)
            
        return RoutingDecision(
            order_id=order.get('order_id', f"order_{datetime.now().timestamp()}"),
            selected_venue=best_venue,
            venue_allocations={best_venue: 1.0},
            expected_slippage=0.001,  # Accept higher slippage
            expected_latency=self.venues[best_venue].average_latency,
            expected_cost=self.venues[best_venue].fee_structure.get('taker', 0.001) * order.get('quantity', 0),
            routing_strategy='aggressive',
            confidence=0.9
        )
        
    async def _passive_routing(self, order: Dict[str, Any], features: Dict[str, float]) -> RoutingDecision:
        """Passive routing to minimize market impact"""
        # Prioritize venues with maker rebates
        best_venue = None
        best_fee = float('inf')
        
        for venue_id, venue in self.venues.items():
            if not venue.is_active:
                continue
                
            maker_fee = venue.fee_structure.get('maker', 0.001)
            if maker_fee < best_fee:
                best_fee = maker_fee
                best_venue = venue_id
                
        if not best_venue:
            return await self._fallback_routing(order)
            
        return RoutingDecision(
            order_id=order.get('order_id', f"order_{datetime.now().timestamp()}"),
            selected_venue=best_venue,
            venue_allocations={best_venue: 1.0},
            expected_slippage=0.0,  # No slippage for passive orders
            expected_latency=self.venues[best_venue].average_latency * 2,  # Slower execution
            expected_cost=best_fee * order.get('quantity', 0),
            routing_strategy='passive',
            confidence=0.8
        )
        
    async def _dark_pool_routing(self, order: Dict[str, Any], features: Dict[str, float]) -> RoutingDecision:
        """Route to dark pools for large orders"""
        dark_pools = [v for v in self.venues.values() if v.venue_type == 'dark_pool' and v.is_active]
        
        if not dark_pools:
            # Fallback to regular venues
            return await self._smart_routing(order, features)
            
        # Select best dark pool
        best_pool = max(dark_pools, key=lambda v: v.liquidity_score)
        
        # Split between dark pool and lit market
        dark_allocation = min(0.7, features['order_size'] / 1000)  # Up to 70% in dark
        lit_allocation = 1.0 - dark_allocation
        
        # Find best lit venue
        lit_venues = [v for v in self.venues.values() if v.venue_type == 'exchange' and v.is_active]
        best_lit = max(lit_venues, key=lambda v: v.liquidity_score) if lit_venues else None
        
        allocations = {best_pool.venue_id: dark_allocation}
        if best_lit:
            allocations[best_lit.venue_id] = lit_allocation
            
        return RoutingDecision(
            order_id=order.get('order_id', f"order_{datetime.now().timestamp()}"),
            selected_venue=best_pool.venue_id,
            venue_allocations=allocations,
            expected_slippage=0.0005,  # Low slippage in dark pools
            expected_latency=best_pool.average_latency,
            expected_cost=sum(self.venues[v].fee_structure.get('taker', 0.001) * alloc * order.get('quantity', 0) 
                            for v, alloc in allocations.items()),
            routing_strategy='dark_seeking',
            confidence=0.85
        )
        
    async def _iceberg_routing(self, order: Dict[str, Any], features: Dict[str, float]) -> RoutingDecision:
        """Iceberg order routing for large orders"""
        # Use multiple venues with small visible quantities
        suitable_venues = [
            v for v in self.venues.values() 
            if v.is_active and 'iceberg' in v.supported_order_types or 'limit' in v.supported_order_types
        ]
        
        if not suitable_venues:
            return await self._fallback_routing(order)
            
        # Distribute across venues
        allocations = {}
        remaining = 1.0
        
        for venue in sorted(suitable_venues, key=lambda v: v.liquidity_score, reverse=True)[:4]:
            allocation = min(0.3, remaining)  # Max 30% per venue
            allocations[venue.venue_id] = allocation
            remaining -= allocation
            
            if remaining <= 0:
                break
                
        return RoutingDecision(
            order_id=order.get('order_id', f"order_{datetime.now().timestamp()}"),
            selected_venue=list(allocations.keys())[0],
            venue_allocations=allocations,
            expected_slippage=0.0002,  # Very low slippage
            expected_latency=100.0,  # Slower execution
            expected_cost=sum(self.venues[v].fee_structure.get('maker', 0.001) * alloc * order.get('quantity', 0) 
                            for v, alloc in allocations.items()),
            routing_strategy='iceberg',
            confidence=0.9
        )
        
    async def _fallback_routing(self, order: Dict[str, Any]) -> RoutingDecision:
        """Fallback routing when other strategies fail"""
        # Route to most liquid venue
        active_venues = [v for v in self.venues.values() if v.is_active]
        if not active_venues:
            raise ValueError("No active venues available")
            
        best_venue = max(active_venues, key=lambda v: v.liquidity_score)
        
        return RoutingDecision(
            order_id=order.get('order_id', f"order_{datetime.now().timestamp()}"),
            selected_venue=best_venue.venue_id,
            venue_allocations={best_venue.venue_id: 1.0},
            expected_slippage=0.001,
            expected_latency=best_venue.average_latency,
            expected_cost=best_venue.fee_structure.get('taker', 0.001) * order.get('quantity', 0),
            routing_strategy='fallback',
            confidence=0.5
        )
        
    def _estimate_slippage(self, allocations: Dict[str, float], order: Dict[str, Any]) -> float:
        """Estimate expected slippage"""
        total_slippage = 0.0
        
        for venue_id, allocation in allocations.items():
            venue = self.venues[venue_id]
            # Base slippage inversely related to liquidity
            base_slippage = 0.001 * (1.0 - venue.liquidity_score)
            
            # Size impact
            size_impact = (order.get('quantity', 0) * allocation) / 10000  # Simplified
            
            total_slippage += (base_slippage + size_impact) * allocation
            
        return total_slippage
        
    def _estimate_latency(self, allocations: Dict[str, float]) -> float:
        """Estimate expected latency"""
        # Weighted average latency
        total_latency = 0.0
        
        for venue_id, allocation in allocations.items():
            venue = self.venues[venue_id]
            total_latency += venue.average_latency * allocation
            
        return total_latency
        
    def _estimate_cost(self, allocations: Dict[str, float], order: Dict[str, Any]) -> float:
        """Estimate total cost including fees"""
        total_cost = 0.0
        quantity = order.get('quantity', 0)
        
        for venue_id, allocation in allocations.items():
            venue = self.venues[venue_id]
            fee_type = 'taker' if order.get('order_type') == 'market' else 'maker'
            fee = venue.fee_structure.get(fee_type, 0.001)
            total_cost += fee * quantity * allocation
            
        return total_cost
        
    async def _cognitive_routing_adjustment(self, decision: RoutingDecision, order: Dict[str, Any]):
        """Adjust routing decision based on cognitive field"""
        if not self.cognitive_field:
            return
            
        # Create routing geoid
        routing_geoid = Geoid(
            semantic_features={
                'type': 'routing_decision',
                'venue': decision.selected_venue,
                'strategy': decision.routing_strategy,
                'order_size': order.get('quantity', 0)
            },
            symbolic_content=f"Route to {decision.selected_venue}"
        )
        
        # Check cognitive coherence
        coherence = await self.cognitive_field.calculate_coherence(routing_geoid)
        
        # Adjust confidence based on coherence
        decision.confidence *= coherence
        
    async def record_execution(self, metrics: ExecutionMetrics):
        """Record actual execution metrics for learning"""
        self.execution_history.append(metrics)
        self.venue_performance[metrics.venue_id].append(metrics)
        
        # Update ML models if enough data
        if len(self.execution_history) > 100 and ML_AVAILABLE:
            await self._update_ml_models()
            
        # Update venue statistics
        venue = self.venues.get(metrics.venue_id)
        if venue:
            # Update average latency
            recent_latencies = [m.actual_latency for m in self.venue_performance[metrics.venue_id][-100:]]
            venue.average_latency = np.mean(recent_latencies) if recent_latencies else venue.average_latency
            
    async def _update_ml_models(self):
        """Update ML models with recent execution data"""
        # This would retrain models with recent data
        # Placeholder for actual implementation
        pass
        
    async def _monitor_venue_performance(self):
        """Monitor venue performance continuously"""
        while self.running:
            try:
                for venue_id, venue in self.venues.items():
                    # Check recent performance
                    recent_metrics = list(self.venue_performance[venue_id])[-50:]
                    
                    if recent_metrics:
                        # Update liquidity score based on fill rates
                        avg_fill_rate = np.mean([m.fill_rate for m in recent_metrics])
                        venue.liquidity_score = 0.7 * venue.liquidity_score + 0.3 * avg_fill_rate
                        
                        # Check if venue should be deactivated
                        if avg_fill_rate < 0.5 or venue.average_latency > 500:
                            logger.warning(f"Poor performance detected for {venue_id}")
                            
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                logger.error(f"Venue monitoring error: {e}")
                
    async def optimize_routing(self):
        """Optimize routing algorithms"""
        # Analyze recent routing decisions
        if self.orders_routed > 0:
            avg_slippage = self.total_slippage / self.orders_routed
            fill_rate = self.successful_fills / self.orders_routed
            
            logger.info(f"Routing performance - Avg slippage: {avg_slippage:.4f}, Fill rate: {fill_rate:.2%}")
            
        # Update venue preferences based on performance
        for venue_id, metrics in self.venue_performance.items():
            if len(metrics) > 10:
                recent = list(metrics)[-10:]
                avg_slippage = np.mean([m.actual_slippage for m in recent])
                
                # Adjust venue score if consistently bad
                if avg_slippage > 0.002:
                    self.venues[venue_id].liquidity_score *= 0.95
                    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get SOR performance metrics"""
        active_venues = sum(1 for v in self.venues.values() if v.is_active)
        
        return {
            'orders_routed': self.orders_routed,
            'average_slippage': self.total_slippage / max(self.orders_routed, 1),
            'fill_rate': self.successful_fills / max(self.orders_routed, 1),
            'active_venues': active_venues,
            'total_venues': len(self.venues),
            'ml_enabled': ML_AVAILABLE,
            'routing_strategies': list(self.routing_strategies.keys())
        }
        
    def shutdown(self):
        """Shutdown SOR"""
        self.running = False


def create_smart_order_router(cognitive_field=None,
                            thermodynamic_engine=None) -> SmartOrderRouter:
    """Factory function to create Smart Order Router"""
    return SmartOrderRouter(
        cognitive_field=cognitive_field,
        thermodynamic_engine=thermodynamic_engine
    ) 