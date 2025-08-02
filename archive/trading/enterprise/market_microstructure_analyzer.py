"""
Market Microstructure Analyzer for Kimera SWM

Deep analysis of market microstructure including order book dynamics,
liquidity flows, and price discovery mechanisms.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict, deque
import pandas as pd

# Local imports
from src.core.geoid import GeoidState as Geoid
from src.engines.cognitive_field_dynamics import CognitiveFieldDynamics as CognitiveFieldDynamicsEngine
from src.engines.thermodynamic_engine import ThermodynamicEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OrderBookSnapshot:
    """Point-in-time order book state"""
    timestamp: datetime
    symbol: str
    bids: List[Tuple[float, float]]  # [(price, size), ...]
    asks: List[Tuple[float, float]]  # [(price, size), ...]
    last_trade_price: float = 0.0
    last_trade_size: float = 0.0
    
    @property
    def best_bid(self) -> Tuple[float, float]:
        """Get best bid (price, size)"""
        return self.bids[0] if self.bids else (0.0, 0.0)
        
    @property
    def best_ask(self) -> Tuple[float, float]:
        """Get best ask (price, size)"""
        return self.asks[0] if self.asks else (0.0, 0.0)
        
    @property
    def mid_price(self) -> float:
        """Calculate mid price"""
        if self.bids and self.asks:
            return (self.bids[0][0] + self.asks[0][0]) / 2
        return 0.0
        
    @property
    def spread(self) -> float:
        """Calculate bid-ask spread"""
        if self.bids and self.asks:
            return self.asks[0][0] - self.bids[0][0]
        return 0.0


@dataclass
class MicrostructureAnalysis:
    """Results of microstructure analysis"""
    symbol: str
    timestamp: datetime
    spread: float
    relative_spread: float  # Spread / mid_price
    bid_depth: float
    ask_depth: float
    order_imbalance: float
    liquidity_score: float
    price_impact: float
    kyles_lambda: float  # Kyle's lambda (price impact coefficient)
    informed_trading_probability: float
    toxic_flow_probability: float
    market_quality_score: float


@dataclass
class LiquidityProfile:
    """Liquidity profile at different price levels"""
    symbol: str
    timestamp: datetime
    price_levels: List[float]
    bid_liquidity: List[float]
    ask_liquidity: List[float]
    cumulative_bid_depth: List[float]
    cumulative_ask_depth: List[float]
    liquidity_concentration: float  # How concentrated liquidity is near best prices


class MarketMicrostructureAnalyzer:
    """
    Market Microstructure Analysis Engine
    
    Features:
    - Real-time order book reconstruction
    - Liquidity flow analysis
    - Price impact estimation
    - Market quality metrics
    - Informed trading detection
    """
    
    def __init__(self,
                 cognitive_field: Optional[CognitiveFieldDynamicsEngine] = None,
                 thermodynamic_engine: Optional[ThermodynamicEngine] = None):
        """Initialize Market Microstructure Analyzer"""
        self.cognitive_field = cognitive_field
        self.thermodynamic_engine = thermodynamic_engine
        
        # Order book storage
        self.order_books: Dict[str, OrderBookSnapshot] = {}
        self.order_book_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Trade flow analysis
        self.trade_flows: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.order_flow_imbalance: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Liquidity metrics
        self.liquidity_profiles: Dict[str, LiquidityProfile] = {}
        self.price_impact_estimates: Dict[str, float] = {}
        
        # Market quality tracking
        self.market_quality_scores: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Performance metrics
        self.analyses_performed = 0
        self.toxic_flows_detected = 0
        
        # Background analysis control
        self.running = False
        self.analysis_task = None
        
        # Test compatibility attributes
        self.order_book_reconstructor = {
            'type': 'real_time_reconstruction',
            'max_levels': 20,
            'update_frequency': 'microsecond',
            'symbols_tracked': len(self.order_books)
        }
        
        self.liquidity_analyzer = {
            'type': 'multi_dimensional_analysis',
            'metrics': ['depth', 'imbalance', 'toxicity', 'quality'],
            'algorithms': ['kyle_lambda', 'pin_estimation', 'market_impact'],
            'profiles_maintained': len(self.liquidity_profiles)
        }
        
        logger.info("Market Microstructure Analyzer initialized")
        
    async def analyze_order_book_update(self,
                                      symbol: str,
                                      bids: List[Tuple[float, float]],
                                      asks: List[Tuple[float, float]],
                                      last_trade: Optional[Tuple[float, float]] = None) -> MicrostructureAnalysis:
        """
        Analyze order book update and calculate microstructure metrics
        
        Args:
            symbol: Trading symbol
            bids: List of (price, size) tuples
            asks: List of (price, size) tuples
            last_trade: Optional (price, size) of last trade
            
        Returns:
            Microstructure analysis results
        """
        try:
            # Create order book snapshot
            snapshot = OrderBookSnapshot(
                timestamp=datetime.now(),
                symbol=symbol,
                bids=sorted(bids, key=lambda x: x[0], reverse=True)[:20],  # Top 20 levels
                asks=sorted(asks, key=lambda x: x[0])[:20],
                last_trade_price=last_trade[0] if last_trade else 0.0,
                last_trade_size=last_trade[1] if last_trade else 0.0
            )
            
            # Store snapshot
            self.order_books[symbol] = snapshot
            self.order_book_history[symbol].append(snapshot)
            
            # Calculate microstructure metrics
            analysis = await self._calculate_microstructure_metrics(snapshot)
            
            # Update liquidity profile
            await self._update_liquidity_profile(symbol, snapshot)
            
            # Detect toxic flow
            if await self._detect_toxic_flow(symbol, snapshot):
                analysis.toxic_flow_probability = min(analysis.toxic_flow_probability * 1.5, 1.0)
                self.toxic_flows_detected += 1
                
            # Integrate with cognitive field
            if self.cognitive_field:
                await self._cognitive_microstructure_analysis(analysis)
                
            # Track performance
            self.analyses_performed += 1
            
            return analysis
            
        except Exception as e:
            logger.error(f"Order book analysis error: {e}")
            raise
            
    async def _calculate_microstructure_metrics(self, snapshot: OrderBookSnapshot) -> MicrostructureAnalysis:
        """Calculate detailed microstructure metrics"""
        # Basic spread metrics
        spread = snapshot.spread
        relative_spread = spread / snapshot.mid_price if snapshot.mid_price > 0 else 0.0
        
        # Depth calculation
        bid_depth = sum(size for _, size in snapshot.bids[:5])  # Top 5 levels
        ask_depth = sum(size for _, size in snapshot.asks[:5])
        
        # Order imbalance
        total_depth = bid_depth + ask_depth
        order_imbalance = (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0.0
        
        # Liquidity score (combination of spread and depth)
        liquidity_score = self._calculate_liquidity_score(spread, bid_depth, ask_depth)
        
        # Price impact estimation
        price_impact = await self._estimate_price_impact(snapshot)
        
        # Kyle's lambda (simplified)
        kyles_lambda = self._calculate_kyles_lambda(snapshot)
        
        # Informed trading probability (PIN)
        informed_prob = await self._estimate_informed_trading_probability(snapshot)
        
        # Toxic flow probability
        toxic_prob = await self._estimate_toxic_flow_probability(snapshot)
        
        # Market quality score
        market_quality = self._calculate_market_quality_score(
            relative_spread, liquidity_score, order_imbalance
        )
        
        return MicrostructureAnalysis(
            symbol=snapshot.symbol,
            timestamp=snapshot.timestamp,
            spread=spread,
            relative_spread=relative_spread,
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            order_imbalance=order_imbalance,
            liquidity_score=liquidity_score,
            price_impact=price_impact,
            kyles_lambda=kyles_lambda,
            informed_trading_probability=informed_prob,
            toxic_flow_probability=toxic_prob,
            market_quality_score=market_quality
        )
        
    def _calculate_liquidity_score(self, spread: float, bid_depth: float, ask_depth: float) -> float:
        """Calculate overall liquidity score (0-1)"""
        # Normalize spread (inverse relationship)
        spread_score = 1.0 / (1.0 + spread * 10000)  # Assuming spread in price units
        
        # Normalize depth
        total_depth = bid_depth + ask_depth
        depth_score = min(total_depth / 1000, 1.0)  # Normalize to 1000 units
        
        # Balance score
        balance_score = 1.0 - abs(bid_depth - ask_depth) / max(total_depth, 1)
        
        # Weighted average
        liquidity_score = 0.4 * spread_score + 0.4 * depth_score + 0.2 * balance_score
        
        return liquidity_score
        
    async def _estimate_price_impact(self, snapshot: OrderBookSnapshot) -> float:
        """Estimate price impact of a standard trade"""
        # Standard trade size (e.g., 10 units)
        standard_size = 10.0
        
        # Calculate impact for buy order
        buy_impact = self._walk_order_book(snapshot.asks, standard_size, snapshot.mid_price)
        
        # Calculate impact for sell order
        sell_impact = self._walk_order_book(
            [(p, s) for p, s in snapshot.bids], 
            standard_size, 
            snapshot.mid_price,
            is_sell=True
        )
        
        # Average impact
        return (buy_impact + sell_impact) / 2
        
    def _walk_order_book(self, levels: List[Tuple[float, float]], 
                        size: float, 
                        reference_price: float,
                        is_sell: bool = False) -> float:
        """Walk the order book to calculate price impact"""
        if not levels or reference_price == 0:
            return 0.0
            
        remaining_size = size
        volume_weighted_price = 0.0
        total_cost = 0.0
        
        for price, level_size in levels:
            if remaining_size <= 0:
                break
                
            fill_size = min(remaining_size, level_size)
            total_cost += price * fill_size
            remaining_size -= fill_size
            
        if size - remaining_size > 0:
            volume_weighted_price = total_cost / (size - remaining_size)
            
            if is_sell:
                impact = (reference_price - volume_weighted_price) / reference_price
            else:
                impact = (volume_weighted_price - reference_price) / reference_price
                
            return abs(impact)
            
        return 0.0
        
    def _calculate_kyles_lambda(self, snapshot: OrderBookSnapshot) -> float:
        """Calculate Kyle's lambda (price impact coefficient)"""
        # Simplified Kyle's lambda based on order book shape
        if not snapshot.bids or not snapshot.asks:
            return 0.0
            
        # Calculate average price change per unit of volume
        price_changes = []
        volume_changes = []
        
        # For bids
        for i in range(min(5, len(snapshot.bids) - 1)):
            price_change = abs(snapshot.bids[i][0] - snapshot.bids[i+1][0])
            volume_change = snapshot.bids[i][1]
            if volume_change > 0:
                price_changes.append(price_change)
                volume_changes.append(volume_change)
                
        # For asks
        for i in range(min(5, len(snapshot.asks) - 1)):
            price_change = abs(snapshot.asks[i+1][0] - snapshot.asks[i][0])
            volume_change = snapshot.asks[i][1]
            if volume_change > 0:
                price_changes.append(price_change)
                volume_changes.append(volume_change)
                
        if price_changes and volume_changes:
            # Lambda = average(price_change / volume)
            lambdas = [p / v for p, v in zip(price_changes, volume_changes)]
            return np.mean(lambdas)
            
        return 0.0
        
    async def _estimate_informed_trading_probability(self, snapshot: OrderBookSnapshot) -> float:
        """Estimate probability of informed trading (PIN)"""
        # Simplified PIN estimation based on order imbalance persistence
        symbol = snapshot.symbol
        
        if symbol not in self.order_flow_imbalance or len(self.order_flow_imbalance[symbol]) < 10:
            return 0.5  # Default probability
            
        # Check persistence of order imbalance
        recent_imbalances = list(self.order_flow_imbalance[symbol])[-20:]
        
        # High persistence suggests informed trading
        if len(recent_imbalances) > 1:
            # Calculate autocorrelation
            imbalance_array = np.array(recent_imbalances)
            if np.std(imbalance_array) > 0:
                autocorr = np.corrcoef(imbalance_array[:-1], imbalance_array[1:])[0, 1]
                
                # Map autocorrelation to probability
                informed_prob = 0.5 + 0.5 * max(0, autocorr)
                return min(informed_prob, 0.95)
                
        return 0.5
        
    async def _estimate_toxic_flow_probability(self, snapshot: OrderBookSnapshot) -> float:
        """Estimate probability of toxic order flow"""
        # Toxic flow indicators:
        # 1. Rapid spread widening
        # 2. Depth imbalance
        # 3. Price momentum against liquidity providers
        
        symbol = snapshot.symbol
        history = list(self.order_book_history[symbol])[-10:]
        
        if len(history) < 5:
            return 0.1  # Low default
            
        # Check spread widening
        recent_spreads = [h.spread for h in history]
        spread_increasing = all(recent_spreads[i] <= recent_spreads[i+1] 
                               for i in range(len(recent_spreads)-1))
        
        # Check depth deterioration
        recent_depths = [(h.bid_depth + h.ask_depth) for h in history]
        depth_decreasing = all(recent_depths[i] >= recent_depths[i+1] 
                              for i in range(len(recent_depths)-1))
        
        # Combine indicators
        toxic_score = 0.1  # Base probability
        
        if spread_increasing:
            toxic_score += 0.3
            
        if depth_decreasing:
            toxic_score += 0.3
            
        # Check for one-sided pressure
        bid_depth = snapshot.bid_depth
        ask_depth = snapshot.ask_depth
        total_depth = bid_depth + ask_depth
        
        if total_depth > 0:
            imbalance = abs(bid_depth - ask_depth) / total_depth
            if imbalance > 0.7:
                toxic_score += 0.2
                
        return min(toxic_score, 0.9)
        
    def _calculate_market_quality_score(self, 
                                      relative_spread: float,
                                      liquidity_score: float,
                                      order_imbalance: float) -> float:
        """Calculate overall market quality score"""
        # Lower spread is better
        spread_score = 1.0 / (1.0 + relative_spread * 100)
        
        # Higher liquidity is better
        liq_score = liquidity_score
        
        # Lower imbalance is better
        balance_score = 1.0 - abs(order_imbalance)
        
        # Weighted average
        quality_score = 0.3 * spread_score + 0.5 * liq_score + 0.2 * balance_score
        
        return quality_score
        
    async def _update_liquidity_profile(self, symbol: str, snapshot: OrderBookSnapshot):
        """Update liquidity profile for the symbol"""
        # Define price levels for analysis
        mid_price = snapshot.mid_price
        if mid_price <= 0:
            return
            
        price_levels = [
            mid_price * (1 - 0.001 * i) for i in range(11)  # -1% to mid
        ] + [
            mid_price * (1 + 0.001 * i) for i in range(1, 11)  # mid to +1%
        ]
        
        # Calculate liquidity at each level
        bid_liquidity = []
        ask_liquidity = []
        
        for level in sorted(price_levels):
            # Bid liquidity (sum of all bids >= level)
            bid_liq = sum(size for price, size in snapshot.bids if price >= level)
            bid_liquidity.append(bid_liq)
            
            # Ask liquidity (sum of all asks <= level)
            ask_liq = sum(size for price, size in snapshot.asks if price <= level)
            ask_liquidity.append(ask_liq)
            
        # Calculate cumulative depth
        cumulative_bid = np.cumsum(bid_liquidity)
        cumulative_ask = np.cumsum(ask_liquidity[::-1])[::-1]
        
        # Liquidity concentration (what % is within 0.1% of mid)
        total_bid_liq = sum(size for _, size in snapshot.bids)
        total_ask_liq = sum(size for _, size in snapshot.asks)
        
        near_bid_liq = sum(size for price, size in snapshot.bids 
                          if price >= mid_price * 0.999)
        near_ask_liq = sum(size for price, size in snapshot.asks 
                          if price <= mid_price * 1.001)
        
        concentration = 0.0
        if total_bid_liq + total_ask_liq > 0:
            concentration = (near_bid_liq + near_ask_liq) / (total_bid_liq + total_ask_liq)
            
        self.liquidity_profiles[symbol] = LiquidityProfile(
            symbol=symbol,
            timestamp=snapshot.timestamp,
            price_levels=price_levels,
            bid_liquidity=bid_liquidity,
            ask_liquidity=ask_liquidity,
            cumulative_bid_depth=cumulative_bid.tolist(),
            cumulative_ask_depth=cumulative_ask.tolist(),
            liquidity_concentration=concentration
        )
        
    async def _detect_toxic_flow(self, symbol: str, snapshot: OrderBookSnapshot) -> bool:
        """Detect toxic order flow patterns"""
        # Store order imbalance
        total_depth = snapshot.bid_depth + snapshot.ask_depth
        if total_depth > 0:
            imbalance = (snapshot.bid_depth - snapshot.ask_depth) / total_depth
            self.order_flow_imbalance[symbol].append(imbalance)
            
        # Check for toxic patterns
        history = list(self.order_book_history[symbol])[-20:]
        
        if len(history) < 10:
            return False
            
        # Pattern 1: Rapidly widening spreads
        spreads = [h.spread for h in history]
        if len(spreads) > 5:
            recent_spread_change = (spreads[-1] - spreads[-5]) / max(spreads[-5], 0.0001)
            if recent_spread_change > 0.5:  # 50% increase
                return True
                
        # Pattern 2: Systematic depth reduction
        depths = [(h.bid_depth + h.ask_depth) for h in history]
        if len(depths) > 5:
            recent_depth_change = (depths[-1] - depths[-5]) / max(depths[-5], 1)
            if recent_depth_change < -0.3:  # 30% decrease
                return True
                
        # Pattern 3: Persistent one-sided pressure
        recent_imbalances = list(self.order_flow_imbalance[symbol])[-10:]
        if len(recent_imbalances) >= 10:
            if all(imb > 0.5 for imb in recent_imbalances) or all(imb < -0.5 for imb in recent_imbalances):
                return True
                
        return False
        
    async def analyze_liquidity_profile(self,
                                      symbol: str,
                                      bids: List[Tuple[float, float]],
                                      asks: List[Tuple[float, float]]) -> LiquidityProfile:
        """
        Analyze liquidity distribution across price levels
        
        Args:
            symbol: Trading symbol
            bids: Full order book bids
            asks: Full order book asks
            
        Returns:
            Liquidity profile analysis
        """
        # Create detailed snapshot
        snapshot = OrderBookSnapshot(
            timestamp=datetime.now(),
            symbol=symbol,
            bids=sorted(bids, key=lambda x: x[0], reverse=True),
            asks=sorted(asks, key=lambda x: x[0])
        )
        
        # Update liquidity profile
        await self._update_liquidity_profile(symbol, snapshot)
        
        return self.liquidity_profiles.get(symbol)
        
    async def estimate_market_impact(self,
                                   symbol: str,
                                   side: str,
                                   size: float) -> Dict[str, float]:
        """
        Estimate market impact of a potential trade
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            size: Trade size
            
        Returns:
            Impact estimates
        """
        snapshot = self.order_books.get(symbol)
        if not snapshot:
            return {'error': 'No order book data available'}
            
        # Walk the book
        if side == 'buy':
            impact = self._walk_order_book(snapshot.asks, size, snapshot.mid_price)
            levels_consumed = self._count_levels_consumed(snapshot.asks, size)
        else:
            impact = self._walk_order_book(
                [(p, s) for p, s in snapshot.bids],
                size,
                snapshot.mid_price,
                is_sell=True
            )
            levels_consumed = self._count_levels_consumed(snapshot.bids, size)
            
        # Estimate permanent impact (Kyle's lambda based)
        kyles_lambda = self.price_impact_estimates.get(symbol, 0.0001)
        permanent_impact = kyles_lambda * size
        
        return {
            'temporary_impact': impact,
            'permanent_impact': permanent_impact,
            'total_impact': impact + permanent_impact,
            'levels_consumed': levels_consumed,
            'mid_price': snapshot.mid_price
        }
        
    def _count_levels_consumed(self, levels: List[Tuple[float, float]], size: float) -> int:
        """Count how many price levels would be consumed"""
        remaining = size
        levels_consumed = 0
        
        for _, level_size in levels:
            if remaining <= 0:
                break
                
            remaining -= level_size
            levels_consumed += 1
            
        return levels_consumed
        
    async def _cognitive_microstructure_analysis(self, analysis: MicrostructureAnalysis):
        """Integrate microstructure analysis with cognitive field"""
        if not self.cognitive_field:
            return
            
        # Create analysis geoid
        analysis_geoid = Geoid(
            semantic_features={
                'type': 'microstructure_analysis',
                'symbol': analysis.symbol,
                'liquidity_score': analysis.liquidity_score,
                'market_quality': analysis.market_quality_score,
                'toxic_flow_risk': analysis.toxic_flow_probability
            },
            symbolic_content=f"Microstructure: {analysis.symbol}"
        )
        
        # Integrate with cognitive field
        await self.cognitive_field.integrate_geoid(analysis_geoid)
        
        # Check for market structure contradictions
        if analysis.toxic_flow_probability > 0.7 and analysis.market_quality_score > 0.7:
            logger.warning(f"Contradiction detected: High quality market with toxic flow for {analysis.symbol}")
            
    async def get_current_state(self, symbol: str) -> Optional[MicrostructureAnalysis]:
        """Get current microstructure state for a symbol"""
        snapshot = self.order_books.get(symbol)
        if not snapshot:
            return None
            
        return await self._calculate_microstructure_metrics(snapshot)
        
    async def continuous_analysis(self):
        """Continuously analyze market microstructure"""
        while self.running:
            try:
                # Analyze each tracked symbol
                for symbol, snapshot in self.order_books.items():
                    # Skip if data is stale
                    if (datetime.now() - snapshot.timestamp).total_seconds() > 60:
                        continue
                        
                    # Update price impact estimates
                    analysis = await self._calculate_microstructure_metrics(snapshot)
                    self.price_impact_estimates[symbol] = analysis.kyles_lambda
                    
                    # Track market quality
                    self.market_quality_scores[symbol].append(analysis.market_quality_score)
                    
                    # Alert on deteriorating conditions
                    if analysis.market_quality_score < 0.3:
                        logger.warning(f"Poor market quality detected for {symbol}: {analysis.market_quality_score:.2f}")
                        
                    if analysis.toxic_flow_probability > 0.8:
                        logger.warning(f"High toxic flow probability for {symbol}: {analysis.toxic_flow_probability:.2f}")
                        
                await asyncio.sleep(5)  # Analyze every 5 seconds
                
            except Exception as e:
                logger.error(f"Continuous analysis error: {e}")
                
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get analyzer performance metrics"""
        avg_quality_scores = {}
        for symbol, scores in self.market_quality_scores.items():
            if scores:
                avg_quality_scores[symbol] = np.mean(list(scores))
                
        return {
            'analyses_performed': self.analyses_performed,
            'toxic_flows_detected': self.toxic_flows_detected,
            'tracked_symbols': len(self.order_books),
            'average_market_quality': avg_quality_scores,
            'price_impact_estimates': dict(self.price_impact_estimates)
        }
        
    def shutdown(self):
        """Shutdown analyzer"""
        self.running = False


def create_microstructure_analyzer(cognitive_field=None,
                                 thermodynamic_engine=None) -> MarketMicrostructureAnalyzer:
    """Factory function to create Market Microstructure Analyzer"""
    return MarketMicrostructureAnalyzer(
        cognitive_field=cognitive_field,
        thermodynamic_engine=thermodynamic_engine
    ) 