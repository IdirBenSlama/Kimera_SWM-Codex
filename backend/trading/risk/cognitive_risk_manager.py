"""
Cognitive Risk Manager for Kimera Ultimate Trading System

Revolutionary risk management using:
- Thermodynamic risk analysis
- Cognitive field-based risk assessment
- Contradiction detection for risk signals
- Dynamic risk limit adjustment
- Multi-dimensional risk scoring
"""

import asyncio
import logging
import time
import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
from enum import Enum

# Kimera cognitive components
from backend.engines.cognitive_field_dynamics import CognitiveFieldDynamics
from backend.engines.contradiction_engine import ContradictionEngine
from backend.engines.thermodynamics import SemanticThermodynamicsEngine

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Risk level classifications"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"
    EXTREME = "extreme"

@dataclass
class RiskLimits:
    """Dynamic risk limits"""
    max_position_size: float = 0.1      # 10% of portfolio
    max_daily_loss: float = 0.05        # 5% daily loss limit
    max_drawdown: float = 0.1           # 10% maximum drawdown
    stop_loss_percent: float = 0.02     # 2% stop loss
    max_correlation_exposure: float = 0.3  # 30% max correlated positions
    volatility_multiplier: float = 2.0   # Risk adjustment for volatility
    
    # Thermodynamic limits
    max_thermal_entropy: float = 2.0    # Maximum system entropy
    min_thermal_efficiency: float = 0.7  # Minimum efficiency threshold
    
    # Cognitive limits
    min_cognitive_confidence: float = 0.6  # Minimum decision confidence
    max_contradiction_score: float = 0.8   # Maximum contradiction tolerance

@dataclass
class CognitiveRiskAssessment:
    """Comprehensive risk assessment result"""
    risk_level: RiskLevel
    risk_score: float                   # 0-1, higher is riskier
    thermal_entropy: float              # Thermodynamic entropy
    cognitive_confidence: float         # Decision confidence
    contradiction_score: float          # Contradiction intensity
    
    # Position sizing
    recommended_position_size: float
    max_position_size: float
    stop_loss_price: float
    take_profit_price: float
    
    # Risk factors
    volatility_risk: float
    correlation_risk: float
    liquidity_risk: float
    market_risk: float
    
    # Reasoning
    risk_factors: List[str]
    recommendations: List[str]
    reasoning: str
    
    # Metadata
    timestamp: datetime
    symbol: str
    side: str

@dataclass
class PortfolioRiskState:
    """Current portfolio risk state"""
    total_exposure: float
    daily_pnl: float
    current_drawdown: float
    correlation_exposure: Dict[str, float]
    thermal_state: Dict[str, float]
    cognitive_state: Dict[str, float]
    active_positions: Dict[str, Any]

class ThermodynamicRiskAnalyzer:
    """Analyze risk using thermodynamic principles"""
    
    def __init__(self):
        self.thermodynamics_engine = SemanticThermodynamicsEngine()
        self.temperature_history = deque(maxlen=100)
        self.entropy_history = deque(maxlen=100)
        self.efficiency_history = deque(maxlen=100)
        
    def calculate_market_temperature(self, market_data: Dict[str, Any]) -> float:
        """Calculate market temperature from volatility and volume"""
        volatility = market_data.get('volatility', 0.02)
        volume = market_data.get('volume', 1000000)
        price_momentum = market_data.get('momentum', 0)
        
        # Thermodynamic temperature calculation
        # High volatility + high volume = high temperature
        base_temperature = volatility * np.log(1 + volume / 1000000)
        momentum_factor = 1 + abs(price_momentum) * 0.5
        
        temperature = base_temperature * momentum_factor
        self.temperature_history.append(temperature)
        
        return temperature
    
    def calculate_market_entropy(self, market_data: Dict[str, Any]) -> float:
        """Calculate market entropy (disorder/uncertainty)"""
        try:
            # Price distribution entropy
            price_changes = market_data.get('price_changes', [0])
            if len(price_changes) > 1:
                # Calculate probability distribution of price changes
                hist, _ = np.histogram(price_changes, bins=10, density=True)
                # Add small epsilon to avoid log(0)
                hist = hist + 1e-10
                entropy = -np.sum(hist * np.log(hist))
            else:
                entropy = 1.0  # High uncertainty with limited data
            
            # Volume entropy
            volume_changes = market_data.get('volume_changes', [0])
            if len(volume_changes) > 1:
                vol_hist, _ = np.histogram(volume_changes, bins=10, density=True)
                vol_hist = vol_hist + 1e-10
                volume_entropy = -np.sum(vol_hist * np.log(vol_hist))
                entropy = (entropy + volume_entropy) / 2
            
            self.entropy_history.append(entropy)
            return entropy
            
        except Exception as e:
            logger.warning(f"Entropy calculation failed: {e}")
            return 1.0  # Default high entropy
    
    def calculate_thermal_efficiency(self, portfolio_state: PortfolioRiskState) -> float:
        """Calculate system thermal efficiency"""
        try:
            # Efficiency = useful work / total energy input
            # In trading: profit / total exposure
            
            total_exposure = portfolio_state.total_exposure
            daily_pnl = portfolio_state.daily_pnl
            
            if total_exposure > 0:
                efficiency = max(0, daily_pnl / total_exposure)
            else:
                efficiency = 0.5  # Neutral efficiency
            
            # Normalize to 0-1 range
            efficiency = min(1.0, max(0.0, efficiency + 0.5))
            
            self.efficiency_history.append(efficiency)
            return efficiency
            
        except Exception as e:
            logger.warning(f"Efficiency calculation failed: {e}")
            return 0.5
    
    def get_thermal_risk_score(self, market_data: Dict[str, Any], 
                              portfolio_state: PortfolioRiskState) -> float:
        """Calculate overall thermal risk score"""
        temperature = self.calculate_market_temperature(market_data)
        entropy = self.calculate_market_entropy(market_data)
        efficiency = self.calculate_thermal_efficiency(portfolio_state)
        
        # Risk increases with temperature and entropy, decreases with efficiency
        thermal_risk = (temperature * 0.4 + entropy * 0.4 + (1 - efficiency) * 0.2)
        
        return min(1.0, max(0.0, thermal_risk))

class CognitiveRiskAnalyzer:
    """Analyze risk using cognitive field dynamics"""
    
    def __init__(self):
        self.cognitive_field = CognitiveFieldDynamics(dimension=256)
        self.contradiction_engine = ContradictionEngine()
        self.risk_memory = {}
        self.confidence_history = deque(maxlen=100)
        
    async def assess_cognitive_risk(self, market_data: Dict[str, Any], 
                                  trading_signal: Dict[str, Any]) -> Tuple[float, float]:
        """Assess risk using cognitive analysis"""
        try:
            # Create market risk embedding
            price = market_data.get('price', 0)
            volume = market_data.get('volume', 0)
            volatility = market_data.get('volatility', 0)
            momentum = market_data.get('momentum', 0)
            
            # Signal characteristics
            signal_confidence = trading_signal.get('confidence', 0.5)
            signal_strength = abs(trading_signal.get('strength', 0))
            
            risk_embedding = torch.tensor([
                price / 100000,           # Normalized price
                volume / 1000000,         # Normalized volume
                volatility * 10,          # Amplified volatility
                momentum * 5,             # Amplified momentum
                signal_confidence,        # Signal confidence
                signal_strength,          # Signal strength
                time.time() % 86400 / 86400,  # Time of day factor
                np.sin(2 * np.pi * time.time() / 604800)  # Weekly cycle
            ], dtype=torch.float32)
            
            # Add to cognitive field
            risk_field = self.cognitive_field.add_geoid(
                f"risk_assessment_{time.time()}",
                risk_embedding
            )
            
            if not risk_field:
                return 0.5, 0.3  # Default medium risk, low confidence
            
            # Analyze field strength as risk indicator
            field_strength = risk_field.field_strength
            resonance = risk_field.resonance_frequency
            
            # High field strength with low resonance = high risk
            # Low field strength with high resonance = low risk
            cognitive_risk = field_strength * (2 - resonance)
            cognitive_risk = min(1.0, max(0.0, cognitive_risk))
            
            # Confidence based on field coherence
            cognitive_confidence = field_strength * resonance
            cognitive_confidence = min(1.0, max(0.0, cognitive_confidence))
            
            self.confidence_history.append(cognitive_confidence)
            
            return cognitive_risk, cognitive_confidence
            
        except Exception as e:
            logger.error(f"Cognitive risk assessment failed: {e}")
            return 0.7, 0.2  # High risk, low confidence on error
    
    async def detect_risk_contradictions(self, market_data: Dict[str, Any], 
                                       trading_signals: List[Dict[str, Any]]) -> float:
        """Detect contradictions in risk signals"""
        try:
            # Create contradiction analysis
            signal_embeddings = []
            
            for signal in trading_signals:
                embedding = torch.tensor([
                    signal.get('confidence', 0.5),
                    signal.get('strength', 0),
                    signal.get('risk_score', 0.5)
                ], dtype=torch.float32)
                signal_embeddings.append(embedding)
            
            if not signal_embeddings:
                return 0.5  # Medium contradiction score
            
            # Detect contradictions between signals
            contradictions = self.contradiction_engine.detect_tension_gradients(signal_embeddings)
            
            # Calculate contradiction intensity
            contradiction_score = len(contradictions) / max(1, len(trading_signals))
            contradiction_score = min(1.0, max(0.0, contradiction_score))
            
            return contradiction_score
            
        except Exception as e:
            logger.error(f"Contradiction detection failed: {e}")
            return 0.5

class CognitiveRiskManager:
    """
    Revolutionary risk management system using cognitive analysis,
    thermodynamic principles, and contradiction detection
    """
    
    def __init__(self, limits: RiskLimits = None):
        self.limits = limits or RiskLimits()
        
        # Risk analyzers
        self.thermal_analyzer = ThermodynamicRiskAnalyzer()
        self.cognitive_analyzer = CognitiveRiskAnalyzer()
        
        # Portfolio state
        self.portfolio_value = 100000.0  # $100k default
        self.active_positions = {}
        self.daily_pnl = 0.0
        self.max_drawdown = 0.0
        self.correlation_matrix = {}
        
        # Risk history
        self.risk_assessments = deque(maxlen=1000)
        self.performance_metrics = {
            'total_assessments': 0,
            'high_risk_rejected': 0,
            'successful_trades': 0,
            'avoided_losses': 0
        }
        
        logger.info("🛡️ Cognitive Risk Manager initialized")
        logger.info(f"   Max position size: {self.limits.max_position_size:.1%}")
        logger.info(f"   Max daily loss: {self.limits.max_daily_loss:.1%}")
        logger.info(f"   Stop loss: {self.limits.stop_loss_percent:.1%}")
    
    async def assess_trade_risk(self, symbol: str, side: str, quantity: float, 
                              price: float, market_data: Dict[str, Any],
                              trading_signals: List[Dict[str, Any]] = None) -> CognitiveRiskAssessment:
        """Comprehensive cognitive risk assessment for a trade"""
        
        start_time = time.time()
        trading_signals = trading_signals or []
        
        # Create portfolio state
        portfolio_state = self.get_current_portfolio_state()
        
        # Thermodynamic risk analysis
        thermal_risk = self.thermal_analyzer.get_thermal_risk_score(market_data, portfolio_state)
        thermal_entropy = self.thermal_analyzer.calculate_market_entropy(market_data)
        
        # Cognitive risk analysis
        primary_signal = trading_signals[0] if trading_signals else {'confidence': 0.5, 'strength': 0}
        cognitive_risk, cognitive_confidence = await self.cognitive_analyzer.assess_cognitive_risk(
            market_data, primary_signal
        )
        
        # Contradiction analysis
        contradiction_score = await self.cognitive_analyzer.detect_risk_contradictions(
            market_data, trading_signals
        )
        
        # Traditional risk factors
        volatility_risk = self.calculate_volatility_risk(market_data)
        correlation_risk = self.calculate_correlation_risk(symbol, quantity, price)
        liquidity_risk = self.calculate_liquidity_risk(market_data)
        market_risk = self.calculate_market_risk(market_data, portfolio_state)
        
        # Combined risk score
        risk_score = self.calculate_combined_risk_score(
            thermal_risk, cognitive_risk, volatility_risk, 
            correlation_risk, liquidity_risk, market_risk
        )
        
        # Determine risk level
        risk_level = self.determine_risk_level(risk_score)
        
        # Position sizing recommendations
        base_position_size = quantity
        recommended_size, max_size = self.calculate_position_sizing(
            symbol, base_position_size, price, risk_score, market_data
        )
        
        # Stop loss and take profit
        stop_loss_price, take_profit_price = self.calculate_exit_prices(
            side, price, risk_score, market_data
        )
        
        # Risk factors and recommendations
        risk_factors = self.identify_risk_factors(
            thermal_risk, cognitive_risk, volatility_risk, 
            correlation_risk, liquidity_risk, market_risk, contradiction_score
        )
        
        recommendations = self.generate_recommendations(risk_level, risk_factors)
        
        # Create comprehensive reasoning
        reasoning = self.generate_risk_reasoning(
            risk_score, risk_level, thermal_entropy, cognitive_confidence,
            contradiction_score, risk_factors
        )
        
        # Create assessment
        assessment = CognitiveRiskAssessment(
            risk_level=risk_level,
            risk_score=risk_score,
            thermal_entropy=thermal_entropy,
            cognitive_confidence=cognitive_confidence,
            contradiction_score=contradiction_score,
            recommended_position_size=recommended_size,
            max_position_size=max_size,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            volatility_risk=volatility_risk,
            correlation_risk=correlation_risk,
            liquidity_risk=liquidity_risk,
            market_risk=market_risk,
            risk_factors=risk_factors,
            recommendations=recommendations,
            reasoning=reasoning,
            timestamp=datetime.now(),
            symbol=symbol,
            side=side
        )
        
        # Record assessment
        self.risk_assessments.append(assessment)
        self.performance_metrics['total_assessments'] += 1
        
        # Log assessment
        processing_time = (time.time() - start_time) * 1000
        logger.info(f"🛡️ Risk Assessment for {symbol} {side.upper()}")
        logger.info(f"   Risk Level: {risk_level.value.upper()} (score: {risk_score:.3f})")
        logger.info(f"   Recommended Size: {recommended_size:.6f} (max: {max_size:.6f})")
        logger.info(f"   Processing Time: {processing_time:.1f}ms")
        
        return assessment
    
    def calculate_combined_risk_score(self, thermal_risk: float, cognitive_risk: float,
                                    volatility_risk: float, correlation_risk: float,
                                    liquidity_risk: float, market_risk: float) -> float:
        """Calculate weighted combined risk score"""
        # Weighted combination of risk factors
        combined_risk = (
            thermal_risk * 0.25 +        # 25% thermodynamic risk
            cognitive_risk * 0.25 +      # 25% cognitive risk
            volatility_risk * 0.20 +     # 20% volatility risk
            correlation_risk * 0.15 +    # 15% correlation risk
            liquidity_risk * 0.10 +      # 10% liquidity risk
            market_risk * 0.05           # 5% general market risk
        )
        
        return min(1.0, max(0.0, combined_risk))
    
    def determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level from score"""
        if risk_score < 0.2:
            return RiskLevel.VERY_LOW
        elif risk_score < 0.4:
            return RiskLevel.LOW
        elif risk_score < 0.6:
            return RiskLevel.MEDIUM
        elif risk_score < 0.8:
            return RiskLevel.HIGH
        elif risk_score < 0.95:
            return RiskLevel.VERY_HIGH
        else:
            return RiskLevel.EXTREME
    
    def calculate_position_sizing(self, symbol: str, base_size: float, price: float,
                                risk_score: float, market_data: Dict[str, Any]) -> Tuple[float, float]:
        """Calculate recommended and maximum position sizes"""
        # Base position value
        position_value = base_size * price
        max_position_value = self.portfolio_value * self.limits.max_position_size
        
        # Risk-adjusted sizing
        risk_multiplier = 1.0 - risk_score  # Lower risk = larger position
        volatility = market_data.get('volatility', 0.02)
        volatility_adjustment = 1.0 / (1.0 + volatility * self.limits.volatility_multiplier)
        
        # Calculate sizes
        max_size = min(base_size, max_position_value / price)
        recommended_size = max_size * risk_multiplier * volatility_adjustment
        
        return recommended_size, max_size
    
    def calculate_exit_prices(self, side: str, entry_price: float, risk_score: float,
                            market_data: Dict[str, Any]) -> Tuple[float, float]:
        """Calculate stop loss and take profit prices"""
        volatility = market_data.get('volatility', 0.02)
        
        # Dynamic stop loss based on risk and volatility
        base_stop_loss = self.limits.stop_loss_percent
        risk_adjusted_stop = base_stop_loss * (1 + risk_score)
        volatility_adjusted_stop = risk_adjusted_stop * (1 + volatility)
        
        # Take profit (risk-reward ratio)
        risk_reward_ratio = 2.0  # 2:1 reward to risk
        take_profit_percent = volatility_adjusted_stop * risk_reward_ratio
        
        if side == 'buy':
            stop_loss_price = entry_price * (1 - volatility_adjusted_stop)
            take_profit_price = entry_price * (1 + take_profit_percent)
        else:  # sell
            stop_loss_price = entry_price * (1 + volatility_adjusted_stop)
            take_profit_price = entry_price * (1 - take_profit_percent)
        
        return stop_loss_price, take_profit_price
    
    def calculate_volatility_risk(self, market_data: Dict[str, Any]) -> float:
        """Calculate volatility-based risk"""
        volatility = market_data.get('volatility', 0.02)
        # Higher volatility = higher risk
        return min(1.0, volatility * 20)  # Normalize to 0-1
    
    def calculate_correlation_risk(self, symbol: str, quantity: float, price: float) -> float:
        """Calculate correlation risk with existing positions"""
        # Simplified correlation risk calculation
        position_value = quantity * price
        total_exposure = sum(pos['value'] for pos in self.active_positions.values())
        
        if total_exposure == 0:
            return 0.0
        
        # Risk increases with concentration
        concentration = position_value / (total_exposure + position_value)
        return min(1.0, concentration * 2)
    
    def calculate_liquidity_risk(self, market_data: Dict[str, Any]) -> float:
        """Calculate liquidity risk"""
        volume = market_data.get('volume', 1000000)
        avg_volume = market_data.get('avg_volume', volume)
        
        # Lower relative volume = higher liquidity risk
        volume_ratio = volume / max(avg_volume, 1)
        liquidity_risk = max(0.0, 1.0 - volume_ratio)
        
        return min(1.0, liquidity_risk)
    
    def calculate_market_risk(self, market_data: Dict[str, Any], 
                            portfolio_state: PortfolioRiskState) -> float:
        """Calculate general market risk"""
        # Market sentiment and trend risk
        momentum = market_data.get('momentum', 0)
        trend_strength = abs(momentum)
        
        # Higher momentum = higher risk (potential reversal)
        market_risk = trend_strength * 0.5
        
        return min(1.0, max(0.0, market_risk))
    
    def identify_risk_factors(self, thermal_risk: float, cognitive_risk: float,
                            volatility_risk: float, correlation_risk: float,
                            liquidity_risk: float, market_risk: float,
                            contradiction_score: float) -> List[str]:
        """Identify specific risk factors"""
        factors = []
        
        if thermal_risk > 0.7:
            factors.append("High thermodynamic entropy detected")
        if cognitive_risk > 0.7:
            factors.append("Cognitive field analysis indicates high risk")
        if volatility_risk > 0.6:
            factors.append("Elevated market volatility")
        if correlation_risk > 0.5:
            factors.append("High correlation with existing positions")
        if liquidity_risk > 0.5:
            factors.append("Limited market liquidity")
        if market_risk > 0.6:
            factors.append("Adverse market conditions")
        if contradiction_score > 0.7:
            factors.append("Contradictory signals detected")
        
        return factors
    
    def generate_recommendations(self, risk_level: RiskLevel, risk_factors: List[str]) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []
        
        if risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH, RiskLevel.EXTREME]:
            recommendations.append("Reduce position size significantly")
            recommendations.append("Use tighter stop losses")
            recommendations.append("Consider avoiding this trade")
        
        if risk_level == RiskLevel.EXTREME:
            recommendations.append("DO NOT TRADE - Risk too high")
        
        if "High thermodynamic entropy" in risk_factors:
            recommendations.append("Wait for market stabilization")
        
        if "Contradictory signals" in risk_factors:
            recommendations.append("Seek additional confirmation")
        
        if "Limited market liquidity" in risk_factors:
            recommendations.append("Use limit orders to avoid slippage")
        
        return recommendations
    
    def generate_risk_reasoning(self, risk_score: float, risk_level: RiskLevel,
                              thermal_entropy: float, cognitive_confidence: float,
                              contradiction_score: float, risk_factors: List[str]) -> str:
        """Generate comprehensive risk reasoning"""
        reasoning = f"Risk Assessment Summary:\n"
        reasoning += f"Overall Risk Score: {risk_score:.3f} ({risk_level.value.upper()})\n"
        reasoning += f"Thermodynamic Entropy: {thermal_entropy:.3f}\n"
        reasoning += f"Cognitive Confidence: {cognitive_confidence:.3f}\n"
        reasoning += f"Contradiction Score: {contradiction_score:.3f}\n\n"
        
        if risk_factors:
            reasoning += "Risk Factors Identified:\n"
            for factor in risk_factors:
                reasoning += f"• {factor}\n"
        else:
            reasoning += "No significant risk factors identified.\n"
        
        return reasoning
    
    def get_current_portfolio_state(self) -> PortfolioRiskState:
        """Get current portfolio risk state"""
        total_exposure = sum(pos.get('value', 0) for pos in self.active_positions.values())
        
        return PortfolioRiskState(
            total_exposure=total_exposure,
            daily_pnl=self.daily_pnl,
            current_drawdown=self.max_drawdown,
            correlation_exposure={},  # Simplified
            thermal_state={},         # Simplified
            cognitive_state={},       # Simplified
            active_positions=dict(self.active_positions)
        )
    
    def update_position(self, symbol: str, quantity: float, price: float, side: str):
        """Update position tracking"""
        position_value = quantity * price
        
        if symbol not in self.active_positions:
            self.active_positions[symbol] = {
                'quantity': 0,
                'value': 0,
                'avg_price': 0,
                'side': side
            }
        
        # Update position
        pos = self.active_positions[symbol]
        if side == pos['side']:
            # Add to position
            total_value = pos['value'] + position_value
            total_quantity = pos['quantity'] + quantity
            pos['avg_price'] = total_value / total_quantity if total_quantity > 0 else price
            pos['quantity'] = total_quantity
            pos['value'] = total_value
        else:
            # Reduce or reverse position
            if quantity >= pos['quantity']:
                # Reverse position
                remaining_quantity = quantity - pos['quantity']
                pos['quantity'] = remaining_quantity
                pos['value'] = remaining_quantity * price
                pos['avg_price'] = price
                pos['side'] = side
            else:
                # Reduce position
                pos['quantity'] -= quantity
                pos['value'] = pos['quantity'] * pos['avg_price']
    
    def update_pnl(self, pnl: float):
        """Update daily P&L and drawdown tracking"""
        self.daily_pnl += pnl
        
        # Update max drawdown
        if pnl < 0:
            current_drawdown = abs(self.daily_pnl) / self.portfolio_value
            self.max_drawdown = max(self.max_drawdown, current_drawdown)
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get comprehensive risk management metrics"""
        recent_assessments = list(self.risk_assessments)[-100:]  # Last 100 assessments
        
        if recent_assessments:
            avg_risk_score = np.mean([a.risk_score for a in recent_assessments])
            risk_level_distribution = {}
            for level in RiskLevel:
                count = sum(1 for a in recent_assessments if a.risk_level == level)
                risk_level_distribution[level.value] = count
        else:
            avg_risk_score = 0
            risk_level_distribution = {}
        
        return {
            'portfolio_value': self.portfolio_value,
            'daily_pnl': self.daily_pnl,
            'max_drawdown': self.max_drawdown,
            'active_positions': len(self.active_positions),
            'total_exposure': sum(pos.get('value', 0) for pos in self.active_positions.values()),
            'avg_risk_score': avg_risk_score,
            'risk_level_distribution': risk_level_distribution,
            'performance_metrics': dict(self.performance_metrics),
            'total_assessments': len(self.risk_assessments)
        }

# Factory function
def create_cognitive_risk_manager(limits: RiskLimits = None) -> CognitiveRiskManager:
    """Create and return cognitive risk manager instance"""
    return CognitiveRiskManager(limits) 