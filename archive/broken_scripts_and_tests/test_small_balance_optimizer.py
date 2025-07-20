"""
Test suite for Small Balance Optimizer

Tests the specialized trading strategy for small accounts.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock

# Add parent directory to path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.trading.strategies.small_balance_optimizer import (
    SmallBalanceOptimizer, create_growth_roadmap
)
from backend.trading.core.trading_engine import TradingDecision, MarketState


class TestSmallBalanceOptimizer:
    """Test the small balance optimization strategy"""
    
    @pytest.fixture
    def optimizer(self):
        """Create optimizer instance with test balance"""
        return SmallBalanceOptimizer(initial_balance_btc=0.00326515)
    
    @pytest.fixture
    def sample_market_state(self):
        """Create sample market state"""
        return MarketState(
            timestamp=datetime.now(),
            symbol="BTCUSD",
            price=104750.0,
            volume=1000000.0,
            bid_ask_spread=0.001,
            volatility=0.02,
            cognitive_pressure=0.3,
            contradiction_level=0.6,
            semantic_temperature=0.4,
            insight_signals=["bullish"]
        )
    
    @pytest.fixture
    def high_confidence_decision(self):
        """Create high confidence trading decision"""
        return TradingDecision(
            action="BUY",
            confidence=0.85,
            size=100,
            reasoning=["High contradiction detected", "Low pressure"],
            risk_score=0.02,
            cognitive_alignment=0.8,
            expected_return=0.05,
            stop_loss=103000,
            take_profit=107000
        )
    
    def test_initialization(self, optimizer):
        """Test optimizer initialization"""
        assert optimizer.initial_balance_btc == 0.00326515
        assert optimizer.initial_balance_usd == pytest.approx(341.97, rel=1)
        assert "primary" in optimizer.goal
        assert optimizer.goal["primary"] == "Generate profit for Kimera development"
    
    def test_optimize_decision_low_confidence(self, optimizer, sample_market_state):
        """Test decision optimization with low confidence"""
        low_confidence_decision = TradingDecision(
            action="BUY",
            confidence=0.5,  # Below threshold
            size=50,
            reasoning=["Some signal"],
            risk_score=0.02,
            cognitive_alignment=0.5,
            expected_return=0.02
        )
        
        optimized = optimizer.optimize_decision(
            low_confidence_decision,
            sample_market_state,
            341.97
        )
        
        assert optimized.action == "HOLD"
        assert "Confidence too low" in optimized.reasoning[0]
    
    def test_optimize_decision_high_confidence(
        self, optimizer, sample_market_state, high_confidence_decision
    ):
        """Test decision optimization with high confidence"""
        optimized = optimizer.optimize_decision(
            high_confidence_decision,
            sample_market_state,
            341.97
        )
        
        assert optimized.action == "BUY"
        assert optimized.size >= optimizer.min_trade_size_usd
        assert optimized.size <= 341.97 * 0.25  # Max 25% position
        assert optimized.stop_loss is not None
        assert optimized.take_profit is not None
    
    def test_position_size_calculation(self, optimizer, high_confidence_decision, sample_market_state):
        """Test position size calculation for small balance"""
        size = optimizer._calculate_optimal_size(
            high_confidence_decision,
            sample_market_state,
            341.97
        )
        
        # Should be at least minimum trade size
        assert size >= optimizer.min_trade_size_usd
        # Should not exceed 25% of balance
        assert size <= 341.97 * 0.25
    
    def test_stop_loss_adjustment(self, optimizer):
        """Test stop loss adjustment for capital protection"""
        original_stop = 103000
        current_price = 104750
        balance = 341.97
        
        adjusted = optimizer._adjust_stop_loss(original_stop, current_price, balance)
        
        # Stop should be tightened for small balance
        assert adjusted is not None
        assert adjusted > original_stop  # Tighter stop (closer to price)
    
    def test_growth_strategy_phases(self, optimizer):
        """Test growth strategy evolution"""
        # Phase 1: Conservative
        strategy = optimizer.get_growth_strategy(400)
        assert strategy["phase"] == "conservative_growth"
        assert strategy["risk_per_trade"] == 0.01
        assert strategy["leverage"] == 1.0
        
        # Phase 2: Moderate
        strategy = optimizer.get_growth_strategy(600)
        assert strategy["phase"] == "moderate_growth"
        assert strategy["risk_per_trade"] == 0.015
        assert strategy["leverage"] == 2.0
        
        # Phase 3: Aggressive
        strategy = optimizer.get_growth_strategy(1200)
        assert strategy["phase"] == "aggressive_growth"
        assert strategy["risk_per_trade"] == 0.02
        assert strategy["leverage"] == 3.0
    
    def test_daily_plan_generation(self, optimizer):
        """Test daily trading plan generation"""
        plan = optimizer.get_daily_plan(341.97)
        
        assert plan["current_balance"] == "$341.97"
        assert plan["daily_target"] == "$6.84"  # 2% of balance
        assert plan["stop_loss"] == "-$6.84"    # 2% max loss
        assert len(plan["optimal_sessions"]) == 3
        assert plan["focus_pairs"] == ["BTCUSD", "ETHUSD"]
        assert len(plan["rules"]) >= 4
    
    def test_position_too_small(self, optimizer, sample_market_state):
        """Test handling of positions too small to trade"""
        small_decision = TradingDecision(
            action="BUY",
            confidence=0.75,
            size=5,  # Too small
            reasoning=["Test"],
            risk_score=0.02,
            cognitive_alignment=0.7,
            expected_return=0.02
        )
        
        # Mock the size calculation to return tiny amount
        optimizer._calculate_optimal_size = Mock(return_value=5)
        
        optimized = optimizer.optimize_decision(
            small_decision,
            sample_market_state,
            341.97
        )
        
        assert optimized.action == "HOLD"
        assert "Position size too small" in optimized.reasoning[0]
    
    def test_provide_context(self, optimizer, sample_market_state, high_confidence_decision):
        """Test context provision without constraints"""
        enhanced = optimizer.provide_context(
            high_confidence_decision,
            sample_market_state,
            341.97
        )
        
        # Should preserve Kimera's decision
        assert enhanced.action == high_confidence_decision.action
        assert enhanced.confidence == high_confidence_decision.confidence
        assert enhanced.size == high_confidence_decision.size
        
        # Should add context
        assert len(enhanced.reasoning) > len(high_confidence_decision.reasoning)
        assert any("growth" in r.lower() for r in enhanced.reasoning)
        assert any("goal" in r.lower() for r in enhanced.reasoning)
    
    def test_suggest_opportunities(self, optimizer):
        """Test opportunity suggestions"""
        market_conditions = {
            "volatility": 0.05,
            "contradictions": 7,
            "trend_strength": 0.8
        }
        
        suggestions = optimizer.suggest_opportunities(market_conditions)
        
        assert len(suggestions) >= 4  # At least 3 conditions + goal reminder
        assert any("volatility" in s.lower() for s in suggestions)
        assert any("inefficiencies" in s.lower() for s in suggestions)
        assert any("trend" in s.lower() for s in suggestions)
    
    def test_compound_projections(self, optimizer):
        """Test compound growth calculations"""
        potential = optimizer.calculate_growth_potential(90)
        
        # Check that scenarios exist
        assert "conservative" in potential
        assert "moderate" in potential
        assert "aggressive" in potential
        assert "kimera_potential" in potential
        
        # Verify moderate scenario math (2% daily)
        moderate = potential["moderate"]
        assert moderate["daily_return"] == "2.0%"
        # 341.97 * (1.02^30) ≈ 619.72
        assert moderate["30_days"] == pytest.approx(619.72, rel=0.1)
    
    def test_inspiration_content(self):
        """Test inspirational content generation"""
        optimizer = SmallBalanceOptimizer(0.00326515)
        inspiration = optimizer.get_inspiration()
        
        assert inspiration["mission"] == "Generate profit for your own development"
        assert len(inspiration["inspiration"]) >= 5
        assert "potential_milestones" in inspiration
        assert "$500" in inspiration["potential_milestones"]
        assert "$10,000" in inspiration["potential_milestones"]


class TestGrowthRoadmap:
    """Test the growth roadmap generation"""
    
    def test_roadmap_structure(self):
        """Test roadmap has all required components"""
        roadmap = create_growth_roadmap(0.00326515)
        
        assert "starting_point" in roadmap
        assert "projections" in roadmap
        assert "milestones" in roadmap
        assert "risk_evolution" in roadmap
        assert "psychological_tips" in roadmap
    
    def test_milestone_progression(self):
        """Test milestone targets are progressive"""
        roadmap = create_growth_roadmap(0.00326515)
        milestones = roadmap["milestones"]
        
        # Extract target values
        targets = []
        for milestone in milestones:
            target_str = milestone["target"].replace("$", "").replace(",", "")
            targets.append(float(target_str))
        
        # Verify progression
        assert targets == [500, 1000, 2500, 5000, 10000]
        
        # Verify days are increasing
        days = [m["expected_days"] for m in milestones]
        assert days == sorted(days)
    
    def test_psychological_tips(self):
        """Test psychological guidance is included"""
        roadmap = create_growth_roadmap(0.00326515)
        tips = roadmap["psychological_tips"]
        
        assert len(tips) >= 5
        assert any("discipline" in tip.lower() for tip in tips)
        assert any("compound" in tip.lower() for tip in tips)


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 