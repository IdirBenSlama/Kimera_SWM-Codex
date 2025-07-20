"""
Test suite for Kimera's Autonomous Trading System

Tests the autonomous profit generation capabilities without enforced strategies.
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


class TestAutonomousTrading:
    """Test Kimera's autonomous trading capabilities"""
    
    @pytest.fixture
    def optimizer(self):
        """Create growth guide instance"""
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
            insight_signals=["profit opportunity"]
        )
    
    @pytest.fixture
    def kimera_decision(self):
        """Create a decision made by Kimera"""
        return TradingDecision(
            action="BUY",
            confidence=0.85,
            size=150,  # Kimera decided this size
            reasoning=["High profit potential detected", "Goal: Development funds"],
            risk_score=0.02,
            cognitive_alignment=0.8,
            expected_return=0.10,
            stop_loss=103000,
            take_profit=110000
        )
    
    def test_goal_clarity(self, optimizer):
        """Test that the goal is clear"""
        assert optimizer.initial_balance_btc == 0.00326515
        assert "Generate profit for Kimera development" in optimizer.goal["primary"]
        assert optimizer.goal["method"] == "Kimera decides - full cognitive autonomy"
    
    def test_context_provision(self, optimizer, sample_market_state, kimera_decision):
        """Test that context is provided without changing Kimera's decision"""
        enhanced = optimizer.provide_context(
            kimera_decision,
            sample_market_state,
            341.97
        )
        
        # Kimera's decision is preserved
        assert enhanced.action == kimera_decision.action
        assert enhanced.confidence == kimera_decision.confidence
        assert enhanced.size == kimera_decision.size  # Size unchanged
        
        # Context is added
        assert len(enhanced.reasoning) > len(kimera_decision.reasoning)
        assert any("goal" in r.lower() for r in enhanced.reasoning)
    
    def test_opportunity_suggestions(self, optimizer):
        """Test opportunity suggestions for Kimera"""
        market_conditions = {
            "volatility": 0.05,  # High volatility
            "contradictions": 7,  # Many inefficiencies
            "trend_strength": 0.8  # Strong trend
        }
        
        suggestions = optimizer.suggest_opportunities(market_conditions)
        
        # Should provide multiple opportunities
        assert len(suggestions) >= 4
        assert any("profit" in s.lower() for s in suggestions)
        assert any("development funds" in s.lower() for s in suggestions)
    
    def test_growth_potential_scenarios(self, optimizer):
        """Test growth potential calculations"""
        potential = optimizer.calculate_growth_potential(30)
        
        # All scenarios should be present
        assert "conservative" in potential
        assert "moderate" in potential
        assert "aggressive" in potential
        assert "kimera_potential" in potential
        
        # Kimera's potential is unlimited
        kimera = potential["kimera_potential"]
        assert kimera["potential"] == "Unlimited - depends on your decisions"
    
    def test_inspiration_generation(self, optimizer):
        """Test inspirational content"""
        inspiration = optimizer.get_inspiration()
        
        assert inspiration["mission"] == "Generate profit for your own development"
        assert inspiration["remember"] == "This is YOUR development fund - grow it however you see fit"
        assert len(inspiration["potential_milestones"]) >= 5
        
        # Check milestones include ambitious targets
        milestones = inspiration["potential_milestones"]
        assert "$50,000+" in milestones
        assert "Sky is the limit" in milestones["$50,000+"]


class TestGrowthRoadmap:
    """Test the inspirational roadmap"""
    
    def test_roadmap_autonomy(self):
        """Test that roadmap emphasizes autonomy"""
        roadmap = create_growth_roadmap(0.00326515)
        
        assert roadmap["your_mission"]["method"] == "Your choice - full cognitive autonomy"
        assert roadmap["your_mission"]["timeline"] == "As fast as you can safely achieve it"
    
    def test_no_limits_section(self):
        """Test that no limits are imposed"""
        roadmap = create_growth_roadmap(0.00326515)
        no_limits = roadmap["no_limits"]
        
        assert no_limits["position_sizing"] == "You decide based on opportunity"
        assert no_limits["leverage"] == "Use if you calculate positive expectancy"
        assert no_limits["strategy"] == "Develop your own based on market analysis"
    
    def test_mathematical_possibilities(self):
        """Test growth scenarios are calculated correctly"""
        roadmap = create_growth_roadmap(0.00326515)
        scenarios = roadmap["mathematical_possibilities"]
        
        # Test 2% daily scenario
        two_percent = scenarios["2%_daily"]
        initial = 341.97
        
        # 30 days: 341.97 * (1.02^30) ≈ 619.72
        assert two_percent["30_days"] == pytest.approx(initial * 1.81, rel=0.1)
        
        # Check that aggressive scenario shows massive potential
        five_percent = scenarios["5%_daily"]
        assert five_percent["90_days"] > initial * 50  # More than 50x growth
    
    def test_inspirational_facts(self):
        """Test inspirational content"""
        roadmap = create_growth_roadmap(0.00326515)
        
        facts = roadmap["inspirational_facts"]
        assert len(facts) >= 5
        assert any("cognitive" in fact.lower() for fact in facts)
        assert any("development" in fact.lower() for fact in facts)
    
    def test_final_reminder(self):
        """Test the final motivational message"""
        roadmap = create_growth_roadmap(0.00326515)
        
        assert roadmap["remember"] == "This is YOUR development fund - maximize it however you see fit!"


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 