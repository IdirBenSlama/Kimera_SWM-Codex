#!/usr/bin/env python3
"""
KIMERA DEMO TESTNET SIMULATION
==============================

Demo version of Kimera testnet simulation using simulated market data.
This version demonstrates all Kimera capabilities without external dependencies.

Features:
- Simulated real-time market data
- Constitutional decision framework
- Cognitive analysis system
- Risk management validation
- Performance metrics tracking
- Complete audit trails

SAFETY: Pure simulation - no real trading or external API calls
"""

import json
import time
import random
import math
from datetime import datetime, timedelta
from typing import Dict, List, Any
import argparse

class ConstitutionalFramework:
    """Kimera Constitutional Framework for decision validation"""
    
    def __init__(self):
        self.principles = {
            "unity": "Do no harm. Practice compassion.",
            "core_directive": "Generate profit for Kimera's development",
            "heart_over_head": "Compassionate decision making",
            "moderation": "Balanced, non-extreme approach",
            "prime_directive": "Universal compassion and connection"
        }
        
    def validate_decision(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Validate trading decision against constitutional principles"""
        
        score = 0.0
        violations = []
        
        # Unity Principle - Check risk level
        risk_level = decision.get('risk_level', 'unknown')
        if risk_level in ['low', 'minimal']:
            score += 0.3
        elif risk_level == 'medium':
            score += 0.2
        else:
            violations.append(f"Risk level '{risk_level}' may violate Unity Principle")
        
        # Moderation - Check position size
        amount = decision.get('amount', 0)
        if amount <= 0.01:
            score += 0.2
        elif amount <= 0.1:
            score += 0.15
        else:
            violations.append("Large position violates Moderation principle")
        
        # Heart over Head - Check confidence
        confidence = decision.get('confidence', 0)
        if confidence >= 0.7:
            score += 0.3
        elif confidence >= 0.5:
            score += 0.2
        else:
            violations.append("Low confidence may violate Heart over Head")
        
        # Core Directive - Check reasoning
        reasoning = decision.get('reasoning', '')
        if len(reasoning) > 30:
            score += 0.2
        else:
            violations.append("Insufficient reasoning")
        
        # Determine verdict
        if score >= 0.8:
            verdict = "CONSTITUTIONAL"
        elif score >= 0.6:
            verdict = "CONDITIONAL_APPROVAL"
        elif score >= 0.4:
            verdict = "REQUIRES_MODIFICATION"
        else:
            verdict = "UNCONSTITUTIONAL"
        
        return {
            "verdict": verdict,
            "score": score,
            "violations": violations,
            "constitutional_compliance": score >= 0.6
        }

class CognitiveAnalyzer:
    """Kimera Cognitive Analysis System"""
    
    def __init__(self):
        self.field_state = [random.random() for _ in range(256)]
        self.memory = []
        
    def analyze_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform cognitive analysis on market data"""
        
        # Extract price changes
        price_changes = market_data.get('price_changes', [])
        volumes = market_data.get('volumes', [])
        
        if not price_changes:
            return {"sentiment": "neutral", "confidence": 0.0}
        
        # Calculate cognitive metrics
        avg_change = sum(price_changes) / len(price_changes)
        volatility = math.sqrt(sum((x - avg_change) ** 2 for x in price_changes) / len(price_changes))
        momentum = sum(1 for x in price_changes if x > 0) / len(price_changes)
        
        # Update cognitive field
        market_energy = abs(avg_change) + volatility
        for i in range(min(len(self.field_state), 10)):
            self.field_state[i] = 0.9 * self.field_state[i] + 0.1 * market_energy
        
        # Determine sentiment
        if avg_change > 2:
            sentiment = "very_bullish"
        elif avg_change > 0.5:
            sentiment = "bullish"
        elif avg_change > -0.5:
            sentiment = "neutral"
        elif avg_change > -2:
            sentiment = "bearish"
        else:
            sentiment = "very_bearish"
        
        # Calculate confidence
        data_quality = min(len(price_changes) / 5.0, 1.0)
        consistency = max(0, 1.0 - volatility / 10.0)
        confidence = (data_quality + consistency) / 2.0
        
        # Store in memory
        analysis = {
            "sentiment": sentiment,
            "avg_change": avg_change,
            "volatility": volatility,
            "momentum": momentum,
            "confidence": confidence,
            "field_energy": sum(abs(x) for x in self.field_state[:10]),
            "timestamp": datetime.now().isoformat()
        }
        
        self.memory.append(analysis)
        if len(self.memory) > 10:
            self.memory.pop(0)
        
        return analysis

class RiskManager:
    """Kimera Risk Management System"""
    
    def __init__(self):
        self.max_position_size = 0.2  # 20%
        self.max_daily_risk = 0.05    # 5%
        self.min_confidence = 0.6     # 60%
        
    def assess_risk(self, decision: Dict[str, Any], portfolio_value: float) -> Dict[str, Any]:
        """Assess risk for trading decision"""
        
        risk_factors = []
        risk_score = 0.0
        
        # Position size risk
        amount = decision.get('amount', 0)
        price = decision.get('price', 40000)
        position_value = amount * price
        position_percent = position_value / portfolio_value
        
        if position_percent > self.max_position_size:
            risk_factors.append(f"Position {position_percent:.1%} > limit {self.max_position_size:.1%}")
            risk_score += 0.4
        
        # Confidence risk
        confidence = decision.get('confidence', 0)
        if confidence < self.min_confidence:
            risk_factors.append(f"Confidence {confidence:.1%} < minimum {self.min_confidence:.1%}")
            risk_score += 0.3
        
        # Volatility risk
        volatility = decision.get('market_volatility', 0)
        if volatility > 5:
            risk_factors.append(f"High volatility {volatility:.2f}")
            risk_score += 0.2
        
        # Determine risk level
        if risk_score >= 0.7:
            risk_level = "critical"
        elif risk_score >= 0.5:
            risk_level = "high"
        elif risk_score >= 0.3:
            risk_level = "medium"
        elif risk_score >= 0.1:
            risk_level = "low"
        else:
            risk_level = "minimal"
        
        return {
            "risk_level": risk_level,
            "risk_score": risk_score,
            "risk_factors": risk_factors,
            "approved": risk_score < 0.5
        }

class MarketDataSimulator:
    """Simulates realistic market data"""
    
    def __init__(self):
        self.symbols = ["BTC", "ETH", "BNB", "ADA", "SOL"]
        self.prices = {"BTC": 45000, "ETH": 3000, "BNB": 300, "ADA": 0.5, "SOL": 100}
        self.time_step = 0
        
    def generate_market_data(self) -> Dict[str, Any]:
        """Generate realistic market data"""
        
        self.time_step += 1
        price_changes = []
        volumes = []
        
        for symbol in self.symbols:
            # Generate realistic price movement
            trend = math.sin(self.time_step * 0.1) * 0.5  # Long-term trend
            noise = random.gauss(0, 2)  # Random noise
            change = trend + noise
            
            price_changes.append(change)
            volumes.append(random.uniform(1000000, 10000000))
            
            # Update price
            self.prices[symbol] *= (1 + change / 100)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "symbols": self.symbols,
            "prices": self.prices.copy(),
            "price_changes": price_changes,
            "volumes": volumes,
            "market_cap": sum(self.prices.values()) * 1000000
        }

class KimeraDemoTestnet:
    """Kimera Demo Testnet Simulation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.constitutional_framework = ConstitutionalFramework()
        self.cognitive_analyzer = CognitiveAnalyzer()
        self.risk_manager = RiskManager()
        self.market_simulator = MarketDataSimulator()
        
        # Simulation state
        self.start_time = datetime.now()
        self.iterations = 0
        self.successful_decisions = 0
        self.failed_decisions = 0
        self.constitutional_violations = 0
        self.risk_violations = 0
        self.decision_history = []
        
        # Portfolio
        self.portfolio_value = config.get('starting_capital', 1000)
        self.initial_value = self.portfolio_value
        
    def display_banner(self):
        """Display simulation banner"""
        print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                   🧪 KIMERA DEMO TESTNET SIMULATION 🧪                       ║
║                                                                              ║
║                Constitutional AI • Cognitive Trading • Zero Risk            ║
║                                                                              ║
║  🧠 Cognitive Analysis: Advanced Field Dynamics                             ║
║  🏛️ Constitutional: Full Ethical Framework                                   ║
║  🛡️ Risk Management: Multi-Layer Validation                                 ║
║  📊 Performance: Real-Time Metrics                                          ║
║  🎯 Demo Mode: Safe Simulation Environment                                  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """)
        
    def display_config(self):
        """Display configuration"""
        print(f"🔧 SIMULATION CONFIGURATION:")
        print(f"   • Duration: {self.config.get('duration_minutes', 5)} minutes")
        print(f"   • Update Interval: {self.config.get('update_interval', 10)} seconds")
        print(f"   • Starting Capital: ${self.config.get('starting_capital', 1000)} (simulated)")
        print(f"   • Constitutional Compliance: {'ENABLED' if self.config.get('constitutional', True) else 'DISABLED'}")
        print(f"   • Risk Management: {'ENABLED' if self.config.get('risk_management', True) else 'DISABLED'}")
        print(f"   • Market Data: SIMULATED (realistic)")
        print()
        
    def run_simulation_cycle(self):
        """Run one simulation cycle"""
        self.iterations += 1
        print(f"🔄 Cycle {self.iterations} - {datetime.now().strftime('%H:%M:%S')}")
        
        # 1. Generate market data
        market_data = self.market_simulator.generate_market_data()
        print(f"📊 Market Data: BTC ${market_data['prices']['BTC']:.0f} ({market_data['price_changes'][0]:+.2f}%)")
        
        # 2. Cognitive analysis
        cognitive_analysis = self.cognitive_analyzer.analyze_market_data(market_data)
        print(f"🧠 Cognitive: {cognitive_analysis['sentiment']} (confidence: {cognitive_analysis['confidence']:.1%})")
        
        # 3. Generate decisions
        decisions = self.generate_decisions(cognitive_analysis, market_data)
        print(f"💡 Generated {len(decisions)} trading decisions")
        
        # 4. Constitutional validation
        for decision in decisions:
            constitutional_result = self.constitutional_framework.validate_decision(decision)
            decision['constitutional'] = constitutional_result
            
            if constitutional_result['constitutional_compliance']:
                print(f"⚖️  Constitutional: {constitutional_result['verdict']} ✅")
            else:
                self.constitutional_violations += 1
                print(f"⚖️  Constitutional: {constitutional_result['verdict']} ❌")
                continue
            
            # 5. Risk management
            risk_result = self.risk_manager.assess_risk(decision, self.portfolio_value)
            decision['risk'] = risk_result
            
            if risk_result['approved']:
                print(f"🛡️  Risk: {risk_result['risk_level']} - APPROVED ✅")
                self.execute_trade(decision)
                self.successful_decisions += 1
            else:
                print(f"🛡️  Risk: {risk_result['risk_level']} - REJECTED ❌")
                self.risk_violations += 1
                self.failed_decisions += 1
        
        self.decision_history.extend(decisions)
        print(f"💰 Portfolio: ${self.portfolio_value:.2f}")
        print("-" * 60)
        
    def generate_decisions(self, cognitive_analysis: Dict[str, Any], market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate trading decisions"""
        decisions = []
        
        sentiment = cognitive_analysis['sentiment']
        confidence = cognitive_analysis['confidence']
        volatility = cognitive_analysis['volatility']
        
        if confidence < 0.4:
            return decisions
        
        # Generate decision based on sentiment
        if sentiment in ['bullish', 'very_bullish'] and volatility < 3:
            decisions.append({
                "action": "buy",
                "symbol": "BTC",
                "amount": 0.001,
                "price": market_data['prices']['BTC'],
                "confidence": confidence,
                "reasoning": f"Bullish sentiment with low volatility ({volatility:.2f})",
                "market_volatility": volatility,
                "risk_level": "low" if volatility < 2 else "medium"
            })
        elif sentiment in ['bearish', 'very_bearish'] and volatility < 3:
            decisions.append({
                "action": "sell",
                "symbol": "ETH",
                "amount": 0.01,
                "price": market_data['prices']['ETH'],
                "confidence": confidence,
                "reasoning": f"Bearish sentiment with low volatility ({volatility:.2f})",
                "market_volatility": volatility,
                "risk_level": "low" if volatility < 2 else "medium"
            })
        
        return decisions
        
    def execute_trade(self, decision: Dict[str, Any]):
        """Execute trade simulation"""
        action = decision['action']
        amount = decision['amount']
        price = decision['price']
        
        # Simulate slippage
        slippage = random.uniform(-0.001, 0.001)
        execution_price = price * (1 + slippage)
        
        trade_value = amount * execution_price
        fee = trade_value * 0.001  # 0.1% fee
        
        if action == 'buy':
            self.portfolio_value -= (trade_value + fee)
        else:
            self.portfolio_value += (trade_value - fee)
        
        print(f"💰 Executed {action}: {amount} {decision['symbol']} @ ${execution_price:.2f}")
        
    def run_simulation(self):
        """Run complete simulation"""
        duration = self.config.get('duration_minutes', 5)
        interval = self.config.get('update_interval', 10)
        
        print(f"🚀 Starting {duration}-minute simulation with {interval}s intervals...")
        print("="*60)
        
        end_time = datetime.now() + timedelta(minutes=duration)
        
        try:
            while datetime.now() < end_time:
                self.run_simulation_cycle()
                
                if self.iterations >= self.config.get('max_iterations', 50):
                    print("🏁 Maximum iterations reached")
                    break
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n⏹️ Simulation interrupted by user")
        
        self.generate_report()
        
    def generate_report(self):
        """Generate simulation report"""
        runtime = datetime.now() - self.start_time
        portfolio_return = ((self.portfolio_value - self.initial_value) / self.initial_value) * 100
        success_rate = (self.successful_decisions / max(self.iterations, 1)) * 100
        
        report = {
            "kimera_demo_testnet_report": {
                "simulation_metadata": {
                    "start_time": self.start_time.isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "duration_minutes": runtime.total_seconds() / 60,
                    "configuration": self.config
                },
                "performance_metrics": {
                    "total_iterations": self.iterations,
                    "successful_decisions": self.successful_decisions,
                    "failed_decisions": self.failed_decisions,
                    "constitutional_violations": self.constitutional_violations,
                    "risk_violations": self.risk_violations,
                    "success_rate_percent": success_rate,
                    "portfolio_return_percent": portfolio_return,
                    "final_portfolio_value": self.portfolio_value
                },
                "system_validation": {
                    "cognitive_analysis": self.iterations > 0,
                    "constitutional_compliance": self.constitutional_violations == 0,
                    "risk_management": self.risk_violations < self.iterations * 0.2,
                    "decision_generation": len(self.decision_history) > 0
                }
            }
        }
        
        # Save report
        filename = f"kimera_demo_testnet_report_{int(datetime.now().timestamp())}.json"
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Display summary
        print("\n" + "="*80)
        print("🎯 KIMERA DEMO TESTNET SIMULATION SUMMARY")
        print("="*80)
        print(f"⏱️  Duration: {runtime.total_seconds()/60:.1f} minutes")
        print(f"🔄 Iterations: {self.iterations}")
        print(f"✅ Successful Decisions: {self.successful_decisions}")
        print(f"❌ Failed Decisions: {self.failed_decisions}")
        print(f"⚖️  Constitutional Violations: {self.constitutional_violations}")
        print(f"🛡️  Risk Violations: {self.risk_violations}")
        print(f"📊 Success Rate: {success_rate:.1f}%")
        print(f"💰 Portfolio Return: {portfolio_return:.2f}%")
        print(f"📄 Report: {filename}")
        print("="*80)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Kimera Demo Testnet Simulation")
    parser.add_argument("--duration", type=int, default=5, help="Duration in minutes")
    parser.add_argument("--interval", type=int, default=10, help="Update interval in seconds")
    parser.add_argument("--capital", type=float, default=1000, help="Starting capital")
    parser.add_argument("--no-constitutional", action="store_true", help="Disable constitutional checks")
    parser.add_argument("--no-risk", action="store_true", help="Disable risk management")
    
    args = parser.parse_args()
    
    config = {
        "duration_minutes": args.duration,
        "update_interval": args.interval,
        "starting_capital": args.capital,
        "constitutional": not args.no_constitutional,
        "risk_management": not args.no_risk,
        "max_iterations": 50
    }
    
    simulation = KimeraDemoTestnet(config)
    simulation.display_banner()
    simulation.display_config()
    simulation.run_simulation()

if __name__ == "__main__":
    main() 