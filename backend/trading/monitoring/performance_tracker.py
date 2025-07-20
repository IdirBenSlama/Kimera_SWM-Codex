"""
Performance Tracker

Tracks and analyzes trading performance with Kimera's cognitive insights.
Provides real-time metrics and decision analysis.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
import json

from backend.trading.core.trading_engine import TradingDecision

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Record of a single trade"""
    timestamp: datetime
    symbol: str
    side: str  # BUY/SELL
    quantity: float
    price: float
    decision: TradingDecision
    order_id: str
    pnl: Optional[float] = None
    cognitive_accuracy: Optional[float] = None


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    cognitive_alignment: float
    decision_accuracy: float


class PerformanceTracker:
    """
    Tracks trading performance and provides analytics.
    Integrates with Kimera's cognitive analysis for decision evaluation.
    """
    
    def __init__(self):
        """Initialize performance tracker"""
        self.trades: List[TradeRecord] = []
        self.daily_returns: List[float] = []
        self.equity_curve: List[float] = []
        self.metrics_history: List[Dict[str, Any]] = []
        
        # Cognitive performance tracking
        self.decision_outcomes: Dict[str, List[float]] = {
            "high_confidence": [],
            "medium_confidence": [],
            "low_confidence": []
        }
        
        self.cognitive_factors: Dict[str, List[float]] = {
            "cognitive_pressure": [],
            "contradiction_level": [],
            "semantic_temperature": []
        }
        
        logger.info("Performance tracker initialized")
    
    async def record_decision(
        self, 
        symbol: str, 
        decision: TradingDecision,
        order: Dict[str, Any]
    ) -> None:
        """Record a trading decision and order"""
        trade = TradeRecord(
            timestamp=datetime.now(),
            symbol=symbol,
            side=decision.action,
            quantity=float(order.get("executedQty", 0)),
            price=float(order.get("price", order.get("avgPrice", 0))),
            decision=decision,
            order_id=str(order.get("orderId", ""))
        )
        
        self.trades.append(trade)
        
        # Track cognitive factors
        if decision.confidence > 0.7:
            confidence_level = "high_confidence"
        elif decision.confidence > 0.4:
            confidence_level = "medium_confidence"
        else:
            confidence_level = "low_confidence"
        
        # We'll update with actual outcome later
        self.decision_outcomes[confidence_level].append(0.0)
        
        logger.info(f"Recorded trade: {symbol} {decision.action} @ {trade.price}")
    
    async def update_trade_outcome(
        self, 
        order_id: str, 
        exit_price: float,
        exit_quantity: float
    ) -> None:
        """Update trade with exit information and calculate P&L"""
        for trade in reversed(self.trades):
            if trade.order_id == order_id and trade.pnl is None:
                # Calculate P&L
                if trade.side == "BUY":
                    trade.pnl = (exit_price - trade.price) * exit_quantity
                else:
                    trade.pnl = (trade.price - exit_price) * exit_quantity
                
                # Calculate cognitive accuracy
                if trade.pnl > 0:
                    trade.cognitive_accuracy = trade.decision.confidence
                else:
                    trade.cognitive_accuracy = 1.0 - trade.decision.confidence
                
                # Update decision outcomes
                confidence_level = self._get_confidence_level(trade.decision.confidence)
                outcome = 1.0 if trade.pnl > 0 else -1.0
                self.decision_outcomes[confidence_level][-1] = outcome
                
                # Update equity curve
                current_equity = self.equity_curve[-1] if self.equity_curve else 0.0
                self.equity_curve.append(current_equity + trade.pnl)
                
                logger.info(
                    f"Trade closed: {trade.symbol} P&L: ${trade.pnl:.2f}, "
                    f"Cognitive accuracy: {trade.cognitive_accuracy:.2f}"
                )
                break
    
    def calculate_metrics(self, period_days: Optional[int] = None) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        # Filter trades by period if specified
        if period_days:
            cutoff_date = datetime.now() - timedelta(days=period_days)
            filtered_trades = [t for t in self.trades if t.timestamp >= cutoff_date]
        else:
            filtered_trades = self.trades
        
        # Basic metrics
        total_trades = len([t for t in filtered_trades if t.pnl is not None])
        
        if total_trades == 0:
            return PerformanceMetrics(
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                total_pnl=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                profit_factor=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                max_drawdown=0.0,
                calmar_ratio=0.0,
                cognitive_alignment=0.0,
                decision_accuracy=0.0
            )
        
        # Win/Loss analysis
        winning_trades = [t for t in filtered_trades if t.pnl and t.pnl > 0]
        losing_trades = [t for t in filtered_trades if t.pnl and t.pnl < 0]
        
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        win_rate = (win_count / total_trades) * 100
        
        # P&L analysis
        total_pnl = sum(t.pnl for t in filtered_trades if t.pnl)
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0.0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0.0
        
        # Profit factor
        gross_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0.0
        gross_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 1.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        
        # Risk-adjusted returns
        returns = self._calculate_returns(filtered_trades)
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        sortino_ratio = self._calculate_sortino_ratio(returns)
        
        # Drawdown analysis
        max_drawdown = self._calculate_max_drawdown()
        calmar_ratio = (total_pnl / max_drawdown) if max_drawdown > 0 else 0.0
        
        # Cognitive performance
        cognitive_alignment = self._calculate_cognitive_alignment(filtered_trades)
        decision_accuracy = self._calculate_decision_accuracy(filtered_trades)
        
        metrics = PerformanceMetrics(
            total_trades=total_trades,
            winning_trades=win_count,
            losing_trades=loss_count,
            win_rate=win_rate,
            total_pnl=total_pnl,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            cognitive_alignment=cognitive_alignment,
            decision_accuracy=decision_accuracy
        )
        
        # Store metrics history
        self.metrics_history.append({
            "timestamp": datetime.now(),
            "metrics": asdict(metrics)
        })
        
        return metrics
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        metrics = self.calculate_metrics()
        
        # Cognitive analysis
        cognitive_analysis = self._analyze_cognitive_performance()
        
        # Time-based analysis
        daily_metrics = self.calculate_metrics(period_days=1)
        weekly_metrics = self.calculate_metrics(period_days=7)
        monthly_metrics = self.calculate_metrics(period_days=30)
        
        return {
            "overall_metrics": asdict(metrics),
            "daily_metrics": asdict(daily_metrics),
            "weekly_metrics": asdict(weekly_metrics),
            "monthly_metrics": asdict(monthly_metrics),
            "cognitive_analysis": cognitive_analysis,
            "equity_curve": self.equity_curve[-100:],  # Last 100 points
            "recent_trades": [
                {
                    "timestamp": t.timestamp.isoformat(),
                    "symbol": t.symbol,
                    "side": t.side,
                    "price": t.price,
                    "pnl": t.pnl,
                    "confidence": t.decision.confidence,
                    "reasoning": t.decision.reasoning
                }
                for t in self.trades[-10:]  # Last 10 trades
            ]
        }
    
    def _calculate_returns(self, trades: List[TradeRecord]) -> List[float]:
        """Calculate returns from trades"""
        returns = []
        equity = 0.0
        
        for trade in trades:
            if trade.pnl is not None:
                previous_equity = equity
                equity += trade.pnl
                
                if previous_equity > 0:
                    returns.append(trade.pnl / previous_equity)
        
        return returns
    
    def _calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio"""
        if not returns:
            return 0.0
        
        excess_returns = [r - risk_free_rate for r in returns]
        mean_return = np.mean(excess_returns)
        std_return = np.std(excess_returns)
        
        if std_return == 0:
            return 0.0
        
        # Annualize (assuming daily returns)
        return (mean_return / std_return) * np.sqrt(252)
    
    def _calculate_sortino_ratio(self, returns: List[float], risk_free_rate: float = 0.0) -> float:
        """Calculate Sortino ratio"""
        if not returns:
            return 0.0
        
        excess_returns = [r - risk_free_rate for r in returns]
        mean_return = np.mean(excess_returns)
        
        # Downside deviation
        negative_returns = [r for r in excess_returns if r < 0]
        if not negative_returns:
            return float('inf') if mean_return > 0 else 0.0
        
        downside_std = np.std(negative_returns)
        
        if downside_std == 0:
            return 0.0
        
        # Annualize
        return (mean_return / downside_std) * np.sqrt(252)
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        if not self.equity_curve:
            return 0.0
        
        peak = self.equity_curve[0]
        max_dd = 0.0
        
        for value in self.equity_curve:
            if value > peak:
                peak = value
            
            drawdown = (peak - value) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def _calculate_cognitive_alignment(self, trades: List[TradeRecord]) -> float:
        """Calculate how well cognitive analysis aligned with outcomes"""
        if not trades:
            return 0.0
        
        alignments = []
        
        for trade in trades:
            if trade.cognitive_accuracy is not None:
                alignments.append(trade.cognitive_accuracy)
        
        return np.mean(alignments) if alignments else 0.0
    
    def _calculate_decision_accuracy(self, trades: List[TradeRecord]) -> float:
        """Calculate decision accuracy by confidence level"""
        correct_decisions = 0
        total_decisions = 0
        
        for trade in trades:
            if trade.pnl is not None:
                total_decisions += 1
                
                # High confidence should result in profit
                if trade.decision.confidence > 0.7 and trade.pnl > 0:
                    correct_decisions += 1
                # Low confidence trades that we avoided (HOLD) count as correct
                elif trade.decision.confidence < 0.3 and trade.decision.action == "HOLD":
                    correct_decisions += 1
                # Medium confidence with small profit/loss is acceptable
                elif 0.3 <= trade.decision.confidence <= 0.7:
                    if abs(trade.pnl) < (trade.price * trade.quantity * 0.01):  # 1% threshold
                        correct_decisions += 0.5
        
        return (correct_decisions / total_decisions * 100) if total_decisions > 0 else 0.0
    
    def _analyze_cognitive_performance(self) -> Dict[str, Any]:
        """Analyze performance by cognitive factors"""
        analysis = {
            "confidence_performance": {},
            "best_conditions": {},
            "worst_conditions": {},
            "recommendations": []
        }
        
        # Analyze by confidence level
        for level, outcomes in self.decision_outcomes.items():
            if outcomes:
                win_rate = sum(1 for o in outcomes if o > 0) / len(outcomes) * 100
                analysis["confidence_performance"][level] = {
                    "trades": len(outcomes),
                    "win_rate": win_rate,
                    "avg_outcome": np.mean(outcomes)
                }
        
        # Find best performing conditions
        profitable_trades = [t for t in self.trades if t.pnl and t.pnl > 0]
        if profitable_trades:
            avg_pressure = np.mean([t.decision.cognitive_alignment for t in profitable_trades])
            analysis["best_conditions"] = {
                "avg_cognitive_alignment": avg_pressure,
                "common_reasoning": self._most_common_reasoning(profitable_trades)
            }
        
        # Find worst performing conditions
        losing_trades = [t for t in self.trades if t.pnl and t.pnl < 0]
        if losing_trades:
            avg_risk = np.mean([t.decision.risk_score for t in losing_trades])
            analysis["worst_conditions"] = {
                "avg_risk_score": avg_risk,
                "common_reasoning": self._most_common_reasoning(losing_trades)
            }
        
        # Generate recommendations
        if analysis["confidence_performance"].get("high_confidence", {}).get("win_rate", 0) < 60:
            analysis["recommendations"].append(
                "High confidence trades underperforming - review confidence calculation"
            )
        
        if analysis["confidence_performance"].get("low_confidence", {}).get("trades", 0) > 30:
            analysis["recommendations"].append(
                "Too many low confidence trades - consider stricter filtering"
            )
        
        return analysis
    
    def _most_common_reasoning(self, trades: List[TradeRecord]) -> List[str]:
        """Find most common reasoning patterns"""
        reasoning_counts = {}
        
        for trade in trades:
            for reason in trade.decision.reasoning:
                reasoning_counts[reason] = reasoning_counts.get(reason, 0) + 1
        
        # Sort by frequency
        sorted_reasons = sorted(
            reasoning_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return [reason for reason, _ in sorted_reasons[:3]]
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Get confidence level category"""
        if confidence > 0.7:
            return "high_confidence"
        elif confidence > 0.4:
            return "medium_confidence"
        else:
            return "low_confidence"
    
    def export_performance_report(self, filepath: str) -> None:
        """Export detailed performance report"""
        report = {
            "generated_at": datetime.now().isoformat(),
            "summary": self.get_performance_summary(),
            "trades": [
                {
                    "timestamp": t.timestamp.isoformat(),
                    "symbol": t.symbol,
                    "side": t.side,
                    "quantity": t.quantity,
                    "price": t.price,
                    "pnl": t.pnl,
                    "decision": {
                        "action": t.decision.action,
                        "confidence": t.decision.confidence,
                        "reasoning": t.decision.reasoning,
                        "cognitive_alignment": t.decision.cognitive_alignment,
                        "expected_return": t.decision.expected_return
                    },
                    "cognitive_accuracy": t.cognitive_accuracy
                }
                for t in self.trades
            ],
            "metrics_history": self.metrics_history
        }
        
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Performance report exported to {filepath}") 