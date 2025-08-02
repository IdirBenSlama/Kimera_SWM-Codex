import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum

# Add the backend directory to the sys.path to allow for engine imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.trading.models import Order
from src.trading.portfolio import Portfolio

class RiskLevel(Enum):
    """Risk level classification"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"

@dataclass
class RiskMetrics:
    """Comprehensive risk metrics"""
    var_95: float  # Value at Risk 95%
    var_99: float  # Value at Risk 99%
    cvar_95: float  # Conditional Value at Risk 95%
    max_drawdown: float  # Maximum drawdown
    current_drawdown: float  # Current drawdown
    sharpe_ratio: float  # Sharpe ratio
    sortino_ratio: float  # Sortino ratio
    volatility: float  # Annualized volatility
    beta: float  # Market beta
    risk_level: RiskLevel  # Overall risk level

class AdvancedRiskManager:
    """
    Advanced risk management system with VaR calculations and comprehensive controls
    """
    
    def __init__(self, 
                 portfolio: Portfolio = None, 
                 max_position_pct: float = 0.2,
                 max_portfolio_risk: float = 0.05,
                 max_drawdown_limit: float = 0.10,
                 var_confidence_level: float = 0.95,
                 lookback_window: int = 252):
        """
        Initialize Advanced Risk Manager
        
        Args:
            portfolio: Portfolio instance to manage (optional)
            max_position_pct: Maximum position size as percentage of portfolio
            max_portfolio_risk: Maximum portfolio risk (daily VaR)
            max_drawdown_limit: Maximum allowed drawdown
            var_confidence_level: Confidence level for VaR calculations
            lookback_window: Historical lookback window for risk calculations
        """
        self.portfolio = portfolio
        self.max_position_pct = max_position_pct
        self.max_portfolio_risk = max_portfolio_risk
        self.max_drawdown_limit = max_drawdown_limit
        self.var_confidence_level = var_confidence_level
        self.lookback_window = lookback_window
        
        # Risk tracking
        self.price_history = {}
        self.return_history = {}
        self.portfolio_value_history = []
        self.max_portfolio_value = 0
        self.risk_metrics_history = []
        
        # Risk-free rate (annualized)
        self.risk_free_rate = 0.02
        
        logging.info(f"Advanced Risk Manager initialized:")
        logging.info(f"  - Max position risk: {self.max_position_pct:.1%}")
        logging.info(f"  - Max portfolio risk: {self.max_portfolio_risk:.1%}")
        logging.info(f"  - Max drawdown limit: {self.max_drawdown_limit:.1%}")
        logging.info(f"  - VaR confidence level: {self.var_confidence_level:.1%}")

    def update_market_data(self, prices: Dict[str, float], timestamp: datetime = None):
        """Update market data for risk calculations"""
        if timestamp is None:
            timestamp = datetime.now()
        
        for symbol, price in prices.items():
            if symbol not in self.price_history:
                self.price_history[symbol] = []
            
            self.price_history[symbol].append({
                'timestamp': timestamp,
                'price': price
            })
            
            # Keep only recent data
            if len(self.price_history[symbol]) > self.lookback_window:
                self.price_history[symbol].pop(0)
        
        # Update returns
        self._update_returns()
        
        # Update portfolio value history
        current_value = self.portfolio.get_total_value(prices)
        self.portfolio_value_history.append({
            'timestamp': timestamp,
            'value': current_value
        })
        
        # Track maximum portfolio value
        self.max_portfolio_value = max(self.max_portfolio_value, current_value)
        
        # Keep only recent portfolio values
        if len(self.portfolio_value_history) > self.lookback_window:
            self.portfolio_value_history.pop(0)
    
    def _update_returns(self):
        """Calculate returns from price history"""
        for symbol, prices in self.price_history.items():
            if len(prices) >= 2:
                returns = []
                for i in range(1, len(prices)):
                    ret = (prices[i]['price'] - prices[i-1]['price']) / prices[i-1]['price']
                    returns.append(ret)
                self.return_history[symbol] = np.array(returns)
    
    def calculate_var(self, returns: np.ndarray, confidence_level: float = 0.95, method: str = 'historical') -> float:
        """
        Calculate Value at Risk (VaR)
        
        Args:
            returns: Array of returns
            confidence_level: Confidence level (0.95 for 95% VaR)
            method: Calculation method ('historical', 'parametric', 'monte_carlo')
            
        Returns:
            VaR value (positive number representing loss)
        """
        if len(returns) < 10:
            return 0.0
        
        try:
            if method == 'historical':
                # Historical simulation method
                return -np.percentile(returns, (1 - confidence_level) * 100)
            
            elif method == 'parametric':
                # Parametric method (assuming normal distribution)
                from scipy.stats import norm
                mean = np.mean(returns)
                std = np.std(returns)
                return -(mean + norm.ppf(1 - confidence_level) * std)
            
            elif method == 'monte_carlo':
                # Monte Carlo simulation
                np.random.seed(42)
                mean = np.mean(returns)
                std = np.std(returns)
                simulated_returns = np.random.normal(mean, std, 10000)
                return -np.percentile(simulated_returns, (1 - confidence_level) * 100)
            
            else:
                return self.calculate_var(returns, confidence_level, 'historical')
                
        except Exception as e:
            logging.error(f"Error calculating VaR: {str(e)}")
            return 0.0
    
    def calculate_cvar(self, returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall)
        
        Args:
            returns: Array of returns
            confidence_level: Confidence level
            
        Returns:
            CVaR value (positive number representing expected loss)
        """
        if len(returns) < 10:
            return 0.0
        
        try:
            var_threshold = -self.calculate_var(returns, confidence_level)
            tail_losses = returns[returns <= var_threshold]
            
            if len(tail_losses) > 0:
                return -np.mean(tail_losses)
            else:
                return -var_threshold
                
        except Exception as e:
            logging.error(f"Error calculating CVaR: {str(e)}")
            return 0.0
    
    def calculate_drawdown(self) -> Tuple[float, float]:
        """
        Calculate current and maximum drawdown
        
        Returns:
            Tuple of (current_drawdown, max_drawdown)
        """
        if len(self.portfolio_value_history) < 2:
            return 0.0, 0.0
        
        values = [entry['value'] for entry in self.portfolio_value_history]
        
        # Calculate running maximum
        running_max = []
        current_max = values[0]
        for value in values:
            current_max = max(current_max, value)
            running_max.append(current_max)
        
        # Calculate drawdowns
        drawdowns = [(values[i] - running_max[i]) / running_max[i] for i in range(len(values))]
        
        current_drawdown = abs(drawdowns[-1]) if drawdowns else 0.0
        max_drawdown = abs(min(drawdowns)) if drawdowns else 0.0
        
        return current_drawdown, max_drawdown
    
    def calculate_risk_metrics(self, prices: Dict[str, float]) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        
        # Portfolio returns
        portfolio_returns = self._calculate_portfolio_returns()
        
        # VaR calculations
        var_95 = self.calculate_var(portfolio_returns, 0.95)
        var_99 = self.calculate_var(portfolio_returns, 0.99)
        cvar_95 = self.calculate_cvar(portfolio_returns, 0.95)
        
        # Drawdown calculations
        current_drawdown, max_drawdown = self.calculate_drawdown()
        
        # Performance metrics
        sharpe_ratio = self._calculate_sharpe_ratio(portfolio_returns)
        sortino_ratio = self._calculate_sortino_ratio(portfolio_returns)
        volatility = np.std(portfolio_returns) * np.sqrt(252) if len(portfolio_returns) > 0 else 0.0
        beta = self._calculate_beta(portfolio_returns)
        
        # Risk level classification
        risk_level = self._classify_risk_level(var_95, current_drawdown, volatility)
        
        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            max_drawdown=max_drawdown,
            current_drawdown=current_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            volatility=volatility,
            beta=beta,
            risk_level=risk_level
        )
    
    def _calculate_portfolio_returns(self) -> np.ndarray:
        """Calculate portfolio returns from value history"""
        if len(self.portfolio_value_history) < 2:
            return np.array([])
        
        values = [entry['value'] for entry in self.portfolio_value_history]
        returns = []
        
        for i in range(1, len(values)):
            if values[i-1] != 0:
                ret = (values[i] - values[i-1]) / values[i-1]
                returns.append(ret)
        
        return np.array(returns)
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 10:
            return 0.0
        
        excess_returns = returns - (self.risk_free_rate / 252)  # Daily risk-free rate
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        if len(returns) < 10:
            return 0.0
        
        excess_returns = returns - (self.risk_free_rate / 252)
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf')
        
        downside_deviation = np.std(downside_returns)
        
        if downside_deviation == 0:
            return 0.0
        
        return np.mean(excess_returns) / downside_deviation * np.sqrt(252)
    
    def _calculate_beta(self, returns: np.ndarray) -> float:
        """Calculate market beta (simplified - assumes market return is available)"""
        # For now, return 1.0 as default beta
        # In a full implementation, this would correlate with market returns
        return 1.0
    
    def _classify_risk_level(self, var_95: float, current_drawdown: float, volatility: float) -> RiskLevel:
        """Classify overall risk level"""
        
        # Risk score based on multiple factors
        risk_score = 0
        
        if var_95 > 0.05:  # 5% daily VaR
            risk_score += 3
        elif var_95 > 0.03:  # 3% daily VaR
            risk_score += 2
        elif var_95 > 0.01:  # 1% daily VaR
            risk_score += 1
        
        if current_drawdown > 0.15:  # 15% drawdown
            risk_score += 3
        elif current_drawdown > 0.10:  # 10% drawdown
            risk_score += 2
        elif current_drawdown > 0.05:  # 5% drawdown
            risk_score += 1
        
        if volatility > 0.40:  # 40% annualized volatility
            risk_score += 3
        elif volatility > 0.25:  # 25% annualized volatility
            risk_score += 2
        elif volatility > 0.15:  # 15% annualized volatility
            risk_score += 1
        
        if risk_score >= 7:
            return RiskLevel.EXTREME
        elif risk_score >= 5:
            return RiskLevel.HIGH
        elif risk_score >= 3:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def check_order(self, order: Order, current_prices: Dict[str, float]) -> Tuple[bool, str]:
        """
        Enhanced order validation with comprehensive risk checks
        
        Args:
            order: Order to validate
            current_prices: Current market prices
            
        Returns:
            Tuple of (is_valid, reason)
        """
        try:
            # Get current price
            price = current_prices.get(order.ticker, 0)
            if price <= 0:
                return False, f"Invalid price for {order.ticker}: {price}"
            
            order_cost = order.quantity * price
            
            # Rule 1: Insufficient Funds
            if order.side == 'buy' and order_cost > self.portfolio.cash:
                return False, f"Insufficient funds for {order.ticker} buy order. Required: ${order_cost:,.2f}, Available: ${self.portfolio.cash:,.2f}"
            
            # Rule 2: Maximum Position Size
            portfolio_value = self.portfolio.get_total_value(current_prices)
            if portfolio_value <= 0:
                return False, "Portfolio value is zero or negative"
            
            max_position_value = portfolio_value * self.max_position_pct
            
            current_position_value = 0
            if order.ticker in self.portfolio.positions:
                current_position_value = abs(self.portfolio.positions[order.ticker].quantity) * price
            
            if order.side == 'buy':
                projected_position_value = current_position_value + order_cost
            else:  # sell
                projected_position_value = max(0, current_position_value - order_cost)
            
            if projected_position_value > max_position_value:
                return False, f"Order for {order.ticker} exceeds max position size. Projected: ${projected_position_value:,.2f}, Max: ${max_position_value:,.2f}"
            
            # Rule 3: Portfolio Risk Limit
            risk_metrics = self.calculate_risk_metrics(current_prices)
            if risk_metrics.var_95 > self.max_portfolio_risk:
                return False, f"Portfolio VaR ({risk_metrics.var_95:.2%}) exceeds limit ({self.max_portfolio_risk:.2%})"
            
            # Rule 4: Drawdown Limit
            if risk_metrics.current_drawdown > self.max_drawdown_limit:
                return False, f"Current drawdown ({risk_metrics.current_drawdown:.2%}) exceeds limit ({self.max_drawdown_limit:.2%})"
            
            # Rule 5: Extreme Risk Level
            if risk_metrics.risk_level == RiskLevel.EXTREME:
                return False, "Portfolio risk level is EXTREME - no new positions allowed"
            
            # Rule 6: Order Size Validation
            if order.quantity <= 0:
                return False, "Order quantity must be positive"
            
            # Rule 7: Price Validation
            if price <= 0:
                return False, f"Invalid price: {price}"
            
            logging.info(f"RISK CHECK PASSED: Order for {order.ticker} is within risk limits")
            return True, "Order approved"
            
        except Exception as e:
            logging.error(f"Error in risk check: {str(e)}")
            return False, f"Risk check failed: {str(e)}"
    
    def get_position_size_recommendation(self, 
                                       symbol: str, 
                                       confidence: float,
                                       current_prices: Dict[str, float],
                                       strategy_risk: float = 0.01) -> float:
        """
        Recommend position size based on risk management principles
        
        Args:
            symbol: Trading symbol
            confidence: Signal confidence (0-1)
            current_prices: Current market prices
            strategy_risk: Risk per trade (fraction of portfolio)
            
        Returns:
            Recommended position size in USD
        """
        try:
            portfolio_value = self.portfolio.get_total_value(current_prices)
            
            # Base risk per trade
            base_risk = portfolio_value * strategy_risk
            
            # Adjust based on confidence
            confidence_adjusted_risk = base_risk * confidence
            
            # Apply portfolio risk limits
            max_position_value = portfolio_value * self.max_position_pct
            
            # Get current risk metrics
            risk_metrics = self.calculate_risk_metrics(current_prices)
            
            # Risk adjustment multiplier based on current risk level
            risk_multiplier = {
                RiskLevel.LOW: 1.0,
                RiskLevel.MEDIUM: 0.8,
                RiskLevel.HIGH: 0.5,
                RiskLevel.EXTREME: 0.1
            }.get(risk_metrics.risk_level, 0.5)
            
            # Calculate final position size
            recommended_size = min(
                confidence_adjusted_risk * risk_multiplier,
                max_position_value
            )
            
            return max(0, recommended_size)
            
        except Exception as e:
            logging.error(f"Error calculating position size: {str(e)}")
            return 0.0
    
    def should_close_position(self, symbol: str, current_prices: Dict[str, float]) -> Tuple[bool, str]:
        """
        Check if a position should be closed due to risk management
        
        Args:
            symbol: Trading symbol
            current_prices: Current market prices
            
        Returns:
            Tuple of (should_close, reason)
        """
        try:
            if symbol not in self.portfolio.positions:
                return False, "No position to close"
            
            # Check drawdown limits
            risk_metrics = self.calculate_risk_metrics(current_prices)
            
            if risk_metrics.current_drawdown > self.max_drawdown_limit:
                return True, f"Drawdown limit exceeded: {risk_metrics.current_drawdown:.2%}"
            
            if risk_metrics.risk_level == RiskLevel.EXTREME:
                return True, "Portfolio risk level is EXTREME"
            
            # Check position-specific risk (if we have return history)
            if symbol in self.return_history and len(self.return_history[symbol]) > 10:
                position_var = self.calculate_var(self.return_history[symbol], 0.95)
                if position_var > 0.10:  # 10% daily VaR for single position
                    return True, f"Position VaR too high: {position_var:.2%}"
            
            return False, "Position within risk limits"
            
        except Exception as e:
            logging.error(f"Error checking position closure: {str(e)}")
            return False, f"Error in position check: {str(e)}"

    # ===================== EXECUTION SYSTEM INTEGRATION METHODS =====================
    
    def validate_position_size(self, symbol: str, quantity: float, price: float) -> bool:
        """
        Validate if a position size is within risk limits
        
        Args:
            symbol: Trading symbol
            quantity: Position quantity
            price: Current price
            
        Returns:
            True if position size is valid
        """
        try:
            position_value = abs(quantity) * price
            
            # Get current portfolio value
            portfolio_value = self.portfolio.get_total_value({symbol: price})
            if portfolio_value <= 0:
                return False
            
            max_position_value = portfolio_value * self.max_position_pct
            
            # Check if position exceeds maximum size
            if position_value > max_position_value:
                logging.warning(f"Position size validation failed: {position_value:.2f} > {max_position_value:.2f}")
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"Error validating position size: {str(e)}")
            return False
    
    def update_daily_pnl(self, pnl: float):
        """
        Update daily PnL tracking
        
        Args:
            pnl: Daily PnL amount
        """
        try:
            if not hasattr(self, 'daily_pnl'):
                self.daily_pnl = 0.0
            
            self.daily_pnl += pnl
            
            # Update PnL history
            if not hasattr(self, 'pnl_history'):
                self.pnl_history = []
            
            self.pnl_history.append({
                'date': datetime.now(),
                'pnl': pnl,
                'cumulative_pnl': self.daily_pnl
            })
            
            # Keep only last 252 days (1 year)
            if len(self.pnl_history) > 252:
                self.pnl_history = self.pnl_history[-252:]
            
            logging.info(f"Daily PnL updated: {pnl:.2f}, Cumulative: {self.daily_pnl:.2f}")
            
        except Exception as e:
            logging.error(f"Error updating daily PnL: {str(e)}")
    
    def validate_risk_score(self, risk_score: float) -> bool:
        """
        Validate if a risk score is acceptable
        
        Args:
            risk_score: Risk score (0-1)
            
        Returns:
            True if risk score is acceptable
        """
        try:
            # Risk score thresholds
            max_risk_score = 0.8  # Maximum acceptable risk score
            
            if risk_score < 0 or risk_score > 1:
                logging.warning(f"Invalid risk score: {risk_score}")
                return False
            
            if risk_score > max_risk_score:
                logging.warning(f"Risk score too high: {risk_score} > {max_risk_score}")
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"Error validating risk score: {str(e)}")
            return False
    
    def calculate_position_size(self, symbol: str, balance: float, risk_per_trade: float, price: float) -> float:
        """
        Calculate optimal position size based on risk management
        
        Args:
            symbol: Trading symbol
            balance: Available balance
            risk_per_trade: Risk per trade (fraction of balance)
            price: Current price
            
        Returns:
            Recommended position size in USD
        """
        try:
            # Base position size based on risk per trade
            base_risk_amount = balance * risk_per_trade
            
            # Apply portfolio risk limits
            max_position_value = balance * self.max_position_pct
            
            # Get current risk metrics if available
            current_risk_level = getattr(self, 'current_risk_level', RiskLevel.LOW)
            
            # Risk adjustment multiplier based on current risk level
            risk_multiplier = {
                RiskLevel.LOW: 1.0,
                RiskLevel.MEDIUM: 0.8,
                RiskLevel.HIGH: 0.5,
                RiskLevel.EXTREME: 0.2
            }.get(current_risk_level, 0.5)
            
            # Calculate final position size
            recommended_size = min(
                base_risk_amount * risk_multiplier,
                max_position_value
            )
            
            # Ensure we don't exceed available balance
            recommended_size = min(recommended_size, balance * 0.9)  # Leave 10% buffer
            
            logging.info(f"Position size calculated for {symbol}: ${recommended_size:.2f}")
            return max(0, recommended_size)
            
        except Exception as e:
            logging.error(f"Error calculating position size: {str(e)}")
            return 0.0
    
    def get_kelly_criterion_size(self, win_rate: float, avg_win: float, avg_loss: float, balance: float) -> float:
        """
        Calculate position size using Kelly Criterion
        
        Args:
            win_rate: Win rate (0-1)
            avg_win: Average win amount
            avg_loss: Average loss amount
            balance: Available balance
            
        Returns:
            Recommended position size using Kelly Criterion
        """
        try:
            if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
                return 0.0
            
            # Kelly Criterion: f = (bp - q) / b
            # where b = avg_win/avg_loss, p = win_rate, q = 1 - win_rate
            b = avg_win / avg_loss
            p = win_rate
            q = 1 - win_rate
            
            kelly_fraction = (b * p - q) / b
            
            # Cap Kelly fraction at 25% to avoid over-leveraging
            kelly_fraction = min(kelly_fraction, 0.25)
            kelly_fraction = max(kelly_fraction, 0.0)
            
            # Apply additional risk management
            conservative_kelly = kelly_fraction * 0.5  # Use half-Kelly for safety
            
            position_size = balance * conservative_kelly
            
            logging.info(f"Kelly criterion position size: ${position_size:.2f} (Kelly: {kelly_fraction:.3f})")
            return position_size
            
        except Exception as e:
            logging.error(f"Error calculating Kelly criterion size: {str(e)}")
            return 0.0
    
    def check_trading_conditions(self, current_prices: Dict[str, float]) -> Tuple[bool, str]:
        """
        Check if current conditions allow for trading
        
        Args:
            current_prices: Current market prices
            
        Returns:
            Tuple of (can_trade, reason)
        """
        try:
            # Get current risk metrics
            risk_metrics = self.calculate_risk_metrics(current_prices)
            
            # Check drawdown limit
            if risk_metrics.current_drawdown > self.max_drawdown_limit:
                return False, f"Drawdown limit exceeded: {risk_metrics.current_drawdown:.2%}"
            
            # Check risk level
            if risk_metrics.risk_level == RiskLevel.EXTREME:
                return False, "Portfolio risk level is EXTREME"
            
            # Check daily PnL if available
            if hasattr(self, 'daily_pnl'):
                daily_loss_limit = -0.05  # 5% daily loss limit
                if self.daily_pnl < daily_loss_limit:
                    return False, f"Daily loss limit exceeded: {self.daily_pnl:.2%}"
            
            # Check portfolio value
            portfolio_value = self.portfolio.get_total_value(current_prices)
            if portfolio_value <= 0:
                return False, "Portfolio value is zero or negative"
            
            return True, "Trading conditions are acceptable"
            
        except Exception as e:
            logging.error(f"Error checking trading conditions: {str(e)}")
            return False, f"Error in trading conditions check: {str(e)}"
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive risk summary
        
        Returns:
            Dictionary with risk metrics and status
        """
        try:
            summary = {
                'max_position_pct': self.max_position_pct,
                'max_portfolio_risk': self.max_portfolio_risk,
                'max_drawdown_limit': self.max_drawdown_limit,
                'var_confidence_level': self.var_confidence_level,
                'daily_pnl': getattr(self, 'daily_pnl', 0.0),
                'current_risk_level': getattr(self, 'current_risk_level', RiskLevel.LOW).value,
                'trading_enabled': True
            }
            
            return summary
            
        except Exception as e:
            logging.error(f"Error generating risk summary: {str(e)}")
            return {'error': str(e)}

# Backward compatibility
RiskManager = AdvancedRiskManager 