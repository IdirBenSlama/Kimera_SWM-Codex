"""
Advanced Portfolio Optimization System for Kimera Trading
Implements modern portfolio theory with state-of-the-art optimization techniques
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    logging.warning("CVXPY not available. Install with: pip install cvxpy")

try:
    from scipy.optimize import minimize, differential_evolution
    from scipy.stats import norm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("SciPy not available")

try:
    from sklearn.covariance import LedoitWolf, OAS, ShrunkCovariance
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class OptimizationObjective(Enum):
    """Portfolio optimization objectives"""
    MEAN_VARIANCE = "mean_variance"
    MEAN_REVERSION = "mean_reversion"
    RISK_PARITY = "risk_parity"
    MAXIMUM_SHARPE = "maximum_sharpe"
    MINIMUM_VARIANCE = "minimum_variance"
    MAXIMUM_DIVERSIFICATION = "maximum_diversification"
    BLACK_LITTERMAN = "black_litterman"


@dataclass
class OptimizationConstraints:
    """Portfolio optimization constraints"""
    min_weight: float = 0.0  # Minimum weight per asset
    max_weight: float = 1.0  # Maximum weight per asset
    max_leverage: float = 1.0  # Maximum total leverage
    max_turnover: float = 1.0  # Maximum portfolio turnover
    sector_limits: Optional[Dict[str, float]] = None  # Sector exposure limits
    asset_groups: Optional[Dict[str, List[str]]] = None  # Asset groupings
    transaction_costs: float = 0.001  # Transaction cost percentage


@dataclass
class OptimizationResult:
    """Portfolio optimization result"""
    weights: np.ndarray
    expected_return: float
    expected_risk: float
    sharpe_ratio: float
    diversification_ratio: float
    max_drawdown_estimate: float
    var_95: float  # Value at Risk 95%
    cvar_95: float  # Conditional Value at Risk 95%
    optimization_status: str
    convergence_info: Dict[str, Any]


class AdvancedPortfolioOptimizer:
    """
    State-of-the-art portfolio optimization system
    Implements multiple optimization objectives and advanced risk models
    """
    
    def __init__(self, 
                 risk_free_rate: float = 0.02,
                 lookback_window: int = 252,
                 rebalance_frequency: int = 22):
        """
        Initialize portfolio optimizer
        
        Args:
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
            lookback_window: Number of periods for historical analysis
            rebalance_frequency: Rebalancing frequency in periods
        """
        self.risk_free_rate = risk_free_rate
        self.lookback_window = lookback_window
        self.rebalance_frequency = rebalance_frequency
        
        # Historical data storage
        self.price_history = {}
        self.return_history = {}
        
        # Risk models
        self.covariance_estimators = {}
        self._initialize_risk_models()
        
        # Optimization cache
        self.optimization_cache = {}
        
        logging.info("Advanced Portfolio Optimizer initialized")
    
    def _initialize_risk_models(self):
        """Initialize covariance estimation models"""
        if SKLEARN_AVAILABLE:
            self.covariance_estimators = {
                'sample': None,  # Sample covariance
                'ledoit_wolf': LedoitWolf(),
                'oas': OAS(),  # Oracle Approximating Shrinkage
                'shrunk': ShrunkCovariance()
            }
    
    def update_price_data(self, prices: Dict[str, float], timestamp: pd.Timestamp):
        """Update price history for portfolio optimization"""
        for asset, price in prices.items():
            if asset not in self.price_history:
                self.price_history[asset] = []
            
            self.price_history[asset].append({
                'timestamp': timestamp,
                'price': price
            })
            
            # Keep only recent data
            if len(self.price_history[asset]) > self.lookback_window:
                self.price_history[asset].pop(0)
        
        # Update returns
        self._update_returns()
    
    def _update_returns(self):
        """Calculate returns from price history"""
        for asset, prices in self.price_history.items():
            if len(prices) >= 2:
                returns = []
                for i in range(1, len(prices)):
                    ret = (prices[i]['price'] - prices[i-1]['price']) / prices[i-1]['price']
                    returns.append(ret)
                self.return_history[asset] = np.array(returns)
    
    def estimate_expected_returns(self, 
                                assets: List[str],
                                method: str = 'historical') -> np.ndarray:
        """
        Estimate expected returns for assets
        
        Args:
            assets: List of asset symbols
            method: Estimation method ('historical', 'capm', 'factor_model')
            
        Returns:
            Expected returns array
        """
        n_assets = len(assets)
        
        # If no assets, return empty array
        if n_assets == 0:
            return np.array([])
        
        returns = np.zeros(n_assets)
        
        for i, asset in enumerate(assets):
            if asset in self.return_history and len(self.return_history[asset]) > 0:
                asset_returns = self.return_history[asset]
                
                # Need at least 5 observations for meaningful statistics
                if len(asset_returns) < 5:
                    returns[i] = 0.001  # Small positive return (0.1% daily)
                    continue
                
                try:
                    if method == 'historical':
                        # Simple historical mean
                        returns[i] = np.mean(asset_returns)
                    elif method == 'exponential':
                        # Exponentially weighted returns
                        weights = np.exp(np.arange(len(asset_returns)) * -0.1)
                        weights = weights / np.sum(weights)
                        returns[i] = np.sum(asset_returns * weights)
                    elif method == 'robust':
                        # Robust estimation (trimmed mean)
                        sorted_returns = np.sort(asset_returns)
                        trim = int(len(sorted_returns) * 0.1)  # Trim 10% from each end
                        if trim > 0:
                            returns[i] = np.mean(sorted_returns[trim:-trim])
                        else:
                            returns[i] = np.mean(sorted_returns)
                    else:
                        returns[i] = np.mean(asset_returns)
                    
                    # Validate return estimate
                    if np.isnan(returns[i]) or np.isinf(returns[i]):
                        returns[i] = 0.001  # Fallback to small positive return
                    
                    # Cap extremely large returns (likely data errors)
                    if abs(returns[i]) > 0.5:  # 50% daily return is unrealistic
                        returns[i] = np.sign(returns[i]) * 0.05  # Cap at 5%
                        
                except Exception as e:
                    logging.warning(f"Error estimating returns for {asset}: {str(e)}")
                    returns[i] = 0.001  # Small positive return as fallback
            else:
                # No data available, use small positive return
                returns[i] = 0.001
        
        return returns
    
    def estimate_covariance_matrix(self, 
                                 assets: List[str],
                                 method: str = 'ledoit_wolf') -> np.ndarray:
        """
        Estimate covariance matrix for assets
        
        Args:
            assets: List of asset symbols
            method: Estimation method ('sample', 'ledoit_wolf', 'oas', 'shrunk')
            
        Returns:
            Covariance matrix
        """
        n_assets = len(assets)
        
        # If no assets, return empty matrix
        if n_assets == 0:
            return np.array([])
        
        # Check if we have any return history
        available_assets = [asset for asset in assets if asset in self.return_history and len(self.return_history[asset]) > 0]
        
        if not available_assets:
            # No data available, return identity matrix scaled by typical market volatility
            return np.eye(n_assets) * 0.01  # 1% daily volatility
        
        # Find minimum length across all assets
        min_length = min(len(self.return_history[asset]) for asset in available_assets)
        
        # Need at least 10 observations for meaningful covariance estimation
        if min_length < 10:
            # Insufficient data, use identity matrix
            return np.eye(n_assets) * 0.01
        
        # Prepare returns matrix
        returns_matrix = []
        for asset in assets:
            if asset in self.return_history and len(self.return_history[asset]) >= min_length:
                returns_matrix.append(self.return_history[asset][-min_length:])
            else:
                # Fill missing data with zeros (conservative approach)
                returns_matrix.append(np.zeros(min_length))
        
        returns_matrix = np.array(returns_matrix).T
        
        # Validate matrix dimensions
        if returns_matrix.shape[0] < 2 or returns_matrix.shape[1] != n_assets:
            return np.eye(n_assets) * 0.01
        
        # Estimate covariance with error handling
        try:
            if method == 'sample' or not SKLEARN_AVAILABLE:
                cov_matrix = np.cov(returns_matrix.T)
            elif method in self.covariance_estimators:
                estimator = self.covariance_estimators[method]
                if estimator is None:
                    cov_matrix = np.cov(returns_matrix.T)
                else:
                    estimator.fit(returns_matrix)
                    cov_matrix = estimator.covariance_
            else:
                cov_matrix = np.cov(returns_matrix.T)
            
            # Validate covariance matrix
            if cov_matrix.shape != (n_assets, n_assets):
                return np.eye(n_assets) * 0.01
            
            # Check for numerical stability
            if np.any(np.isnan(cov_matrix)) or np.any(np.isinf(cov_matrix)):
                return np.eye(n_assets) * 0.01
            
            # Ensure positive definiteness
            eigenvals = np.linalg.eigvals(cov_matrix)
            if np.any(eigenvals <= 0):
                # Add regularization to make positive definite
                cov_matrix += np.eye(n_assets) * 1e-6
            
            return cov_matrix
            
        except Exception as e:
            logging.error(f"Error estimating covariance matrix: {str(e)}")
            return np.eye(n_assets) * 0.01
    
    def optimize_portfolio(self,
                         assets: List[str],
                         current_weights: Optional[np.ndarray] = None,
                         objective: OptimizationObjective = OptimizationObjective.MEAN_VARIANCE,
                         constraints: Optional[OptimizationConstraints] = None) -> OptimizationResult:
        """
        Optimize portfolio weights
        
        Args:
            assets: List of asset symbols
            current_weights: Current portfolio weights
            objective: Optimization objective
            constraints: Portfolio constraints
            
        Returns:
            Optimization result
        """
        if not CVXPY_AVAILABLE:
            return self._fallback_optimization(assets, current_weights, constraints)
        
        # Default constraints
        if constraints is None:
            constraints = OptimizationConstraints()
        
        # Estimate parameters
        expected_returns = self.estimate_expected_returns(assets)
        covariance_matrix = self.estimate_covariance_matrix(assets)
        
        n_assets = len(assets)
        
        # Optimization variables
        weights = cp.Variable(n_assets)
        
        # Portfolio return and risk
        portfolio_return = expected_returns.T @ weights
        portfolio_risk = cp.quad_form(weights, covariance_matrix)
        
        # Objective function
        if objective == OptimizationObjective.MEAN_VARIANCE:
            # Mean-variance optimization with risk aversion parameter
            risk_aversion = 1.0
            objective_func = cp.Maximize(portfolio_return - risk_aversion * portfolio_risk)
        
        elif objective == OptimizationObjective.MAXIMUM_SHARPE:
            # Maximum Sharpe ratio (approximated)
            objective_func = cp.Maximize(portfolio_return - self.risk_free_rate)
            
        elif objective == OptimizationObjective.MINIMUM_VARIANCE:
            # Minimum variance
            objective_func = cp.Minimize(portfolio_risk)
            
        elif objective == OptimizationObjective.RISK_PARITY:
            # Risk parity (equal risk contribution)
            objective_func = self._risk_parity_objective(weights, covariance_matrix)
            
        else:
            # Default to mean-variance
            objective_func = cp.Maximize(portfolio_return - portfolio_risk)
        
        # Constraints
        constraint_list = []
        
        # Weights sum to 1 (fully invested)
        constraint_list.append(cp.sum(weights) == 1)
        
        # Weight bounds
        constraint_list.append(weights >= constraints.min_weight)
        constraint_list.append(weights <= constraints.max_weight)
        
        # Leverage constraint
        if constraints.max_leverage < 1.0:
            constraint_list.append(cp.norm(weights, 1) <= constraints.max_leverage)
        
        # Turnover constraint
        if current_weights is not None and constraints.max_turnover < 2.0:
            turnover = cp.norm(weights - current_weights, 1)
            constraint_list.append(turnover <= constraints.max_turnover)
        
        # Sector constraints
        if constraints.sector_limits and constraints.asset_groups:
            for sector, limit in constraints.sector_limits.items():
                if sector in constraints.asset_groups:
                    sector_assets = constraints.asset_groups[sector]
                    sector_indices = [i for i, asset in enumerate(assets) if asset in sector_assets]
                    if sector_indices:
                        sector_weight = cp.sum([weights[i] for i in sector_indices])
                        constraint_list.append(sector_weight <= limit)
        
        # Solve optimization problem
        problem = cp.Problem(objective_func, constraint_list)
        
        try:
            problem.solve(solver=cp.OSQP, verbose=False)
            
            if problem.status == cp.OPTIMAL:
                optimal_weights = weights.value
                
                # Calculate performance metrics
                result = self._calculate_performance_metrics(
                    optimal_weights, expected_returns, covariance_matrix, assets
                )
                result.optimization_status = "OPTIMAL"
                result.convergence_info = {
                    'solver_status': problem.status,
                    'solve_time': problem.solver_stats.solve_time if hasattr(problem.solver_stats, 'solve_time') else 0,
                    'iterations': problem.solver_stats.num_iters if hasattr(problem.solver_stats, 'num_iters') else 0
                }
                
                return result
            
            else:
                logging.warning(f"Optimization failed with status: {problem.status}")
                return self._fallback_optimization(assets, current_weights, constraints)
        
        except Exception as e:
            logging.error(f"Optimization error: {str(e)}")
            return self._fallback_optimization(assets, current_weights, constraints)
    
    def _risk_parity_objective(self, weights, covariance_matrix):
        """Risk parity objective function"""
        # Approximate risk parity using penalty method
        n = len(weights)
        portfolio_risk = cp.quad_form(weights, covariance_matrix)
        
        # Risk contributions
        risk_contributions = []
        for i in range(n):
            risk_contrib = weights[i] * (covariance_matrix @ weights)[i] / portfolio_risk
            risk_contributions.append(risk_contrib)
        
        # Minimize variance of risk contributions
        risk_contrib_var = cp.sum([(rc - 1/n)**2 for rc in risk_contributions])
        
        return cp.Minimize(risk_contrib_var)
    
    def _calculate_performance_metrics(self,
                                     weights: np.ndarray,
                                     expected_returns: np.ndarray,
                                     covariance_matrix: np.ndarray,
                                     assets: List[str]) -> OptimizationResult:
        """Calculate portfolio performance metrics"""
        
        # Expected return and risk
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
        portfolio_risk = np.sqrt(portfolio_variance)
        
        # Sharpe ratio
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0
        
        # Diversification ratio
        individual_risks = np.sqrt(np.diag(covariance_matrix))
        weighted_avg_risk = np.dot(weights, individual_risks)
        diversification_ratio = weighted_avg_risk / portfolio_risk if portfolio_risk > 0 else 1
        
        # Value at Risk (95%)
        var_95 = -norm.ppf(0.05) * portfolio_risk
        
        # Conditional Value at Risk (95%)
        cvar_95 = var_95 * norm.pdf(norm.ppf(0.05)) / 0.05
        
        # Maximum drawdown estimate (rough approximation)
        max_drawdown_estimate = 2 * portfolio_risk  # Simplified estimate
        
        return OptimizationResult(
            weights=weights,
            expected_return=portfolio_return,
            expected_risk=portfolio_risk,
            sharpe_ratio=sharpe_ratio,
            diversification_ratio=diversification_ratio,
            max_drawdown_estimate=max_drawdown_estimate,
            var_95=var_95,
            cvar_95=cvar_95,
            optimization_status="CALCULATED",
            convergence_info={}
        )
    
    def _fallback_optimization(self,
                             assets: List[str],
                             current_weights: Optional[np.ndarray] = None,
                             constraints: Optional[OptimizationConstraints] = None) -> OptimizationResult:
        """Fallback optimization using simple methods"""
        
        if constraints is None:
            constraints = OptimizationConstraints()
        
        n_assets = len(assets)
        
        # Simple equal weight portfolio
        weights = np.ones(n_assets) / n_assets
        
        # Apply weight constraints
        weights = np.clip(weights, constraints.min_weight, constraints.max_weight)
        weights = weights / np.sum(weights)  # Renormalize
        
        # Estimate returns and covariance for metrics
        expected_returns = self.estimate_expected_returns(assets)
        covariance_matrix = self.estimate_covariance_matrix(assets)
        
        return self._calculate_performance_metrics(weights, expected_returns, covariance_matrix, assets)
    
    def rebalance_portfolio(self,
                          current_positions: Dict[str, float],
                          target_weights: np.ndarray,
                          assets: List[str],
                          total_value: float) -> Dict[str, float]:
        """
        Calculate rebalancing trades
        
        Args:
            current_positions: Current position sizes
            target_weights: Target portfolio weights
            assets: List of assets
            total_value: Total portfolio value
            
        Returns:
            Dictionary of trades to execute
        """
        trades = {}
        
        for i, asset in enumerate(assets):
            current_value = current_positions.get(asset, 0)
            target_value = target_weights[i] * total_value
            trade_value = target_value - current_value
            
            if abs(trade_value) > total_value * 0.001:  # Minimum trade threshold
                trades[asset] = trade_value
        
        return trades
    
    def analyze_portfolio_risk(self,
                             weights: np.ndarray,
                             assets: List[str]) -> Dict[str, Any]:
        """
        Comprehensive portfolio risk analysis
        
        Args:
            weights: Portfolio weights
            assets: List of assets
            
        Returns:
            Risk analysis results
        """
        covariance_matrix = self.estimate_covariance_matrix(assets)
        expected_returns = self.estimate_expected_returns(assets)
        
        # Portfolio metrics
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
        portfolio_risk = np.sqrt(portfolio_variance)
        
        # Risk decomposition
        risk_contributions = {}
        total_risk_squared = portfolio_variance
        
        for i, asset in enumerate(assets):
            marginal_risk = np.dot(covariance_matrix[i], weights)
            risk_contribution = weights[i] * marginal_risk / total_risk_squared
            risk_contributions[asset] = risk_contribution
        
        # Concentration metrics
        herfindahl_index = np.sum(weights**2)
        effective_number_assets = 1 / herfindahl_index
        
        return {
            'portfolio_return': portfolio_return,
            'portfolio_risk': portfolio_risk,
            'sharpe_ratio': (portfolio_return - self.risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0,
            'risk_contributions': risk_contributions,
            'herfindahl_index': herfindahl_index,
            'effective_number_assets': effective_number_assets,
            'var_95': -norm.ppf(0.05) * portfolio_risk,
            'max_weight': np.max(weights),
            'min_weight': np.min(weights)
        }


def create_portfolio_optimizer() -> AdvancedPortfolioOptimizer:
    """Factory function to create portfolio optimizer"""
    return AdvancedPortfolioOptimizer()


# Example usage and testing
if __name__ == "__main__":
    # Test the portfolio optimizer
    optimizer = create_portfolio_optimizer()
    
    # Simulate some price data
    assets = ['BTC', 'ETH', 'ADA', 'DOT', 'LINK']
    
    # Generate random price history
    np.random.seed(42)
    for i in range(100):
        prices = {}
        for j, asset in enumerate(assets):
            base_price = 100 * (j + 1)
            price = base_price * (1 + np.random.normal(0, 0.02))
            prices[asset] = price
        
        optimizer.update_price_data(prices, pd.Timestamp.now() + pd.Timedelta(days=i))
    
    logger.info("Testing portfolio optimization...")
    
    # Test different optimization objectives
    objectives = [
        OptimizationObjective.MEAN_VARIANCE,
        OptimizationObjective.MINIMUM_VARIANCE,
        OptimizationObjective.MAXIMUM_SHARPE
    ]
    
    for objective in objectives:
        result = optimizer.optimize_portfolio(assets, objective=objective)
        logger.info(f"\n{objective.value}:")
        logger.info(f"  Expected Return: {result.expected_return:.4f}")
        logger.info(f"  Expected Risk: {result.expected_risk:.4f}")
        logger.info(f"  Sharpe Ratio: {result.sharpe_ratio:.4f}")
        logger.info(f"  Weights: {[f'{w:.3f}' for w in result.weights]}")
        logger.info(f"  Status: {result.optimization_status}")
    
    # Test risk analysis
    equal_weights = np.ones(len(assets)) / len(assets)
    risk_analysis = optimizer.analyze_portfolio_risk(equal_weights, assets)
    logger.info(f"\nEqual Weight Portfolio Risk Analysis:")
    logger.info(f"  Portfolio Risk: {risk_analysis['portfolio_risk']:.4f}")
    logger.info(f"  Sharpe Ratio: {risk_analysis['sharpe_ratio']:.4f}")
    logger.info(f"  Effective Number of Assets: {risk_analysis['effective_number_assets']:.2f}")
    logger.info(f"  VaR 95%: {risk_analysis['var_95']:.4f}")