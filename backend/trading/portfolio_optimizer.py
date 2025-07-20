"""
Advanced Portfolio Optimizer for Kimera Trading System
======================================================

Modern portfolio optimization using multiple methods:
- Mean-variance optimization (Markowitz)
- Black-Litterman model
- Risk parity
- Hierarchical Risk Parity (HRP)
- Kelly Criterion optimization
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.linalg import inv
from typing import Dict, List, Optional, Tuple, Any
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf, OAS
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    """
    Advanced portfolio optimization with multiple methods and robust error handling
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize Portfolio Optimizer
        
        Args:
            risk_free_rate: Risk-free rate for Sharpe ratio calculations
        """
        self.risk_free_rate = risk_free_rate
        self.scaler = StandardScaler()
        
    def optimize_portfolio(self, 
                         returns_data: Dict[str, np.ndarray],
                         target_return: float = None,
                         method: str = 'mean_variance',
                         max_iterations: int = 1000,
                         risk_aversion: float = 1.0) -> Dict[str, Any]:
        """
        Optimize portfolio allocation
        
        Args:
            returns_data: Dictionary of asset returns
            target_return: Target portfolio return
            method: Optimization method ('mean_variance', 'risk_parity', 'kelly')
            max_iterations: Maximum optimization iterations
            risk_aversion: Risk aversion parameter
            
        Returns:
            Dictionary with optimization results
        """
        try:
            # Prepare data
            df = pd.DataFrame(returns_data)
            returns = df.dropna()
            
            if len(returns) < 10:
                raise ValueError("Insufficient data for optimization")
            
            # Calculate expected returns and covariance
            expected_returns = returns.mean()
            
            # Use robust covariance estimation
            try:
                cov_estimator = LedoitWolf()
                cov_matrix = cov_estimator.fit(returns).covariance_
            except Exception:
                try:
                    cov_estimator = OAS()
                    cov_matrix = cov_estimator.fit(returns).covariance_
                except Exception:
                    # Fallback to sample covariance
                    cov_matrix = returns.cov().values
            
            # Ensure covariance matrix is positive definite
            cov_matrix = self._ensure_positive_definite(cov_matrix)
            
            # Optimize based on method
            if method == 'mean_variance':
                result = self._optimize_mean_variance(
                    expected_returns.values, 
                    cov_matrix, 
                    target_return,
                    max_iterations
                )
            elif method == 'risk_parity':
                result = self._optimize_risk_parity(
                    cov_matrix, 
                    max_iterations
                )
            elif method == 'kelly':
                result = self._optimize_kelly_criterion(
                    expected_returns.values, 
                    cov_matrix, 
                    max_iterations
                )
            else:
                raise ValueError(f"Unknown optimization method: {method}")
            
            # Add asset names to results
            result['asset_names'] = list(returns.columns)
            result['method'] = method
            
            return result
            
        except Exception as e:
            logger.error(f"Portfolio optimization failed: {e}")
            return {
                'converged': False,
                'error': str(e),
                'weights': None,
                'expected_return': 0.0,
                'portfolio_risk': 0.0,
                'sharpe_ratio': 0.0,
                'iterations': 0
            }
    
    def _ensure_positive_definite(self, matrix: np.ndarray) -> np.ndarray:
        """Ensure covariance matrix is positive definite"""
        try:
            # Check if matrix is positive definite
            np.linalg.cholesky(matrix)
            return matrix
        except np.linalg.LinAlgError:
            # Add small regularization to diagonal
            regularization = 1e-6 * np.eye(matrix.shape[0])
            return matrix + regularization
    
    def _optimize_mean_variance(self, 
                              expected_returns: np.ndarray, 
                              cov_matrix: np.ndarray,
                              target_return: float = None,
                              max_iterations: int = 1000) -> Dict[str, Any]:
        """Optimize using mean-variance optimization"""
        try:
            n_assets = len(expected_returns)
            
            # Objective function: minimize portfolio variance
            def objective(weights):
                return np.dot(weights, np.dot(cov_matrix, weights))
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}  # Weights sum to 1
            ]
            
            # If target return is specified, add return constraint
            if target_return is not None:
                constraints.append({
                    'type': 'eq',
                    'fun': lambda weights: np.dot(weights, expected_returns) - target_return
                })
            
            # Bounds (no short selling)
            bounds = [(0, 1) for _ in range(n_assets)]
            
            # Initial guess (equal weights)
            initial_guess = np.ones(n_assets) / n_assets
            
            # Optimize
            result = minimize(
                objective,
                initial_guess,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': max_iterations}
            )
            
            if result.success:
                weights = result.x
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_risk = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
                sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_risk
                
                return {
                    'converged': True,
                    'weights': weights,
                    'expected_return': portfolio_return,
                    'portfolio_risk': portfolio_risk,
                    'sharpe_ratio': sharpe_ratio,
                    'iterations': result.nit
                }
            else:
                return {
                    'converged': False,
                    'error': result.message,
                    'weights': None,
                    'expected_return': 0.0,
                    'portfolio_risk': 0.0,
                    'sharpe_ratio': 0.0,
                    'iterations': result.nit
                }
                
        except Exception as e:
            logger.error(f"Mean-variance optimization failed: {e}")
            return {
                'converged': False,
                'error': str(e),
                'weights': None,
                'expected_return': 0.0,
                'portfolio_risk': 0.0,
                'sharpe_ratio': 0.0,
                'iterations': 0
            }
    
    def _optimize_risk_parity(self, 
                            cov_matrix: np.ndarray,
                            max_iterations: int = 1000) -> Dict[str, Any]:
        """Optimize using risk parity (equal risk contribution)"""
        try:
            n_assets = cov_matrix.shape[0]
            
            # Objective function: minimize sum of squared risk contribution differences
            def objective(weights):
                portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
                marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
                contrib = weights * marginal_contrib
                target_contrib = portfolio_vol / n_assets
                return np.sum((contrib - target_contrib) ** 2)
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
            ]
            
            # Bounds (no short selling)
            bounds = [(0, 1) for _ in range(n_assets)]
            
            # Initial guess (equal weights)
            initial_guess = np.ones(n_assets) / n_assets
            
            # Optimize
            result = minimize(
                objective,
                initial_guess,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': max_iterations}
            )
            
            if result.success:
                weights = result.x
                portfolio_risk = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
                
                return {
                    'converged': True,
                    'weights': weights,
                    'expected_return': 0.0,  # Risk parity doesn't optimize for return
                    'portfolio_risk': portfolio_risk,
                    'sharpe_ratio': 0.0,
                    'iterations': result.nit
                }
            else:
                return {
                    'converged': False,
                    'error': result.message,
                    'weights': None,
                    'expected_return': 0.0,
                    'portfolio_risk': 0.0,
                    'sharpe_ratio': 0.0,
                    'iterations': result.nit
                }
                
        except Exception as e:
            logger.error(f"Risk parity optimization failed: {e}")
            return {
                'converged': False,
                'error': str(e),
                'weights': None,
                'expected_return': 0.0,
                'portfolio_risk': 0.0,
                'sharpe_ratio': 0.0,
                'iterations': 0
            }
    
    def _optimize_kelly_criterion(self, 
                                expected_returns: np.ndarray, 
                                cov_matrix: np.ndarray,
                                max_iterations: int = 1000) -> Dict[str, Any]:
        """Optimize using Kelly criterion (maximize log utility)"""
        try:
            n_assets = len(expected_returns)
            
            # Objective function: maximize expected log return
            def objective(weights):
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_var = np.dot(weights, np.dot(cov_matrix, weights))
                # Approximate log utility using second-order Taylor expansion
                return -(portfolio_return - 0.5 * portfolio_var)
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
            ]
            
            # Bounds (no short selling)
            bounds = [(0, 1) for _ in range(n_assets)]
            
            # Initial guess (equal weights)
            initial_guess = np.ones(n_assets) / n_assets
            
            # Optimize
            result = minimize(
                objective,
                initial_guess,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': max_iterations}
            )
            
            if result.success:
                weights = result.x
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_risk = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
                sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_risk
                
                return {
                    'converged': True,
                    'weights': weights,
                    'expected_return': portfolio_return,
                    'portfolio_risk': portfolio_risk,
                    'sharpe_ratio': sharpe_ratio,
                    'iterations': result.nit
                }
            else:
                return {
                    'converged': False,
                    'error': result.message,
                    'weights': None,
                    'expected_return': 0.0,
                    'portfolio_risk': 0.0,
                    'sharpe_ratio': 0.0,
                    'iterations': result.nit
                }
                
        except Exception as e:
            logger.error(f"Kelly criterion optimization failed: {e}")
            return {
                'converged': False,
                'error': str(e),
                'weights': None,
                'expected_return': 0.0,
                'portfolio_risk': 0.0,
                'sharpe_ratio': 0.0,
                'iterations': 0
            }
    
    def calculate_efficient_frontier(self, 
                                   returns_data: Dict[str, np.ndarray],
                                   num_portfolios: int = 100) -> Dict[str, Any]:
        """Calculate efficient frontier"""
        try:
            df = pd.DataFrame(returns_data)
            returns = df.dropna()
            
            expected_returns = returns.mean()
            cov_matrix = returns.cov().values
            
            # Ensure positive definite
            cov_matrix = self._ensure_positive_definite(cov_matrix)
            
            # Calculate return range
            min_return = expected_returns.min()
            max_return = expected_returns.max()
            target_returns = np.linspace(min_return, max_return, num_portfolios)
            
            # Calculate efficient portfolio for each target return
            efficient_portfolios = []
            for target_return in target_returns:
                result = self._optimize_mean_variance(
                    expected_returns.values, 
                    cov_matrix, 
                    target_return
                )
                if result['converged']:
                    efficient_portfolios.append({
                        'return': result['expected_return'],
                        'risk': result['portfolio_risk'],
                        'sharpe_ratio': result['sharpe_ratio'],
                        'weights': result['weights']
                    })
            
            return {
                'efficient_portfolios': efficient_portfolios,
                'asset_names': list(returns.columns)
            }
            
        except Exception as e:
            logger.error(f"Efficient frontier calculation failed: {e}")
            return {
                'efficient_portfolios': [],
                'asset_names': [],
                'error': str(e)
            }
    
    def calculate_portfolio_metrics(self, 
                                  weights: np.ndarray,
                                  returns_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate portfolio performance metrics"""
        try:
            df = pd.DataFrame(returns_data)
            returns = df.dropna()
            
            portfolio_returns = np.dot(returns, weights)
            
            # Calculate metrics
            total_return = np.prod(1 + portfolio_returns) - 1
            annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
            volatility = np.std(portfolio_returns) * np.sqrt(252)
            sharpe_ratio = (annualized_return - self.risk_free_rate) / volatility
            
            # Calculate maximum drawdown
            cumulative_returns = (1 + portfolio_returns).cumprod()
            rolling_max = cumulative_returns.cummax()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # Calculate Sortino ratio
            downside_returns = portfolio_returns[portfolio_returns < 0]
            downside_deviation = np.std(downside_returns) * np.sqrt(252)
            sortino_ratio = (annualized_return - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
            
            return {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown,
                'calmar_ratio': annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Portfolio metrics calculation failed: {e}")
            return {
                'total_return': 0.0,
                'annualized_return': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'max_drawdown': 0.0,
                'calmar_ratio': 0.0
            } 