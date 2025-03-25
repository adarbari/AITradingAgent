"""
Portfolio Optimizer for the multi-agent trading system.
Implements Modern Portfolio Theory and related optimization strategies.
"""
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import scipy.optimize as sco

class PortfolioOptimizer:
    """
    Provides portfolio optimization capabilities using Modern Portfolio Theory and other strategies.
    
    Features:
    1. Efficient frontier optimization (Markowitz model)
    2. Maximum Sharpe ratio portfolio
    3. Minimum volatility portfolio
    4. Risk parity portfolio
    5. Integration of sentiment data for optimization adjustments
    6. Custom constraints for position sizing and sector exposure
    """
    
    def __init__(self, risk_free_rate: float = 0.02, verbose: int = 0):
        """
        Initialize the portfolio optimizer.
        
        Args:
            risk_free_rate (float): Annual risk-free rate (default: 2%)
            verbose (int): Verbosity level (0: silent, 1: normal, 2: detailed)
        """
        self.risk_free_rate = risk_free_rate
        self.verbose = verbose
        self.daily_risk_free_rate = (1 + risk_free_rate) ** (1/252) - 1

    def get_portfolio_stats(self, weights: np.ndarray, returns: pd.DataFrame, cov_matrix: pd.DataFrame) -> Tuple[float, float, float]:
        """
        Calculate portfolio expected return, volatility and Sharpe ratio.
        
        Args:
            weights (np.ndarray): Portfolio weights
            returns (pd.DataFrame): Historical returns data
            cov_matrix (pd.DataFrame): Covariance matrix of returns
            
        Returns:
            Tuple[float, float, float]: (Expected return, volatility, Sharpe ratio)
        """
        # Expected portfolio return (annualized)
        port_return = np.sum(returns.mean() * weights) * 252
        
        # Expected portfolio volatility (annualized)
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
        
        # Sharpe Ratio
        sharpe_ratio = (port_return - self.risk_free_rate) / port_volatility
        
        return port_return, port_volatility, sharpe_ratio

    def negative_sharpe(self, weights: np.ndarray, returns: pd.DataFrame, cov_matrix: pd.DataFrame) -> float:
        """
        Negative Sharpe ratio for minimization.
        
        Args:
            weights (np.ndarray): Portfolio weights
            returns (pd.DataFrame): Historical returns data
            cov_matrix (pd.DataFrame): Covariance matrix of returns
            
        Returns:
            float: Negative Sharpe ratio
        """
        return -self.get_portfolio_stats(weights, returns, cov_matrix)[2]

    def portfolio_volatility(self, weights: np.ndarray, cov_matrix: pd.DataFrame) -> float:
        """
        Calculate portfolio volatility (standard deviation).
        
        Args:
            weights (np.ndarray): Portfolio weights
            cov_matrix (pd.DataFrame): Covariance matrix of returns
            
        Returns:
            float: Portfolio volatility
        """
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))

    def optimize_portfolio(self, returns: pd.DataFrame, constraints: Dict[str, Any] = None,
                           objective: str = 'sharpe', sentiment_scores: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Optimize portfolio based on specified objective.
        
        Args:
            returns (pd.DataFrame): Historical returns data
            constraints (Dict[str, Any], optional): Optimization constraints
            objective (str): Optimization objective - 'sharpe', 'min_volatility', or 'return'
            sentiment_scores (Dict[str, float], optional): Sentiment scores by symbol
            
        Returns:
            Dict[str, Any]: Optimized portfolio weights and statistics
        """
        # Calculate covariance matrix
        cov_matrix = returns.cov()
        
        # Define constraints
        if constraints is None:
            constraints = {}
        
        # Default constraints
        n_assets = len(returns.columns)
        default_bounds = (0, 1)  # Default: Long only, max 100% in any one asset
        max_position = constraints.get('max_position', 1.0)
        
        # Apply sentiment adjustments to expected returns if available
        adjusted_returns = returns.copy()
        if sentiment_scores:
            # Scale sentiment effect: how much it influences returns prediction
            sentiment_effect_scale = constraints.get('sentiment_effect_scale', 0.002)
            
            for symbol, score in sentiment_scores.items():
                if symbol in adjusted_returns.columns:
                    # Adjust expected return based on sentiment
                    # Positive sentiment increases expected return, negative reduces it
                    adjustment = score * sentiment_effect_scale
                    adjusted_returns[symbol] += adjustment
        
        # Initial weights: equal allocation
        init_weights = np.array([1/n_assets] * n_assets)
        
        # Define bounds
        bounds = [(0, max_position) for _ in range(n_assets)]
        
        # Sum of weights = 1 constraint
        constraints_opt = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        # Add sector constraints if provided
        sector_constraints = constraints.get('sector_constraints', {})
        if sector_constraints:
            for sector, (sector_indices, max_exposure) in sector_constraints.items():
                constraints_opt.append({
                    'type': 'ineq', 
                    'fun': lambda x, idx=sector_indices: max_exposure - np.sum(x[idx])
                })
        
        # Optimize based on objective
        if objective == 'sharpe':
            # Maximize Sharpe ratio
            result = sco.minimize(
                self.negative_sharpe, 
                init_weights, 
                args=(adjusted_returns, cov_matrix),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_opt
            )
        elif objective == 'min_volatility':
            # Minimize volatility
            result = sco.minimize(
                self.portfolio_volatility, 
                init_weights, 
                args=(cov_matrix),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_opt
            )
        elif objective == 'return':
            # Maximize return subject to target volatility
            target_volatility = constraints.get('target_volatility', 0.15)
            # This requires a more complex optimization setup
            # TODO: Implement return maximization with volatility constraint
            result = sco.minimize(
                self.negative_sharpe,  # Using Sharpe as a proxy for now 
                init_weights, 
                args=(adjusted_returns, cov_matrix),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_opt
            )
        else:
            raise ValueError(f"Unsupported objective: {objective}")
            
        # Check if optimization succeeded
        if not result.success:
            if self.verbose > 0:
                print(f"Optimization failed: {result.message}")
            # Fall back to equal weights
            weights = init_weights
        else:
            weights = result.x
            
        # Calculate portfolio statistics
        portfolio_return, portfolio_volatility, sharpe_ratio = self.get_portfolio_stats(
            weights, returns, cov_matrix
        )
        
        # Create result dictionary with symbol-to-weight mapping
        weights_dict = {symbol: weight for symbol, weight in zip(returns.columns, weights)}
        
        # Sort by weight (descending)
        sorted_weights = {k: v for k, v in sorted(weights_dict.items(), key=lambda item: item[1], reverse=True)}
        
        return {
            'weights': sorted_weights,
            'return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'objective': objective,
            'risk_free_rate': self.risk_free_rate
        }
        
    def generate_efficient_frontier(self, returns: pd.DataFrame, points: int = 20) -> Dict[str, np.ndarray]:
        """
        Generate the efficient frontier.
        
        Args:
            returns (pd.DataFrame): Historical returns data
            points (int): Number of points to generate on the frontier
            
        Returns:
            Dict[str, np.ndarray]: Volatility and returns for the efficient frontier
        """
        # Calculate covariance matrix
        cov_matrix = returns.cov()
        
        # Set bounds and constraints
        n_assets = len(returns.columns)
        bounds = [(0, 1) for _ in range(n_assets)]
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        # Get min volatility portfolio
        init_weights = np.array([1/n_assets] * n_assets)
        min_vol_result = sco.minimize(
            self.portfolio_volatility, 
            init_weights, 
            args=(cov_matrix),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        min_vol_weights = min_vol_result.x
        min_vol_return, min_vol_volatility, _ = self.get_portfolio_stats(min_vol_weights, returns, cov_matrix)
        
        # Get max return portfolio (100% in the asset with highest return)
        max_return_idx = returns.mean().idxmax()
        max_return = returns.mean()[max_return_idx] * 252
        
        # Generate efficient frontier points
        target_returns = np.linspace(min_vol_return, max_return, points)
        volatilities = []
        
        # For each target return, find the portfolio with minimum volatility
        for target in target_returns:
            # Add return constraint
            return_constraint = {'type': 'eq', 'fun': lambda x: sum(x * returns.mean() * 252) - target}
            constraints_with_return = constraints + [return_constraint]
            
            # Find minimum volatility for this target return
            result = sco.minimize(
                self.portfolio_volatility, 
                init_weights, 
                args=(cov_matrix),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_with_return
            )
            
            if result.success:
                volatilities.append(self.portfolio_volatility(result.x, cov_matrix))
            else:
                # If optimization fails, use a fallback approach
                volatilities.append(np.nan)
        
        # Convert to numpy arrays
        volatilities = np.array(volatilities)
        
        # Remove any NaN values
        valid_indices = ~np.isnan(volatilities)
        clean_returns = target_returns[valid_indices]
        clean_volatilities = volatilities[valid_indices]
        
        return {
            'returns': clean_returns,
            'volatilities': clean_volatilities
        }
        
    def calculate_risk_parity_portfolio(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate risk parity portfolio - each asset contributes equally to portfolio risk.
        
        Args:
            returns (pd.DataFrame): Historical returns data
            
        Returns:
            Dict[str, Any]: Risk parity portfolio weights and statistics
        """
        # Calculate covariance matrix
        cov_matrix = returns.cov()
        n_assets = len(returns.columns)
        
        def risk_contribution(weights, cov_matrix):
            """Calculate risk contribution of each asset"""
            port_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            marginal_contribution = np.dot(cov_matrix, weights)
            risk_contribution = np.multiply(marginal_contribution, weights) / port_variance
            return risk_contribution
        
        def risk_parity_objective(weights, cov_matrix):
            """Objective function for risk parity - minimize variance of risk contributions"""
            risk_contrib = risk_contribution(weights, cov_matrix)
            target_contrib = 1/n_assets
            # Return sum of squared deviations from target
            return np.sum((risk_contrib - target_contrib)**2)
        
        # Constraints and bounds
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = [(0.01, 1) for _ in range(n_assets)]  # Min 1% in each asset
        
        # Initial weights: equal allocation
        init_weights = np.array([1/n_assets] * n_assets)
        
        # Risk parity optimization
        result = sco.minimize(
            risk_parity_objective, 
            init_weights, 
            args=(cov_matrix),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        # Check if optimization succeeded
        if not result.success:
            if self.verbose > 0:
                print(f"Risk parity optimization failed: {result.message}")
            # Fall back to equal weights
            weights = init_weights
        else:
            weights = result.x
            
        # Calculate portfolio statistics
        portfolio_return, portfolio_volatility, sharpe_ratio = self.get_portfolio_stats(
            weights, returns, cov_matrix
        )
        
        # Create result dictionary with symbol-to-weight mapping
        weights_dict = {symbol: weight for symbol, weight in zip(returns.columns, weights)}
        
        # Sort by weight (descending)
        sorted_weights = {k: v for k, v in sorted(weights_dict.items(), key=lambda item: item[1], reverse=True)}
        
        return {
            'weights': sorted_weights,
            'return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'objective': 'risk_parity',
            'risk_free_rate': self.risk_free_rate
        } 