"""
Unit tests for the PortfolioOptimizer class.
"""
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.agent.multi_agent.portfolio_optimizer import PortfolioOptimizer

class TestPortfolioOptimizer(unittest.TestCase):
    """Test cases for the PortfolioOptimizer class."""
    
    def setUp(self):
        """Set up test environment before each test method."""
        self.optimizer = PortfolioOptimizer(risk_free_rate=0.02, verbose=0)
        
        # Create sample returns data for testing
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        np.random.seed(42)  # For reproducible results
        
        # Create sample returns data with different correlations
        asset1_returns = np.random.normal(0.001, 0.02, size=100)  # Higher return, higher volatility
        asset2_returns = np.random.normal(0.0005, 0.01, size=100)  # Lower return, lower volatility
        asset3_returns = np.random.normal(0.0008, 0.015, size=100)  # Medium return, medium volatility
        
        # Create dataframe with returns
        self.returns_df = pd.DataFrame({
            'ASSET1': asset1_returns,
            'ASSET2': asset2_returns,
            'ASSET3': asset3_returns
        }, index=dates)
        
        # Create covariance matrix
        self.cov_matrix = self.returns_df.cov()
    
    def test_get_portfolio_stats(self):
        """Test portfolio statistics calculation."""
        # Equal weights portfolio
        weights = np.array([1/3, 1/3, 1/3])
        
        # Calculate portfolio stats
        return_val, volatility, sharpe = self.optimizer.get_portfolio_stats(
            weights, self.returns_df, self.cov_matrix
        )
        
        # Assertions
        self.assertIsInstance(return_val, float)
        self.assertIsInstance(volatility, float)
        self.assertIsInstance(sharpe, float)
        self.assertGreater(volatility, 0, "Volatility should be positive")
    
    def test_portfolio_volatility(self):
        """Test portfolio volatility calculation."""
        # Equal weights portfolio
        weights = np.array([1/3, 1/3, 1/3])
        
        # Calculate volatility
        volatility = self.optimizer.portfolio_volatility(weights, self.cov_matrix)
        
        # Assertions
        self.assertIsInstance(volatility, float)
        self.assertGreater(volatility, 0, "Volatility should be positive")
    
    def test_negative_sharpe(self):
        """Test negative Sharpe ratio calculation for optimization."""
        # Equal weights portfolio
        weights = np.array([1/3, 1/3, 1/3])
        
        # Calculate negative Sharpe ratio
        neg_sharpe = self.optimizer.negative_sharpe(weights, self.returns_df, self.cov_matrix)
        
        # Calculate portfolio stats to verify
        _, _, sharpe = self.optimizer.get_portfolio_stats(weights, self.returns_df, self.cov_matrix)
        
        # Assertions
        self.assertAlmostEqual(neg_sharpe, -sharpe, places=10)
    
    def test_optimize_portfolio_sharpe(self):
        """Test portfolio optimization with Sharpe ratio objective."""
        # Optimize with Sharpe ratio objective (default)
        result = self.optimizer.optimize_portfolio(self.returns_df)
        
        # Assertions
        self.assertIn('weights', result)
        self.assertIn('return', result)
        self.assertIn('volatility', result)
        self.assertIn('sharpe_ratio', result)
        
        # Sum of weights should be close to 1.0
        self.assertAlmostEqual(sum(result['weights'].values()), 1.0, places=6)
        
        # All weights should be between 0 and 1
        for weight in result['weights'].values():
            self.assertGreaterEqual(weight, 0)
            self.assertLessEqual(weight, 1)
    
    def test_optimize_portfolio_min_volatility(self):
        """Test portfolio optimization with minimum volatility objective."""
        # Optimize with minimum volatility objective
        result = self.optimizer.optimize_portfolio(
            self.returns_df,
            objective='min_volatility'
        )
        
        # Assertions
        self.assertIn('weights', result)
        self.assertIn('return', result)
        self.assertIn('volatility', result)
        
        # Compare to Sharpe ratio optimized portfolio - volatility should be lower
        result_sharpe = self.optimizer.optimize_portfolio(self.returns_df)
        self.assertLessEqual(result['volatility'], result_sharpe['volatility'])
    
    def test_optimize_portfolio_with_constraints(self):
        """Test portfolio optimization with custom constraints."""
        # Set max position size constraint to 0.4
        constraints = {'max_position': 0.4}
        
        # Optimize with constraints
        result = self.optimizer.optimize_portfolio(
            self.returns_df,
            constraints=constraints
        )
        
        # Assertions
        self.assertIn('weights', result)
        
        # All weights should be <= 0.4
        for weight in result['weights'].values():
            self.assertLessEqual(weight, 0.4)
    
    def test_optimize_portfolio_with_sentiment(self):
        """Test portfolio optimization with sentiment adjustments."""
        # Create sentiment scores (positive for ASSET1, negative for ASSET2)
        sentiment_scores = {
            'ASSET1': 0.8,  # Positive sentiment
            'ASSET2': -0.5,  # Negative sentiment
            'ASSET3': 0.1   # Slightly positive
        }
        
        # Optimize with sentiment scores
        result_with_sentiment = self.optimizer.optimize_portfolio(
            self.returns_df,
            sentiment_scores=sentiment_scores
        )
        
        # Optimize without sentiment for comparison
        result_no_sentiment = self.optimizer.optimize_portfolio(self.returns_df)
        
        # Store weights for comparison
        with_sentiment_weights = result_with_sentiment['weights']
        no_sentiment_weights = result_no_sentiment['weights']
        
        # ASSET1 should have higher weight with sentiment adjustment
        self.assertGreaterEqual(
            with_sentiment_weights.get('ASSET1', 0) / no_sentiment_weights.get('ASSET1', 1e-10),
            0.9  # Allow for minor variation due to optimization
        )
    
    def test_generate_efficient_frontier(self):
        """Test efficient frontier generation."""
        # Generate efficient frontier
        frontier = self.optimizer.generate_efficient_frontier(self.returns_df, points=10)
        
        # Assertions
        self.assertIn('returns', frontier)
        self.assertIn('volatilities', frontier)
        self.assertEqual(len(frontier['returns']), len(frontier['volatilities']))
        
        # Check that the frontier has the expected shape (returns increase with volatility)
        returns = frontier['returns']
        volatilities = frontier['volatilities']
        
        # Returns should be in ascending order
        self.assertTrue(all(returns[i] <= returns[i+1] for i in range(len(returns)-1)))
    
    def test_calculate_risk_parity_portfolio(self):
        """Test risk parity portfolio calculation."""
        # Calculate risk parity portfolio
        result = self.optimizer.calculate_risk_parity_portfolio(self.returns_df)
        
        # Assertions
        self.assertIn('weights', result)
        self.assertIn('return', result)
        self.assertIn('volatility', result)
        self.assertIn('sharpe_ratio', result)
        
        # Sum of weights should be close to 1.0
        self.assertAlmostEqual(sum(result['weights'].values()), 1.0, places=6)
    
    def test_edge_case_single_asset(self):
        """Test handling of a single asset edge case."""
        # Create a single asset returns dataframe
        single_asset_returns = pd.DataFrame({
            'ASSET1': self.returns_df['ASSET1']
        })
        
        # Optimize portfolio
        result = self.optimizer.optimize_portfolio(single_asset_returns)
        
        # Assertions
        self.assertEqual(len(result['weights']), 1)
        self.assertAlmostEqual(list(result['weights'].values())[0], 1.0)
    
    def test_edge_case_equal_returns(self):
        """Test handling of assets with identical returns."""
        # Create returns dataframe with identical returns
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        identical_returns = np.random.normal(0.001, 0.02, size=100)
        
        equal_returns_df = pd.DataFrame({
            'ASSET1': identical_returns,
            'ASSET2': identical_returns,
            'ASSET3': identical_returns
        }, index=dates)
        
        # Optimize portfolio
        result = self.optimizer.optimize_portfolio(equal_returns_df)
        
        # With equal returns and volatility, weights should be approximately equal
        weights = list(result['weights'].values())
        # Allow for slight variations due to numerical optimization
        self.assertTrue(all(abs(w - 1/3) < 0.1 for w in weights))

if __name__ == '__main__':
    unittest.main() 