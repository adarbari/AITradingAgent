"""
Tests for the SafeTradingEnvironment class
"""
import pytest
import numpy as np
from src.agent import SafeTradingEnvironment


class TestSafeTradingEnvironment:
    """Test cases for the SafeTradingEnvironment class"""

    def test_initialization(self, sample_price_data, sample_features):
        """Test initialization of the safe environment"""
        prices = sample_price_data['Close'].values
        features = sample_features
        
        env = SafeTradingEnvironment(
            prices=prices,
            features=features,
            initial_balance=10000,
            transaction_fee_percent=0.001
        )
        
        # Check that the environment has the correct properties
        assert env.initial_balance == 10000
        assert env.transaction_fee_percent == 0.001
        assert env.max_steps == min(len(prices), len(features)) - 1
        assert len(env.portfolio_values) == 0
    
    def test_calculate_max_shares(self, sample_price_data, sample_features):
        """Test the calculation of maximum shares that can be bought"""
        prices = sample_price_data['Close'].values
        features = sample_features
        
        env = SafeTradingEnvironment(
            prices=prices,
            features=features,
            initial_balance=10000,
            transaction_fee_percent=0.001
        )
        
        # Test with different prices
        max_shares_at_price_100 = env._calculate_max_shares(100)
        expected_shares = int(10000 // (100 * 1.001))
        assert max_shares_at_price_100 == expected_shares
        
        max_shares_at_price_50 = env._calculate_max_shares(50)
        expected_shares = int(10000 // (50 * 1.001))
        assert max_shares_at_price_50 == expected_shares
    
    def test_safe_bounds_checking(self):
        """Test that the environment handles index bounds safely"""
        # Create artificial price and feature data with different lengths
        prices = np.array([100, 101, 102, 103])
        features = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ])  # One less than prices
        
        env = SafeTradingEnvironment(
            prices=prices,
            features=features,
            initial_balance=10000,
            transaction_fee_percent=0.001
        )
        
        # Ensure the initialization worked properly despite different lengths
        assert env.max_steps == 2  # min(4, 3) - 1
        assert len(env.prices) == 3  # Safe truncation to match features
        assert len(env.features) == 3
        
        # Reset to initialize portfolio values
        env.reset()
        
        # Take the first step - should not terminate (current_step becomes 1)
        _, _, terminated, _, _ = env.step(0)  # 0 = Buy action
        assert not terminated
        assert env.current_step == 1
        
        # Take the second step - should not terminate yet (current_step becomes 2)
        _, _, terminated, _, _ = env.step(0)  # 0 = Buy action
        # Environment terminates when step >= max_steps, 
        # and our max_steps is 2 with current_step now at 2
        assert terminated
        assert env.current_step == 2
        
        # Try to step one more time - should still be at the max_steps and terminated
        _, _, terminated, _, _ = env.step(0)
        assert terminated
        assert env.current_step == 2  # Should stay at max_steps and not increment further
    
    def test_safe_get_observation(self, sample_price_data, sample_features):
        """Test that _get_observation handles index bounds safely"""
        prices = sample_price_data['Close'].values[:5]
        features = sample_features[:5]
        
        env = SafeTradingEnvironment(
            prices=prices,
            features=features,
            initial_balance=10000,
            transaction_fee_percent=0.001
        )
        
        # Manually set current_step beyond bounds
        env.current_step = len(features)
        
        # This should not raise an IndexError
        observation = env._get_observation()
        
        # Should contain account_info and features
        assert 'account_info' in observation
        assert 'features' in observation
        
        # Should have used the last valid index
        assert np.array_equal(observation['features'], features[-1])

    def test_buy_action(self, sample_price_data, sample_features):
        """Test taking a buy action in the safe environment"""
        prices = sample_price_data['Close'].values[:5]
        features = sample_features[:5]
        
        env = SafeTradingEnvironment(
            prices=prices,
            features=features,
            initial_balance=10000,
            transaction_fee_percent=0.001
        )
        
        # Reset to initialize portfolio values
        env.reset()
        
        # Take a buy action (action 0)
        observation, reward, terminated, truncated, info = env.step(0)
        
        # Should have bought shares
        assert env.shares_held > 0
        assert env.cash_balance < 10000
        
        # Check that the reward is calculated
        assert isinstance(reward, float)
        
    def test_sell_action(self, sample_price_data, sample_features):
        """Test taking a sell action in the safe environment"""
        prices = sample_price_data['Close'].values[:5]
        features = sample_features[:5]
        
        env = SafeTradingEnvironment(
            prices=prices,
            features=features,
            initial_balance=10000,
            transaction_fee_percent=0.001
        )
        
        # Reset to initialize portfolio values
        env.reset()
        
        # First buy some shares
        env.step(0)
        
        # Record state after buying
        cash_after_buy = env.cash_balance
        shares_after_buy = env.shares_held
        
        # Then sell
        observation, reward, terminated, truncated, info = env.step(1)
        
        # Should have sold shares
        assert env.shares_held == 0
        assert env.cash_balance > cash_after_buy
        
        # Check that the reward is calculated
        assert isinstance(reward, float)
    
    def test_input_validation_none_values(self):
        """Test that the environment correctly validates None inputs"""
        with pytest.raises(ValueError, match="Prices cannot be None"):
            SafeTradingEnvironment(
                prices=None,
                features=np.array([[1, 2, 3]]),
                initial_balance=10000,
                transaction_fee_percent=0.001
            )
        
        with pytest.raises(ValueError, match="Features cannot be None"):
            SafeTradingEnvironment(
                prices=np.array([100, 101, 102]),
                features=None,
                initial_balance=10000,
                transaction_fee_percent=0.001
            )
    
    def test_input_validation_empty_arrays(self):
        """Test that the environment correctly validates empty arrays"""
        with pytest.raises(ValueError, match="Prices array cannot be empty"):
            SafeTradingEnvironment(
                prices=np.array([]),
                features=np.array([[1, 2, 3]]),
                initial_balance=10000,
                transaction_fee_percent=0.001
            )
        
        with pytest.raises(ValueError, match="Features array cannot be empty"):
            SafeTradingEnvironment(
                prices=np.array([100, 101, 102]),
                features=np.array([]),
                initial_balance=10000,
                transaction_fee_percent=0.001
            )
    
    def test_input_validation_types(self):
        """Test that the environment correctly validates input types"""
        with pytest.raises(ValueError, match="Prices must be a list or numpy array"):
            SafeTradingEnvironment(
                prices="invalid",
                features=np.array([[1, 2, 3]]),
                initial_balance=10000,
                transaction_fee_percent=0.001
            )
        
        with pytest.raises(ValueError, match="Features must be a list or numpy array"):
            SafeTradingEnvironment(
                prices=np.array([100, 101, 102]),
                features="invalid",
                initial_balance=10000,
                transaction_fee_percent=0.001
            )
    
    def test_input_validation_negative_prices(self):
        """Test that the environment correctly validates negative prices"""
        with pytest.raises(ValueError, match="Prices must be positive values"):
            SafeTradingEnvironment(
                prices=np.array([100, -101, 102]),
                features=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                initial_balance=10000,
                transaction_fee_percent=0.001
            )
    
    def test_input_validation_invalid_balance_and_fees(self):
        """Test that the environment correctly validates balance and fees"""
        with pytest.raises(ValueError, match="Initial balance must be positive"):
            SafeTradingEnvironment(
                prices=np.array([100, 101, 102]),
                features=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                initial_balance=0,
                transaction_fee_percent=0.001
            )
        
        with pytest.raises(ValueError, match="Transaction fee must be between 0 and 1"):
            SafeTradingEnvironment(
                prices=np.array([100, 101, 102]),
                features=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                initial_balance=10000,
                transaction_fee_percent=1.5
            ) 