"""
Tests for the TradingEnvironment class
"""
import pytest
import numpy as np
import gymnasium as gym
from src.agent.trading_env import TradingEnvironment


class TestTradingEnvironment:
    """Test cases for the TradingEnvironment class"""

    def test_initialization(self, sample_price_data, sample_features):
        """Test initialization of the environment"""
        prices = sample_price_data['Close'].values
        features = sample_features
        
        env = TradingEnvironment(
            prices=prices,
            features=features,
            initial_balance=10000,
            transaction_fee_percent=0.001
        )
        
        # Check that the environment has the correct properties
        assert env.initial_balance == 10000
        assert env.transaction_fee_percent == 0.001
        assert np.array_equal(env.prices, prices)
        assert np.array_equal(env.features, features)
        assert env.current_step == 0
        assert env.current_price == prices[0]
        assert env.cash_balance == 10000
        assert env.shares_held == 0
        assert env.total_net_worth == 10000
        assert env.total_shares_bought == 0
        assert env.total_shares_sold == 0
        
        # Check that action and observation spaces are correctly defined
        assert isinstance(env.action_space, gym.spaces.Box)
        assert isinstance(env.observation_space, gym.spaces.Box)
        
        # Check action space bounds (action is percentage of portfolio to invest)
        assert env.action_space.low[0] == -1.0
        assert env.action_space.high[0] == 1.0
        
        # Check observation space (should include features plus portfolio state)
        expected_obs_shape = (features.shape[1] + 5,)
        assert env.observation_space.shape == expected_obs_shape

    def test_reset(self, sample_price_data, sample_features):
        """Test resetting the environment"""
        prices = sample_price_data['Close'].values
        features = sample_features
        
        env = TradingEnvironment(
            prices=prices,
            features=features,
            initial_balance=10000,
            transaction_fee_percent=0.001
        )
        
        # Make some changes to the environment
        env.current_step = 10
        env.cash_balance = 5000
        env.shares_held = 50
        
        # Reset the environment
        obs, info = env.reset()
        
        # Check that the environment has been reset
        assert env.current_step == 0
        assert env.cash_balance == 10000
        assert env.shares_held == 0
        assert env.total_net_worth == 10000
        assert env.total_shares_bought == 0
        assert env.total_shares_sold == 0
        
        # Check that the observation is correct
        assert isinstance(obs, np.ndarray)
        assert obs.shape == env.observation_space.shape
        
        # Check specific observation components
        # First part should be the features at step 0
        assert np.array_equal(obs[:features.shape[1]], features[0])
        
        # Last 5 values should contain portfolio information
        assert obs[-5] == 10000  # Cash balance
        assert obs[-4] == 0      # Shares held
        assert obs[-3] == 10000  # Net worth
        assert obs[-2] == 0      # Previous action
        assert pytest.approx(obs[-1]) == pytest.approx(prices[0])  # Current price

    def test_step_buy(self, sample_price_data, sample_features):
        """Test taking a buy action in the environment"""
        prices = sample_price_data['Close'].values
        features = sample_features
        
        env = TradingEnvironment(
            prices=prices,
            features=features,
            initial_balance=10000,
            transaction_fee_percent=0.001
        )
        
        # Take a buy action (75% of portfolio)
        action = np.array([0.75])
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Calculate expected values
        expected_cash_spent = 10000 * 0.75
        expected_shares_bought = (expected_cash_spent * (1 - 0.001)) / prices[0]
        expected_cash_balance = 10000 - expected_cash_spent
        expected_net_worth = expected_cash_balance + (expected_shares_bought * prices[1])
        
        # Check that the environment has been updated correctly
        assert env.current_step == 1
        assert env.cash_balance == pytest.approx(expected_cash_balance)
        assert env.shares_held == pytest.approx(expected_shares_bought)
        assert env.total_shares_bought == pytest.approx(expected_shares_bought)
        assert env.total_shares_sold == 0
        
        # Check reward (should be based on net worth change)
        expected_reward = (expected_net_worth - 10000) / 10000
        assert reward == pytest.approx(expected_reward)
        
        # Check observation
        assert isinstance(obs, np.ndarray)
        assert obs.shape == env.observation_space.shape
        
        # Check specific observation components
        assert np.array_equal(obs[:features.shape[1]], features[1])
        assert obs[-5] == pytest.approx(expected_cash_balance)
        assert obs[-4] == pytest.approx(expected_shares_bought)
        assert obs[-3] == pytest.approx(expected_net_worth)
        assert obs[-2] == 0.75  # Previous action
        assert obs[-1] == pytest.approx(prices[1])  # Current price

    def test_step_sell(self, sample_price_data, sample_features):
        """Test taking a sell action in the environment"""
        prices = sample_price_data['Close'].values
        features = sample_features
        
        env = TradingEnvironment(
            prices=prices,
            features=features,
            initial_balance=10000,
            transaction_fee_percent=0.001
        )
        
        # First buy some shares
        action = np.array([0.5])
        env.step(action)
        
        initial_cash = env.cash_balance
        initial_shares = env.shares_held
        initial_net_worth = env.total_net_worth
        
        # Then sell 80% of shares
        action = np.array([-0.8])
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Calculate expected values
        shares_to_sell = initial_shares * 0.8
        cash_gained = shares_to_sell * prices[1]
        fee = cash_gained * 0.001
        expected_cash_balance = initial_cash + cash_gained - fee
        expected_shares_held = initial_shares - shares_to_sell
        expected_net_worth = expected_cash_balance + (expected_shares_held * prices[2])
        
        # Check that the environment has been updated correctly
        assert env.current_step == 2
        assert env.cash_balance == pytest.approx(expected_cash_balance, rel=1e-4)
        assert env.shares_held == pytest.approx(expected_shares_held)
        assert env.total_shares_sold == pytest.approx(shares_to_sell)
        
        # Check reward
        expected_reward = (expected_net_worth - initial_net_worth) / initial_net_worth
        assert reward == pytest.approx(expected_reward)
        
        # Check observation
        assert isinstance(obs, np.ndarray)
        assert obs.shape == env.observation_space.shape
        
        # Check specific observation components
        assert np.array_equal(obs[:features.shape[1]], features[2])
        assert obs[-5] == pytest.approx(expected_cash_balance, rel=1e-4)
        assert obs[-4] == pytest.approx(expected_shares_held)
        assert obs[-3] == pytest.approx(expected_net_worth)
        assert obs[-2] == pytest.approx(-0.8)  # Previous action
        assert obs[-1] == pytest.approx(prices[2])  # Current price

    def test_termination(self, sample_price_data, sample_features):
        """Test that the environment terminates correctly"""
        prices = sample_price_data['Close'].values
        features = sample_features
        
        env = TradingEnvironment(
            prices=prices,
            features=features,
            initial_balance=10000,
            transaction_fee_percent=0.001
        )
        
        # Step through the environment until just before the end
        for _ in range(len(prices) - 2):
            action = np.array([0.0])  # No action
            _, _, terminated, _, _ = env.step(action)
            assert not terminated
        
        # Take the final step
        action = np.array([0.0])
        _, _, terminated, truncated, _ = env.step(action)
        
        # Check that the environment has terminated
        assert terminated
        assert env.current_step == len(prices) - 1

    def test_render(self, sample_price_data, sample_features):
        """Test rendering the environment"""
        prices = sample_price_data['Close'].values
        features = sample_features
        
        env = TradingEnvironment(
            prices=prices,
            features=features,
            initial_balance=10000,
            transaction_fee_percent=0.001
        )
        
        # Take some actions to create a trading history
        actions = [0.5, -0.3, 0.2, -0.1, 0.0]
        for action in actions:
            env.step(np.array([action]))
        
        # Test render method (it should return None since it prints to console)
        rendered = env.render()
        assert rendered is None 