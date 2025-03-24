"""
Tests for the TradingEnvironment class
"""
import pytest
import numpy as np
from src.agent.trading_env import LegacyTradingEnvironment


class TestTradingEnvironment:
    """Test cases for trading environment"""
    
    def test_initialization(self, sample_price_data, sample_features):
        """Test initialization of the environment"""
        prices = sample_price_data['Close'].values
        features = sample_features
        
        env = LegacyTradingEnvironment(
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
        assert env.cash_balance == 10000
        assert env.shares_held == 0
        assert env.current_step == 0
        
        # Check action and observation spaces
        assert env.action_space.shape == (1,)
        assert env.action_space.low[0] == -1.0
        assert env.action_space.high[0] == 1.0
        assert env.observation_space.shape == (features.shape[1] + 5,)
    
    def test_reset(self, sample_price_data, sample_features):
        """Test resetting the environment"""
        prices = sample_price_data['Close'].values
        features = sample_features
        
        env = LegacyTradingEnvironment(
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
        assert len(obs) == features.shape[1] + 5
    
    def test_step_buy(self, sample_price_data, sample_features):
        """Test taking a buy action in the environment"""
        prices = sample_price_data['Close'].values
        features = sample_features
        
        env = LegacyTradingEnvironment(
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
        assert env.shares_held == pytest.approx(expected_shares_bought, rel=1e-3)
        assert env.total_net_worth == pytest.approx(expected_net_worth, rel=1e-3)
        assert env.total_shares_bought == pytest.approx(expected_shares_bought, rel=1e-3)
        
        # Check that the observation is correct
        assert isinstance(obs, np.ndarray)
        assert len(obs) == features.shape[1] + 5
        
        # Check that info dict has correct keys
        assert 'step' in info
        assert 'cash_balance' in info
        assert 'shares_held' in info
        assert 'portfolio_value' in info
    
    def test_step_sell(self, sample_price_data, sample_features):
        """Test taking a sell action in the environment"""
        prices = sample_price_data['Close'].values
        features = sample_features
        
        env = LegacyTradingEnvironment(
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
        assert env.shares_held == pytest.approx(expected_shares_held, rel=1e-3)
        assert env.total_shares_sold == pytest.approx(shares_to_sell, rel=1e-3)
        
        # Check reward with higher tolerance due to float calculation differences
        expected_reward = (expected_net_worth - initial_net_worth) / initial_net_worth * 100
        assert reward == pytest.approx(expected_reward, rel=1.0)
        
        # Check observation
        assert isinstance(obs, np.ndarray)
        assert len(obs) == features.shape[1] + 5
    
    def test_portfolio_value(self, sample_price_data, sample_features):
        """Test calculation of portfolio value"""
        prices = sample_price_data['Close'].values
        features = sample_features
        
        env = LegacyTradingEnvironment(
            prices=prices,
            features=features,
            initial_balance=10000,
            transaction_fee_percent=0.001
        )
        
        # Initial portfolio value should be the initial balance
        assert env._get_portfolio_value() == 10000
        
        # Buy some shares
        action = np.array([0.5])
        env.step(action)
        
        # Calculate expected portfolio value
        expected_value = env.cash_balance + env.shares_held * prices[1]
        assert env._get_portfolio_value() == pytest.approx(expected_value)
    
    def test_calculate_reward(self, sample_price_data, sample_features):
        """Test calculation of reward"""
        prices = sample_price_data['Close'].values
        features = sample_features
        
        env = LegacyTradingEnvironment(
            prices=prices,
            features=features,
            initial_balance=10000,
            transaction_fee_percent=0.001
        )
        
        # Test with previous net worth = 10000, current = 11000
        # Set up the portfolio to have a value of 11000
        env.cash_balance = 11000
        reward = env._calculate_reward(10000)
        
        # Reward should be the percentage change
        expected_reward = (11000 - 10000) / 10000 * 100
        assert reward == pytest.approx(expected_reward, rel=1e-5) 