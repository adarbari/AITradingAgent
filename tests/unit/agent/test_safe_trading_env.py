"""
Tests for the TradingEnvironment class (safer implementation)
"""
import pytest
import numpy as np
import gymnasium as gym
from src.agent.trading_env import TradingEnvironment, LegacyTradingEnvironment


class TestTradingEnvironment:
    """Test cases for the TradingEnvironment class (safer implementation)"""

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
        
        # Use approx instead of array_equal due to float32 vs float64 conversion
        assert len(env.prices) == len(prices)
        for i in range(len(prices)):
            assert env.prices[i] == pytest.approx(prices[i], rel=1e-5)
        
        assert np.array_equal(env.features, features)
        assert env.current_step == 0
        assert env.current_price == pytest.approx(prices[0], rel=1e-5)
        assert env.cash_balance == 10000
        assert env.shares_held == 0
        assert env._total_net_worth == 10000
        assert env._total_shares_bought == 0
        assert env._total_shares_sold == 0
        
        # Check that action and observation spaces are correctly defined
        assert isinstance(env.action_space, gym.spaces.Box)
        assert isinstance(env.observation_space, gym.spaces.Dict)
        
        # Check action space bounds (action is percentage of portfolio to invest)
        assert env.action_space.low[0] == -1.0
        assert env.action_space.high[0] == 1.0
        
        # Check observation space (should include features plus portfolio state)
        assert 'features' in env.observation_space.spaces
        assert 'portfolio' in env.observation_space.spaces
        
        expected_feature_shape = (features.shape[1],)
        expected_portfolio_shape = (5,)
        
        assert env.observation_space.spaces['features'].shape == expected_feature_shape
        assert env.observation_space.spaces['portfolio'].shape == expected_portfolio_shape

    def test_calculate_max_shares(self, sample_price_data, sample_features):
        """Test calculation of maximum purchasable shares"""
        prices = sample_price_data['Close'].values
        features = sample_features
        
        env = TradingEnvironment(
            prices=prices,
            features=features,
            initial_balance=10000,
            transaction_fee_percent=0.001
        )
        
        # Test with a typical case
        max_shares = env._calculate_max_shares(9000, prices[0], 0.001)
        expected_shares = 9000 * (1 - 0.001) / prices[0]
        assert max_shares == pytest.approx(expected_shares)
        
        # Test with zero cash
        max_shares = env._calculate_max_shares(0, prices[0], 0.001)
        assert max_shares == 0
        
        # Test with zero price (should be properly handled)
        max_shares = env._calculate_max_shares(1000, 0, 0.001)
        assert max_shares == 0
    
    def test_input_validation(self, sample_price_data, sample_features):
        """Test input validation during initialization"""
        prices = sample_price_data['Close'].values
        features = sample_features
        
        # Test with valid inputs (should not raise)
        TradingEnvironment(
            prices=prices,
            features=features,
            initial_balance=10000,
            transaction_fee_percent=0.001
        )
        
        # Test with incompatible prices and features lengths
        with pytest.raises(ValueError, match="must have the same length"):
            TradingEnvironment(
                prices=prices[:-1],  # One shorter
                features=features,
                initial_balance=10000,
                transaction_fee_percent=0.001
            )
        
        # Test with negative initial balance
        with pytest.raises(ValueError, match="must be positive"):
            TradingEnvironment(
                prices=prices,
                features=features,
                initial_balance=-1000,
                transaction_fee_percent=0.001
            )
        
        # Test with invalid transaction fee
        with pytest.raises(ValueError, match="between 0 and 1"):
            TradingEnvironment(
                prices=prices,
                features=features,
                initial_balance=10000,
                transaction_fee_percent=1.5
            )
        
        # Test with negative prices
        with pytest.raises(ValueError, match="must be non-negative"):
            TradingEnvironment(
                prices=np.array([-1.0] + list(prices[1:])),
                features=features,
                initial_balance=10000,
                transaction_fee_percent=0.001
            )

    def test_safe_bounds_checking(self, sample_price_data, sample_features):
        """Test bounds checking for action and state"""
        prices = sample_price_data['Close'].values
        features = sample_features
        
        env = TradingEnvironment(
            prices=prices,
            features=features,
            initial_balance=10000,
            transaction_fee_percent=0.001
        )
        
        # Test with an action outside bounds
        action = np.array([1.5])  # Greater than max allowed (1.0)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Action should be clipped to 1.0
        assert info['actual_action'] == pytest.approx(1.0)
        
        # Test with extreme negative action
        action = np.array([-2.0])  # Less than min allowed (-1.0)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Action should be clipped to -1.0
        assert info['actual_action'] == pytest.approx(-1.0)

    def test_buy_action(self, sample_price_data, sample_features):
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
        assert env.shares_held == pytest.approx(expected_shares_bought, rel=1e-3)
        assert env._total_net_worth == pytest.approx(expected_net_worth, rel=1e-3)
        assert env._total_shares_bought == pytest.approx(expected_shares_bought, rel=1e-3)
        assert env._total_shares_sold == 0
        
        # Check reward (should be based on net worth change)
        expected_reward = (expected_net_worth - 10000) / 10000
        assert reward == pytest.approx(expected_reward, rel=1e-2)
        
        # Check observation
        assert isinstance(obs, dict)
        assert 'features' in obs
        assert 'portfolio' in obs
        assert np.array_equal(obs['features'], features[1])
        
        portfolio = obs['portfolio']
        assert portfolio[0] == pytest.approx(expected_cash_balance)
        assert portfolio[1] == pytest.approx(expected_shares_bought)
        assert portfolio[2] == pytest.approx(expected_net_worth)
        assert portfolio[3] == pytest.approx(0.75)  # Previous action
        assert portfolio[4] == pytest.approx(prices[1])  # Current price
        
        # Check info dict
        assert 'step' in info
        assert 'cash_balance' in info
        assert 'shares_held' in info
        assert 'portfolio_value' in info
        assert 'actual_action' in info
        
        assert info['step'] == 1
        assert info['cash_balance'] == pytest.approx(expected_cash_balance)
        assert info['shares_held'] == pytest.approx(expected_shares_bought)
        assert info['portfolio_value'] == pytest.approx(expected_net_worth)
        assert info['actual_action'] == pytest.approx(0.75)

    def test_sell_action(self, sample_price_data, sample_features):
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
        initial_net_worth = env._total_net_worth
        
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
        assert env._total_shares_sold == pytest.approx(shares_to_sell)
        
        # Check reward
        expected_reward = (expected_net_worth - initial_net_worth) / initial_net_worth
        assert reward == pytest.approx(expected_reward, rel=1e-3)
        
        # Check observation
        assert isinstance(obs, dict)
        assert 'features' in obs
        assert 'portfolio' in obs
        
        # Check info dict
        assert 'step' in info
        assert 'cash_balance' in info
        assert 'shares_held' in info
        assert 'portfolio_value' in info
        assert 'actual_action' in info

    def test_edge_case_selling_with_no_shares(self, sample_price_data, sample_features):
        """Test attempting to sell when no shares are held"""
        prices = sample_price_data['Close'].values
        features = sample_features
        
        env = TradingEnvironment(
            prices=prices,
            features=features,
            initial_balance=10000,
            transaction_fee_percent=0.001
        )
        
        # Try to sell when no shares are held
        action = np.array([-0.5])
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Should have no effect on cash or shares
        assert env.cash_balance == 10000
        assert env.shares_held == 0
        assert env._total_shares_sold == 0
        
        # The actual action should be 0
        assert info['actual_action'] == 0

    def test_buying_with_insufficient_cash(self, sample_price_data, sample_features):
        """Test buying with insufficient cash balance"""
        prices = sample_price_data['Close'].values
        features = sample_features
        
        env = TradingEnvironment(
            prices=prices,
            features=features,
            initial_balance=100,  # Small initial balance
            transaction_fee_percent=0.001
        )
        
        # Step the environment to a point where the price is high
        high_price_idx = np.argmax(prices)
        for _ in range(high_price_idx):
            env.step(np.array([0.0]))
        
        # Try to buy with a small cash balance
        initial_cash = env.cash_balance
        action = np.array([1.0])  # Try to use all cash
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Should buy as many shares as possible
        expected_shares = env._calculate_max_shares(
            initial_cash, env.prices[env.current_step - 1], env.transaction_fee_percent
        )
        
        # Cash should be reduced appropriately
        if expected_shares > 0:
            expected_cash = initial_cash - (expected_shares * env.prices[env.current_step - 1] * (1 + env.transaction_fee_percent))
            assert env.cash_balance == pytest.approx(expected_cash, abs=1e-4)
            assert env.shares_held == pytest.approx(expected_shares)
        else:
            # If can't buy any shares, cash should remain the same
            assert env.cash_balance == initial_cash
            assert env.shares_held == 0

    def test_zero_action(self, sample_price_data, sample_features):
        """Test taking no action (hold)"""
        prices = sample_price_data['Close'].values
        features = sample_features
        
        env = TradingEnvironment(
            prices=prices,
            features=features,
            initial_balance=10000,
            transaction_fee_percent=0.001
        )
        
        # Take a zero action (hold)
        action = np.array([0.0])
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Cash and shares should remain the same
        assert env.cash_balance == 10000
        assert env.shares_held == 0
        
        # Net worth should only change due to price changes
        assert env._total_net_worth == 10000
        
        # Check observation
        assert isinstance(obs, dict)
        assert 'features' in obs
        assert 'portfolio' in obs
        
        # The zero action should be recorded in info
        assert info['actual_action'] == 0.0

    def test_terminal_state(self, sample_price_data, sample_features):
        """Test that the environment terminates at the end of the data"""
        prices = sample_price_data['Close'].values
        features = sample_features
        
        env = TradingEnvironment(
            prices=prices,
            features=features,
            initial_balance=10000,
            transaction_fee_percent=0.001
        )
        
        # Step through all but the last step
        for _ in range(len(prices) - 2):
            obs, reward, terminated, truncated, info = env.step(np.array([0.0]))
            assert not terminated
            assert not truncated
        
        # Take the last step
        obs, reward, terminated, truncated, info = env.step(np.array([0.0]))
        
        # Should be terminated
        assert terminated
        assert not truncated
        assert env.current_step == len(prices) - 1 