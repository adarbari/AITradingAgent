"""
Tests for the BaseTradingEnvironment abstract class
"""
import pytest
from unittest.mock import MagicMock
import numpy as np
from src.agent.base_trading_env import BaseTradingEnvironment


class TestBaseTradingEnvironment:
    """Test cases for the BaseTradingEnvironment abstract class"""
    
    def test_abstract_methods_implementation(self):
        """Test that abstract methods need to be implemented"""
        # Verify that we cannot instantiate the abstract class directly
        with pytest.raises(TypeError):
            BaseTradingEnvironment(
                prices=np.array([100, 101, 102]),
                features=np.array([[1, 2], [3, 4], [5, 6]]),
                initial_balance=10000,
                transaction_fee_percent=0.001
            )
            
    def test_inheritance(self):
        """Test that the BaseTradingEnvironment can be inherited from and extended"""
        # Create a concrete subclass with all methods implemented
        class CompleteTradingEnv(BaseTradingEnvironment):
            def __init__(self, prices, features, initial_balance=10000, transaction_fee_percent=0.001, **kwargs):
                super().__init__(prices, features, initial_balance, transaction_fee_percent, **kwargs)
                self.action_space = MagicMock()
                self.observation_space = MagicMock()
                self.prices = prices
                self.features = features
                self.initial_balance = initial_balance
                self.transaction_fee_percent = transaction_fee_percent
            
            def reset(self, **kwargs):
                return {"observation": "mock"}, {"info": "mock"}
                
            def step(self, action):
                return {"observation": "mock"}, 0.0, False, False, {"info": "mock"}
                
            def _get_observation(self):
                return {"observation": "mock"}
                
            def _get_portfolio_value(self):
                return 10000.0
                
            def _calculate_reward(self, action=None):
                return 0.0
                
            def _get_info(self):
                return {"info": "mock"}
        
        # Create an instance
        env = CompleteTradingEnv(
            prices=np.array([100, 101, 102]),
            features=np.array([[1, 2], [3, 4], [5, 6]]),
            initial_balance=10000,
            transaction_fee_percent=0.001
        )
        
        # Test that the implementation can be used without errors
        obs_info = env.reset()
        assert isinstance(obs_info, tuple)
        
        step_result = env.step(None)
        assert isinstance(step_result, tuple)
        
        obs = env._get_observation()
        assert isinstance(obs, dict)
        
        value = env._get_portfolio_value()
        assert isinstance(value, float)
        
        reward = env._calculate_reward()
        assert isinstance(reward, float)
        
        info = env._get_info()
        assert isinstance(info, dict) 