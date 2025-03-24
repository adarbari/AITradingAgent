"""
Base interface for trading environments
"""
from abc import ABC, abstractmethod
from gymnasium import Env


class BaseTradingEnvironment(Env, ABC):
    """
    Abstract base class for trading environments
    
    This defines the interface that all trading environments must implement
    to ensure interchangeability.
    """
    
    @abstractmethod
    def __init__(self, prices, features, initial_balance=10000, transaction_fee_percent=0.001, **kwargs):
        """
        Initialize the trading environment
        
        Args:
            prices (numpy.ndarray): Array of price data
            features (numpy.ndarray): Array of feature data
            initial_balance (float): Starting balance for trading
            transaction_fee_percent (float): Fee percentage for transactions
            **kwargs: Additional keyword arguments specific to the implementation
        """
        super().__init__()
    
    @abstractmethod
    def reset(self, **kwargs):
        """
        Reset the environment to initial state
        
        Args:
            **kwargs: Additional keyword arguments specific to the implementation
            
        Returns:
            tuple: (observation, info)
        """
        pass
    
    @abstractmethod
    def step(self, action):
        """
        Execute one step in the environment
        
        Args:
            action: Action to take in the environment
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        pass
    
    @abstractmethod
    def _get_observation(self):
        """
        Get the current observation
        
        Returns:
            dict: The current observation
        """
        pass
    
    @abstractmethod
    def _get_portfolio_value(self):
        """
        Calculate the current portfolio value
        
        Returns:
            float: Current portfolio value
        """
        pass
    
    @abstractmethod
    def _calculate_reward(self, action=None):
        """
        Calculate the reward for the current step
        
        Args:
            action: The action taken (optional)
            
        Returns:
            float: The calculated reward
        """
        pass
    
    @abstractmethod
    def _get_info(self):
        """
        Get additional information about the current state
        
        Returns:
            dict: Additional information
        """
        pass 