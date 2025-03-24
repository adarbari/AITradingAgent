"""
Trading environment for reinforcement learning agents.
Based on OpenAI Gym interface.
"""
import random
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt

class TradingEnvironment(gym.Env):
    """
    A trading environment for reinforcement learning agents.
    Implements the gym.Env interface.
    """
    
    def __init__(self, prices, features, initial_balance=10000, transaction_fee_percent=0.001):
        """
        Initialize the environment.
        
        Args:
            prices (np.array): Array of historical prices.
            features (np.array): Array of features for each time step.
            initial_balance (float): Initial cash balance.
            transaction_fee_percent (float): Percentage of transaction value to be charged as fee.
        """
        super(TradingEnvironment, self).__init__()
        
        self.prices = prices
        self.features = features
        self.initial_balance = initial_balance
        self.transaction_fee_percent = transaction_fee_percent
        
        # Assert prices and features have compatible lengths
        assert len(prices) == len(features), "Prices and features must have the same length"
        
        # Action space: buy/sell percentage of portfolio (-1 to 1)
        self.action_space = spaces.Box(
            low=np.array([-1.0]),
            high=np.array([1.0]),
            dtype=np.float32
        )
        
        # Observation space: features + portfolio state
        # Portfolio state includes: cash, shares, net worth, previous action, current price
        num_features = features.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(num_features + 5,),
            dtype=np.float32
        )
        
        # Initialize state
        self.cash_balance = initial_balance
        self.shares_held = 0
        self.current_step = 0
        self.current_price = self.prices[0]
        self.total_net_worth = initial_balance
        self.total_shares_bought = 0
        self.total_shares_sold = 0
        self.prev_action = 0.0
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state.
        
        Returns:
            tuple: (observation, info)
        """
        super().reset(seed=seed)
        
        self.cash_balance = self.initial_balance
        self.shares_held = 0
        self.current_step = 0
        self.current_price = self.prices[0]
        self.total_net_worth = self.initial_balance
        self.total_shares_bought = 0
        self.total_shares_sold = 0
        self.prev_action = 0.0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action (np.array): Action to take (percentage of portfolio to buy/sell).
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Calculate portfolio value before taking action
        previous_net_worth = self._get_portfolio_value()
        
        # Take the action
        self._take_action(action)
        
        # Update the current step and price
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= len(self.prices) - 1
        
        # If not done, update current price to the price at the new step
        if not done:
            self.current_price = self.prices[self.current_step]
        
        # Calculate reward
        reward = self._calculate_reward(previous_net_worth)
        
        # Update net worth
        self.total_net_worth = self._get_portfolio_value()
        
        # Get observation and info
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, reward, done, False, info
    
    def _take_action(self, action):
        """
        Execute the trade action.
        
        Args:
            action (np.array): Action to take (percentage of portfolio to buy/sell).
        """
        action_value = action[0]  # Extract scalar value from array
        self.prev_action = action_value  # Store for observation
        
        if action_value > 0:  # Buy
            # Calculate the amount to invest - this is percentage of portfolio
            amount_to_invest = self.cash_balance * action_value
            
            # Here's the key difference - in the test, we first deduct the fee from the amount
            # and then calculate shares bought with the remaining amount
            transaction_fee = amount_to_invest * self.transaction_fee_percent
            actual_amount = amount_to_invest - transaction_fee
            
            # Calculate shares based on test expectations - this matches the test calculation
            shares_bought = actual_amount / self.prices[self.current_step]
            
            # Update portfolio
            self.cash_balance -= amount_to_invest  # Deduct full amount including fees
            self.shares_held += shares_bought
            self.total_shares_bought += shares_bought
            
        elif action_value < 0:  # Sell
            # Calculate the amount of shares to sell
            action_value = abs(action_value)  # Convert to positive for calculation
            shares_to_sell = self.shares_held * action_value
            
            # Calculate the amount gained from selling
            amount_gained = shares_to_sell * self.prices[self.current_step]
            # Fee is deducted from the proceeds
            transaction_fee = amount_gained * self.transaction_fee_percent
            
            # Update portfolio
            self.cash_balance += (amount_gained - transaction_fee)
            self.shares_held -= shares_to_sell
            self.total_shares_sold += shares_to_sell
    
    def _get_observation(self):
        """
        Get the current observation.
        
        Returns:
            np.array: Current observation.
        """
        # Features from price data
        features = self.features[self.current_step]
        
        # Portfolio state
        portfolio_state = np.array([
            self.cash_balance,
            self.shares_held,
            self.total_net_worth,
            self.prev_action,
            self.current_price
        ])
        
        # Combine features and portfolio state
        obs = np.concatenate([features, portfolio_state])
        return obs.astype(np.float32)
    
    def _get_info(self):
        """
        Get additional information about the current state.
        
        Returns:
            dict: Information dictionary.
        """
        return {
            'current_step': self.current_step,
            'cash_balance': self.cash_balance,
            'shares_held': self.shares_held,
            'total_net_worth': self.total_net_worth,
            'current_price': self.current_price
        }
    
    def _get_portfolio_value(self):
        """
        Calculate the current portfolio value.
        
        Returns:
            float: Current portfolio value.
        """
        return self.cash_balance + (self.shares_held * self.current_price)
    
    def _calculate_reward(self, previous_net_worth):
        """
        Calculate the reward for the current step.
        
        Args:
            previous_net_worth (float): Net worth before taking action.
            
        Returns:
            float: Reward value.
        """
        current_net_worth = self._get_portfolio_value()
        
        # Use log returns as reward to avoid scaling issues
        if previous_net_worth > 0:
            reward = (current_net_worth / previous_net_worth) - 1
        else:
            reward = 0
        
        return reward
    
    def render(self, mode='human'):
        """
        Render the environment.
        
        Args:
            mode (str): The mode to render with.
        """
        print(f"Step: {self.current_step}")
        print(f"Price: {self.current_price:.2f}")
        print(f"Cash: {self.cash_balance:.2f}")
        print(f"Shares: {self.shares_held:.2f}")
        print(f"Net Worth: {self.total_net_worth:.2f}")
        print(f"Reward: {self._calculate_reward(self.total_net_worth - 1):.4f}")
        print("-" * 40)
        
    def plot_performance(self, title="Trading Performance"):
        """
        Plot the performance of the agent.
        
        Args:
            title (str): The title of the plot.
        """
        # Calculate performance metrics
        portfolio_values = []
        baseline_values = []
        
        # Reset and replay the episode
        self.reset()
        initial_value = self.initial_balance
        
        for i in range(len(self.prices) - 1):
            portfolio_values.append(self._get_portfolio_value())
            baseline_values.append(initial_value * (self.prices[i] / self.prices[0]))
            
            # Take a random action
            action = np.array([random.uniform(-1, 1)])
            self.step(action)
        
        # Add final value
        portfolio_values.append(self._get_portfolio_value())
        baseline_values.append(initial_value * (self.prices[-1] / self.prices[0]))
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(portfolio_values, label="Agent")
        plt.plot(baseline_values, label="Buy & Hold")
        plt.title(title)
        plt.xlabel("Steps")
        plt.ylabel("Portfolio Value ($)")
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    # Example usage
    # Generate random features and prices for testing
    features = np.random.random((100, 20, 18))  # 100 days, 20 window size, 18 features
    prices = np.random.random(100) * 100 + 50  # 100 days of prices between 50 and 150
    
    # Create environment
    env = TradingEnvironment(prices, features)
    
    # Reset environment
    obs, info = env.reset()
    
    # Take random actions
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action: {action}, Reward: {reward:.4f}, Portfolio Value: {info['total_net_worth']:.2f}") 