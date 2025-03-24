#!/usr/bin/env python3
"""
Train PPO models using synthetic or real data up to a specified date and backtest
them for a future period, comparing performance against a market index.
"""
import os
import glob
import argparse
import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
from src.agent.trading_env import TradingEnvironment
from src.data import BaseDataFetcher, SyntheticDataFetcher, YahooDataFetcher, DataFetcherFactory
from src.backtest import Backtester
from src.models import ModelTrainer
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import warnings
from pathlib import Path
import traceback

# Create results and models directories if they don't exist
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('data/cache', exist_ok=True)

# Create a subclass of TradingEnvironment that adds safety checks
class SafeTradingEnvironment(TradingEnvironment):
    """A TradingEnvironment with added safety checks to prevent index errors"""
    
    def __init__(self, prices, features, initial_balance=10000, transaction_fee_percent=0.001):
        """
        Initialize the environment with safety measures
        
        Args:
            prices: Array of price data
            features: Array of feature data
            initial_balance: Starting balance for trading
            transaction_fee_percent: Fee percentage for transactions
        """
        # Calculate a safe max_steps value based on the data size
        self.max_steps = min(len(prices), len(features)) - 1
        
        super().__init__(prices=prices, features=features, initial_balance=initial_balance, 
                         transaction_fee_percent=transaction_fee_percent)
    
    def _calculate_max_shares(self, price):
        """
        Calculate the maximum number of shares that can be bought with current balance
        """
        max_shares = self.balance // (price * (1 + self.transaction_fee_percent))
        return int(max_shares)
    
    def _calculate_reward(self, action=None):
        """
        Calculate the reward based on the portfolio value change
        
        Args:
            action: The action taken (optional, not used in this implementation)
            
        Returns:
            float: The calculated reward
        """
        # Get current portfolio value
        current_value = self._get_portfolio_value()
        
        # Calculate reward based on portfolio value change
        if len(self.portfolio_values) > 0:
            previous_value = self.portfolio_values[-1]
            reward = (current_value - previous_value) / previous_value
        else:
            reward = 0
        
        return reward
    
    def _get_observation(self):
        """
        Get current observation with safety checks
        """
        # Ensure current_step doesn't exceed the size of features
        if self.current_step >= len(self.features):
            self.current_step = len(self.features) - 1
        
        # Get the current features
        features = self.features[self.current_step]
        
        # Create account info (3 elements to match the expected shape, with float32 type)
        account_info = np.array([
            float(self.balance),
            float(self.shares_held),
            float(self.cost_basis)
        ], dtype=np.float32)
        
        # Create observation dictionary
        observation = {
            'account_info': account_info,
            'features': features
        }
        
        return observation
    
    def step(self, action):
        """
        Update environment state based on agent action with safety checks
        """
        # Update current step with bounds checking
        self.current_step += 1
        if self.current_step >= min(len(self.prices), len(self.features)):
            # If we've reached the end of our data, terminate the episode
            self.current_step = min(len(self.prices), len(self.features)) - 1  # Ensure we don't go out of bounds
            
            # Get the current observation
            observation = self._get_observation()
            
            # Calculate the final reward
            reward = self._calculate_reward(action)
            
            # Set terminated to True since we've reached the end of the data
            terminated = True
            truncated = False
            
            # Get additional info
            info = self._get_info()
            
            # Append the current portfolio value to our list
            self.portfolio_values.append(self._get_portfolio_value())
            
            return observation, reward, terminated, truncated, info
        
        # Get current price
        current_price = self.prices[self.current_step]
        
        # Process the action
        action_type = action
        
        if action_type == 0:  # Buy
            shares_to_buy = self._calculate_max_shares(current_price)
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price
                transaction_fee = cost * self.transaction_fee_percent
                total_cost = cost + transaction_fee
                
                self.balance -= total_cost
                self.shares_held += shares_to_buy
                self.total_transaction_fees += transaction_fee
                self.cost_basis = current_price
                
        elif action_type == 1:  # Sell
            if self.shares_held > 0:
                proceeds = self.shares_held * current_price
                transaction_fee = proceeds * self.transaction_fee_percent
                total_proceeds = proceeds - transaction_fee
                
                self.balance += total_proceeds
                self.shares_held = 0
                self.total_transaction_fees += transaction_fee
                self.cost_basis = 0
                
        # Calculate reward, check if done, get observation and info
        reward = self._calculate_reward(action)
        terminated = self.current_step >= min(self.max_steps, len(self.prices) - 1, len(self.features) - 1)
        truncated = False
        observation = self._get_observation()
        info = self._get_info()
        
        # Append the current portfolio value to our list
        self.portfolio_values.append(self._get_portfolio_value())
        
        return observation, reward, terminated, truncated, info

def main():
    """Main function to run the script"""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train and backtest trading models')
    parser.add_argument('--data_source', type=str, default='synthetic', 
                        choices=['synthetic', 'yahoo'],
                        help='Data source to use (synthetic or yahoo)')
    parser.add_argument('--train_start', type=str, default='2020-01-01',
                        help='Start date for training data (YYYY-MM-DD)')
    parser.add_argument('--train_end', type=str, default='2023-12-31',
                        help='End date for training data (YYYY-MM-DD)')
    parser.add_argument('--test_start', type=str, default='2024-01-01',
                        help='Start date for test data (YYYY-MM-DD)')
    parser.add_argument('--test_end', type=str, default='2024-03-31',
                        help='End date for test data (YYYY-MM-DD)')
    parser.add_argument('--symbols', type=str, default='AAPL,AMZN',
                        help='Comma-separated list of stock symbols')
    
    args = parser.parse_args()
    
    # Convert symbols string to list
    symbols = args.symbols.split(',')
    
    print(f"Using {args.data_source} data")
    print(f"Training period: {args.train_start} to {args.train_end}")
    print(f"Testing period: {args.test_start} to {args.test_end}")
    print(f"Symbols: {', '.join(symbols)}")
    
    # Initialize the model trainer
    model_trainer = ModelTrainer(models_dir='models')
    
    # Train models for each symbol
    trained_models = {}
    for symbol in symbols:
        model = model_trainer.train_model(
            symbol=symbol, 
            train_start_date=args.train_start, 
            train_end_date=args.train_end, 
            data_source=args.data_source,
            data_fetcher_factory=DataFetcherFactory,
            trading_env_class=SafeTradingEnvironment
        )
        if model:
            trained_models[symbol] = model
    
    if not trained_models:
        print("No models were successfully trained. Exiting.")
        return
    
    # Initialize the backtester
    backtester = Backtester(results_dir='results')
    
    # Backtest models
    results_dict = {}
    for symbol in trained_models.keys():
        model_name = f"ppo_{symbol}_{args.train_start.split('-')[0]}_{args.train_end.split('-')[0]}"
        results = backtester.backtest_model(
            model_name=model_name,
            test_start_date=args.test_start,
            test_end_date=args.test_end,
            data_source=args.data_source,
            data_fetcher_factory=DataFetcherFactory,
            trading_env_class=SafeTradingEnvironment
        )
        if results:
            results_dict[symbol] = results
    
    if not results_dict:
        print("No models were successfully backtested. Exiting.")
        return
    
    # Get NASDAQ performance for comparison
    nasdaq_performance = backtester.get_market_performance(
        symbol="^IXIC",
        test_start_date=args.test_start,
        test_end_date=args.test_end,
        data_source=args.data_source,
        data_fetcher_factory=DataFetcherFactory
    )
    
    # Plot comparison even with just one model
    backtester.plot_comparison(
        results_dict=results_dict,
        market_performance=nasdaq_performance,
        test_start_date=args.test_start,
        test_end_date=args.test_end,
        market_name="NASDAQ"
    )
    
    print("\nTraining and backtesting completed successfully!")

if __name__ == "__main__":
    main() 