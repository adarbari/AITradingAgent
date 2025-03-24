#!/usr/bin/env python3
"""
Script for training and backtesting an AMZN trading model for specific date ranges.
Uses the basic components directly to avoid issues with the training manager.
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from stable_baselines3 import PPO

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.yahoo_data_fetcher import YahooDataFetcher
from src.models.trainer import ModelTrainer
from src.backtest.backtester import Backtester
from src.agent.trading_env import TradingEnvironment
from src.feature_engineering import process_features
from src.feature_engineering.cache import FeatureCache

def calculate_max_drawdown(portfolio_values):
    """
    Calculate the maximum drawdown of a portfolio.
    
    Parameters:
    -----------
    portfolio_values: list or numpy.array
        List of portfolio values over time
    
    Returns:
    --------
    float
        Maximum drawdown as a decimal (not percentage)
    """
    portfolio_values = np.array(portfolio_values)
    peak_values = np.maximum.accumulate(portfolio_values)
    drawdowns = (portfolio_values - peak_values) / peak_values
    max_drawdown = abs(min(drawdowns)) if len(drawdowns) > 0 else 0
    return max_drawdown

def train_amzn_model(train_start, train_end, model_dir, timesteps=20000, feature_set="standard"):
    """
    Train a model for AMZN using the specified date range.
    
    Args:
        train_start (str): Start date in YYYY-MM-DD format
        train_end (str): End date in YYYY-MM-DD format
        model_dir (str): Directory to save the model
        timesteps (int): Number of timesteps for training
        feature_set (str): Feature set configuration
        
    Returns:
        tuple: (model, model_path)
    """
    print(f"Training AMZN model from {train_start} to {train_end}")
    
    # Create models directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Fetch training data
    data_fetcher = YahooDataFetcher()
    symbol = "AMZN"
    data = data_fetcher.fetch_ticker_data(symbol, train_start, train_end)
    
    if data is None or len(data) == 0:
        print(f"Error: No data available for {symbol} from {train_start} to {train_end}")
        return None, None
    
    print(f"Fetched {len(data)} days of data for training")
    
    # Generate features
    print(f"Using feature engineering with feature set: {feature_set}")
    cache = FeatureCache(cache_dir=".feature_cache", enable_cache=True, verbose=True)
    cache_key = cache.get_cache_key(symbol, train_start, train_end, feature_set)
    
    # Check if features are cached
    cached_features = cache.load(cache_key)
    if cached_features is not None:
        print("Using cached features")
        features = cached_features
    else:
        print("Computing features from data")
        features = process_features(data, feature_set=feature_set, verbose=True)
        cache.save(features, cache_key)
    
    print(f"Prepared {len(features)} data points with {len(features.columns)} features for training")
    
    # Initialize model trainer
    model_trainer = ModelTrainer(models_dir=model_dir, verbose=1)
    
    # Train the model
    model_path = model_trainer.train_model(
        env_class=TradingEnvironment,
        prices=data['Close'].values,
        features=features.values,
        symbol=symbol,
        train_start=train_start,
        train_end=train_end,
        total_timesteps=timesteps
    )
    
    # Load the model
    model = PPO.load(model_path)
    
    return model, model_path

def backtest_amzn_model(model_path, test_start, test_end, results_dir, feature_set="standard"):
    """
    Backtest a trained model on AMZN data for the specified date range.
    
    Args:
        model_path (str): Path to the saved model
        test_start (str): Start date in YYYY-MM-DD format
        test_end (str): End date in YYYY-MM-DD format
        results_dir (str): Directory to save results
        feature_set (str): Feature set configuration
        
    Returns:
        dict: Results of the backtest
    """
    print(f"Backtesting model {model_path} from {test_start} to {test_end}")
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Fetch test data
    data_fetcher = YahooDataFetcher()
    symbol = "AMZN"
    test_data = data_fetcher.fetch_ticker_data(symbol, test_start, test_end)
    
    if test_data is None or len(test_data) == 0:
        print(f"Error: No data available for {symbol} from {test_start} to {test_end}")
        return None
    
    print(f"Fetched {len(test_data)} days of data for backtesting")
    
    # Generate features
    print(f"Using feature engineering with feature set: {feature_set}")
    cache = FeatureCache(cache_dir=".feature_cache", enable_cache=True, verbose=True)
    cache_key = cache.get_cache_key(symbol, test_start, test_end, feature_set)
    
    # Check if features are cached
    cached_features = cache.load(cache_key)
    if cached_features is not None:
        print("Using cached features")
        features = cached_features
    else:
        print("Computing features from data")
        features = process_features(test_data, feature_set=feature_set, verbose=True)
        cache.save(features, cache_key)
    
    print(f"Prepared {len(features)} data points with {len(features.columns)} features for testing")
    
    # Load model
    model = PPO.load(model_path)
    
    # Create trading environment
    env = TradingEnvironment(
        prices=test_data['Close'].values,
        features=features.values,
        initial_balance=10000,
        transaction_fee_percent=0.001
    )
    
    # Run manual backtest since the Backtester class doesn't accept the right parameters
    print("Running backtest...")
    
    # Reset environment
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        obs, _ = reset_result  # New Gym API returns (obs, info)
    else:
        obs = reset_result  # Old API returns just obs
    
    # Run simulation
    done = False
    portfolio_values = []
    actions = []
    
    while not done:
        action, _ = model.predict(obs)
        step_result = env.step(action)
        
        # Handle different gym interfaces
        if len(step_result) == 5:  # New Gym API: obs, reward, terminated, truncated, info
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:  # Old Gym API: obs, reward, done, info
            obs, reward, done, info = step_result
            
        portfolio_values.append(info['portfolio_value'])
        actions.append(action)
    
    # Calculate metrics manually
    initial_price = test_data['Close'].iloc[0]
    final_price = test_data['Close'].iloc[-1]
    buy_hold_return = (final_price - initial_price) / initial_price
    
    initial_value = portfolio_values[0]
    final_value = portfolio_values[-1]
    strategy_return = (final_value - initial_value) / initial_value
    
    # Calculate outperformance
    outperformance = strategy_return - buy_hold_return
    
    # Calculate other metrics
    total_trades = sum(1 for i in range(1, len(actions)) if actions[i] != actions[i-1])
    
    # Sharpe ratio (simplified)
    if len(portfolio_values) > 1:
        daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)  # Annualized
    else:
        sharpe_ratio = 0
    
    max_drawdown = calculate_max_drawdown(portfolio_values)
    
    # Compile metrics
    metrics = {
        'symbol': symbol,
        'test_period': f"{test_start} to {test_end}",
        'initial_value': initial_value,
        'final_value': final_value,
        'strategy_return': strategy_return,
        'buy_hold_return': buy_hold_return,
        'outperformance': outperformance,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'total_trades': total_trades
    }
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(test_data.index[:len(portfolio_values)], portfolio_values, label='Trading Strategy')
    plt.plot(test_data.index[:len(portfolio_values)], 
             [initial_price * (1 + buy_hold_return * i / len(portfolio_values)) * (initial_value / initial_price) 
              for i in range(len(portfolio_values))], 
             label='Buy and Hold')
    plt.title(f'Backtest Results for {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plot_path = os.path.join(results_dir, f"{symbol}_backtest_plot.png")
    plt.savefig(plot_path)
    plt.close()
    metrics['plot_path'] = plot_path
    
    # Save results to JSON
    results_path = os.path.join(results_dir, f"{symbol}_backtest_results.json")
    pd.Series(metrics).to_json(results_path)
    print(f"Results saved to {results_path}")
    
    # Print summary
    print("\nBacktest Results:")
    print(f"Initial portfolio value: ${metrics['initial_value']:.2f}")
    print(f"Final portfolio value: ${metrics['final_value']:.2f}")
    print(f"Strategy return: {metrics['strategy_return']*100:.2f}%")
    print(f"Buy & Hold return: {metrics['buy_hold_return']*100:.2f}%")
    print(f"Outperformance: {metrics['outperformance']*100:.2f}%")
    print(f"Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max drawdown: {metrics['max_drawdown']*100:.2f}%")
    print(f"Total trades: {metrics['total_trades']}")
    
    return metrics

def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description="Train and backtest AMZN model")
    
    # Training options
    parser.add_argument("--train", action="store_true", help="Train a new model")
    parser.add_argument("--train-start", type=str, default="2023-01-01", 
                        help="Training start date (YYYY-MM-DD)")
    parser.add_argument("--train-end", type=str, default="2024-12-31",
                        help="Training end date (YYYY-MM-DD)")
    parser.add_argument("--timesteps", type=int, default=20000,
                        help="Training timesteps")
    
    # Backtest options
    parser.add_argument("--backtest", action="store_true", help="Backtest the model")
    parser.add_argument("--test-start", type=str, default="2025-01-01",
                        help="Test start date (YYYY-MM-DD)")
    parser.add_argument("--test-end", type=str, default="2025-03-15",
                        help="Test end date (YYYY-MM-DD)")
    
    # Other options
    parser.add_argument("--feature-set", type=str, default="standard",
                        help="Feature set to use")
    parser.add_argument("--model-dir", type=str, default="models",
                        help="Directory to save models")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Directory to save results")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Specific model path (will be auto-generated if not provided)")
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Auto-generate model path if not provided
    if not args.model_path:
        args.model_path = os.path.join(args.model_dir, f"ppo_AMZN_2023_2024_model")
    
    # Train model if requested
    if args.train:
        model, model_path = train_amzn_model(
            train_start=args.train_start,
            train_end=args.train_end,
            model_dir=args.model_dir,
            timesteps=args.timesteps,
            feature_set=args.feature_set
        )
        
        if model is None:
            print("Training failed.")
            return
            
        args.model_path = model_path
    
    # Backtest model if requested
    if args.backtest:
        if not os.path.exists(f"{args.model_path}.zip"):
            print(f"Error: Model file {args.model_path}.zip does not exist.")
            return
            
        results = backtest_amzn_model(
            model_path=args.model_path,
            test_start=args.test_start,
            test_end=args.test_end,
            results_dir=args.results_dir,
            feature_set=args.feature_set
        )
        
        if results is None:
            print("Backtesting failed.")
            return

if __name__ == "__main__":
    main() 