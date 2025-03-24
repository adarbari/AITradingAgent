#!/usr/bin/env python3
"""
Generic script for training and backtesting trading models on historical stock data.
Handles multiple data sources, feature engineering, and comprehensive result visualization.
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from datetime import datetime, timedelta
import yfinance as yf
from stable_baselines3 import PPO

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data import DataFetcherFactory
from src.models import ModelTrainer
from src.backtest import Backtester
from src.agent.trading_env import TradingEnvironment
from src.train.trainer import TrainingManager
from src.utils.feature_utils import prepare_robust_features, prepare_features_from_indicators, get_data

# Import feature engineering module
from src.feature_engineering import process_features, FeatureRegistry, FEATURE_CONFIGS
from src.feature_engineering.pipeline import FeaturePipeline
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

def train_model(symbol, train_start, train_end, model_path=None, 
                timesteps=100000, feature_count=21, data_source="yahoo",
                trading_env_class=TradingEnvironment, verbose=1,
                save_model=True, synthetic_params=None, force_retrain=False,
                feature_set="standard"):
    """
    Train a trading agent on historical data.
    
    Args:
        symbol (str): Symbol to train on
        train_start (str): Start date for training data
        train_end (str): End date for training data
        model_path (str, optional): Path to save model to
        timesteps (int): Number of timesteps to train for
        feature_count (int): Number of features to use
        data_source (str): Source of data ("yfinance", "synthetic")
        trading_env_class (class): Trading environment class
        verbose (int): Verbosity level
        save_model (bool): Whether to save the model
        synthetic_params (dict): Parameters for synthetic data generation
        force_retrain (bool): Force retraining even if cached model exists
        feature_set (str): Feature set configuration to use
    
    Returns:
        tuple: (trained_model, model_path)
    """
    print(f"Training model for {symbol} from {train_start} to {train_end}")
    
    # Create models directory if needed
    models_dir = os.path.dirname(model_path) if model_path else "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Fetch training data
    data = get_data(symbol, train_start, train_end, data_source, synthetic_params)
    if data is None or len(data) == 0:
        print(f"Error: No data available for {symbol} from {train_start} to {train_end}")
        return None, None
    
    print(f"Fetched {len(data)} days of data for training")
    
    # Generate features using the feature engineering module
    print(f"Using feature engineering module with feature set: {feature_set}")
    # Initialize feature cache
    cache = FeatureCache(cache_dir=".feature_cache", enable_cache=True, verbose=verbose > 0)
    cache_key = cache.get_cache_key(symbol, train_start, train_end, feature_set)
    
    # Check if features are cached
    cached_features = cache.load(cache_key)
    if cached_features is not None and not force_retrain:
        print("Using cached features")
        features = cached_features
    else:
        print("Computing features from data")
        features = process_features(data, feature_set=feature_set, verbose=verbose > 0)
        cache.save(features, cache_key)
    
    print(f"Prepared {len(features)} data points with {len(features.columns)} features for training")
    
    # Use the TrainingManager to get the model (will use cache if available)
    training_manager = TrainingManager(models_dir=models_dir, verbose=verbose)
    
    # Create model parameters dictionary
    model_params = None  # Default to None for now
    
    # Get or train the model
    model, path = training_manager.get_model(
        symbol=symbol,
        train_start=train_start,
        train_end=train_end,
        feature_count=len(features.columns),
        data_source=data_source,
        timesteps=timesteps,
        force_train=force_retrain,
        synthetic_params=synthetic_params,
        model_params=model_params,
        features=features,
        prices=data['Close'].values
    )
    
    return model, path

def fetch_and_prepare_data(symbol, start_date, end_date, data_source='yahoo', min_data_points=5):
    """
    Fetch and prepare data for training or backtesting.
    Handles error cases and ensures minimum data requirements.
    
    Args:
        symbol (str): Stock symbol
        start_date (str): Start date (YYYY-MM-DD)
        end_date (str): End date (YYYY-MM-DD)
        data_source (str): Source for data ('yahoo' or 'synthetic')
        min_data_points (int): Minimum number of data points required
        
    Returns:
        pd.DataFrame: Prepared data with technical indicators
    """
    # Use the centralized get_data function from feature_utils
    data = get_data(symbol, start_date, end_date, data_source)
    
    # Ensure we have at least the minimum data points
    if data is None or len(data) < min_data_points:
        print(f"Warning: Not enough data points ({len(data) if data is not None else 0}). At least {min_data_points} are required.")
        print("Generating additional synthetic data to supplement...")
        
        # Generate more synthetic data by extending the date range
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        extended_end = start_dt + timedelta(days=30)  # Always ensure at least a month of data
        extended_end_str = extended_end.strftime("%Y-%m-%d")
            
        # Get synthetic data with extended range
        data = get_data(symbol, start_date, extended_end_str, data_source="synthetic")
        print(f"Using {len(data)} data points for analysis.")
    
    return data

def backtest_model(model_path, symbol, test_start, test_end, data_source='yahoo', feature_set="standard"):
    """
    Backtest a trained model on test data
    
    Parameters:
    -----------
    model_path: str
        Path to the trained model
    symbol: str
        Stock symbol to backtest
    test_start: str
        Start date for test data (YYYY-MM-DD)
    test_end: str
        End date for test data (YYYY-MM-DD)
    data_source: str
        Source for data fetching ('yahoo' or 'synthetic')
    feature_set: str
        Feature set configuration to use
    
    Returns:
    --------
    dict
        Dictionary containing backtest results
    """
    from datetime import datetime  # Import datetime for timestamp
    
    print(f"Backtesting model from {test_start} to {test_end}")
    
    # Fetch and prepare test data
    test_data = fetch_and_prepare_data(symbol, test_start, test_end, data_source)
    
    # Generate features using the feature engineering module
    print(f"Using feature engineering module with feature set: {feature_set}")
    # Initialize feature cache
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
    
    # Create and run the trading environment
    env = TradingEnvironment(
        prices=test_data['Close'].values,
        features=features,
        initial_balance=10000,
        transaction_fee_percent=0.001
    )
    # Handle different gym interfaces
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        obs, _ = reset_result  # New Gym API returns (obs, info)
    else:
        obs = reset_result  # Old API returns just obs
    
    # Load model
    model = PPO.load(model_path)
    
    # Run backtest
    done = False
    portfolio_values = []
    actions_taken = []
    
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
        actions_taken.append(action)
    
    # Check if any steps were taken in the backtest
    if not portfolio_values:
        return {
            "error": "No trading steps were performed in backtest. Check if the test data set is valid."
        }
    
    # Get buy and hold results
    initial_price = test_data['Close'].iloc[0]
    final_price = test_data['Close'].iloc[-1]
    buy_hold_return = (final_price - initial_price) / initial_price
    
    # Calculate strategy returns
    initial_value = portfolio_values[0]
    final_value = portfolio_values[-1]
    strategy_return = (final_value - initial_value) / initial_value
    
    # Create returns DataFrame
    date_range = test_data.index[:len(portfolio_values)]
    returns_df = pd.DataFrame({
        'Date': date_range,
        'Strategy': portfolio_values,
        'Buy_Hold': [initial_price * (1 + buy_hold_return * i / len(portfolio_values)) for i in range(len(portfolio_values))]
    })
    returns_df.set_index('Date', inplace=True)
    
    # Calculate metrics
    total_trades = sum(1 for i in range(1, len(actions_taken)) if actions_taken[i] != actions_taken[i-1])
    sharpe_ratio = np.mean(np.diff(portfolio_values) / portfolio_values[:-1]) / np.std(np.diff(portfolio_values) / portfolio_values[:-1]) if len(portfolio_values) > 1 else 0
    max_drawdown = calculate_max_drawdown(portfolio_values)
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(returns_df.index, returns_df['Strategy'], label='Trading Strategy')
    plt.plot(returns_df.index, returns_df['Buy_Hold'], label='Buy and Hold')
    plt.title(f'Backtest Results for {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    
    # Save plot
    results_dir = os.path.join('results', symbol)
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(results_dir, f'backtest_{timestamp}.png')
    plt.savefig(plot_path)
    print(f"Backtest plot saved to {plot_path}")
    
    # Save portfolio values
    portfolio_values_path = os.path.join(results_dir, f'portfolio_values_{timestamp}.csv')
    returns_df.to_csv(portfolio_values_path)
    print(f"Portfolio values saved to {portfolio_values_path}")
    
    # Return results
    return {
        "symbol": symbol,
        "model_path": model_path,
        "test_period": f"{test_start} to {test_end}",
        "initial_value": initial_value,
        "final_value": final_value,
        "strategy_return": strategy_return,
        "buy_hold_return": buy_hold_return,
        "outperformance": strategy_return - buy_hold_return,
        "total_trades": total_trades,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "plot_path": plot_path,
        "portfolio_values_path": portfolio_values_path
    }

def main():
    """Main function to parse arguments and run training/backtesting."""
    parser = argparse.ArgumentParser(description="Train and backtest trading models")
    
    # General arguments
    parser.add_argument("--symbol", type=str, default="AAPL",
                       help="Stock symbol to train on (default: AAPL)")
    parser.add_argument("--data-source", type=str, default="yahoo",
                       choices=["yahoo", "synthetic"],
                       help="Source of data (default: yahoo)")
    parser.add_argument("--feature-count", type=int, default=21,
                       help="Number of features to use (default: 21)")
    parser.add_argument("--feature-set", type=str, default="standard",
                       help="Feature set configuration to use (default: standard)")
    
    # Training arguments
    parser.add_argument("--train", action="store_true",
                       help="Train a new model")
    parser.add_argument("--train-start", type=str, default="2020-01-01",
                       help="Start date for training data (default: 2020-01-01)")
    parser.add_argument("--train-end", type=str, default="2022-12-31",
                       help="End date for training data (default: 2022-12-31)")
    parser.add_argument("--timesteps", type=int, default=100000,
                       help="Training timesteps (default: 100000)")
    parser.add_argument("--force", action="store_true", 
                       help="Force retraining even if a cached model exists")
    
    # Backtesting arguments
    parser.add_argument("--backtest", action="store_true",
                       help="Backtest the model")
    parser.add_argument("--test-start", type=str, default="2023-01-01",
                       help="Start date for testing data (default: 2023-01-01)")
    parser.add_argument("--test-end", type=str, 
                       default=(datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"),
                       help="End date for testing data (default: yesterday)")
    parser.add_argument("--fee", type=float, default=0.001,
                       help="Transaction fee percentage (default: 0.001)")
    
    # Model arguments
    parser.add_argument("--model-path", type=str,
                       help="Path to load or save model (default: auto-generated)")
    
    # Directory arguments
    parser.add_argument("--models-dir", type=str, default="models",
                       help="Directory to save models (default: models)")
    parser.add_argument("--results-dir", type=str, default="results",
                       help="Directory to save results (default: results)")
    
    # Synthetic data arguments
    parser.add_argument("--synthetic-initial-price", type=float, default=100.0,
                       help="Initial price for synthetic data (default: 100.0)")
    parser.add_argument("--synthetic-volatility", type=float, default=0.01,
                       help="Volatility for synthetic data (default: 0.01)")
    parser.add_argument("--synthetic-drift", type=float, default=0.0001,
                       help="Drift for synthetic data (default: 0.0001)")
    
    args = parser.parse_args()
    
    # Normalize data source name for backwards compatibility
    if args.data_source.lower() == "yfinance":
        args.data_source = "yahoo"
        
    # Create synthetic parameters dict if using synthetic data
    synthetic_params = None
    if args.data_source == "synthetic":
        synthetic_params = {
            "initial_price": args.synthetic_initial_price,
            "volatility": args.synthetic_volatility,
            "drift": args.synthetic_drift
        }
    
    # Create directories if they don't exist
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Auto-generate model path if not provided
    if not args.model_path:
        model_name = f"ppo_{args.symbol}_{args.train_start.split('-')[0]}_{args.train_end.split('-')[0]}"
        args.model_path = os.path.join(args.models_dir, model_name)
    
    # Train model if requested
    model = None
    if args.train:
        model, args.model_path = train_model(
            symbol=args.symbol,
            train_start=args.train_start,
            train_end=args.train_end,
            model_path=args.model_path,
            timesteps=args.timesteps,
            feature_count=args.feature_count,
            data_source=args.data_source,
            synthetic_params=synthetic_params,
            force_retrain=args.force,
            feature_set=args.feature_set
        )
    
    # Backtest model if requested
    if args.backtest:
        if not args.train and not os.path.exists(args.model_path):
            print(f"Error: Model file {args.model_path} does not exist. Train a model first or provide a valid model path.")
            return
        
        model_path = args.model_path
        results = backtest_model(
            model_path=model_path,
            symbol=args.symbol,
            test_start=args.test_start,
            test_end=args.test_end,
            data_source=args.data_source,
            feature_set=args.feature_set
        )
        
        if results and not results.get('error'):
            print("\nBacktest Results Summary:")
            print(f"Symbol: {results['symbol']}")
            print(f"Test Period: {results['test_period']}")
            print(f"Initial Value: ${results['initial_value']:.2f}")
            print(f"Final Value: ${results['final_value']:.2f}")
            print(f"Strategy Return: {results['strategy_return']*100:.2f}%")
            print(f"Buy & Hold Return: {results['buy_hold_return']*100:.2f}%")
            print(f"Outperformance: {results['outperformance']*100:.2f}%")
            print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
            print(f"Max Drawdown: {results['max_drawdown']*100:.2f}%")
            print(f"Total Trades: {results['total_trades']}")
            print(f"Results saved to {results['plot_path']}")
        elif results and results.get('error'):
            print(f"Error in backtesting: {results['error']}")
        else:
            print("Backtesting failed to return valid results.")


if __name__ == "__main__":
    main() 