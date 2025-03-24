#!/usr/bin/env python3
"""
Script for training and backtesting trading models on historical stock data.
"""
import os
import sys
import argparse
import gymnasium as gym
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data import DataFetcherFactory
from src.models import ModelTrainer
from src.backtest import Backtester
from src.agent.trading_env import TradingEnvironment


def train_model(data_fetcher_type, symbol, start_date, end_date,
               trading_env_class=TradingEnvironment, model_path=None, 
               timesteps=100000, verbose=1):
    """
    Train a trading agent on historical data.
    
    Args:
        data_fetcher_type (str): Type of data fetcher to use (yahoo, synthetic, csv)
        symbol (str): Symbol to fetch data for
        start_date (str): Start date for training data
        end_date (str): End date for training data
        trading_env_class (class): Class of trading environment to use
        model_path (str, optional): Path to save model to
        timesteps (int): Number of timesteps to train for
        verbose (int): Verbosity level
    
    Returns:
        tuple: (trained_model, model_path)
    """
    print(f"Training model for {symbol} from {start_date} to {end_date}")
    
    # Create data fetcher
    data_fetcher = DataFetcherFactory.create_data_fetcher(data_fetcher_type)
    
    # Fetch training data
    print("Fetching training data...")
    training_data = data_fetcher.fetch_data(symbol, start_date, end_date)
    
    # Check if we got data
    if training_data is None or len(training_data) == 0:
        print(f"Error: No training data available for {symbol} from {start_date} to {end_date}.")
        return None, None
        
    print(f"Fetched {len(training_data)} data points for training.")
    
    # Add technical indicators
    print("Adding technical indicators...")
    training_data = data_fetcher.add_technical_indicators(training_data)
    
    # Prepare data for agent
    print("Preparing data for the agent...")
    result = data_fetcher.prepare_data_for_agent(training_data)
    
    # Handle different return types from prepare_data_for_agent
    if isinstance(result, tuple) and len(result) == 2:
        # SyntheticDataFetcher returns (prices, features)
        prices, features = result
    else:
        # BaseDataFetcher returns just features, which may be windowed data
        features = result
        # If features length is less than original data, it's using windows
        # Adjust prices to match the length of features
        if len(features) < len(training_data):
            window_size = 20  # Default window size
            prices = training_data['Close'].values[window_size-1:]
        else:
            prices = training_data['Close'].values
    
    # Check if we have enough data
    if len(features) == 0:
        print(f"Error: Not enough data points after preparation.")
        return None, None
    
    # Final check to ensure lengths match
    if len(prices) != len(features):
        print(f"Warning: Adjusting price data length from {len(prices)} to {len(features)} to match features")
        if len(prices) > len(features):
            prices = prices[-len(features):]
        else:
            # This should not happen, but just in case
            features = features[-len(prices):]
    
    print(f"Prepared {len(features)} data points for training.")
    
    # Create training environment
    env = trading_env_class(
        prices=prices,
        features=features,
        initial_balance=10000,
        transaction_fee_percent=0.001
    )
    
    # Choose appropriate policy based on observation space type
    if isinstance(env.observation_space, gym.spaces.Dict):
        policy = "MultiInputPolicy"  # For dictionary observation spaces
    else:
        policy = "MlpPolicy"  # For Box observation spaces
    
    # Train agent
    print(f"Training model with {timesteps} timesteps...")
    
    # Create model trainer
    trainer = ModelTrainer(models_dir=os.path.dirname(model_path) if model_path else "models", 
                          verbose=verbose)
    
    # Train the model
    path = trainer.train_model(
        env_class=trading_env_class,
        prices=prices,
        features=features,
        symbol=symbol,
        train_start=start_date,
        train_end=end_date,
        total_timesteps=timesteps
    )
    
    # Load the trained model
    model = trainer.load_model(path)
    
    print(f"Model trained and saved to {path}")
    
    return model, path


def backtest_model(model_path, symbol, start_date, end_date, data_source, 
                  results_dir="results", trading_env_class=TradingEnvironment):
    """
    Backtest a trained model on historical data.
    
    Args:
        model_path (str): Path to the trained model
        symbol (str): Symbol to test on
        start_date (str): Start date for testing data
        end_date (str): End date for testing data
        data_source (str): Source of data (e.g., 'yahoo', 'synthetic')
        results_dir (str): Directory to save results
        trading_env_class (class): Trading environment class
        
    Returns:
        dict: Results of the backtest
    """
    print(f"Backtesting model for {symbol} from {start_date} to {end_date}")
    
    # Create backtester
    backtester = Backtester(results_dir=results_dir)
    
    try:
        # Run backtest
        results = backtester.backtest_model(
            model_path=model_path,
            symbol=symbol,
            test_start=start_date,
            test_end=end_date,
            data_source=data_source,
            env_class=trading_env_class
        )
        
        # Print results
        print("\nBacktest Results:")
        print(f"Total Return: {results['total_return']:.2f}%")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
        
        # Save results
        results_file = os.path.join(results_dir, f"{symbol}_backtest_results.json")
        backtester.save_results(results, results_file)
        print(f"Results saved to {results_file}")
        
        # Print path to the generated plot
        print(f"Performance plot saved to {results['plot_path']}")
        
        return results
        
    except Exception as e:
        print(f"Error during backtesting: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main function to parse arguments and run training/backtesting."""
    parser = argparse.ArgumentParser(description="Train and backtest trading models")
    
    # General arguments
    parser.add_argument("--symbol", type=str, default="AMZN",
                       help="Stock symbol to train on (default: AMZN)")
    parser.add_argument("--data-source", type=str, default="yahoo",
                       choices=["yahoo", "synthetic", "csv"],
                       help="Source of data (default: yahoo)")
    
    # Training arguments
    parser.add_argument("--train", action="store_true",
                       help="Train a new model")
    parser.add_argument("--train-start", type=str, default="2023-01-01",
                       help="Start date for training data (default: 2023-01-01)")
    parser.add_argument("--train-end", type=str, default="2024-12-31",
                       help="End date for training data (default: 2024-12-31)")
    parser.add_argument("--timesteps", type=int, default=100000,
                       help="Training timesteps (default: 100000)")
    
    # Backtesting arguments
    parser.add_argument("--backtest", action="store_true",
                       help="Backtest the model")
    parser.add_argument("--test-start", type=str, default="2025-01-01",
                       help="Start date for testing data (default: 2025-01-01)")
    parser.add_argument("--test-end", type=str, 
                       default=(datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"),
                       help="End date for testing data (default: current date + 30 days)")
    
    # Model arguments
    parser.add_argument("--model-path", type=str,
                       help="Path to load or save model (default: auto-generated)")
    
    # Directory arguments
    parser.add_argument("--models-dir", type=str, default="models",
                       help="Directory to store models (default: models)")
    parser.add_argument("--results-dir", type=str, default="results",
                       help="Directory to store results (default: results)")
    
    # Parse arguments
    args = parser.parse_args()
    
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
            data_fetcher_type=args.data_source,
            symbol=args.symbol,
            start_date=args.train_start,
            end_date=args.train_end,
            model_path=args.model_path,
            timesteps=args.timesteps
        )
    
    # Backtest model if requested
    if args.backtest:
        if not args.model_path or not os.path.exists(f"{args.model_path}.zip"):
            print(f"Error: Model file not found at {args.model_path}.zip")
            print("Please train a model first or provide a valid model path.")
            return
        
        backtest_model(
            model_path=args.model_path,
            symbol=args.symbol,
            start_date=args.test_start,
            end_date=args.test_end,
            data_source=args.data_source,
            results_dir=args.results_dir
        )


if __name__ == "__main__":
    main() 