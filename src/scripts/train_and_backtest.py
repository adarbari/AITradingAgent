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

def prepare_robust_features(data, feature_count=21):
    """
    Prepare features for the trading agent with robust error handling.
    
    Args:
        data (pd.DataFrame): Raw price data with OHLCV columns
        feature_count (int): Expected number of features
        
    Returns:
        np.array: Processed features with shape (n_samples, feature_count)
    """
    # Calculate technical indicators with robust error handling
    features = []
    
    # Price data
    close_prices = data['Close'].values
    
    # 1. Price changes
    price_returns = np.diff(close_prices, prepend=close_prices[0]) / np.maximum(close_prices, 1e-8)
    price_returns = np.nan_to_num(price_returns, nan=0.0, posinf=0.0, neginf=0.0)
    features.append(price_returns)
    
    # 2. Volatility (rolling std of returns)
    vol = pd.Series(price_returns).rolling(window=5).std().fillna(0).values
    vol = np.nan_to_num(vol, nan=0.0)
    features.append(vol)
    
    # 3. Volume changes
    volume = np.maximum(data['Volume'].values, 1)  # Ensure no zeros
    volume_changes = np.diff(volume, prepend=volume[0]) / volume
    volume_changes = np.nan_to_num(volume_changes, nan=0.0, posinf=0.0, neginf=0.0)
    features.append(volume_changes)
    
    # 4. Price momentum
    momentum = pd.Series(close_prices).pct_change(periods=5).fillna(0).values
    momentum = np.nan_to_num(momentum, nan=0.0, posinf=0.0, neginf=0.0)
    features.append(momentum)
    
    # 5. High-Low range
    high_low_range = (data['High'].values - data['Low'].values) / np.maximum(data['Close'].values, 1e-8)
    high_low_range = np.nan_to_num(high_low_range, nan=0.0, posinf=0.0, neginf=0.0)
    features.append(high_low_range)
    
    # If we need more features to match the expected count
    if feature_count > 5:
        # 6-10: Moving averages
        for period in [5, 10, 20, 50, 100]:
            ma = pd.Series(close_prices).rolling(window=min(period, len(close_prices))).mean().fillna(0).values
            ma = np.maximum(ma, 1e-8)  # Avoid division by zero
            ma_ratio = ma / np.maximum(close_prices, 1e-8)
            ma_ratio = np.nan_to_num(ma_ratio, nan=1.0, posinf=1.0, neginf=1.0)
            features.append(ma_ratio)
        
        # 11-15: RSI for different periods
        for period in [5, 10, 14, 20, 30]:
            delta = pd.Series(close_prices).diff().fillna(0)
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(window=min(period, len(gain))).mean().fillna(0)
            avg_loss = loss.rolling(window=min(period, len(loss))).mean().fillna(0)
            
            # Calculate RS and RSI
            rs = np.where(avg_loss < 1e-8, 1.0, avg_gain / np.maximum(avg_loss, 1e-8))
            rs = np.nan_to_num(rs, nan=1.0, posinf=1.0, neginf=1.0)
            rsi = 100 - (100 / (1 + rs))
            rsi = np.nan_to_num(rsi, nan=50.0)  # Default to neutral RSI
            features.append(rsi)
        
        # 16-18: Bollinger Bands
        for period in [10, 20, 30]:
            ma = pd.Series(close_prices).rolling(window=min(period, len(close_prices))).mean().fillna(0).values
            std = pd.Series(close_prices).rolling(window=min(period, len(close_prices))).std().fillna(0).values
            
            # Avoid division by zero
            ma = np.maximum(ma, 1e-8)
            
            upper_band = (ma + 2 * std) / np.maximum(close_prices, 1e-8)
            lower_band = (ma - 2 * std) / np.maximum(close_prices, 1e-8)
            
            # This can sometimes be zero, so add a small epsilon
            bandwidth = (upper_band - lower_band) / (ma + 1e-8)
            bandwidth = np.nan_to_num(bandwidth, nan=0.0, posinf=0.0, neginf=0.0)
            features.append(bandwidth)
        
        # 19-21: MACD
        ema12 = pd.Series(close_prices).ewm(span=12).mean().values
        ema26 = pd.Series(close_prices).ewm(span=26).mean().values
        macd = ema12 - ema26
        signal = pd.Series(macd).ewm(span=9).mean().values
        hist = macd - signal
        
        macd_feature = macd / np.maximum(close_prices, 1e-8)
        signal_feature = signal / np.maximum(close_prices, 1e-8)
        hist_feature = hist / np.maximum(close_prices, 1e-8)
        
        macd_feature = np.nan_to_num(macd_feature, nan=0.0, posinf=0.0, neginf=0.0)
        signal_feature = np.nan_to_num(signal_feature, nan=0.0, posinf=0.0, neginf=0.0)
        hist_feature = np.nan_to_num(hist_feature, nan=0.0, posinf=0.0, neginf=0.0)
        
        features.append(macd_feature)
        features.append(signal_feature)
        features.append(hist_feature)
    
    # Stack features into a 2D array
    features = np.stack(features, axis=1)
    
    # Final check for any remaining NaNs or infinities
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Ensure we have the right number of features
    if features.shape[1] < feature_count:
        # Pad with zeros if needed
        padding = np.zeros((features.shape[0], feature_count - features.shape[1]))
        features = np.concatenate([features, padding], axis=1)
    elif features.shape[1] > feature_count:
        # Trim if we have too many
        features = features[:, :feature_count]
    
    return features

def get_data(symbol, start_date, end_date, data_source="yfinance", synthetic_params=None):
    """
    Get data for training or testing.
    
    Args:
        symbol (str): Stock symbol
        start_date (str): Start date (YYYY-MM-DD)
        end_date (str): End date (YYYY-MM-DD)
        data_source (str): Source of data ("yfinance", "synthetic")
        synthetic_params (dict): Parameters for synthetic data generation
        
    Returns:
        pd.DataFrame: OHLCV data
    """
    if data_source == "yfinance":
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            if len(data) == 0:
                print(f"No data available for {symbol} from {start_date} to {end_date}. Using synthetic data.")
                data_source = "synthetic"
            else:
                return data
        except Exception as e:
            print(f"Error fetching {symbol} data: {e}. Using synthetic data.")
            data_source = "synthetic"
    
    if data_source == "synthetic":
        if synthetic_params is None:
            synthetic_params = {
                "initial_price": 100.0,
                "volatility": 0.02,
                "drift": 0.001,
                "volume_min": 1000000,
                "volume_max": 5000000
            }
        
        # Generate synthetic data
        days = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
        n_days = len(days)
        
        # Generate a random walk with drift for closing prices
        np.random.seed(42)  # For reproducibility
        daily_returns = np.random.normal(synthetic_params["drift"], 
                                         synthetic_params["volatility"], 
                                         n_days)
        
        # Calculate price series
        prices = np.zeros(n_days)
        prices[0] = synthetic_params["initial_price"]
        for i in range(1, n_days):
            prices[i] = prices[i-1] * (1 + daily_returns[i])
        
        # Create DataFrame
        df = pd.DataFrame(index=days)
        df['Close'] = prices
        df['Open'] = df['Close'] * (1 - np.random.normal(0, 0.005, n_days))
        df['High'] = df['Close'] * (1 + np.random.normal(0.005, 0.005, n_days))
        df['Low'] = df['Close'] * (1 - np.random.normal(0.005, 0.005, n_days))
        df['Volume'] = np.random.randint(synthetic_params["volume_min"], 
                                         synthetic_params["volume_max"], 
                                         size=n_days)
        
        # Ensure High is always highest and Low is always lowest
        for i in range(n_days):
            values = [df['Open'].iloc[i], df['Close'].iloc[i], df['High'].iloc[i], df['Low'].iloc[i]]
            df.loc[df.index[i], 'High'] = max(values)
            df.loc[df.index[i], 'Low'] = min(values)
        
        print(f"Generated synthetic data for {symbol} from {start_date} to {end_date}")
        return df
    
    return None

def train_model(symbol, train_start, train_end, model_path=None, 
                timesteps=100000, feature_count=21, data_source="yfinance",
                trading_env_class=TradingEnvironment, verbose=1,
                save_model=True, synthetic_params=None):
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
    
    Returns:
        tuple: (trained_model, model_path)
    """
    print(f"Training model for {symbol} from {train_start} to {train_end}")
    
    # Get training data
    print("Fetching training data...")
    training_data = get_data(symbol, train_start, train_end, data_source, synthetic_params)
    
    if training_data is None or len(training_data) == 0:
        print(f"Error: No training data available for {symbol} from {train_start} to {train_end}.")
        return None, None
    
    print(f"Fetched {len(training_data)} data points for training.")
    
    # Prepare features for the model
    print("Preparing data for the agent...")
    features = prepare_robust_features(training_data, feature_count)
    prices = training_data['Close'].values
    
    print(f"Prepared {len(features)} data points with {features.shape[1]} features.")
    
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
    
    # Create model trainer
    models_dir = os.path.dirname(model_path) if model_path else "models"
    os.makedirs(models_dir, exist_ok=True)
    
    trainer = ModelTrainer(models_dir=models_dir, verbose=verbose)
    
    # Train the model
    print(f"Training model with {timesteps} timesteps...")
    path = trainer.train_model(
        env_class=trading_env_class,
        prices=prices,
        features=features,
        symbol=symbol,
        train_start=train_start,
        train_end=train_end,
        total_timesteps=timesteps
    )
    
    # Load the trained model
    model = trainer.load_model(path)
    
    print(f"Model trained and saved to {path}")
    
    return model, path

def backtest_model(model_path, symbol, test_start, test_end, 
                  feature_count=21, data_source="yfinance", 
                  results_dir="results", trading_env_class=TradingEnvironment,
                  synthetic_params=None, transaction_fee_percent=0.001):
    """
    Backtest a trained model on historical data.
    
    Args:
        model_path (str): Path to the trained model
        symbol (str): Symbol to test on
        test_start (str): Start date for testing data
        test_end (str): End date for testing data
        feature_count (int): Number of features to use
        data_source (str): Source of data ("yfinance", "synthetic")
        results_dir (str): Directory to save results
        trading_env_class (class): Trading environment class to use
        synthetic_params (dict): Parameters for synthetic data generation
        transaction_fee_percent (float): Fee percentage for transactions
        
    Returns:
        dict: Results of the backtest
    """
    print(f"Backtesting model for {symbol} from {test_start} to {test_end}")
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Get test data
    print("Fetching test data...")
    test_data = get_data(symbol, test_start, test_end, data_source, synthetic_params)
    
    if test_data is None or len(test_data) == 0:
        print(f"Error: No test data available for {symbol} from {test_start} to {test_end}.")
        return None
    
    print(f"Fetched {len(test_data)} data points for testing.")
    
    # Prepare features for the model
    print("Preparing data for the agent...")
    features = prepare_robust_features(test_data, feature_count)
    prices = test_data['Close'].values
    
    print(f"Prepared {len(features)} data points with {features.shape[1]} features.")
    
    # Create environment
    env = trading_env_class(
        prices=prices,
        features=features,
        initial_balance=10000,
        transaction_fee_percent=transaction_fee_percent
    )
    
    # Load the trained model
    print(f"Loading model from {model_path}...")
    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Initialize tracking variables
    portfolio_values = []
    actions_taken = []
    positions = []
    current_step = 0
    
    # Reset the environment
    obs, _ = env.reset()
    
    # Run the backtest
    print("Running backtest...")
    while current_step < len(prices) - 1:
        # Get action from model
        try:
            action, _ = model.predict(obs, deterministic=True)
        except Exception as e:
            print(f"Error predicting action: {e}")
            break
        
        # Take step in environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Record data
        portfolio_values.append(info['portfolio_value'])
        actions_taken.append(info['actual_action'])
        positions.append(info.get('position', 0))  # Position (shares held)
        
        current_step += 1
        if terminated or truncated:
            break
    
    # Calculate performance metrics
    portfolio_values = np.array(portfolio_values)
    initial_value = portfolio_values[0]
    final_value = portfolio_values[-1]
    
    total_return = (final_value - initial_value) / initial_value * 100
    
    # Handle edge case where length is too short
    if len(portfolio_values) > 1:
        daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        sharpe_ratio = np.mean(daily_returns) / (np.std(daily_returns) + 1e-10) * np.sqrt(252)
        max_drawdown = np.min(portfolio_values / np.maximum.accumulate(portfolio_values) + 1e-10) - 1
    else:
        daily_returns = [0]
        sharpe_ratio = 0
        max_drawdown = 0
    
    # Buy and hold benchmark
    benchmark_initial = prices[0]
    benchmark_final = prices[-1]
    benchmark_return = (benchmark_final - benchmark_initial) / benchmark_initial * 100
    
    # Print results
    print("\nBacktest Results:")
    print(f"Initial Portfolio: ${initial_value:.2f}")
    print(f"Final Portfolio: ${final_value:.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: {max_drawdown*100:.2f}%")
    print(f"\nBenchmark (Buy & Hold {symbol}):")
    print(f"Total Return: {benchmark_return:.2f}%")
    
    # Create returns DataFrame
    returns_df = pd.DataFrame(
        daily_returns,
        index=test_data.index[1:len(portfolio_values)],
        columns=['returns']
    )
    
    # Plot results
    plt.figure(figsize=(14, 12))
    
    # Portfolio value
    plt.subplot(4, 1, 1)
    plt.plot(test_data.index[:len(portfolio_values)], portfolio_values)
    plt.title(f'{symbol} Trading Strategy Performance ({test_start} to {test_end})')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True)
    
    # Price chart
    plt.subplot(4, 1, 2)
    plt.plot(test_data.index, prices)
    plt.title(f'{symbol} Price')
    plt.ylabel('Price ($)')
    plt.grid(True)
    
    # Actions
    plt.subplot(4, 1, 3)
    plt.plot(test_data.index[:len(actions_taken)], actions_taken)
    plt.title('Trading Actions')
    plt.ylabel('Action (-1 to 1)')
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.grid(True)
    
    # Cumulative returns comparison
    plt.subplot(4, 1, 4)
    strategy_returns = (1 + np.array(daily_returns)).cumprod() - 1
    buy_hold_returns = (prices[:len(portfolio_values)] / prices[0] - 1)
    
    plt.plot(test_data.index[1:len(portfolio_values)], strategy_returns * 100, label='AI Strategy')
    plt.plot(test_data.index[:len(portfolio_values)], buy_hold_returns * 100, label=f'Buy & Hold {symbol}')
    plt.title('Cumulative Returns Comparison (%)')
    plt.ylabel('Return (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(results_dir, f'{symbol}_backtest_results.png')
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Results chart saved to {plot_path}")
    
    # Save detailed results to CSV
    results_df = pd.DataFrame({
        'Date': test_data.index[:len(portfolio_values)],
        'Price': prices[:len(portfolio_values)],
        'Portfolio_Value': portfolio_values,
        'Action': actions_taken,
        'Position': positions[:len(portfolio_values)] if len(positions) > 0 else [0] * len(portfolio_values)
    })
    
    csv_path = os.path.join(results_dir, f'{symbol}_backtest_details.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"Detailed results saved to {csv_path}")
    
    return {
        'returns': returns_df,
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'initial_value': initial_value,
        'final_value': final_value,
        'portfolio_values': portfolio_values,
        'actions': actions_taken,
        'benchmark_return': benchmark_return,
        'plot_path': plot_path,
        'csv_path': csv_path
    }

def main():
    """Main function to parse arguments and run training/backtesting."""
    parser = argparse.ArgumentParser(description="Train and backtest trading models")
    
    # General arguments
    parser.add_argument("--symbol", type=str, default="AAPL",
                       help="Stock symbol to train on (default: AAPL)")
    parser.add_argument("--data-source", type=str, default="yfinance",
                       choices=["yfinance", "synthetic"],
                       help="Source of data (default: yfinance)")
    parser.add_argument("--feature-count", type=int, default=21,
                       help="Number of features to use (default: 21)")
    
    # Training arguments
    parser.add_argument("--train", action="store_true",
                       help="Train a new model")
    parser.add_argument("--train-start", type=str, default="2020-01-01",
                       help="Start date for training data (default: 2020-01-01)")
    parser.add_argument("--train-end", type=str, default="2022-12-31",
                       help="End date for training data (default: 2022-12-31)")
    parser.add_argument("--timesteps", type=int, default=100000,
                       help="Training timesteps (default: 100000)")
    
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
                       help="Directory to store models (default: models)")
    parser.add_argument("--results-dir", type=str, default="results",
                       help="Directory to store results (default: results)")
    
    # Synthetic data parameters
    parser.add_argument("--synthetic-initial-price", type=float, default=100.0,
                       help="Initial price for synthetic data (default: 100.0)")
    parser.add_argument("--synthetic-volatility", type=float, default=0.02,
                       help="Volatility for synthetic data (default: 0.02)")
    parser.add_argument("--synthetic-drift", type=float, default=0.001,
                       help="Drift for synthetic data (default: 0.001)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Set up synthetic parameters
    synthetic_params = {
        "initial_price": args.synthetic_initial_price,
        "volatility": args.synthetic_volatility,
        "drift": args.synthetic_drift,
        "volume_min": 1000000,
        "volume_max": 5000000
    }
    
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
            synthetic_params=synthetic_params
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
            test_start=args.test_start,
            test_end=args.test_end,
            feature_count=args.feature_count,
            data_source=args.data_source,
            results_dir=args.results_dir,
            synthetic_params=synthetic_params,
            transaction_fee_percent=args.fee
        )


if __name__ == "__main__":
    main() 