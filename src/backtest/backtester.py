"""
Backtesting module for evaluating trading strategy performance.
"""
import os
import traceback
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import joblib
from stable_baselines3 import PPO
import pandas_datareader.data as web
import gymnasium as gym
import yfinance as yf

from .base_backtester import BaseBacktester
from .benchmarks import BenchmarkFactory
from ..agent.trading_env import TradingEnvironment
from ..utils.helpers import (
    calculate_returns, 
    calculate_sharpe_ratio, 
    calculate_drawdown,
    plot_portfolio_performance,
    plot_trade_history,
    save_performance_metrics
)
from src.data import DataFetcherFactory


class Backtester(BaseBacktester):
    """
    Backtester for evaluating trading strategies.
    """
    
    def __init__(self, results_dir='results'):
        """
        Initialize the backtester.
        
        Args:
            results_dir (str): Directory to store backtest results
        """
        super().__init__(results_dir)
        
        # Define default benchmarks
        self.benchmarks = [
            "buy_and_hold",  # Buy and hold the stock
            "sp500",        # S&P 500 index
            "nasdaq"        # NASDAQ index
        ]
    
    def run_benchmarks(self, test_data, symbol, initial_balance=10000):
        """
        Run benchmark comparisons.
        
        Args:
            test_data (pd.DataFrame): Test data with OHLCV columns
            symbol (str): Symbol to run benchmarks for
            initial_balance (float): Initial balance for benchmarks
            
        Returns:
            dict: Dictionary containing benchmark results
        """
        benchmark_results = {}
        
        # Get start and end dates from test data
        start_date = test_data.index[0].strftime('%Y-%m-%d')
        end_date = test_data.index[-1].strftime('%Y-%m-%d')
        
        # Run each benchmark
        for benchmark_type in self.benchmarks:
            try:
                # Create benchmark instance
                benchmark = BenchmarkFactory.create_benchmark(benchmark_type, symbol, initial_balance)
                
                # Calculate returns
                returns_df = benchmark.calculate_returns(start_date, end_date)
                
                if returns_df.empty:
                    print(f"Warning: Empty returns DataFrame for {benchmark_type} benchmark")
                    continue
                
                # Calculate metrics
                portfolio_values = returns_df['value'].values
                daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
                
                total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
                sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)  # Annualized
                max_drawdown = np.min(portfolio_values / np.maximum.accumulate(portfolio_values)) - 1
                
                # Store results
                benchmark_results[benchmark_type] = {
                    'name': benchmark.name,  # Include the benchmark name
                    'returns_df': pd.DataFrame(
                        daily_returns,
                        index=test_data.index[1:len(portfolio_values)],
                        columns=['returns']
                    ),
                    'total_return': total_return,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown
                }
            except Exception as e:
                print(f"Error calculating {benchmark_type} benchmark: {e}")
                continue
        
        return benchmark_results

    def backtest_model(self, model_path, symbol, test_start, test_end, data_source="yfinance", env_class=None):
        """
        Backtest a trained model on historical data.
        
        Args:
            model_path (str): Path to the saved model
            symbol (str): Stock symbol to backtest on
            test_start (str): Start date for testing (YYYY-MM-DD)
            test_end (str): End date for testing (YYYY-MM-DD)
            data_source (str): Source of data ("yfinance" or "synthetic")
            env_class (class): Environment class to use (must be a gym.Env subclass)
            
        Returns:
            dict: Dictionary containing backtest results
        """
        print(f"Backtesting model {model_path} on {symbol} from {test_start} to {test_end}")
        
        # Get test data
        if data_source == "yfinance":
            ticker = yf.Ticker(symbol)
            test_data = ticker.history(start=test_start, end=test_end)
        else:
            # Generate synthetic data for testing
            days = pd.date_range(start=test_start, end=test_end, freq='D')
            test_data = pd.DataFrame(index=days)
            test_data['Close'] = np.linspace(100, 150, len(days))  # Simple linear price trend
            test_data['Open'] = test_data['Close'] - 1
            test_data['High'] = test_data['Close'] + 2
            test_data['Low'] = test_data['Close'] - 2
            test_data['Volume'] = np.random.randint(1000000, 2000000, size=len(days))
            print(f"Generated {len(days)} days of synthetic data for {symbol}")
        
        # Prepare data for the agent
        prices = test_data['Close'].values
        features = self.prepare_data_for_agent(test_data)
        
        # Create environment
        env = env_class(
            prices=prices,
            features=features,
            initial_balance=10000,
            transaction_fee_percent=0.001
        )
        
        # Load the trained model
        model = PPO.load(model_path)
        
        # Initialize tracking variables
        portfolio_values = []
        actions_taken = []
        current_step = 0
        
        # Reset the environment
        obs, _ = env.reset()
        
        # Run the backtest
        while current_step < len(prices) - 1:
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            
            # Take step in environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Record portfolio value and action
            portfolio_values.append(info['portfolio_value'])
            actions_taken.append(info['actual_action'])
            
            current_step += 1
            if terminated or truncated:
                break
        
        # Calculate daily returns
        portfolio_values = np.array(portfolio_values)
        daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Create returns DataFrame
        returns_df = pd.DataFrame(
            daily_returns,
            index=test_data.index[1:len(portfolio_values)],
            columns=['returns']
        )
        
        # Calculate performance metrics
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)  # Annualized
        max_drawdown = np.min(portfolio_values / np.maximum.accumulate(portfolio_values)) - 1
        
        # Run benchmark comparisons
        benchmark_results = self.run_benchmarks(test_data, symbol, initial_balance=10000)
        
        # Save results
        results = {
            'returns': returns_df,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'benchmark_results': benchmark_results,
            'portfolio_values': portfolio_values,
            'actions': actions_taken
        }
        
        # Save results to file
        results_file = os.path.join(self.results_dir, f"{symbol}_backtest_results.json")
        self.save_results(results, results_file)
        
        # Plot comparison
        self.plot_comparison(symbol, results)
        
        return results
    
    def plot_comparison(self, symbol, results):
        """Plot performance comparison between strategy and benchmarks"""
        plt.figure(figsize=(12, 6))

        # Normalize portfolio returns
        normalized_portfolio = results['returns']['returns'] + 1
        plt.plot(normalized_portfolio.index, normalized_portfolio, label='Trading Strategy')

        # Plot each benchmark
        for name, benchmark in results['benchmark_results'].items():
            benchmark_df = benchmark['returns_df']
            normalized_benchmark = benchmark_df['returns'] + 1
            plt.plot(normalized_benchmark.index, normalized_benchmark, label=name)

        plt.title(f'Performance Comparison - {symbol}')
        plt.xlabel('Date')
        plt.ylabel('Normalized Portfolio Value')
        plt.legend()
        plt.grid(True)

        # Save plot
        plt.savefig(os.path.join(self.results_dir, f'{symbol}_market_comparison.png'))
        plt.close()

    def save_results(self, results, file_path):
        """Save backtest results to file"""
        # Create a copy of results to modify
        if 'portfolio_values' in results:
            # Format from backtest_model
            json_results = {
                'final_value': float(results['portfolio_values'][-1]),
                'initial_value': float(results['portfolio_values'][0]),
                'total_return': float(results['total_return']),
                'sharpe_ratio': float(results['sharpe_ratio']),
                'max_drawdown': float(results['max_drawdown']),
                'benchmark_results': {}
            }
        else:
            # Format from test_save_and_load_results
            json_results = {
                'final_value': float(results['final_value']),
                'initial_value': float(results['initial_value']),
                'total_return': float(results['total_return']),
                'sharpe_ratio': float(results['sharpe_ratio']),
                'max_drawdown': float(results['max_drawdown']),
                'benchmark_results': {}
            }

        # Save returns DataFrame separately
        returns_csv_path = file_path.replace('.json', '_returns.csv')
        if isinstance(results['returns'], list):
            # Convert list to DataFrame if needed
            returns_df = pd.DataFrame({'returns': results['returns']})
        else:
            returns_df = results['returns']
        returns_df.to_csv(returns_csv_path)

        # Convert benchmark DataFrames to CSV
        for name, benchmark in results['benchmark_results'].items():
            benchmark_csv_path = os.path.join(os.path.dirname(file_path), f'benchmark_{name}_returns.csv')
            benchmark['returns_df'].to_csv(benchmark_csv_path)
            
            # Store benchmark metrics
            json_results['benchmark_results'][name] = {
                'name': benchmark['name'],  # Use the actual benchmark name
                'total_return': float(benchmark['total_return']),
                'sharpe_ratio': float(benchmark['sharpe_ratio']),
                'max_drawdown': float(benchmark['max_drawdown']),
                'returns_csv_path': benchmark_csv_path
            }

        # Save JSON results
        with open(file_path, 'w') as f:
            json.dump(json_results, f)

        print(f"Returns data saved to {returns_csv_path}")
        print(f"Results saved to {file_path}")

    def load_results(self, file_path):
        """Load backtest results from file"""
        # Load JSON results
        with open(file_path, 'r') as f:
            json_results = json.load(f)

        # Load returns DataFrame
        returns_csv_path = file_path.replace('.json', '_returns.csv')
        returns_df = pd.read_csv(returns_csv_path, index_col=0, parse_dates=True)

        # Create results dictionary
        results = {
            'returns': returns_df,
            'final_value': json_results['final_value'],
            'initial_value': json_results['initial_value'],
            'total_return': json_results['total_return'],
            'sharpe_ratio': json_results['sharpe_ratio'],
            'max_drawdown': json_results['max_drawdown'],
            'benchmark_results': {},
            # Reconstruct portfolio_values and actions for backwards compatibility
            'portfolio_values': [json_results['initial_value'], json_results['final_value']],
            'actions': []  # Empty list as we don't save actions
        }

        # Load benchmark results
        for name, benchmark in json_results['benchmark_results'].items():
            benchmark_df = pd.read_csv(benchmark['returns_csv_path'], index_col=0, parse_dates=True)
            results['benchmark_results'][name] = {
                'name': benchmark['name'],  # Use the name from the loaded JSON
                'returns_df': benchmark_df,
                'total_return': benchmark['total_return'],
                'sharpe_ratio': benchmark['sharpe_ratio'],
                'max_drawdown': benchmark['max_drawdown']
            }

        return results

    def prepare_data_for_agent(self, data):
        """
        Prepare data for the trading agent.
        
        Args:
            data (pd.DataFrame): Raw price data with OHLCV columns
            
        Returns:
            np.array: Processed features for the agent
        """
        # Calculate basic technical indicators
        features = []
        
        # Price changes
        close_prices = data['Close'].values
        features.append(np.diff(close_prices, prepend=close_prices[0]) / close_prices)  # Returns
        
        # Volatility (rolling std of returns)
        returns = np.diff(close_prices, prepend=close_prices[0]) / close_prices
        vol = pd.Series(returns).rolling(window=5).std().fillna(0).values
        features.append(vol)
        
        # Volume changes
        volume = data['Volume'].values
        features.append(np.diff(volume, prepend=volume[0]) / volume)
        
        # Price momentum
        momentum = pd.Series(close_prices).pct_change(periods=5).fillna(0).values
        features.append(momentum)
        
        # High-Low range
        high_low_range = (data['High'].values - data['Low'].values) / data['Close'].values
        features.append(high_low_range)
        
        # Stack features into a 2D array
        features = np.stack(features, axis=1)
        
        return features 