#!/usr/bin/env python3
"""
Script to compare different model configurations and find the best one for a given stock.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import argparse
from datetime import datetime, timedelta
from copy import deepcopy

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.scripts.train_and_backtest import backtest_model
from src.train.trainer import TrainingManager
from src.backtest import Backtester
from src.agent.trading_env import TradingEnvironment
from src.data.yahoo_data_fetcher import YahooDataFetcher

def evaluate_model_configurations(symbol, configurations, test_period, results_dir="results/comparison"):
    """
    Train and evaluate multiple model configurations.
    
    Args:
        symbol (str): Stock symbol
        configurations (list): List of configuration dictionaries
        test_period (dict): Dict with 'start' and 'end' keys for test period
        results_dir (str): Directory to save results
        
    Returns:
        pd.DataFrame: Results dataframe
    """
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    results = []
    
    for i, config in enumerate(configurations):
        print(f"\n========== Configuration {i+1}/{len(configurations)} ==========")
        print(json.dumps(config, indent=2))
        
        # Define a unique name for this configuration
        config_name = f"Config_{i+1}"
        
        # Train model with this configuration
        model_path = f"{results_dir}/{symbol}_{config_name}.zip"
        
        # Use the training manager
        trainer = TrainingManager()
        trainer.train(
            symbol=symbol,
            start_date=config['train_start'],
            end_date=config['train_end'],
            model_output_path=model_path,
            feature_count=config.get('feature_count', 21),
            data_source=config.get('data_source', 'yahoo'),
            timesteps=config.get('timesteps', 100000),
            force_train=True
        )
        
        # Backtest the model
        backtest_results = backtest_model(
            model_path=model_path,
            symbol=symbol,
            test_start=test_period['start'],
            test_end=test_period['end'],
            data_source=config.get('data_source', 'yahoo')
        )
        
        # Store results
        results.append({
            'Configuration': config_name,
            'Train Period': f"{config['train_start']} to {config['train_end']}",
            'Timesteps': config['timesteps'],
            'Total Return (%)': backtest_results['total_return'],
            'Sharpe Ratio': backtest_results['sharpe_ratio'],
            'Max Drawdown (%)': backtest_results['max_drawdown'],
            'Win Rate (%)': backtest_results['win_rate'],
            'N Trades': backtest_results['n_trades']
        })
        
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    csv_path = os.path.join(results_dir, f"{symbol}_comparison_results.csv")
    results_df.to_csv(csv_path, index=False)
    
    return results_df


def plot_model_comparison(results_df, symbol, results_dir):
    """
    Create a visual comparison of model performances.
    
    Args:
        results_df (pd.DataFrame): Performance summary dataframe
        symbol (str): Stock symbol
        results_dir (str): Directory to save the plot
        
    Returns:
        str: Path to the saved plot
    """
    plt.figure(figsize=(12, 8))
    
    # Plot total return vs sharpe ratio
    plt.subplot(2, 1, 1)
    plt.scatter(results_df['Total Return (%)'], results_df['Sharpe Ratio'], 
                s=100, alpha=0.7, c=range(len(results_df)), cmap='viridis')
    
    # Add labels to points
    for i, row in results_df.iterrows():
        plt.annotate(row['Configuration'], 
                    (row['Total Return (%)'], row['Sharpe Ratio']),
                    xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Total Return (%)')
    plt.ylabel('Sharpe Ratio')
    plt.title(f'Performance Comparison of {symbol} Models - Return vs Sharpe')
    plt.grid(True, alpha=0.3)
    
    # Plot total return vs max drawdown
    plt.subplot(2, 1, 2)
    plt.scatter(results_df['Total Return (%)'], results_df['Max Drawdown (%)'],
                s=100, alpha=0.7, c=range(len(results_df)), cmap='viridis')
    
    # Add labels to points
    for i, row in results_df.iterrows():
        plt.annotate(row['Configuration'], 
                    (row['Total Return (%)'], row['Max Drawdown (%)']),
                    xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Total Return (%)')
    plt.ylabel('Max Drawdown (%)')
    plt.title(f'Performance Comparison of {symbol} Models - Return vs Drawdown')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(results_dir, f"{symbol}_model_comparison.png")
    plt.savefig(plot_path)
    
    return plot_path


def define_default_configurations():
    """Define a set of default configurations to test."""
    current_year = datetime.now().year
    
    return [
        # Different training periods
        {
            'train_start': f"{current_year-3}-01-01",
            'train_end': f"{current_year-1}-12-31",
            'timesteps': 100000,
            'data_source': 'yahoo',
            'feature_count': 21
        },
        {
            'train_start': f"{current_year-2}-01-01", 
            'train_end': f"{current_year-1}-12-31",
            'timesteps': 100000,
            'data_source': 'yahoo',
            'feature_count': 21
        },
        {
            'train_start': f"{current_year-2}-01-01",
            'train_end': f"{current_year-1}-06-30",
            'timesteps': 100000,
            'data_source': 'yahoo',
            'feature_count': 21
        },
        
        # Different training durations (same end date)
        {
            'train_start': f"{current_year-2}-01-01",
            'train_end': f"{current_year-1}-12-31",
            'timesteps': 50000,
            'data_source': 'yahoo',
            'feature_count': 21
        },
        {
            'train_start': f"{current_year-2}-01-01",
            'train_end': f"{current_year-1}-12-31",
            'timesteps': 200000,
            'data_source': 'yahoo',
            'feature_count': 21
        },
    ]


def main():
    """Main function to compare model configurations."""
    parser = argparse.ArgumentParser(description="Compare different trading model configurations")
    
    # General arguments
    parser.add_argument("--symbol", type=str, default="AMZN",
                       help="Stock symbol to train on (default: AMZN)")
    
    # Test period arguments
    parser.add_argument("--test-start", type=str, default="2024-01-01",
                       help="Start date for testing data (default: beginning of current year)")
    parser.add_argument("--test-end", type=str, 
                       default=datetime.now().strftime("%Y-%m-%d"),
                       help="End date for testing data (default: current date)")
    
    # Directory arguments
    parser.add_argument("--results-dir", type=str, default="results/comparison",
                       help="Directory to store results (default: results/comparison)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create test period tuple
    test_period = {'start': args.test_start, 'end': args.test_end}
    
    # Get default configurations
    configurations = define_default_configurations()
    
    print(f"\nComparing model configurations for {args.symbol}")
    print(f"Test period: {test_period['start']} to {test_period['end']}")
    print(f"Number of configurations to evaluate: {len(configurations)}")
    
    # Evaluate configurations
    results = evaluate_model_configurations(
        symbol=args.symbol,
        configurations=configurations,
        test_period=test_period,
        results_dir=args.results_dir
    )
    
    # Display the results
    print("\nModel Configurations Performance Summary:")
    print(results.to_string(index=False))
    
    # Identify the best models
    best_return_model = results.loc[results['Total Return (%)'].idxmax()]
    best_sharpe_model = results.loc[results['Sharpe Ratio'].idxmax()]
    best_drawdown_model = results.loc[results['Max Drawdown (%)'].idxmax()]
    
    print("\nBest model by Total Return:", best_return_model['Configuration'])
    print("Best model by Sharpe Ratio:", best_sharpe_model['Configuration'])
    print("Best model by Max Drawdown:", best_drawdown_model['Configuration'])


if __name__ == "__main__":
    main() 