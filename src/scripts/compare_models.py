#!/usr/bin/env python3
"""
Script to compare different model configurations and find the best one for a given stock.
"""
import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.scripts.train_and_backtest import train_model, backtest_model

def evaluate_model_configurations(symbol, configurations, test_period, results_dir="results/comparison"):
    """
    Evaluate multiple model configurations and compare their performance.
    
    Args:
        symbol (str): Stock symbol to train on
        configurations (list): List of configuration dictionaries
        test_period (tuple): (test_start, test_end) dates
        results_dir (str): Directory to save results
        
    Returns:
        pd.DataFrame: Performance summary of all models
    """
    os.makedirs(results_dir, exist_ok=True)
    
    # Create a dataframe to store results
    results_summary = pd.DataFrame(columns=[
        'Configuration', 'Training Period', 'Total Timesteps', 
        'Total Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)'
    ])
    
    # Evaluate each configuration
    for i, config in enumerate(configurations):
        print(f"\n=== Testing Configuration {i+1}/{len(configurations)} ===")
        print(f"Training period: {config['train_start']} to {config['train_end']}")
        print(f"Timesteps: {config['timesteps']}")
        
        # Create a model directory
        model_name = f"{symbol}_{i+1}_model"
        model_path = os.path.join(results_dir, model_name)
        
        # Train the model
        model, model_path = train_model(
            data_fetcher_type=config.get('data_source', 'yahoo'),
            symbol=symbol,
            start_date=config['train_start'],
            end_date=config['train_end'],
            model_path=model_path,
            timesteps=config['timesteps']
        )
        
        if model is None:
            print(f"Failed to train model for configuration {i+1}")
            continue
        
        # Backtest the model
        test_start, test_end = test_period
        results = backtest_model(
            model_path=model_path,
            symbol=symbol,
            start_date=test_start,
            end_date=test_end,
            data_source=config.get('data_source', 'yahoo'),
            results_dir=os.path.join(results_dir, f"{symbol}_{i+1}")
        )
        
        if results is None:
            print(f"Failed to backtest model for configuration {i+1}")
            continue
        
        # Add to summary
        results_summary.loc[i] = [
            f"Config {i+1}",
            f"{config['train_start']} to {config['train_end']}",
            config['timesteps'],
            results['total_return'],
            results['sharpe_ratio'],
            results['max_drawdown']
        ]
    
    # Save summary to CSV
    summary_file = os.path.join(results_dir, f"{symbol}_model_comparison.csv")
    results_summary.to_csv(summary_file, index=False)
    print(f"\nComparison results saved to {summary_file}")
    
    # Plot comparison
    plot_path = plot_model_comparison(results_summary, symbol, results_dir)
    print(f"Comparison plot saved to {plot_path}")
    
    return results_summary


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
            'timesteps': 100000
        },
        {
            'train_start': f"{current_year-2}-01-01", 
            'train_end': f"{current_year-1}-12-31",
            'timesteps': 100000
        },
        {
            'train_start': f"{current_year-2}-01-01",
            'train_end': f"{current_year-1}-06-30",
            'timesteps': 100000
        },
        
        # Different training durations (same end date)
        {
            'train_start': f"{current_year-2}-01-01",
            'train_end': f"{current_year-1}-12-31",
            'timesteps': 50000
        },
        {
            'train_start': f"{current_year-2}-01-01",
            'train_end': f"{current_year-1}-12-31",
            'timesteps': 200000
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
    test_period = (args.test_start, args.test_end)
    
    # Get default configurations
    configurations = define_default_configurations()
    
    print(f"\nComparing model configurations for {args.symbol}")
    print(f"Test period: {test_period[0]} to {test_period[1]}")
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