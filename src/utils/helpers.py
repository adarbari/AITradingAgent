"""
Utility functions for the trading agent.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


def create_directories(*dirs):
    """
    Create directories if they don't exist.
    
    Args:
        *dirs: Variable number of directory paths
    """
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)


def calculate_returns(initial_portfolio_value, final_portfolio_value):
    """
    Calculate percentage returns.
    
    Args:
        initial_portfolio_value (float): Initial portfolio value
        final_portfolio_value (float): Final portfolio value
        
    Returns:
        float: Percentage return
    """
    return ((final_portfolio_value - initial_portfolio_value) / initial_portfolio_value) * 100


def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """
    Calculate the Sharpe ratio.
    
    Args:
        returns (list or numpy.array): Daily returns
        risk_free_rate (float, optional): Risk-free rate. Default is 0.0.
        
    Returns:
        float: Sharpe ratio
    """
    returns_array = np.array(returns)
    excess_returns = returns_array - risk_free_rate
    
    if np.std(excess_returns) == 0:
        return 0
    
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)  # Annualized


def calculate_drawdown(portfolio_values):
    """
    Calculate maximum drawdown.
    
    Args:
        portfolio_values (list or numpy.array): Portfolio values over time
        
    Returns:
        float: Maximum drawdown percentage
    """
    portfolio_values = np.array(portfolio_values)
    
    # Running maximum
    running_max = np.maximum.accumulate(portfolio_values)
    
    # Drawdown in percentage terms
    drawdown = (portfolio_values - running_max) / running_max * 100
    
    return np.min(drawdown)


def plot_portfolio_performance(dates, portfolio_values, benchmark_values=None, save_path=None):
    """
    Plot portfolio performance over time.
    
    Args:
        dates (list): List of dates
        portfolio_values (list): List of portfolio values
        benchmark_values (list, optional): List of benchmark values for comparison
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    plt.plot(dates, portfolio_values, label='Portfolio')
    
    if benchmark_values is not None:
        plt.plot(dates, benchmark_values, label='Benchmark', linestyle='--')
    
    plt.title('Portfolio Performance')
    plt.xlabel('Date')
    plt.ylabel('Value ($)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.tight_layout()
    plt.show()


def plot_trade_history(dates, actions, prices, save_path=None):
    """
    Plot trade history with buy/sell signals.
    
    Args:
        dates (list): List of dates
        actions (list): List of actions (buy/sell/hold)
        prices (list): List of prices
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    plt.plot(dates, prices, label='Price')
    
    buy_dates = [dates[i] for i, action in enumerate(actions) if action == 'buy']
    buy_prices = [prices[i] for i, action in enumerate(actions) if action == 'buy']
    
    sell_dates = [dates[i] for i, action in enumerate(actions) if action == 'sell']
    sell_prices = [prices[i] for i, action in enumerate(actions) if action == 'sell']
    
    plt.scatter(buy_dates, buy_prices, color='green', marker='^', s=100, label='Buy')
    plt.scatter(sell_dates, sell_prices, color='red', marker='v', s=100, label='Sell')
    
    plt.title('Trade History')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.tight_layout()
    plt.show()


def save_performance_metrics(metrics, file_path):
    """
    Save performance metrics to a file.
    
    Args:
        metrics (dict): Dictionary of performance metrics
        file_path (str): Path to save the metrics
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Format metrics
    formatted_metrics = [f"Timestamp: {timestamp}"]
    for key, value in metrics.items():
        if isinstance(value, float):
            formatted_metrics.append(f"{key}: {value:.2f}")
        else:
            formatted_metrics.append(f"{key}: {value}")
    
    # Write to file
    with open(file_path, 'a') as f:
        f.write('\n' + '\n'.join(formatted_metrics) + '\n' + '-'*50 + '\n') 