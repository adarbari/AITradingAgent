#!/usr/bin/env python
"""
Example script demonstrating multi-asset portfolio optimization
using Modern Portfolio Theory.
"""
import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import DataManager
from src.agent.multi_agent import PortfolioManagementAgent, AgentInput

def run_multi_asset_optimization():
    """Run a multi-asset portfolio optimization example."""
    # Initialize the data manager
    data_manager = DataManager(market_data_source="yahoo", verbose=1)
    
    # Initialize the portfolio management agent
    portfolio_agent = PortfolioManagementAgent(data_manager=data_manager, verbose=1)
    
    # Define a set of symbols to include in the portfolio
    symbols = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'JNJ']
    
    # Define date range for historical data
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365*3)).strftime("%Y-%m-%d")  # 3 years of data
    
    print(f"\nRunning multi-asset portfolio optimization for {len(symbols)} assets...")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Symbols: {', '.join(symbols)}")
    
    # Create agent input with optimization request
    request = f"Optimize portfolio for {', '.join(symbols)} with a moderate risk tolerance."
    agent_input = AgentInput(
        request=request,
        context={
            "symbols": symbols,
            "date_range": {
                "start_date": start_date,
                "end_date": end_date
            },
            "risk_tolerance": "moderate"
        }
    )
    
    # Process the optimization request
    result = portfolio_agent.process(agent_input)
    
    # Print the result
    print("\nPortfolio Optimization Results:\n")
    print(result.response)
    
    # If we have efficient frontier data, visualize it
    if result.data and 'efficient_frontier' in result.data:
        plot_efficient_frontier(
            result.data['efficient_frontier'],
            result.data.get('volatility', 0),
            result.data.get('return', 0),
            result.data.get('risk_parity', {})
        )
    
    # Compare different risk profiles
    compare_risk_profiles(portfolio_agent, symbols, start_date, end_date)

def compare_risk_profiles(portfolio_agent, symbols, start_date, end_date):
    """Compare portfolio optimization across different risk profiles."""
    risk_profiles = ['conservative', 'moderate', 'aggressive']
    results = {}
    
    print("\nComparing Different Risk Profiles:\n")
    
    for risk in risk_profiles:
        # Create agent input with the risk profile
        request = f"Optimize portfolio for {', '.join(symbols)} with a {risk} risk tolerance."
        agent_input = AgentInput(
            request=request,
            context={
                "symbols": symbols,
                "date_range": {
                    "start_date": start_date,
                    "end_date": end_date
                },
                "risk_tolerance": risk
            }
        )
        
        # Process the optimization request
        result = portfolio_agent.process(agent_input)
        results[risk] = result.data
        
        # Print key metrics
        expected_return = result.data.get("return", 0) * 100
        volatility = result.data.get("volatility", 0) * 100
        sharpe_ratio = result.data.get("sharpe_ratio", 0)
        
        print(f"{risk.capitalize()} Portfolio:")
        print(f"  - Expected Return: {expected_return:.2f}%")
        print(f"  - Expected Volatility: {volatility:.2f}%")
        print(f"  - Sharpe Ratio: {sharpe_ratio:.2f}")
        
        # Print top 3 allocations
        weights = result.data.get("formatted_weights", {})
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"  - Top 3 Allocations:")
        for symbol, allocation in sorted_weights:
            print(f"    - {symbol}: {allocation:.2f}%")
        print()
    
    # Plot comparison of allocations
    plot_allocation_comparison(results)

def plot_efficient_frontier(frontier_data, current_volatility, current_return, risk_parity=None):
    """Plot the efficient frontier with the current portfolio marked."""
    plt.figure(figsize=(10, 6))
    
    # Plot efficient frontier
    plt.plot(frontier_data['volatilities'], frontier_data['returns'], 'b-', linewidth=2, label='Efficient Frontier')
    
    # Mark current portfolio
    plt.scatter(current_volatility, current_return, marker='o', color='red', s=100, label='Optimized Portfolio')
    
    # Mark risk parity portfolio if available
    if risk_parity:
        rp_vol = risk_parity.get('volatility', 0)
        rp_ret = risk_parity.get('return', 0)
        plt.scatter(rp_vol, rp_ret, marker='s', color='green', s=100, label='Risk Parity')
    
    plt.title('Efficient Frontier')
    plt.xlabel('Expected Volatility (Standard Deviation)')
    plt.ylabel('Expected Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_allocation_comparison(results):
    """Plot comparison of allocations across different risk profiles."""
    risk_profiles = ['conservative', 'moderate', 'aggressive']
    
    # Get unique symbols across all portfolios
    all_symbols = set()
    for risk in risk_profiles:
        weights = results[risk].get('weights', {})
        all_symbols.update(weights.keys())
    
    # Create data for plotting
    data = []
    for symbol in all_symbols:
        symbol_data = [symbol]
        for risk in risk_profiles:
            weights = results[risk].get('weights', {})
            weight = weights.get(symbol, 0) * 100  # Convert to percentage
            symbol_data.append(weight)
        data.append(symbol_data)
    
    # Sort by average allocation
    data.sort(key=lambda x: sum(x[1:]) / len(x[1:]), reverse=True)
    
    # Only show top 10 allocations
    data = data[:10]
    
    # Prepare data for plotting
    symbols = [row[0] for row in data]
    conservative = [row[1] for row in data]
    moderate = [row[2] for row in data]
    aggressive = [row[3] for row in data]
    
    # Create plot
    x = np.arange(len(symbols))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(x - width, conservative, width, label='Conservative')
    ax.bar(x, moderate, width, label='Moderate')
    ax.bar(x + width, aggressive, width, label='Aggressive')
    
    ax.set_ylabel('Allocation (%)')
    ax.set_title('Portfolio Allocation Comparison by Risk Profile')
    ax.set_xticks(x)
    ax.set_xticklabels(symbols)
    ax.legend()
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_multi_asset_optimization() 