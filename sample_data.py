#!/usr/bin/env python3
"""
Script to generate synthetic stock data for training and backtesting.
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def generate_synthetic_stock_data(symbol, start_date, end_date, initial_price=100, volatility=0.01, trend=0.0001):
    """
    Generate synthetic stock data with a random walk model.
    
    Args:
        symbol (str): Stock symbol
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        initial_price (float): Initial stock price
        volatility (float): Daily volatility (standard deviation)
        trend (float): Daily trend (drift)
    
    Returns:
        pd.DataFrame: Synthetic stock data
    """
    # Convert dates to datetime
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Generate date range
    date_range = pd.date_range(start=start, end=end, freq='B')  # Business days
    
    # Generate random returns
    np.random.seed(42)  # For reproducibility
    returns = np.random.normal(trend, volatility, size=len(date_range))
    
    # Cumulative returns
    cumulative_returns = np.exp(np.cumsum(returns))
    
    # Calculate prices
    prices = initial_price * cumulative_returns
    
    # Generate daily data
    data = pd.DataFrame(index=date_range)
    data['Open'] = prices * (1 - 0.002 * np.random.randn(len(date_range)))
    data['High'] = np.maximum(prices * (1 + 0.004 * np.random.rand(len(date_range))), data['Open'])
    data['Low'] = np.minimum(prices * (1 - 0.004 * np.random.rand(len(date_range))), data['Open'])
    data['Close'] = prices
    data['Adj Close'] = prices
    data['Volume'] = np.random.randint(100000, 1000000, size=len(date_range))
    
    # Add some patterns and seasonality
    # Only add the crash and recovery if there are enough data points
    if len(date_range) > 100:  # Ensure we have enough data points
        # Calculate indices with proper bounds checking
        crash_start = len(date_range) // 3
        crash_length = min(20, (len(date_range) - crash_start) // 2)
        crash_end = crash_start + crash_length
        
        recovery_length = min(40, len(date_range) - crash_end - 1)
        recovery_end = crash_end + recovery_length
        
        # Safety check
        if crash_end <= len(date_range) and recovery_end <= len(date_range):
            # Create factor arrays
            crash_factor = np.linspace(1.0, 0.7, crash_end - crash_start)
            recovery_factor = np.linspace(0.7, 1.1, recovery_end - crash_end)
            
            # Apply factors
            for i in range(crash_start, crash_end):
                data.iloc[i, :5] *= crash_factor[i - crash_start]
            
            for i in range(crash_end, recovery_end):
                data.iloc[i, :5] *= recovery_factor[i - crash_end]
    
    return data

def generate_multiple_stocks(symbols, start_date, end_date, correlation=0.7):
    """
    Generate synthetic data for multiple stocks with correlation.
    
    Args:
        symbols (list): List of stock symbols
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        correlation (float): Correlation between stocks
    
    Returns:
        dict: Dictionary of DataFrames for each symbol
    """
    # Generate the first stock as a reference
    reference_stock = generate_synthetic_stock_data(
        symbols[0], start_date, end_date, 
        initial_price=np.random.randint(50, 200)
    )
    
    stock_data = {symbols[0]: reference_stock}
    
    # Generate correlated stocks
    for symbol in symbols[1:]:
        # Get reference returns
        ref_returns = reference_stock['Close'].pct_change().dropna().values
        
        # Create correlated returns
        np.random.seed(hash(symbol) % 2**32)  # Different seed for each stock
        
        # Start with a different initial price
        initial_price = np.random.randint(50, 200)
        
        # Create correlated stock by mixing reference returns with random returns
        correlated_stock = generate_synthetic_stock_data(
            symbol, start_date, end_date, 
            initial_price=initial_price,
            volatility=0.01 + 0.005 * np.random.rand()  # Slightly different volatility
        )
        
        stock_data[symbol] = correlated_stock
    
    return stock_data

def generate_market_index(stock_data, index_name="^IXIC", weights=None):
    """
    Generate a market index based on the given stocks.
    
    Args:
        stock_data (dict): Dictionary of stock DataFrames
        index_name (str): Name of the index
        weights (list, optional): List of weights for each stock
    
    Returns:
        pd.DataFrame: Market index data
    """
    # Extract all unique dates from all stocks
    all_dates = set()
    for df in stock_data.values():
        all_dates.update(df.index)
    
    all_dates = sorted(all_dates)
    
    # Create an empty index DataFrame
    index_data = pd.DataFrame(index=all_dates)
    
    # If weights not provided, use equal weights
    if weights is None:
        weights = [1.0 / len(stock_data)] * len(stock_data)
    
    # Normalize weights to sum to 1
    weights = np.array(weights) / sum(weights)
    
    # Compute the weighted average of all stock prices
    close_prices = pd.DataFrame(index=all_dates)
    
    for i, (symbol, df) in enumerate(stock_data.items()):
        close_prices[symbol] = df['Close']
    
    # Forward fill missing values
    close_prices.ffill(inplace=True)
    
    # Calculate weighted index
    weighted_sum = close_prices.dot(weights)
    
    # Scale to start at 10,000
    initial_value = weighted_sum.iloc[0]
    scaled_index = (weighted_sum / initial_value) * 10000
    
    # Create the index DataFrame
    index_data['Open'] = scaled_index * (1 - 0.001 * np.random.randn(len(all_dates)))
    index_data['High'] = np.maximum(scaled_index * (1 + 0.002 * np.random.rand(len(all_dates))), index_data['Open'])
    index_data['Low'] = np.minimum(scaled_index * (1 - 0.002 * np.random.rand(len(all_dates))), index_data['Open'])
    index_data['Close'] = scaled_index
    index_data['Adj Close'] = scaled_index
    index_data['Volume'] = np.random.randint(1000000, 10000000, size=len(all_dates))
    
    return index_data

def add_technical_indicators(data):
    """
    Add basic technical indicators to a DataFrame.
    
    Args:
        data (pd.DataFrame): OHLCV data
    
    Returns:
        pd.DataFrame: Data with technical indicators
    """
    df = data.copy()
    
    # Moving averages
    df['sma_5'] = df['Close'].rolling(window=5).mean()
    df['sma_20'] = df['Close'].rolling(window=20).mean()
    df['sma_50'] = df['Close'].rolling(window=50).mean()
    df['ema_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    # MACD
    df['MACD_12_26_9'] = df['ema_12'] - df['ema_26']
    df['MACDs_12_26_9'] = df['MACD_12_26_9'].ewm(span=9, adjust=False).mean()
    df['MACDh_12_26_9'] = df['MACD_12_26_9'] - df['MACDs_12_26_9']
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BBM_5_2.0'] = df['Close'].rolling(window=5).mean()
    df['BBstd_5_2.0'] = df['Close'].rolling(window=5).std()
    df['BBU_5_2.0'] = df['BBM_5_2.0'] + 2 * df['BBstd_5_2.0']
    df['BBL_5_2.0'] = df['BBM_5_2.0'] - 2 * df['BBstd_5_2.0']
    df['BBB_5_2.0'] = df['BBU_5_2.0'] - df['BBL_5_2.0']
    
    # Replace NaN values with forward fill then backward fill
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    
    return df

def save_data_to_csv(data_dict, output_dir='data'):
    """
    Save data to CSV files.
    
    Args:
        data_dict (dict): Dictionary of DataFrames
        output_dir (str): Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for symbol, df in data_dict.items():
        filename = f"{output_dir}/{symbol}.csv"
        df.to_csv(filename)
        print(f"Saved {symbol} data to {filename}")

def plot_data(data_dict, output_dir='plots'):
    """
    Plot stock data.
    
    Args:
        data_dict (dict): Dictionary of DataFrames
        output_dir (str): Output directory for plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot all stocks on one chart
    plt.figure(figsize=(12, 6))
    
    for symbol, df in data_dict.items():
        # Normalize to percentage change from start
        normalized = (df['Close'] / df['Close'].iloc[0] - 1) * 100
        plt.plot(df.index, normalized, label=symbol)
    
    plt.title('Percentage Change from Start')
    plt.xlabel('Date')
    plt.ylabel('Percent Change (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig(f"{output_dir}/all_stocks.png")
    plt.close()
    
    # Plot each stock individually with volume
    for symbol, df in data_dict.items():
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # Price plot
        ax1.plot(df.index, df['Close'], label='Close Price')
        ax1.plot(df.index, df['sma_20'], label='SMA (20)', alpha=0.7)
        ax1.plot(df.index, df['ema_26'], label='EMA (26)', alpha=0.7)
        ax1.fill_between(df.index, df['BBL_5_2.0'], df['BBU_5_2.0'], color='gray', alpha=0.2, label='Bollinger Bands')
        
        ax1.set_title(f'{symbol} Stock Price')
        ax1.set_ylabel('Price')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        
        # Volume plot
        ax2.bar(df.index, df['Volume'], color='blue', alpha=0.5)
        ax2.set_title('Volume')
        ax2.set_ylabel('Volume')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{symbol}_price.png")
        plt.close()
    
    # Plot technical indicators for the first stock
    symbol = list(data_dict.keys())[0]
    df = data_dict[symbol]
    
    # RSI plot
    plt.figure(figsize=(12, 4))
    plt.plot(df.index, df['rsi_14'], color='purple')
    plt.axhline(y=70, color='red', linestyle='--', alpha=0.5)
    plt.axhline(y=30, color='green', linestyle='--', alpha=0.5)
    plt.title(f'{symbol} RSI (14)')
    plt.ylabel('RSI')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/{symbol}_rsi.png")
    plt.close()
    
    # MACD plot
    plt.figure(figsize=(12, 4))
    plt.plot(df.index, df['MACD_12_26_9'], label='MACD', color='blue')
    plt.plot(df.index, df['MACDs_12_26_9'], label='Signal', color='red')
    plt.bar(df.index, df['MACDh_12_26_9'], label='Histogram', color='gray', alpha=0.5)
    plt.title(f'{symbol} MACD (12, 26, 9)')
    plt.ylabel('MACD')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(f"{output_dir}/{symbol}_macd.png")
    plt.close()

def main():
    """
    Main function to generate and save synthetic stock data.
    """
    # Create directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Define date ranges
    train_start_date = "2020-01-01"
    train_end_date = "2023-12-31"
    test_start_date = "2024-01-01"
    test_end_date = "2024-03-22"  # Current date as of writing
    
    # Generate training data
    print(f"Generating training data from {train_start_date} to {train_end_date}...")
    training_stocks = generate_multiple_stocks(
        symbols=['AAPL', 'AMZN'], 
        start_date=train_start_date, 
        end_date=train_end_date
    )
    
    # Add technical indicators
    for symbol, df in training_stocks.items():
        training_stocks[symbol] = add_technical_indicators(df)
    
    # Generate NASDAQ index for training period
    training_nasdaq = generate_market_index(training_stocks, index_name="^IXIC")
    training_nasdaq = add_technical_indicators(training_nasdaq)
    
    # Add the index to the data dictionary
    training_stocks['^IXIC'] = training_nasdaq
    
    # Generate test data
    print(f"Generating test data from {test_start_date} to {test_end_date}...")
    test_stocks = generate_multiple_stocks(
        symbols=['AAPL', 'AMZN'], 
        start_date=test_start_date, 
        end_date=test_end_date
    )
    
    # Add technical indicators
    for symbol, df in test_stocks.items():
        test_stocks[symbol] = add_technical_indicators(df)
    
    # Generate NASDAQ index for test period
    test_nasdaq = generate_market_index(test_stocks, index_name="^IXIC")
    test_nasdaq = add_technical_indicators(test_nasdaq)
    
    # Add the index to the data dictionary
    test_stocks['^IXIC'] = test_nasdaq
    
    # Save data to CSV files
    print("Saving training data...")
    save_data_to_csv(training_stocks, output_dir='data/train')
    
    print("Saving test data...")
    save_data_to_csv(test_stocks, output_dir='data/test')
    
    # Plot data
    print("Plotting data...")
    plot_data(training_stocks, output_dir='plots/train')
    plot_data(test_stocks, output_dir='plots/test')
    
    print("Data generation complete!")

if __name__ == "__main__":
    main() 