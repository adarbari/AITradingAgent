"""
Synthetic data fetcher for generating test data
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from .base_data_fetcher import BaseDataFetcher

class SyntheticDataFetcher(BaseDataFetcher):
    """Class for generating synthetic stock data for testing"""
    
    def __init__(self):
        """Initialize the synthetic data fetcher"""
        self.base_dir = "data/synthetic"
        # Create directory if it doesn't exist
        os.makedirs(self.base_dir, exist_ok=True)
        
    def fetch_data(self, symbol, start_date, end_date):
        """
        Generate synthetic data for a given symbol and date range.
        
        Args:
            symbol (str): The stock symbol.
            start_date (str): Start date in YYYY-MM-DD format.
            end_date (str): End date in YYYY-MM-DD format.
            
        Returns:
            pd.DataFrame: Dataframe with synthetic stock data.
        """
        # Parse dates
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Check if data file already exists
        file_path = os.path.join(self.base_dir, f"{symbol}.csv")
        if os.path.exists(file_path):
            # Read existing data
            df = pd.read_csv(file_path, parse_dates=['Date'])
            # Filter for date range
            df = df[(df['Date'] >= start) & (df['Date'] <= end)]
            if not df.empty:
                print(f"Loaded {len(df)} days of synthetic data for {symbol}")
                return df
        
        # Generate date range
        date_range = pd.date_range(start=start, end=end, freq='B')  # Business days
        
        # Generate synthetic price data
        np.random.seed(42)  # For reproducibility
        
        # Generate an initial price for each symbol
        initial_price = np.random.uniform(50, 500)
        
        # Generate daily returns with a slight upward bias
        daily_returns = np.random.normal(0.0005, 0.015, size=len(date_range))
        
        # Compute prices using cumulative returns
        cumulative_returns = np.cumprod(1 + daily_returns)
        prices = initial_price * cumulative_returns
        
        # Generate OHLC and volume data
        data = []
        for i, date in enumerate(date_range):
            # Base price for the day
            base_price = prices[i]
            
            # Generate Open, High, Low, Close around the base price
            daily_volatility = np.random.uniform(0.01, 0.03)
            open_price = base_price * np.random.uniform(0.99, 1.01)
            high_price = base_price * np.random.uniform(1.0, 1.0 + daily_volatility)
            low_price = base_price * np.random.uniform(1.0 - daily_volatility, 1.0)
            close_price = base_price * np.random.uniform(0.99, 1.01)
            
            # Ensure High >= Open, Close, Low and Low <= Open, Close
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            
            # Generate volume
            volume = np.random.randint(100000, 10000000)
            
            data.append({
                'Date': date,
                'Open': open_price,
                'High': high_price,
                'Low': low_price,
                'Close': close_price,
                'Volume': volume,
                'Adj Close': close_price  # Assuming no adjustments for simplicity
            })
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Save to file for future use
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        
        print(f"Generated {len(df)} days of synthetic data for {symbol}")
        
        return df
        
    def add_technical_indicators(self, df):
        """
        Add technical indicators to the dataframe.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data.
            
        Returns:
            pd.DataFrame: DataFrame with added technical indicators.
        """
        # Call the parent class implementation
        return super().add_technical_indicators(df)
    
    def prepare_data_for_agent(self, df, window_size=20):
        """
        Prepare data for the trading agent.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data and technical indicators.
            window_size (int): Size of the lookback window.
            
        Returns:
            tuple: (prices, features) where prices is a numpy array of close prices
                  and features is a numpy array of normalized feature data.
        """
        # First add technical indicators if they don't exist
        if 'SMA_20' not in df.columns:
            df = self.add_technical_indicators(df)
        
        # Get the closing prices
        prices = df['Close'].values
        
        # Define features to use in the normalized representation
        feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume', 
            'SMA_5', 'SMA_10', 'SMA_20',
            'EMA_5', 'EMA_10', 'EMA_20',
            'MACD', 'MACD_Signal', 'MACD_Hist', 'RSI_14',
            'Middle_Band', 'Upper_Band', 'Lower_Band', 
            'ATR_14', 'ADX_14', 'Return'
        ]
        
        # Ensure all expected columns exist, filling with zeros if not
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        # Create a normalized DataFrame
        norm_df = df.copy()
        
        # Normalize each feature column
        for col in feature_columns:
            min_val = norm_df[col].min()
            max_val = norm_df[col].max()
            if max_val > min_val:  # Avoid division by zero
                norm_df[col] = (norm_df[col] - min_val) / (max_val - min_val)
            else:
                norm_df[col] = 0  # If all values are the same, set to 0
        
        # Extract features for each day
        features = []
        for i in range(len(df)):
            # Get values for the current day
            feature_vector = norm_df.iloc[i][feature_columns].values
            features.append(feature_vector)
        
        features = np.array(features, dtype=np.float32)
        
        return prices, features 