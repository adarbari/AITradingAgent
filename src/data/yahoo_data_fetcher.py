"""
Yahoo Finance data fetcher for retrieving real stock data
"""
import pandas as pd
import numpy as np
import os
import time
import random
import yfinance as yf
from pandas_datareader import data as pdr
from datetime import datetime, timedelta
from .base_data_fetcher import BaseDataFetcher

class YahooDataFetcher(BaseDataFetcher):
    """Class for fetching stock data from Yahoo Finance"""
    
    def __init__(self):
        """Initialize the Yahoo Finance data fetcher"""
        self.cache_dir = "data/cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.max_retries = 3
        self.retry_delay = 2  # seconds
        
    def fetch_data(self, symbol, start_date, end_date):
        """
        Fetch stock data from Yahoo Finance for a given symbol and date range.
        Implements a caching mechanism and retry logic with fallback to synthetic data.
        
        Args:
            symbol (str): The stock symbol.
            start_date (str): Start date in YYYY-MM-DD format.
            end_date (str): End date in YYYY-MM-DD format.
            
        Returns:
            pd.DataFrame: Dataframe with stock data.
        """
        # Parse dates
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Define cache file path
        cache_file = os.path.join(self.cache_dir, f"{symbol}_{start_date.replace('-', '')}_{end_date.replace('-', '')}.csv")
        
        # Check if cached data exists
        if os.path.exists(cache_file):
            try:
                df = pd.read_csv(cache_file, parse_dates=['Date'])
                print(f"Loaded cached data for {symbol} from {cache_file}")
                return df
            except Exception as e:
                print(f"Error reading cache file: {e}")
        
        # Try to fetch data using pandas_datareader
        attempts = 0
        df = None
        
        while attempts < self.max_retries:
            try:
                print(f"Attempt {attempts + 1}: Fetching {symbol} data using pandas_datareader")
                yf.pdr_override()
                df = pdr.get_data_yahoo(symbol, start=start, end=end)
                
                # Reset index to make Date a column
                df = df.reset_index()
                
                # Validate data
                if df.empty:
                    raise ValueError(f"Empty dataframe returned for {symbol}")
                
                required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    raise ValueError(f"Missing required columns: {missing_columns}")
                
                # Standardize column names
                df.columns = [col if col in required_columns else col for col in df.columns]
                
                # Cache the data
                df.to_csv(cache_file, index=False)
                print(f"Successfully fetched and cached {len(df)} days of data for {symbol}")
                
                # Add technical indicators
                df = self.add_technical_indicators(df)
                return df
                
            except Exception as e:
                print(f"Error fetching {symbol} with pandas_datareader: {str(e)}")
                attempts += 1
                
                # Try yfinance directly as a fallback
                if attempts == self.max_retries - 1:
                    try:
                        print(f"Trying direct yfinance for {symbol}")
                        ticker = yf.Ticker(symbol)
                        df = ticker.history(start=start, end=end)
                        
                        if df.empty:
                            raise ValueError(f"Empty dataframe returned from yfinance for {symbol}")
                        
                        # Reset index to make Date a column
                        df = df.reset_index()
                        
                        # Standardize column names
                        df.columns = [col.title() if col.lower() in ['open', 'high', 'low', 'close', 'volume'] else col for col in df.columns]
                        df.rename(columns={'Stock Splits': 'Splits', 'Dividends': 'Dividends', 'Date': 'Date'}, inplace=True)
                        
                        # Cache the data
                        df.to_csv(cache_file, index=False)
                        print(f"Successfully fetched and cached {len(df)} days of data for {symbol} using yfinance directly")
                        
                        # Add technical indicators
                        df = self.add_technical_indicators(df)
                        return df
                        
                    except Exception as e2:
                        print(f"Error with direct yfinance for {symbol}: {str(e2)}")
                
                # Exponential backoff
                sleep_time = self.retry_delay * (2 ** attempts) + random.uniform(0, 1)
                print(f"Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
        
        # If all attempts fail, try to use synthetic data
        print(f"All attempts to fetch {symbol} failed. Trying to use synthetic data...")
        
        # Check if synthetic data exists
        synthetic_file = os.path.join("data/synthetic", f"{symbol}.csv")
        if os.path.exists(synthetic_file):
            try:
                df = pd.read_csv(synthetic_file, parse_dates=['Date'])
                df = df[(df['Date'] >= start) & (df['Date'] <= end)]
                if not df.empty:
                    print(f"Using synthetic data for {symbol}")
                    df.to_csv(cache_file, index=False)
                    return self.add_technical_indicators(df)
            except Exception as e:
                print(f"Error loading synthetic data: {e}")
        
        # Generate synthetic data as a last resort
        print(f"Generating synthetic data for {symbol}")
        from .synthetic_data_fetcher import SyntheticDataFetcher
        synthetic_fetcher = SyntheticDataFetcher()
        df = synthetic_fetcher.fetch_data(symbol, start_date, end_date)
        df.to_csv(cache_file, index=False)
        
        return df 