"""
Abstract base class for data fetching strategies
"""
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

class BaseDataFetcher(ABC):
    """Abstract base class for data fetching strategies"""
    
    @abstractmethod
    def fetch_data(self, symbol, start_date, end_date):
        """Fetch data for a given symbol and date range"""
        pass
    
    def add_technical_indicators(self, df):
        """Add technical indicators to the dataframe"""
        # Add Simple Moving Averages
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        
        # Add Exponential Moving Averages
        df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
        df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        
        # Add MACD
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # Add RSI (14-period)
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df['RSI_14'] = 100 - (100 / (1 + rs))
        
        # Add Bollinger Bands (20-day, 2 standard deviations)
        df['Middle_Band'] = df['Close'].rolling(window=20).mean()
        df['BB_Std'] = df['Close'].rolling(window=20).std()
        df['Upper_Band'] = df['Middle_Band'] + 2 * df['BB_Std']
        df['Lower_Band'] = df['Middle_Band'] - 2 * df['BB_Std']
        
        # Add Average True Range (ATR)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR_14'] = true_range.rolling(14).mean()
        
        # Add Average Directional Index (ADX)
        # True Range
        tr1 = df['High'] - df['Low']
        tr2 = abs(df['High'] - df['Close'].shift())
        tr3 = abs(df['Low'] - df['Close'].shift())
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        # Directional Movement
        up_move = df['High'] - df['High'].shift()
        down_move = df['Low'].shift() - df['Low']
        # Positive Directional Movement (+DM)
        pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        # Negative Directional Movement (-DM)
        neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        # Exponential moving averages of +DM, -DM, and TR
        tr_14 = tr.rolling(window=14).mean()
        pos_dm_14 = pd.Series(pos_dm).rolling(window=14).mean()
        neg_dm_14 = pd.Series(neg_dm).rolling(window=14).mean()
        # Positive Directional Index (+DI) and Negative Directional Index (-DI)
        pos_di_14 = 100 * (pos_dm_14 / tr_14)
        neg_di_14 = 100 * (neg_dm_14 / tr_14)
        # Average Directional Index (ADX)
        dx = 100 * abs(pos_di_14 - neg_di_14) / (pos_di_14 + neg_di_14)
        df['ADX_14'] = dx.rolling(window=14).mean()
        
        # Calculate daily returns
        df['Return'] = df['Close'].pct_change()
        
        # Forward fill NaN values
        df.fillna(method='ffill', inplace=True)
        # Backward fill any remaining NaN values at the beginning
        df.fillna(method='bfill', inplace=True)
        
        return df
    
    def prepare_data_for_agent(self, df, window_size=20):
        """Prepare data for the trading agent"""
        # Normalize features using min-max scaling
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 
                    'SMA_5', 'SMA_10', 'SMA_20',
                    'EMA_5', 'EMA_10', 'EMA_20',
                    'MACD', 'MACD_Signal', 'MACD_Hist', 'RSI_14',
                    'Middle_Band', 'Upper_Band', 'Lower_Band', 
                    'ATR_14', 'ADX_14', 'Return']
        
        # Create a copy of the dataframe to avoid modifying the original
        norm_df = df.copy()
        
        # Normalize each feature
        for feature in features:
            if feature in norm_df.columns:
                min_val = norm_df[feature].min()
                max_val = norm_df[feature].max()
                if max_val > min_val:  # Avoid division by zero
                    norm_df[feature] = (norm_df[feature] - min_val) / (max_val - min_val)
                else:
                    norm_df[feature] = 0  # If all values are the same, set to 0
        
        # Create rolling windows of data
        windows = []
        for i in range(len(norm_df) - window_size + 1):  # +1 to include the final window
            window_data = []
            for feature in features:
                if feature in norm_df.columns:
                    window_data.append(norm_df.iloc[i][feature])
                else:
                    window_data.append(0)  # Default value if feature is missing
            windows.append(window_data)
        
        # Convert to float32 to match expected type
        return np.array(windows, dtype=np.float32) if windows else np.array([]) 