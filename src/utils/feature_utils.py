"""
Utilities for feature preparation and data handling.
"""
import numpy as np
import pandas as pd
import yfinance as yf


def prepare_features_from_indicators(features_df, expected_feature_count=21, verbose=False):
    """
    Prepare features from a DataFrame that already contains technical indicators.
    Handles column conversion, NaN values, normalization, and feature count matching.
    
    Args:
        features_df (pd.DataFrame): DataFrame containing technical indicators
        expected_feature_count (int): Expected number of features
        verbose (bool): Whether to print information about the transformation
        
    Returns:
        pd.DataFrame: Processed features DataFrame
    """
    features = features_df.copy()
    
    # First, ensure that all columns are numeric
    for col in features.columns:
        if not pd.api.types.is_numeric_dtype(features[col]):
            if pd.api.types.is_datetime64_any_dtype(features[col]):
                if verbose:
                    print(f"Converting datetime column {col} to numeric timestamp")
                features[col] = features[col].astype(np.int64) // 10**9  # Convert to Unix timestamp seconds
            else:
                if verbose:
                    print(f"Converting non-numeric column {col} to numeric values")
                try:
                    features[col] = pd.to_numeric(features[col], errors='coerce')
                except:
                    if verbose:
                        print(f"Could not convert column {col} to numeric, dropping it")
                    features = features.drop(columns=[col])
    
    # Handle NaN values
    features = features.fillna(0)  # Replace NaN with zeros
    
    # Apply simple normalization to avoid extreme values
    for col in features.columns:
        if features[col].std() > 0:
            features[col] = (features[col] - features[col].mean()) / features[col].std()
        else:
            features[col] = 0  # If std is 0, set all values to 0
    
    # Ensure we have the expected number of features
    if len(features.columns) < expected_feature_count:
        if verbose:
            print(f"Warning: Expected {expected_feature_count} features but only {len(features.columns)} are available.")
            print("Adding dummy features to match the expected count...")
        
        # Add missing features with zeros
        for i in range(len(features.columns), expected_feature_count):
            feature_name = f"dummy_feature_{i}"
            features[feature_name] = 0.0
    
    elif len(features.columns) > expected_feature_count:
        if verbose:
            print(f"Warning: More features than expected ({len(features.columns)} vs {expected_feature_count}).")
            print(f"Keeping only the first {expected_feature_count} features...")
        features = features.iloc[:, :expected_feature_count]
    
    # Final check for NaN or infinite values
    if np.isnan(features.values).any() or np.isinf(features.values).any():
        if verbose:
            print("Warning: NaN or infinite values detected after processing. Replacing with zeros.")
        features = features.replace([np.inf, -np.inf, np.nan], 0)
    
    if verbose:
        print(f"Prepared {len(features)} data points with {len(features.columns)} features")
    
    return features


def prepare_robust_features(data, feature_count=21, verbose=False):
    """
    Prepare features for the trading agent with robust error handling.
    
    Args:
        data (pd.DataFrame): Raw price data with OHLCV columns
        feature_count (int): Expected number of features
        verbose (bool): Whether to print information about the transformation
        
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