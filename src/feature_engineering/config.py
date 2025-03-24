"""
Feature Configuration Module

Defines standard feature sets and provides versioning mechanisms.
"""
import hashlib
import json
from typing import Dict, List, Any


class FeatureConfig:
    """
    Handles feature configuration and versioning.
    """
    
    CURRENT_VERSION = "1.0.0"
    
    @staticmethod
    def get_version_hash(feature_list: List[str], normalization_params: Dict[str, Any] = None) -> str:
        """
        Generate a hash for a specific feature configuration.
        
        Args:
            feature_list (List[str]): List of features
            normalization_params (Dict[str, Any]): Normalization parameters
            
        Returns:
            str: Hash of the configuration
        """
        config = {
            "features": sorted(feature_list),
            "normalization": normalization_params or {},
            "version": FeatureConfig.CURRENT_VERSION
        }
        
        # Create a deterministic string representation
        config_str = json.dumps(config, sort_keys=True)
        
        # Return the hash
        return hashlib.md5(config_str.encode()).hexdigest()
    
    @staticmethod
    def save_config(feature_list: List[str], 
                    normalization_params: Dict[str, Any] = None,
                    filename: str = "feature_config.json") -> None:
        """
        Save a feature configuration to file.
        
        Args:
            feature_list (List[str]): List of features
            normalization_params (Dict[str, Any]): Normalization parameters
            filename (str): File to save to
        """
        config = {
            "features": feature_list,
            "normalization": normalization_params or {},
            "version": FeatureConfig.CURRENT_VERSION,
            "hash": FeatureConfig.get_version_hash(feature_list, normalization_params)
        }
        
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
    
    @staticmethod
    def load_config(filename: str = "feature_config.json") -> Dict[str, Any]:
        """
        Load a feature configuration from file.
        
        Args:
            filename (str): File to load from
            
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        with open(filename, 'r') as f:
            return json.load(f)


# Define standard feature configurations
FEATURE_CONFIGS = {
    # Minimal set with essential features
    "minimal": [
        "price_change",         # Daily price change
        "volatility",           # Price volatility
        "volume_change",        # Volume change
        "high_low_range",       # Daily trading range
        "rsi_14"                # Relative Strength Index (14)
    ],
    
    # Standard set for most models
    "standard": [
        # Trend indicators
        "price_change",         # Daily price change
        "sma_5",                # 5-day Simple Moving Average
        "sma_10",               # 10-day Simple Moving Average
        "sma_20",               # 20-day Simple Moving Average
        "ema_12",               # 12-day Exponential Moving Average
        "ema_26",               # 26-day Exponential Moving Average
        
        # Volatility indicators
        "volatility",           # Price volatility
        "bollinger_bandwidth",  # Bollinger Bands Width
        "high_low_range",       # Daily trading range
        
        # Momentum indicators
        "rsi_14",               # Relative Strength Index (14)
        "macd",                 # MACD Line
        "macd_signal",          # MACD Signal Line
        "macd_histogram",       # MACD Histogram
        "momentum_5",           # 5-day Momentum
        
        # Volume indicators
        "volume_change",        # Volume change
        "volume_sma_ratio",     # Volume vs SMA ratio
        
        # Additional
        "day_of_week",          # Day of week (0-6)
        "month",                # Month (1-12)
        "vwap_distance",        # Distance from VWAP
        "atr_14",               # Average True Range (14)
        "stoch_k"               # Stochastic Oscillator %K
    ],
    
    # Advanced set with more sophisticated features
    "advanced": [
        # All standard features
        "price_change", "sma_5", "sma_10", "sma_20", "ema_12", "ema_26",
        "volatility", "bollinger_bandwidth", "high_low_range",
        "rsi_14", "macd", "macd_signal", "macd_histogram", "momentum_5",
        "volume_change", "volume_sma_ratio",
        "day_of_week", "month", "vwap_distance", "atr_14", "stoch_k",
        
        # Additional trend indicators
        "sma_50", "sma_200",    # 50 and 200-day SMAs
        "ichimoku_a", "ichimoku_b", # Ichimoku Cloud lines
        
        # Additional volatility indicators
        "atr_ratio",            # ATR as percentage of price
        "bollinger_position",   # Position within Bollinger Bands
        
        # Additional momentum indicators
        "rsi_2", "rsi_5",       # Short-term RSI values
        "stoch_d",              # Stochastic Oscillator %D
        "cci_20",               # Commodity Channel Index
        "williams_r",           # Williams %R
        
        # Additional volume indicators
        "obv",                  # On-Balance Volume
        "cmf_20",               # Chaikin Money Flow
        "mfi_14",               # Money Flow Index
        
        # Mean reversion indicators
        "distance_from_sma_20", # Distance from 20-day SMA
        "distance_from_sma_50", # Distance from 50-day SMA
        
        # Trend strength indicators
        "adx_14",               # Average Directional Index
        "dmi_plus",             # DMI+ (Positive Directional Movement)
        "dmi_minus",            # DMI- (Negative Directional Movement)
        
        # Seasonal/Cyclical
        "day_of_month",         # Day of month (1-31)
        "quarter",              # Quarter (1-4)
        
        # Price patterns
        "doji",                 # Doji candlestick pattern
        "engulfing",            # Engulfing pattern
        "hammer"                # Hammer pattern
    ]
} 