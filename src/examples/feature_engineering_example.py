#!/usr/bin/env python3
"""
Example script demonstrating the use of the feature engineering module.
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import feature engineering module
from src.feature_engineering import process_features, FeatureRegistry, FEATURE_CONFIGS
from src.feature_engineering.pipeline import FeaturePipeline
from src.feature_engineering.cache import FeatureCache

# Import data fetching utilities
from src.utils.feature_utils import get_data


def main():
    """Main function to demonstrate feature engineering."""
    # Fetch some example data
    symbol = "AAPL"
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    
    print(f"Fetching data for {symbol} from {start_date} to {end_date}")
    data = get_data(symbol, start_date, end_date)
    
    if data is None or len(data) == 0:
        print("No data available. Using synthetic data instead.")
        # Generate synthetic data
        data = get_data(symbol, start_date, end_date, data_source="synthetic")
    
    print(f"Got {len(data)} data points")
    
    # Initialize feature cache
    cache = FeatureCache(cache_dir=".feature_cache", enable_cache=True, verbose=True)
    cache_key = cache.get_cache_key(symbol, start_date, end_date, "standard")
    
    # Check if features are cached
    cached_features = cache.load(cache_key)
    if cached_features is not None:
        print("Using cached features")
        features = cached_features
    else:
        print("Computing features from data")
        
        # Method 1: Use the convenience function
        features1 = process_features(data, feature_set="standard", verbose=True)
        
        # Method 2: Use the pipeline directly
        pipeline = FeaturePipeline(
            feature_list=FEATURE_CONFIGS["standard"],
            feature_count=21,
            verbose=True
        )
        features2 = pipeline.process(data)
        
        # Method 3: Compute specific features
        # Get a list of all registered features
        all_features = FeatureRegistry.list_features()
        print(f"All registered features: {len(all_features)}")
        
        # List features by category
        categories = FeatureRegistry.list_categories()
        print("Feature categories:")
        for category in categories:
            features_in_category = FeatureRegistry.get_features_by_category(category)
            print(f"  {category}: {len(features_in_category)} features")
        
        # Compute specific features
        specific_features = [
            "price_change",
            "volatility",
            "rsi_14",
            "macd",
            "volume_change",
            "day_of_week"
        ]
        features3 = FeatureRegistry.compute_features(specific_features, data)
        
        # Use method 1 for this example
        features = features1
        
        # Cache the results
        cache.save(features, cache_key)
    
    # Display feature statistics
    print("\nFeature Statistics:")
    print(f"Number of features: {len(features.columns)}")
    print(f"Number of samples: {len(features)}")
    
    # Display feature correlation matrix
    correlation = features.corr()
    plt.figure(figsize=(12, 10))
    plt.matshow(correlation, fignum=1)
    plt.title(f"Feature Correlation Matrix for {symbol}")
    plt.colorbar()
    plt.tight_layout()
    
    # Create a directory for plots if it doesn't exist
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/feature_correlation_{symbol}.png")
    
    # List highest correlated feature pairs
    print("\nHighest Correlated Feature Pairs:")
    corr_pairs = []
    for i in range(len(correlation.columns)):
        for j in range(i+1, len(correlation.columns)):
            corr_pairs.append((
                correlation.columns[i],
                correlation.columns[j],
                abs(correlation.iloc[i, j])
            ))
    
    # Sort by correlation (descending) and display top 5
    corr_pairs.sort(key=lambda x: x[2], reverse=True)
    for feature1, feature2, corr in corr_pairs[:5]:
        print(f"  {feature1} and {feature2}: {corr:.3f}")
    
    # Plot a few features over time
    plt.figure(figsize=(12, 8))
    
    # Select a few interesting features to plot
    features_to_plot = ["price_change", "rsi_14", "volatility", "volume_change"]
    if set(features_to_plot).issubset(set(features.columns)):
        for i, feature in enumerate(features_to_plot):
            plt.subplot(len(features_to_plot), 1, i+1)
            plt.plot(features.index, features[feature])
            plt.title(feature)
            plt.tight_layout()
        
        plt.savefig(f"plots/feature_timeseries_{symbol}.png")
    else:
        print(f"Some features not available among: {features_to_plot}")
        print(f"Available features: {features.columns.tolist()}")
    
    print("\nCache statistics:")
    stats = cache.get_cache_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main() 