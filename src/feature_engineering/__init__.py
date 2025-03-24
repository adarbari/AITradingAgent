"""
Feature Engineering Package

This package provides a comprehensive framework for feature engineering in trading systems.
It includes feature registry, pipeline architecture, and a variety of finance-specific features.
"""

from .registry import FeatureRegistry
from .pipeline import FeaturePipeline
from .config import FeatureConfig, FEATURE_CONFIGS
from .cache import FeatureCache

# Import features package to ensure all feature functions are registered
from . import features

# Re-export key APIs for convenience
__all__ = [
    'FeatureRegistry',
    'FeaturePipeline',
    'FeatureConfig',
    'FeatureCache',
    'FEATURE_CONFIGS',
    'process_features',
]

# Convenience function for processing features
def process_features(data, feature_set="standard", feature_count=21, verbose=False):
    """
    Process features from raw OHLCV data using a predefined feature set.
    
    Args:
        data (pd.DataFrame): Raw OHLCV data
        feature_set (str or list): Name of the feature set to use 
                          ("minimal", "standard", "advanced") or a list of features
        feature_count (int): Expected number of features
        verbose (bool): Whether to print information during processing
        
    Returns:
        pd.DataFrame: Processed features
    """
    from .config import FEATURE_CONFIGS
    from .pipeline import FeaturePipeline
    
    # Use the feature list from config, or use the feature set name as a feature list
    if isinstance(feature_set, list):
        feature_list = feature_set
    else:
        feature_list = FEATURE_CONFIGS.get(feature_set, feature_set.split(','))
    
    # Create and run the feature pipeline
    pipeline = FeaturePipeline(feature_list=feature_list, feature_count=feature_count, verbose=verbose)
    return pipeline.process(data) 