# Feature Engineering Module

This module provides a comprehensive and extensible framework for feature engineering in the AI Trading Agent project. It focuses on transforming raw market data into meaningful features that can be used for model training and prediction.

## Module Structure

```
src/feature_engineering/
├── __init__.py           # Public exports and main entry points
├── registry.py           # Feature registry for registering and discovering features
├── pipeline.py           # Feature processing pipeline
├── config.py             # Feature set configurations
├── cache.py              # Feature caching implementation
├── features/             # Feature implementation grouped by category
│   ├── __init__.py       # Category initialization
│   ├── price_features.py # Price-based features
│   ├── volume_features.py # Volume-based features
│   ├── momentum_features.py # Momentum indicators
│   ├── trend_features.py # Trend-based features
│   ├── volatility_features.py # Volatility-based features
│   └── seasonal_features.py # Time/seasonal features
└── README.md             # This documentation file
```

## Key Components

### Feature Registry

The `FeatureRegistry` is a central repository for all feature generation functions. It allows for:
- Dynamic registration of feature functions
- Feature discovery by name or category
- Feature computation with automatic dependency resolution

### Feature Pipeline

The `FeaturePipeline` handles the sequential processing of features, including:
- Feature computation in the appropriate order
- Normalization and scaling
- Missing value handling
- Feature selection

### Feature Cache

The `FeatureCache` provides caching capabilities to avoid redundant computations:
- Disk-based caching with automatic invalidation
- Cache key generation based on input data characteristics
- Cache statistics tracking

### Feature Sets

Predefined feature sets in `config.py` allow for easy selection of groups of features for different trading strategies.

## Using the Module

### Basic Usage

```python
from src.feature_engineering import process_features
from src.utils.feature_utils import get_data

# Get stock data
data = get_data("AAPL", "2022-01-01", "2023-01-01")

# Process features using a predefined feature set
features = process_features(data, feature_set="standard")
```

### Advanced Usage

```python
from src.feature_engineering import FeatureRegistry, FEATURE_CONFIGS
from src.feature_engineering.pipeline import FeaturePipeline

# Get a list of all available features
all_features = FeatureRegistry.list_features()

# Create a custom pipeline with specific features
pipeline = FeaturePipeline(
    feature_list=["price_change", "rsi_14", "bollinger_bandwidth", "volume_sma_ratio"],
    feature_count=4
)

# Process data through the pipeline
features = pipeline.process(data)
```

## Adding New Features

### 1. Create a New Feature Function

```python
# In src/feature_engineering/features/custom_features.py
from src.feature_engineering.registry import FeatureRegistry

@FeatureRegistry.register(name="my_custom_feature", category="custom")
def calculate_my_feature(data):
    """
    Calculate a custom feature.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with OHLCV data
        
    Returns:
    --------
    pandas.Series
        The calculated feature values
    """
    # Feature calculation logic
    result = data['Close'] / data['Open'] - 1
    
    return result
```

### 2. Import the Module

Make sure your new feature module is imported in `src/feature_engineering/features/__init__.py`:

```python
# In src/feature_engineering/features/__init__.py
from . import custom_features
```

### 3. Use the New Feature

```python
# The feature is automatically available in the registry
from src.feature_engineering import FeatureRegistry

# Check if it's registered
assert "my_custom_feature" in FeatureRegistry.list_features()

# Use it in feature processing
custom_features = FeatureRegistry.compute_features(["my_custom_feature"], data)
```

## Feature Categories

The module organizes features into the following categories:

1. **Price Features**: Features derived from price action (open, high, low, close)
2. **Volume Features**: Features based on trading volume
3. **Momentum Features**: Indicators measuring the rate of price change
4. **Trend Features**: Indicators identifying market trends
5. **Volatility Features**: Measures of market volatility
6. **Seasonal Features**: Time-based patterns and seasonality

## Best Practices

1. **Feature Documentation**: Always document your features with docstrings that explain what the feature measures and how it's calculated.
2. **Handling Missing Data**: Ensure your feature functions handle NaN values appropriately.
3. **Normalization**: Consider normalizing your feature outputs to a standard range (usually -1 to 1 or 0 to 1).
4. **Performance**: For computationally expensive features, implement caching or efficient calculation methods.
5. **Testing**: Add tests for new features to ensure correct calculation.

## Example Usage

See `src/examples/feature_engineering_example.py` for a complete example of using the feature engineering module. 