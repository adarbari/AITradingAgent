# Feature Engineering Module Refactoring

## Overview

The feature engineering code in the AI Trading Agent project has been restructured to improve maintainability, extensibility, and code organization. This document explains the changes made, how to use the new feature engineering module, and the benefits of the new architecture.

## Changes Made

### 1. Reorganized Code Structure

The feature engineering code has been moved from utility files into a dedicated module with the following structure:

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
└── README.md             # Module documentation
```

### 2. Feature Registry Pattern

Implemented a feature registry pattern that allows:
- Dynamic registration of feature calculation functions
- Categorization of features by type (price, volume, momentum, etc.)
- Easy discovery and usage of available features
- Automatic dependency resolution

### 3. Modular Feature Implementation

Each feature category is now implemented in its own file, making it easier to:
- Find specific features
- Add new features
- Organize features by logical grouping
- Document feature behavior

### 4. Feature Pipeline

Added a feature processing pipeline that handles:
- Sequential computation of features
- Data cleaning and normalization
- Missing value handling
- Feature selection and combination

### 5. Caching System

Implemented a cache system for computed features to:
- Avoid redundant computations
- Speed up model training and backtesting
- Provide cache invalidation when needed
- Track cache usage statistics

### 6. Feature Configuration

Created a configuration system for defining feature sets:
- Standard feature sets for different trading strategies
- Customizable feature combinations
- Feature set versioning

## How to Use the New Module

### Basic Usage

```python
from src.feature_engineering import process_features
from src.utils.feature_utils import get_data

# Get stock data
data = get_data("AAPL", "2022-01-01", "2023-01-01")

# Process features using a predefined feature set
features = process_features(data, feature_set="standard")
```

### Using in train_and_backtest.py

The `train_and_backtest.py` script has been updated to use the new feature engineering module. You can specify the feature set to use with the `--feature-set` parameter:

```bash
python src/scripts/train_and_backtest.py --symbol AAPL --feature-set standard --train --backtest
```

### Creating Custom Feature Sets

You can define custom feature sets in `src/feature_engineering/config.py`:

```python
FEATURE_CONFIGS = {
    "custom_set": [
        "price_change",
        "volatility",
        "rsi_14",
        "volume_change",
        "day_of_week"
    ]
}
```

### Adding New Features

To add a new feature, create a function in the appropriate category file:

```python
# In src/feature_engineering/features/custom_features.py
from src.feature_engineering.registry import FeatureRegistry

@FeatureRegistry.register(name="my_feature", category="custom")
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

## Benefits of the New Architecture

### 1. Modularity and Organization

- Each feature type is organized in its own file
- Features are categorized by their functional purpose
- Clear separation of concerns between feature generation, pipeline, and caching

### 2. Extensibility

- Easy to add new features without modifying existing code
- New categories can be added seamlessly
- The registry pattern allows for automatic discovery of features

### 3. Maintainability

- Better documentation of features and their behavior
- Centralized configuration of feature sets
- Consistent feature registration and implementation pattern

### 4. Performance

- Caching mechanism avoids redundant calculations
- More efficient feature computation pipeline
- Better error handling and logging

### 5. Reusability

- Features can be reused across different parts of the application
- Consistent interface for feature computation
- Clear dependency management

### 6. Testing

- Easier to test individual features in isolation
- Better organization for test files
- Improved testability with clear interfaces

## Examples

See `src/examples/feature_engineering_example.py` for a complete example of how to use the new feature engineering module, including:

- Fetching data
- Computing features
- Using the cache
- Exploring feature correlations
- Visualizing feature behavior

## Migration Guide

If you have existing code that uses the old feature engineering approach, here's how to migrate:

1. Replace direct feature calculation with registry calls:
   
   **Old approach**:
   ```python
   # Manual feature calculation
   features['price_change'] = data['Close'] / data['Close'].shift(1) - 1
   ```
   
   **New approach**:
   ```python
   # Using the registry
   from src.feature_engineering import FeatureRegistry
   features['price_change'] = FeatureRegistry.compute_feature('price_change', data)
   ```

2. Replace feature preparation with the process_features function:
   
   **Old approach**:
   ```python
   # Manual feature preparation
   features = prepare_features_from_indicators(data, expected_feature_count=21)
   ```
   
   **New approach**:
   ```python
   # Using the process_features function
   from src.feature_engineering import process_features
   features = process_features(data, feature_set="standard")
   ```

## Conclusion

The new feature engineering module provides a more organized, maintainable, and extensible way to manage features in the AI Trading Agent project. By centralizing feature logic and implementing a registry pattern, we've made it easier to add new features and ensure consistency across the codebase. 