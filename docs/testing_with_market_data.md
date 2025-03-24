# Testing with Market Data

## Overview

This document explains how to test your trading models with both real and synthetic market data, with a focus on making tests suitable for CI environments.

## Approaches to Testing with Market Data

### 1. Using Real Market Data (Cached)

For tests that use real historical market data (like stock prices from 2023-2025), the best approach is to fetch the data once and cache it for future test runs:

```python
@pytest.fixture(scope="class", autouse=True)
def setup_test_data():
    # Initialize Yahoo data fetcher
    yahoo_fetcher = YahooDataFetcher()
    
    # Check if data is already cached
    cache_file = yahoo_fetcher._get_cache_path("AAPL", "2023-01-01", "2023-12-31")
    
    # Fetch data if not in cache
    if not os.path.exists(cache_file):
        print("Fetching real market data")
        data = yahoo_fetcher.fetch_data("AAPL", "2023-01-01", "2023-12-31")
    else:
        print("Using cached market data")
```

Benefits of this approach:
- Uses real market data for accurate tests
- Only fetches data once, then reuses cache
- No API calls in CI environment after first run
- Tests reflect real market conditions

### 2. Using Synthetic Data (For Future or Unavailable Data)

For testing with future dates or when you can't access real market data, the synthetic data approach is useful:

```python
@pytest.fixture(scope="class", autouse=True)
def setup_synthetic_test_data():
    synthetic_fetcher = SyntheticDataFetcher()
    yahoo_fetcher = YahooDataFetcher()
    
    # Generate synthetic data
    data = synthetic_fetcher.fetch_data("AAPL", "2026-01-01", "2026-12-31")
    
    # Save to the Yahoo data cache
    cache_file = yahoo_fetcher._get_cache_path("AAPL", "2026-01-01", "2026-12-31")
    data.to_csv(cache_file, index=False)
```

Why use synthetic data:
- **CI/CD Compatibility**: External API calls can fail or be rate-limited in CI environments
- **Future Date Testing**: Testing on future dates (beyond current date) requires synthetic data
- **Reproducibility**: Tests using real market APIs may produce different results over time
- **Speed**: No network calls means faster test execution
- **Offline Development**: Develop and test without internet connection

## How Synthetic Data is Generated

The `SyntheticDataFetcher` class generates realistic OHLCV (Open, High, Low, Close, Volume) data with properties similar to real market data:

- Daily returns follow a normal distribution with a slight upward bias
- Creates realistic volatility between daily high and low prices
- Ensures proper relationships between OHLC values (High ≥ Open, Close ≥ Low)
- Maintains continuity between trading days

## How the YahooDataFetcher Caching Works

The `YahooDataFetcher` class has built-in caching:

1. When you call `fetch_data()` or `fetch_ticker_data()`, it first checks for cached files
2. If a file exists in the cache directory, it loads the data from there
3. Only if the file doesn't exist does it make an API call to Yahoo Finance
4. After fetching data, it saves it to the cache directory for future use

Cache files are named using this pattern:
```
{symbol}_{start_date}_{end_date}.csv
```

For ticker format data (with Date as index):
```
{symbol}_ticker_{start_date}_{end_date}.csv
```

## Feature Caching

In addition to market data caching, our system also caches computed features:

1. Features are computed from market data using the `process_features()` function
2. The `FeatureCache` class saves these features to avoid redundant calculations
3. A unique cache key is generated based on the symbol, date range, and feature set

```python
feature_cache = FeatureCache()
cache_key = feature_cache.get_cache_key("AAPL", "2023-01-01", "2023-12-31", "standard")
features = feature_cache.load(cache_key)

if features is None:
    # Features not cached, compute them
    features = process_features(data, feature_set="standard")
    feature_cache.save(features, cache_key)
```

## Example: Testing with Both Historical and Future Data

See `tests/integration/test_amzn_train_backtest.py` for a complete example that:
1. First tries to use cached data
2. Fetches real data if the cache is empty
3. Processes and caches features for faster test runs

## Best Practices

1. **Prefer real data when available** - Use actual market data for historical periods
2. **Cache aggressively** - Once data is fetched, ensure it's cached for future test runs
3. **Use synthetic data sparingly** - Limit synthetic data to future dates or when real data is unavailable
4. **Maintain cache in version control** - If possible, include cache files in the repo for CI
5. **Clearly document data sources** - Make it clear which tests use real vs. synthetic data
6. **Seed the random number generator** for reproducible synthetic tests
7. **Use pytest fixtures** to handle the data setup and teardown

## Limitations and Considerations

- **Cache invalidation**: You may need to periodically clear cache for tests that should use fresh market data
- **Test reliability**: Tests using real data might produce different results over time as more data becomes available
- **CI environment**: Ensure CI has access to cache directories, or pre-populate them as part of the CI setup
- **Synthetic data limitations**: Synthetic data lacks real-world anomalies and market events, and has different statistical properties than real data

For final validation of trading strategies, always use real market data when possible. 