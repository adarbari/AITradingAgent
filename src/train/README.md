# Training Module

This module provides functionality for training and caching models for different stock symbols. The key components are:

## Components

- **TrainingManager**: Core class responsible for managing model training and caching.
- **CLI**: Command-line interface for training and managing cached models.

## Features

1. **Caching**: Models are cached based on their training parameters and can be reused for multiple backtest runs.
2. **Hash-based Identification**: Models are identified by a hash of their configuration parameters, ensuring that identical configurations are reused.
3. **Forced Retraining**: Option to force retraining even if a cached model exists.
4. **Cache Management**: Commands to list and clear cached models.
5. **Data Source Integration**: Uses the DataFetcherFactory to fetch data from different sources.

## Architecture

The training module is designed to integrate with the existing data fetching infrastructure:

1. The `TrainingManager` uses the `DataFetcherFactory` to create the appropriate data fetcher (Yahoo or Synthetic).
2. Data fetchers handle fetching raw OHLCV data and adding technical indicators.
3. Feature engineering is applied to prepare data for model training.
4. The training environment is created using processed features and prices.
5. After training, models are cached for future use.

## Usage

### From Python

```python
from src.train.trainer import TrainingManager
from src.data import DataFetcherFactory

# Create a training manager
trainer = TrainingManager(models_dir="models")

# Get (or train if not cached) a model
model, model_path = trainer.get_model(
    symbol="AAPL",
    train_start="2020-01-01",
    train_end="2022-12-31",
    feature_count=21,
    data_source="yfinance",
    timesteps=100000
)

# List cached models
cached_models = trainer.list_cached_models(symbol="AAPL")
```

### From Command Line

Train a model:
```bash
python -m src.train.cli train --symbols AAPL MSFT --train-start 2020-01-01 --train-end 2022-12-31
```

List cached models:
```bash
python -m src.train.cli list
```

List cached models for a specific symbol:
```bash
python -m src.train.cli list --symbol AAPL
```

Clear cached models:
```bash
python -m src.train.cli clear --symbol AAPL --older-than 2022-01-01
```

## Benefits

1. **Reduced Training Time**: Avoid retraining models with the same parameters multiple times.
2. **Reproducibility**: Cached models ensure consistent results across multiple runs.
3. **Organization**: Models are systematically named and stored with their metadata.
4. **Memory Efficiency**: Only load the models you need when you need them.
5. **Integration**: Leverages existing data fetching infrastructure.
6. **Robustness**: Graceful fallback to synthetic data when real data is unavailable.

## How Caching Works

The TrainingManager creates a hash of the configuration parameters and stores it in a JSON file. When a model is requested, it checks if a model with the same hash already exists. If it does, the cached model is loaded; otherwise, a new model is trained.

## Data Sources

The module supports multiple data sources through the DataFetcherFactory:

1. **Yahoo Finance** (`yfinance`): Fetches real-world historical stock data.
2. **Synthetic** (`synthetic`): Generates synthetic data for testing and development.

If a data fetcher fails to retrieve data, the system automatically falls back to synthetic data generation to ensure training can continue. 