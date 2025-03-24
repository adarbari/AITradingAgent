# AI Trading Agent - Project Context

## Project Overview

This project implements a reinforcement learning-based trading agent that can be trained on historical market data and backtest trading strategies. The system uses Proximal Policy Optimization (PPO) algorithms and can work with both synthetic and real market data.

## Architecture and Design Principles

### 1. Modular Design

The project follows a modular architecture with clear separation of concerns:

- **Data Module** (`src/data/`): Handles data acquisition, processing, and feature engineering.
- **Models Module** (`src/models/`): Manages model training, saving, and loading.
- **Backtest Module** (`src/backtest/`): Provides backtesting functionality and performance evaluation.
- **Agent Module** (`src/agent/`): Contains the trading environment implementation for reinforcement learning.
- **Utils Module** (`src/utils/`): Contains utility functions and helpers.
- **Scripts Module** (`src/scripts/`): Contains executable scripts for training, backtesting, and model comparison.

### 2. Interface-based Design

All major components implement abstract base classes (interfaces) to ensure:

- **Dependency Inversion Principle**: High-level modules depend on abstractions, not concrete implementations.
- **Interchangeability**: Different implementations can be swapped without changing other parts of the system.
- **Extension over Modification**: New functionality is added by creating new implementations of existing interfaces.

The key interfaces are:
- `BaseDataFetcher`: Interface for data acquisition strategies
- `BaseModelTrainer`: Interface for model training algorithms
- `BaseBacktester`: Interface for backtesting strategies
- `BaseTradingEnvironment`: Interface for trading environments

### 3. Factory Pattern

The system uses factory patterns to create appropriate implementations:
- `DataFetcherFactory`: Creates appropriate data fetcher instances based on the data source.

### 4. Consistent Error Handling

- All modules follow consistent error handling with appropriate logging and exception management.
- Failed operations return `None` with proper error messages rather than raising exceptions.

## Code Organization

### Directory Structure

```
AITradingAgent/
├── src/
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── base_trading_env.py
│   │   └── trading_env.py
│   ├── backtest/
│   │   ├── __init__.py
│   │   ├── base_backtester.py
│   │   └── backtester.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── base_data_fetcher.py
│   │   ├── data_fetcher_factory.py
│   │   ├── synthetic_data_fetcher.py
│   │   └── yahoo_data_fetcher.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_trainer.py
│   │   └── trainer.py
│   ├── scripts/
│   │   ├── train_and_backtest.py
│   │   ├── train_and_backtest_amzn.py
│   │   └── compare_models.py
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
├── models/  # Saved models
├── data/
│   ├── cache/  # Cached data
│   └── test/   # Test data
└── results/    # Backtest results
```

### Module Responsibilities

#### Data Module

- `BaseDataFetcher`: Abstract interface for all data fetchers
- `SyntheticDataFetcher`: Generates synthetic market data
- `YahooDataFetcher`: Fetches real market data from Yahoo Finance
- `DataFetcherFactory`: Creates appropriate data fetcher instances

#### Models Module

- `BaseModelTrainer`: Abstract interface for model trainers
- `ModelTrainer`: Implementation using PPO algorithm from stable-baselines3

#### Backtest Module

- `BaseBacktester`: Abstract interface for backtesting systems
- `Backtester`: Implementation for evaluating trading strategies

#### Agent Module

- `BaseTradingEnvironment`: Abstract interface for trading environments
- `TradingEnvironment`: Implementation of a Gym environment for RL trading

#### Scripts Module

- `train_and_backtest.py`: Main script for training and backtesting models with various configuration options
- `train_and_backtest_amzn.py`: Specialized script for training and backtesting AMZN stock
- `compare_models.py`: Script to train and compare multiple model configurations

## Coding Conventions

### 1. Method and Class Structure

- Every class has a clear, single responsibility
- All public methods have descriptive docstrings with:
  - Brief description
  - Args section with parameter descriptions
  - Returns section with return value descriptions
- Abstract methods use the `@abstractmethod` decorator

### 2. Error Handling

- Use try/except blocks around operations that might fail
- Provide meaningful error messages
- Return `None` for failed operations rather than raising exceptions
- Use verbose levels to control the detail of error output

### 3. Parameter Passing

- Use named parameters for clarity
- Use `**kwargs` for flexibility and future extensibility
- Provide sensible defaults for optional parameters

## Extension Guidelines

When adding new functionality:

1. **New Data Sources**: 
   - Create a new class implementing `BaseDataFetcher`
   - Add the new data source to the `DataFetcherFactory`

2. **New Model Types**:
   - Create a new class implementing `BaseModelTrainer`
   - Update the imports in the main script or create a model factory

3. **New Backtesting Strategies**:
   - Create a new class implementing `BaseBacktester`

4. **New Trading Environments**:
   - Create a new class implementing `BaseTradingEnvironment`

5. **New Scripts**:
   - Place new scripts in the `src/scripts` directory
   - Follow the existing pattern for argument parsing and error handling
   - Reuse existing components through proper imports

## Main Workflow

The main workflow in the scripts:

1. Parse command-line arguments
2. Create a ModelTrainer instance
3. Train models for each specified symbol
4. Create a Backtester instance
5. Backtest the trained models
6. Compare performance against market benchmarks
7. Generate and save performance metrics and visualizations

### Script-Specific Workflows

#### train_and_backtest.py
- Generic script for training and backtesting any stock with configurable parameters

#### train_and_backtest_amzn.py
- Specialized script for Amazon (AMZN) stock with predefined parameters

#### compare_models.py
- Trains multiple model configurations with different parameters
- Compares their performance using various metrics
- Identifies the best performing models

## Future Considerations

- Support for additional model types beyond PPO
- More sophisticated backtesting metrics and visualizations
- Portfolio-level optimization across multiple assets
- Real-time trading capabilities
- Hyperparameter optimization for model training 