# AI Trading Agent

[![Tests](https://github.com/adarbari/AITradingAgent/actions/workflows/tests.yml/badge.svg)](https://github.com/adarbari/AITradingAgent/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/adarbari/AITradingAgent/branch/main/graph/badge.svg)](https://codecov.io/gh/adarbari/AITradingAgent)

An artificial intelligence trading agent that uses reinforcement learning to trade financial assets.

## Overview

This project implements a trading agent using reinforcement learning techniques, specifically Proximal Policy Optimization (PPO), to make investment decisions in financial markets. The agent learns from historical market data or synthetic data and makes decisions to buy, sell, or hold assets to maximize returns.

## Project Structure

```
├── src                     # Source code
│   ├── agent               # Trading agent implementation
│   ├── backtest            # Backtesting framework
│   ├── data                # Data fetching and processing
│   ├── models              # ML models for trading
│   └── scripts             # Execution scripts
│       ├── train_and_backtest.py  # Train and backtest models
│       └── compare_models.py      # Compare different model configurations
├── tests                   # Test files
│   ├── unit                # Unit tests
│   └── integration         # Integration tests
├── models                  # Saved model files
└── results                 # Results from backtesting
```

## Features

- Reinforcement learning-based trading agent
- Support for both real market data and synthetic data
- Backtesting framework to evaluate strategy performance
- Customizable model parameters
- Visualization of trading performance
- Model configuration comparison tools

## Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/adarbari/AITradingAgent.git
   cd AITradingAgent
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Install Git hooks for development:
   ```
   python scripts/install_hooks.py
   ```

## Usage

### Training and Backtesting a Model

To train a model on historical data and backtest it:

```bash
python src/scripts/train_and_backtest.py --train --backtest --symbol AMZN --train-start 2022-01-01 --train-end 2023-12-31 --test-start 2024-01-01 --test-end 2024-04-30
```

### Comparing Different Model Configurations

To compare different model configurations and find the best one:

```bash
python src/scripts/compare_models.py --symbol AMZN --test-start 2024-01-01 --test-end 2024-04-30
```

## Running Tests

To run all tests:
```
python scripts/run_tests.py
```

To run only unit tests:
```
python scripts/run_tests.py --unit
```

To run only integration tests:
```
python scripts/run_tests.py --integration
```

## Development

### Testing Guidelines

This project follows strict testing guidelines to ensure code quality and reliability. For detailed testing practices, please refer to the [TESTING.md](TESTING.md) file.

### Key Testing Rules

1. Every new feature or bug fix must include appropriate test cases
2. All tests must pass before submitting a pull request
3. Use `scripts/generate_test.py` to generate test templates for new classes/functions

### Generating Tests

To automatically generate test templates for new code:
```
python scripts/generate_test.py path/to/your/file.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.