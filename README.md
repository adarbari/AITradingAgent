# AI Trading Agent

[![Tests](https://github.com/abhinavdarbari/AITradingAgent/actions/workflows/tests.yml/badge.svg)](https://github.com/abhinavdarbari/AITradingAgent/actions/workflows/tests.yml)

An artificial intelligence trading agent that uses reinforcement learning to trade financial assets.

## Overview

This project implements a trading agent using reinforcement learning techniques, specifically Proximal Policy Optimization (PPO), to make investment decisions in financial markets. The agent learns from historical market data or synthetic data and makes decisions to buy, sell, or hold assets to maximize returns.

## Project Structure

```
├── src                     # Source code
│   ├── agent               # Trading agent implementation
│   ├── backtest            # Backtesting framework
│   ├── data                # Data fetching and processing
│   └── models              # ML models for trading
├── tests                   # Test files
│   ├── unit                # Unit tests
│   └── integration         # Integration tests
├── scripts                 # Utility scripts
└── models                  # Saved model files
```

## Features

- Reinforcement learning-based trading agent
- Support for both real market data and synthetic data
- Backtesting framework to evaluate strategy performance
- Customizable model parameters
- Visualization of trading performance

## Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/abhinavdarbari/AITradingAgent.git
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

## Running Tests

To run all tests:
```
python run_tests.py
```

To run only unit tests:
```
python run_tests.py --unit
```

To run only integration tests:
```
python run_tests.py --integration
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