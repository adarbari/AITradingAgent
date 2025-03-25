# AI Trading Agent

[![Tests](https://github.com/adarbari/AITradingAgent/actions/workflows/tests.yml/badge.svg)](https://github.com/adarbari/AITradingAgent/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/adarbari/AITradingAgent/branch/main/graph/badge.svg)](https://codecov.io/gh/adarbari/AITradingAgent)

An artificial intelligence trading agent that uses reinforcement learning and multi-agent systems to trade financial assets.

## Overview

This project implements trading agents using both reinforcement learning techniques and multi-agent architectures to make investment decisions in financial markets. The reinforcement learning agent (using Proximal Policy Optimization - PPO) learns from historical market data or synthetic data, while the multi-agent system leverages specialized agents for market analysis, sentiment analysis, and decision-making to create more sophisticated trading strategies.

## Project Structure

```
├── src                     # Source code
│   ├── agent               # Trading agent implementation
│   │   ├── multi_agent     # Multi-agent system components
│   │   ├── trading_env.py  # RL environment
│   │   └── trading_agent.py# RL agent
│   ├── backtest            # Backtesting framework
│   ├── data                # Data fetching and processing
│   ├── models              # ML models for trading
│   ├── utils               # Utility functions
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
- Multi-agent trading system with specialized agents
- Support for both real market data and synthetic data
- Integration with news sentiment and other data sources
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

### Training and Backtesting a Reinforcement Learning Model

To train a model on historical data and backtest it:

```bash
python src/scripts/train_and_backtest.py --train --backtest --symbol AMZN --train-start 2022-01-01 --train-end 2023-12-31 --test-start 2024-01-01 --test-end 2024-04-30
```

### Using the Multi-Agent Trading System

To analyze a stock using the multi-agent system:

```bash
python src/examples/multi_agent_example.py --symbol AAPL --days 90 --request "Analyze Apple's recent performance and provide a trading recommendation"
```

If you have an OpenAI API key and want to use LLM-based analysis:

```bash
python src/examples/multi_agent_example.py --symbol AAPL --api-key YOUR_OPENAI_API_KEY
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

### Multi-Agent Architecture

The multi-agent system uses LangGraph to coordinate specialized agents:

1. **Market Analysis Agent**: Analyzes technical indicators and price patterns
2. **Sentiment Analysis Agent**: Processes news and social media sentiment
3. **Strategy Agent**: Develops trading strategies based on inputs from other agents
4. **Risk Management Agent**: Evaluates and controls risk exposure
5. **Execution Agent**: Handles trade execution decisions

### Data Abstraction

The system uses a unified data management layer that abstracts various data sources:

- Market data (prices, volumes, technical indicators)
- News sentiment data
- Economic indicators
- Social media sentiment

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

## Multi-Asset Portfolio Optimization

The project includes a sophisticated multi-asset portfolio optimization capability that uses Modern Portfolio Theory and sentiment analysis to optimize portfolio allocations.

### Features

- **Efficient Frontier Calculation**: Calculates the efficient frontier showing the optimal risk/return tradeoff.
- **Multiple Optimization Objectives**: Supports maximizing Sharpe ratio, minimizing volatility, and maximizing returns.
- **Risk Parity Portfolio**: Implements risk parity allocation where each asset contributes equally to portfolio risk.
- **Sentiment Integration**: Incorporates news and social media sentiment to adjust expected returns.
- **Risk Tolerance Profiles**: Supports conservative, moderate, and aggressive risk profiles.
- **Visualization**: Includes tools to visualize the efficient frontier and compare allocation strategies.

### Example Usage

To optimize a portfolio, you can use the example script:

```bash
python examples/multi_asset_optimization_example.py
```

This will run a multi-asset portfolio optimization for a set of popular stocks, compare different risk profiles, and display visualizations of the results.

### Testing

Comprehensive unit and integration tests have been added to ensure the reliability and correctness of the portfolio optimization functionality.

#### Running Tests

You can run all tests using the test runner script:

```bash
./run_tests.py
```

To run only unit tests:

```bash
./run_tests.py --unit
```

To run only integration tests:

```bash
./run_tests.py --integration
```

#### Test Coverage

The tests cover:

1. **Unit Tests**:
   - PortfolioOptimizer class methods
   - Different optimization objectives
   - Sentiment data integration
   - Constraint handling
   - Edge cases

2. **Integration Tests**:
   - End-to-end portfolio optimization workflow
   - Real data retrieval and processing
   - Risk parity portfolio calculation
   - Efficient frontier generation
   - Sentiment integration workflow

### Dependencies

The portfolio optimization functionality requires the following packages:
- pandas
- numpy
- scipy (for optimization)
- matplotlib (for visualization)

These dependencies are included in the project's requirements.txt file.