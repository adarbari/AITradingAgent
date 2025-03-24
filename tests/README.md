# AI Trading Agent Test Suite

This directory contains tests for the AI Trading Agent project using pytest.

## Test Structure

- `conftest.py`: Contains shared fixtures and utilities for tests
- `unit/`: Unit tests for individual components
  - `data/`: Tests for data fetching components
  - `models/`: Tests for model training components
  - `agent/`: Tests for the trading environment
  - `backtest/`: Tests for backtesting functionality
- `integration/`: Integration tests for the full pipeline

## Running Tests

You can use the `run_tests.py` script to run tests with various options:

```bash
# Run all tests
python3 ./run_tests.py --all

# Run only unit tests
python3 ./run_tests.py --unit

# Run only integration tests
python3 ./run_tests.py --integration

# Run tests with verbose output
python3 ./run_tests.py --all -v

# Run a specific test file
python3 ./run_tests.py --file tests/unit/data/test_data_fetcher_factory.py

# Skip slow tests
python3 ./run_tests.py --all --skip-slow

# Generate code coverage report
python3 ./run_tests.py --all --coverage
```

## Test-Driven Development

The tests in this suite are designed to support test-driven development (TDD). When implementing new features:

1. Write tests first that define the expected behavior
2. Run the tests to verify they fail
3. Implement the functionality to make the tests pass
4. Refactor the code while ensuring the tests still pass

## Notes on Implementation

Currently, not all tests pass as the implementation is in progress. The following components have passing tests:

- `DataFetcherFactory`
- Basic functionality of `SyntheticDataFetcher`
- Basic initialization of `Backtester` and `ModelTrainer`

The remaining tests are waiting for their corresponding implementations to be completed or updated. 