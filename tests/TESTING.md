# Testing the AI Trading Agent

This document describes the test suite for the AI Trading Agent project.

## Test Structure

The test structure follows standard practices for Python projects:

- `tests/unit/`: Unit tests for individual components
  - `tests/unit/feature_engineering/`: Unit tests for the feature engineering module
  - `tests/unit/models/`: Unit tests for ML models
  - `tests/unit/train/`: Unit tests for training pipelines
  - `tests/unit/backtest/`: Unit tests for backtesting logic
  - `tests/unit/agent/`: Unit tests for the agent implementation
  - `tests/unit/utils/`: Unit tests for utility functions
  - `tests/unit/data/`: Unit tests for data handling
  - `tests/unit/scripts/`: Unit tests for scripts

- `tests/integration/`: Integration tests for combined components
  - Contains tests that validate interactions between multiple components
  - Tests real-world usage scenarios on multiple components

- `tests/test_feature_engineering.py`: Top-level feature engineering test suite
- `tests/conftest.py`: Shared fixtures for tests
- `tests/run_tests.py`: Script to run tests with different configurations

## Feature Engineering Tests

The feature engineering module has comprehensive test coverage:

### Unit Tests (`tests/unit/feature_engineering/`)

- **Registry Tests** (`test_registry.py`): Tests for the feature registry system
  - Feature registration and retrieval
  - Category management
  - Metadata handling
  - Feature computation

- **Pipeline Tests** (`test_pipeline.py`): Tests for the feature processing pipeline
  - Feature validation
  - Feature generation
  - Normalization methods (z-score, min-max, robust)
  - Feature count handling
  - Data cleanup

- **Cache Tests** (`test_cache.py`): Tests for the feature caching system
  - Cache key generation
  - Save and load operations
  - Cache invalidation and expiration
  - Statistics tracking

- **Config Tests** (`test_config.py`): Tests for feature configurations
  - Version management
  - Config serialization
  - Predefined feature sets (minimal, standard, advanced)

- **Feature Implementation Tests** (`test_features.py`): Tests for specific feature implementations
  - Price features
  - Volume features
  - Momentum features
  - Trend features
  - Volatility features
  - Seasonal features

### Integration Tests (`tests/integration/test_feature_engineering_integration.py`)

The integration tests validate the complete feature engineering pipeline within the broader context of the trading agent:

- End-to-end pipeline tests
- Feature category coverage tests
- Data format flexibility tests
- Trading environment integration
- Model training integration
- Performance testing
- Multi-stock processing

## Running Tests

You can run tests using the `run_tests.py` script:

```bash
# Run all tests
python tests/run_tests.py

# Run only unit tests
python tests/run_tests.py --unit

# Run only integration tests
python tests/run_tests.py --integration

# Run with verbose output
python tests/run_tests.py --verbose

# Run tests matching a pattern
python tests/run_tests.py --pattern test_registry

# Run a specific test file or directory
python tests/run_tests.py --test tests/unit/feature_engineering/test_features.py
```

## Test Philosophy

This project follows a test-driven development (TDD) approach where possible. All new features should be accompanied by appropriate tests. The goal is to have high coverage for critical components, particularly those involving complex logic or calculations.

## Current Status

- Feature Engineering: Comprehensive unit and integration tests
- Training and Backtesting: Functional tests covering basic workflow
- Model Components: Core functionality tests
- Agent: Basic tests for action selection and reward calculation

Some components are still under active development, and their tests may not all pass until implementation is complete.

## Test Data

Tests use a combination of:
- Synthetic data generated for test purposes
- Small sample datasets that ship with the repository
- Mock objects for external dependencies

No real API calls should be made during tests unless explicitly configured to do so.

## Writing New Tests

When adding new functionality, follow these guidelines:

1. Create unit tests for individual functions/methods
2. Create integration tests for component interactions
3. Use fixtures from `conftest.py` where appropriate
4. Ensure tests run quickly and don't depend on external services
5. Test both normal operation and edge cases/error handling 