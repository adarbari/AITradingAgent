# Testing Guidelines for AITradingAgent

This document outlines the testing requirements and best practices for the AITradingAgent project.

## Testing Requirements

### Code Coverage

- **Aim for at least 80% code coverage** for new code
- All critical paths should be tested
- Complex decision logic should have tests for each branch

### Types of Tests Required

1. **Unit Tests**
   - All classes and functions should have unit tests
   - Test each component in isolation
   - Mock external dependencies

2. **Integration Tests**
   - Test components working together
   - End-to-end workflows should be tested
   - Include at least one full pipeline test

3. **Performance Tests** (where applicable)
   - Tests for model performance metrics
   - Tests for execution speed of critical components

## When to Add Tests

- **New Features**: Every new feature must have corresponding test cases
- **Bug Fixes**: Each bug fix must include a test that would have caught the bug
- **Refactoring**: When refactoring code, ensure existing tests pass and add new ones if needed

## Testing Structure

- Unit tests go in `tests/unit/[module_name]/`
- Integration tests go in `tests/integration/`
- Test files should follow the naming convention: `test_[file_being_tested].py`
- Script tests should be in `tests/unit/scripts/` to test functionality in `src/scripts/`

## Script Testing

Scripts in `src/scripts/` should be tested for:

- Command-line argument parsing
- Integration with other components
- Proper error handling
- Expected output formats

Examples of script tests:
- `test_train_and_backtest.py` for testing the training and backtesting script
- `test_compare_models.py` for testing the model comparison functionality

## Test Naming Convention

- Test methods should be named `test_[method_name]_[scenario]`
- Example: `test_buy_insufficient_funds`

## Mocks and Dependencies

- Use `unittest.mock` or `pytest-mock` for mocking
- Inject dependencies to make testing easier
- Mock external APIs and services

## Test Environment

- Tests should be able to run in a CI environment
- No tests should depend on external services unless properly mocked
- Use fixture data instead of real API calls

## When to Run Tests

- **Before Committing**: Run unit tests before each commit
- **Before Creating a PR**: Run all tests (unit and integration)
- **In CI Pipeline**: Tests will run automatically on push and PR

## Troubleshooting Common Test Issues

1. **Flaky Tests**
   - Avoid time-dependent tests
   - Ensure test isolation
   - Clean up resources after tests

2. **Slow Tests**
   - Use appropriate fixtures
   - Mock external dependencies
   - Consider separating slow tests if necessary

3. **Test Dependencies**
   - Tests should be independent
   - Avoid tests that depend on other tests

## Using Test Generation Tool

The project includes a test generation tool to help create test templates:

```bash
python scripts/generate_test.py path/to/your/file.py
```

This will:
- Generate a test file with templates for each class and function
- Place it in the appropriate test directory
- Include basic test structure to get you started

## Code Review Checklist for Tests

Before submitting code, ensure:

- [ ] Tests exist for all new code
- [ ] Tests cover edge cases and error conditions
- [ ] All tests pass
- [ ] Tests are readable and maintainable
- [ ] Test coverage meets the minimum requirement

## Additional Resources

- [pytest Documentation](https://docs.pytest.org/)
- [unittest.mock Documentation](https://docs.python.org/3/library/unittest.mock.html)
- [Python Testing Best Practices](https://docs.python-guide.org/writing/tests/) 