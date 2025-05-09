# AI Trading Agent - Rules for AI Assistance

When providing assistance with this project, please adhere to the following guidelines:

## 1. Code Organization Rules

- **Maintain Modular Structure**: Keep the 6 main modules separate - data, models, backtest, agent, utils, and scripts.
- **Respect Interface Boundaries**: Any new implementation must adhere to its respective base interface.
- **Package Organization**: New files should be placed in the appropriate module directory with proper imports in the `__init__.py` file.
- **Single Responsibility**: Each class should have a focused purpose; avoid creating overly complex classes.
- **Script Organization**: All executable scripts should be placed in the `src/scripts/` directory.

## 2. Interface Implementation Rules

- **Implement All Abstract Methods**: When creating a new class that implements a base interface, all abstract methods must be implemented.
- **Maintain Method Signatures**: Keep the same parameter structure for interface implementations, using `**kwargs` for extensions.
- **Error Handling**: Follow the established error handling pattern, returning `None` for failures with appropriate messaging.
- **Documentation**: All methods need proper docstrings following the established format.

## 3. Code Extension Guidelines

- **New Data Sources**: 
  - Must implement `BaseDataFetcher` with all abstract methods
  - Should be registered in `DataFetcherFactory`
  - Handle connection and data format errors gracefully

- **New Model Types**:
  - Must implement `BaseModelTrainer` with all abstract methods
  - Follow the same load/save model approach
  - Handle model training failures gracefully

- **New Backtesting Strategies**:
  - Must implement `BaseBacktester` with all abstract methods
  - Maintain the same result format for compatibility

- **New Trading Environments**:
  - Must implement `BaseTradingEnvironment`
  - Need to conform to the Gym API requirements
  - Should handle boundary conditions (beginning/end of data)

- **New Scripts**:
  - Place in the `src/scripts/` directory
  - Follow consistent argument parsing patterns
  - Include proper error handling and logging
  - Provide clear usage instructions in docstrings
  - Reuse existing components through proper imports

## 4. Design Principles to Follow

- **Dependency Inversion**: High-level modules should depend on abstractions, not concrete implementations.
- **Factory Pattern**: Use factories for creating concrete implementations.
- **Composition Over Inheritance**: Prefer composing objects rather than deep inheritance hierarchies.
- **Parameter Flexibility**: Allow for parameter customization through `**kwargs` while providing sensible defaults.

## 5. Specific Coding Patterns

### Error Handling Pattern
```python
try:
    # Operation that might fail
    result = perform_operation()
    return result
except Exception as e:
    print(f"Error occurred: {e}")
    if self.verbose >= 2:
        traceback.print_exc()
    return None
```

### Interface Implementation Pattern
```python
from .base_something import BaseSomething

class ConcreteSomething(BaseSomething):
    def __init__(self, specific_param, **kwargs):
        self.specific_param = specific_param
        
    def interface_method(self, param1, **kwargs):
        """
        Implementation of interface_method
        
        Args:
            param1: Description
            **kwargs: Additional parameters
            
        Returns:
            Result description
        """
        # Implementation
```

### Factory Pattern
```python
@staticmethod
def create_something(type_name, **kwargs):
    if type_name == "type1":
        return Type1Something(**kwargs)
    elif type_name == "type2":
        return Type2Something(**kwargs)
    else:
        print(f"Unknown type: {type_name}")
        return None
```

### Script Pattern
```python
#!/usr/bin/env python3
"""
Description of what the script does.
"""
import argparse
import os
import sys

# Add project root to sys.path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def main():
    """Main function that handles the workflow of the script."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Script description")
    parser.add_argument("--some-arg", type=str, default="default", help="Description of argument")
    args = parser.parse_args()
    
    # Script logic here
    
    # Handle errors and provide feedback
    print("Operation completed successfully")

if __name__ == "__main__":
    main()
```

## 6. Testing Approach

- **Always Add Tests**: Every new class or module must have corresponding test files
  - Place unit tests in `tests/unit/[module_name]/`
  - Place script tests in `tests/unit/scripts/`
  - Name test files with `test_` prefix matching the file being tested 
  - Cover all public methods and edge cases
  - Target minimum 80% code coverage
  
- **Test Structure and Fixtures**:
  - Use pytest fixtures for shared test data
  - Test for success cases, edge cases, and failure modes
  - Isolate tests with proper mocking when needed

- **Automated Test Execution**:
  - Run tests after every significant code change
  - Execute with `python run_tests.py` before committing
  - Tests must pass before considering the code complete

- **Test First Development**:
  - When possible, write test cases before implementing features
  - Use `scripts/generate_test.py` to generate test templates
  - Fill in test implementations to validate requirements

- Always run the main script after making changes to verify functionality
- Test with synthetic data first before testing with real data sources
- Provide a concrete example of how to use any new functionality you add

## 7. Command Line Interface

- Maintain the existing command-line argument structure
- If adding new options, follow the same argparse pattern
- Provide default values for new options
- For scripts in `src/scripts/`, ensure consistent argument naming across different scripts
- Document all command-line arguments in the script's docstring

By following these guidelines, you'll help maintain the structural integrity and design philosophy of the project. 