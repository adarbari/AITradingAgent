# AI Trading Agent - Rules for AI Assistance

When providing assistance with this project, please adhere to the following guidelines:

## 1. Code Organization Rules

- **Maintain Modular Structure**: Keep the 5 main modules separate - data, models, backtest, agent, and utils.
- **Respect Interface Boundaries**: Any new implementation must adhere to its respective base interface.
- **Package Organization**: New files should be placed in the appropriate module directory with proper imports in the `__init__.py` file.
- **Single Responsibility**: Each class should have a focused purpose; avoid creating overly complex classes.

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

## 6. Testing Approach

- Always run the main script after making changes to verify functionality
- Test with synthetic data first before testing with real data sources
- Provide a concrete example of how to use any new functionality you add

## 7. Command Line Interface

- Maintain the existing command-line argument structure
- If adding new options, follow the same argparse pattern
- Provide default values for new options

By following these guidelines, you'll help maintain the structural integrity and design philosophy of the project. 