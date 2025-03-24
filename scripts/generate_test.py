#!/usr/bin/env python3
"""
Script to generate test files for new code.
Usage:
    python3 generate_test.py <path_to_file>
Example:
    python3 generate_test.py src/models/trainer.py
"""
import os
import sys
import re
from pathlib import Path

TEST_TEMPLATES = {
    "class": '''"""
Tests for the {class_name} class
"""
import pytest
from unittest.mock import MagicMock, patch

from {module_path} import {class_name}


class Test{class_name}:
    """Test cases for the {class_name} class"""

    def test_initialization(self):
        """Test initialization of the {class_name}"""
        # Implement test here
        pass
{additional_methods}
''',

    "method": '''    def test_{method_name}(self):
        """Test {method_description}"""
        # Implement test here
        pass
''',

    "function": '''"""
Tests for the {function_name} function
"""
import pytest
from unittest.mock import MagicMock, patch

from {module_path} import {function_name}


def test_{function_name}():
    """Test the {function_name} function"""
    # Implement test here
    pass
'''
}


def extract_classes_and_methods(file_path):
    """Extract classes and methods from a Python file."""
    with open(file_path, 'r') as f:
        content = f.read()

    # Extract module docstring
    module_docstring = re.search(r'^"""(.+?)"""', content, re.DOTALL)
    module_description = module_docstring.group(1).strip() if module_docstring else ""

    # Find all class definitions
    class_pattern = r'class\s+(\w+)(?:\(.*?\))?:'
    classes = re.findall(class_pattern, content)
    
    # Find all method definitions within classes
    methods_by_class = {}
    for class_name in classes:
        # Find class content - look for the class definition until the next class or end of file
        class_pattern = rf'class\s+{class_name}(?:\(.*?\))?:(.*?)(?:(?:^class\s+)|(?:\Z))'
        class_match = re.search(class_pattern, content, re.DOTALL | re.MULTILINE)
        
        if class_match:
            class_content = class_match.group(1)
            # Find methods with proper indentation
            method_pattern = r'    def\s+(\w+)\s*\('
            methods = re.findall(method_pattern, class_content)
            # Exclude __init__, private methods, and methods starting with _
            methods = [m for m in methods if m != '__init__' and not m.startswith('_')]
            methods_by_class[class_name] = methods
    
    # Find standalone functions (not inside a class)
    function_pattern = r'^def\s+(\w+)\s*\('
    functions = re.findall(function_pattern, content, re.MULTILINE)
    
    return {
        "classes": methods_by_class,
        "functions": functions,
        "module_description": module_description
    }


def create_test_file(file_path):
    """Create a test file for the given Python file."""
    file_path = Path(file_path)
    
    # Extract module path
    relative_path = file_path.relative_to(Path.cwd())
    module_path = str(relative_path).replace('.py', '').replace('/', '.')
    
    # Determine test file path
    if str(relative_path).startswith('src/'):
        # For src files, place tests in tests/unit
        parts = relative_path.parts[1:]  # Remove 'src'
        test_dir = Path('tests/unit').joinpath(*parts[:-1])
        test_file = test_dir / f"test_{parts[-1]}"
    else:
        # For other files, place tests in the same directory with test_ prefix
        test_dir = file_path.parent
        test_file = test_dir / f"test_{file_path.name}"
    
    # Create test directory if it doesn't exist
    os.makedirs(test_dir, exist_ok=True)
    
    # Extract classes and functions
    extracted = extract_classes_and_methods(file_path)
    
    # Generate test content
    test_content = []
    
    # Generate class tests
    for class_name, methods in extracted["classes"].items():
        additional_methods = ""
        for method in methods:
            method_description = method.replace('_', ' ')
            additional_methods += TEST_TEMPLATES["method"].format(
                method_name=method,
                method_description=method_description
            )
        
        test_content.append(TEST_TEMPLATES["class"].format(
            class_name=class_name,
            module_path=module_path,
            additional_methods=additional_methods
        ))
    
    # Generate function tests
    for function_name in extracted["functions"]:
        test_content.append(TEST_TEMPLATES["function"].format(
            function_name=function_name,
            module_path=module_path
        ))
    
    # Write the test file
    with open(test_file, 'w') as f:
        f.write('\n'.join(test_content))
    
    print(f"Created test file: {test_file}")
    return test_file


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 generate_test.py <path_to_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    if not os.path.isfile(file_path) or not file_path.endswith('.py'):
        print(f"Error: {file_path} is not a Python file")
        sys.exit(1)
    
    test_file = create_test_file(file_path) 