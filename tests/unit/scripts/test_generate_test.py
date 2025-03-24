"""
Tests for the generate_test.py script
"""
import os
import sys
import pytest
import tempfile
from unittest.mock import patch, mock_open
from pathlib import Path

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / 'scripts'))

from generate_test import extract_classes_and_methods, create_test_file


class TestGenerateTest:
    """Test cases for the generate_test.py script"""

    def test_extract_classes_and_methods(self):
        """Test extracting classes and methods from a Python file"""
        test_content = '''"""
Sample module
"""

class TestClass:
    """Test class docstring"""
    
    def __init__(self, param):
        self.param = param
    
    def method1(self, arg):
        """Method 1 docstring"""
        return arg
    
    def _private_method(self):
        """Private method docstring"""
        pass
    
    def method2(self):
        """Method 2 docstring"""
        return self.param


def standalone_function():
    """Standalone function docstring"""
    return True
'''
        
        # Mock open to return our test content
        with patch('builtins.open', mock_open(read_data=test_content)):
            result = extract_classes_and_methods('dummy_path.py')
        
        # Check results
        assert 'classes' in result
        assert 'functions' in result
        assert 'module_description' in result
        
        # Check classes and methods
        assert 'TestClass' in result['classes']
        methods = result['classes']['TestClass']
        assert 'method1' in methods
        assert 'method2' in methods
        assert '_private_method' not in methods  # Private methods should be excluded
        
        # Check functions
        assert 'standalone_function' in result['functions']
        
        # Check module description
        assert result['module_description'] == 'Sample module'
    
    def test_create_test_file(self):
        """Test creating a test file"""
        # Create a temporary directory and file
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a mock src directory structure
            src_dir = os.path.join(temp_dir, 'src', 'package')
            os.makedirs(src_dir, exist_ok=True)
            
            # Create a test module in src directory
            test_file_path = os.path.join(src_dir, 'test_module.py')
            with open(test_file_path, 'w') as f:
                f.write('''"""
Sample module
"""

class TestClass:
    def method1(self):
        pass
''')
            
            # Mock path.relative_to and cwd
            with patch('pathlib.Path.relative_to', return_value=Path('src/package/test_module.py')):
                with patch('pathlib.Path.cwd', return_value=Path(temp_dir)):
                    with patch('os.makedirs') as mock_makedirs:
                        with patch('builtins.open', mock_open()) as mock_file:
                            create_test_file(test_file_path)
            
            # Check that the appropriate directories would be created
            mock_makedirs.assert_called()
            
            # Check that the file write was called
            mock_file.assert_called() 