import os
import sys
import argparse
import unittest
import pytest

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_unit_tests(verbose=False, pattern=None):
    """Run all unit tests"""
    print("Running unit tests...")
    if pattern:
        print(f"Using pattern: {pattern}")
    
    # Discover and run all unit tests in the tests/unit directory
    test_loader = unittest.TestLoader()
    
    # Include feature engineering tests in test discovery
    unit_test_directories = [
        os.path.join(os.path.dirname(__file__), 'unit'),
        os.path.join(os.path.dirname(__file__), 'unit', 'feature_engineering')
    ]
    
    test_suite = unittest.TestSuite()
    for directory in unit_test_directories:
        if os.path.exists(directory):
            suite = test_loader.discover(directory, pattern=pattern or 'test_*.py')
            test_suite.addTest(suite)
    
    # Also include the main feature_engineering test file in the tests directory
    if pattern is None or 'test_feature_engineering.py' in pattern:
        feature_test_path = os.path.join(os.path.dirname(__file__), 'test_feature_engineering.py')
        if os.path.exists(feature_test_path):
            suite = test_loader.discover(os.path.dirname(__file__), pattern='test_feature_engineering.py')
            test_suite.addTest(suite)
    
    test_runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    result = test_runner.run(test_suite)
    return result.wasSuccessful()


def run_integration_tests(verbose=False, pattern=None):
    """Run all integration tests"""
    print("Running integration tests...")
    if pattern:
        print(f"Using pattern: {pattern}")
    
    # Use pytest for integration tests
    args = ['-xvs' if verbose else '-xs']
    integration_dir = os.path.join(os.path.dirname(__file__), 'integration')
    
    if pattern:
        args.append(f"{integration_dir}/{pattern}")
    else:
        args.append(integration_dir)
    
    result = pytest.main(args)
    return result == 0


def run_specific_test(test_path, verbose=False):
    """Run a specific test file or directory"""
    print(f"Running specific test: {test_path}")
    
    # Check if the path exists
    if not os.path.exists(test_path):
        print(f"Error: {test_path} does not exist")
        return False
    
    # If it's a directory, use discovery
    if os.path.isdir(test_path):
        test_loader = unittest.TestLoader()
        test_suite = test_loader.discover(test_path, pattern='test_*.py')
        test_runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
        result = test_runner.run(test_suite)
        return result.wasSuccessful()
    
    # If it's a file, run with pytest
    else:
        args = ['-xvs' if verbose else '-xs', test_path]
        result = pytest.main(args)
        return result == 0


def run_all_tests(verbose=False, pattern=None):
    """Run all tests - unit and integration"""
    unit_success = run_unit_tests(verbose, pattern)
    integration_success = run_integration_tests(verbose, pattern)
    return unit_success and integration_success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run tests for AI Trading Agent')
    parser.add_argument('--unit', action='store_true', help='Run unit tests only')
    parser.add_argument('--integration', action='store_true', help='Run integration tests only')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--pattern', '-p', type=str, help='Test pattern to match')
    parser.add_argument('--test', '-t', type=str, help='Run a specific test file or directory')
    
    args = parser.parse_args()
    
    # If no specific command is provided, run all tests
    run_all = args.all or (not args.unit and not args.integration and not args.test)
    
    success = True
    
    if args.test:
        success = run_specific_test(args.test, args.verbose)
    else:
        if args.unit or run_all:
            success = success and run_unit_tests(args.verbose, args.pattern)
        
        if args.integration or run_all:
            success = success and run_integration_tests(args.verbose, args.pattern)
    
    # Exit with appropriate status
    sys.exit(0 if success else 1) 