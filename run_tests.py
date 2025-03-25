#!/usr/bin/env python
"""
Test runner script for portfolio optimization functionality.

This script discovers and runs all unit and integration tests related to
the portfolio optimization functionality. It provides a convenient way to
verify that all tests are passing.
"""
import unittest
import os
import sys

def run_all_tests():
    """Discover and run all tests."""
    # Add project root to sys.path
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(project_root)
    
    # Create test suite for unit tests
    print("Discovering unit tests...")
    unit_test_suite = unittest.defaultTestLoader.discover(
        os.path.join(project_root, 'tests', 'unit', 'agent', 'multi_agent'),
        pattern='test*portfolio*.py'
    )
    
    # Create test suite for integration tests
    print("Discovering integration tests...")
    integration_test_suite = unittest.defaultTestLoader.discover(
        os.path.join(project_root, 'tests', 'integration'),
        pattern='test*portfolio*.py'
    )
    
    # Create a combined test suite
    all_tests = unittest.TestSuite([unit_test_suite, integration_test_suite])
    
    # Create a test runner
    test_runner = unittest.TextTestRunner(verbosity=2)
    
    # Run the tests
    print("\nRunning all portfolio optimization tests...")
    print("=" * 70)
    result = test_runner.run(all_tests)
    
    # Print summary
    print("\nTest Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    # Return True if all tests passed, False otherwise
    return len(result.failures) == 0 and len(result.errors) == 0

def run_unit_tests_only():
    """Run only unit tests."""
    # Add project root to sys.path
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(project_root)
    
    # Create test suite for unit tests
    print("Discovering unit tests...")
    unit_test_suite = unittest.defaultTestLoader.discover(
        os.path.join(project_root, 'tests', 'unit', 'agent', 'multi_agent'),
        pattern='test*portfolio*.py'
    )
    
    # Create a test runner
    test_runner = unittest.TextTestRunner(verbosity=2)
    
    # Run the tests
    print("\nRunning unit tests for portfolio optimization...")
    print("=" * 70)
    result = test_runner.run(unit_test_suite)
    
    # Print summary
    print("\nUnit Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    # Return True if all tests passed, False otherwise
    return len(result.failures) == 0 and len(result.errors) == 0

def run_integration_tests_only():
    """Run only integration tests."""
    # Add project root to sys.path
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(project_root)
    
    # Create test suite for integration tests
    print("Discovering integration tests...")
    integration_test_suite = unittest.defaultTestLoader.discover(
        os.path.join(project_root, 'tests', 'integration'),
        pattern='test*portfolio*.py'
    )
    
    # Create a test runner
    test_runner = unittest.TextTestRunner(verbosity=2)
    
    # Run the tests
    print("\nRunning integration tests for portfolio optimization...")
    print("=" * 70)
    result = test_runner.run(integration_test_suite)
    
    # Print summary
    print("\nIntegration Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    # Return True if all tests passed, False otherwise
    return len(result.failures) == 0 and len(result.errors) == 0

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Run portfolio optimization tests")
    parser.add_argument('--unit', action='store_true', help='Run only unit tests')
    parser.add_argument('--integration', action='store_true', help='Run only integration tests')
    args = parser.parse_args()
    
    if args.unit:
        success = run_unit_tests_only()
    elif args.integration:
        success = run_integration_tests_only()
    else:
        success = run_all_tests()
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1) 