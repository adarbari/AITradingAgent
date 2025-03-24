#!/usr/bin/env python3
"""
Test runner script for the AI Trading Agent project.
Provides different command-line options for running tests.
"""
import os
import sys
import argparse
import subprocess


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Run tests for the AI Trading Agent project')
    
    parser.add_argument('--unit', action='store_true', help='Run unit tests only')
    parser.add_argument('--integration', action='store_true', help='Run integration tests only')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('--coverage', action='store_true', help='Generate code coverage report')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--file', help='Run tests from a specific file')
    parser.add_argument('--test', help='Run a specific test (format: TestClass.test_method)')
    parser.add_argument('--skip-slow', action='store_true', help='Skip slow tests')
    
    return parser.parse_args()


def run_tests(args):
    """Run tests based on the command-line arguments."""
    cmd = ['python3', '-m', 'pytest']
    
    # Set verbosity
    if args.verbose:
        cmd.append('-v')
    
    # Add coverage if requested
    if args.coverage:
        cmd.extend(['--cov=src', '--cov-report=term', '--cov-report=html'])
    
    # Add test selection based on arguments
    if args.unit:
        cmd.append('tests/unit')
    elif args.integration:
        cmd.append('tests/integration')
    elif args.file:
        cmd.append(args.file)
    elif args.test:
        # For specific test, format is 'TestClass.test_method'
        if '.' in args.test:
            class_name, method_name = args.test.split('.')
            cmd.append(f'tests/unit -k "{class_name} and {method_name}"')
        else:
            cmd.append(f'-k {args.test}')
    else:
        # Default to all tests if nothing specific is provided
        cmd.append('tests/')
    
    # Skip slow tests if requested
    if args.skip_slow:
        cmd.append('-m "not slow"')
    
    # Join the command parts
    cmd_str = ' '.join(cmd)
    
    # Print what we're running
    print(f"Running: {cmd_str}\n")
    
    # Execute the tests
    return subprocess.call(cmd_str, shell=True)


def main():
    """Main function."""
    args = parse_args()
    
    # Ensure pytest and required packages are installed
    try:
        import pytest
        import pytest_cov
    except ImportError:
        print("Error: pytest or pytest-cov not installed.")
        print("Install with: pip3 install pytest pytest-cov pytest-mock")
        sys.exit(1)
    
    # Run the tests
    return run_tests(args)


if __name__ == '__main__':
    sys.exit(main()) 