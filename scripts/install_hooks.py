#!/usr/bin/env python3
"""
Script to install Git hooks for the AITradingAgent project.
This script installs pre-commit hooks and a post-commit hook that suggests
generating tests for new or modified Python files.
"""
import os
import stat
import shutil
import subprocess
from pathlib import Path

# Define the Git hooks directory
HOOKS_DIR = Path('.git/hooks')

# Content for post-commit hook
POST_COMMIT_HOOK = """#!/bin/bash
# Post-commit hook to suggest generating tests for new Python files

# Get list of added or modified Python files that aren't test files
files=$(git diff --name-only HEAD HEAD~1 | grep '\.py$' | grep -v 'test_' | grep -v '__pycache__')

# Check if any Python files were changed
if [ -n "$files" ]; then
    echo ""
    echo "======================================================"
    echo "The following Python files were changed:"
    echo "$files"
    echo ""
    echo "Consider generating tests for these files using:"
    echo ""
    
    # For each file, suggest generating tests if it's in src/ and doesn't have tests
    for file in $files; do
        if [[ $file == src/* ]]; then
            test_file="tests/unit/${file#src/}"
            test_file="${test_file%.*}"
            test_dir=$(dirname "tests/unit/${file#src/}")
            test_file="$test_dir/test_$(basename "$file")"
            
            # Check if a test file already exists
            if [ ! -f "$test_file" ]; then
                echo "python scripts/generate_test.py $file"
            fi
        fi
    done
    
    echo "======================================================"
    echo ""
fi
"""

def install_hooks():
    """Install Git hooks for the project."""
    # Create hooks directory if it doesn't exist
    if not HOOKS_DIR.exists():
        print(f"Creating hooks directory: {HOOKS_DIR}")
        HOOKS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Write post-commit hook
    post_commit_path = HOOKS_DIR / 'post-commit'
    with open(post_commit_path, 'w') as f:
        f.write(POST_COMMIT_HOOK)
    
    # Make post-commit hook executable
    post_commit_path.chmod(post_commit_path.stat().st_mode | stat.S_IEXEC)
    print(f"Installed post-commit hook to {post_commit_path}")
    
    # Check if pre-commit is installed
    try:
        subprocess.run(['pre-commit', '--version'], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Installing pre-commit...")
        subprocess.run(['pip', 'install', 'pre-commit'], check=True)
    
    # Install pre-commit hooks
    print("Installing pre-commit hooks...")
    subprocess.run(['pre-commit', 'install'], check=True)
    
    print("\nGit hooks installed successfully!")
    print("Pre-commit hooks will run unit tests before each commit.")
    print("Post-commit hook will suggest generating tests for new Python files.")
    print("\nNote: You can bypass pre-commit hooks with git commit --no-verify")

if __name__ == "__main__":
    install_hooks() 