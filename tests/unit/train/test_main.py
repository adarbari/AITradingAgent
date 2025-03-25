#!/usr/bin/env python3
"""
Tests for the src/train/__main__.py entry point
"""

def test_main_module_import():
    """Test that the main module can be imported without errors"""
    # This test merely checks if the module can be imported without errors
    try:
        import src.train.__main__
        assert True
    except ImportError:
        assert False, "Failed to import src.train.__main__"

def test_main_module_content():
    """Test that the main module contains the expected content"""
    import src.train.__main__
    import inspect
    content = inspect.getsource(src.train.__main__)
    # Check that the module imports main from cli
    assert "from .cli import main" in content
    # Check that the module has a main block
    assert "if __name__ == \"__main__\"" in content
    # Check that the module calls main in the main block
    assert "main()" in content 