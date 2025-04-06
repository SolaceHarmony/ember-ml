#!/usr/bin/env python
"""
Test the NumPy usage detection tool.

This test ensures that the detection tool correctly identifies NumPy usage,
precision-reducing casts, tensor conversions, and Python operators.
"""

import os
import sys
import pytest
import tempfile
from unittest.mock import patch
import importlib.util

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the detection tool
spec = importlib.util.spec_from_file_location(
    "detect_numpy_usage", 
    os.path.join(os.path.dirname(__file__), "..", "utils", "detect_numpy_usage.py")
)
detect_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(detect_module)

class TestDetectNumpyUsage:
    """Test the NumPy usage detection tool."""

    def test_find_python_files(self):
        """Test the find_python_files function."""
        # Create a temporary directory with some Python files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create some Python files
            with open(os.path.join(temp_dir, "file1.py"), "w") as f:
                f.write("# Test file 1")
            with open(os.path.join(temp_dir, "file2.py"), "w") as f:
                f.write("# Test file 2")
            # Create a subdirectory with a Python file
            os.makedirs(os.path.join(temp_dir, "subdir"))
            with open(os.path.join(temp_dir, "subdir", "file3.py"), "w") as f:
                f.write("# Test file 3")
            
            # Call the find_python_files function
            python_files = detect_module.find_python_files(temp_dir)
            
            # Check that all Python files were found
            assert len(python_files) == 3
            assert any(f.endswith("file1.py") for f in python_files)
            assert any(f.endswith("file2.py") for f in python_files)
            assert any(f.endswith("file3.py") for f in python_files)

    def test_check_numpy_import(self):
        """Test the check_numpy_import function."""
        # Create a temporary file with NumPy imports
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w+") as temp_file:
            temp_file.write("""
import numpy as np
from numpy import array, zeros
import numpy
            """)
            temp_file.flush()
            
            # Call the check_numpy_import function
            has_numpy, numpy_imports = detect_module.check_numpy_import(temp_file.name)
            
            # Check that NumPy imports were detected
            assert has_numpy
            assert len(numpy_imports) == 4
            assert "np" in numpy_imports
            assert "array" in numpy_imports
            assert "zeros" in numpy_imports
            assert "numpy" in numpy_imports

    def test_check_ast_for_issues(self):
        """Test the check_ast_for_issues function."""
        # Create a temporary file with various issues
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w+") as temp_file:
            temp_file.write("""
import numpy as np

def test_function():
    # NumPy usage
    arr = np.array([1, 2, 3])
    
    # Precision-reducing cast
    x = float(arr[0])
    
    # Tensor conversion
    numpy_arr = arr.numpy()
    
    # Python operators
    result = arr + 1
    result = arr - 1
    result = arr * 2
    result = arr / 2
    
    return result
            """)
            temp_file.flush()
            
            # Call the check_ast_for_issues function
            has_numpy, numpy_imports, numpy_usages, precision_casts, tensor_conversions, python_operators = detect_module.check_ast_for_issues(temp_file.name)
            
            # Check that all issues were detected
            assert has_numpy
            assert len(numpy_imports) >= 1  # May detect multiple imports
            assert len(numpy_usages) >= 1
            assert len(precision_casts) == 1
            assert len(tensor_conversions) >= 1  # May detect multiple conversions
            assert len(python_operators) == 4  # +, -, *, /

    def test_analyze_file(self):
        """Test the analyze_file function."""
        # Create a temporary file with various issues
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w+") as temp_file:
            temp_file.write("""
import numpy as np

def test_function():
    # NumPy usage
    arr = np.array([1, 2, 3])
    
    # Precision-reducing cast
    x = float(arr[0])
    
    # Tensor conversion
    numpy_arr = arr.numpy()
    
    # Python operators
    result = arr + 1
    
    return result
            """)
            temp_file.flush()
            
            # Call the analyze_file function
            result = detect_module.analyze_file(temp_file.name)
            
            # Check that all issues were detected
            assert result["has_numpy"]
            assert len(result["imports"]) >= 1  # May detect multiple imports
            assert len(result["usages"]) >= 1
            assert len(result["precision_casts"]) == 1
            assert len(result["tensor_conversions"]) >= 1  # May detect multiple conversions
            assert len(result["python_operators"]) == 1

    @patch('builtins.print')  # Mock print to avoid output
    def test_print_results(self, mock_print):
        """Test the print_results function."""
        # Create a simple results list
        results = [
            {
                "file": "file1.py",
                "has_numpy": True,
                "imports": ["np"],
                "usages": ["np.array"],
                "precision_casts": [{"type": "float", "location": "test_function:5", "line": 5}],
                "tensor_conversions": [{"type": "tensor.numpy()", "location": "test_function:8", "line": 8}],
                "python_operators": [{"type": "+", "location": "test_function:11", "line": 11}]
            }
        ]
        
        # Call the print_results function with different show flags
        detect_module.print_results(results, verbose=True, show_numpy=True, show_precision=False, 
                                   show_conversion=False, show_operators=False)
        detect_module.print_results(results, verbose=True, show_numpy=False, show_precision=True, 
                                   show_conversion=False, show_operators=False)
        detect_module.print_results(results, verbose=True, show_numpy=False, show_precision=False, 
                                   show_conversion=True, show_operators=False)
        detect_module.print_results(results, verbose=True, show_numpy=False, show_precision=False, 
                                   show_conversion=False, show_operators=True)
        
        # Check that print was called multiple times
        assert mock_print.call_count > 10

    @patch('builtins.print')  # Mock print to avoid output
    def test_main_function_imports(self, mock_print):
        """Test that the main function can be imported without errors."""
        # Just check that the main function exists and is callable
        assert callable(detect_module.main)