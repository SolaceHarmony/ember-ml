#!/usr/bin/env python
"""
Test the purified random operations benchmark script.

This test ensures that the benchmark script runs correctly after our changes
to replace Python operators with appropriate comments and documentation.
"""

import os
import sys
import pytest
from unittest.mock import patch
import importlib.util

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the benchmark script
import ember_ml.benchmarks.compare_random_ops_purified as benchmark_module

class TestCompareRandomOpsPurified:
    """Test the purified random operations benchmark script."""

    def test_time_function(self):
        """Test the time_function function."""
        # Define a simple function to time
        def dummy_function(x):
            return x * 2
        
        # Call the time_function
        import sys
        import os
        print(f"Current working directory: {os.getcwd()}")
        with open(os.path.join(os.getcwd(), "benchmark_module_dir.txt"), "w") as f:
            sys.stdout = f
            print(dir(benchmark_module))
        sys.stdout = sys.__stdout__
        elapsed_time, result = benchmark_module.time_function(dummy_function, 5)
        
        # Check that the result is correct
        assert result == 10
        
        # Check that the elapsed time is a float and greater than zero
        assert isinstance(elapsed_time, float)
        assert elapsed_time >= 0

    def test_benchmark_function(self):
        """Test the benchmark_function function."""
        # Define a simple function to benchmark
        def dummy_function(x):
            return x * 2
        
        # Call the benchmark_function
        avg_time, result = benchmark_module.benchmark_function(dummy_function, 3, 5)
        
        # Check that the result is correct
        assert result == 10
        
        # Check that the average time is a float and greater than zero
        assert isinstance(avg_time, float)
        assert avg_time >= 0

    def test_calculate_statistics(self):
        """Test the calculate_statistics function."""
        from ember_ml import ops
        
        # Create a simple tensor
        tensor = ops.ones((10,))
        
        # Call the calculate_statistics function
        mean, std, min_val, max_val = benchmark_module.calculate_statistics(tensor)
        
        # Check that the statistics are correct
        assert pytest.approx(mean, abs=1e-5) == 1.0
        assert pytest.approx(std, abs=1e-5) == 0.0
        assert pytest.approx(min_val, abs=1e-5) == 1.0
        assert pytest.approx(max_val, abs=1e-5) == 1.0

    @patch('matplotlib.pyplot.savefig')  # Mock the savefig function to avoid creating files
    @patch('matplotlib.pyplot.close')    # Mock the close function
    def test_plot_histograms(self, mock_close, mock_savefig):
        """Test the plot_histograms function."""
        # Create a simple backend results dictionary
        backend_results = {
            'numpy': {
                'normal': {
                    'result': [0.1, 0.2, 0.3, 0.4, 0.5],
                    'mean': 0.3,
                    'std': 0.15
                }
            }
        }
        
        # Call the plot_histograms function
        benchmark_module.plot_histograms(backend_results, 'normal')
        
        # Check that savefig was called
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()

    @patch('os.makedirs')  # Mock makedirs to avoid creating directories
    @patch('builtins.print')  # Mock print to avoid output
    def test_main_function_imports(self, mock_print, mock_makedirs):
        """Test that the main function can be imported without errors."""
        # Just check that the main function exists and is callable
        assert callable(benchmark_module.main)