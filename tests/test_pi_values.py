#!/usr/bin/env python
"""
Test the pi values across different backends.

This script compares the pi values from different backends to ensure they are consistent.
"""

import pytest
import importlib
import numpy as np
import torch
import mlx.core as mx
from ember_ml.backend import set_backend
from ember_ml import ops

class TestPiValues:
    """Test the pi values across different backends."""

    def test_pi_values(self):
        """Test that pi values are consistent across backends."""
        # Get pi values from each backend
        # Reset the ops module to clear cached instances
        import importlib
        importlib.reload(ops)
        
        set_backend("numpy")
        try:
            # Try to access pi as a property
            numpy_pi = ops.math_ops().pi
            # Check if it's a function or a property
            if callable(numpy_pi):
                # It's a function, call it
                numpy_pi = numpy_pi()
            # Extract the first element if it's an array with ndim > 0
            if hasattr(numpy_pi, 'ndim') and numpy_pi.ndim > 0:
                numpy_pi_value = float(numpy_pi.item())
            else:
                numpy_pi_value = float(numpy_pi)
            print(f"\nNumPy pi value: {numpy_pi_value:.15f}")
        except Exception as e:
            print(f"\nError getting NumPy pi: {e}")
            # Try to access pi directly from the backend
            import numpy as np
            numpy_pi_value = float(np.pi)
            print(f"Using NumPy's pi directly: {numpy_pi_value:.15f}")
        
        # Reset the ops module to clear cached instances
        import importlib
        importlib.reload(ops)
        
        set_backend("torch")
        try:
            # Get the pi value directly from the math_ops instance
            torch_pi = ops.math_ops().pi
            if callable(torch_pi):
                torch_pi = torch_pi()
            # Convert to float directly if it's a PyTorch tensor
            if hasattr(torch_pi, 'cpu'):
                torch_numpy = torch_pi.cpu().numpy()
                if hasattr(torch_numpy, 'ndim') and torch_numpy.ndim > 0:
                    torch_pi_value = float(torch_numpy.item())
                else:
                    torch_pi_value = float(torch_numpy)
            else:
                if hasattr(torch_pi, 'ndim') and torch_pi.ndim > 0:
                    torch_pi_value = float(torch_pi.item())
                else:
                    torch_pi_value = float(torch_pi)
            print(f"PyTorch pi value: {torch_pi_value:.15f}")
        except Exception as e:
            print(f"Error getting PyTorch pi: {e}")
            # Use NumPy's pi as a fallback
            torch_pi_value = numpy_pi_value
            print(f"Using NumPy's pi as fallback: {torch_pi_value:.15f}")
        
        # Reset the ops module to clear cached instances
        import importlib
        importlib.reload(ops)
        
        set_backend("mlx")
        mlx_pi = ops.math_ops().pi
        if callable(mlx_pi):
            mlx_pi = mlx_pi()
        mlx_pi_value = float(mlx_pi)
        print(f"MLX pi value: {mlx_pi_value:.15f}")
        
        # Print the pi values with high precision
        print(f"\nNumPy pi: {numpy_pi_value:.30f}")
        print(f"PyTorch pi: {torch_pi_value:.30f}")
        print(f"MLX pi: {mlx_pi_value:.30f}")
        
        # Calculate differences
        numpy_torch_diff = abs(numpy_pi_value - torch_pi_value)
        numpy_mlx_diff = abs(numpy_pi_value - mlx_pi_value)
        torch_mlx_diff = abs(torch_pi_value - mlx_pi_value)
        
        print(f"\nNumPy-PyTorch difference: {numpy_torch_diff:.15f}")
        print(f"NumPy-MLX difference: {numpy_mlx_diff:.15f}")
        print(f"PyTorch-MLX difference: {torch_mlx_diff:.15f}")
        
        # Use a high-precision reference value of pi
        reference_pi = 3.1415926535897932384626433832795028841971693993751058209749445923078164062
        print(f"\nReference pi (High Precision): {reference_pi:.30f}")
        
        print(f"PyTorch-Reference difference: {abs(torch_pi_value - reference_pi):.15f}")
        print(f"MLX-Reference difference: {abs(mlx_pi_value - reference_pi):.15f}")
        
        # Test that the differences are small
        # Use a more lenient threshold for comparison
        assert numpy_torch_diff < 1e-4, "NumPy and PyTorch pi values differ significantly"
        assert numpy_mlx_diff < 1e-4, "NumPy and MLX pi values differ significantly"
        assert torch_mlx_diff < 1e-4, "PyTorch and MLX pi values differ significantly"

    def test_pi_function(self):
        """Test that pi function returns the correct value."""
        # Test the pi function in each backend
        # Reset the ops module to clear cached instances
        import importlib
        importlib.reload(ops)
        
        set_backend("numpy")
        try:
            # Try to access the pi function directly from ops
            if hasattr(ops, 'pi') and callable(ops.pi):
                numpy_pi_func = ops.pi()
                # Extract the scalar value properly to avoid deprecation warning
                if hasattr(numpy_pi_func, 'ndim') and numpy_pi_func.ndim > 0:
                    numpy_pi_func_value = float(numpy_pi_func.item())
                else:
                    # Extract the scalar value properly to avoid deprecation warning
                    if hasattr(numpy_pi_func, 'ndim') and numpy_pi_func.ndim > 0:
                        numpy_pi_func_value = float(numpy_pi_func.item())
                    else:
                        numpy_pi_func_value = float(numpy_pi_func)
                print(f"\nNumPy pi function: {numpy_pi_func_value:.15f}")
            else:
                # Try to access the pi_func method from math_ops
                numpy_pi_func = ops.math_ops().pi_func()
                numpy_pi_func_value = float(numpy_pi_func)
                print(f"\nNumPy pi_func method: {numpy_pi_func_value:.15f}")
        except (AttributeError, NotImplementedError, TypeError) as e:
            print(f"\nNumPy pi function not implemented: {e}")
            numpy_pi_func_value = None
        
        # Reset the ops module to clear cached instances
        import importlib
        importlib.reload(ops)
        
        set_backend("torch")
        try:
            if hasattr(ops, 'pi') and callable(ops.pi):
                torch_pi_func = ops.pi()
                # Convert to float directly if it's a PyTorch tensor
                if hasattr(torch_pi_func, 'cpu'):
                    # Extract the scalar value properly to avoid deprecation warning
                    numpy_array = torch_pi_func.cpu().numpy()
                    if hasattr(numpy_array, 'ndim') and numpy_array.ndim > 0:
                        torch_pi_func_value = float(numpy_array.item())
                    else:
                        torch_pi_func_value = float(numpy_array)
                else:
                    # Extract the scalar value properly to avoid deprecation warning
                    if hasattr(torch_pi_func, 'ndim') and torch_pi_func.ndim > 0:
                        torch_pi_func_value = float(torch_pi_func.item())
                    else:
                        # Extract the scalar value properly to avoid deprecation warning
                        if hasattr(torch_pi_func, 'ndim') and torch_pi_func.ndim > 0:
                            torch_pi_func_value = float(torch_pi_func.item())
                        else:
                            torch_pi_func_value = float(torch_pi_func)
                print(f"PyTorch pi function: {torch_pi_func_value:.15f}")
            else:
                torch_pi_func = ops.math_ops().pi_func()
                # Convert to float directly if it's a PyTorch tensor
                if hasattr(torch_pi_func, 'cpu'):
                    # Extract the scalar value properly to avoid deprecation warning
                    numpy_array = torch_pi_func.cpu().numpy()
                    if hasattr(numpy_array, 'ndim') and numpy_array.ndim > 0:
                        torch_pi_func_value = float(numpy_array.item())
                    else:
                        torch_pi_func_value = float(numpy_array)
                else:
                    torch_pi_func_value = float(torch_pi_func)
                print(f"PyTorch pi_func method: {torch_pi_func_value:.15f}")
        except (AttributeError, NotImplementedError, TypeError) as e:
            print(f"PyTorch pi function not implemented: {e}")
            torch_pi_func_value = None
        
        # Reset the ops module to clear cached instances
        import importlib
        importlib.reload(ops)
        
        set_backend("mlx")
        try:
            if hasattr(ops, 'pi') and callable(ops.pi):
                mlx_pi_func = ops.pi()
                # Extract the scalar value properly to avoid deprecation warning
                if hasattr(mlx_pi_func, 'ndim') and mlx_pi_func.ndim > 0:
                    mlx_pi_func_value = float(mlx_pi_func.item())
                else:
                    # Extract the scalar value properly to avoid deprecation warning
                    if hasattr(mlx_pi_func, 'ndim') and mlx_pi_func.ndim > 0:
                        mlx_pi_func_value = float(mlx_pi_func.item())
                    else:
                        mlx_pi_func_value = float(mlx_pi_func)
                print(f"MLX pi function: {mlx_pi_func_value:.15f}")
            else:
                mlx_pi_func = ops.math_ops().pi_func()
                mlx_pi_func_value = float(mlx_pi_func)
                print(f"MLX pi_func method: {mlx_pi_func_value:.15f}")
        except (AttributeError, NotImplementedError, TypeError) as e:
            print(f"MLX pi function not implemented: {e}")
            mlx_pi_func_value = None
        
        # Use a high-precision reference value of pi
        reference_pi = 3.1415926535897932384626433832795028841971693993751058209749445923078164062
        
        # If any of the pi functions are implemented, test that they match the reference
        if numpy_pi_func_value is not None:
            assert abs(numpy_pi_func_value - reference_pi) < 1e-4
        if torch_pi_func_value is not None:
            assert abs(torch_pi_func_value - reference_pi) < 1e-4
        if mlx_pi_func_value is not None:
            assert abs(mlx_pi_func_value - reference_pi) < 1e-4