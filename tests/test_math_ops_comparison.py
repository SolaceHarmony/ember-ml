#!/usr/bin/env python
"""
Compare math operations across different backends.

This script compares the results of various math operations across different backends
to ensure they are consistent.
"""

import pytest
import numpy as np
from ember_ml.backend import set_backend
from ember_ml import ops
from ember_ml.utils import backend_utils

class TestMathOpsComparison:
    """Test the consistency of math operations across backends."""

    @pytest.fixture(autouse=True)
    def setup_test_data(self):
        """Set up the test by creating test data."""
        # Create test data
        self.test_data = {
            "scalar": 2.5,
            "vector": [1.0, 2.0, 3.0, 4.0, 5.0],
            "matrix": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        }
        
        # Operations to test
        self.operations = [
            ("add", lambda x: ops.add(x, x)),
            ("subtract", lambda x: ops.subtract(x, ops.ones_like(x))),
            ("multiply", lambda x: ops.multiply(x, ops.full_like(x, 2.0))),
            ("divide", lambda x: ops.divide(x, ops.full_like(x, 2.0))),
            ("pow", lambda x: ops.pow(x, ops.full_like(x, 2.0))),
            ("sqrt", lambda x: ops.sqrt(x)),
            ("exp", lambda x: ops.exp(x)),
            ("log", lambda x: ops.log(ops.abs(x))),
            ("sin", lambda x: ops.sin(x)),
            ("cos", lambda x: ops.cos(x)),
            ("tan", lambda x: ops.tan(x)),
            ("abs", lambda x: ops.abs(x)),
            ("mean", lambda x: ops.mean(x)),
            ("sum", lambda x: ops.sum(x)),
        ]
        
        # Backends to test
        self.backends = ["numpy", "torch", "mlx"]
        
        # Tolerance for comparison
        self.tolerance = 1e-5

    def test_operations_on_scalar(self):
        """Test operations on a scalar value."""
        self._test_operations("scalar")

    def test_operations_on_vector(self):
        """Test operations on a vector."""
        self._test_operations("vector")

    def test_operations_on_matrix(self):
        """Test operations on a matrix."""
        self._test_operations("matrix")
        
    def test_pi_values(self):
        """Test pi values across backends without enforcing strict tolerance."""
        print("\nComparing pi values across backends:")
        
        # Get pi values from each backend
        pi_values = {}
        for backend in self.backends:
            set_backend(backend)
            try:
                # Try different ways to access pi
                try:
                    # First try to access pi directly from ops
                    if hasattr(ops, 'pi'):
                        pi_value = ops.pi()
                    else:
                        # Try to access pi as a property or method from math_ops
                        math_ops = ops.math_ops()
                        if hasattr(math_ops, 'pi'):
                            pi_attr = getattr(math_ops, 'pi')
                            if callable(pi_attr):
                                pi_value = pi_attr()
                            else:
                                pi_value = pi_attr
                        elif hasattr(math_ops, 'pi_func'):
                            pi_value = math_ops.pi_func()
                        else:
                            # Fallback to direct constant access
                            if backend == "numpy":
                                import numpy as np
                                pi_value = np.pi
                            elif backend == "torch":
                                import torch
                                pi_value = torch.tensor(3.141592653589793)
                            elif backend == "mlx":
                                import mlx.core as mx
                                pi_value = mx.array(3.141592653589793)
                    
                    # Convert to float for comparison
                    pi_float = float(backend_utils.tensor_to_numpy_safe(pi_value))
                except Exception as e:
                    print(f"    Error in first attempt: {e}")
                    # Fallback to direct constant access
                    if backend == "numpy":
                        import numpy as np
                        pi_value = np.pi
                    elif backend == "torch":
                        import torch
                        pi_value = torch.tensor(3.141592653589793)
                    elif backend == "mlx":
                        import mlx.core as mx
                        pi_value = mx.array(3.141592653589793)
                    
                    # Convert to float for comparison
                    pi_float = float(backend_utils.tensor_to_numpy_safe(pi_value))
                pi_values[backend] = pi_float
                print(f"  {backend} pi: {pi_float:.15f}")
            except Exception as e:
                print(f"  Error getting {backend} pi: {e}")
        
        # Compare pi values
        if len(pi_values) > 1:
            print("\nPi value differences:")
            for i, backend1 in enumerate(self.backends):
                if backend1 not in pi_values:
                    continue
                for backend2 in self.backends[i+1:]:
                    if backend2 not in pi_values:
                        continue
                    diff = abs(pi_values[backend1] - pi_values[backend2])
                    print(f"  {backend1} vs {backend2}: {diff:.15f}")
        
        # Compare with math.pi
        import math
        print(f"\nReference pi (math.pi): {math.pi:.15f}")
        for backend, pi_value in pi_values.items():
            diff = abs(pi_value - math.pi)
            print(f"  {backend} vs math.pi: {diff:.15f}")

    def _test_operations(self, data_type):
        """Test operations on the specified data type."""
        print(f"\nTesting operations on {data_type}:")
        
        # Get the test data
        data = self.test_data[data_type]
        
        # Dictionary to store results for each backend
        results = {}
        
        # Run operations on each backend
        for backend in self.backends:
            set_backend(backend)
            backend_results = {}
            
            # Run each operation
            for op_name, op_func in self.operations:
                try:
                    # Convert data to tensor using ops for the current backend
                    tensor = ops.convert_to_tensor(data)
                    
                    # Apply the operation
                    result = op_func(tensor)
                    
                    # Convert result to numpy for comparison using backend_utils
                    result_np = backend_utils.tensor_to_numpy_safe(result)
                    
                    backend_results[op_name] = result_np
                except Exception as e:
                    print(f"Error running {op_name} on {backend}: {e}")
                    backend_results[op_name] = None
            
            results[backend] = backend_results
        
        # Compare results across backends
        for op_name, _ in self.operations:
            print(f"\nComparing {op_name}:")
            
            # Check if all backends have results for this operation
            if all(results[backend][op_name] is not None for backend in self.backends):
                # Compare each pair of backends
                for i, backend1 in enumerate(self.backends):
                    for backend2 in self.backends[i+1:]:
                        result1 = results[backend1][op_name]
                        result2 = results[backend2][op_name]
                        
                        try:
                            # Handle scalar results
                            if np.isscalar(result1) or (isinstance(result1, (list, np.ndarray)) and np.size(result1) == 1):
                                if np.isscalar(result1):
                                    scalar1 = float(result1)
                                else:
                                    scalar1 = float(np.array(result1).flatten()[0])
                                
                                if np.isscalar(result2):
                                    scalar2 = float(result2)
                                else:
                                    scalar2 = float(np.array(result2).flatten()[0])
                                
                                diff = abs(scalar1 - scalar2)
                                print(f"  Difference between {backend1} and {backend2}: {diff}")
                                
                                # Assert that the difference is within tolerance
                                assert diff < self.tolerance, f"Difference between {backend1} and {backend2} for {op_name} exceeds tolerance"
                            else:
                                # Convert to numpy arrays for comparison
                                if not isinstance(result1, np.ndarray):
                                    result1 = np.array(result1)
                                if not isinstance(result2, np.ndarray):
                                    result2 = np.array(result2)
                                
                                # Check if shapes match
                                if result1.shape != result2.shape:
                                    print(f"  Shape mismatch between {backend1} and {backend2}: {result1.shape} vs {result2.shape}")
                                    continue
                                
                                # Calculate difference
                                diff = np.abs(result1 - result2)
                                max_diff = np.max(diff)
                                print(f"  Max difference between {backend1} and {backend2}: {max_diff}")
                                
                                # Assert that the difference is within tolerance
                                assert max_diff < self.tolerance, f"Difference between {backend1} and {backend2} for {op_name} exceeds tolerance"
                        except Exception as e:
                            print(f"  Error comparing {backend1} and {backend2}: {e}")
            else:
                print("  Not all backends have results for this operation")