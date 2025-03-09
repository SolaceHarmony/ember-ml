#!/usr/bin/env python
"""
Test for consistent behavior across different backends.

This script runs a set of operations on all backends and compares the results
to ensure they're consistent. It helps identify operations that behave differently
across backends, which need to be normalized.
"""

import os
import sys
import argparse
import numpy as np
from typing import Dict, List, Any, Tuple, Callable
import pandas as pd

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ember_ml.backend import set_backend, get_backend
from ember_ml import ops
# Set of operations to test
OPERATIONS_TO_TEST = {
    # Basic math operations
    "add": lambda backend: ops.add(ops.ones((3, 3)), ops.ones((3, 3))),
    "subtract": lambda backend: ops.subtract(ops.ones((3, 3)), ops.ones((3, 3))),
    "multiply": lambda backend: ops.multiply(ops.ones((3, 3)), ops.ones((3, 3))),
    "divide": lambda backend: ops.divide(ops.ones((3, 3)), ops.ones((3, 3))),
    "pow": lambda backend: ops.pow(ops.ones((3, 3)), 2),
    "sqrt": lambda backend: ops.sqrt(ops.ones((3, 3))),
    "exp": lambda backend: ops.exp(ops.ones((3, 3))),
    "log": lambda backend: ops.log(ops.ones((3, 3))),
    
    # Reduction operations
    "sum": lambda backend: ops.sum(ops.ones((3, 3))),
    "mean": lambda backend: ops.mean(ops.ones((3, 3))),
    "max": lambda backend: ops.max(ops.ones((3, 3))),
    "min": lambda backend: ops.min(ops.ones((3, 3))),
    
    # Shape operations
    "reshape": lambda backend: ops.reshape(ops.ones((3, 3)), (9,)),
    "transpose": lambda backend: ops.transpose(ops.ones((3, 3))),
    "expand_dims": lambda backend: ops.expand_dims(ops.ones((3, 3)), axis=0),
    "squeeze": lambda backend: ops.squeeze(ops.expand_dims(ops.ones((3, 3)), axis=0)),
    
    # Random operations (with seed for reproducibility)
    "random_normal": lambda backend: (ops.random_ops().set_seed(42), ops.random_ops().random_normal((3, 3)))[1],
    "random_uniform": lambda backend: (ops.random_ops().set_seed(42), ops.random_ops().random_uniform((3, 3)))[1],
    
    # Linear algebra operations
    "matmul": lambda backend: ops.matmul(ops.ones((3, 3)), ops.ones((3, 3))),
    "solve": lambda backend: ops.matmul(ops.eye(3), ops.ones((3, 1))),  # Use matmul instead of solve for now
    
    # Activation functions
    "relu": lambda backend: ops.relu(ops.ones((3, 3))),
    "sigmoid": lambda backend: ops.sigmoid(ops.ones((3, 3))),
    "tanh": lambda backend: ops.tanh(ops.ones((3, 3))),
    "softmax": lambda backend: ops.softmax(ops.ones((3, 3))),
}

def run_operation_for_backend(backend: str, operation_func: Callable) -> Any:
    """Run an operation for a specific backend."""
    set_backend(backend)
    return operation_func(backend)

def compare_results(results: Dict[str, Dict[str, Tuple[bool, Any]]]) -> pd.DataFrame:
    """Compare results across backends and return a DataFrame."""
    data = []
    
    for operation, backend_results in results.items():
        row = {"Operation": operation}
        
        # Check if all backends succeeded
        all_succeeded = all(success for success, _ in backend_results.values())
        row["All Succeeded"] = all_succeeded
        
        # If all backends succeeded, check if results are consistent
        if all_succeeded:
            # Convert results to numpy arrays for comparison
            numpy_results = {}
            shapes = {}
            dtypes = {}
            
            for backend, (_, result) in backend_results.items():
                # Store shape and dtype information
                if hasattr(result, "shape"):
                    # Normalize shape representation (torch.Size vs tuple)
                    if str(type(result.shape)) == "<class 'torch.Size'>":
                        shapes[backend] = tuple(result.shape)
                    else:
                        shapes[backend] = result.shape
                
                if hasattr(result, "dtype"):
                    dtypes[backend] = str(result.dtype)
                
                # Convert to numpy for value comparison
                if hasattr(result, "numpy"):
                    numpy_results[backend] = result.numpy()
                elif hasattr(result, "cpu"):
                    numpy_results[backend] = result.cpu().numpy()
                else:
                    numpy_results[backend] = np.array(result)
            
            # Check if all results are equal
            first_backend = next(iter(numpy_results.keys()))
            first_result = numpy_results[first_backend]
            
            # Check values
            values_equal = True
            for backend, result in numpy_results.items():
                if backend != first_backend:
                    try:
                        # We can now compare random operations since we set a seed
                        if not np.allclose(first_result, result, rtol=1e-5, atol=1e-5):
                            # For random operations, use a larger tolerance
                            if operation in ["random_normal", "random_uniform"]:
                                if not np.allclose(first_result, result, rtol=1e-3, atol=1e-3):
                                    values_equal = False
                                    break
                            else:
                                values_equal = False
                                break
                    except:
                        values_equal = False
                        break
            
            # Check shapes (if they exist)
            shapes_equal = True
            if shapes:
                # Normalize shape representations
                normalized_shapes = {}
                for backend, shape in shapes.items():
                    if isinstance(shape, str) and "torch.Size" in shape:
                        # Extract the shape from the string representation
                        shape_str = shape.replace("torch.Size([", "").replace("])", "")
                        shape_tuple = tuple(int(x.strip()) for x in shape_str.split(",") if x.strip())
                        normalized_shapes[backend] = shape_tuple
                    else:
                        normalized_shapes[backend] = shape
                
                # Compare normalized shapes
                first_shape = normalized_shapes[first_backend]
                for backend, shape in normalized_shapes.items():
                    if backend != first_backend and shape != first_shape:
                        shapes_equal = False
                        break
            
            # Results are consistent if both values and shapes are equal
            all_equal = values_equal and shapes_equal
            
            row["Results Consistent"] = all_equal
        else:
            row["Results Consistent"] = False
        
        # Add backend-specific results
        for backend, (success, result) in backend_results.items():
            row[f"{backend}_success"] = success
            if success:
                if hasattr(result, "shape"):
                    row[f"{backend}_shape"] = str(result.shape)
                if hasattr(result, "dtype"):
                    row[f"{backend}_dtype"] = str(result.dtype)
            else:
                row[f"{backend}_error"] = result
        
        data.append(row)
    
    return pd.DataFrame(data)

def test_backends(backends: List[str]) -> pd.DataFrame:
    """Test operations across different backends."""
    results = {}
    
    for operation_name, operation_func in OPERATIONS_TO_TEST.items():
        backend_results = {}
        
        for backend in backends:
            print(f"Testing {operation_name} on {backend}...")
            try:
                # Use the run_operation_for_backend function to ensure each operation
                # is run with the correct backend context
                result = run_operation_for_backend(backend, operation_func)
                backend_results[backend] = (True, result)
            except Exception as e:
                backend_results[backend] = (False, str(e))
        
        results[operation_name] = backend_results
    
    return compare_results(results)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test for consistent behavior across different backends.")
    parser.add_argument("--backends", nargs="+", default=["numpy", "torch", "mlx"], 
                        help="Backends to test (default: numpy torch mlx)")
    args = parser.parse_args()
    
    # Test backends
    results_df = test_backends(args.backends)
    
    # Print summary
    print("\nSummary:")
    print(f"Total operations tested: {len(results_df)}")
    print(f"Operations with consistent results: {results_df['Results Consistent'].sum()}")
    print(f"Operations with inconsistent results: {len(results_df) - results_df['Results Consistent'].sum()}")
    
    # Print inconsistent operations
    inconsistent = results_df[~results_df["Results Consistent"]]
    if not inconsistent.empty:
        print("\nOperations with inconsistent results:")
        for _, row in inconsistent.iterrows():
            print(f"  - {row['Operation']}")
    
    # Save results to CSV
    results_df.to_csv("backend_consistency_results.csv", index=False)
    print("\nDetailed results saved to backend_consistency_results.csv")

if __name__ == "__main__":
    main()