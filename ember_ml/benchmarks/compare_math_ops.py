#!/usr/bin/env python
"""
Compare math operations across different backends using ember_ml.ops.

This script benchmarks and compares the math operations in ember_ml.ops
across different backends (numpy, torch, mlx) to ensure consistent results.
"""

import time
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Callable, Any
import pandas as pd
import os

from ember_ml.backend import set_backend, get_backend
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.utils import backend_utils

# Shape for testing
SHAPE = (1000, 100)
NUM_RUNS = 5

def time_function(func: Callable, *args, **kwargs) -> Tuple[float, Any]:
    """Time a function and return the execution time and result."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    # NOTE: We use Python's subtraction operator here because we're working with
    # Python's built-in time.time() values, not tensors. This is an appropriate
    # use case for Python operators.
    elapsed_time = end_time - start_time
    return elapsed_time, result

def benchmark_function(func: Callable, num_runs: int, *args, **kwargs) -> Tuple[float, Any]:
    """Benchmark a function over multiple runs and return the average time and last result."""
    total_time = 0
    result = None
    for _ in range(num_runs):
        run_time, result = time_function(func, *args, **kwargs)
        # NOTE: We use Python's addition operator here because we're working with
        # Python's built-in float values, not tensors. This is an appropriate
        # use case for Python operators.
        total_time += run_time
    # NOTE: We use Python's division operator here because we're working with
    # Python's built-in float values, not tensors. This is an appropriate
    # use case for Python operators.
    avg_time = total_time / num_runs
    return avg_time, result

def calculate_statistics(tensor):
    """Calculate statistics using ops module instead of NumPy."""
    # Use ops module for all calculations
    mean_val = ops.mean(tensor)
    
    # Calculate standard deviation: sqrt(mean((x - mean(x))^2))
    # Use ops.full with the shape of the tensor instead of full_like
    tensor_shape = ops.shape(tensor)
    mean_tensor = ops.full(tensor_shape, mean_val)
    diff = ops.subtract(tensor, mean_tensor)
    # Use multiply instead of square since square might not be available in all backends
    squared_diff = ops.multiply(diff, diff)
    variance = ops.mean(squared_diff)
    std_val = ops.sqrt(variance)
    
    # Min and max
    min_val = ops.min(tensor)
    max_val = ops.max(tensor)
    
    # Convert to Python native types only at the end to avoid precision loss
    # Use the backend's native conversion method
    backend = get_backend()
    if backend == "mlx":
        mean_val = mean_val.item()
        std_val = std_val.item()
        min_val = min_val.item()
        max_val = max_val.item()
    elif backend == "torch":
        mean_val = mean_val.item()
        std_val = std_val.item()
        min_val = min_val.item()
        max_val = max_val.item()
    else:  # numpy
        mean_val = mean_val.item() if hasattr(mean_val, 'item') else float(mean_val)
        std_val = std_val.item() if hasattr(std_val, 'item') else float(std_val)
        min_val = min_val.item() if hasattr(min_val, 'item') else float(min_val)
        max_val = max_val.item() if hasattr(max_val, 'item') else float(max_val)
    
    return mean_val, std_val, min_val, max_val

def run_benchmarks_for_backend(backend_name: str) -> Dict:
    """Run benchmarks for a specific backend."""
    print(f"\nSetting backend to {backend_name}...")
    set_backend(backend_name)
    
    results = {}
    
    # Create test tensors
    x = ops.ones(SHAPE)
    y = ops.full(SHAPE, 2.0)
    
    # Basic arithmetic operations
    print(f"Benchmarking add operation with {backend_name} backend...")
    add_time, add_result = benchmark_function(ops.add, NUM_RUNS, x, y)
    
    print(f"Benchmarking subtract operation with {backend_name} backend...")
    subtract_time, subtract_result = benchmark_function(ops.subtract, NUM_RUNS, x, y)
    
    print(f"Benchmarking multiply operation with {backend_name} backend...")
    multiply_time, multiply_result = benchmark_function(ops.multiply, NUM_RUNS, x, y)
    
    print(f"Benchmarking divide operation with {backend_name} backend...")
    divide_time, divide_result = benchmark_function(ops.divide, NUM_RUNS, x, y)
    
    # Element-wise operations
    print(f"Benchmarking pow operation with {backend_name} backend...")
    pow_time, pow_result = benchmark_function(ops.pow, NUM_RUNS, x, y)
    
    print(f"Benchmarking abs operation with {backend_name} backend...")
    neg_x = ops.subtract(tensor.zeros_like(x), x)  # Create negative values
    abs_time, abs_result = benchmark_function(ops.abs, NUM_RUNS, neg_x)
    
    # Reduction operations
    print(f"Benchmarking mean operation with {backend_name} backend...")
    mean_time, mean_result = benchmark_function(ops.mean, NUM_RUNS, x)
    
    # Activation functions
    print(f"Benchmarking softmax operation with {backend_name} backend...")
    softmax_time, softmax_result = benchmark_function(ops.softmax, NUM_RUNS, x)
    
    # Calculate statistics before converting to numpy
    add_mean, add_std, add_min, add_max = calculate_statistics(add_result)
    subtract_mean, subtract_std, subtract_min, subtract_max = calculate_statistics(subtract_result)
    multiply_mean, multiply_std, multiply_min, multiply_max = calculate_statistics(multiply_result)
    divide_mean, divide_std, divide_min, divide_max = calculate_statistics(divide_result)
    pow_mean, pow_std, pow_min, pow_max = calculate_statistics(pow_result)
    abs_mean, abs_std, abs_min, abs_max = calculate_statistics(abs_result)
    mean_val = mean_result.item() if hasattr(mean_result, 'item') else float(mean_result)
    
    # Convert results to numpy only for saving to file and plotting
    add_np = backend_utils.tensor_to_numpy_safe(add_result)
    subtract_np = backend_utils.tensor_to_numpy_safe(subtract_result)
    multiply_np = backend_utils.tensor_to_numpy_safe(multiply_result)
    divide_np = backend_utils.tensor_to_numpy_safe(divide_result)
    pow_np = backend_utils.tensor_to_numpy_safe(pow_result)
    abs_np = backend_utils.tensor_to_numpy_safe(abs_result)
    softmax_np = backend_utils.tensor_to_numpy_safe(softmax_result)
    
    # Save the raw distributions for comparison
    os.makedirs("outputs/plots", exist_ok=True)
    
    # Save as text files with full precision
    # Handle both numpy arrays and lists
    def flatten_and_slice(arr, n=100):
        """Flatten array or list and take first n elements."""
        if hasattr(arr, 'flatten'):
            return arr.flatten()[:n]
        elif isinstance(arr, list):
            # If it's a list of lists, flatten it
            if arr and isinstance(arr[0], list):
                return [item for sublist in arr[:n] for item in sublist]
            return arr[:n]
        return arr[:n]  # Default case
    
    with open(f"outputs/plots/{backend_name}_add.txt", "w") as f:
        f.write(",".join([str(x) for x in flatten_and_slice(add_np)]))
    with open(f"outputs/plots/{backend_name}_subtract.txt", "w") as f:
        f.write(",".join([str(x) for x in flatten_and_slice(subtract_np)]))
    with open(f"outputs/plots/{backend_name}_multiply.txt", "w") as f:
        f.write(",".join([str(x) for x in flatten_and_slice(multiply_np)]))
    with open(f"outputs/plots/{backend_name}_divide.txt", "w") as f:
        f.write(",".join([str(x) for x in flatten_and_slice(divide_np)]))
    with open(f"outputs/plots/{backend_name}_pow.txt", "w") as f:
        f.write(",".join([str(x) for x in flatten_and_slice(pow_np)]))
    with open(f"outputs/plots/{backend_name}_abs.txt", "w") as f:
        f.write(",".join([str(x) for x in flatten_and_slice(abs_np)]))
    with open(f"outputs/plots/{backend_name}_softmax.txt", "w") as f:
        f.write(",".join([str(x) for x in flatten_and_slice(softmax_np)]))
    
    results = {
        "add": {
            "time": add_time,
            "mean": add_mean,
            "std": add_std,
            "min": add_min,
            "max": add_max,
            "result": add_np,  # Keep numpy array for plotting
        },
        "subtract": {
            "time": subtract_time,
            "mean": subtract_mean,
            "std": subtract_std,
            "min": subtract_min,
            "max": subtract_max,
            "result": subtract_np,
        },
        "multiply": {
            "time": multiply_time,
            "mean": multiply_mean,
            "std": multiply_std,
            "min": multiply_min,
            "max": multiply_max,
            "result": multiply_np,
        },
        "divide": {
            "time": divide_time,
            "mean": divide_mean,
            "std": divide_std,
            "min": divide_min,
            "max": divide_max,
            "result": divide_np,
        },
        "pow": {
            "time": pow_time,
            "mean": pow_mean,
            "std": pow_std,
            "min": pow_min,
            "max": pow_max,
            "result": pow_np,
        },
        "abs": {
            "time": abs_time,
            "mean": abs_mean,
            "std": abs_std,
            "min": abs_min,
            "max": abs_max,
            "result": abs_np,
        },
        "mean": {
            "time": mean_time,
            "result": mean_val,
        },
        "softmax": {
            "time": softmax_time,
            "result": softmax_np,
        },
    }
    
    return results

def plot_histograms(backend_results: Dict[str, Dict], operation: str, num_bins: int = 50):
    """Plot histograms of the operation results for each backend."""
    plt.figure(figsize=(15, 5))
    
    for i, (backend, results) in enumerate(backend_results.items()):
        # Get the data for this operation and backend
        data = results[operation]["result"]
        
        # Handle list data for histograms
        if isinstance(data, list):
            # If it's a list of lists, flatten it
            if data and isinstance(data[0], list):
                data = [item for sublist in data for item in sublist]
        
        # For large arrays, sample a subset for the histogram
        if hasattr(data, 'size') and data.size > 10000:
            indices = np.random.choice(data.size, 10000, replace=False)
            data = data.flatten()[indices]
        elif len(data) > 10000:
            indices = np.random.choice(len(data), 10000, replace=False)
            data = [data[i] for i in indices]
        
        # Create subplot
        # NOTE: We use Python's addition operator here because we're working with
        # Python's built-in integer values for subplot indexing, not tensors.
        # This is an appropriate use case for Python operators.
        subplot_index = i + 1
        plt.subplot(1, len(backend_results), subplot_index)
        plt.hist(data, bins=num_bins, alpha=0.7)
        plt.title(f"{backend.capitalize()} - {operation.capitalize()}")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        
        # Add statistics with high precision
        if "mean" in results[operation]:
            mean = results[operation]["mean"]
            std = results[operation]["std"]
            plt.text(0.05, 0.95, f"Mean: {mean:.10g}\nStd: {std:.10g}",
                    transform=plt.gca().transAxes, verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(f"outputs/plots/{operation}_comparison.png", dpi=300)
    plt.close()

def compare_operations(backend_results: Dict[str, Dict]):
    """Compare the actual operation results across backends."""
    operations = ["add", "subtract", "multiply", "divide", "pow", "abs", "mean", "softmax"]
    
    for op in operations:
        # Skip operations that don't have histogram data
        if op == "mean":
            continue
            
        # Plot histograms
        plot_histograms(backend_results, op)
        
        # Compare the first 10 values from each backend
        print(f"\nFirst 10 values for {op} operation:")
        for backend, results in backend_results.items():
            values = results[op]["result"][:10]
            print(f"{backend}: {values}")
        
        # Calculate statistical measures
        if len(backend_results) > 1 and op != "mean" and op != "softmax":
            backends = list(backend_results.keys())
            for i in range(len(backends)):
                # NOTE: We use Python's addition operator here because we're working with
                # Python's built-in integer values for loop indexing, not tensors.
                # This is an appropriate use case for Python operators.
                next_index = i + 1
                for j in range(next_index, len(backends)):
                    backend1 = backends[i]
                    backend2 = backends[j]
                    
                    # Get the data
                    data1 = backend_results[backend1][op]["result"]
                    data2 = backend_results[backend2][op]["result"]
                    
                    # NOTE: We use Python's subtraction operator here because we're working with
                    # Python's built-in float values from our statistics calculations, not tensors.
                    # This is an appropriate use case for Python operators.
                    mean1 = backend_results[backend1][op]["mean"]
                    mean2 = backend_results[backend2][op]["mean"]
                    mean_diff = abs(mean1 - mean2)
                    
                    std1 = backend_results[backend1][op]["std"]
                    std2 = backend_results[backend2][op]["std"]
                    std_diff = abs(std1 - std2)
                    
                    print(f"Mean difference between {backend1} and {backend2} for {op}: {mean_diff:.10g}")
                    print(f"Std difference between {backend1} and {backend2} for {op}: {std_diff:.10g}")
    
    # Special handling for mean operation
    if "mean" in operations:
        print("\nMean operation results:")
        for backend, results in backend_results.items():
            mean_val = results["mean"]["result"]
            print(f"{backend}: {mean_val}")
        
        # Compare mean values across backends
        if len(backend_results) > 1:
            backends = list(backend_results.keys())
            for i in range(len(backends)):
                next_index = i + 1
                for j in range(next_index, len(backends)):
                    backend1 = backends[i]
                    backend2 = backends[j]
                    
                    mean1 = backend_results[backend1]["mean"]["result"]
                    mean2 = backend_results[backend2]["mean"]["result"]
                    mean_diff = abs(mean1 - mean2)
                    
                    print(f"Mean difference between {backend1} and {backend2}: {mean_diff:.10g}")

def create_performance_table(backend_results: Dict[str, Dict]) -> pd.DataFrame:
    """Create a performance comparison table."""
    data = []
    
    for operation in ["add", "subtract", "multiply", "divide", "pow", "abs", "mean", "softmax"]:
        row = {"Operation": operation}
        
        for backend in backend_results.keys():
            row[f"{backend}_time"] = backend_results[backend][operation]["time"]
            if operation != "mean" and operation != "softmax":
                row[f"{backend}_mean"] = backend_results[backend][operation]["mean"]
                row[f"{backend}_std"] = backend_results[backend][operation]["std"]
        
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Set display precision for pandas
    pd.set_option('display.precision', 10)
    
    return df

def main():
    """Run benchmarks for all backends and compare results."""
    # Create outputs directory if it doesn't exist
    os.makedirs("outputs/plots", exist_ok=True)
    
    # Available backends
    backends = ["numpy", "torch", "mlx"]
    
    # Run benchmarks for each backend
    backend_results = {}
    for backend in backends:
        try:
            backend_results[backend] = run_benchmarks_for_backend(backend)
        except Exception as e:
            print(f"Error running benchmarks for {backend} backend: {e}")
    
    # Compare the actual operations
    print("\nComparing operations across backends...")
    compare_operations(backend_results)
    
    # Create performance comparison table
    performance_df = create_performance_table(backend_results)
    print("\nPerformance Comparison:")
    print(performance_df)
    
    # Save results to CSV with full precision
    performance_df.to_csv("outputs/plots/math_ops_performance.csv", index=False, float_format='%.15g')
    
    # Create summary table with speedups
    summary_data = []
    for operation in ["add", "subtract", "multiply", "divide", "pow", "abs", "mean", "softmax"]:
        if "numpy" in backend_results:
            numpy_time = backend_results["numpy"][operation]["time"]
            row = {
                "Operation": operation,
                "NumPy Time (s)": numpy_time,
            }
            
            for backend in ["torch", "mlx"]:
                if backend in backend_results:
                    backend_time = backend_results[backend][operation]["time"]
                    # NOTE: We use Python's division operator here because we're working with
                    # Python's built-in float values from our timing measurements, not tensors.
                    # This is an appropriate use case for Python operators.
                    speedup = numpy_time / backend_time
                    row[f"{backend.capitalize()} Time (s)"] = backend_time
                    row[f"{backend.capitalize()} Speedup"] = speedup
            
            summary_data.append(row)
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        print("\nSpeedup Summary:")
        print(summary_df)
        
        # Save summary to CSV with full precision
        summary_df.to_csv("outputs/plots/math_ops_speedup.csv", index=False, float_format='%.15g')
    
    print("\nBenchmark complete. Results saved to outputs/plots/")

if __name__ == "__main__":
    main()