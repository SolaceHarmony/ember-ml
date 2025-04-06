#!/usr/bin/env python
"""
Compare random operations across different backends using ember_ml.ops.

This script benchmarks and compares the random operations in ember_ml.ops
across different backends (numpy, torch, mlx) without using NumPy directly.
"""

import time
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Callable, Any
import pandas as pd
import os

from ember_ml.backend import set_backend, get_backend
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.utils import backend_utils

# Shape for testing
SHAPE = (10000,)
NUM_RUNS = 5
SEED = 42

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

def calculate_statistics(tensor_var):
    """Calculate statistics using ops module instead of NumPy."""
    # Use ops module for all calculations
    mean_val = ops.mean(tensor_var)
    
    # Calculate standard deviation: sqrt(mean((x - mean(x))^2))
    # Use tensor.full with the shape of the tensor instead of full_like
    tensor_shape = tensor.shape(tensor_var)
    mean_tensor = tensor.full(tensor_shape, mean_val)
    squared_diff = ops.square(ops.subtract(tensor_var, mean_tensor))
    variance = ops.mean(squared_diff)
    std_val = ops.sqrt(variance)
    
    # Min and max
    min_val = ops.stats.min(tensor_var)
    max_val = ops.stats.max(tensor_var)
    
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
    tensor.set_seed(SEED)
    
    results = {}
    
    # Normal distribution
    print(f"Benchmarking normal distribution with {backend_name} backend...")
    normal_time, normal_result = benchmark_function(
        tensor.random_normal, NUM_RUNS, SHAPE, 0.0, 1.0
    )
    
    # Uniform distribution
    print(f"Benchmarking uniform distribution with {backend_name} backend...")
    uniform_time, uniform_result = benchmark_function(
        tensor.random_uniform, NUM_RUNS, SHAPE, 0.0, 1.0
    )
    
  
    # Exponential distribution
    print(f"Benchmarking exponential distribution with {backend_name} backend...")
    exponential_time, exponential_result = benchmark_function(
        tensor.random_exponential, NUM_RUNS, SHAPE, 1.0
    )
    
    # Gamma distribution
    print(f"Benchmarking gamma distribution with {backend_name} backend...")
    gamma_time, gamma_result = benchmark_function(
        tensor.random_gamma, NUM_RUNS, SHAPE, 2.0, 1.0
    )
    
    # Poisson distribution
    print(f"Benchmarking poisson distribution with {backend_name} backend...")
    poisson_time, poisson_result = benchmark_function(
        tensor.random_poisson, NUM_RUNS, SHAPE, 5.0
    )
    
    # Categorical distribution
    print(f"Benchmarking categorical distribution with {backend_name} backend...")
    # Create logits for 5 categories
    logits = tensor.ones((SHAPE[0], 5))
    categorical_time, categorical_result = benchmark_function(
        tensor.random_categorical, NUM_RUNS, logits, 1
    )
    
    # Calculate statistics before converting to numpy
    normal_mean, normal_std, normal_min, normal_max = calculate_statistics(normal_result)
    uniform_mean, uniform_std, uniform_min, uniform_max = calculate_statistics(uniform_result)
    exponential_mean, exponential_std, exponential_min, exponential_max = calculate_statistics(exponential_result)
    gamma_mean, gamma_std, gamma_min, gamma_max = calculate_statistics(gamma_result)
    poisson_mean, poisson_std, poisson_min, poisson_max = calculate_statistics(poisson_result)
    categorical_mean, categorical_std, categorical_min, categorical_max = calculate_statistics(categorical_result)
    
    # Convert results to numpy only for saving to file and plotting
    normal_np = backend_utils.tensor_to_numpy_safe(normal_result)
    uniform_np = backend_utils.tensor_to_numpy_safe(uniform_result)
    exponential_np = backend_utils.tensor_to_numpy_safe(exponential_result)
    gamma_np = backend_utils.tensor_to_numpy_safe(gamma_result)
    poisson_np = backend_utils.tensor_to_numpy_safe(poisson_result)
    categorical_np = backend_utils.tensor_to_numpy_safe(categorical_result)
    
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
    
    with open(f"outputs/plots/{backend_name}_normal.txt", "w") as f:
        f.write(",".join([str(x) for x in flatten_and_slice(normal_np)]))
    with open(f"outputs/plots/{backend_name}_uniform.txt", "w") as f:
        f.write(",".join([str(x) for x in flatten_and_slice(uniform_np)]))
    with open(f"outputs/plots/{backend_name}_exponential.txt", "w") as f:
        f.write(",".join([str(x) for x in flatten_and_slice(exponential_np)]))
    with open(f"outputs/plots/{backend_name}_gamma.txt", "w") as f:
        f.write(",".join([str(x) for x in flatten_and_slice(gamma_np)]))
    with open(f"outputs/plots/{backend_name}_poisson.txt", "w") as f:
        f.write(",".join([str(x) for x in flatten_and_slice(poisson_np)]))
    with open(f"outputs/plots/{backend_name}_categorical.txt", "w") as f:
        f.write(",".join([str(x) for x in flatten_and_slice(categorical_np)]))
    
    results = {
        "normal": {
            "time": normal_time,
            "mean": normal_mean,
            "std": normal_std,
            "min": normal_min,
            "max": normal_max,
            "result": normal_np,  # Keep numpy array for plotting
        },
        "uniform": {
            "time": uniform_time,
            "mean": uniform_mean,
            "std": uniform_std,
            "min": uniform_min,
            "max": uniform_max,
            "result": uniform_np,
        },
        "exponential": {
            "time": exponential_time,
            "mean": exponential_mean,
            "std": exponential_std,
            "min": exponential_min,
            "max": exponential_max,
            "result": exponential_np,
        },
        "gamma": {
            "time": gamma_time,
            "mean": gamma_mean,
            "std": gamma_std,
            "min": gamma_min,
            "max": gamma_max,
            "result": gamma_np,
        },
        "poisson": {
            "time": poisson_time,
            "mean": poisson_mean,
            "std": poisson_std,
            "min": poisson_min,
            "max": poisson_max,
            "result": poisson_np,
        },
        "categorical": {
            "time": categorical_time,
            "mean": categorical_mean,
            "std": categorical_std,
            "min": categorical_min,
            "max": categorical_max,
            "result": categorical_np,
        },
    }
    
    return results

def plot_histograms(backend_results: Dict[str, Dict], distribution: str, num_bins: int = 50):
    """Plot histograms of the distribution for each backend."""
    plt.figure(figsize=(15, 5))
    
    for i, (backend, results) in enumerate(backend_results.items()):
        # Get the data for this distribution and backend
        data = results[distribution]["result"]
        
        # Handle list data for histograms
        if isinstance(data, list):
            # If it's a list of lists, flatten it
            if data and isinstance(data[0], list):
                data = [item for sublist in data for item in sublist]
        
        # Create subplot
        # NOTE: We use Python's addition operator here because we're working with
        # Python's built-in integer values for subplot indexing, not tensors.
        # This is an appropriate use case for Python operators.
        subplot_index = i + 1
        plt.subplot(1, len(backend_results), subplot_index)
        plt.hist(data, bins=num_bins, alpha=0.7)
        plt.title(f"{backend.capitalize()} - {distribution.capitalize()}")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        
        # Add statistics with high precision
        mean = results[distribution]["mean"]
        std = results[distribution]["std"]
        plt.text(0.05, 0.95, f"Mean: {mean:.10g}\nStd: {std:.10g}",
                 transform=plt.gca().transAxes, verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(f"outputs/plots/{distribution}_comparison.png", dpi=300)
    plt.close()

def compare_distributions(backend_results: Dict[str, Dict]):
    """Compare the actual distributions across backends."""
    distributions = ["normal", "uniform", "exponential", "gamma", "poisson", "categorical"]
    
    for dist in distributions:
        # Plot histograms
        plot_histograms(backend_results, dist)
        
        # Compare the first 10 values from each backend
        print(f"\nFirst 10 values for {dist} distribution:")
        for backend, results in backend_results.items():
            values = results[dist]["result"][:10]
            print(f"{backend}: {values}")
        
        # Calculate statistical measures
        if len(backend_results) > 1:
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
                    data1 = backend_results[backend1][dist]["result"]
                    data2 = backend_results[backend2][dist]["result"]
                    
                    # NOTE: We use Python's subtraction operator here because we're working with
                    # Python's built-in float values from our statistics calculations, not tensors.
                    # This is an appropriate use case for Python operators.
                    mean1 = backend_results[backend1][dist]["mean"]
                    mean2 = backend_results[backend2][dist]["mean"]
                    mean_diff = abs(mean1 - mean2)
                    
                    std1 = backend_results[backend1][dist]["std"]
                    std2 = backend_results[backend2][dist]["std"]
                    std_diff = abs(std1 - std2)
                    
                    print(f"Mean difference between {backend1} and {backend2} for {dist}: {mean_diff:.10g}")
                    print(f"Std difference between {backend1} and {backend2} for {dist}: {std_diff:.10g}")

def create_performance_table(backend_results: Dict[str, Dict]) -> pd.DataFrame:
    """Create a performance comparison table."""
    data = []
    
    for distribution in ["normal", "uniform", "exponential", "gamma", "poisson", "categorical"]:
        row = {"Distribution": distribution}
        
        for backend in backend_results.keys():
            row[f"{backend}_time"] = backend_results[backend][distribution]["time"]
            row[f"{backend}_mean"] = backend_results[backend][distribution]["mean"]
            row[f"{backend}_std"] = backend_results[backend][distribution]["std"]
        
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
    
    # Compare the actual distributions
    print("\nComparing distributions across backends...")
    compare_distributions(backend_results)
    
    # Create performance comparison table
    performance_df = create_performance_table(backend_results)
    print("\nPerformance Comparison:")
    print(performance_df)
    
    # Save results to CSV with full precision
    performance_df.to_csv("outputs/plots/random_ops_performance.csv", index=False, float_format='%.15g')
    
    # Create summary table with speedups
    summary_data = []
    for distribution in ["normal", "uniform", "exponential", "gamma", "poisson", "categorical"]:
        if "numpy" in backend_results:
            numpy_time = backend_results["numpy"][distribution]["time"]
            row = {
                "Distribution": distribution,
                "NumPy Time (s)": numpy_time,
            }
            
            for backend in ["torch", "mlx"]:
                if backend in backend_results:
                    backend_time = backend_results[backend][distribution]["time"]
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
        summary_df.to_csv("outputs/plots/random_ops_speedup.csv", index=False, float_format='%.15g')
    
    print("\nBenchmark complete. Results saved to outputs/plots/")

if __name__ == "__main__":
    main()