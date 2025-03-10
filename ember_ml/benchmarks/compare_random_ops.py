#!/usr/bin/env python
"""
Compare random operations across different backends using ember_ml.ops.

This script benchmarks and compares the random operations in ember_ml.ops
across different backends (numpy, torch, mlx).
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Callable, Any
import pandas as pd

from ember_ml.backend import set_backend, get_backend
from ember_ml import ops
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
    return end_time - start_time, result

def benchmark_function(func: Callable, num_runs: int, *args, **kwargs) -> Tuple[float, Any]:
    """Benchmark a function over multiple runs and return the average time and last result."""
    total_time = 0
    result = None
    for _ in range(num_runs):
        run_time, result = time_function(func, *args, **kwargs)
        total_time += run_time
    return total_time / num_runs, result

def run_benchmarks_for_backend(backend_name: str) -> Dict:
    """Run benchmarks for a specific backend."""
    print(f"\nSetting backend to {backend_name}...")
    set_backend(backend_name)
    ops.set_seed(SEED)
    
    results = {}
    
    # Normal distribution
    print(f"Benchmarking normal distribution with {backend_name} backend...")
    normal_time, normal_result = benchmark_function(
        ops.random_normal, NUM_RUNS, SHAPE, 0.0, 1.0
    )
    
    # Uniform distribution
    print(f"Benchmarking uniform distribution with {backend_name} backend...")
    uniform_time, uniform_result = benchmark_function(
        ops.random_uniform, NUM_RUNS, SHAPE, 0.0, 1.0
    )
    
    # Get the random_ops instance to access the methods we implemented
    random_ops_instance = ops.random_ops()
    
    # Exponential distribution
    print(f"Benchmarking exponential distribution with {backend_name} backend...")
    exponential_time, exponential_result = benchmark_function(
        random_ops_instance.random_exponential, NUM_RUNS, SHAPE, 1.0
    )
    
    # Gamma distribution
    print(f"Benchmarking gamma distribution with {backend_name} backend...")
    gamma_time, gamma_result = benchmark_function(
        random_ops_instance.random_gamma, NUM_RUNS, SHAPE, 2.0, 1.0
    )
    
    # Poisson distribution
    print(f"Benchmarking poisson distribution with {backend_name} backend...")
    poisson_time, poisson_result = benchmark_function(
        random_ops_instance.random_poisson, NUM_RUNS, SHAPE, 5.0
    )
    
    # Categorical distribution
    print(f"Benchmarking categorical distribution with {backend_name} backend...")
    # Create logits for 5 categories
    logits = ops.ones((SHAPE[0], 5))
    categorical_time, categorical_result = benchmark_function(
        random_ops_instance.random_categorical, NUM_RUNS, logits, 1
    )
    
    # Convert results to numpy for consistent handling
    normal_np = backend_utils.tensor_to_numpy_safe(normal_result)
    uniform_np = backend_utils.tensor_to_numpy_safe(uniform_result)
    exponential_np = backend_utils.tensor_to_numpy_safe(exponential_result)
    gamma_np = backend_utils.tensor_to_numpy_safe(gamma_result)
    poisson_np = backend_utils.tensor_to_numpy_safe(poisson_result)
    categorical_np = backend_utils.tensor_to_numpy_safe(categorical_result)
    
    # Save the raw distributions for comparison
    np.save(f"outputs/plots/{backend_name}_normal.npy", normal_np)
    np.save(f"outputs/plots/{backend_name}_uniform.npy", uniform_np)
    np.save(f"outputs/plots/{backend_name}_exponential.npy", exponential_np)
    np.save(f"outputs/plots/{backend_name}_gamma.npy", gamma_np)
    np.save(f"outputs/plots/{backend_name}_poisson.npy", poisson_np)
    np.save(f"outputs/plots/{backend_name}_categorical.npy", categorical_np)
    
    results = {
        "normal": {
            "time": normal_time,
            "mean": np.mean(normal_np),
            "std": np.std(normal_np),
            "min": np.min(normal_np),
            "max": np.max(normal_np),
            "result": normal_np,
        },
        "uniform": {
            "time": uniform_time,
            "mean": np.mean(uniform_np),
            "std": np.std(uniform_np),
            "min": np.min(uniform_np),
            "max": np.max(uniform_np),
            "result": uniform_np,
        },
        "exponential": {
            "time": exponential_time,
            "mean": np.mean(exponential_np),
            "std": np.std(exponential_np),
            "min": np.min(exponential_np),
            "max": np.max(exponential_np),
            "result": exponential_np,
        },
        "gamma": {
            "time": gamma_time,
            "mean": np.mean(gamma_np),
            "std": np.std(gamma_np),
            "min": np.min(gamma_np),
            "max": np.max(gamma_np),
            "result": gamma_np,
        },
        "poisson": {
            "time": poisson_time,
            "mean": np.mean(poisson_np),
            "std": np.std(poisson_np),
            "min": np.min(poisson_np),
            "max": np.max(poisson_np),
            "result": poisson_np,
        },
        "categorical": {
            "time": categorical_time,
            "mean": np.mean(categorical_np),
            "std": np.std(categorical_np),
            "min": np.min(categorical_np),
            "max": np.max(categorical_np),
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
        
        # Create subplot
        plt.subplot(1, len(backend_results), i + 1)
        plt.hist(data, bins=num_bins, alpha=0.7)
        plt.title(f"{backend.capitalize()} - {distribution.capitalize()}")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        
        # Add statistics
        mean = results[distribution]["mean"]
        std = results[distribution]["std"]
        plt.text(0.05, 0.95, f"Mean: {mean:.6f}\nStd: {std:.6f}",
                 transform=plt.gca().transAxes, verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(f"outputs/plots/{distribution}_comparison.png")
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
        
        # Calculate KL divergence or other statistical measures
        if len(backend_results) > 1:
            backends = list(backend_results.keys())
            for i in range(len(backends)):
                for j in range(i+1, len(backends)):
                    backend1 = backends[i]
                    backend2 = backends[j]
                    
                    # Get the data
                    data1 = backend_results[backend1][dist]["result"]
                    data2 = backend_results[backend2][dist]["result"]
                    
                    # Calculate basic statistics
                    mean_diff = np.abs(np.mean(data1) - np.mean(data2))
                    std_diff = np.abs(np.std(data1) - np.std(data2))
                    
                    # Calculate correlation
                    if dist != "categorical" and dist != "poisson":  # Only for continuous distributions
                        correlation = np.corrcoef(data1, data2)[0, 1]
                        print(f"Correlation between {backend1} and {backend2} for {dist}: {correlation}")
                    
                    print(f"Mean difference between {backend1} and {backend2} for {dist}: {mean_diff}")
                    print(f"Std difference between {backend1} and {backend2} for {dist}: {std_diff}")

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
    return df

def main():
    """Run benchmarks for all backends and compare results."""
    # Create outputs directory if it doesn't exist
    import os
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
    
    # Save results to CSV
    performance_df.to_csv("outputs/plots/random_ops_performance.csv", index=False)
    
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
                    speedup = numpy_time / backend_time
                    row[f"{backend.capitalize()} Time (s)"] = backend_time
                    row[f"{backend.capitalize()} Speedup"] = speedup
            
            summary_data.append(row)
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        print("\nSpeedup Summary:")
        print(summary_df)
        
        # Save summary to CSV
        summary_df.to_csv("outputs/plots/random_ops_speedup.csv", index=False)
    
    print("\nBenchmark complete. Results saved to outputs/plots/")

if __name__ == "__main__":
    main()