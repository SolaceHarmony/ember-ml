"""
Compare the performance of the original PyTorch backend with the optimized version.

This script benchmarks matrix operations using both backends to demonstrate
the performance improvements from our optimizations.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import importlib

# Import the original PyTorch backend
import ember_ml.backend.torch_backend as torch_backend_original

# Import the optimized PyTorch backend
import ember_ml.backend.torch_backend_optimized as torch_backend_optimized

def benchmark_operation(operation_name, sizes=[2000, 4000, 8000], repeats=10):
    """Benchmark an operation with both backends."""
    results = {
        'original': [],
        'optimized': []
    }
    
    for size in sizes:
        print(f"\nBenchmarking {operation_name} with size {size}...")
        
        # Benchmark original backend
        try:
            # Create tensors with original backend
            a_orig = torch_backend_original.random_normal((size, size))
            b_orig = torch_backend_original.random_normal((size, size))
            
            # Warm-up
            if operation_name == "matmul":
                _ = torch_backend_original.matmul(a_orig, b_orig)
            
            # Benchmark
            start_time = time.time()
            for _ in range(repeats):
                if operation_name == "matmul":
                    result = torch_backend_original.matmul(a_orig, b_orig)
            end_time = time.time()
            
            avg_time_orig = (end_time - start_time) / repeats
            results['original'].append(avg_time_orig)
            print(f"  Original backend: {avg_time_orig:.4f}s")
            
        except Exception as e:
            print(f"  Error with original backend: {e}")
            results['original'].append(None)
        
        # Benchmark optimized backend
        try:
            # Create tensors with optimized backend
            a_opt = torch_backend_optimized.random_normal((size, size))
            b_opt = torch_backend_optimized.random_normal((size, size))
            
            # Warm-up
            if operation_name == "matmul":
                _ = torch_backend_optimized.matmul(a_opt, b_opt)
            
            # Benchmark
            start_time = time.time()
            for _ in range(repeats):
                if operation_name == "matmul":
                    result = torch_backend_optimized.matmul(a_opt, b_opt)
            end_time = time.time()
            
            avg_time_opt = (end_time - start_time) / repeats
            results['optimized'].append(avg_time_opt)
            print(f"  Optimized backend: {avg_time_opt:.4f}s")
            
            # Calculate speedup
            if avg_time_orig is not None and avg_time_opt is not None:
                speedup = avg_time_orig / avg_time_opt
                print(f"  Speedup: {speedup:.2f}x")
            
        except Exception as e:
            print(f"  Error with optimized backend: {e}")
            results['optimized'].append(None)
    
    return results

def benchmark_complex_operation(operation_name="complex_matmul", sizes=[1000, 2000, 4000], repeats=5):
    """Benchmark a more complex operation that involves multiple matrix operations."""
    results = {
        'original': [],
        'optimized': []
    }
    
    for size in sizes:
        print(f"\nBenchmarking {operation_name} with size {size}...")
        
        # Benchmark original backend
        try:
            # Create tensors with original backend
            a_orig = torch_backend_original.random_normal((size, size))
            b_orig = torch_backend_original.random_normal((size, size))
            c_orig = torch_backend_original.random_normal((size, size))
            
            # Warm-up
            _ = torch_backend_original.matmul(
                torch_backend_original.add(
                    torch_backend_original.matmul(a_orig, b_orig),
                    c_orig
                ),
                torch_backend_original.transpose(b_orig)
            )
            
            # Benchmark
            start_time = time.time()
            for _ in range(repeats):
                # (A @ B + C) @ B.T
                result = torch_backend_original.matmul(
                    torch_backend_original.add(
                        torch_backend_original.matmul(a_orig, b_orig),
                        c_orig
                    ),
                    torch_backend_original.transpose(b_orig)
                )
            end_time = time.time()
            
            avg_time_orig = (end_time - start_time) / repeats
            results['original'].append(avg_time_orig)
            print(f"  Original backend: {avg_time_orig:.4f}s")
            
        except Exception as e:
            print(f"  Error with original backend: {e}")
            results['original'].append(None)
        
        # Benchmark optimized backend
        try:
            # Create tensors with optimized backend
            a_opt = torch_backend_optimized.random_normal((size, size))
            b_opt = torch_backend_optimized.random_normal((size, size))
            c_opt = torch_backend_optimized.random_normal((size, size))
            
            # Warm-up
            _ = torch_backend_optimized.matmul(
                torch_backend_optimized.add(
                    torch_backend_optimized.matmul(a_opt, b_opt),
                    c_opt
                ),
                torch_backend_optimized.transpose(b_opt)
            )
            
            # Benchmark
            start_time = time.time()
            for _ in range(repeats):
                # (A @ B + C) @ B.T
                result = torch_backend_optimized.matmul(
                    torch_backend_optimized.add(
                        torch_backend_optimized.matmul(a_opt, b_opt),
                        c_opt
                    ),
                    torch_backend_optimized.transpose(b_opt)
                )
            end_time = time.time()
            
            avg_time_opt = (end_time - start_time) / repeats
            results['optimized'].append(avg_time_opt)
            print(f"  Optimized backend: {avg_time_opt:.4f}s")
            
            # Calculate speedup
            if avg_time_orig is not None and avg_time_opt is not None:
                speedup = avg_time_orig / avg_time_opt
                print(f"  Speedup: {speedup:.2f}x")
            
        except Exception as e:
            print(f"  Error with optimized backend: {e}")
            results['optimized'].append(None)
    
    return results

def plot_results(results, operation_name, sizes):
    """Plot benchmark results."""
    plt.figure(figsize=(10, 6))
    
    # Plot original backend
    if all(t is not None for t in results['original']):
        plt.plot(sizes, results['original'], 'b-o', label='Original Backend')
    
    # Plot optimized backend
    if all(t is not None for t in results['optimized']):
        plt.plot(sizes, results['optimized'], 'g-s', label='Optimized Backend')
    
    plt.title(f"{operation_name} Performance Comparison")
    plt.xlabel("Matrix Size")
    plt.ylabel("Time (seconds)")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{operation_name}_comparison.png")
    
    # Also create a bar chart for speedup
    if (all(t is not None for t in results['original']) and 
        all(t is not None for t in results['optimized'])):
        
        plt.figure(figsize=(10, 6))
        speedups = [orig / opt for orig, opt in zip(results['original'], results['optimized'])]
        
        plt.bar(range(len(sizes)), speedups, tick_label=[str(s) for s in sizes])
        plt.axhline(y=1.0, color='r', linestyle='-', alpha=0.3)
        plt.title(f"{operation_name} Speedup (Optimized vs Original)")
        plt.xlabel("Matrix Size")
        plt.ylabel("Speedup Factor (higher is better)")
        plt.grid(True, axis='y')
        plt.savefig(f"{operation_name}_speedup.png")

def print_backend_info():
    """Print information about both backends."""
    print("Original PyTorch Backend:")
    print(f"  Version: {torch_backend_original.__version__}")
    print(f"  Default device: {torch_backend_original.DEFAULT_DEVICE if hasattr(torch_backend_original, 'DEFAULT_DEVICE') else 'Not specified'}")
    print(f"  Default precision: {torch_backend_original.default_float_type}")
    print(f"  CUDA available: {torch_backend_original.has_gpu}")
    print(f"  MPS available: {torch_backend_original.has_mps}")
    
    print("\nOptimized PyTorch Backend:")
    print(f"  Version: {torch_backend_optimized.__version__}")
    print(f"  Default device: {torch_backend_optimized.DEFAULT_DEVICE}")
    print(f"  Default precision: {torch_backend_optimized.DEFAULT_PRECISION}")
    print(f"  CUDA available: {torch_backend_optimized.has_gpu}")
    print(f"  MPS available: {torch_backend_optimized.has_mps}")

def main():
    """Main function to run benchmarks."""
    print("PyTorch Backend Comparison")
    print("=========================")
    
    # Print backend information
    print_backend_info()
    
    # Matrix sizes to benchmark
    sizes = [1000, 2000, 4000]
    
    # Run matrix multiplication benchmark
    print("\nRunning matrix multiplication benchmark...")
    matmul_results = benchmark_operation("matmul", sizes=sizes)
    
    # Run complex operation benchmark
    print("\nRunning complex operation benchmark...")
    complex_results = benchmark_complex_operation("complex_matmul", sizes=sizes)
    
    # Plot results
    plot_results(matmul_results, "Matrix Multiplication", sizes)
    plot_results(complex_results, "Complex Matrix Operation", sizes)
    
    print("\nBenchmarks completed! Results saved as PNG files.")

if __name__ == "__main__":
    main()