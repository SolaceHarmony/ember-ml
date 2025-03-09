"""
Demonstration of PyTorch's MPS (Metal Performance Shaders) backend on Apple Silicon.

This script shows how to use the PyTorch backend with MPS acceleration
for matrix operations on Apple Silicon devices.
"""

import time
import numpy as np
import ember_ml as nl

def benchmark_matrix_multiply(backend, device=None, sizes=[1000, 2000, 4000]):
    """Benchmark matrix multiplication with different backends and devices."""
    print(f"\n--- Benchmarking {backend} backend" + (f" on {device}" if device else "") + " ---")
    
    # Set the backend
    nl.set_backend(backend)
    
    for size in sizes:
        # Create random matrices
        a = nl.random_normal((size, size), device=device)
        b = nl.random_normal((size, size), device=device)
        
        # Warm-up
        _ = nl.matmul(a, b)
        
        # Benchmark
        start_time = time.time()
        for _ in range(3):
            _ = nl.matmul(a, b)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 3
        print(f"  Matrix size {size}x{size}: {avg_time:.4f} seconds")

def main():
    """Main function to demonstrate PyTorch MPS backend."""
    print("PyTorch MPS Backend Demonstration")
    print("=================================")
    
    # Check if MPS is available
    import torch
    has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    has_cuda = torch.cuda.is_available()
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {has_mps}")
    print(f"CUDA available: {has_cuda}")
    
    # Benchmark NumPy backend (CPU only)
    benchmark_matrix_multiply('numpy')
    
    # Benchmark PyTorch backend on CPU
    benchmark_matrix_multiply('torch', device='cpu')
    
    # Benchmark PyTorch backend on MPS if available
    if has_mps:
        benchmark_matrix_multiply('torch', device='mps')
    
    # Benchmark PyTorch backend on CUDA if available
    if has_cuda:
        benchmark_matrix_multiply('torch', device='cuda')
    
    # Benchmark MLX backend (optimized for Apple Silicon)
    benchmark_matrix_multiply('mlx')
    
    print("\nDemonstration completed!")

if __name__ == "__main__":
    main()