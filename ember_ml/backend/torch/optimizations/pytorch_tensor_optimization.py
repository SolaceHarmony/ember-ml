"""
PyTorch Tensor Optimization Benchmark

This script demonstrates the performance impact of different tensor types,
precision levels, and device optimizations in PyTorch.
"""

import time
import numpy as np
import torch
import matplotlib.pyplot as plt

def benchmark_operation(operation_name, tensor_fn, sizes=[1000, 2000, 4000], 
                        dtypes=[torch.float32, torch.float16, torch.bfloat16], 
                        devices=['cpu', 'mps']):
    """Benchmark a tensor operation with different configurations."""
    results = {}
    
    for device_name in devices:
        # Skip unavailable devices
        if device_name == 'cuda' and not torch.cuda.is_available():
            continue
        if device_name == 'mps' and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            continue
            
        device = torch.device(device_name)
        results[device_name] = {}
        
        for dtype in dtypes:
            # Skip unsupported dtype/device combinations
            if device_name == 'cpu' and dtype == torch.bfloat16 and not torch.cpu.is_bf16_supported():
                continue
                
            results[device_name][str(dtype)] = []
            
            for size in sizes:
                # Create tensors and move to device
                try:
                    tensors = tensor_fn(size, dtype, device)
                    
                    # Warm-up
                    if operation_name == "matmul":
                        _ = torch.matmul(tensors[0], tensors[1])
                    elif operation_name == "conv2d":
                        _ = torch.nn.functional.conv2d(tensors[0], tensors[1], padding=1)
                    
                    # Benchmark
                    start_time = time.time()
                    for _ in range(5):
                        if operation_name == "matmul":
                            result = torch.matmul(tensors[0], tensors[1])
                        elif operation_name == "conv2d":
                            result = torch.nn.functional.conv2d(tensors[0], tensors[1], padding=1)
                        
                        # Ensure computation is complete
                        if device_name == 'cuda':
                            torch.cuda.synchronize()
                        elif device_name == 'mps':
                            torch.mps.synchronize()
                    end_time = time.time()
                    
                    avg_time = (end_time - start_time) / 5
                    results[device_name][str(dtype)].append(avg_time)
                    print(f"{operation_name} - Size: {size}, Device: {device_name}, Dtype: {dtype}, Time: {avg_time:.4f}s")
                    
                except Exception as e:
                    print(f"Error with {operation_name}, Size: {size}, Device: {device_name}, Dtype: {dtype}: {e}")
                    results[device_name][str(dtype)].append(None)
    
    return results

def create_matmul_tensors(size, dtype, device):
    """Create tensors for matrix multiplication benchmark."""
    a = torch.randn(size, size, dtype=dtype, device=device)
    b = torch.randn(size, size, dtype=dtype, device=device)
    return a, b

def create_conv_tensors(size, dtype, device):
    """Create tensors for convolution benchmark."""
    # For convolution, we'll use a square image with batch size 1 and 3 channels
    # The kernel will have 16 output channels and 3x3 size
    input_tensor = torch.randn(1, 3, size, size, dtype=dtype, device=device)
    kernel = torch.randn(16, 3, 3, 3, dtype=dtype, device=device)
    return input_tensor, kernel

def plot_results(results, operation_name, sizes):
    """Plot benchmark results."""
    plt.figure(figsize=(12, 8))
    
    # Set up colors and markers for different configurations
    colors = {'cpu': 'blue', 'mps': 'green', 'cuda': 'red'}
    markers = {str(torch.float32): 'o', str(torch.float16): 's', str(torch.bfloat16): '^'}
    
    for device_name, device_results in results.items():
        for dtype, times in device_results.items():
            if all(t is not None for t in times):  # Only plot if all benchmarks succeeded
                label = f"{device_name} - {dtype}"
                plt.plot(sizes, times, label=label, 
                         color=colors.get(device_name, 'black'),
                         marker=markers.get(dtype, 'x'))
    
    plt.title(f"{operation_name} Performance by Device and Data Type")
    plt.xlabel("Matrix/Tensor Size")
    plt.ylabel("Time (seconds)")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{operation_name}_benchmark.png")

def main():
    """Main function to run benchmarks."""
    print("PyTorch Tensor Optimization Benchmark")
    print("=====================================")
    
    # Print system information
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}")
    print(f"CPU BFloat16 supported: {torch.cpu.is_bf16_supported() if hasattr(torch.cpu, 'is_bf16_supported') else False}")
    
    # Determine available devices
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        devices.append('mps')
    
    # Determine available dtypes
    dtypes = [torch.float32, torch.float16]
    if hasattr(torch.cpu, 'is_bf16_supported') and torch.cpu.is_bf16_supported():
        dtypes.append(torch.bfloat16)
    
    # Matrix sizes to benchmark
    sizes = [512, 1024, 2048]
    
    # Run matrix multiplication benchmark
    print("\nRunning matrix multiplication benchmark...")
    matmul_results = benchmark_operation("matmul", create_matmul_tensors, 
                                         sizes=sizes, dtypes=dtypes, devices=devices)
    
    # Run convolution benchmark
    print("\nRunning convolution benchmark...")
    conv_sizes = [128, 256, 512]  # Smaller sizes for convolution as it's more intensive
    conv_results = benchmark_operation("conv2d", create_conv_tensors, 
                                      sizes=conv_sizes, dtypes=dtypes, devices=devices)
    
    # Plot results
    plot_results(matmul_results, "Matrix Multiplication", sizes)
    plot_results(conv_results, "2D Convolution", conv_sizes)
    
    print("\nBenchmarks completed! Results saved as PNG files.")

if __name__ == "__main__":
    main()