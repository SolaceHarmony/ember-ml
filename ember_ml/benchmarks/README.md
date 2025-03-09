# Benchmarks

This directory contains benchmark and comparison scripts for evaluating the performance of different backends and implementations.

## Files

- `compare_torch_backends_improved.py`: Improved version of the PyTorch backend comparison script
- `compare_torch_backends.py`: Original script for comparing different PyTorch backends
- `memory_transfer_analysis_fixed.py`: Fixed version of the memory transfer analysis script
- `memory_transfer_analysis.py`: Original script for analyzing memory transfer performance
- `pytorch_tensor_optimization.py`: Script for optimizing PyTorch tensor operations

## Usage

These scripts can be run directly to benchmark different aspects of the EmberHarmony framework:

```bash
# Compare PyTorch backends
python benchmarks/compare_torch_backends_improved.py

# Analyze memory transfer performance
python benchmarks/memory_transfer_analysis_fixed.py
```

The results of these benchmarks can help identify performance bottlenecks and guide optimization efforts.