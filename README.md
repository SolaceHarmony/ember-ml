# Ember ML: Hardware-Optimized Neural Networks

This repository contains a modern machine learning library that uses cutting-edge neural networks with hardware optimization. Ember ML implements various neuron types based on recent research papers and supports multiple backends (PyTorch, MLX, NumPy, and more) to run efficiently on different hardware platforms (CUDA, Apple Metal, and other exotic platforms).

## Overview

The project focuses on several key areas:

1. **Hardware-Optimized Neural Networks**: Implementation of cutting-edge neural network architectures optimized for different hardware platforms
2. **Multi-Backend Support**: Backend-agnostic tensor operations that work with PyTorch, MLX, NumPy, and other computational backends
3. **Feature Extraction**: Tools for extracting features from BigQuery tables for use in neural networks
4. **Liquid Neural Networks**: Design and implementation of liquid neural networks and other advanced architectures

## Documentation

All user documentation has been organized into the following directories:

- **[docs/api](docs/api/)**: API reference documentation for all modules
- **[docs/tutorials](docs/tutorials/)**: Step-by-step guides for common tasks
- **[docs/examples](docs/examples/)**: Example code and usage patterns
- **[docs/plans](docs/plans/)**: Development plans and roadmaps

For more information, see the [documentation index](docs/index.md).

## Key Components

### Neural Network Architectures

The project implements various cutting-edge neural network architectures:

- Liquid Neural Networks (LNN): Dynamic networks with adaptive connectivity
- Neural Circuit Policies (NCP): Biologically-inspired neural architectures
- Stride-Aware Continuous-time Fully Connected (CfC) networks
- Specialized attention mechanisms and temporal processing units

### Multi-Backend Support

The project implements backend-agnostic tensor operations that can use different computational backends:

- MLX (optimized for Apple Silicon)
- PyTorch (for CUDA and other GPU platforms)
- NumPy (for CPU computation)
- Future support for additional backends

### Feature Extraction

The project includes tools for extracting features from BigQuery tables, including:

- `TerabyteFeatureExtractor`: Extracts features from BigQuery tables
- `TerabyteTemporalStrideProcessor`: Processes temporal data with variable strides

## Getting Started

1. Clone the repository
2. Install the required packages: `pip install -r requirements.txt`
3. Choose your backend:
   ```python
   from ember_ml.backend import set_backend
   
   # Use MLX (optimized for Apple Silicon)
   set_backend('mlx')
   
   # Or use PyTorch
   # set_backend('torch')
   
   # Or use NumPy
   # set_backend('numpy')
   ```
4. Run the example scripts in the `examples/` directory to see different neural network architectures in action

## Example Usage

```python
import ember_ml
from ember_ml import ops

# Create a liquid neural network
model = ember_ml.models.LiquidNeuralNetwork(
    input_size=10,
    hidden_size=32,
    output_size=1
)

# Create input tensor
x = ops.random.normal(shape=(100, 10))

# Forward pass
output = model(x)
```

=======
