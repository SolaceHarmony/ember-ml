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

- **[docs/feature_extraction](docs/feature_extraction/)**: Documentation for feature extraction components
- **[docs/architecture](docs/architecture/)**: Documentation for system architecture
- **[docs/testing](docs/testing/)**: Documentation for testing procedures
- **[docs/notebook](docs/notebook/)**: Documentation for notebook usage
- **[docs/api](docs/api/)**: API reference documentation
  - **[docs/api/tensor.md](docs/api/tensor.md)**: Documentation for the tensor module
  - **[docs/api/tensor_architecture.md](docs/api/tensor_architecture.md)**: Detailed explanation of the tensor operations architecture
- **[docs/tutorials](docs/tutorials/)**: Tutorials for using the library
- **[docs/examples](docs/examples/)**: Example code and usage patterns

Internal development documentation has been moved to a separate location.

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

The tensor operations follow a function-first design pattern, where each operation is implemented as a standalone function that can be called directly or through a method on a tensor class. For more details, see the [Tensor Operations Architecture](docs/api/tensor_architecture.md) document.

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
from ember_ml.nn.tensor import EmberTensor

# Create a liquid neural network
model = ember_ml.models.LiquidNeuralNetwork(
    input_size=10,
    hidden_size=32,
    output_size=1
)

# Create input tensor
x = EmberTensor.random_normal(shape=(100, 10))

# Forward pass
output = model(x)
