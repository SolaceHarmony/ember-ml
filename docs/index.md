# Ember ML Documentation

Welcome to the Ember ML documentation. Ember ML is a hardware-optimized neural network library that supports multiple backends (PyTorch, MLX, NumPy) to run efficiently on different hardware platforms (CUDA, Apple Metal, and other platforms).

## Documentation Sections

- [API Reference](api/index.md): Detailed API documentation for all modules
  - [Frontend Usage Guide](api/frontend_usage_guide.md): Comprehensive guide on using the Ember ML frontend
  - [Tensor Architecture](api/tensor_architecture.md): Detailed explanation of the tensor operations architecture
- [Architecture](architecture/index.md): System architecture and design principles
  - [Ember ML Architecture](architecture/ember_ml_architecture.md): Comprehensive overview of the Ember ML architecture
  - [Function-First Design](architecture/function_first_design.md): Detailed explanation of the function-first design pattern
- [Tutorials](tutorials/index.md): Step-by-step guides for common tasks
- [Examples](examples/index.md): Code examples and use cases
- [Plans](plans/): Development plans and roadmaps

## Quick Start

### Installation

```bash
pip install ember-ml
```

### Basic Usage

```python
import ember_ml
from ember_ml.nn.tensor import EmberTensor
from ember_ml import ops

# Set the backend
ember_ml.backend.set_backend('mlx')  # or 'torch' or 'numpy'

# Create a tensor
tensor = EmberTensor([[1, 2, 3], [4, 5, 6]])

# Perform operations
result = ops.matmul(tensor, EmberTensor([[1], [2], [3]]))
print(result)  # EmberTensor([[14], [32]])
```

For more detailed instructions, see the [Getting Started](tutorials/getting_started.md) guide and the [Frontend Usage Guide](api/frontend_usage_guide.md).

## Key Features

- **Hardware-Optimized Neural Networks**: Implementation of cutting-edge neural network architectures optimized for different hardware platforms
- **Multi-Backend Support**: Backend-agnostic tensor operations that work with PyTorch, MLX, NumPy, and other computational backends
- **Function-First Design**: Efficient memory usage through separation of functions from class implementations
- **Liquid Neural Networks**: Design and implementation of liquid neural networks and other advanced architectures
- **Neural Circuit Policies**: Biologically-inspired neural architectures with custom wiring configurations

## Core Components

### Neural Network Architectures

The project implements various cutting-edge neural network architectures:

- Liquid Neural Networks (LNN): Dynamic networks with adaptive connectivity
- Neural Circuit Policies (NCP): Biologically-inspired neural architectures
- Stride-Aware Continuous-time Fully Connected (CfC) networks
- Specialized attention mechanisms and temporal processing units

For more details, see the [Architecture Documentation](architecture/ember_ml_architecture.md).

### Multi-Backend Support

The project implements backend-agnostic tensor operations that can use different computational backends:

- MLX (optimized for Apple Silicon)
- PyTorch (for CUDA and other GPU platforms)
- NumPy (for CPU computation)
- Future support for additional backends

### Feature Extraction

The project includes tools for extracting features from large datasets, including:

- `TerabyteFeatureExtractor`: Extracts features from large datasets
- `TemporalStrideProcessor`: Processes temporal data with variable strides

## Getting Help

If you encounter any issues or have questions:

1. Check the tutorials and examples in this documentation
2. Search for similar issues in the GitHub repository
3. Ask a question in the Discussion forum

## License

Ember ML is released under the MIT License.