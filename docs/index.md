# Ember ML Documentation

Welcome to the Ember ML documentation. Ember ML is a library for hardware-optimized neural networks with multiple backend support (PyTorch, MLX, NumPy), focusing on performance, scalability, and ease of use.

## Documentation Sections

- [Architecture](architecture/index.md): System architecture and design principles
  - [Ember ML Architecture](architecture/ember_ml_architecture.md): Comprehensive overview of the Ember ML architecture
  - [Function-First Design](architecture/function_first_design.md): Detailed explanation of the function-first design pattern
- [Feature Extraction](feature_extraction/index.md): Documentation for feature extraction components
- [Notebook](notebook/index.md): Documentation related to Jupyter notebooks and fixes
- [Testing](testing/index.md): Testing procedures and test plans
- [API Reference](api/index.md): Detailed API documentation
  - [Frontend Usage Guide](api/frontend_usage_guide.md): Comprehensive guide on using the Ember ML frontend
  - [Tensor Architecture](api/tensor_architecture.md): Detailed explanation of the tensor operations architecture
- [Tutorials](tutorials/index.md): Step-by-step guides for common tasks
- [Examples](examples/index.md): Code examples and use cases
- [Troubleshooting](troubleshooting/index.md): Solutions for common issues
- [Development](development/index.md): Guidelines for contributors

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

## Architecture Overview

Ember ML is designed with a modular architecture that separates concerns and promotes reusability:

- **Tensor Operations Framework**: Backend-agnostic tensor operations with a function-first design pattern
- **Neural Network Framework**: Modular neural network components with a focus on advanced architectures
- **Backend Abstraction**: Unified API across different computational backends
- **Memory Optimization**: Efficient memory usage through careful design patterns

For more details, see the [Architecture Documentation](architecture/ember_ml_architecture.md).

## Use Cases

Ember ML is particularly well-suited for:

- **Hardware-Optimized Neural Networks**: Running neural networks efficiently on different hardware platforms
- **Advanced Neural Architectures**: Implementing cutting-edge neural network architectures
- **Multi-Backend Deployment**: Deploying models across different computational backends
- **Memory-Constrained Environments**: Running models in environments with limited memory

## Getting Help

If you encounter any issues or have questions:

1. Check the [Troubleshooting](troubleshooting/index.md) guide
2. Search for similar issues in the [GitHub repository](https://github.com/your-org/ember-ml/issues)
3. Ask a question in the [Discussion forum](https://github.com/your-org/ember-ml/discussions)

## Contributing

We welcome contributions to Ember ML! See the [Development Guide](development/index.md) for information on how to contribute.

## License

Ember ML is released under the [MIT License](https://opensource.org/licenses/MIT).