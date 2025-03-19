# Ember ML Documentation

Welcome to the Ember ML documentation. Ember ML is a hardware-optimized neural network library that supports multiple backends (PyTorch, MLX, NumPy) to run efficiently on different hardware platforms (CUDA, Apple Metal, and other platforms).

## Documentation Sections

- [API Reference](api/index.md): Detailed API documentation for all modules
=======
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

# Set the backend (optional, auto-selects by default)
eh.set_backend('torch')  # or 'numpy', 'mlx'

# Create a liquid neural network
model = eh.models.LiquidNeuralNetwork(
    input_size=10,
    hidden_size=32,
    output_size=1
)

# Create input tensor
x = ops.random.normal(shape=(100, 10))

# Forward pass
output = model(x)
=======
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

- **Backend Abstraction**: Automatically selects the optimal computational backend (MLX, PyTorch, or NumPy)
- **Neural Network Architectures**: Implementation of cutting-edge neural network architectures like LTC, NCP, and more
- **Feature Extraction**: Tools for extracting features from large datasets
- **Hardware Optimization**: Optimized for different hardware platforms (CUDA, Apple Metal, etc.)
=======
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
=======
- **Tensor Operations Framework**: Backend-agnostic tensor operations with a function-first design pattern
- **Neural Network Framework**: Modular neural network components with a focus on advanced architectures
- **Backend Abstraction**: Unified API across different computational backends
- **Memory Optimization**: Efficient memory usage through careful design patterns

For more details, see the [Architecture Documentation](architecture/ember_ml_architecture.md).

### Multi-Backend Support

The project implements backend-agnostic tensor operations that can use different computational backends:

- MLX (optimized for Apple Silicon)
- PyTorch (for CUDA and other GPU platforms)
- NumPy (for CPU computation)
- Future support for additional backends
- **Hardware-Optimized Neural Networks**: Running neural networks efficiently on different hardware platforms
- **Advanced Neural Architectures**: Implementing cutting-edge neural network architectures
- **Multi-Backend Deployment**: Deploying models across different computational backends
- **Memory-Constrained Environments**: Running models in environments with limited memory

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