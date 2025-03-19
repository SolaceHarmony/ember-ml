# API Reference

This section contains detailed API documentation for Ember ML.

## Core Modules

- **`ember_ml.ops`**: Core operations for tensor manipulation
  - `tensor`: Tensor creation and manipulation
  - `math`: Mathematical operations
  - `random`: Random number generation
  - `solver`: Linear algebra operations

- **`ember_ml.backend`**: Backend abstraction system
  - `numpy`: NumPy backend implementation
  - `torch`: PyTorch backend implementation
  - `mlx`: MLX backend implementation
  - `ember`: Ember backend implementation

- **`ember_ml.features`**: Feature extraction and processing
  - `TerabyteFeatureExtractor`: Extracts features from large datasets
  - `TemporalStrideProcessor`: Processes temporal data with variable strides
  - `GenericFeatureEngineer`: General-purpose feature engineering
  - `GenericTypeDetector`: Automatic column type detection

- **`ember_ml.models`**: Machine learning models
  - `liquid`: Liquid neural networks
  - `rbm`: Restricted Boltzmann Machines

## Neural Network Components

- **`ember_ml.core`**: Core neural implementations
  - `ltc`: Liquid Time Constant neurons
  - `geometric`: Geometric processing
  - `spherical_ltc`: Spherical variants
  - `stride_aware_cfc`: Stride-Aware Continuous-time Fully Connected cells

- **`ember_ml.attention`**: Attention mechanisms
  - `temporal`: Time-based attention
  - `causal`: Causal attention
  - `multiscale_ltc`: Multiscale LTC attention

- **`ember_ml.nn`**: Neural network components
  - `wirings`: Network connectivity patterns
  - `modules`: Neural network modules

## Utility Modules

- **`ember_ml.utils`**: Utility functions
  - `math_helpers`: Mathematical utilities
  - `metrics`: Evaluation metrics
  - `visualization`: Plotting tools

- **`ember_ml.initializers`**: Weight initialization
  - `glorot`: Glorot/Xavier initialization
  - `binomial`: Binomial initialization

For detailed documentation on specific functions and classes, refer to the docstrings in the source code.
=======
- `ember_ml.nn.tensor`: Backend-agnostic tensor implementation
- `ember_ml.ops`: Core operations for tensor manipulation
- `ember_ml.nn.modules`: Neural network modules and components
- `ember_ml.backend`: Backend abstraction system
- `ember_ml.initializers`: Weight initialization functions
- `ember_ml.nn.wirings`: Neural circuit policy wiring configurations

## Comprehensive Guides

- [Frontend Usage Guide](frontend_usage_guide.md): Comprehensive guide on using the Ember ML frontend, including tensor operations, neural network components, and backend selection
- [Tensor Module](tensor.md): Overview of the tensor module and its usage
- [Tensor Operations Architecture](tensor_architecture.md): Detailed explanation of the tensor operations architecture

## Tensor Operations

The `ember_ml.nn.tensor` module provides a backend-agnostic tensor implementation that works with any backend (NumPy, PyTorch, MLX) using the backend abstraction layer. Key components include:

- `EmberTensor`: A backend-agnostic tensor class that delegates operations to the current backend
- `EmberDType`: A backend-agnostic data type class that represents data types across different backends
- Common tensor operations: Creation, manipulation, and conversion functions

For detailed usage examples, see the [Frontend Usage Guide](frontend_usage_guide.md).

## Neural Network Components

The `ember_ml.nn.modules` module provides neural network components that can be combined to create complex models. Key components include:

- Basic modules: `Linear`, `Activation`, `Sequential`
- Recurrent networks: `RNN`, `LSTM`, `GRU`, `LTC`, `CFC`
- Neural circuit policies: `NCP`, `AutoNCP`
- Restricted Boltzmann Machines: `RestrictedBoltzmannMachine`

For detailed usage examples, see the [Frontend Usage Guide](frontend_usage_guide.md).

## Backend System

The `ember_ml.backend` module provides a backend abstraction system that allows switching between different computational backends. Key functions include:

- `set_backend(name)`: Sets the active backend
- `get_backend()`: Gets the current backend
- `get_device()`: Gets the current device

For detailed usage examples, see the [Frontend Usage Guide](frontend_usage_guide.md).

## Function Reference

Detailed function reference documentation will be added soon.
