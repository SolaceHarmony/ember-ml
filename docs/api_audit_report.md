# Ember ML API Audit Report

This document provides a comprehensive audit of the Ember ML API based on analysis of the `__init__.py` files throughout the codebase.

## Overview

Ember ML is a backend-agnostic neural network library that provides a unified interface for neural network operations across different backends (NumPy, PyTorch, MLX). The library is organized into several key modules:

- **Core**: Main module with backend selection functionality
- **Backend**: Backend-specific implementations (NumPy, PyTorch, MLX)
- **Neural Networks (nn)**: Neural network primitives and components
- **Operations (ops)**: Mathematical operations abstracted across backends
- **Tensor**: Backend-agnostic tensor implementation
- **Features**: Feature extraction and transformation operations
- **Data**: Data handling utilities
- **Models**: Pre-built model architectures
- **Training**: Training utilities
- **Visualization**: Visualization tools
- **Wave**: Wave-specific functionality
- **Utils**: Utility functions

## Module Structure and Signatures

### Core Module (`ember_ml/__init__.py`)

```python
def set_backend(backend_name: Union[str, Literal['numpy', 'torch', 'mlx']]) -> None
```

**Key Imports**:
- `data`
- `models`
- `nn`
- `ops`
- `training`
- `visualization`
- `wave`
- `utils`

**Version**: 0.2.0

### Neural Networks Module (`ember_ml/nn/__init__.py`)

The `nn` module provides foundational building blocks for constructing neural networks with backend-agnostic implementations.

**Key Components**:

- **From Modulation**:
  - `DopamineState`
  - `DopamineModulator`

- **From Attention**:
  - `LTCNeuronWithAttention`

- **From Modules**:
  - `Module`
  - `Parameter`
  - `BaseModule`
  - `NeuronMap`
  - `FullyConnectedMap`
  - `RandomMap`
  - `NCPMap`
  - `NCP`
  - `AutoNCP`

- **From Container**:
  - `Sequential`

- **From Tensor**:
  - `TensorInterface`
  - `EmberTensor`

- **From Modules RNN**:
  - `RNN`, `LSTM`, `GRU`
  - `RNNCell`, `LSTMCell`, `GRUCell`
  - `StrideAware`, `StrideAwareCfC`, `StrideAwareCell`
  - `CfC`, `CfCCell`, `WiredCfCCell`
  - `StrideAwareWiredCfCCell`, `LTC`, `LTCCell`

### Tensor Module (`ember_ml/nn/tensor/__init__.py`)

The tensor module provides a backend-agnostic tensor implementation that works with any backend.

**Key Classes**:
- `TensorInterface`: Abstract interface for tensors
- `DTypeInterface`: Abstract interface for data types
- `EmberTensor`: Main tensor implementation 
- `EmberDType`, `DType`: Data type implementations

**Key Functions**:
```python
def array(data, dtype=None, device=None, requires_grad=False) -> EmberTensor
def convert_to_tensor(data: Any, dtype=None, device=None, requires_grad=False) -> EmberTensor
```

**Tensor Operations**:
- Creation: `zeros`, `ones`, `eye`, `arange`, `linspace`, `zeros_like`, `ones_like`, `full`, `full_like`
- Manipulation: `reshape`, `transpose`, `concatenate`, `stack`, `split`, `expand_dims`, `squeeze`, `tile`, `gather`, `scatter`, `tensor_scatter_nd_update`, `slice`, `slice_update`, `cast`, `copy`, `pad`
- Conversion: `to_numpy`, `item`, `shape`
- Random: `random_uniform`, `random_normal`, `maximum`, `random_bernoulli`, `random_gamma`, `random_exponential`, `random_poisson`, `random_categorical`, `random_permutation`, `shuffle`, `set_seed`, `get_seed`

**Data Types**:
- `float32`, `float64`, `int32`, `int64`, `bool_`, `int8`, `int16`, `uint8`, `uint16`, `uint32`, `uint64`, `float16`
- Data type operations: `get_dtype`, `to_dtype_str`, `from_dtype_str`

### Modules Module (`ember_ml/nn/modules/__init__.py`)

The modules module provides backend-agnostic implementations of neural network modules.

#### Base Module Architecture

The foundation of Ember ML's neural network implementation is built on the `BaseModule` class (exposed as `Module`), which provides a PyTorch-like interface for building neural networks:

```python
class BaseModule:
    def __init__(self)
    def forward(self, *args, **kwargs)  # Abstract method
    def build(self, input_shape)        # Deferred initialization
    def __call__(self, *args, **kwargs) # Handles build and forwards
    def register_parameter(self, name, param)
    def register_buffer(self, name, buffer)
    def add_module(self, name, module)
    def named_parameters(self, prefix='', recurse=True)
    def parameters(self, recurse=True)
    def named_buffers(self, prefix='', recurse=True)
    def buffers(self, recurse=True)
    def named_modules(self, prefix='', memo=None)
    def modules()
    def train(self, mode=True)
    def eval()
    def to(self, device=None, dtype=None)
    def zero_grad()
    def get_config()
    @classmethod
    def from_config(cls, config)
```

The `Parameter` class wraps tensors that require gradients:

```python
class Parameter:
    def __init__(self, data, requires_grad=True)
```

#### Cell and Wiring Architecture

Ember ML implements a sophisticated recurrent neural network architecture with specialized cells and wiring configurations:

**Base Cell Classes**:
- `ModuleCell`: Abstract base class for all recurrent cells
  ```python
  class ModuleCell(Module):
      def __init__(self, input_size, hidden_size, activation="tanh", use_bias=True, **kwargs)
      @property
      def state_size(self) -> Union[int, List[int]]
      @property
      def output_size(self) -> int
      def forward(self, inputs, state=None, **kwargs)  # Abstract method
      def reset_state(self, batch_size=1)
  ```

- `ModuleWiredCell`: Abstract base class for cells with configurable wiring
  ```python
  class ModuleWiredCell(ModuleCell):
      def __init__(self, neuron_map, mode="default", **kwargs)
      @property
      def state_size(self)
      @property
      def layer_sizes(self)
      @property
      def num_layers(self)
      @property
      def sensory_size(self)
      @property
      def motor_size(self)
      @property
      def output_size(self)
      @property
      def synapse_count(self)
      @property
      def sensory_synapse_count(self)
      def build(self, input_shape)
  ```

**Neural Connectivity**:
- `NeuronMap`: Base class for neural connectivity patterns
  ```python
  class NeuronMap:
      def __init__(self, units, output_dim=None, input_dim=None, sparsity_level=0.5, seed=None)
      def build(self, input_dim=None)  # Abstract method
      def set_input_dim(self, input_dim)
      def is_built(self)
      def get_input_mask(self)
      def get_recurrent_mask(self)
      def get_output_mask(self)
      def add_synapse(self, src, dest, polarity)
      def add_sensory_synapse(self, src, dest, polarity)
      @property
      def synapse_count(self)
      @property
      def sensory_synapse_count(self)
  ```

- `NCPMap`: Neural Circuit Policy connectivity pattern
  ```python
  class NCPMap(NeuronMap):
      def __init__(self, inter_neurons, motor_neurons, sensory_neurons=0, sparsity_level=0.5, seed=None,
                  sensory_to_inter_sparsity=None, sensory_to_motor_sparsity=None,
                  inter_to_inter_sparsity=None, inter_to_motor_sparsity=None,
                  motor_to_motor_sparsity=None, motor_to_inter_sparsity=None)
      def build(self, input_dim=None)
      def get_neuron_groups(self)
  ```

- `FullyConnectedMap`: Fully connected neural map
- `RandomMap`: Random connectivity map with configurable sparsity

**Neural Circuit Policy Implementation**:
- `NCP`: Neural Circuit Policy implementation
  ```python
  class NCP(Module):
      def __init__(self, neuron_map, activation="tanh", use_bias=True,
                 kernel_initializer="glorot_uniform", recurrent_initializer="orthogonal",
                 bias_initializer="zeros", dtype=None)
      def build(self, input_shape)
      def forward(self, inputs, state=None, return_state=False)
      def reset_state(self)
  ```

- `AutoNCP`: Automatic NCP generator
  ```python
  class AutoNCP(NCP):
      def __init__(self, units, output_size, sparsity_level=0.5, seed=None, activation="tanh",
                 use_bias=True, kernel_initializer="glorot_uniform",
                 recurrent_initializer="orthogonal", bias_initializer="zeros", dtype=None)
  ```

#### Basic Layers

- `Dense`: Fully connected layer implementation
  ```python
  class Dense(Module):
      def __init__(self, input_dim, units, activation=None, use_bias=True)
      def forward(self, x)
  ```

#### RNN Modules

**Basic RNN Cells**:
- `RNNCell`: Basic RNN cell
- `LSTMCell`: Long Short-Term Memory cell
- `GRUCell`: Gated Recurrent Unit cell

**Advanced RNN Cells**:
- `CfCCell`: Closed-form Continuous-time cell
- `WiredCfCCell`: Wired CfC cell
- `LTCCell`: Liquid Time-Constant cell
- `StrideAwareCell`: Stride-aware cell
- `StrideAwareWiredCfCCell`: Stride-aware wired CfC cell

**RNN Layers**:
- `RNN`: Basic RNN layer
- `LSTM`: Long Short-Term Memory layer
- `GRU`: Gated Recurrent Unit layer
- `CfC`: Closed-form Continuous-time layer
- `LTC`: Liquid Time-Constant layer
- `StrideAware`: Stride-aware base class
- `StrideAwareCfC`: Stride-aware CfC layer

**Activation Modules**:
- `ReLU`, `Tanh`, `Sigmoid`, `Softmax`, `Softplus`, `LeCunTanh`, `Dropout`

### RNN Module (`ember_ml/nn/modules/rnn/__init__.py`)

This module provides implementations of various RNN layers with a focus on specialized cells for continuous-time dynamics and multi-timescale processing.

#### Specialized Continuous-time Cells

**Closed-form Continuous-time (CfC)**:
- `CfCCell`: Cell implementation of the Closed-form Continuous-time RNN
  - Handles irregular time intervals between inputs
  - Implements continuous-time dynamics using closed-form solutions
  - Enables efficient processing of time-series data with variable sampling rates

- `CfC`: Layer wrapper for CfCCell that processes sequences

**Wired CfC**:
- `WiredCfCCell`: CfC cell with configurable neural connectivity patterns
  - Combines continuous-time dynamics with neural circuit policies
  - Allows for sparse, structured connectivity between neurons
  
**Liquid Time-Constant (LTC)**:
- `LTCCell`: Liquid Time-Constant neuron implementation
  - Features dynamically adjustable time constants
  - Implements neuron models with more biologically plausible dynamics
  - Supports adaptive timescales that respond to input characteristics
  
- `LTC`: Layer wrapper for LTCCell that processes sequences

#### Multi-timescale Processing

**Stride-aware Cells**:
- `StrideAwareCell`: Base cell for handling multiple timescales
  - Processes inputs at different temporal resolutions
  - Maintains efficiency by adapting computation to input frequency
  
- `StrideAwareWiredCfCCell`: Combines wired connectivity with stride-awareness
  - Integrates neural circuit policies with multi-timescale processing
  - Supports both continuous-time dynamics and variable input rates
  
- `StrideAwareCfC`: Layer implementation of stride-aware CfC
  - Wrapper for processing sequences with varying timescales

#### Traditional RNN Implementations

**Standard Cells**:
- `RNNCell`: Basic RNN cell with customizable activation functions
- `LSTMCell`: Long Short-Term Memory cell with gates for controlling information flow
- `GRUCell`: Gated Recurrent Unit cell with simplified gating mechanism

**Layer Wrappers**:
- `RNN`: Basic RNN layer that processes sequences using RNNCell
- `LSTM`: Long Short-Term Memory layer for sequence processing
- `GRU`: Gated Recurrent Unit layer for sequence processing
- `StrideAware`: Base class for all stride-aware layers

### Wiring Module (`ember_ml/nn/modules/wiring/__init__.py`)

This module provides implementations defining neural connectivity structures, which are a key part of Ember ML's biologically-inspired neural network architecture.

#### Neural Connectivity Architecture

The wiring module implements the concept of "neural maps" (previously called "wiring") that define how neurons connect to each other. These connectivity patterns are inspired by biological neural circuits and allow for more structured and efficient neural networks.

**Core Concept**:
- Neural connectivity is defined by adjacency matrices that specify which neurons connect to which
- Connections can have positive or negative polarity (excitatory or inhibitory)
- Sparse connectivity patterns can be defined to reduce computational complexity and improve generalization

**Base Class**:
- `NeuronMap`: Abstract base class for all neural connectivity patterns
  - Defines interfaces for building connectivity masks
  - Handles synapse management and tracking
  - Provides methods for querying connectivity properties
  - Supports serialization and deserialization

**Specialized Maps**:

- `NCPMap`: Neural Circuit Policy connectivity pattern
  - Divides neurons into three groups: sensory, inter, and motor neurons
  - Implements biologically-inspired connectivity between groups
  - Configurable sparsity levels for different connection types
  - Supports custom connection probabilities between neuron groups

- `FullyConnectedMap`: Dense connectivity pattern
  - Every neuron connects to every other neuron
  - Provides a baseline for comparison with sparse maps

- `RandomMap`: Stochastic connectivity pattern
  - Randomly connects neurons based on probability
  - Configurable sparsity level
  - Reproducible patterns with seed control

The wiring module integrates closely with the RNN module to create biologically-inspired recurrent neural networks with structured connectivity patterns, enabling more efficient and specialized neural architectures.
### Activations Module (`ember_ml/nn/modules/activations/__init__.py`)

This module provides activation functions and modules with a backend-agnostic implementation strategy that combines object-oriented Module classes with dynamically aliased functional operations.

#### Architecture

The activations module employs a sophisticated design that enables seamless backend switching:

1. **Module Classes**: Object-oriented implementations of activation functions
2. **Functional Operations**: Backend-specific implementations accessed through dynamic aliasing
3. **Dynamic Backend Integration**: Automatic updating of function references when the backend changes

#### Dynamic Function Aliasing

The module uses a runtime function aliasing mechanism to switch between backend implementations:

```python
def _update_activation_aliases():
    """Dynamically updates this module's namespace with backend activation functions."""
    global _aliased_backend_activations
    backend_name = get_backend()

    # Avoid re-aliasing if backend hasn't changed
    if backend_name == _aliased_backend_activations:
        return

    # Get backend activation module
    backend_module = get_activations_module()
    current_module = sys.modules[__name__]
    
    # Update function references for each activation operation
    for func_name in _ACTIVATION_OPS_LIST:
        try:
            backend_function = getattr(backend_activations, func_name)
            setattr(current_module, func_name, backend_function)
            globals()[func_name] = backend_function
        except AttributeError:
            # Handle missing implementations
            setattr(current_module, func_name, None)
            globals()[func_name] = None
```

This approach allows the module to maintain consistent function names while using backend-specific implementations.

#### Module Implementations

**Standard Activation Functions**:
- `ReLU`: Rectified Linear Unit implementation
  ```python
  class ReLU(Module):
      def forward(self, x: tensor.EmberTensor) -> tensor.EmberTensor:
          return relu(x)  # Calls the backend-specific function
  ```
- `Sigmoid`: Sigmoid activation (σ(x) = 1/(1+e^(-x)))
- `Tanh`: Hyperbolic tangent activation
- `Softmax`: Softmax normalization function
- `Softplus`: Smooth approximation of ReLU

**Specialized Activation Functions**:
- `LeCunTanh`: Scaled tanh variant for improved training dynamics
  ```python
  class LeCunTanh(Module):
      def __init__(self):
          self.scale_factor = tensor.convert_to_tensor(0.66666667)
          self.amplitude = tensor.convert_to_tensor(1.7159)
          
      def forward(self, x):
          scaled_x = ops.multiply(self.scale_factor, x)
          tanh_x = tanh(scaled_x)
          return ops.multiply(self.amplitude, tanh_x)
  ```

**Regularization**:
- `Dropout`: Stochastic regularization technique
  ```python
  class Dropout(Module):
      def __init__(self, rate: float, seed: int | None = None):
          self.rate = rate
          self.seed = seed
          
      def forward(self, x, training=True):
          # Only apply dropout during training
          if training and self.rate > 0:
              keep_prob = 1.0 - self.rate
              mask = tensor.random_bernoulli(tensor.shape(x), p=keep_prob)
              scale = 1.0 / keep_prob
              return ops.multiply(ops.multiply(x, mask), scale)
          else:
              return x
  ```

**Function Aliases** (dynamically imported from backend):
- `relu`, `sigmoid`, `tanh`, `softmax`, `softplus`

#### Integration with Backend System

The activations module integrates with the backend system to ensure that:
1. Function aliases are updated when the backend changes
2. Module implementations use the correct backend functions
3. Operations maintain consistent behavior across different backends

This architecture enables seamless switching between NumPy, PyTorch, and MLX backends while maintaining a consistent API for activation functions.
- `softplus`

### Features Module (`ember_ml/nn/features/__init__.py`)

This module provides feature extraction and transformation operations.

**Interfaces**:
- `PCAInterface`: Principal Component Analysis interface
- `StandardizeInterface`: Standardization interface
- `NormalizeInterface`: Normalization interface
- `TensorFeaturesInterface`: Tensor features interface

**Implementations**:
- `PCA`: Principal Component Analysis implementation

**Key Functions**:
```python
def get_features() -> str
def set_features(features_name: str) -> None
def pca_features() -> PCAInterface
def standardize_features() -> StandardizeInterface
def normalize_features() -> NormalizeInterface
def tensor_features() -> TensorFeaturesInterface
```

**Operations**:
- `fit`, `transform`, `fit_transform`, `inverse_transform`: PCA operations
- `one_hot`, `scatter`: Tensor features operations

### Attention Module (`ember_ml/nn/attention/__init__.py`)

This module provides implementations of specialized neurons with attention mechanisms.

**Core Attention Classes**:
- `BaseAttention`: Base attention class
- `AttentionLayer`: Attention layer
- `MultiHeadAttention`: Multi-head attention

**Specific Implementations**:
- `CausalAttention`: Causal attention mechanism
- `PredictionAttention`: Prediction-oriented attention
- `TemporalAttention`: Temporal attention mechanism

**Utilities/Supporting Classes**:
- `AttentionMask`: Attention masking
- `AttentionScore`: Attention scoring
- `AttentionState`: Attention state tracking
- `PositionalEncoding`: Positional encoding
- `LTCNeuronWithAttention`: LTC neuron with attention

### Modulation Module (`ember_ml/nn/modulation/__init__.py`)

This module provides implementations of neural modulation and neuromodulatory systems.

**Classes**:
- `DopamineModulator`: Implements dopamine-based modulation
- `DopamineState`: Represents dopamine system state

### Container Module (`ember_ml/nn/container/__init__.py`)

This module provides container modules for neural networks, which manage collections of layers and their forward propagation.

#### Architecture

The container module implements a flexible architecture for composing neural network layers:

1. **Module Composition**: Layers can be combined in sequential or nested structures
2. **Container Interfaces**: Abstract interfaces define common container operations
3. **Backend Agnosticism**: Implementations work across different backend libraries

#### Module Components

**Sequential Container**:
```python
class Sequential(Module):
    def __init__(self, *args)
    def forward(self, x)
    def append(self, module)
    def __getitem__(self, idx)  # Support for indexing and slicing
    def __len__()
    def __iter__()
```

The Sequential container enables straightforward model construction:

```python
# Example usage
model = Sequential(
    Linear(10, 20),
    ReLU(),
    Linear(20, 1)
)
# or
model = Sequential()
model.add_module('fc1', Linear(10, 20))
model.add_module('relu', ReLU())
model.add_module('fc2', Linear(20, 1))
```

**Normalization Layers**:
```python
class BatchNormalization(Module):
    def __init__(self, epsilon=1e-5, momentum=0.9)
    def forward(self, x)
```

BatchNormalization provides training-time batch statistics and running averages:
- Normalizes activations to have zero mean and unit variance
- Maintains running statistics for inference
- Includes learnable scale (gamma) and shift (beta) parameters

**Linear Layer**:
```python
class Linear(Module, ContainerInterfaces):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None)
    def forward(self, x)
```

The Linear layer implements a fully-connected transformation:
- Uses Kaiming initialization (He initialization) for weights
- Supports optional bias term
- Optimized for different backends

#### Container Interface

The module defines a comprehensive interface for container operations:

```python
class ContainerInterfaces(ABC):
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor
    @abstractmethod
    def add(self, layer: Any) -> None
    @abstractmethod
    def build(self, input_shape: Union[List[int], Tuple[int, ...]]) -> None
    @abstractmethod
    def get_config(self) -> Dict[str, Any]
    @abstractmethod
    def state_dict(self) -> Dict[str, Any]
    @abstractmethod
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None
```

This interface ensures consistent behavior across different container implementations and supports:
- Forward propagation of inputs
- Dynamic addition of layers
- Shape inference and building
- Serialization and deserialization
- State management

#### Backend Management

The container module includes functions for managing backend-specific implementations:

```python
def get_container() -> str  # Get current container implementation
def set_container(container_name: str) -> None  # Set container implementation
```

These functions integrate with the backend system to ensure consistent behavior across different backends.

### Operations Module (`ember_ml/ops/__init__.py`)

This module provides a dynamic interface for mathematical operations that are backend-agnostic.

**Math Operations**:
- Basic: `add`, `subtract`, `multiply`, `divide`, `matmul`, `dot`
- Statistical: `mean`, `sum`, `max`, `min`, `var`, `median`, `std`, `percentile`
- Mathematical: `exp`, `log`, `log10`, `log2`, `pow`, `sqrt`, `square`, `abs`, `sign`
- Trigonometric: `sin`, `cos`, `tan`, `sinh`, `cosh`
- Others: `clip`, `negative`, `mod`, `floor_divide`, `sort`, `gradient`, `cumsum`, `eigh`, `pi`, `power`

**Comparison Operations**:
- `equal`, `not_equal`, `less`, `less_equal`, `greater`, `greater_equal`
- `logical_and`, `logical_or`, `logical_not`, `logical_xor`
- `allclose`, `isclose`, `all`, `where`, `isnan`

**Device Operations**:
- `to_device`, `get_device`, `get_available_devices`
- `memory_usage`, `memory_info`, `synchronize`
- `set_default_device`, `get_default_device`, `is_available`

**I/O Operations**:
- `save`, `load`

**Loss Functions**:
- `mse`, `mean_absolute_error`
- `binary_crossentropy`, `categorical_crossentropy`
- `sparse_categorical_crossentropy`
- `huber_loss`, `log_cosh_loss`

**Vector Operations**:
- `normalize_vector`, `compute_energy_stability`
- `compute_interference_strength`, `compute_phase_coherence`
- `partial_interference`, `euclidean_distance`, `cosine_similarity`
- `exponential_decay`
- FFT: `fft`, `ifft`, `fft2`, `ifft2`, `fftn`, `ifftn`, `rfft`, `irfft`, `rfft2`, `irfft2`, `rfftn`, `irfftn`
- `gaussian`

**Feature Operations**:
- `pca`, `transform`, `inverse_transform`, `standardize`, `normalize`

**Backend Management**:
```python
def set_backend(backend: str) -> None
def get_backend() -> str
def auto_select_backend() -> Tuple[str, Optional[str]]
```

### Backend Module (`ember_ml/backend/__init__.py`)

This module provides backend implementations for different libraries.

**Key Functions**:
```python
def get_backend() -> str
def set_backend(backend: str) -> None
def get_backend_module() -> ModuleType
def get_device(tensor=None) -> str
def set_device(device) -> None
def auto_select_backend() -> Tuple[str, Optional[str]]
```

**Available Backends**:
- `numpy`: NumPy backend
- `torch`: PyTorch backend
- `mlx`: MLX backend

## Models Module (`ember_ml/models/__init__.py`)

The models module provides backend-agnostic implementations of machine learning models using the ops abstraction layer.

**Key Components**:

### Restricted Boltzmann Machine (RBM) (`ember_ml/models/rbm/__init__.py`)

```python
def RBMModule(...)
def contrastive_divergence_step(...)
def train_rbm(...)
def transform_in_chunks(...)
def save_rbm(...)
def load_rbm(...)
```

**Aliases**:
- `RestrictedBoltzmannMachine = RBMModule` (for backward compatibility)

### Attention Models (`ember_ml/models/attention/__init__.py`)

- `CausalAttention`: Causal attention implementation
- `AttentionState`: State management for attention mechanisms

### Liquid State Machines (`ember_ml/models/liquid/__init__.py`)

- `liquid_anomaly_detector`: Anomaly detection using liquid state machines
- `liquidtrainer`: Training utilities for liquid state machines

## Training Module (`ember_ml/training/__init__.py`)

The training module provides backend-agnostic implementations of training components.

**Key Components**:

### Optimizers (`ember_ml/training/optimizer/__init__.py`)

- `Optimizer`: Base optimizer class
- `SGD`: Stochastic Gradient Descent optimizer
- `Adam`: Adam optimizer

### Loss Functions (`ember_ml/training/loss/__init__.py`)

- `Loss`: Base loss class
- `MSELoss`: Mean Squared Error loss
- `CrossEntropyLoss`: Cross Entropy loss

### Hebbian Learning (`ember_ml/training/hebbian/__init__.py`)

This module appears to be a placeholder for future implementation of Hebbian learning algorithms.

## Backend Implementations

Ember ML achieves backend-agnosticism through consistent implementations across different backend libraries. The three supported backends are:

### NumPy Backend (`ember_ml/backend/numpy`)

The NumPy backend provides implementations of all core operations using NumPy as the underlying computation library. It exports the following categories of operations:

- **Math Operations**: Basic arithmetic, statistical functions, and mathematical operations
- **Comparison Operations**: Equality and inequality comparisons, logical operations
- **Device Operations**: CPU device management (NumPy is CPU-only)
- **Linear Algebra Operations**: Matrix operations like solve, inverse, eigenvalue decomposition
- **I/O Operations**: Save and load functionality
- **Vector Operations**: FFT, vector normalization, distance metrics
- **Loss Functions**: Common loss functions for training neural networks
- **Feature Operations**: PCA, standardization, normalization
- **Tensor Operations**: Creation, manipulation, and utility functions for tensors
- **Activation Functions**: Common neural network activation functions

### PyTorch Backend (`ember_ml/backend/torch`)

The PyTorch backend provides implementations using PyTorch, supporting both CPU and GPU computation. It exports operations in the same categories as the NumPy backend, but implemented using PyTorch's tensor operations and device management capabilities.

Key differences from NumPy backend:
- Full GPU support through CUDA and MPS (Apple Silicon)
- Native autograd functionality for gradients
- Optimized tensor operations for deep learning

### MLX Backend (`ember_ml/backend/mlx`)

The MLX backend provides implementations using Apple's MLX framework, which is optimized for Apple Silicon hardware. It exports operations in the same categories as the other backends, implemented using MLX's tensor operations.

Key features:
- Optimized for Apple Silicon (M1/M2/M3 chips)
- Supports both CPU and GPU (Metal) computation
- Lazy evaluation for optimized computation graphs

## Backend Switching Mechanism

The library includes a sophisticated backend switching mechanism:

```python
# Set backend explicitly
ember_ml.set_backend('numpy')  # or 'torch', 'mlx'

# Auto-select best backend based on available hardware
backend, device = ember_ml.auto_select_backend()
```

When a backend is switched:
1. The `_CURRENT_BACKEND` is updated
2. The backend selection is saved to a configuration file
3. The operations module (`ops`) is reloaded to update all function aliases
4. Other modules with backend-specific implementations are updated

This mechanism allows code written using Ember ML to run without modification across different backends, with the library handling all the backend-specific details internally.
## Data Module (`ember_ml/data/__init__.py`)

The data module provides tools for loading and preprocessing data.

**Key Components**:

- `GenericCSVLoader`: Loader for CSV data
- `GenericTypeDetector`: Detector for data types

## Visualization Module (`ember_ml/visualization/__init__.py`)

The visualization module provides tools for visualizing neural networks and data.

**Key Components**:

- `RBMVisualizer`: Visualizer for Restricted Boltzmann Machines

## Wave Module (`ember_ml/wave/__init__.py`)

The wave module provides implementations of wave-based neural processing.

**Key Components**:

- `HarmonicProcessor`: Processor for harmonic wave-based neural processing
- `FrequencyAnalyzer`: Analyzer for frequency components
- `WaveSynthesizer`: Synthesizer for wave generation

**Submodules**:
- `binary`: Binary wave implementations
- `harmonic`: Harmonic wave implementations
- `memory`: Wave memory analysis
- `audio`: Audio processing
- `limb`: Limb-specific wave processing

## Summary

The Ember ML library provides a comprehensive, backend-agnostic neural network API with support for:

1. **Multiple backends**: NumPy, PyTorch, and MLX with consistent interfaces
2. **Tensor operations**: A unified tensor interface that works across backends
3. **Neural network modules**: Core building blocks for neural networks
4. **RNN implementations**: Various RNN architectures including specialized cells
5. **Feature extraction**: PCA and other feature transformations
6. **Attention mechanisms**: Various attention implementations
7. **Neuromodulation**: Dopamine-based modulation
8. **Mathematical operations**: Comprehensive set of backend-agnostic operations
9. **Models**: Implementations of machine learning models like RBM and Liquid State Machines
10. **Training**: Optimizers and loss functions for model training
11. **Data handling**: Tools for loading and preprocessing data
12. **Visualization**: Tools for visualizing models and data
13. **Wave processing**: Wave-based neural processing implementations
8. **Mathematical operations**: Comprehensive set of backend-agnostic operations

The API is well-structured with clear module boundaries and consistent naming conventions. Each module exposes its functionality through a well-defined `__init__.py` file, making the library easy to use and extend. The backend-agnostic design allows for seamless switching between different computation backends (NumPy, PyTorch, MLX), enabling the same code to run efficiently on different hardware platforms.