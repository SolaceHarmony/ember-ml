# Ember ML Backend Architecture

This document provides a detailed analysis of Ember ML's backend architecture, focusing on how the library achieves backend agnosticism and seamlessly switches between different computation libraries.

## Overview

Ember ML's backend architecture is designed to provide a unified interface for neural network operations that can work with different backend libraries (NumPy, PyTorch, MLX). This is achieved through a sophisticated system of function aliasing, dynamic imports, and consistent interfaces.

## Core Components

### 1. Backend Management System

The backbone of Ember ML's backend agnosticism is its backend management system, which handles:

- Backend selection and switching
- Configuration persistence
- Device management
- Automatic backend detection

```python
# Key functions in backend management
def get_backend() -> str
def set_backend(backend: str) -> None
def get_backend_module() -> ModuleType
def auto_select_backend() -> Tuple[str, Optional[str]]
```

#### Backend Selection

The library supports multiple strategies for backend selection:

1. **Explicit Selection**: Users can directly specify which backend to use
   ```python
   ember_ml.set_backend('numpy')  # or 'torch', 'mlx'
   ```

2. **Configuration File**: Backend preferences are stored in `~/.ember/backend`
   ```python
   def _get_backend_from_file():
       try:
           return EMBER_BACKEND_FILE.read_text().strip()
       except Exception:
           return None
   ```

3. **Environment Variable**: The `EMBER_ML_BACKEND` environment variable can specify the backend

4. **Auto-Detection**: The library can automatically select the optimal backend based on available hardware
   ```python
   def auto_select_backend():
       """Automatically select the best backend based on the available hardware."""
       # Check for PyTorch with CUDA
       if _check_torch_cuda():
           return 'torch', 'cuda'
       # Check for PyTorch with MPS (Apple Silicon)
       if _check_torch_mps():
           return 'torch', 'mps'
       # Check for PyTorch
       if _check_torch():
           return 'torch', 'cpu'
       
       # Check for MLX (Apple Silicon)
       if platform.system() == 'Darwin' and platform.machine() == 'arm64':
           try:
               import mlx.core
               return 'mlx', None
           except ImportError:
               pass
   ```

### 2. Dynamic Function Aliasing

A key aspect of the backend architecture is the dynamic function aliasing system, which:

- Maintains a consistent API across different backends
- Updates function references when the backend changes
- Handles missing implementations gracefully

```python
def _update_ops_aliases():
    """Dynamically updates the ops module's namespace with backend functions."""
    global _aliased_backend
    backend_name = get_backend()

    # Only update if the backend has changed
    if backend_name == _aliased_backend:
        return

    backend_module = get_backend_module()
    current_ops_module = sys.modules[__name__]

    for func_name in _MASTER_OPS_LIST:
        try:
            backend_function = getattr(backend_module, func_name)
            setattr(current_ops_module, func_name, backend_function)
            globals()[func_name] = backend_function
        except AttributeError:
            setattr(current_ops_module, func_name, None)
            globals()[func_name] = None
```

This approach allows the library to:
- Expose a consistent set of functions regardless of the backend
- Switch backends at runtime without requiring code changes
- Gracefully handle missing implementations in specific backends

### 3. Type Definitions and Handling

The backend architecture includes a sophisticated type system that:

- Defines consistent type aliases across backends
- Handles runtime vs. type-checking type definitions
- Provides backend-specific type conversions

```python
# Basic type aliases
type Numeric = Union[int, float]
type Shape = Sequence[int]
type ShapeLike = Union[int, List[int], Tuple[int, ...], Shape]

# Runtime vs. type-checking definitions
if TYPE_CHECKING == True:
    # These imports are for type checking only
    type TensorTypes = Union[
        NumpyArray,
        Any,  # NumpyTensor
        Any,  # EmberTensor
        Parameter # Add Parameter here
    ]
else:
    # Runtime definitions (simplified)
    type TensorTypes = Any
```

### 4. Specialized Module-Level Aliasing

Several modules in Ember ML implement their own dynamic function aliasing mechanisms that coordinate with the main backend system:

#### Statistics Operations (`ember_ml/ops/stats/__init__.py`)

The statistics module maintains its own list of operations and aliasing mechanism:

```python
# Master list of statistical functions expected to be aliased
_STATS_OPS_LIST = [
    'mean', 'var', 'median', 'std', 'percentile',
    'max', 'min', 'sum', 'cumsum',
    'argmax', 'sort', 'argsort', 'gaussian',
]

def _update_stats_aliases():
    """Dynamically updates this module's namespace with backend stats functions."""
    global _aliased_backend_stats
    backend_name = get_backend()

    if backend_name == _aliased_backend_stats:
        return

    backend_module = get_stats_module()
    current_module = sys.modules[__name__]
    
    for func_name in _STATS_OPS_LIST:
        try:
            backend_function = getattr(backend_module, func_name)
            setattr(current_module, func_name, backend_function)
            globals()[func_name] = backend_function
        except AttributeError:
            setattr(current_module, func_name, None)
            globals()[func_name] = None
```

#### Linear Algebra Operations (`ember_ml/ops/linearalg/__init__.py`)

Similar to the statistics module, the linear algebra module maintains its own list of operations:

```python
# Master list of linear algebra functions expected to be aliased
_LINEARALG_OPS_LIST = [
    'solve', 'inv', 'svd', 'eig', 'eigvals', 'det', 'norm', 'qr',
    'cholesky', 'lstsq', 'diag', 'diagonal',
]
```

#### Activation Functions (`ember_ml/nn/modules/activations/__init__.py`)

The activations module combines object-oriented Module classes with dynamically aliased functional operations:

```python
# Import Module classes
from ember_ml.nn.modules.activations.relu_module import ReLU
from ember_ml.nn.modules.activations.sigmoid_module import Sigmoid
# ...

# Master list of activation functions expected to be aliased
_ACTIVATION_OPS_LIST = [
    'relu', 'sigmoid', 'tanh', 'softmax', 'softplus',
]

def _update_activation_aliases():
    """Dynamically updates this module's namespace with backend activation functions."""
    global _aliased_backend_activations
    backend_name = get_backend()

    if backend_name == _aliased_backend_activations:
        return

    backend_module = get_activations_module()
    current_module = sys.modules[__name__]
    
    for func_name in _ACTIVATION_OPS_LIST:
        try:
            backend_function = getattr(backend_activations, func_name)
            setattr(current_module, func_name, backend_function)
            globals()[func_name] = backend_function
        except AttributeError:
            setattr(current_module, func_name, None)
            globals()[func_name] = None
```

### 5. Class-Based vs. Function-Based Approaches

Ember ML employs two different approaches to backend agnosticism:

#### Function-Based Approach (Current)

Instead of using class-based implementations with inheritance (previous approach), Ember ML now primarily uses direct function implementations with dynamic imports:

```python
def add(x: TensorLike, y: TensorLike) -> np.ndarray:
    """Add two NumPy arrays element-wise."""
    from ember_ml.backend.numpy.tensor.tensor import NumpyTensor
    tensor_ops = NumpyTensor()
    return np.add(tensor_ops.convert_to_tensor(x), tensor_ops.convert_to_tensor(y))
```

Key aspects of this pattern:
- Dynamic imports to avoid circular dependencies
- Consistent function signatures across backends
- Local instantiation of helper classes when needed
- Type conversion using backend-specific methods

#### Hybrid Approach: Factory and Function Aliasing (`nn.features`)

The features module (`ember_ml/nn/features/__init__.py`) employs a hybrid approach, combining factory function aliasing for stateful components with direct function aliasing for stateless operations:

```python
# Master list of feature *factory functions* and *direct operations*
_FEATURES_OPS_LIST = [
    # Factories for stateful classes
    'pca',
    'standardize_features',
    'normalize_features',
    # Direct stateless operations
    'one_hot',
    'scatter',
]

def _update_features_aliases():
    """Dynamically updates this module's namespace with backend/common feature functions/factories."""
    # ... (similar aliasing logic as ops.stats) ...
    for op_name in _FEATURES_OPS_LIST:
        try:
            # Get the factory function or direct operation from the loaded module
            op_function = getattr(features_ops_module, op_name)
            setattr(current_module, op_name, op_function)
            globals()[op_name] = op_function
        except AttributeError:
            # Handle missing implementations
            setattr(current_module, op_name, None)
            globals()[op_name] = None
```

**How it works**:

1.  **Stateful Components (PCA, Standardizer, Normalizer)**:
    *   Classes like `PCA` are defined in `nn/features/common/`.
    *   Factory functions (e.g., `pca()`) are defined in `nn/features/common/__init__.py` to instantiate these classes (`def pca(): return PCA()`).
    *   The `nn/features/__init__.py` aliases these *factory functions* (`pca`, `standardize_features`, `normalize_features`).
    *   Users call the factory function (`features.pca()`) to get a stateful instance and then call methods on it (`instance.fit(data)`).

2.  **Stateless Operations (one_hot, scatter)**:
    *   Standalone functions (`one_hot`, `scatter`) are defined in `nn/features/common/tensor_features.py`.
    *   These functions are imported and exported directly by `nn/features/common/__init__.py`.
    *   The `nn/features/__init__.py` aliases these *direct functions* (`one_hot`, `scatter`).
    *   Users call these functions directly (`features.one_hot(...)`).

**Benefits**:

*   Unifies the backend switching mechanism using function aliasing.
*   Provides a natural object-oriented API for stateful components like PCA.
*   Offers a direct functional API for stateless operations.
*   Removes the need for separate interface definitions and complex instance caching in the `__init__`.

### 5. Tensor Module Architecture

The tensor module (`ember_ml/nn/tensor/__init__.py`) takes a unique approach to backend agnosticism:

```python
# Import interfaces
from ember_ml.nn.tensor.interfaces import TensorInterface
from ember_ml.nn.tensor.interfaces.dtype import DTypeInterface

# Import common implementation
from ember_ml.nn.tensor.common import EmberTensor
from ember_ml.nn.tensor.common.dtypes import (
    EmberDType, DType, dtype as dtype_instance,
    get_dtype, to_dtype_str, from_dtype_str
)

# Import tensor operations from common
from ember_ml.nn.tensor.common import (
    zeros, ones, eye, arange, linspace,
    zeros_like, ones_like, full, full_like,
    reshape, transpose, concatenate, stack, split,
    # ...and many more
)
```

The tensor module uses a combination of approaches:

1. **Common Implementation**: Instead of dynamic backend switching, it provides a common implementation (`EmberTensor`) that wraps backend-specific tensors
2. **Conversion Functions**: It includes functions to convert between different tensor types
3. **Direct Imports**: It imports operations directly from the common implementation

This approach provides several advantages:
- Consistent API regardless of the backend
- Transparent handling of different tensor types
- Simple conversion between backends

The `convert_to_tensor` function is a key component of this architecture:

```python
def convert_to_tensor(data: Any, dtype=None, device=None, requires_grad=False):
    """Create a tensor from data."""
    # If already an EmberTensor, return it directly
    if type(EmberTensor) == type(data):
        return data
    
    # Convert to backend tensor first using the internal function
    from ember_ml.nn.tensor.common import _convert_to_backend_tensor
    backend_tensor = _convert_to_backend_tensor(data, dtype=dtype)
    
    # Wrap in EmberTensor
    return EmberTensor(backend_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
```

### 6. Operations Module

The operations (`ops`) module serves as the primary interface for backend-agnostic operations:

- Maintains a master list of supported operations
- Dynamically aliases functions from the active backend
- Handles backend switching and updates

```python
# Master list of operations
_MASTER_OPS_LIST = [
    # Math
    'add', 'subtract', 'multiply', 'divide', 'matmul', 'dot', 'mean', 'sum', 'max', 'min',
    # Comparison
    'equal', 'not_equal', 'less', 'less_equal', 'greater', 'greater_equal',
    # Device
    'to_device', 'get_device', 'get_available_devices',
    # And many more...
]
```

The module includes a specialized `set_backend` function that:
1. Calls the original backend setting function
2. Updates aliases in the ops module
3. Triggers updates in related modules (stats, activations)

```python
def set_backend(backend: str):
    """Sets the backend and updates ops aliases."""
    original_set_backend(backend)
    _update_ops_aliases()
    # Trigger updates in other aliasing modules
    from ember_ml.nn.modules.activations import _update_activation_aliases
    _update_activation_aliases()
```

## Backend-Specific Implementations

Each backend provides consistent implementations of the operations defined in the master list:

### NumPy Backend

```python
# NumPy implementation of add
def add(x: TensorLike, y: TensorLike) -> np.ndarray:
    from ember_ml.backend.numpy.tensor.tensor import NumpyTensor
    tensor_ops = NumpyTensor()
    return np.add(tensor_ops.convert_to_tensor(x), tensor_ops.convert_to_tensor(y))
```

### PyTorch Backend

```python
# PyTorch implementation of add
def add(x: TensorLike, y: TensorLike) -> torch.Tensor:
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    return torch.add(tensor_ops.convert_to_tensor(x), tensor_ops.convert_to_tensor(y))
```

### MLX Backend

```python
# MLX implementation of add (conceptual)
def add(x: TensorLike, y: TensorLike) -> mx.array:
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor_ops = MLXTensor()
    return mx.add(tensor_ops.convert_to_tensor(x), tensor_ops.convert_to_tensor(y))
```

## Device Management

The backend architecture includes a device management system that:

- Provides consistent device handling across backends
- Adapts to backend-specific device capabilities
- Handles device-specific operations transparently

```python
def get_device(tensor=None):
    """Get the current device."""
    backend = get_backend()
    
    if tensor is not None:
        # If a tensor is provided, try to get its device
        if hasattr(tensor, 'device'):
            return str(tensor.device)
    
    if backend == 'numpy':
        return 'cpu'
    elif backend == 'torch':
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    elif backend == 'mlx':
        import mlx.core as mx
        return mx.default_device().type
```

## Key Design Patterns

### 1. Dynamic Imports

To avoid circular dependencies, the library uses dynamic imports within functions:

```python
def matmul(x, y):
    from ember_ml.backend.numpy.tensor.tensor import NumpyTensor
    tensor_ops = NumpyTensor()
    # Implementation...
```

### 2. Function Aliasing

Instead of inheritance-based polymorphism, the library uses function aliasing:

```python
# In ops module
backend_function = getattr(backend_module, func_name)
setattr(current_ops_module, func_name, backend_function)
globals()[func_name] = backend_function
```

### 3. Consistent Interfaces

All backends implement the same function signatures:

```python
# Same signature across all backends
def add(x: TensorLike, y: TensorLike) -> TensorType:
    # Backend-specific implementation
```

### 4. Lazy Loading

The library uses lazy loading to minimize import time and avoid circular dependencies:

```python
def get_backend_module():
    global _CURRENT_BACKEND_MODULE
    
    if _CURRENT_BACKEND_MODULE is None:
        # Import the backend module only when needed
        backend = get_backend()
        _CURRENT_BACKEND_MODULE = importlib.import_module(_BACKENDS[backend])
    
    return _CURRENT_BACKEND_MODULE
```

## Summary

Ember ML's backend architecture represents a sophisticated approach to backend-agnostic machine learning:

1. **Unified Interface**: Consistent API across different backends
2. **Dynamic Switching**: Runtime backend selection without code changes
3. **Function Aliasing**: Direct function references instead of class-based wrappers
4. **Type System**: Comprehensive type definitions for static checking
5. **Device Management**: Transparent handling of different compute devices

This architecture enables users to write code once and run it on different hardware platforms with optimal performance, regardless of whether they're using NumPy, PyTorch, or MLX as the underlying computation library.