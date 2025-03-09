# EmberHarmony Operations

This directory contains the operations module for EmberHarmony, providing a unified interface for tensor operations across different backends.

## Overview

The `ops` module is a key component of EmberHarmony's backend-agnostic design. It provides a consistent API for tensor operations that works across different backends (NumPy, PyTorch, MLX), allowing users to write code that is independent of the underlying tensor library.

## Module Structure

The operations module is organized into several files:

- **__init__.py**: Exports all operations from the submodules
- **tensor.py**: Basic tensor operations (creation, manipulation, etc.)
- **math.py**: Mathematical operations (addition, multiplication, etc.)
- **random.py**: Random number generation operations

## Key Features

### Backend Agnosticism

The `ops` module delegates operations to the current backend, making it possible to write code that works with any backend:

```python
from emberharmony import ops

# Create tensors
x = ops.random_normal((3, 4))
y = ops.random_normal((4, 5))

# Perform operations
z = ops.matmul(x, y)
w = ops.relu(z)
```

This code will work with any backend (NumPy, PyTorch, MLX) without any changes.

### Consistent API

The `ops` module provides a consistent API across different backends, making it easier to write and maintain code:

```python
from emberharmony import ops

# NumPy-like API
x = ops.zeros((3, 4))
y = ops.ones((3, 4))
z = ops.add(x, y)

# PyTorch-like API
a = ops.relu(z)
b = ops.sigmoid(z)
c = ops.tanh(z)
```

### Type Conversion

The `ops` module provides functions for converting between different tensor types and NumPy arrays:

```python
from emberharmony import ops
import numpy as np

# Create a NumPy array
x_np = np.random.randn(3, 4)

# Convert to a tensor
x = ops.convert_to_tensor(x_np)

# Perform operations
y = ops.relu(x)

# Convert back to NumPy
y_np = ops.to_numpy(y)
```

## Available Operations

### Tensor Operations

- **Creation**: `zeros`, `ones`, `eye`, `random_normal`, `random_uniform`, etc.
- **Manipulation**: `reshape`, `transpose`, `concat`, `stack`, `split`, etc.
- **Indexing**: `gather`, `scatter`, `slice`, etc.
- **Information**: `shape`, `size`, `dtype`, `device`, etc.

### Mathematical Operations

- **Basic**: `add`, `subtract`, `multiply`, `divide`, etc.
- **Linear Algebra**: `matmul`, `dot`, `norm`, `svd`, etc.
- **Reduction**: `sum`, `mean`, `max`, `min`, etc.
- **Activation**: `relu`, `sigmoid`, `tanh`, `softmax`, etc.

### Random Operations

- **Generation**: `random_normal`, `random_uniform`, `random_bernoulli`, etc.
- **Seeding**: `set_random_seed`, `get_random_seed`, etc.

## Usage Examples

### Basic Usage

```python
import emberharmony as eh
from emberharmony import ops

# Set the backend
eh.set_backend('torch')

# Create tensors
x = ops.random_normal((3, 4))
y = ops.random_normal((4, 5))

# Perform operations
z = ops.matmul(x, y)
w = ops.relu(z)

print(f"x shape: {ops.shape(x)}")
print(f"y shape: {ops.shape(y)}")
print(f"z shape: {ops.shape(z)}")
print(f"w shape: {ops.shape(w)}")
```

### Neural Network Operations

```python
import emberharmony as eh
from emberharmony import ops

# Set the backend
eh.set_backend('torch')

# Create random data
batch_size = 32
input_size = 10
hidden_size = 20
output_size = 5

# Create random input
x = ops.random_normal((batch_size, input_size))

# Create random weights and biases
w1 = ops.random_normal((input_size, hidden_size))
b1 = ops.zeros((hidden_size,))
w2 = ops.random_normal((hidden_size, output_size))
b2 = ops.zeros((output_size,))

# Forward pass
h = ops.relu(ops.add(ops.matmul(x, w1), b1))
y = ops.softmax(ops.add(ops.matmul(h, w2), b2))

print(f"Input shape: {ops.shape(x)}")
print(f"Hidden layer shape: {ops.shape(h)}")
print(f"Output shape: {ops.shape(y)}")
```

### Custom Operations

You can create custom operations that work with any backend by using the existing operations:

```python
import emberharmony as eh
from emberharmony import ops

def custom_activation(x, alpha=0.1):
    """
    Custom activation function: leaky ReLU.
    
    Args:
        x: Input tensor
        alpha: Slope for negative values
        
    Returns:
        Output tensor
    """
    return ops.where(ops.greater_equal(x, 0), x, ops.multiply(alpha, x))

# Set the backend
eh.set_backend('torch')

# Create a tensor
x = ops.random_normal((3, 4))

# Apply the custom activation
y = custom_activation(x)

print(f"x: {ops.to_numpy(x)}")
print(f"y: {ops.to_numpy(y)}")
```

## Implementation Details

### Operation Delegation

The `ops` module delegates operations to the current backend using the `get_backend` function from the `backend` module:

```python
from emberharmony.backend import get_backend

def zeros(shape, dtype=None):
    """
    Create a tensor of zeros.
    
    Args:
        shape: Shape of the tensor
        dtype: Data type of the tensor
        
    Returns:
        A tensor of zeros
    """
    return get_backend().zeros(shape, dtype)
```

### Type Checking

The `ops` module includes type checking to ensure that the inputs to operations are valid:

```python
def matmul(a, b):
    """
    Matrix multiplication.
    
    Args:
        a: First tensor
        b: Second tensor
        
    Returns:
        Result of matrix multiplication
    """
    if not isinstance(a, (np.ndarray, torch.Tensor, mlx.core.array)):
        raise TypeError(f"Expected tensor, got {type(a)}")
    if not isinstance(b, (np.ndarray, torch.Tensor, mlx.core.array)):
        raise TypeError(f"Expected tensor, got {type(b)}")
    
    return get_backend().matmul(a, b)
```

### Error Handling

The `ops` module includes error handling to provide helpful error messages:

```python
def reshape(x, shape):
    """
    Reshape a tensor.
    
    Args:
        x: Input tensor
        shape: New shape
        
    Returns:
        Reshaped tensor
    """
    try:
        return get_backend().reshape(x, shape)
    except Exception as e:
        raise ValueError(f"Failed to reshape tensor of shape {get_backend().shape(x)} to shape {shape}: {e}")
```

## Relationship with Other Modules

The `ops` module works closely with the `backend` module to provide a unified interface for tensor operations across different backends. The `backend` module provides the actual implementation of the operations, while the `ops` module provides a consistent API for using those operations.

For more information on the `backend` module, see the [Backend documentation](../backend/README.md).