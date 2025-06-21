# Ember ML API Reorganization: User's Migration Guide

This guide demonstrates how to transition from the current Ember ML API to the new, more intuitive API structure. We'll walk through common usage patterns and show how they improve with the new API.

## Table of Contents

1. [API Changes Overview](#api-changes-overview)
2. [Basic Tensor Operations](#basic-tensor-operations)
3. [Common Operations](#common-operations)
4. [Specialized Operations](#specialized-operations)
5. [Full Example Comparison](#full-example-comparison)
6. [Migration Timeline](#migration-timeline)

## API Changes Overview

The Ember ML API has been reorganized to provide a more intuitive experience:

1. **Tensor creation functions** are now available at the top level
2. **Common operations** are consolidated at the top level
3. **Specialized operations** are organized into logical categories
4. **Backend handling** is simpler and more consistent

## Basic Tensor Operations

### Creating Tensors

**Before:**
```python
from ember_ml.nn import tensor

# Create tensors
x = tensor.array([1, 2, 3])
zeros = tensor.zeros((3, 3))
ones = tensor.ones_like(x)
```

**After:**
```python
import ember_ml as em

# Create tensors - cleaner and more intuitive
x = em.array([1, 2, 3])
zeros = em.zeros((3, 3))
ones = em.ones_like(x)
```

### Tensor Manipulation

**Before:**
```python
from ember_ml.nn import tensor

# Manipulate tensors
x = tensor.array([[1, 2], [3, 4]])
reshaped = tensor.reshape(x, (4,))
transposed = tensor.transpose(x)
```

**After:**
```python
import ember_ml as em

# Manipulate tensors - all at the top level
x = em.array([[1, 2], [3, 4]])
reshaped = em.reshape(x, (4,))
transposed = em.transpose(x)
```

## Common Operations

### Math Operations

**Before:**
```python
from ember_ml.nn import tensor
from ember_ml import ops

# Mixed imports for operations
x = tensor.array([1, 2, 3])
y = tensor.array([4, 5, 6])
added = ops.add(x, y)
multiplied = ops.multiply(x, y)
```

**After:**
```python
import ember_ml as em

# Single import, consistent pattern
x = em.array([1, 2, 3])
y = em.array([4, 5, 6])
added = em.add(x, y)
multiplied = em.multiply(x, y)
```

### Matrix Operations

**Before:**
```python
from ember_ml.nn import tensor
from ember_ml import ops

# Create matrices
a = tensor.array([[1, 2], [3, 4]])
b = tensor.array([[5, 6], [7, 8]])
# Matrix multiplication from ops
c = ops.matmul(a, b)
```

**After:**
```python
import ember_ml as em

# Create matrices and multiply - all from same module
a = em.array([[1, 2], [3, 4]])
b = em.array([[5, 6], [7, 8]])
c = em.matmul(a, b)
```

## Specialized Operations

### Linear Algebra

**Before:**
```python
from ember_ml import ops
from ember_ml.nn import tensor

# Create a matrix
a = tensor.array([[1, 2], [3, 4]])
# Access linear algebra through ops
u, s, v = ops.linearalg.svd(a)
```

**After:**
```python
import ember_ml as em

# Create a matrix
a = em.array([[1, 2], [3, 4]])
# Access through specialized module
u, s, v = em.linalg.svd(a)

# Alternative direct import
from ember_ml import linalg
u, s, v = linalg.svd(a)

# Also available at top level
u, s, v = em.svd(a)
```

### Statistics

**Before:**
```python
from ember_ml import ops
from ember_ml.nn import tensor

# Create data
data = tensor.array([1, 2, 3, 4, 5])
# Calculate statistics
mean = ops.stats.mean(data)
std = ops.stats.std(data)
```

**After:**
```python
import ember_ml as em

# Create data
data = em.array([1, 2, 3, 4, 5])
# Calculate statistics - cleaner organization
mean = em.stats.mean(data)
std = em.stats.std(data)

# Also available at top level
mean = em.mean(data)
std = em.std(data)
```

### Activation Functions

**Before:**
```python
from ember_ml.nn.modules.activations import relu, sigmoid
from ember_ml.nn import tensor

# Create data
x = tensor.array([-1, 0, 1, 2])
# Apply activations
y1 = relu(x)
y2 = sigmoid(x)
```

**After:**
```python
import ember_ml as em

# Create data
x = em.array([-1, 0, 1, 2])
# Apply activations - consistent pattern
y1 = em.activations.relu(x)
y2 = em.activations.sigmoid(x)

# Also available at top level
y1 = em.relu(x)
y2 = em.sigmoid(x)
```

## Full Example Comparison

### LSTM Example (Before)

```python
from ember_ml import ops
from ember_ml.nn.modules.rnn import LSTM
from ember_ml.nn import Module, Sequential, tensor
from ember_ml.training import Optimizer, Loss

def generate_sine_wave_data(num_samples=1000, seq_length=100, num_features=1):
    """Generate sine wave data for sequence prediction."""
    # Generate time points
    t = tensor.linspace(0, 2 * ops.pi, seq_length)
    
    # Generate sine waves with random phase shifts
    X = tensor.zeros((num_samples, seq_length, num_features))
    y = tensor.zeros((num_samples, seq_length, num_features))
    
    for i in range(num_samples):
        # Random phase shift
        phase_shift = tensor.random_uniform(0, 2 * ops.pi)
        
        # Generate sine wave with phase shift
        signal = ops.sin(ops.add(t, phase_shift))
        
        # Add some noise
        noise = tensor.random_normal(0, 0.1, seq_length)
        noisy_signal = ops.add(signal, noise)
        
        # Store input and target
        X = tensor.tensor_scatter_nd_update(X, indices, noisy_signal)
        y = tensor.tensor_scatter_nd_update(y, indices, signal)
    
    return X, y

# Train LSTM model
X, y = generate_sine_wave_data()
model = Sequential([LSTM(input_size=1, hidden_size=64, output_size=1)])
```

### LSTM Example (After)

```python
import ember_ml as em
from ember_ml.nn.modules.rnn import LSTM
from ember_ml.nn import Module, Sequential
from ember_ml.training import Optimizer, Loss

def generate_sine_wave_data(num_samples=1000, seq_length=100, num_features=1):
    """Generate sine wave data for sequence prediction."""
    # Generate time points - cleaner API
    t = em.linspace(0, 2 * em.pi, seq_length)
    
    # Generate sine waves with random phase shifts
    X = em.zeros((num_samples, seq_length, num_features))
    y = em.zeros((num_samples, seq_length, num_features))
    
    for i in range(num_samples):
        # Random phase shift - consistent API
        phase_shift = em.random.uniform(0, 2 * em.pi)
        
        # Generate sine wave with phase shift
        signal = em.sin(em.add(t, phase_shift))
        
        # Add some noise - all from same namespace
        noise = em.random.normal(0, 0.1, seq_length)
        noisy_signal = em.add(signal, noise)
        
        # Store input and target
        X = em.tensor_scatter_nd_update(X, indices, noisy_signal)
        y = em.tensor_scatter_nd_update(y, indices, signal)
    
    return X, y

# Train LSTM model
X, y = generate_sine_wave_data()
model = Sequential([LSTM(input_size=1, hidden_size=64, output_size=1)])
```

## Migration Timeline

We understand that updating your code to use the new API may take time. Here's our migration timeline:

1. **Release 1.0 (June 2025)**
   - New API introduced
   - Old API still works without warnings

2. **Release 1.1 (September 2025)**
   - Old API marked as deprecated with warnings
   - Documentation fully updated for new API

3. **Release 2.0 (December 2025)**
   - Old API paths removed
   - Full transition to new API

To help with migration, we provide:
- This migration guide
- Updated examples
- Automatic code transformation scripts (coming soon)

### Quick Migration Steps

1. Replace imports:
   ```python
   # Before
   from ember_ml.nn import tensor
   from ember_ml import ops
   
   # After
   import ember_ml as em
   ```

2. Update tensor creation:
   ```python
   # Before
   x = tensor.array([1, 2, 3])
   
   # After
   x = em.array([1, 2, 3])
   ```

3. Update operations:
   ```python
   # Before
   y = ops.add(x, x)
   z = tensor.reshape(y, (3, 1))
   
   # After
   y = em.add(x, x)
   z = em.reshape(y, (3, 1))
   ```

4. Update specialized operations:
   ```python
   # Before
   result = ops.linearalg.svd(z)
   
   # After
   result = em.linalg.svd(z)
   # or
   result = em.svd(z)
   ```

With these changes, your code will be cleaner, more intuitive, and easier to maintain.
