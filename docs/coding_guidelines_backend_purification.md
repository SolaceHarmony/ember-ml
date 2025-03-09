# Backend Purification Coding Guidelines

This document provides guidelines for writing backend-agnostic code in the EmberHarmony framework. Following these guidelines will help ensure that your code works consistently across different backends (NumPy, PyTorch, MLX) while maintaining precision and performance.

## Core Principles

1. **Stay within the ops abstraction layer**
2. **Avoid unnecessary type conversions**
3. **Preserve numerical precision**
4. **Optimize for backend-specific execution**

## Avoiding NumPy Dependencies

### ❌ Bad Patterns

```python
# Direct NumPy import
import numpy as np

# Using NumPy functions directly
def calculate_statistics(data):
    mean = np.mean(data)
    std = np.std(data)
    return mean, std

# Converting tensors to NumPy
def process_tensor(tensor):
    numpy_array = tensor.numpy()  # Conversion to NumPy
    result = numpy_array * 2
    return torch.from_numpy(result)  # Conversion back to tensor
```

### ✅ Good Patterns

```python
# Use the ops abstraction layer
from emberharmony import ops

# Use ops functions instead of NumPy
def calculate_statistics(data):
    mean = ops.mean(data)
    # Calculate standard deviation using ops
    variance = ops.mean(ops.square(ops.subtract(data, mean)))
    std = ops.sqrt(variance)
    return mean, std

# Keep operations within the tensor framework
def process_tensor(tensor):
    return ops.multiply(tensor, 2)
```

## Preserving Numerical Precision

### ❌ Precision-Reducing Patterns

```python
# Converting to Python scalar types loses precision
def normalize(tensor):
    min_val = float(ops.min(tensor))  # Precision loss
    max_val = float(ops.max(tensor))  # Precision loss
    range_val = max_val - min_val
    return ops.divide(ops.subtract(tensor, min_val), range_val)

# Integer division can lose precision
def calculate_average(values):
    return ops.sum(values) / len(values)  # Integer division if len(values) is an integer
```

### ✅ Precision-Preserving Patterns

```python
# Keep values as tensors to preserve precision
def normalize(tensor):
    min_val = ops.min(tensor)
    max_val = ops.max(tensor)
    range_val = ops.subtract(max_val, min_val)
    return ops.divide(ops.subtract(tensor, min_val), range_val)

# Use ops.mean or ensure float division
def calculate_average(values):
    return ops.mean(values)  # Preferred approach
    # Or if you need to use division:
    # return ops.divide(ops.sum(values), ops.full((), len(values), dtype=ops.float32))
```

## Performance Considerations

### ❌ Inefficient Patterns

```python
# Unnecessary conversions between backends
def process_data(tensor):
    # Convert to NumPy, process, then convert back
    numpy_data = tensor.numpy()
    result = np.log(1 + np.abs(numpy_data))
    return ops.convert_to_tensor(result)

# Repeated small operations instead of vectorized operations
def calculate_distances(points):
    distances = []
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            dist = ops.sqrt(ops.sum(ops.square(ops.subtract(points[i], points[j]))))
            distances.append(dist)
    return distances

# Using Python operators for arithmetic
def normalize_data(tensor, mean, std):
    # Python operators force eager execution and break lazy evaluation
    return (tensor - mean) / std  # Uses Python's - and / operators
```

### ✅ Efficient Patterns

```python
# Stay within the same backend
def process_data(tensor):
    # All operations use ops
    return ops.log(ops.add(1, ops.abs(tensor)))

# Use vectorized operations
def calculate_distances(points):
    # Expand dimensions to create pairs
    points_i = ops.expand_dims(points, axis=1)  # Shape: [N, 1, D]
    points_j = ops.expand_dims(points, axis=0)  # Shape: [1, N, D]
    
    # Calculate pairwise differences and distances in one vectorized operation
    diff = ops.subtract(points_i, points_j)  # Shape: [N, N, D]
    sq_diff = ops.square(diff)  # Shape: [N, N, D]
    sq_dist = ops.sum(sq_diff, axis=2)  # Shape: [N, N]
    
    # Extract upper triangular part (excluding diagonal)
    indices = ops.triu_indices(len(points), k=1)
    distances = ops.sqrt(sq_dist[indices])
    
    return distances

# Use ops functions for arithmetic
def normalize_data(tensor, mean, std):
    # Using ops functions preserves lazy evaluation in supported backends
    return ops.divide(ops.subtract(tensor, mean), std)
```

## Avoiding Python Operators

### ❌ Python Operators (Break Lazy Evaluation)

```python
# These operators force eager execution and break optimization opportunities
def compute_loss(predictions, targets):
    diff = predictions - targets  # Python subtraction operator
    squared_diff = diff ** 2      # Python power operator
    loss = squared_diff.mean()    # Method call is okay if it's a backend method
    return loss

# Mixing Python operators with ops functions
def mixed_operations(a, b, c):
    # This breaks the computation graph and forces intermediate results
    return ops.multiply(a + b, c)  # Python addition breaks lazy evaluation chain
```

### ✅ Ops Functions (Preserve Lazy Evaluation)

```python
# These preserve the computation graph and enable optimizations
def compute_loss(predictions, targets):
    diff = ops.subtract(predictions, targets)
    squared_diff = ops.square(diff)  # or ops.power(diff, 2)
    loss = ops.mean(squared_diff)
    return loss

# Consistent use of ops functions
def mixed_operations(a, b, c):
    # This preserves the computation graph
    return ops.multiply(ops.add(a, b), c)
```

### Why This Matters

Using Python operators (`+`, `-`, `*`, `/`, `**`) instead of ops functions has several negative consequences:

1. **Breaks Lazy Evaluation**: MLX and PyTorch can optimize operations by fusing them together, but only if they're part of the same computation graph.

2. **Forces Synchronization**: Python operators force the evaluation of tensors, which can cause unnecessary synchronization between CPU and accelerators.

3. **Prevents Optimizations**: Backend-specific optimizations like kernel fusion, memory access optimization, and parallel execution are lost.

4. **Performance Impact**: The difference can be dramatic - up to 10-100x slower for complex operations due to the overhead of multiple kernel launches and memory transfers.

Even simple operations like `tensor + 1` should be written as `ops.add(tensor, 1)` to preserve these optimization opportunities.

## Type Handling

### ❌ Problematic Type Handling

```python
# Using NumPy dtypes
from numpy import dtype as np_dtype

def create_tensor(shape, dtype=np_dtype('float32')):
    return ops.zeros(shape, dtype=dtype)

# Manual type checking with isinstance
def process_input(x):
    if isinstance(x, np.ndarray):
        # NumPy-specific code
        pass
    elif isinstance(x, torch.Tensor):
        # PyTorch-specific code
        pass
    elif hasattr(x, 'numpy'):  # Check for MLX array
        # MLX-specific code
        pass
```

### ✅ Backend-Agnostic Type Handling

```python
# Use ops dtypes
def create_tensor(shape, dtype=ops.float32):
    return ops.zeros(shape, dtype=dtype)

# Use ops conversion functions
def process_input(x):
    # Convert to tensor in the current backend
    tensor = ops.convert_to_tensor(x)
    # Process using ops functions
    result = ops.some_operation(tensor)
    return result
```

## Random Operations

### ❌ Backend-Specific Random Operations

```python
# Direct use of backend-specific random functions
def generate_samples(shape):
    if get_backend() == 'numpy':
        return np.random.normal(0, 1, shape)
    elif get_backend() == 'torch':
        return torch.randn(shape)
    elif get_backend() == 'mlx':
        return mx.random.normal(shape=shape)
```

### ✅ Backend-Agnostic Random Operations

```python
# Use ops random functions
def generate_samples(shape):
    return ops.random_normal(shape, mean=0.0, stddev=1.0)
```

## Handling Device Placement

### ❌ Backend-Specific Device Handling

```python
# Direct use of backend-specific device placement
def place_on_device(tensor, device='cuda'):
    if get_backend() == 'numpy':
        # NumPy doesn't have device placement
        return tensor
    elif get_backend() == 'torch':
        return tensor.to(device)
    elif get_backend() == 'mlx':
        # MLX has different device handling
        return mx.array(tensor, device=device)
```

### ✅ Backend-Agnostic Device Handling

```python
# Use ops device functions
def place_on_device(tensor, device='cuda'):
    return ops.to_device(tensor, device)
```

## Impact on Precision and Speed

### Precision Impact Examples

| Pattern | Impact on Precision | Example |
|---------|---------------------|---------|
| `float(tensor)` | Loss of precision, especially for large or small values | `1e20` becomes `1.0e20` which may lose least significant digits |
| `tensor.numpy()` | Potential loss of precision during conversion | Half-precision (float16) tensors may be converted to higher precision, masking precision issues |
| Integer division | Truncation of fractional part | `3/2` becomes `1` instead of `1.5` |
| Mixed precision operations | Results computed at lowest precision | `float16 + float32` computed at float16 precision |

### Performance Impact Examples

| Pattern | Impact on Performance | Example Slowdown |
|---------|------------------------|------------------|
| Backend switching | Memory transfers, serialization overhead | Up to 10-100x slower for large tensors |
| Python loops vs. vectorized ops | Interpreter overhead vs. optimized kernels | 10-1000x slower depending on operation |
| Unnecessary copies | Memory bandwidth and allocation overhead | 2-5x slower for large tensors |
| Scalar extraction | Forces synchronization in async backends | Can block entire pipeline, causing 10-100x slowdown |
| Python operators (+, -, *, /) | Breaks lazy evaluation, prevents kernel fusion | 2-10x slower for simple ops, up to 100x for complex chains |
| Mixed ops/Python operations | Fragments computation graph, forces materialization | 5-20x slower due to extra memory transfers |

## Testing for Backend Independence

1. **Run with different backends**: Test your code with all supported backends
2. **Check numerical consistency**: Results should be identical (or very close) across backends
3. **Use the detection tool**: Run `utils/detect_numpy_usage.py` on your code to find issues
4. **Test with different precision**: Run with float32 and float16 to catch precision issues

## Common Pitfalls and Solutions

| Pitfall | Solution |
|---------|----------|
| Implicit NumPy dependency in third-party libraries | Wrap library calls in backend-specific adapter functions |
| Hardcoded data types | Use ops.dtype constants (ops.float32, ops.int64, etc.) |
| Backend-specific tensor attributes | Use ops utility functions (ops.shape, ops.ndim, etc.) |
| Scalar indexing that forces evaluation | Keep operations vectorized as long as possible |
| In-place operations | Use functional style with explicit outputs |
| Python arithmetic operators (+, -, *, /, **) | Use ops.add, ops.subtract, ops.multiply, ops.divide, ops.power |
| Mixing Python operators with ops functions | Consistently use ops functions for all operations in a computation chain |
| Assuming eager execution | Design for lazy evaluation to enable backend optimizations |

## Conclusion

Following these guidelines will help ensure that your code works consistently across different backends while maintaining precision and performance. The key is to stay within the ops abstraction layer and avoid unnecessary conversions or backend-specific code.

Remember that backend purification is not just about removing direct NumPy imports, but about writing truly backend-agnostic code that can run efficiently on any supported hardware.