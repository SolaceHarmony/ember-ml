# Operations (ops) Module

The `ember_ml.ops` module provides a comprehensive set of backend-agnostic operations for tensor manipulation, mathematical operations, device management, and more. These operations follow a consistent API across different backends (NumPy, PyTorch, MLX).

## Importing

```python
from ember_ml import ops
```

## Mathematical Operations

| Function | Description |
|----------|-------------|
| `ops.add(x, y)` | Element-wise addition of tensors |
| `ops.subtract(x, y)` | Element-wise subtraction of tensors |
| `ops.multiply(x, y)` | Element-wise multiplication of tensors |
| `ops.divide(x, y)` | Element-wise division of tensors |
| `ops.floor_divide(x, y)` | Element-wise floor division of tensors |
| `ops.dot(x, y)` | Dot product of tensors |
| `ops.matmul(x, y)` | Matrix multiplication of tensors |
| `ops.gather(params, indices, axis=0)` | Gather slices from params according to indices |
| `ops.exp(x)` | Element-wise exponential of tensor |
| `ops.log(x)` | Element-wise natural logarithm of tensor |
| `ops.log10(x)` | Element-wise base-10 logarithm of tensor |
| `ops.log2(x)` | Element-wise base-2 logarithm of tensor |
| `ops.pow(x, y)` | Element-wise power function |
| `ops.sqrt(x)` | Element-wise square root of tensor |
| `ops.square(x)` | Element-wise square of tensor |
| `ops.abs(x)` | Element-wise absolute value of tensor |
| `ops.negative(x)` | Element-wise negation of tensor |
| `ops.sign(x)` | Element-wise sign of tensor |
| `ops.clip(x, min_val, max_val)` | Element-wise clipping of tensor values |
| `ops.gradient(f, x, dx=1.0, axis=None, edge_order=1)` | Compute the gradient of a function |
| `ops.eigh(x)` | Compute eigenvalues and eigenvectors of a Hermitian matrix |

### Trigonometric Functions

| Function | Description |
|----------|-------------|
| `ops.sin(x)` | Element-wise sine of tensor |
| `ops.cos(x)` | Element-wise cosine of tensor |
| `ops.tan(x)` | Element-wise tangent of tensor |
| `ops.sinh(x)` | Element-wise hyperbolic sine of tensor |
| `ops.cosh(x)` | Element-wise hyperbolic cosine of tensor |
| `ops.tanh(x)` | Element-wise hyperbolic tangent of tensor |

### Activation Functions

| Function | Description |
|----------|-------------|
| `ops.sigmoid(x)` | Element-wise sigmoid of tensor |
| `ops.softplus(x)` | Element-wise softplus of tensor |
| `ops.relu(x)` | Element-wise rectified linear unit of tensor |
| `ops.softmax(x, axis=-1)` | Softmax activation function |
| `ops.get_activation(activation)` | Get activation function by name |

## Device Operations

| Function | Description |
|----------|-------------|
| `ops.to_device(x, device)` | Move tensor to the specified device |
| `ops.get_device(x)` | Get the device of a tensor |
| `ops.get_available_devices()` | Get a list of available devices |
| `ops.memory_usage(device=None)` | Get memory usage for the specified device |
| `ops.memory_info(device=None)` | Get detailed memory information for the specified device |

## Feature Operations

| Function | Description |
|----------|-------------|
| `ops.pca(x, n_components=None, svd_solver='auto')` | Perform principal component analysis |
| `ops.transform(x, components)` | Transform data using principal components |
| `ops.inverse_transform(x, components, mean=None)` | Inverse transform data |
| `ops.standardize(x, mean=None, std=None, axis=0)` | Standardize data |
| `ops.normalize(x, norm='l2', axis=1)` | Normalize data |

## Comparison Operations

| Function | Description |
|----------|-------------|
| `ops.equal(x, y)` | Element-wise equality comparison |
| `ops.not_equal(x, y)` | Element-wise inequality comparison |
| `ops.less(x, y)` | Element-wise less-than comparison |
| `ops.less_equal(x, y)` | Element-wise less-than-or-equal comparison |
| `ops.greater(x, y)` | Element-wise greater-than comparison |
| `ops.greater_equal(x, y)` | Element-wise greater-than-or-equal comparison |
| `ops.logical_and(x, y)` | Element-wise logical AND |
| `ops.logical_or(x, y)` | Element-wise logical OR |
| `ops.logical_not(x)` | Element-wise logical NOT |
| `ops.logical_xor(x, y)` | Element-wise logical XOR |
| `ops.allclose(x, y, rtol=1e-5, atol=1e-8)` | Returns whether all elements are close |
| `ops.isclose(x, y, rtol=1e-5, atol=1e-8)` | Returns whether each element is close |
| `ops.all(x, axis=None, keepdims=False)` | Test whether all elements evaluate to True |
| `ops.where(condition, x, y)` | Return elements chosen from x or y depending on condition |
| `ops.isnan(x)` | Test element-wise for NaN |

## I/O Operations

| Function | Description |
|----------|-------------|
| `ops.save(obj, path)` | Save object to file |
| `ops.load(path)` | Load object from file |

## Loss Operations

| Function | Description |
|----------|-------------|
| `ops.mean_squared_error(y_true, y_pred)` | Mean squared error loss |
| `ops.mean_absolute_error(y_true, y_pred)` | Mean absolute error loss |
| `ops.binary_crossentropy(y_true, y_pred, from_logits=False)` | Binary crossentropy loss |
| `ops.categorical_crossentropy(y_true, y_pred, from_logits=False)` | Categorical crossentropy loss |
| `ops.sparse_categorical_crossentropy(y_true, y_pred, from_logits=False)` | Sparse categorical crossentropy loss |
| `ops.huber_loss(y_true, y_pred, delta=1.0)` | Huber loss |
| `ops.log_cosh_loss(y_true, y_pred)` | Logarithm of the hyperbolic cosine loss |

## Vector Operations

| Function | Description |
|----------|-------------|
| `ops.normalize_vector(x, axis=None)` | Normalize a vector or matrix |
| `ops.compute_energy_stability(x, axis=None)` | Compute energy stability of a vector |
| `ops.compute_interference_strength(x, y)` | Compute interference strength between vectors |
| `ops.compute_phase_coherence(x, y)` | Compute phase coherence between vectors |
| `ops.partial_interference(x, y, mask)` | Compute partial interference between vectors |
| `ops.euclidean_distance(x, y)` | Compute Euclidean distance between vectors |
| `ops.cosine_similarity(x, y)` | Compute cosine similarity between vectors |
| `ops.exponential_decay(x, rate=0.1)` | Apply exponential decay to a vector |
| `ops.gaussian(x, mean=0.0, std=1.0)` | Apply Gaussian function to a vector |

## Backend Management

| Function | Description |
|----------|-------------|
| `ops.get_ops()` | Get the current ops implementation name |
| `ops.set_ops(ops_name)` | Set the current ops implementation |
| `ops.set_backend(backend_name)` | Set the current backend |

## Examples

### Basic Mathematical Operations

```python
import numpy as np
from ember_ml import ops
from ember_ml.nn import tensor

# Create tensors
x = tensor.convert_to_tensor([1, 2, 3])
y = tensor.convert_to_tensor([4, 5, 6])

# Add tensors
result = ops.add(x, y)  # [5, 7, 9]

# Matrix multiplication
a = tensor.convert_to_tensor([[1, 2], [3, 4]])
b = tensor.convert_to_tensor([[5, 6], [7, 8]])
result = ops.matmul(a, b)  # [[19, 22], [43, 50]]
```

### Activation Functions

```python
import numpy as np
from ember_ml import ops
from ember_ml.nn import tensor

# Create a tensor
x = tensor.convert_to_tensor([-2, -1, 0, 1, 2])

# Apply activation functions
sigmoid_result = ops.sigmoid(x)  # [0.12, 0.27, 0.5, 0.73, 0.88]
relu_result = ops.relu(x)  # [0, 0, 0, 1, 2]
tanh_result = ops.tanh(x)  # [-0.96, -0.76, 0, 0.76, 0.96]
```

### Device Management

```python
from ember_ml import ops

# Get available devices
devices = ops.get_available_devices()  # ['cpu', 'cuda:0', 'mps']

# Check memory usage
memory_info = ops.memory_info('cuda:0')
print(f"Total memory: {memory_info['total']}")
print(f"Used memory: {memory_info['used']}")
print(f"Free memory: {memory_info['free']}")
```

### Loss Functions

```python
from ember_ml import ops
from ember_ml.nn import tensor

# Create true and predicted values
y_true = tensor.convert_to_tensor([0, 1, 0, 1])
y_pred = tensor.convert_to_tensor([0.1, 0.9, 0.2, 0.8])

# Compute loss
mse = ops.mean_squared_error(y_true, y_pred)
mae = ops.mean_absolute_error(y_true, y_pred)
bce = ops.binary_crossentropy(y_true, y_pred)
```

## Notes

- All operations are backend-agnostic and work with any backend (NumPy, PyTorch, MLX).
- The operations follow a consistent API across different backends.
- Most operations support broadcasting, similar to NumPy and other array libraries.
- For tensor creation and manipulation, use the `ember_ml.nn.tensor` module.
- For statistical operations, use the `ember_ml.ops.stats` module.
- For linear algebra operations, use the `ember_ml.ops.linearalg` module.