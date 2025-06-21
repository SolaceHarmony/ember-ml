# Statistical Operations (`ops.stats`)

The `ops.stats` module provides a comprehensive set of statistical operations for tensor analysis. These operations are backend-agnostic and follow a consistent API, making it easy to work with tensors regardless of the underlying backend.

**Important Note on Input/Output Types:** Functions within the `ops.stats` module accept a variety of tensor-like inputs, including native backend tensors (e.g., `mlx.core.array`), `EmberTensor` objects, `Parameter` objects, NumPy arrays, and Python lists/tuples/scalars. The backend implementation automatically handles converting these inputs and unwrapping objects like `Parameter` and `EmberTensor` to access the underlying native tensor data needed for computation. These functions return results as **native backend tensors**, not `EmberTensor` instances.

## Importing

```python
from ember_ml import ops, tensor # For creating example tensors

# Access stats functions via ops.stats
# e.g., stats.mean(...)
```

## Available Functions

### Basic Statistics

#### `stats.mean(x, axis=None, keepdims=False)`

Compute the mean of a tensor along specified axes.

**Parameters:**
- `x`: Input tensor
- `axis`: Axis or axes along which to compute the mean. If None, compute over all axes.
- `keepdims`: Whether to keep the reduced dimensions with length 1.

**Returns:**
- Mean of the tensor.

**Example:**
```python
x = tensor.convert_to_tensor([[1, 2, 3], [4, 5, 6]])
mean_all = stats.mean(x)  # Mean of all elements
mean_rows = stats.mean(x, axis=1)  # Mean of each row
mean_cols = stats.mean(x, axis=0)  # Mean of each column
```

#### `stats.sum(x, axis=None, keepdims=False)`

Compute the sum along the specified axis.

**Parameters:**
- `x`: Input tensor
- `axis`: Axis or axes along which to compute the sum. If None, compute over all axes.
- `keepdims`: Whether to keep the reduced dimensions with length 1.

**Returns:**
- Sum of the tensor.

**Example:**
```python
x = tensor.convert_to_tensor([[1, 2, 3], [4, 5, 6]])
sum_all = stats.sum(x)  # Sum of all elements
sum_rows = stats.sum(x, axis=1)  # Sum of each row
sum_cols = stats.sum(x, axis=0)  # Sum of each column
```

#### `stats.var(x, axis=None, keepdims=False, ddof=0)`

Compute the variance along the specified axis.

**Parameters:**
- `x`: Input tensor
- `axis`: Axis or axes along which to compute the variance. If None, compute over all axes.
- `keepdims`: Whether to keep the reduced dimensions with length 1.
- `ddof`: Delta degrees of freedom. The divisor is `N - ddof`, where `N` is the number of elements.

**Returns:**
- Variance of the tensor.

**Example:**
```python
x = tensor.convert_to_tensor([[1, 2, 3], [4, 5, 6]])
var_all = stats.var(x)  # Variance of all elements
var_rows = stats.var(x, axis=1)  # Variance of each row
var_cols = stats.var(x, axis=0)  # Variance of each column
```

#### `stats.std(x, axis=None, keepdims=False, ddof=0)`

Compute the standard deviation along the specified axis.

**Parameters:**
- `x`: Input tensor
- `axis`: Axis or axes along which to compute the standard deviation. If None, compute over all axes.
- `keepdims`: Whether to keep the reduced dimensions with length 1.
- `ddof`: Delta degrees of freedom. The divisor is `N - ddof`, where `N` is the number of elements.

**Returns:**
- Standard deviation of the tensor.

**Example:**
```python
x = tensor.convert_to_tensor([[1, 2, 3], [4, 5, 6]])
std_all = stats.std(x)  # Standard deviation of all elements
std_rows = stats.std(x, axis=1)  # Standard deviation of each row
std_cols = stats.std(x, axis=0)  # Standard deviation of each column
```

#### `stats.median(x, axis=None, keepdims=False)`

Compute the median along the specified axis.

**Parameters:**
- `x`: Input tensor
- `axis`: Axis or axes along which to compute the median. If None, compute over all axes.
- `keepdims`: Whether to keep the reduced dimensions with length 1.

**Returns:**
- Median of the tensor.

**Example:**
```python
x = tensor.convert_to_tensor([[1, 2, 3], [4, 5, 6]])
median_all = stats.median(x)  # Median of all elements
median_rows = stats.median(x, axis=1)  # Median of each row
median_cols = stats.median(x, axis=0)  # Median of each column
```

#### `stats.percentile(x, q, axis=None, keepdims=False)`

Compute the q-th percentile along the specified axis.

**Parameters:**
- `x`: Input tensor
- `q`: Percentile(s) to compute, in range [0, 100]
- `axis`: Axis or axes along which to compute the percentile. If None, compute over all axes.
- `keepdims`: Whether to keep the reduced dimensions with length 1.

**Returns:**
- q-th percentile of the tensor.

**Example:**
```python
x = tensor.convert_to_tensor([[1, 2, 3], [4, 5, 6]])
p25 = stats.percentile(x, 25)  # 25th percentile
p50 = stats.percentile(x, 50)  # 50th percentile (same as median)
p75 = stats.percentile(x, 75)  # 75th percentile
```

#### `stats.min(x, axis=None, keepdims=False)`

Compute the minimum value along the specified axis.

**Parameters:**
- `x`: Input tensor
- `axis`: Axis or axes along which to compute the minimum. If None, compute over all axes.
- `keepdims`: Whether to keep the reduced dimensions with length 1.

**Returns:**
- Minimum value of the tensor.

**Example:**
```python
x = tensor.convert_to_tensor([[1, 2, 3], [4, 5, 6]])
min_all = stats.min(x)  # Minimum of all elements
min_rows = stats.min(x, axis=1)  # Minimum of each row
min_cols = stats.min(x, axis=0)  # Minimum of each column
```

#### `stats.max(x, axis=None, keepdims=False)`

Compute the maximum value along the specified axis.

**Parameters:**
- `x`: Input tensor
- `axis`: Axis or axes along which to compute the maximum. If None, compute over all axes.
- `keepdims`: Whether to keep the reduced dimensions with length 1.

**Returns:**
- Maximum value of the tensor.

**Example:**
```python
x = tensor.convert_to_tensor([[1, 2, 3], [4, 5, 6]])
max_all = stats.max(x)  # Maximum of all elements
max_rows = stats.max(x, axis=1)  # Maximum of each row
max_cols = stats.max(x, axis=0)  # Maximum of each column
```

### Cumulative Operations

#### `stats.cumsum(x, axis=None)`

Compute the cumulative sum along the specified axis.

**Parameters:**
- `x`: Input tensor
- `axis`: Axis along which to compute the cumulative sum. If None, the array is flattened.

**Returns:**
- Cumulative sum of the tensor.

**Example:**
```python
x = tensor.convert_to_tensor([1, 2, 3, 4, 5])
cumsum_result = stats.cumsum(x)  # [1, 3, 6, 10, 15]
```

### Indices and Sorting

#### `stats.argmax(x, axis=None, keepdims=False)`

Returns the indices of the maximum values along an axis.

**Parameters:**
- `x`: Input tensor
- `axis`: Axis along which to compute the argmax. If None, the array is flattened.
- `keepdims`: Whether to keep the reduced dimensions with length 1.

**Returns:**
- Indices of the maximum values.

**Example:**
```python
x = tensor.convert_to_tensor([[1, 5, 3], [4, 2, 6]])
argmax_all = stats.argmax(x)  # Index of maximum in flattened tensor
argmax_rows = stats.argmax(x, axis=1)  # Index of maximum in each row
argmax_cols = stats.argmax(x, axis=0)  # Index of maximum in each column
```

#### `stats.sort(x, axis=-1, descending=False)`

Sort a tensor along the specified axis.

**Parameters:**
- `x`: Input tensor
- `axis`: Axis along which to sort. Default is -1 (last axis).
- `descending`: Whether to sort in descending order.

**Returns:**
- Sorted tensor.

**Example:**
```python
x = tensor.convert_to_tensor([[3, 1, 2], [6, 5, 4]])
sorted_rows = stats.sort(x, axis=1)  # Sort each row
sorted_desc = stats.sort(x, descending=True)  # Sort in descending order
```

#### `stats.argsort(x, axis=-1, descending=False)`

Returns the indices that would sort a tensor along the specified axis.

**Parameters:**
- `x`: Input tensor
- `axis`: Axis along which to sort. Default is -1 (last axis).
- `descending`: Whether to sort in descending order.

**Returns:**
- Indices that would sort the tensor.

**Example:**
```python
x = tensor.convert_to_tensor([[3, 1, 2], [6, 5, 4]])
argsort_rows = stats.argsort(x, axis=1)  # Indices to sort each row
argsort_desc = stats.argsort(x, descending=True)  # Indices for descending sort
```

### Other

#### `stats.gaussian(x, mean=0.0, std=1.0)`

Apply Gaussian function element-wise to a tensor. Note: This is often categorized under vector operations but is aliased here.

**Parameters:**
- `x`: Input tensor
- `mean`: Mean of the Gaussian distribution
- `std`: Standard deviation of the Gaussian distribution

**Returns:**
- Tensor with Gaussian function applied.

## Notes on Tensor Operations vs. Statistical Operations

In Ember ML, there is a distinction between tensor operations and statistical operations:

- **Tensor Operations** (in `ember_ml.nn.tensor` module): These are operations that manipulate tensor structure, such as reshape, transpose, concatenate, etc., and typically return `EmberTensor` objects.

- **Statistical Operations** (in `ember_ml.ops.stats` module): These are operations that compute statistical properties of tensors, such as mean, variance, percentiles, etc., and return native backend tensors.

Sort and argsort are available in both modules, as they can be considered both tensor manipulation operations and statistical operations depending on the context. Check the specific module documentation for return type details if needed.