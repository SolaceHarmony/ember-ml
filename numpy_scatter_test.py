"""
Standalone test for NumPy scatter operations.
This will help us debug and fix the implementation without EmberTensor overhead.
"""

import numpy as np
from builtins import slice as py_slice

def scatter(data, indices, dim_size=None, aggr="add", axis=0):
    """
    Scatter values from data into a new tensor.
    
    Args:
        data: Source tensor containing values to scatter
        indices: Indices where to scatter the values
        dim_size: Size of the output tensor along the given axis. If None, uses the maximum index + 1
        aggr: Aggregation method to use for duplicate indices ("add", "max", "min", "mean", "softmax")
        axis: Axis along which to scatter
        
    Returns:
        Tensor with scattered values
    """
    # Convert inputs to NumPy arrays
    data_array = np.array(data, dtype=np.float32)
    indices_array = np.array(indices, dtype=np.int32)
    
    # Ensure indices are integers
    indices_int = indices_array.astype(np.int32)
    
    # Handle dim_size
    if dim_size is None:
        computed_dim_size = int(np.max(indices_int) + 1)
    else:
        computed_dim_size = int(dim_size)
    
    # Create output shape
    output_shape = list(data_array.shape)
    output_shape[axis] = computed_dim_size
    
    # Initialize output tensor based on operation
    if aggr == "add" or aggr == "mean" or aggr == "softmax":
        output = np.zeros(output_shape, dtype=data_array.dtype)
    elif aggr == "max":
        output = np.full(output_shape, -np.inf, dtype=data_array.dtype)
    elif aggr == "min":
        output = np.full(output_shape, np.inf, dtype=data_array.dtype)
    else:
        raise ValueError(f"Unknown operation: {aggr}")
    
    # Handle 1D case (most common)
    if indices_int.ndim == 1:
        for i, idx in enumerate(indices_int):
            # Select the appropriate slice from data_array
            if axis == 0:
                # If scattering along axis 0, select the i-th element
                src_value = data_array[i]
                # Create the output index
                out_idx = tuple([idx] + [py_slice(None)] * (len(output_shape) - 1))
            else:
                # For other axes, we need to create more complex indexing
                idx_tuple = tuple(py_slice(None) if j != axis else i for j in range(data_array.ndim))
                src_value = data_array[idx_tuple]
                # Create the output index
                out_idx = tuple(py_slice(None) if j != axis else idx for j in range(output.ndim))
            
            # Apply the aggregation method
            if aggr == "add":
                output[out_idx] += src_value
            elif aggr == "max":
                output[out_idx] = np.maximum(output[out_idx], src_value)
            elif aggr == "min":
                output[out_idx] = np.minimum(output[out_idx], src_value)
            elif aggr == "mean":
                # For mean, we need to count occurrences and divide later
                output[out_idx] += src_value
                # TODO: Implement proper mean aggregation
            elif aggr == "softmax":
                # TODO: Implement softmax aggregation
                output[out_idx] += src_value
    
    return output

def scatter_add(src, index, dim_size=None, axis=0):
    """Scatter values using addition."""
    return scatter(src, index, dim_size, "add", axis)

def scatter_max(src, index, dim_size=None, axis=0):
    """Scatter values using maximum."""
    return scatter(src, index, dim_size, "max", axis)

def scatter_min(src, index, dim_size=None, axis=0):
    """Scatter values using minimum."""
    return scatter(src, index, dim_size, "min", axis)

def scatter_mean(values, index, dim_size=None, axis=0):
    """Scatter values and compute mean."""
    # Debug prints
    print("\n== NumPy scatter_mean debug ==")
    print("values:", values)
    print("index:", index)
    print("dim_size:", dim_size)
    
    # Use a dictionary to group values by their target indices
    values_by_idx = {}
    for i, idx in enumerate(index):
        idx_val = int(idx)
        if idx_val not in values_by_idx:
            values_by_idx[idx_val] = []
        values_by_idx[idx_val].append(values[i])
    
    print("Values grouped by index:", values_by_idx)
    
    # Determine output size
    if dim_size is None:
        computed_dim_size = max(values_by_idx.keys()) + 1
    else:
        computed_dim_size = dim_size
    
    # Create output array and compute means
    result = np.zeros(computed_dim_size, dtype=np.float32)
    for idx, vals in values_by_idx.items():
        result[idx] = np.mean(vals)
    
    print("Final result:", result)
    return result

def scatter_softmax(values, index, dim_size=None, axis=0):
    """Scatter values and compute softmax."""
    # First compute max for numerical stability
    max_vals = scatter(values, index, dim_size, "max", axis)
    
    # Compute exp(x - max)
    values_array = np.array(values, dtype=np.float32)
    exp_vals = np.exp(values_array - max_vals)
    
    # Sum exp values
    sum_exp = scatter(exp_vals, index, dim_size, "add", axis)
    
    # Compute softmax
    return np.divide(exp_vals, sum_exp)

# Test functions
def test_scatter_add():
    print("\nTesting scatter_add")
    src = np.array([1.0, 0.0, 2.0], dtype=np.float32)
    index = np.array([0, 2, 0], dtype=np.int32)
    result = scatter_add(src, index, dim_size=3)
    print("Result:", result)
    expected = np.array([3.0, 0.0, 0.0], dtype=np.float32)
    print("Expected:", expected)
    print("Match:", np.allclose(result, expected))

def test_scatter_max():
    print("\nTesting scatter_max")
    src = np.array([1.0, 5.0, 3.0], dtype=np.float32)
    index = np.array([0, 0, 1], dtype=np.int32)
    result = scatter_max(src, index, dim_size=2)
    print("Result:", result)
    expected = np.array([5.0, 3.0], dtype=np.float32)
    print("Expected:", expected)
    print("Match:", np.allclose(result, expected))

def test_scatter_mean():
    print("\nTesting scatter_mean")
    src = np.array([1.0, 5.0, 3.0], dtype=np.float32)
    index = np.array([0, 0, 1], dtype=np.int32)
    result = scatter_mean(src, index, dim_size=2)
    print("Result:", result)
    expected = np.array([3.0, 3.0], dtype=np.float32)
    print("Expected:", expected)
    print("Match:", np.allclose(result, expected))

def test_multi_dim():
    print("\nTesting multi-dimensional scatter")
    src = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    index = np.array([0, 2, 0], dtype=np.int32)
    result = scatter_add(src, index, dim_size=3, axis=0)
    print("Result:", result)
    expected = np.array([[6.0, 8.0], [0.0, 0.0], [3.0, 4.0]], dtype=np.float32)
    print("Expected:", expected)
    print("Match:", np.allclose(result, expected))

if __name__ == "__main__":
    test_scatter_add()
    test_scatter_max()
    test_scatter_mean()
    test_multi_dim()