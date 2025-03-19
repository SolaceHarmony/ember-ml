"""NumPy tensor indexing operations."""

from typing import Any, List, Literal, Optional, Sequence, Union

import numpy as np

from ember_ml.backend.numpy.types import (
    TensorLike, Shape
)

def slice_tensor(tensor: TensorLike, starts: Shape, sizes: Shape) -> np.ndarray:
    """Extract a slice from a tensor."""
    # Convert input to NumPy array
    from ember_ml.backend.numpy.tensor.tensor import NumpyTensor
    Tensor = NumpyTensor()
    tensor_array = Tensor.convert_to_tensor(tensor)
    
    # Create a list of slice objects for each dimension
    slice_objects = []
    for i, (start, size) in enumerate(zip(starts, sizes)):
        if size == -1:
            # -1 means "all remaining elements in this dimension"
            slice_objects.append(slice(start, None))
        else:
            # Use np.add instead of + operator
            end = np.add(start, size)
            slice_objects.append(slice(start, end))
    
    # Extract the slice
    return tensor_array[tuple(slice_objects)]

def gather(tensor: TensorLike, indices: TensorLike, axis: int = 0) -> np.ndarray:
    """Gather slices from a tensor along an axis."""
    # Convert inputs to NumPy arrays
    from ember_ml.backend.numpy.tensor.tensor import NumpyTensor
    Tensor = NumpyTensor()
    tensor_array = Tensor.convert_to_tensor(tensor)
    indices_array = Tensor.convert_to_tensor(indices)
    
    # Ensure indices are integers
    indices_int = indices_array.astype(np.int64)
    
    # Use take operation for gathering
    return np.take(tensor_array, indices_int, axis=axis)

def tensor_scatter_nd_update(tensor: TensorLike, indices: TensorLike, updates: TensorLike) -> np.ndarray:
    """Update tensor elements at given indices."""
    # Convert inputs to NumPy arrays
    from ember_ml.backend.numpy.tensor.tensor import NumpyTensor
    Tensor = NumpyTensor()
    tensor_array = Tensor.convert_to_tensor(tensor)
    indices_array = Tensor.convert_to_tensor(indices)
    updates_array = Tensor.convert_to_tensor(updates)
    
    # Create a copy of the tensor
    result = tensor_array.copy()
    
    # Ensure indices are integers
    indices_array = indices_array.astype(np.int64)
    
    # Iterate over the indices and apply updates
    for i in range(indices_array.shape[0]):
        # Extract indices for this update
        idx = []
        for j in range(indices_array.shape[1]):
            # Get each dimension's index value
            idx.append(indices_array[i, j])
        
        # Apply the update directly using tuple indexing
        result[tuple(idx)] = updates_array[i]
    
    return result

def slice_update(tensor: TensorLike, slices: TensorLike, updates: Optional[TensorLike] = None) -> np.ndarray:
    """Update a tensor at specific indices."""
    # Convert inputs to NumPy arrays
    from ember_ml.backend.numpy.tensor.tensor import NumpyTensor
    Tensor = NumpyTensor()
    tensor_array = Tensor.convert_to_tensor(tensor)
    slices_array = Tensor.convert_to_tensor(slices)
    
    # Create a copy of the input tensor
    result = tensor_array.copy()
    
    # If updates is None, return slice of tensor
    if updates is None:
        # Handle scalar indices (0-d arrays)
        if slices_array.ndim == 0:
            # For scalar indices, just return the element at that index
            return tensor_array[int(slices_array)]
        else:
            # Create a list of slice objects for each dimension
            slice_objects = []
            for i, start in enumerate(slices_array):
                slice_objects.append(slice(start, start + 1))
            
            # Extract the slice
            return tensor_array[tuple(slice_objects)]
    
    # Convert updates to NumPy array
    updates_array = Tensor.convert_to_tensor(updates)
    
    # Create a list of slice objects for each dimension
    slice_objects = []
    for i, (start, size) in enumerate(zip(slices_array, updates_array.shape)):
        # Use np.add instead of + operator
        end = np.add(start, size)
        slice_objects.append(slice(start, end))
    
    # Update the tensor at the specified indices
    result[tuple(slice_objects)] = updates_array
    
    return result

def scatter(data: TensorLike, indices: TensorLike, dim_size: Optional[Union[int, np.ndarray]] = None,
           aggr: Literal["add", "max", "min", "mean", "softmax"] = "add", axis: int = 0) -> np.ndarray:
    """Scatter values into a new tensor."""
    # Convert inputs to NumPy arrays
    from ember_ml.backend.numpy.tensor.tensor import NumpyTensor
    Tensor = NumpyTensor()
    data_array = Tensor.convert_to_tensor(data)
    indices_array = Tensor.convert_to_tensor(indices)
    
    # Ensure indices are integers
    indices_int = indices_array.astype(np.int64)
    
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
                out_idx = tuple([idx] + [slice(None)] * (len(output_shape) - 1))
            else:
                # For other axes, we need to create more complex indexing
                idx_tuple = tuple(slice(None) if j != axis else i for j in range(data_array.ndim))
                src_value = data_array[idx_tuple]
                # Create the output index
                out_idx = tuple(slice(None) if j != axis else idx for j in range(output.ndim))
            
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

# Helper functions for scatter operations
def scatter_add(src: TensorLike, index: TensorLike, dim_size: int, axis: int = 0) -> np.ndarray:
    """Scatter values using addition."""
    return scatter(src, index, dim_size, "add", axis)

def scatter_max(src: TensorLike, index: TensorLike, dim_size: int, axis: int = 0) -> np.ndarray:
    """Scatter values using maximum."""
    return scatter(src, index, dim_size, "max", axis)

def scatter_min(src: TensorLike, index: TensorLike, dim_size: int, axis: int = 0) -> np.ndarray:
    """Scatter values using minimum."""
    return scatter(src, index, dim_size, "min", axis)

def scatter_mean(values: TensorLike, index: TensorLike, dim_size: int, axis: int = 0) -> np.ndarray:
    """Scatter values and compute mean."""
    # First compute sum
    sum_result = scatter(values, index, dim_size, "add", axis)
    
    # Then compute count
    from ember_ml.backend.numpy.tensor.tensor import NumpyTensor
    Tensor = NumpyTensor()
    ones = np.ones_like(Tensor.convert_to_tensor(values))
    count = scatter(ones, index, dim_size, "add", axis)
    
    # Avoid division by zero
    count = np.where(count == 0, np.ones_like(count), count)
    
    # Compute mean
    return np.divide(sum_result, count)

def scatter_softmax(values: TensorLike, index: TensorLike, dim_size: int, axis: int = 0) -> np.ndarray:
    """Scatter values and compute softmax."""
    # First compute max for numerical stability
    max_vals = scatter(values, index, dim_size, "max", axis)
    
    # Compute exp(x - max)
    from ember_ml.backend.numpy.tensor.tensor import NumpyTensor
    Tensor = NumpyTensor()
    values_array = Tensor.convert_to_tensor(values)
    exp_vals = np.exp(values_array - max_vals)
    
    # Sum exp values
    sum_exp = scatter(exp_vals, index, dim_size, "add", axis)
    
    # Compute softmax
    return np.divide(exp_vals, sum_exp)

# Alias for backward compatibility
slice = slice_tensor