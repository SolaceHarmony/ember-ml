"""NumPy tensor indexing operations."""

import numpy as np
from typing import Union, Sequence

# Type aliases
Shape = Union[int, Sequence[int]]

def slice_tensor(tensor_obj, tensor, starts, sizes):
    """
    Extract a slice from a tensor.
    
    Args:
        tensor_obj: NumpyTensor instance
        tensor: Input tensor
        starts: Starting indices for each dimension
        sizes: Size of the slice in each dimension. A value of -1 means "all remaining elements in this dimension"
        
    Returns:
        Sliced tensor
    """
    tensor_np = tensor_obj.convert_to_tensor(tensor)
    
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
    return tensor_np[tuple(slice_objects)]

def slice_update(tensor_obj, tensor, starts, updates):
    """
    Update a slice of a tensor with new values.
    
    Args:
        tensor_obj: NumpyTensor instance
        tensor: Input tensor to update
        starts: Starting indices for each dimension
        updates: Values to insert at the specified indices
        
    Returns:
        Updated tensor
    """
    tensor_np = tensor_obj.convert_to_tensor(tensor)
    updates_np = tensor_obj.convert_to_tensor(updates)
    
    # Create a copy of the input tensor
    result = tensor_np.copy()
    
    # Create a list of slice objects for each dimension
    slice_objects = []
    for i, (start, size) in enumerate(zip(starts, updates_np.shape)):
        # Use np.add instead of + operator
        end = np.add(start, size)
        slice_objects.append(slice(start, end))
    
    # Update the tensor at the specified indices
    result[tuple(slice_objects)] = updates_np
    
    return result

def gather(tensor_obj, tensor, indices, axis=0):
    """
    Gather slices from a tensor along an axis.
    
    Args:
        tensor_obj: NumpyTensor instance
        tensor: Input tensor
        indices: Indices of slices to gather
        axis: Axis along which to gather
        
    Returns:
        Gathered tensor
    """
    tensor_np = tensor_obj.convert_to_tensor(tensor)
    indices_np = tensor_obj.convert_to_tensor(indices)
    
    # Convert indices to integers
    indices_np = indices_np.astype(np.int64)
    
    return np.take(tensor_np, indices_np, axis=axis)

def scatter(tensor_obj, src, index, dim_size=None, aggr="add", axis=0):
    """
    Scatter values from src into a new tensor of size dim_size along the given axis.
    
    Args:
        tensor_obj: NumpyTensor instance
        src: Source tensor containing values to scatter
        index: Indices where to scatter the values
        dim_size: Size of the output tensor along the given axis. If None, uses the maximum index + 1
        aggr: Aggregation method to use for duplicate indices ("add", "max", "mean", "softmax", "min")
        axis: Axis along which to scatter
        
    Returns:
        Tensor with scattered values
    """
    # Convert inputs to NumPy arrays
    src_np = tensor_obj.convert_to_tensor(src)
    index_np = tensor_obj.convert_to_tensor(index)
    
    # Ensure indices are integers
    index_np = index_np.astype(np.int64)
    
    # Determine the output shape
    if dim_size is None:
        dim_size = np.add(np.max(index_np), 1)
    
    # Create output shape
    output_shape = list(src_np.shape)
    output_shape[axis] = dim_size
    
    # Initialize output tensor with zeros
    if aggr == "min":
        # For min aggregation, initialize with large values
        output = np.full(output_shape, np.finfo(src_np.dtype).max, dtype=src_np.dtype)
    else:
        output = np.zeros(output_shape, dtype=src_np.dtype)
    
    # Handle 1D case (most common)
    if index_np.ndim == 1:
        for i, idx in enumerate(index_np):
            # Select the appropriate slice from src_np
            if axis == 0:
                # If scattering along axis 0, select the i-th element
                src_value = src_np[i]
                # Create the output index
                # Use list and tuple conversion instead of + operator
                out_idx_list = [idx]
                for j in src_value.shape:
                    out_idx_list.append(range(j))
                out_idx = tuple(out_idx_list)
            else:
                # For other axes, we need to create more complex indexing
                idx_tuple = tuple(slice(None) if j != axis else i for j in range(src_np.ndim))
                src_value = src_np[idx_tuple]
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
    else:
        # Handle multi-dimensional indices
        # This is a simplified implementation that may not handle all cases
        for i in range(src_np.shape[0]):
            idx = index_np[i]
            src_value = src_np[i]
            
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

def tensor_scatter_nd_update(tensor_obj, tensor, indices, updates):
    """
    Updates values of a tensor at specified indices.
    
    Args:
        tensor_obj: NumpyTensor instance
        tensor: Input tensor to update
        indices: Indices at which to update values (N-dimensional indices)
        updates: Values to insert at the specified indices
        
    Returns:
        Updated tensor
    """
    # Create a copy of the tensor
    tensor_np = tensor_obj.convert_to_tensor(tensor)
    indices_np = tensor_obj.convert_to_tensor(indices)
    updates_np = tensor_obj.convert_to_tensor(updates)
    
    # Ensure indices are integers
    indices_np = indices_np.astype(np.int64)
    
    # Create a copy of the tensor
    result = tensor_np.copy()
    
    # Iterate over the indices and apply updates
    for i in range(indices_np.shape[0]):
        # Extract indices for this update
        idx = []
        for j in range(indices_np.shape[1]):
            # Get each dimension's index value
            idx.append(indices_np[i, j])
        
        # Apply the update directly using tuple indexing
        result[tuple(idx)] = updates_np[i]
    
    return result