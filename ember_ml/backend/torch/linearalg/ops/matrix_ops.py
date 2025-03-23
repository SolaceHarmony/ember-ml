"""
PyTorch matrix operations for ember_ml.

This module provides PyTorch implementations of matrix operations.
"""

import torch
from typing import Union, Tuple, Optional, Literal

# Import from tensor_ops
from ember_ml.backend.torch.types import TensorLike
from ember_ml.backend.torch.linearalg.ops.decomp_ops import svd
from ember_ml.backend.torch.tensor import TorchDType
from ember_ml.backend.torch.types import OrdLike

dtype_obj = TorchDType()

def norm(x: TensorLike, 
         ord: OrdLike = None, 
         axis: Optional[Union[int, Tuple[int, ...]]] = None, 
         keepdim: bool = False) -> torch.Tensor:    
    """
    Compute the matrix or vector norm.
    
    Args:
        x: Input matrix or vector
        ord: Order of the norm
        axis: Axis along which to compute the norm
        keepdim: Whether to keep the reduced dimensions
    
    Returns:
        Norm of the matrix or vector
    """
    # Convert input to torch array
    from ember_ml.backend.torch.tensor import TorchTensor
    Tensor = TorchTensor()
    x_array = Tensor.convert_to_tensor(x)
    
    # Default values
    if ord is None:
        if axis is None:
            # Default to Frobenius norm for matrices, L2 norm for vectors
            if x_array.ndim > 1:  # Use ndim instead of len(shape)
                ord = 'fro'
            else:
                ord = 2
        else:
            # Default to L2 norm along the specified axis
            ord = 2
    
    # Vector norm
    if axis is not None or x_array.ndim == 1:  # Use ndim instead of len(shape)
        if axis is None:
            axis = 0
        
        if ord == 'inf':
            # L-infinity norm (maximum absolute value)
            result = torch.max(torch.abs(x_array), axis=axis)
        elif ord == 1:
            # L1 norm (sum of absolute values)
            result = torch.sum(torch.abs(x_array), axis=axis)
        elif ord == 2:
            # L2 norm (Euclidean norm)
            result = torch.sqrt(torch.sum(torch.square(x_array), axis=axis))
        else:
            # General Lp norm
            if isinstance(ord, (int, float)):
                result = torch.power(
                    torch.sum(torch.power(torch.abs(x_array), ord), axis=axis),
                    torch.divide(torch.tensor(1.0), torch.tensor(ord))
                )
            else:
                # Handle case where ord is a string (shouldn't happen after our fixes)
                raise ValueError(f"Invalid norm order: {ord}")
    
    # Matrix norm
    else:
        if ord == 'fro':
            # Frobenius norm
            result = torch.sqrt(torch.sum(torch.square(x_array)))
        elif ord == 'nuc':
            # Nuclear norm (sum of singular values)
            s_values = svd(x_array, compute_uv=False)
            if isinstance(s_values, tuple):
                # Handle case where svd returns a tuple
                result = torch.sum(s_values[0])
            else:
                # Handle case where svd returns an array
                result = torch.sum(s_values)
        elif ord == 1:
            # Maximum absolute column sum
            result = torch.max(torch.sum(torch.abs(x_array), axis=0))
        elif ord == 'inf':
            # Maximum absolute row sum
            result = torch.max(torch.sum(torch.abs(x_array), axis=1))
        elif ord == -1:
            # Minimum absolute column sum
            result = torch.min(torch.sum(torch.abs(x_array), axis=0))
        elif ord == '-inf':
            # Minimum absolute row sum
            result = torch.min(torch.sum(torch.abs(x_array), axis=1))
        else:
            # For other matrix norms, use the singular values
            s_values = svd(x_array, compute_uv=False)
            if isinstance(s_values, tuple):
                # Handle case where svd returns a tuple
                s_array = s_values[0]
            else:
                # Handle case where svd returns an array
                s_array = s_values
                
            if ord == 2:
                # Spectral norm (maximum singular value)
                result = s_array[0]
            elif ord == -2:
                # Minimum singular value
                result = s_array[-1]
            else:
                raise ValueError(f"Invalid norm order: {ord}")
    
    # Keep dimensions if requested
    if keepdim and axis is not None:
        # Reshape to keep dimensions
        if isinstance(axis, tuple):
            shape = list(x_array.shape)
            for ax in sorted(axis, reverse=True):
                shape[ax] = 1
            result = torch.reshape(result, tuple(shape))
        else:
            shape = list(x_array.shape)
            shape[axis] = 1
            result = torch.reshape(result, tuple(shape))
    
    return result    

def det(a: TensorLike) -> torch.Tensor:
    """
    Compute the determinant of a square matrix.
    
    Args:
        a: Input square matrix
        
    Returns:
        Determinant of the matrix
    """
    # Convert input to torch.Tensor array
    from ember_ml.backend.torch.tensor import TorchTensor
    Tensor = TorchTensor()
    a_array = Tensor.convert_to_tensor(a)

    
    # Get matrix dimensions
    n = a_array.shape[0]
    assert a_array.shape[1] == n, "Matrix must be square"
    
    # Special cases for small matrices
    if torch.equal(n, torch.tensor(1)):
        return a_array[0, 0]
    elif torch.equal(n, torch.tensor(2)):
        term1 = torch.multiply(a_array[0, 0], a_array[1, 1])
        term2 = torch.multiply(a_array[0, 1], a_array[1, 0])
        return torch.subtract(term1, term2)
    
    # For larger matrices, use LU decomposition
    # This is a simplified implementation and may not be numerically stable
    # For a more robust implementation, consider using a dedicated algorithm
    
    # Make a copy of the matrix
    a_copy = torch.tensor(a_array)
    
    # Initialize determinant
    det_value = torch.tensor(1.0, dtype=a_array.dtype)
    
    # Gaussian elimination
    for i in range(n):
        # Find pivot
        pivot = a_copy[i, i]
        
        # Update determinant
        det_value = torch.multiply(det_value, pivot)
        
        # If pivot is zero, determinant is zero
        if torch.less(torch.abs(pivot), torch.tensor(1e-10)):
            return torch.tensor(0.0, dtype=a_array.dtype)
        
        # Eliminate below
        # Use direct integer calculation
        i_plus_1_int = i + 1
        for j in range(i_plus_1_int, n):
            factor = torch.divide(a_copy[j, i], pivot)
            
            # Calculate the new row
            new_row = torch.subtract(a_copy[j, i:], torch.multiply(factor, a_copy[i, i:]))
            
            # Update a_copy using direct indexing
            for k in range(i, n):
                a_copy[j, k] = new_row[k - i]
    
    return det_value

def diag(x: TensorLike, k: int = 0) -> torch.Tensor:
    """
    Extract a diagonal or construct a diagonal matrix.
    
    Args:
        x: Input array. If x is 2-D, return the k-th diagonal.
           If x is 1-D, return a 2-D array with x on the k-th diagonal.
        k: Diagonal offset. Use k>0 for diagonals above the main diagonal,
           and k<0 for diagonals below the main diagonal.
            
    Returns:
        The extracted diagonal or constructed diagonal matrix.
    """
    # Convert input to torch array
    from ember_ml.backend.torch.tensor import TorchTensor
    Tensor = TorchTensor()
    x_array = Tensor.convert_to_tensor(x)
    
    # Check if input is 1-D or 2-D
    if x_array.ndim == 1:
        # Construct a diagonal matrix
        n = x_array.shape[0]
        
        # Calculate the size of the output matrix
        m = torch.where(torch.greater_equal(torch.tensor(k), torch.tensor(0)),
                     torch.add(n, k),
                     torch.subtract(n, -k))
            
        # Ensure we use a compatible dtype (not int64)
        dtype = x_array.dtype
        if dtype == torch.int64:
            dtype = torch.int32
            
        # Create a zero matrix
        result = torch.zeros((m, m), dtype=dtype)
        
        # Import the scatter function from indexing
        from ember_ml.backend.torch.tensor.ops.indexing import scatter_add
        
        # Fill the diagonal
        # Use torch.greater_equal for comparison
        is_non_negative = torch.greater_equal(torch.tensor(k), torch.tensor(0))
        
        for i in range(n):
            # Create a copy of the result
            result_copy = result.clone()
            
            # Use torch.where to conditionally select the indices
            row = torch.where(is_non_negative,
                          torch.tensor(i),
                          torch.subtract(torch.tensor(i), torch.tensor(k)))
            col = torch.where(is_non_negative,
                          torch.add(torch.tensor(i), torch.tensor(k)),
                          torch.tensor(i))
            
            # Update the element directly
            result_copy[int(row.item()), int(col.item())] += x_array[i].item()            
            result = result_copy
                
        return result
    
    elif x_array.ndim == 2:
        # Extract a diagonal
        rows, cols = x_array.shape
        
        # Calculate the length of the diagonal
        # Use torch.greater_equal, torch.subtract, torch.add, and torch.minimum for operations
        is_non_negative = torch.greater_equal(torch.tensor(k), torch.tensor(0))
        diag_len_if_positive = torch.minimum(torch.tensor(rows), torch.subtract(torch.tensor(cols), torch.tensor(k)))
        diag_len_if_negative = torch.minimum(torch.add(torch.tensor(rows), torch.tensor(k)), torch.tensor(cols))
        diag_len = torch.where(is_non_negative, diag_len_if_positive, diag_len_if_negative).item()
            
        # Use torch.less_equal for comparison
        if torch.less_equal(torch.tensor(diag_len), torch.tensor(0)):
            # Empty diagonal
            return torch.tensor([], dtype=x_array.dtype)
            
        # Ensure we use a compatible dtype (not int64)
        dtype = x_array.dtype
        if dtype == torch.int64:
            dtype = torch.int32
            
        # Create an array to hold the diagonal
        result = torch.zeros((diag_len,), dtype=dtype)
        
        # Extract the diagonal
        # Use torch.greater_equal for comparison
        is_non_negative = torch.greater_equal(torch.tensor(k), torch.tensor(0))
        
        for i in range(diag_len):
            # Create a copy of the result
            result_copy = torch.tensor(result)
            
            # Use torch.where to conditionally select the indices
            row = torch.where(is_non_negative,
                          torch.tensor(i),
                          torch.subtract(torch.tensor(i), torch.tensor(k)))
            col = torch.where(is_non_negative,
                          torch.add(torch.tensor(i), torch.tensor(k)),
                          torch.tensor(i))
            
            # Update the element directly
            result_copy.index_copy_(0, torch.tensor([i]), x_array[i, i].unsqueeze(0))            
            result = result_copy
                
        return result
    
    else:
        raise ValueError("Input must be 1-D or 2-D")

def diagonal(x: TensorLike, offset: int = 0, axis1: int = 0, axis2: int = 1) -> torch.Tensor:
    """
    Return specified diagonals of an array.
    
    Args:
        x: Input array
        offset: Offset of the diagonal from the main diagonal
        axis1: First axis of the 2-D sub-arrays from which the diagonals should be taken
        axis2: Second axis of the 2-D sub-arrays from which the diagonals should be taken
        
    Returns:
        Array of diagonals
    """
    # Convert input to Torch array
    from ember_ml.backend.torch.tensor import TorchTensor
    Tensor = TorchTensor()
    x_array = Tensor.convert_to_tensor(x)
    
    # Check if input has at least 2 dimensions
    if x_array.ndim < 2:
        raise ValueError("Input must have at least 2 dimensions")
        
    # Ensure axis1 and axis2 are different
    # Use torch.equal for comparison
    if torch.equal(torch.tensor(axis1), torch.tensor(axis2)):
        raise ValueError("axis1 and axis2 must be different")
        
    # Normalize axes
    ndim = x_array.ndim
    if axis1 < 0:
        axis1 += ndim
    if axis2 < 0:
        axis2 += ndim
        
    # Ensure axes are valid
    # Use torch.less, torch.greater_equal, torch.logical_or for comparisons
    axis1_invalid = torch.logical_or(
        torch.less(torch.tensor(axis1), torch.tensor(0)),
        torch.greater_equal(torch.tensor(axis1), torch.tensor(ndim))
    )
    axis2_invalid = torch.logical_or(
        torch.less(torch.tensor(axis2), torch.tensor(0)),
        torch.greater_equal(torch.tensor(axis2), torch.tensor(ndim))
    )
    
    if torch.logical_or(axis1_invalid, axis2_invalid).item():
        raise ValueError("axis1 and axis2 must be within the dimensions of the input array")
        
    # Get the shape of the input array
    shape = x_array.shape
    
    # Calculate the length of the diagonal
    # Use torch.greater_equal, torch.maximum, torch.minimum, torch.subtract, torch.add for operations
    is_non_negative = torch.greater_equal(torch.tensor(offset), torch.tensor(0))
    
    # Calculate diagonal length for positive offset
    diag_len_if_positive = torch.maximum(
        torch.tensor(0),
        torch.minimum(
            torch.tensor(shape[axis1]),
            torch.subtract(torch.tensor(shape[axis2]), torch.tensor(offset))
        )
    )
    
    # Calculate diagonal length for negative offset
    diag_len_if_negative = torch.maximum(
        torch.tensor(0),
        torch.minimum(
            torch.add(torch.tensor(shape[axis1]), torch.tensor(offset)),
            torch.tensor(shape[axis2])
        )
    )
    
    # Select the appropriate length based on offset sign
    diag_len = torch.where(is_non_negative, diag_len_if_positive, diag_len_if_negative).item()
        
    # Use torch.equal for comparison
    if torch.equal(torch.tensor(diag_len), torch.tensor(0)):
        # Empty diagonal
        return torch.tensor([], dtype=x_array.dtype)
        
    # Create an array to hold the diagonal
    result_shape = list(shape)
    result_shape.pop(max(axis1, axis2))
    result_shape.pop(min(axis1, axis2))
    result_shape.append(diag_len)
    
    # Ensure we use a compatible dtype (not int64)
    dtype = x_array.dtype
    if dtype == torch.int64:
        dtype = torch.int32
    
    result = torch.zeros(tuple(result_shape), dtype=dtype)
    
    # Extract the diagonal
    # This is a simplified implementation that works for common cases
    # For a more general implementation, we would need to handle arbitrary axes
    
    # Handle the case where axis1 and axis2 are the first two dimensions
    # Use torch.equal and torch.logical_or for comparisons
    is_first_two_dims = torch.logical_or(
        torch.logical_and(
            torch.equal(torch.tensor(axis1), torch.tensor(0)),
            torch.equal(torch.tensor(axis2), torch.tensor(1))
        ),
        torch.logical_and(
            torch.equal(torch.tensor(axis1), torch.tensor(1)),
            torch.equal(torch.tensor(axis2), torch.tensor(0))
        )
    )
    
    if is_first_two_dims.item():
        # Transpose if needed
        # Use torch.greater for comparison
        if torch.greater(torch.tensor(axis1), torch.tensor(axis2)).item():
            x_array = torch.transpose(x_array, (1, 0) + tuple(range(2, ndim)))
            
        # Extract the diagonal
        # Use torch.greater_equal for comparison
        if torch.greater_equal(torch.tensor(offset), torch.tensor(0)).item():
            for i in range(diag_len):
                # Get the slice for the current diagonal element
                slices = [i, i + offset] + [slice(None)] * (ndim - 2)
                
                # Get the diagonal element
                diag_element = x_array[tuple(slices)]
                
                # Ensure diag_element is not int64
                if diag_element.dtype == torch.int64:
                    diag_element = diag_element.astype(torch.int32)
                
                # Set the result
                result_slices = [slice(None)] * (ndim - 2) + [i]
                # Use direct assignment for updating
                result_copy = torch.tensor(result)
                # Use add instead of direct assignment
                result_copy = result_copy.at[tuple(result_slices)].add(diag_element)
                result = result_copy
        else:
            for i in range(diag_len):
                # Get the slice for the current diagonal element
                slices = [i - offset, i] + [slice(None)] * (ndim - 2)
                
                # Get the diagonal element
                diag_element = x_array[tuple(slices)]
                
                # Ensure diag_element is not int64
                if diag_element.dtype == torch.int64:
                    diag_element = diag_element.astype(torch.int32)
                
                # Set the result
                result_slices = [slice(None)] * (ndim - 2) + [i]
                # Use direct assignment for updating
                result_copy = torch.tensor(result).copy()
                # Use add instead of direct assignment
                result_copy = result_copy.at[tuple(result_slices)].add(diag_element)
                result = result_copy
    else:
        # For arbitrary axes, we need to permute the dimensions
        # Create a permutation that brings axis1 and axis2 to the front
        perm = list(range(ndim))
        perm.remove(axis1)
        perm.remove(axis2)
        perm = [axis1, axis2] + perm
        
        # Transpose the array to bring the specified axes to the front
        x_transposed = torch.transpose(x_array, perm)
        
        # Now we can extract the diagonal from the first two dimensions
        # Use torch.greater_equal for comparison
        if torch.greater_equal(torch.tensor(offset), torch.tensor(0)).item():
            for i in range(diag_len):
                # Get the slice for the current diagonal element
                slices = [i, i + offset] + [slice(None)] * (ndim - 2)
                
                # Get the diagonal element
                diag_element = x_transposed[tuple(slices)]
                
                # Ensure diag_element is not int64
                if diag_element.dtype == torch.int64:
                    diag_element = diag_element.astype(torch.int32)
                
                # Set the result
                result_slices = [slice(None)] * (ndim - 2) + [i]
                # Use direct assignment for updating
                result_copy = torch.tensor(result)
                # Use add instead of direct assignment
                result_copy = result_copy.at[tuple(result_slices)].add(diag_element)
                result = result_copy
        else:
            for i in range(diag_len):
                # Get the slice for the current diagonal element
                slices = [i - offset, i] + [slice(None)] * (ndim - 2)
                
                # Get the diagonal element
                diag_element = x_transposed[tuple(slices)]
                
                # Ensure diag_element is not int64
                if diag_element.dtype == torch.int64:
                    diag_element = diag_element.astype(torch.int32)
                
                # Set the result
                result_slices = [slice(None)] * (ndim - 2) + [i]
                # Use direct assignment for updating
                result_copy = torch.tensor(result)
                # Use add instead of direct assignment
                result_copy = result_copy.at[tuple(result_slices)].add(diag_element)
                result = result_copy
        
    return result