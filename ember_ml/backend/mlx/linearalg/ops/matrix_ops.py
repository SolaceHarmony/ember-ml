"""
MLX matrix linear algebra operations for ember_ml.

This module provides MLX operations.
"""

import mlx.core as mx
from typing import Union, Tuple, Optional, Literal

# Import from tensor_ops
from ember_ml.backend.mlx.types import TensorLike
from ember_ml.backend.mlx.linearalg.ops.decomp_ops import svd
from ember_ml.backend.mlx.tensor import MLXDType

dtype_obj = MLXDType()

def norm(x: TensorLike, 
         ord: Optional[Union[int, str]] = None, 
         axis: Optional[Union[int, Tuple[int, ...]]] = None, 
         keepdim: bool = False) -> mx.array:
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
    # Convert input to MLX array
    from ember_ml.backend.mlx.tensor import MLXTensor
    Tensor = MLXTensor()
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
            result = mx.max(mx.abs(x_array), axis=axis)
        elif ord == 1:
            # L1 norm (sum of absolute values)
            result = mx.sum(mx.abs(x_array), axis=axis)
        elif ord == 2:
            # L2 norm (Euclidean norm)
            result = mx.sqrt(mx.sum(mx.square(x_array), axis=axis))
        else:
            # General Lp norm
            if isinstance(ord, (int, float)):
                result = mx.power(
                    mx.sum(mx.power(mx.abs(x_array), ord), axis=axis),
                    mx.divide(mx.array(1.0), mx.array(ord))
                )
            else:
                # Handle case where ord is a string (shouldn't happen after our fixes)
                raise ValueError(f"Invalid norm order: {ord}")
    
    # Matrix norm
    else:
        if ord == 'fro':
            # Frobenius norm
            result = mx.sqrt(mx.sum(mx.square(x_array)))
        elif ord == 'nuc':
            # Nuclear norm (sum of singular values)
            s_values = svd(x_array, compute_uv=False)
            if isinstance(s_values, tuple):
                # Handle case where svd returns a tuple
                result = mx.sum(s_values[0])
            else:
                # Handle case where svd returns an array
                result = mx.sum(s_values)
        elif ord == 1:
            # Maximum absolute column sum
            result = mx.max(mx.sum(mx.abs(x_array), axis=0))
        elif ord == 'inf':
            # Maximum absolute row sum
            result = mx.max(mx.sum(mx.abs(x_array), axis=1))
        elif ord == -1:
            # Minimum absolute column sum
            result = mx.min(mx.sum(mx.abs(x_array), axis=0))
        elif ord == '-inf':
            # Minimum absolute row sum
            result = mx.min(mx.sum(mx.abs(x_array), axis=1))
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
            result = mx.reshape(result, tuple(shape))
        else:
            shape = list(x_array.shape)
            shape[axis] = 1
            result = mx.reshape(result, tuple(shape))
    
    return result

def det(a: TensorLike) -> mx.array:
    """
    Compute the determinant of a square matrix.
    
    Args:
        a: Input square matrix
    
    Returns:
        Determinant of the matrix
    """
    # Convert input to MLX array
    from ember_ml.backend.mlx.tensor import MLXTensor
    Tensor = MLXTensor()
    a_array = Tensor.convert_to_tensor(a)

    
    # Get matrix dimensions
    n = a_array.shape[0]
    assert a_array.shape[1] == n, "Matrix must be square"
    
    # Special cases for small matrices
    if mx.equal(n, mx.array(1)):
        return a_array[0, 0]
    elif mx.equal(n, mx.array(2)):
        term1 = mx.multiply(a_array[0, 0], a_array[1, 1])
        term2 = mx.multiply(a_array[0, 1], a_array[1, 0])
        return mx.subtract(term1, term2)
    
    # For larger matrices, use LU decomposition
    # This is a simplified implementation and may not be numerically stable
    # For a more robust implementation, consider using a dedicated algorithm
    
    # Make a copy of the matrix
    a_copy = mx.array(a_array)
    
    # Initialize determinant
    det_value = mx.array(1.0, dtype=a_array.dtype)
    
    # Gaussian elimination
    for i in range(n):
        # Find pivot
        pivot = a_copy[i, i]
        
        # Update determinant
        det_value = mx.multiply(det_value, pivot)
        
        # If pivot is zero, determinant is zero
        if mx.less(mx.abs(pivot), mx.array(1e-10)):
            return mx.array(0.0, dtype=a_array.dtype)
        
        # Eliminate below
        # Use direct integer calculation
        i_plus_1_int = i + 1
        for j in range(i_plus_1_int, n):
            factor = mx.divide(a_copy[j, i], pivot)
            
            # Calculate the new row
            new_row = mx.subtract(a_copy[j, i:], mx.multiply(factor, a_copy[i, i:]))
            
            # Update a_copy using direct indexing
            for k in range(i, n):
                a_copy[j, k] = new_row[k - i]
    
    return det_value

def diag(x: TensorLike, k: int = 0) -> mx.array:
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
    # Convert input to MLX array
    from ember_ml.backend.mlx.tensor import MLXTensor
    Tensor = MLXTensor()
    x_array = Tensor.convert_to_tensor(x)
    
    # Check if input is 1-D or 2-D
    if x_array.ndim == 1:
        # Construct a diagonal matrix
        n = x_array.shape[0]
        
        # Calculate the size of the output matrix
        if k >= 0:
            m = n + k
        else:
            m = n - k
            
        # Create a zero matrix
        result = mx.zeros((m, m), dtype=x_array.dtype)
        
        # Fill the diagonal
        if k >= 0:
            # Diagonal above the main diagonal
            for i in range(n):
                result = result.at[i, i + k].set(x_array[i])
        else:
            # Diagonal below the main diagonal
            for i in range(n):
                result = result.at[i - k, i].set(x_array[i])
                
        return result
    
    elif x_array.ndim == 2:
        # Extract a diagonal
        rows, cols = x_array.shape
        
        # Calculate the length of the diagonal
        if k >= 0:
            diag_len = min(rows, cols - k)
        else:
            diag_len = min(rows + k, cols)
            
        if diag_len <= 0:
            # Empty diagonal
            return mx.array([], dtype=x_array.dtype)
            
        # Create an array to hold the diagonal
        result = mx.zeros((diag_len,), dtype=x_array.dtype)
        
        # Extract the diagonal
        if k >= 0:
            # Diagonal above the main diagonal
            for i in range(diag_len):
                result = result.at[i].set(x_array[i, i + k])
        else:
            # Diagonal below the main diagonal
            for i in range(diag_len):
                result = result.at[i].set(x_array[i - k, i])
                
        return result
    
    else:
        raise ValueError("Input must be 1-D or 2-D")

def diagonal(x: TensorLike, offset: int = 0, axis1: int = 0, axis2: int = 1) -> mx.array:
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
    # Convert input to MLX array
    from ember_ml.backend.mlx.tensor import MLXTensor
    Tensor = MLXTensor()
    x_array = Tensor.convert_to_tensor(x)
    
    # Check if input has at least 2 dimensions
    if x_array.ndim < 2:
        raise ValueError("Input must have at least 2 dimensions")
        
    # Ensure axis1 and axis2 are different
    if axis1 == axis2:
        raise ValueError("axis1 and axis2 must be different")
        
    # Normalize axes
    ndim = x_array.ndim
    if axis1 < 0:
        axis1 += ndim
    if axis2 < 0:
        axis2 += ndim
        
    # Ensure axes are valid
    if axis1 < 0 or axis1 >= ndim or axis2 < 0 or axis2 >= ndim:
        raise ValueError("axis1 and axis2 must be within the dimensions of the input array")
        
    # Get the shape of the input array
    shape = x_array.shape
    
    # Calculate the length of the diagonal
    if offset >= 0:
        diag_len = max(0, min(shape[axis1], shape[axis2] - offset))
    else:
        diag_len = max(0, min(shape[axis1] + offset, shape[axis2]))
        
    if diag_len == 0:
        # Empty diagonal
        return mx.array([], dtype=x_array.dtype)
        
    # Create an array to hold the diagonal
    result_shape = list(shape)
    result_shape.pop(max(axis1, axis2))
    result_shape.pop(min(axis1, axis2))
    result_shape.append(diag_len)
    result = mx.zeros(tuple(result_shape), dtype=x_array.dtype)
    
    # Extract the diagonal
    # This is a simplified implementation that works for common cases
    # For a more general implementation, we would need to handle arbitrary axes
    
    # Handle the case where axis1 and axis2 are the first two dimensions
    if (axis1 == 0 and axis2 == 1) or (axis1 == 1 and axis2 == 0):
        # Transpose if needed
        if axis1 > axis2:
            x_array = mx.transpose(x_array, (1, 0) + tuple(range(2, ndim)))
            
        # Extract the diagonal
        if offset >= 0:
            for i in range(diag_len):
                # Get the slice for the current diagonal element
                slices = [i, i + offset] + [slice(None)] * (ndim - 2)
                
                # Get the diagonal element
                diag_element = x_array[tuple(slices)]
                
                # Set the result
                result_slices = [slice(None)] * (ndim - 2) + [i]
                result = result.at[tuple(result_slices)].set(diag_element)
        else:
            for i in range(diag_len):
                # Get the slice for the current diagonal element
                slices = [i - offset, i] + [slice(None)] * (ndim - 2)
                
                # Get the diagonal element
                diag_element = x_array[tuple(slices)]
                
                # Set the result
                result_slices = [slice(None)] * (ndim - 2) + [i]
                result = result.at[tuple(result_slices)].set(diag_element)
    else:
        # For arbitrary axes, we need to permute the dimensions
        # This is a more complex case that would require a more general implementation
        # For now, we'll raise an error
        raise NotImplementedError("diagonal with arbitrary axes is not yet implemented")
        
    return result