"""
MLX implementation of orthogonal matrix operations.

This module provides MLX-specific implementation of orthogonal matrix operations,
leveraging high-precision computing techniques for numerical stability.
"""

import math
import mlx.core as mx
from typing import Tuple, Optional, Any, Union

from ember_ml.backend.mlx.linearalg.decomp_ops_hpc import HPC16x8
from ember_ml.backend.mlx.types import TensorLike

# Helper for double-single precision arithmetic
def _add_double_single(a_high, a_low, b_high, b_low):
    """Helper for double-single precision arithmetic."""
    s = a_high + b_high
    e = (a_high - s) + b_high + a_low + b_low
    return s, e

# MLX-specific QR decomposition
def _custom_qr(matrix_high, matrix_low):
    """MLX-specific QR decomposition with increased numerical stability."""
    rows, cols = matrix_high.shape
    
    # Use HPC implementation for non-square matrices
    if rows != cols:
        # Note: The provided code returned Q, R, None. Assuming None was placeholder.
        matrix_hpc = HPC16x8.from_array(matrix_high)  # Convert to HPC format
        q_hpc, r_hpc = matrix_hpc.qr()  # HPC QR decomposition 
        # Convert back to standard float32 mx.array for consistency
        return q_hpc.to_float32(), r_hpc.to_float32() 
    
    # Square matrix case - use existing implementation from provided code
    q_high = mx.zeros((rows, cols), dtype=mx.float32)
    # R matrix is cols x cols
    r_high = mx.zeros((cols, cols), dtype=mx.float32) 
    # Low part for R might not be needed if only Q is used by orthogonal, but keep for now
    r_low  = mx.zeros((cols, cols), dtype=mx.float32) 

    temp_v_high = mx.array(matrix_high) # Work on a copy
    temp_v_low = mx.array(matrix_low)   # Work on a copy

    for i in range(cols):
        v_high, v_low = temp_v_high[:, i], temp_v_low[:, i]

        for j in range(i):
            # Ensure correct shapes for matmul: (1, rows) @ (rows, 1) -> (1, 1)
            qj_high_row = q_high[:, j].reshape(1, -1)
            
            # Calculate r_high[j, i]
            r_val_high = mx.matmul(qj_high_row, v_high.reshape(-1, 1)).item()
            # Update using array indexing
            r_high_new = mx.array(r_high)
            r_high_new[j, i] = r_val_high
            r_high = r_high_new

            # Calculate r_low[j, i] (assuming q_low is zero initially)
            # Simplified r_low calculation as q_low is zero
            r_val_low = mx.matmul(qj_high_row, v_low.reshape(-1, 1)).item()
            # Update using array indexing
            r_low_new = mx.array(r_low)
            r_low_new[j, i] = r_val_low
            r_low = r_low_new

            # Calculate projection
            proj_high = mx.matmul(q_high[:, j].reshape(-1, 1), mx.array(r_val_high).reshape(1, 1))
            proj_low  = mx.matmul(q_high[:, j].reshape(-1, 1), mx.array(r_val_low).reshape(1, 1))
            
            # Subtract projection using double-single arithmetic
            v_high, v_low = _add_double_single(v_high, v_low, -proj_high[:, 0], -proj_low[:, 0])
        
        # Update the temporary matrix columns after projections
        temp_v_high_new = mx.array(temp_v_high)
        temp_v_high_new[:, i] = v_high
        temp_v_high = temp_v_high_new
        
        temp_v_low_new = mx.array(temp_v_low)
        temp_v_low_new[:, i] = v_low
        temp_v_low = temp_v_low_new

        # Calculate norm
        norm_high = mx.linalg.norm(v_high)
        if norm_high < 1e-10:
            # Consider raising an error or handling degenerate cases differently
            # For now, keep original behavior
             raise ValueError(f"Column norm too small (col={i}). Check initialization.")

        # Normalize and store in Q
        q_col = (v_high / norm_high).astype(mx.float32)
        q_high_new = mx.array(q_high)
        q_high_new[:, i] = q_col
        q_high = q_high_new
        
        # Update R diagonal
        r_high_new = mx.array(r_high)
        r_high_new[i, i] = norm_high
        r_high = r_high_new
        # r_low diagonal update might depend on v_low and norm calculation details omitted here

    # Return Q and R (high parts, low part of R might not be fully necessary/correct here)
    return q_high, r_high # Returning R low part might be misleading if not fully computed

def orthogonal(shape: Union[Tuple[int, ...], TensorLike], gain: float = 1.0, dtype: Optional[Any] = None, device: Optional[str] = None) -> mx.array:
    """
    MLX-specific orthogonal matrix initialization with improved numerical stability.
    
    Uses HPC implementation for non-square matrices to handle MLX limitations.
    
    Args:
        shape: Shape of the tensor to initialize. Must have at least 2 dimensions.
        gain: Multiplicative factor to apply to the orthogonal matrix.
        dtype: Data type of the tensor (optional, not used in MLX implementation).
        device: Device to place the tensor on (optional, not used in MLX implementation).
        
    Returns:
        A random orthogonal matrix of the specified shape.
        
    Raises:
        ValueError: If shape has fewer than 2 dimensions.
    """
    if isinstance(shape, mx.array):
        # If shape is an MLX array, convert to tuple of Python integers
        shape_tuple = tuple(int(dim.item()) if hasattr(dim, 'item') else int(dim) for dim in shape)
    else:
        # Otherwise, assume it's already a tuple or list
        shape_tuple = tuple(int(dim.item()) if hasattr(dim, 'item') else int(dim) for dim in shape)
    
    if len(shape_tuple) < 2:
        raise ValueError("Shape must have at least 2 dimensions")

    rows, cols = shape_tuple[0], math.prod(shape_tuple[1:])
    size = max(rows, cols) # Create a square matrix for QR

    # Generate a random matrix (high part)
    matrix_high = mx.random.normal(
        shape=(size, size),
        dtype=mx.float32,
        loc=0.0,
        scale=1.0
    )
    
    if rows != cols:
        # Use HPC path for non-square based on original logic
        matrix_hpc = HPC16x8.from_array(matrix_high)
        q_high, _ = matrix_hpc.qr() # We only need Q
        
        # Check if q_high is already an MLX array or an HPC object
        if hasattr(q_high, 'to_float32'):
            q_high = q_high.to_float32() # Convert HPC object to standard float32
    else:
        # Square matrix - use custom QR path
        # Generate low part for stability
        matrix_low = mx.random.normal(
            shape=(size, size), 
            dtype=mx.float32,
            loc=0, 
            scale=1e-7 # Small scale for low part
        )
        # Perform custom QR decomposition
        q_high, _ = _custom_qr(matrix_high, matrix_low) # We only need Q
    
    # Take the relevant part of Q and reshape
    q_high = q_high[:rows, :cols] 
    
    # Apply gain and reshape
    return mx.multiply(gain, q_high.reshape(shape_tuple)) # Use mx.multiply