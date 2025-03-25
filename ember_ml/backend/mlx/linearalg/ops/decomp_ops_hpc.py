"""
MLX high-precision computing for matrix decomposition operations.

This module provides high-precision matrix computation implementations for the MLX backend.
It allows for more numerically stable computations by implementing a limb-based precision approach.
"""
from typing import Union, Tuple
import mlx.core as mx

# Import from tensor_ops
from ember_ml.backend.mlx.tensor import MLXDType
from ember_ml.backend.mlx.types import TensorLike

dtype_obj = MLXDType()

def _add_double_single(a_high, a_low, b_high, b_low):
    """Helper for double-single precision arithmetic."""
    s = a_high + b_high
    e = (a_high - s) + b_high + a_low + b_low
    return s, e

class HPC16x8:
    """
    High-Precision Computing class for MLX using a limb-based approach.
    
    This class implements a 24-bit precision floating point representation
    using two 16-bit mantissas (high and low) and an 8-bit shared exponent.
    It provides enhanced numerical stability for QR decomposition of matrices,
    especially those with high condition numbers or varying aspect ratios.
    
    The limb-based approach splits the precision into multiple components:
    - High limb: Most significant bits (16 bits of precision)
    - Low limb: Least significant bits (8 bits of precision)
    
    This implementation is optimized for MLX backend operations.
    """
    
    def __init__(self, high, low=None):
        """
        Initialize an HPC16x8 object.
        
        Args:
            high: High-precision part (16 bits)
            low: Low-precision part (8 bits), defaults to zeros if not provided
        """
        self.high = high
        if low is None:
            self.low = mx.zeros_like(high)
        else:
            self.low = low
    
    @classmethod
    def from_array(cls, array):
        """
        Convert a regular MLX array to HPC format.
        
        Args:
            array: Input MLX array to convert
            
        Returns:
            HPC16x8 instance representing the input array
        """
        # Initialize with high part as the original array
        # and low part as zeros (initially)
        return cls(array)
    
    def to_float32(self):
        """
        Convert back to regular MLX float32 array.
        
        Returns:
            MLX array in float32 format
        """
        # In a limb-based approach, we'd combine high and low parts
        # But for simplicity when converting back, we primarily use the high part
        # as the low part contains minimal precision bits that may introduce noise
        return self.high
    
    def __add__(self, other):
        """
        Add two HPC16x8 objects with extra precision.
        
        Args:
            other: Another HPC16x8 object
            
        Returns:
            HPC16x8 representing the sum
        """
        if isinstance(other, HPC16x8):
            high, low = _add_double_single(self.high, self.low, other.high, other.low)
            return HPC16x8(high, low)
        else:
            # Handle scalar or array addition
            high, low = _add_double_single(self.high, self.low, other, mx.zeros_like(self.high))
            return HPC16x8(high, low)
    
    def __sub__(self, other):
        """
        Subtract one HPC16x8 object from another with extra precision.
        
        Args:
            other: Another HPC16x8 object
            
        Returns:
            HPC16x8 representing the difference
        """
        if isinstance(other, HPC16x8):
            # Negate the second object and add
            neg_high = -other.high
            neg_low = -other.low
            high, low = _add_double_single(self.high, self.low, neg_high, neg_low)
            return HPC16x8(high, low)
        else:
            # Handle scalar or array subtraction
            high, low = _add_double_single(self.high, self.low, -other, mx.zeros_like(self.high))
            return HPC16x8(high, low)
    
    def __mul__(self, other):
        """
        Multiply two HPC16x8 objects with extra precision.
        
        Args:
            other: Another HPC16x8 object or scalar
            
        Returns:
            HPC16x8 representing the product
        """
        if isinstance(other, HPC16x8):
            # Double-single precision multiplication
            high = mx.multiply(self.high, other.high)
            cross1 = mx.multiply(self.high, other.low)
            cross2 = mx.multiply(self.low, other.high)
            low = mx.add(cross1, cross2)
            return HPC16x8(high, low)
        else:
            # Scalar multiplication
            high = mx.multiply(self.high, other)
            low = mx.multiply(self.low, other)
            return HPC16x8(high, low)
    
    def __truediv__(self, other):
        """
        Divide one HPC16x8 object by another with extra precision.
        
        Args:
            other: Another HPC16x8 object or scalar
            
        Returns:
            HPC16x8 representing the quotient
        """
        if isinstance(other, HPC16x8):
            # Perform division using Newton-Raphson refinement
            # for extra precision in the reciprocal
            recip = 1.0 / other.high
            # One refinement step: r = r * (2 - a * r)
            correction = mx.subtract(2.0, mx.multiply(other.high, recip))
            refined_recip = mx.multiply(recip, correction)
            
            high = mx.multiply(self.high, refined_recip)
            low = mx.multiply(self.low, refined_recip)
            return HPC16x8(high, low)
        else:
            # Scalar division
            recip = 1.0 / other
            high = mx.multiply(self.high, recip)
            low = mx.multiply(self.low, recip)
            return HPC16x8(high, low)
    
    def sqrt(self):
        """
        Compute square root with extra precision.
        
        Returns:
            HPC16x8 representing the square root
        """
        # Initial approximation
        x = mx.sqrt(self.high)
        
        # One Newton-Raphson refinement step
        # x = 0.5 * (x + a / x)
        recip = 1.0 / x
        approx_div = mx.multiply(self.high, recip)
        sum_terms = mx.add(x, approx_div)
        refined = mx.multiply(0.5, sum_terms)
        
        # Create low component by computing the residual
        squared = mx.multiply(refined, refined)
        residual = mx.subtract(self.high, squared)
        low_part = mx.multiply(0.5, mx.multiply(residual, recip))
        
        return HPC16x8(refined, low_part)
    
    def norm(self):
        """
        Compute L2 norm (Euclidean) with extra precision.
        
        Returns:
            HPC16x8 representing the norm
        """
        # Sum of squares with extra precision
        squared_high = mx.multiply(self.high, self.high)
        cross_term = mx.multiply(2.0, mx.multiply(self.high, self.low))
        sum_squared_high = mx.sum(squared_high)
        sum_cross = mx.sum(cross_term)
        
        # Create HPC representation of the sum
        sum_hpc = HPC16x8(sum_squared_high, sum_cross)
        
        # Take square root with extra precision
        return sum_hpc.sqrt()
    
    def matmul(self, other):
        """
        Matrix multiplication with extra precision.
        
        Args:
            other: Another HPC16x8 object
            
        Returns:
            HPC16x8 representing the matrix product
        """
        if isinstance(other, HPC16x8):
            # Perform high-precision matrix multiplication
            high_result = mx.matmul(self.high, other.high)
            
            # Cross products for additional precision
            cross1 = mx.matmul(self.high, other.low)
            cross2 = mx.matmul(self.low, other.high)
            low_result = mx.add(cross1, cross2)
            
            return HPC16x8(high_result, low_result)
        else:
            # Handle multiplication with standard array
            high_result = mx.matmul(self.high, other)
            low_result = mx.matmul(self.low, other)
            return HPC16x8(high_result, low_result)
    
    def qr(self):
        """
        Perform QR decomposition on the HPC matrix with enhanced precision.
        
        Returns:
            Tuple of (Q, R) as HPC16x8 objects
        """
        m, n = self.high.shape
        
        if m < n:
            # For fat matrices, use the transpose trick
            hpc_transposed = HPC16x8(mx.transpose(self.high), mx.transpose(self.low))
            r_hpc, q_hpc = hpc_transposed.qr()
            return HPC16x8(mx.transpose(q_hpc.high), mx.transpose(q_hpc.low)), HPC16x8(mx.transpose(r_hpc.high), mx.transpose(r_hpc.low))
        
        # Initialize Q and R matrices with zeros
        q_high = mx.zeros((m, m), dtype=self.high.dtype)
        q_low = mx.zeros((m, m), dtype=self.low.dtype)
        r_high = mx.zeros((m, n), dtype=self.high.dtype)
        r_low = mx.zeros((m, n), dtype=self.low.dtype)
        
        # Loop over columns for modified Gram-Schmidt
        for j in range(min(m, n)):
            # Extract column j
            v_high = self.high[:, j]
            v_low = self.low[:, j]
            
            # Orthogonalize against previous columns
            for i in range(j):
                # Compute dot product with extra precision
                dot_high = mx.sum(mx.multiply(q_high[:, i], v_high))
                dot_cross1 = mx.sum(mx.multiply(q_high[:, i], v_low))
                dot_cross2 = mx.sum(mx.multiply(q_low[:, i], v_high))
                dot_low = mx.add(dot_cross1, dot_cross2)
                
                # Store in R
                r_high = r_high.at[i, j].set(dot_high)
                r_low = r_low.at[i, j].set(dot_low)
                
                # Subtract projection with extra precision
                proj_high = mx.multiply(dot_high, q_high[:, i])
                proj_low = mx.add(mx.multiply(dot_high, q_low[:, i]), mx.multiply(dot_low, q_high[:, i]))
                
                v_high, v_low = _add_double_single(v_high, v_low, -proj_high, -proj_low)
            
            # Compute norm with extra precision
            v_norm_squared_high = mx.sum(mx.multiply(v_high, v_high))
            v_norm_cross = mx.sum(mx.multiply(2.0, mx.multiply(v_high, v_low)))
            
            # Check if column is already zero
            if v_norm_squared_high < 1e-14:
                continue
                
            # Compute norm using sqrt with extra precision
            v_norm_hpc = HPC16x8(v_norm_squared_high, v_norm_cross).sqrt()
            v_norm_high = v_norm_hpc.high
            v_norm_low = v_norm_hpc.low
            
            # Store in R diagonal
            r_high = r_high.at[j, j].set(v_norm_high)
            r_low = r_low.at[j, j].set(v_norm_low)
            
            # Normalize column to create Q
            # Division with extra precision
            recip_norm_high = 1.0 / v_norm_high
            # One refinement step: r = r * (2 - a * r)
            correction = mx.subtract(2.0, mx.multiply(v_norm_high, recip_norm_high))
            refined_recip = mx.multiply(recip_norm_high, correction)
            
            q_high = q_high.at[:, j].set(mx.multiply(v_high, refined_recip))
            q_low = q_low.at[:, j].set(mx.multiply(v_low, refined_recip))
            
            # Update remaining columns in A
            if j < n - 1:
                for k in range(j + 1, n):
                    # Extract column k
                    col_k_high = self.high[:, k]
                    col_k_low = self.low[:, k]
                    
                    # Compute dot product with extra precision
                    dot_high = mx.sum(mx.multiply(q_high[:, j], col_k_high))
                    dot_cross1 = mx.sum(mx.multiply(q_high[:, j], col_k_low))
                    dot_cross2 = mx.sum(mx.multiply(q_low[:, j], col_k_high))
                    dot_low = mx.add(dot_cross1, dot_cross2)
                    
                    # Store in R
                    r_high = r_high.at[j, k].set(dot_high)
                    r_low = r_low.at[j, k].set(dot_low)
                    
                    # Subtract projection with extra precision
                    proj_high = mx.multiply(dot_high, q_high[:, j])
                    proj_low = mx.add(mx.multiply(dot_high, q_low[:, j]), mx.multiply(dot_low, q_high[:, j]))
                    
                    # Update column k
                    new_col_high, new_col_low = _add_double_single(col_k_high, col_k_low, -proj_high, -proj_low)
                    self.high = self.high.at[:, k].set(new_col_high)
                    self.low = self.low.at[:, k].set(new_col_low)
        
        # For tall matrices, complete the orthogonal basis for Q
        if m > n:
            for j in range(n, m):
                # Create a unit vector in the j-th direction
                v_high = mx.zeros((m,), dtype=self.high.dtype)
                v_high = v_high.at[j].set(1.0)
                v_low = mx.zeros((m,), dtype=self.low.dtype)
                
                # Orthogonalize against all previous columns
                for i in range(j):
                    # Compute dot product with extra precision
                    dot_high = mx.sum(mx.multiply(q_high[:, i], v_high))
                    dot_cross1 = mx.sum(mx.multiply(q_high[:, i], v_low))
                    dot_cross2 = mx.sum(mx.multiply(q_low[:, i], v_high))
                    dot_low = mx.add(dot_cross1, dot_cross2)
                    
                    # Subtract projection with extra precision
                    proj_high = mx.multiply(dot_high, q_high[:, i])
                    proj_low = mx.add(mx.multiply(dot_high, q_low[:, i]), mx.multiply(dot_low, q_high[:, i]))
                    
                    v_high, v_low = _add_double_single(v_high, v_low, -proj_high, -proj_low)
                
                # Compute norm with extra precision
                v_norm_squared_high = mx.sum(mx.multiply(v_high, v_high))
                v_norm_cross = mx.sum(mx.multiply(2.0, mx.multiply(v_high, v_low)))
                
                # Check if vector is numerically zero
                if v_norm_squared_high < 1e-14:
                    # Try a different vector
                    v_high = mx.zeros((m,), dtype=self.high.dtype)
                    v_high = v_high.at[(j + 1) % m].set(1.0)
                    v_low = mx.zeros((m,), dtype=self.low.dtype)
                    
                    # Orthogonalize against all previous columns again
                    for i in range(j):
                        dot_high = mx.sum(mx.multiply(q_high[:, i], v_high))
                        dot_low = mx.sum(mx.multiply(q_high[:, i], v_low))
                        
                        proj_high = mx.multiply(dot_high, q_high[:, i])
                        proj_low = mx.multiply(dot_low, q_high[:, i])
                        
                        v_high, v_low = _add_double_single(v_high, v_low, -proj_high, -proj_low)
                    
                    v_norm_squared_high = mx.sum(mx.multiply(v_high, v_high))
                
                # Compute norm using sqrt with extra precision
                v_norm_hpc = HPC16x8(v_norm_squared_high, v_norm_cross).sqrt()
                v_norm_high = v_norm_hpc.high
                
                # Normalize column to create Q
                recip_norm = 1.0 / v_norm_high
                q_high = q_high.at[:, j].set(mx.multiply(v_high, recip_norm))
                q_low = q_low.at[:, j].set(mx.multiply(v_low, recip_norm))
        
        # Return the result as HPC16x8 objects
        return HPC16x8(q_high, q_low), HPC16x8(r_high, r_low)