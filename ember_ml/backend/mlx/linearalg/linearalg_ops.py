"""
MLX solver operations for ember_ml.

This module provides MLX implementations of solver operations.
"""

import mlx.core as mx

# Import from tensor_ops
from ember_ml.backend.mlx.tensor import MLXDType
from typing import Literal, Union, Tuple, Optional
from ember_ml.backend.mlx.types import TensorLike, Axis, OrdLike

dtype_obj = MLXDType()


class MLXLinearAlgOps:
    """MLX linear algebra operations."""
    
    def solve(self, a : TensorLike, b : TensorLike) -> mx.array:
        """
        Solve a linear system of equations Ax = b for x.
        
        Args:
            a: Coefficient matrix A
            b: Right-hand side vector or matrix b
            
        Returns:
            Solution to the system of equations
        """
        from ember_ml.backend.mlx.linearalg.ops import solve as solve_func
        return solve_func(a, b)


    def inv(self, a: TensorLike) -> mx.array:
        """
        Compute the inverse of a square matrix.
        
        Args:
            a: Input square matrix
            
        Returns:
            Inverse of the matrix
        """
        from ember_ml.backend.mlx.linearalg.ops import inv as inv_func
        return inv_func(a)
    
    def svd(self, a: TensorLike, full_matrices=True, compute_uv=True) -> Union[mx.array, Tuple[mx.array, mx.array, mx.array]]:
        """
        Compute the singular value decomposition of a matrix.
        
        Args:
            a: Input matrix
            full_matrices: If True, return full U and Vh matrices
            compute_uv: If True, compute U and Vh matrices
            
        Returns:
            If compute_uv is True, returns (U, S, Vh), otherwise returns S
        """
        from ember_ml.backend.mlx.linearalg.ops import svd as svd_func
        return svd_func(a, full_matrices=full_matrices, compute_uv=compute_uv)

    def det(self, a: TensorLike) -> mx.array:
        """
        Compute the determinant of a square matrix.
        
        Args:
            a: Input square matrix
            
        Returns:
            Determinant of the matrix
        """
        from ember_ml.backend.mlx.linearalg.ops import det as det_func
        return det_func(a)

    def norm(self, x: TensorLike, ord: OrdLike = None, 
             axis: Axis = None, 
             keepdims: bool = False) -> mx.array:
        """
        Compute the matrix or vector norm.
        
        Args:
            x: Input matrix or vector
            ord: Order of the norm
            axis: Axis along which to compute the norm
            keepdims: Whether to keep the reduced dimensions
            
        Returns:
            Norm of the matrix or vector
        """
        from ember_ml.backend.mlx.linearalg.ops import norm as norm_func
        return norm_func(x, ord=ord, axis=axis, keepdims=keepdims)
    
    def qr(self, a: TensorLike, mode: Literal["reduced", "complete", "r", "raw"] = "reduced") -> Tuple[mx.array, mx.array]:
        """
        Compute the QR decomposition of a matrix.
        
        Args:
            a: Input matrix
            mode: Mode of decomposition ('reduced', 'complete', 'r', 'raw')
            
        Returns:
            Tuple of (Q, R) matrices
        """
        from ember_ml.backend.mlx.linearalg.ops import qr as qr_func
        return qr_func(a, mode=mode)


    def cholesky(self, a: TensorLike):
        """
        Compute the Cholesky decomposition of a positive definite matrix.
        
        Args:
            a: Input positive definite matrix
            
        Returns:
            Lower triangular matrix L such that L @ L.T = A
        """
        from ember_ml.backend.mlx.linearalg.ops import cholesky as cholesky_func
        return cholesky_func(a)

    
    def lstsq(self, a: TensorLike, b: TensorLike, rcond: Optional[float] = None):
        """
        Compute the least-squares solution to a linear matrix equation.
        
        Args:
            a: Coefficient matrix
            b: Dependent variable
            rcond: Cutoff for small singular values
            
        Returns:
            Tuple of (solution, residuals, rank, singular values)
        """
        from ember_ml.backend.mlx.linearalg.ops import lstsq as lstsq_func
        return lstsq_func(a, b, rcond=rcond)

    def eig(self, a: TensorLike) -> Tuple[mx.array, mx.array]:
        """
        Compute the eigenvalues and eigenvectors of a square matrix using power iteration.
        
        Args:
            a: Input square matrix
        
        Returns:
            Tuple of (eigenvalues, eigenvectors)
        
        Notes:
            This is a simplified implementation using power iteration.
            For large matrices or high precision requirements, consider using
            a more sophisticated algorithm.
        """
        from ember_ml.backend.mlx.linearalg.ops import eig as eig_func
        return eig_func(a)
    
    
    def eigvals(self, a: TensorLike) -> mx.array:
        """
        Compute the eigenvalues of a square matrix.
        
        Args:
            a: Input square matrix
        
        Returns:
            Eigenvalues of the matrix
        """
        from ember_ml.backend.mlx.linearalg.ops import eigvals as eigvals_func
        return eigvals_func(a)
    
    def diag(self, x: TensorLike, k: int = 0) -> mx.array:
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
        from ember_ml.backend.mlx.linearalg.ops import diag as diag_func
        return diag_func(x, k=k)
    
    def diagonal(self, x: TensorLike, offset: int = 0, axis1: int = 0, axis2: int = 1):
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
        from ember_ml.backend.mlx.linearalg.ops import diagonal as diagonal_func
        return diagonal_func(x = x, offset = offset, axis1 = axis1, axis2=axis2)