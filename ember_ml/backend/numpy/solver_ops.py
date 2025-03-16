"""
NumPy solver operations for ember_ml.

This module provides NumPy implementations of solver operations.
"""

import numpy as np
from typing import Union, Tuple, Optional, Sequence, Literal

# Type aliases
Shape = Union[int, Sequence[int]]
DType = Union[np.dtype, str, None]
QRMode = Literal['reduced', 'complete', 'r', 'raw']

# Import from tensor_ops
from ember_ml.backend.numpy.tensor import NumpyTensor, NumpyDType

convert_to_tensor = NumpyTensor().convert_to_tensor
dtype = NumpyDType()

def inv(A: np.ndarray) -> np.ndarray:
    """
    Inverts a square matrix using Gauss-Jordan elimination.
    
    Args:
        A: Square matrix to invert
        
    Returns:
        Inverse of matrix A
    """
    # Convert input to NumPy array
    A = convert_to_tensor(A)
    
    # Get matrix dimensions
    n = A.shape[0]
    assert A.shape[1] == n, "Matrix must be square"
    
    # Create augmented matrix [A|I]
    I = np.eye(n, dtype=A.dtype)
    aug = np.concatenate([A, I], axis=1)
    
    # Create a copy of the augmented matrix that we can modify
    aug_copy = convert_to_tensor(aug)
    
    # Gauss-Jordan elimination
    for i in range(n):
        # Find pivot
        pivot = aug_copy[i, i]
        
        # Scale pivot row
        pivot_row = np.divide(aug_copy[i], pivot)
        
        # Create a new augmented matrix with the updated row
        rows = []
        for j in range(n):
            if j == i:
                rows.append(pivot_row)
            else:
                # Eliminate from other rows
                factor = aug_copy[j, i]
                rows.append(np.subtract(aug_copy[j], np.multiply(factor, pivot_row)))
        
        # Reconstruct the augmented matrix
        aug_copy = np.stack(rows)
    
    # Extract inverse from augmented matrix
    inv_A = aug_copy[:, n:]
    
    return inv_A


def solve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve a linear system of equations Ax = b for x using NumPy backend.
    
    Args:
        a: Coefficient matrix A
        b: Right-hand side vector or matrix b
    
    Returns:
        Solution to the system of equations
    
    Notes:
        Uses custom Gauss-Jordan elimination to compute the inverse of A,
        then multiplies by b to get the solution: x = A^(-1) * b.
    """
    # Convert inputs to NumPy arrays
    a_array = convert_to_tensor(a)
    b_array = convert_to_tensor(b)
    
    # Compute the inverse of a using our custom implementation
    a_inv = inv(a_array)
    
    # Multiply the inverse by b to get the solution
    return np.matmul(a_inv, b_array)


def svd(a: np.ndarray, full_matrices: bool = True, compute_uv: bool = True, hermitian: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Compute the singular value decomposition of a matrix using power iteration.
    
    Args:
        a: Input matrix
        full_matrices: If True, return full U and Vh matrices
        compute_uv: If True, compute U and Vh matrices
        hermitian: If True, a is Hermitian (symmetric if real-valued)
    
    Returns:
        If compute_uv is True, returns (U, S, Vh), otherwise returns S
    """
    # Convert input to NumPy array
    a_array = convert_to_tensor(a)
    
    # Use NumPy's built-in SVD function
    if compute_uv:
        u, s, vh = np.linalg.svd(a_array, full_matrices=full_matrices, compute_uv=True, hermitian=hermitian)
        return u, s, vh
    else:
        s = np.linalg.svd(a_array, full_matrices=full_matrices, compute_uv=False, hermitian=hermitian)
        return s


def eig(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the eigenvalues and eigenvectors of a square matrix.
    
    Args:
        a: Input square matrix
    
    Returns:
        Tuple of (eigenvalues, eigenvectors)
    """
    # Convert input to NumPy array
    a_array = convert_to_tensor(a)
    
    # Use NumPy's built-in eigendecomposition function
    return np.linalg.eig(a_array)


def eigvals(a: np.ndarray) -> np.ndarray:
    """
    Compute the eigenvalues of a square matrix.
    
    Args:
        a: Input square matrix
    
    Returns:
        Eigenvalues of the matrix
    """
    # Convert input to NumPy array
    a_array = convert_to_tensor(a)
    
    # Use NumPy's built-in eigenvalues function
    return np.linalg.eigvals(a_array)


def det(a: np.ndarray) -> np.ndarray:
    """
    Compute the determinant of a square matrix.
    
    Args:
        a: Input square matrix
    
    Returns:
        Determinant of the matrix
    """
    # Convert input to NumPy array
    a_array = convert_to_tensor(a)
    
    # Use NumPy's built-in determinant function
    return np.linalg.det(a_array)

def norm(x: np.ndarray, ord: Optional[Union[Literal['fro', 'nuc'], float]] = None, axis: None = None, keepdims: bool = False) -> Union[float, np.ndarray]:
    """
    Compute the matrix or vector norm.
    
    Args:
        x: Input matrix or vector
        ord: Order of the norm ('fro', 'nuc', float, or None)
        axis: Axis along which to compute the norm (must be None for matrix norm)
        keepdims: Whether to keep the reduced dimensions
    
    Returns:
        Norm of the matrix or vector
    """
    # Convert input to NumPy array
    x_array = convert_to_tensor(x)
    
    # Default values
    if ord is None:
        if axis is None:
            # Default to Frobenius norm for matrices, L2 norm for vectors
            if len(x_array.shape) > 1:
                ord = 'fro'
            else:
                ord = 2
        else:
            # Default to L2 norm along the specified axis
            ord = 2
    
    # Use NumPy's built-in norm function
    result = np.linalg.norm(x_array, ord=ord, axis=axis, keepdims=keepdims)
    
    # Ensure the result is the correct type
    if isinstance(result, np.ndarray):
        return result
    else:
        # Use item() to extract scalar value without precision loss
        return result.item()


def qr(a: np.ndarray, mode: QRMode = 'reduced') -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the QR decomposition of a matrix.
    
    Args:
        a: Input matrix
        mode: Mode of decomposition ('reduced', 'complete', 'r', 'raw')
    
    Returns:
        Tuple of (Q, R) matrices
    """
    # Convert input to NumPy array
    a_array = convert_to_tensor(a)
    
    # Use NumPy's built-in QR decomposition function
    return np.linalg.qr(a_array, mode=mode)


def cholesky(a: np.ndarray) -> np.ndarray:
    """
    Compute the Cholesky decomposition of a positive definite matrix.
    
    Args:
        a: Input positive definite matrix
    
    Returns:
        Lower triangular matrix L such that L @ L.T = A
    """
    # Convert input to NumPy array
    a_array = convert_to_tensor(a)
    
    # Use NumPy's built-in Cholesky decomposition function
    return np.linalg.cholesky(a_array)


def lstsq(a: np.ndarray, b: np.ndarray, rcond: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """
    Compute the least-squares solution to a linear matrix equation.
    
    Args:
        a: Coefficient matrix
        b: Dependent variable
        rcond: Cutoff for small singular values
    
    Returns:
        Tuple of (solution, residuals, rank, singular values)
    """
    # Convert inputs to NumPy arrays
    a_array = convert_to_tensor(a)
    b_array = convert_to_tensor(b)
    
    # Use NumPy's built-in least-squares function
    return np.linalg.lstsq(a_array, b_array, rcond=rcond)


class NumpySolverOps:
    """NumPy implementation of solver operations."""
    
    def solve(self, a, b):
        """Solve a linear system of equations Ax = b for x."""
        return solve(a, b)
    
    def svd(self, a, full_matrices=True, compute_uv=True, hermitian=False):
        """Compute the singular value decomposition of a matrix."""
        return svd(a, full_matrices=full_matrices, compute_uv=compute_uv, hermitian=hermitian)
    
    def eig(self, a):
        """Compute the eigenvalues and eigenvectors of a square matrix."""
        return eig(a)
    
    def eigvals(self, a):
        """Compute the eigenvalues of a square matrix."""
        return eigvals(a)
    
    def inv(self, a):
        """Compute the inverse of a square matrix."""
        return inv(a)
    
    def det(self, a):
        """Compute the determinant of a square matrix."""
        return det(a)
    
    def norm(self, x, ord=None, axis=None, keepdims=False):
        """Compute the matrix or vector norm."""
        return norm(x, ord=ord, axis=axis, keepdims=keepdims)
    
    def qr(self, a, mode='reduced'):
        """Compute the QR decomposition of a matrix."""
        return qr(a, mode=mode)
    
    def cholesky(self, a):
        """Compute the Cholesky decomposition of a matrix."""
        return cholesky(a)
    
    def lstsq(self, a, b, rcond=None):
        """Compute the least-squares solution to a linear matrix equation."""
        return lstsq(a, b, rcond=rcond)