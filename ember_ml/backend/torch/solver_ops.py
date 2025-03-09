"""
PyTorch solver operations for EmberHarmony.

This module provides PyTorch implementations of solver operations.
"""

import torch
from typing import Any, Optional, Union, Tuple

# Import from tensor_ops
from ember_ml.backend.torch.tensor_ops import convert_to_tensor


def solve(a: Any, b: Any) -> torch.Tensor:
    """
    Solve a linear system of equations Ax = b for x using PyTorch backend.
    
    Args:
        a: Coefficient matrix A
        b: Right-hand side vector or matrix b
        
    Returns:
        Solution to the system of equations
        
    Notes:
        Uses torch.linalg.solve which requires a to be square and of full-rank.
    """
    # Convert inputs to PyTorch tensors with the correct dtype
    a_tensor = convert_to_tensor(a, dtype=torch.float32)
    b_tensor = convert_to_tensor(b, dtype=torch.float32)
    
    # Solve the system using torch.linalg.solve
    # This function handles broadcasting and shape checking automatically
    return torch.linalg.solve(a_tensor, b_tensor)


def svd(a: Any, full_matrices: bool = True, compute_uv: bool = True) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Compute the singular value decomposition of a matrix.
    
    Args:
        a: Input matrix
        full_matrices: If True, return full U and Vh matrices
        compute_uv: If True, compute U and Vh matrices
    
    Returns:
        If compute_uv is True, returns (U, S, Vh), otherwise returns S
    """
    a_tensor = convert_to_tensor(a, dtype=torch.float32)
    
    if compute_uv:
        U, S, Vh = torch.linalg.svd(a_tensor, full_matrices=full_matrices)
        return U, S, Vh
    else:
        # PyTorch doesn't have a direct way to compute only S without U and V,
        # so we compute them and return only S
        _, S, _ = torch.linalg.svd(a_tensor, full_matrices=full_matrices)
        return S


def eig(a: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the eigenvalues and eigenvectors of a square matrix.
    
    Args:
        a: Input square matrix
    
    Returns:
        Tuple of (eigenvalues, eigenvectors)
    """
    a_tensor = convert_to_tensor(a, dtype=torch.float32)
    
    # PyTorch's eig function is deprecated, use torch.linalg.eig instead
    eigenvalues, eigenvectors = torch.linalg.eig(a_tensor)
    
    return eigenvalues, eigenvectors


def eigvals(a: Any) -> torch.Tensor:
    """
    Compute the eigenvalues of a square matrix.
    
    Args:
        a: Input square matrix
    
    Returns:
        Eigenvalues of the matrix
    """
    a_tensor = convert_to_tensor(a, dtype=torch.float32)
    
    # PyTorch doesn't have a direct eigvals function, so we use eig and return only the eigenvalues
    eigenvalues, _ = torch.linalg.eig(a_tensor)
    
    return eigenvalues


def inv(a: Any) -> torch.Tensor:
    """
    Compute the inverse of a square matrix.
    
    Args:
        a: Input square matrix
    
    Returns:
        Inverse of the matrix
    """
    a_tensor = convert_to_tensor(a, dtype=torch.float32)
    
    return torch.linalg.inv(a_tensor)


def det(a: Any) -> torch.Tensor:
    """
    Compute the determinant of a square matrix.
    
    Args:
        a: Input square matrix
    
    Returns:
        Determinant of the matrix
    """
    a_tensor = convert_to_tensor(a, dtype=torch.float32)
    
    return torch.linalg.det(a_tensor)


def norm(x: Any, ord: Optional[Union[int, str]] = None, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdim: bool = False) -> torch.Tensor:
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
    x_tensor = convert_to_tensor(x, dtype=torch.float32)
    
    # Handle the case where axis is a tuple of two integers
    dim = axis
    
    return torch.linalg.norm(x_tensor, ord=ord, dim=dim, keepdim=keepdim)


def qr(a: Any, mode: str = 'reduced') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the QR decomposition of a matrix.
    
    Args:
        a: Input matrix
        mode: Mode of decomposition ('reduced', 'complete', 'r', 'raw')
    
    Returns:
        Tuple of (Q, R) matrices
    """
    a_tensor = convert_to_tensor(a, dtype=torch.float32)
    
    # PyTorch's qr function supports 'reduced' and 'complete' modes
    if mode in ['reduced', 'complete']:
        Q, R = torch.linalg.qr(a_tensor, mode=mode)
        return Q, R
    elif mode == 'r':
        # For 'r' mode, we compute QR and return only R
        _, R = torch.linalg.qr(a_tensor, mode='reduced')
        return R
    else:
        raise ValueError(f"Unsupported mode: {mode}. PyTorch supports 'reduced' and 'complete' modes.")


def cholesky(a: Any) -> torch.Tensor:
    """
    Compute the Cholesky decomposition of a matrix.
    
    Args:
        a: Input matrix
    
    Returns:
        Cholesky decomposition of the matrix
    """
    a_tensor = convert_to_tensor(a, dtype=torch.float32)
    
    return torch.linalg.cholesky(a_tensor)


def lstsq(a: Any, b: Any, rcond: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the least-squares solution to a linear matrix equation.
    
    Args:
        a: Coefficient matrix
        b: Dependent variable
        rcond: Cutoff for small singular values
    
    Returns:
        Tuple of (solution, residuals, rank, singular values)
    """
    a_tensor = convert_to_tensor(a, dtype=torch.float32)
    b_tensor = convert_to_tensor(b, dtype=torch.float32)
    
    # PyTorch's lstsq function is deprecated, use torch.linalg.lstsq instead
    # However, torch.linalg.lstsq returns a different format than numpy.linalg.lstsq
    # We need to adapt the output to match numpy's format
    
    # Use torch.linalg.lstsq
    solution = torch.linalg.lstsq(a_tensor, b_tensor, rcond=rcond).solution
    
    # Compute residuals
    diff = torch.subtract(b_tensor, torch.matmul(a_tensor, solution))
    residuals = torch.pow(torch.norm(diff, dim=0), 2)
    
    # Compute rank and singular values
    U, S, Vh = torch.linalg.svd(a_tensor)
    rcond_value = rcond if rcond is not None else 1e-15
    rcond_tensor = convert_to_tensor(rcond_value, dtype=torch.float32)
    threshold = torch.multiply(rcond_tensor, S[0])
    rank = torch.sum(torch.gt(S, threshold))
    
    return solution, residuals, rank, S


class TorchSolverOps:
    """PyTorch implementation of solver operations."""
    
    def solve(self, a, b):
        """Solve a linear system of equations."""
        return solve(a, b)
    
    def svd(self, a, full_matrices=True, compute_uv=True):
        """Compute the singular value decomposition of a matrix."""
        return svd(a, full_matrices=full_matrices, compute_uv=compute_uv)
    
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
        return norm(x, ord=ord, axis=axis, keepdim=keepdims)
    
    def qr(self, a, mode='reduced'):
        """Compute the QR decomposition of a matrix."""
        return qr(a, mode=mode)
    
    def cholesky(self, a):
        """Compute the Cholesky decomposition of a matrix."""
        return cholesky(a)
    
    def lstsq(self, a, b, rcond=None):
        """Compute the least-squares solution to a linear matrix equation."""
        return lstsq(a, b, rcond=rcond)