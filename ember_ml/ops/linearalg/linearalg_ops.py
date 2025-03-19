"""
Linear Algebra operations interface.

This module defines the abstract interface for solver operations.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, Union

class LinearAlgOps(ABC):
    """Abstract interface for solver operations."""
    
    @abstractmethod
    def solve(self, a: Any, b: Any) -> Any:
        """
        Solve a linear system of equations Ax = b for x.
        
        Args:
            a: Coefficient matrix A
            b: Right-hand side vector or matrix b
            
        Returns:
            Solution to the system of equations
        """
        pass
    
    @abstractmethod
    def inv(self, a: Any) -> Any:
        """
        Compute the inverse of a square matrix.
        
        Args:
            a: Input square matrix
            
        Returns:
            Inverse of the matrix
        """
        pass
    
    @abstractmethod
    def det(self, a: Any) -> Any:
        """
        Compute the determinant of a square matrix.
        
        Args:
            a: Input square matrix
            
        Returns:
            Determinant of the matrix
        """
        pass
    
    @abstractmethod
    def norm(self, x: Any, ord: Optional[Union[int, str]] = None,
             axis: Optional[Union[int, Tuple[int, ...]]] = None,
             keepdims: bool = False) -> Any:
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
        pass
    
    @abstractmethod
    def qr(self, a: Any, mode: str = 'reduced') -> Tuple[Any, Any]:
        """
        Compute the QR decomposition of a matrix.
        
        Args:
            a: Input matrix
            mode: Mode of decomposition ('reduced', 'complete', 'r', 'raw')
            
        Returns:
            Tuple of (Q, R) matrices
        """
        pass
    
    @abstractmethod
    def svd(self, a: Any, full_matrices: bool = True, compute_uv: bool = True) -> Union[Any, Tuple[Any, Any, Any]]:
        """
        Compute the singular value decomposition of a matrix.
        
        Args:
            a: Input matrix
            full_matrices: If True, return full U and Vh matrices
            compute_uv: If True, compute U and Vh matrices
            
        Returns:
            If compute_uv is True, returns (U, S, Vh), otherwise returns S
        """
        pass
    
    @abstractmethod
    def cholesky(self, a: Any) -> Any:
        """
        Compute the Cholesky decomposition of a positive definite matrix.
        
        Args:
            a: Input positive definite matrix
            
        Returns:
            Lower triangular matrix L such that L @ L.T = A
        """
        pass
    
    @abstractmethod
    def lstsq(self, a: Any, b: Any, rcond: Optional[float] = None) -> Tuple[Any, Any, Any, Any]:
        """
        Compute the least-squares solution to a linear matrix equation.
        
        Args:
            a: Coefficient matrix
            b: Dependent variable
            rcond: Cutoff for small singular values
            
        Returns:
            Tuple of (solution, residuals, rank, singular values)
        """
        pass
    
    @abstractmethod
    def eig(self, a: Any) -> Tuple[Any, Any]:
        """
        Compute the eigenvalues and eigenvectors of a square matrix.
        
        Args:
            a: Input square matrix
            
        Returns:
            Tuple of (eigenvalues, eigenvectors)
        """
        pass
    
    @abstractmethod
    def eigvals(self, a: Any) -> Any:
        """
        Compute the eigenvalues of a square matrix.
        
        Args:
            a: Input square matrix
            
        Returns:
            Eigenvalues of the matrix
        """
        pass
    
    @abstractmethod
    def diag(self, x: Any, k: int = 0):
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
        pass
    
    @abstractmethod
    def diagonal(self, x: Any, offset: int = 0, axis1: int = 0, axis2: int = 1) -> Any:
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
        pass
    
    @abstractmethod
    def eigvals(self, a: Any) -> Any:
        """
        Compute the eigenvalues of a square matrix.
        
        Args:
            a: Input square matrix
            
        Returns:
            Eigenvalues of the matrix
        """
        pass
    
    