"""
Implementation of the solve operation for different backends.

This file provides implementations of the solve operation for NumPy, PyTorch, and MLX backends.
The solve operation solves a linear system of equations Ax = b for x.
"""

def numpy_solve(a, b):
    """
    Solve a linear system of equations Ax = b for x using NumPy backend.
    
    Parameters
    ----------
    a : tensor
        Coefficient matrix A.
    b : tensor
        Right-hand side vector or matrix b.
    
    Returns
    -------
    tensor
        Solution to the system of equations.
    
    Notes
    -----
    Uses numpy.linalg.solve which requires a to be square and of full-rank.
    """
    import numpy as np
    return np.linalg.solve(a, b)

def torch_solve(a, b):
    """
    Solve a linear system of equations Ax = b for x using PyTorch backend.
    
    Parameters
    ----------
    a : tensor
        Coefficient matrix A.
    b : tensor
        Right-hand side vector or matrix b.
    
    Returns
    -------
    tensor
        Solution to the system of equations.
    
    Notes
    -----
    Uses torch.linalg.solve which requires a to be square and of full-rank.
    """
    import torch
    return torch.linalg.solve(a, b)

def mlx_solve(a, b):
    """
    Solve a linear system of equations Ax = b for x using MLX backend.
    
    Parameters
    ----------
    a : tensor
        Coefficient matrix A.
    b : tensor
        Right-hand side vector or matrix b.
    
    Returns
    -------
    tensor
        Solution to the system of equations.
    
    Notes
    -----
    Uses mlx.core.linalg.solve which requires a to be square and of full-rank.
    """
    import mlx.core as mx
    return mx.linalg.solve(a, b)