"""
Linear Algebra operations module.

This module provides a unified interface to linear algebra operations from the current backend
(NumPy, PyTorch, MLX) using the proxy module pattern. It dynamically forwards
attribute access to the appropriate backend module.
"""

# Import the linearalg proxy from the ops proxy module
from ember_ml.ops.proxy import linearalg as linearalg_proxy

# Import all operations from the linearalg proxy
solve = linearalg_proxy.solve
inv = linearalg_proxy.inv
svd = linearalg_proxy.svd
eig = linearalg_proxy.eig
eigh = linearalg_proxy.eigh
eigvals = linearalg_proxy.eigvals
det = linearalg_proxy.det
norm = linearalg_proxy.norm
qr = linearalg_proxy.qr
cholesky = linearalg_proxy.cholesky
lstsq = linearalg_proxy.lstsq
diag = linearalg_proxy.diag
diagonal = linearalg_proxy.diagonal
orthogonal = linearalg_proxy.orthogonal
HPC16x8 = linearalg_proxy.HPC16x8

# Define __all__ to include all operations
__all__ = [
    'solve',
    'inv',
    'svd',
    'eig',
    'eigh',
    'eigvals',
    'det',
    'norm',
    'qr',
    'cholesky',
    'lstsq',
    'diag',
    'diagonal',
    'orthogonal',
    'HPC16x8'
]
