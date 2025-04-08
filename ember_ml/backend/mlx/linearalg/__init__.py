"""MLX linear algebra operations for ember_ml."""

# Removed MLXLinearAlgOps import
# from ember_ml.backend.mlx.linearalg.linearalg_ops import MLXLinearAlgOps

# Import directly from moved files using absolute paths
from ember_ml.backend.mlx.linearalg.decomp_ops import qr, svd, cholesky, eig, eigvals, eigh # Added eigh
from ember_ml.backend.mlx.linearalg.inverses_ops import inv # Assuming function is here
from ember_ml.backend.mlx.linearalg.matrix_ops import det, norm, diag, diagonal # Assuming functions are here
from ember_ml.backend.mlx.linearalg.solvers_ops import solve, lstsq # Assuming functions are here
# Note: decomp_ops_hpc.py and qr_128.py might contain specialized versions not directly imported here

__all__ = [
    # "MLXLinearAlgOps", # Removed class export
    "norm",
    "inv",
    "solve",
    "eig",
    "eigvals",
    "eigh", # Added eigh
    "qr",
    "det",
    "cholesky",
    "lstsq",
    "svd",
    "diag",
    "diagonal"
]