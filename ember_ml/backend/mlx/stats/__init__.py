"""
MLX statistical operations for ember_ml.

This module provides MLX implementations of statistical operations.
"""

from ember_ml.backend.mlx.stats.stats_ops import MLXStatsOps
from ember_ml.backend.mlx.stats.ops import (
    # Descriptive statistics
    median,
    std,
    percentile
)

__all__ = [
    "MLXStatsOps",
    "median",
    "std",
    "percentile"
]