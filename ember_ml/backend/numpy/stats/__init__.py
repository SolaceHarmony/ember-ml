"""
NumPy statistical operations for ember_ml.

This module provides NumPy implementations of statistical operations.
"""

from ember_ml.backend.numpy.stats.stats_ops import NumpyStatsOps
from ember_ml.backend.numpy.stats.ops import (
    # Descriptive statistics
    median,
    std,
    percentile
)

__all__ = [
    "NumpyStatsOps",
    "median",
    "std",
    "percentile"
]