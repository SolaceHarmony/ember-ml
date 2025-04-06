"""
PyTorch statistical operations for ember_ml.

This module provides PyTorch implementations of statistical operations.
"""

from ember_ml.backend.torch.stats.stats_ops import TorchStatsOps
from ember_ml.backend.torch.stats.ops import (
    # Descriptive statistics
    median,
    std,
    percentile
)

__all__ = [
    "TorchStatsOps",
    "median",
    "std",
    "percentile"
]