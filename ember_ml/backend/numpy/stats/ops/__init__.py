"""
NumPy statistical operations implementations.

This module provides NumPy implementations of statistical operations.
"""

from ember_ml.backend.numpy.stats.ops.descriptive import (
    mean,
    var,
    median,
    std,
    percentile,
    max,
    min,
    sum,
    cumsum,
    argmax,
    sort,
    argsort
)

__all__ = [
    "mean",
    "var",
    "median",
    "std",
    "percentile",
    "max",
    "min",
    "sum",
    "cumsum",
    "argmax",
    "sort",
    "argsort"
]