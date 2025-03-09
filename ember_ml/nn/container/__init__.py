"""
Container modules for emberharmony.

This module provides backend-agnostic implementations of container modules
that work with any backend (NumPy, PyTorch, MLX).
"""

from ember_ml.nn.container.sequential import Sequential

__all__ = ['Sequential']