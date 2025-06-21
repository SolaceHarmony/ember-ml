"""
Container modules for ember_ml.

This module provides backend-agnostic implementations of container modules
that work with any backend (NumPy, PyTorch, MLX).
"""

from ember_ml.nn.layers.linear import Linear
from ember_ml.nn.layers.dropout import Dropout
from ember_ml.nn.layers.sequential import Sequential
from ember_ml.nn.layers.batch_normalization import BatchNormalization

# Export all functions and classes
__all__ = [
    
    # Operations
    'Linear',
    'Dropout',
    'Sequential',
    'BatchNormalization',
    # Removed 'Dense' export
]