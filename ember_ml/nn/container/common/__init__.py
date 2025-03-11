"""
Common container implementations.

This module provides backend-agnostic implementations of container modules
using the ops abstraction layer.
"""

from ember_ml.nn.container.common.linear import Linear
from ember_ml.nn.container.common.dropout import Dropout
from ember_ml.nn.container.common.sequential import Sequential

__all__ = [
    'Linear',
    'Dropout',
    'Sequential',
]