"""
MLX backend configuration for ember_ml.

This module provides configuration settings for the MLX backend.
"""

import mlx.core as mx

# Default device for MLX operations
DEFAULT_DEVICE = mx.default_device().type

# Default data type for MLX operations
DEFAULT_DTYPE = mx.float32