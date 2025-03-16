"""
Simple test script for MLX tensor cast operation.

This script tests the standalone cast() function directly.
"""

import mlx.core as mx
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.abspath('.'))

# Import the MLXTensor class and _validate_dtype function directly
from ember_ml.backend.mlx.tensor.tensor import MLXTensor, _validate_dtype
from ember_ml.backend.mlx.tensor.dtype import MLXDType

# Import the cast function directly
from ember_ml.backend.mlx.tensor.ops.casting import cast

# Create a test tensor
tensor = mx.array([1, 2, 3], dtype=mx.float32)
print("Original tensor dtype:", tensor.dtype)

# Create an MLXTensor instance
tensor_obj = MLXTensor()

# Test the standalone cast() function
result = cast(tensor_obj, tensor, 'float64')
print("Cast result dtype:", result.dtype)

# Verify that the dtype changed
print("Dtype changed:", tensor.dtype != result.dtype)