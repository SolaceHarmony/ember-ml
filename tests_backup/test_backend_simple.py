"""
Simple test for the EmberTensor backend.

This script demonstrates using the EmberTensor backend for basic tensor operations.
"""

from ember_ml import ops
from ember_ml.backend import set_backend, get_backend
from ember_ml.nn.tensor import EmberTensor
from ember_ml.nn import tensor
# Save the current backend
original_backend = get_backend()

# Set the backend to numpy
print(f"Setting backend to 'numpy' (was '{original_backend}')")
set_backend('numpy')

# Verify the backend is set to ember
current_backend = get_backend()
print(f"Current backend: {current_backend}")

# Create a tensor
print("\nCreating a tensor...")
tensor_obj = tensor.convert_to_tensor([1, 2, 3])
print(f"Tensor: {tensor_obj}")
print(f"Type: {type(tensor_obj)}")
print(f"Shape: {tensor_obj.shape}")

# Create a tensor of zeros
print("\nCreating a tensor of zeros...")
zeros_tensor = tensor.zeros((2, 3))
print(f"Zeros tensor: {zeros_tensor}")
print(f"Shape: {zeros_tensor.shape}")

# Create a tensor of ones
print("\nCreating a tensor of ones...")
ones_tensor = tensor.ones((2, 3))
print(f"Ones tensor: {ones_tensor}")
print(f"Shape: {ones_tensor.shape}")

# Reshape a tensor
print("\nReshaping a tensor...")
tensor_obj = tensor.convert_to_tensor([1, 2, 3, 4, 5, 6])
reshaped = tensor.reshape(tensor_obj, (2, 3))
print(f"Original tensor: {tensor_obj}")
print(f"Reshaped tensor: {reshaped}")
print(f"Shape: {reshaped.shape}")

# Expand dimensions of a tensor
print("\nExpanding dimensions of a tensor...")
tensor_obj = tensor.convert_to_tensor([1, 2, 3])
expanded = tensor.expand_dims(tensor_obj, 0)
print(f"Original tensor: {tensor_obj}")
print(f"Expanded tensor: {expanded}")
print(f"Shape: {expanded.shape}")

# Restore the original backend
print(f"\nRestoring backend to '{original_backend}'")
if original_backend is not None:
    set_backend(original_backend)
else:
    # If original_backend was None, use the default backend
    set_backend('numpy')
print(f"Current backend: {get_backend()}")