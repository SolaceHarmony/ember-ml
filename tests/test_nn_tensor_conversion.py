import pytest
import importlib
import typing
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn.tensor.common.ember_tensor import EmberTensor
import numpy as np # Okay to import numpy here for testing output types

# Assume conftest.py provides 'backend' fixture

@pytest.fixture(params=[
    (tensor.int32, tensor.float32),
    (tensor.float32, tensor.int64),
    (tensor.float64, tensor.bool_),
    (tensor.bool_, tensor.int8),
    (tensor.float32, tensor.float64), # Add float->float
    (tensor.int16, tensor.int64),     # Add int->int
])
def dtype_pair(request):
    """Fixture providing pairs of dtypes for casting tests."""
    return request.param

# Helper function to get allowed types (EmberTensor + raw backend type)
def get_allowed_tensor_types(backend_name: str) -> tuple:
    """Gets the tuple of allowed return types for tensor operations."""
    allowed_types = (EmberTensor,)
    if backend_name == 'numpy':
        # np is imported at module level
        allowed_types += (np.ndarray,)
    elif backend_name == 'torch':
        try:
            import torch # Import locally only when needed for this check
            allowed_types += (torch.Tensor,)
        except ImportError:
            print("Torch not installed, skipping torch type check.")
            pass # Ignore if torch isn't installed
    elif backend_name == 'mlx':
        try:
            import mlx.core as mx # Import locally only when needed for this check
            allowed_types += (mx.array,)
        except ImportError:
             print("MLX not installed, skipping mlx type check.")
             pass # Ignore if mlx isn't installed
    return allowed_types


def test_tensor_cast(backend, dtype_pair):
    # --- MLX float64 GPU Check (Moved to the beginning) ---
    original_dtype_str = str(dtype_pair[0])
    target_dtype_str = str(dtype_pair[1])
    if backend == 'mlx' and 'float64' in [original_dtype_str, target_dtype_str]:
        try:
            import mlx.core as mx
            # Attempt to create AND cast a float64 array to check for support
            temp_array = mx.array(1.0, dtype=mx.float32) # Create a safe type first
            _ = temp_array.astype(mx.float64) # Now attempt the cast
        except ValueError as e:
             # Make check case-insensitive and check substring
            if "float64 is not supported" in str(e).lower():
                pytest.skip(f"Skipping MLX float64 test: {e}")
            else:
                # Re-raise if it's a different ValueError
                raise e
        except ImportError:
             pytest.skip("MLX not installed, skipping float64 check.") # Skip if MLX isn't even installed
    # --- End MLX float64 Check ---

    """Tests tensor.cast between different data types."""
    ops.set_backend(backend)
    original_dtype, target_dtype = dtype_pair
    
    # Create an initial tensor
    # Handle bool creation carefully
    if original_dtype == tensor.bool_:
         # Need True/False values
         initial_data = [[True, False], [False, True]]
         t_original = tensor.array(initial_data, dtype=original_dtype)
    elif 'int' in str(original_dtype):
         initial_data = [[1, 0], [5, -2]]
         t_original = tensor.array(initial_data, dtype=original_dtype)
    else: # float
         initial_data = [[1.5, 0.0], [5.2, -2.9]]
         t_original = tensor.array(initial_data, dtype=original_dtype)

    # Cast the tensor
    t_casted = tensor.cast(t_original, target_dtype)

    # Assertions
    # Assertions
    allowed_types = get_allowed_tensor_types(backend)
    assert isinstance(t_casted, allowed_types), f"Cast result type ({type(t_casted)}) is not EmberTensor or the expected raw backend type for '{backend}'"
    assert tensor.shape(t_casted) == tensor.shape(t_original), "Cast changed shape"
    # Compare dtype names for reliable comparison across types
    casted_dtype_str = tensor.to_dtype_str(tensor.dtype(t_casted))
    target_dtype_str = str(target_dtype) # EmberDType's __str__ returns its name
    assert casted_dtype_str == target_dtype_str, f"Cast failed: expected dtype '{target_dtype_str}', got '{casted_dtype_str}'"
    
    # Optional: Check if values are reasonably preserved (e.g., float -> int truncation)
    # Convert back to numpy for easier value checks in test logic (but use ops for actual assertions if possible)
    np_original = tensor.to_numpy(t_original)
    np_casted = tensor.to_numpy(t_casted)

    # Example: Check if int(1.5) became 1
    if original_dtype == tensor.float32 and 'int' in str(target_dtype):
        assert np_casted[0, 0] == 1, "Cast float->int truncation failed"
    # Example: Check if bool(5) became True
    if 'int' in str(original_dtype) and target_dtype == tensor.bool_:
         assert np_casted[1, 0] is True, "Cast int->bool conversion failed (non-zero)"
         assert np_casted[0, 1] is False, "Cast int->bool conversion failed (zero)"
    # Example: Check bool -> int
    if original_dtype == tensor.bool_ and 'int' in str(target_dtype):
         assert np_casted[0, 0] == 1, "Cast bool->int conversion failed (True)"
         assert np_casted[0, 1] == 0, "Cast bool->int conversion failed (False)"


def test_tensor_to_numpy(backend):
    """Tests tensor.to_numpy conversion."""
    ops.set_backend(backend)
    data = [[1.1, 2.2], [3.3, 4.4]]
    t_ember = tensor.convert_to_tensor(data)
    
    # Convert to numpy
    t_numpy = tensor.to_numpy(t_ember)

    # Check type and content
    assert isinstance(t_numpy, np.ndarray), "to_numpy did not return np.ndarray"
    assert np.allclose(t_numpy, np.array(data)), "to_numpy content mismatch"
    assert t_numpy.shape == tuple(tensor.shape(t_ember)), "to_numpy shape mismatch"

def test_tensor_item(backend):
    """Tests tensor.item for scalar tensors."""
    ops.set_backend(backend)
    
    # Integer scalar
    t_int = tensor.convert_to_tensor(42)
    item_int = tensor.item(t_int)
    assert isinstance(item_int, (int, np.integer)), "item() did not return int/np.integer for int tensor"
    assert item_int == 42, "item() value mismatch for int tensor"

    # Float scalar
    t_float = tensor.convert_to_tensor(3.14)
    item_float = tensor.item(t_float)
    # Type might be float or np.float - check approximate value
    assert isinstance(item_float, (float, np.floating)), "item() did not return float/np.floating for float tensor"
    assert abs(item_float - 3.14) < 1e-6, "item() value mismatch for float tensor"

    # Boolean scalar
    t_bool = tensor.convert_to_tensor(True)
    item_bool = tensor.item(t_bool)
    # Type might be bool or np.bool_
    assert isinstance(item_bool, (bool, np.bool_)), "item() did not return bool/np.bool_ for bool tensor"
    assert item_bool is True, "item() value mismatch for bool tensor"

    # Test on non-scalar tensor (should raise error)
    t_non_scalar = tensor.convert_to_tensor([1, 2])
    with pytest.raises(Exception): # Use more specific exception if known (e.g., ValueError, TypeError)
        tensor.item(t_non_scalar)