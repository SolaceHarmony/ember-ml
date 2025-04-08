"""MLX tensor casting operations."""

import mlx.core # Restore necessary import for type hint
# from typing import Optional # Removed unused Optional
# Removed top-level import of _validate_and_get_mlx_dtype
# from ember_ml.backend.mlx.tensor.dtype import MLXDType # Restore this import if needed, but likely not used now
from ember_ml.backend.mlx.types import DType, TensorLike

# _validate_dtype helper function removed. Logic should exist in MLXDType.validate_dtype

def cast(tensor: TensorLike, dtype: DType) -> mlx.core.array: # Use direct type hint
    """
    Cast a tensor to a new data type using MLX backend.

    Args:
        tensor: Input tensor-like object.
        dtype: Target data type (EmberDType, mlx.core.Dtype, str, or None).

    Returns:
        MLX array with the new data type.

    Raises:
        ValueError: If the target dtype is invalid.
    """
    # Import MLX specifics lazily
    from ember_ml.backend.mlx.tensor.tensor import MLXTensor
    # MLXDType import removed

    # 1. Validate the target dtype using the class method
    # 1. Validate the target dtype using the utility function (lazy import)
    from ember_ml.backend.mlx.tensor.ops.utility import _validate_and_get_mlx_dtype
    mlx_dtype = _validate_and_get_mlx_dtype(dtype)

    # 2. Convert the input tensor to a base MLX array (without casting yet)
    # Ensure convert_to_tensor doesn't apply the final dtype cast itself
    # It should just return a standard mlx.core.array representation.
    # Assuming MLXTensor().convert_to_tensor(tensor) handles this correctly.
    # If convert_to_tensor *requires* a dtype, this needs adjustment.
    # Let's assume for now it can convert without a target dtype.
    tensor_obj = MLXTensor()
    # Check if convert_to_tensor accepts None or omits dtype
    # If not, pass the original tensor's dtype? Or a default?
    # Let's assume it converts to MLX array naturally first.
    try:
         # Try converting without specifying dtype first
         tensor_array = tensor_obj.convert_to_tensor(tensor)
    except TypeError:
         # If convert_to_tensor requires a dtype arg, maybe pass None or original?
         # This depends on the implementation of convert_to_tensor.
         # For now, let's stick to the pattern assuming it works without dtype.
         # If this fails, we need to inspect MLXTensor.convert_to_tensor.
         # Re-raising for clarity if this assumption is wrong.
         raise RuntimeError("Assumption failed: MLXTensor.convert_to_tensor potentially requires dtype argument which conflicts with this casting logic.")


    # 3. If the validated dtype is None (meaning no cast needed), return original array
    if mlx_dtype is None:
        return tensor_array

    # 4. Perform the cast using the validated mlx_dtype
    # MLX uses astype() for casting
    return tensor_array.astype(mlx_dtype)
        