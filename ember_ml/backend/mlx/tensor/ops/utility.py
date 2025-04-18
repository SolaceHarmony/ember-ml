"""MLX tensor utility operations."""

from typing import Any, Optional, Sequence, Tuple, Union

import mlx.core as mx # Ensure mx is imported
import numpy as np # Use standard alias for clarity, though we avoid direct calls
# Removed DTypeHandler import
from ember_ml.backend.mlx.types import TensorLike, DType

# Removed DTypeHandler instance

# --- NEW DTYPE VALIDATION FUNCTION ---
def _validate_and_get_mlx_dtype(dtype: Optional[Any]) -> Optional[mx.Dtype]:
    """
    Validate and convert input dtype to an MLX Dtype.

    Args:
        dtype: Input dtype (string, EmberDType, MLX Dtype, None)

    Returns:
        Validated MLX Dtype or None
    """
    if dtype is None:
        return None

    # If it's already an MLX dtype, return it
    if isinstance(dtype, mx.Dtype):
        return dtype

    # Handle string dtypes or objects with a 'name' attribute
    dtype_name = None
    if isinstance(dtype, str):
        dtype_name = dtype
    elif hasattr(dtype, 'name'): # Handles EmberDType
        dtype_name = str(dtype.name)

    if dtype_name:
        # Map dtype names to MLX dtypes
        if dtype_name == 'float32':
            return mx.float32
        elif dtype_name == 'float64': # Map float64 to float32 for MLX
             return mx.float32
        elif dtype_name == 'int32':
            return mx.int32
        elif dtype_name == 'int64':
            return mx.int64
        elif dtype_name in ('bool', 'bool_'):
            return mx.bool_ # Fallback
        elif dtype_name == 'int8':
            return mx.int8
        elif dtype_name == 'int16':
            return mx.int16
        elif dtype_name == 'uint8':
            return mx.uint8
        elif dtype_name == 'uint16':
            return mx.uint16
        elif dtype_name == 'uint32':
            return mx.uint32
        elif dtype_name == 'uint64':
            return mx.uint64
        elif dtype_name == 'float16':
            return mx.float16
        elif dtype_name == 'complex64':
             # Check if complex64 is actually supported by the installed mlx version
             if hasattr(mx, 'complex64'):
                 return mx.complex64
             else:
                 raise ValueError(f"MLX backend does not support complex64 dtype.")
        else:
            raise ValueError(f"Unknown data type name: {dtype_name}")

    # If it's not a string, EmberDType, or MLX Dtype, it's invalid
    raise ValueError(f"Invalid dtype: {dtype} of type {type(dtype)}")
# --- END NEW DTYPE VALIDATION FUNCTION ---

def _convert_input(x: TensorLike) -> Any:
    """
    Convert input to MLX array.

    Handles various input types:
    - MLX arrays (returned as-is)
    - NumPy arrays (converted to MLX arrays)
    - MLXTensor objects (extract underlying data)
    - EmberTensor objects (extract underlying data)
    - Python scalars (int, float, bool)
    - Python sequences (list, tuple)
    - None values (converted to default zero tensor)

    Special handling for:
    - 0D tensors (scalars)
    - 1D tensors (vectors)
    - 2D tensors (matrices)
    - Higher dimensional tensors

    Args:
        x: Input data to convert

    Returns:
        MLX array

    Raises:
        ValueError: If the input cannot be converted to an MLX array
    """
    # Handle None values
    if x is None:
        # Create a default zero tensor (scalar)
        from ember_ml.backend.mlx.types import default_float
        return mx.array(0.0, dtype=default_float)
        
    # Already an MLX array - check by type and module
    # Using isinstance for potentially more robust check
    if isinstance(x, mx.array):
         return x
    # Fallback check if isinstance fails for some reason
    elif (hasattr(x, '__class__') and
        hasattr(x.__class__, '__module__') and
        x.__class__.__module__ == 'mlx.core' and
        x.__class__.__name__ == 'array'):
        return x

    # Handle MLXTensor objects
    if (hasattr(x, '__class__') and
        hasattr(x.__class__, '__name__') and
        x.__class__.__name__ == 'MLXTensor'):
        # Assuming MLXTensor._tensor holds an mx.array
        if hasattr(x, '_tensor') and isinstance(x._tensor, mx.array):
             return x._tensor
        else:
             # Handle case where _tensor might not be set or is wrong type
             raise ValueError(f"MLXTensor does not have a valid '_tensor' attribute: {x}")


    # Handle EmberTensor objects
    if (hasattr(x, '__class__') and
        hasattr(x.__class__, '__name__') and
        x.__class__.__name__ == 'EmberTensor'):
        if hasattr(x, '_tensor'):
             # The tensor within EmberTensor might be from another backend,
             # so we recursively call _convert_input to handle it properly.
             return _convert_input(x._tensor)
        else:
          raise ValueError(f"EmberTensor does not have a '_tensor' attribute: {x}")

    # Handle Parameter objects
    # Check by class name to avoid direct import which might cause circular dependencies
    if (hasattr(x, '__class__') and
        hasattr(x.__class__, '__name__') and
        x.__class__.__name__ == 'Parameter'):
        if hasattr(x, 'data'):
            # Recursively convert the underlying data
            return _convert_input(x.data)
        else:
            raise ValueError(f"Parameter object does not have a 'data' attribute: {x}")

    # Check for NumPy arrays by type name rather than direct import
    if (hasattr(x, '__class__') and
        x.__class__.__module__ == 'numpy' and
        x.__class__.__name__ == 'ndarray'):
        # Ensure float64 numpy arrays become float32 mlx arrays
        if x.dtype == np.float64:
            return mx.array(x, dtype=mx.float32)
        else:
            # Let MLX handle other numpy dtype conversions
            return mx.array(x)

    # Handle NumPy scalar types using hasattr
    if (hasattr(x, 'item') and # Check for item method common to numpy scalars
        hasattr(x, '__class__') and
        hasattr(x.__class__, '__module__') and
        x.__class__.__module__ == 'numpy'):
        try:
            # Convert NumPy scalar to its Python equivalent, then to MLX array
            # Use the new validation function to get the appropriate MLX dtype
            py_scalar = x.item()
            mlx_dtype = _validate_and_get_mlx_dtype(x.dtype) # Get MLX dtype from numpy dtype
            if mlx_dtype:
                 return mx.array(py_scalar, dtype=mlx_dtype)
            else:
                 # Fallback if dtype validation fails
                 return mx.array(py_scalar)
        except Exception as e:
             raise ValueError(f"Cannot convert NumPy scalar {type(x)} to MLX array: {e}")

    # Handle Python scalars (int, float, bool), EXCLUDING NumPy scalars handled above
    is_python_scalar = isinstance(x, (int, float, bool))
    is_numpy_scalar = (hasattr(x, 'item') and hasattr(x, '__class__') and hasattr(x.__class__, '__module__') and x.__class__.__module__ == 'numpy')

    if is_python_scalar and not is_numpy_scalar:
        try:
            # Import default_float and default_int for type conversion
            from ember_ml.backend.mlx.types import default_float, default_int

            # Map Python types to default MLX types
            if isinstance(x, float):
                # Convert to MLX float32
                return mx.array(x, dtype=default_float)
            elif isinstance(x, int):
                return mx.array(x, dtype=default_int)
            else:  # bool
                return mx.array(x, dtype=getattr(mx, 'bool_', mx.uint8)) # Use bool if available
        except Exception as e:
            raise ValueError(f"Cannot convert Python scalar {type(x)} to MLX array: {e}")

    # Handle Python sequences (potential 1D or higher tensors) recursively
    if isinstance(x, (list, tuple)):
       try:
            # Convert items first before creating the final array
            converted_items = [_convert_input(item) for item in x]
            # Let MLX determine the best dtype from the converted items
            return mx.array(converted_items)
       except Exception as e:
            # Safely get item types, handling potential errors
            try:
                item_types = [type(item).__name__ for item in x[:10]]  # Limit to first 10 items for safety
                if len(x) > 10:
                    item_types.append("...")
            except Exception:
                item_types = ["<unknown>"]
            raise ValueError(f"Cannot convert sequence {type(x)} with item types {item_types} to MLX array: {str(e)}")


    # For any other type, reject it with a corrected list of supported types
    raise ValueError(f"Cannot convert {type(x)} to MLX array. Supported types: Python scalars/sequences, NumPy scalars/arrays, MLXTensor, EmberTensor, Parameter.")

def _convert_to_tensor(data: TensorLike, dtype: Optional[DType] = None, device: Optional[str] = None) -> mx.array:
    """
    Convert input to MLX array with specific dtype handling.

    Args:
        data: Input data
        dtype: Optional desired data type (string, EmberDType, MLX Dtype).
        device: Ignored for MLX backend

    Returns:
        MLX array
    """
    # Initial conversion using the refined _convert_input
    tensor = _convert_input(data)
    current_mlx_dtype = tensor.dtype

    # Validate and get the target MLX dtype
    target_mlx_dtype = _validate_and_get_mlx_dtype(dtype) # Use the new validation function

    # Apply dtype conversion if necessary
    if target_mlx_dtype is not None and target_mlx_dtype != current_mlx_dtype:
        tensor = tensor.astype(target_mlx_dtype)
    # If target_mlx_dtype is None (meaning user passed dtype=None),
    # we keep the dtype determined by _convert_input or MLX's default.
    # No need for default int/float logic here as _convert_input handles initial conversion.

    # Ensure proper dimensionality for scalars (redundant check removed, handled in _convert_input)
    # if isinstance(data, (int, float, bool)) and tensor.ndim > 0:
    #    tensor = mx.reshape(tensor, ())

    return tensor

def to_numpy(data: TensorLike) -> np.ndarray: # Changed return type hint
    """
    Convert an MLX array to a NumPy array.

    IMPORTANT: This function is provided ONLY for visualization/plotting libraries
    that specifically require NumPy arrays. It should NOT be used for general tensor
    conversions or operations. Ember ML has a zero backend design where EmberTensor
    relies entirely on the selected backend for representation.

    Args:
        data: Input MLX array or compatible type

    Returns:
        NumPy array
    """
    # Convert input to MLX array first
    tensor_data = _convert_input(data)
    # Use the built-in numpy conversion
    return np.array(tensor_data)


def item(data: TensorLike) -> Union[int, float, bool]:
    """
    Extract the scalar value from a tensor.

    Args:
        data: Input tensor containing a single element

    Returns:
        Standard Python scalar (int, float, or bool)
    """
    tensor_array = _convert_input(data)

    if tensor_array.size != 1:
        raise ValueError(f"item() can only be called on tensors with a single element, but got size {tensor_array.size}")

    # Get the raw value using MLX item()
    raw_value = tensor_array.item()

    # MLX item() returns Python types directly
    return raw_value


def shape(data: TensorLike) -> Tuple[int, ...]:
    """
    Get the shape of a tensor.

    Args:
        data: Input array

    Returns:
        Shape of the array
    """
    # No need for MLXTensor instance, just convert and get shape
    return _convert_input(data).shape

def dtype(data: TensorLike) -> str: # Return type is now string
    """
    Get the string representation of a tensor's data type.

    Args:
        data: Input array

    Returns:
        String representation of the array's data type (e.g., 'float32', 'int64').
    """
    from ember_ml.backend.mlx.tensor.dtype import MLXDType # For converting native to string

    native_dtype = _convert_input(data).dtype

    # Convert native MLX dtype to string representation
    mlx_dtype_helper = MLXDType()
    dtype_str = mlx_dtype_helper.to_dtype_str(native_dtype)

    return dtype_str


def copy(data: TensorLike) -> mx.array:
    """
    Create a copy of an MLX array.

    Args:
        data: Input array

    Returns:
        Copy of the array
    """
    # MLX arrays are immutable, converting ensures a distinct object if needed,
    # but fundamentally the data buffer might be shared until modification (if mutable).
    # For safety/clarity, explicitly use mx.array() which should handle copying if necessary.
    tensor_array = _convert_input(data)
    return mx.array(tensor_array) # Use mx.array to ensure it's a standard MLX array copy


def var(data: TensorLike, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False, ddof: int = 0) -> mx.array: # Added ddof
    """
    Compute the variance of a tensor.

    Args:
        data: Input array
        axis: Axis or axes along which to compute the variance
        keepdims: Whether to keep the dimensions or not
        ddof: Delta degrees of freedom. The divisor used in calculations is N - ddof,
              where N represents the number of elements. Default: 0.

    Returns:
        Variance of the array
    """
    tensor_array = _convert_input(data)
    # Pass ddof to mx.var
    return mx.var(tensor_array, axis=axis, keepdims=keepdims, ddof=ddof)


def sort(data: TensorLike, axis: int = -1, descending: bool = False) -> mx.array:
    """
    Sort a tensor along the given axis.

    Args:
        data: Input array
        axis: Axis along which to sort
        descending: Whether to sort in descending order

    Returns:
        Sorted array
    """
    tensor_array = _convert_input(data)
    sorted_array = mx.sort(tensor_array, axis=axis)

    if descending:
        # Create a list of slice objects for each dimension
        slices = [slice(None)] * tensor_array.ndim
        # Reverse the array along the specified axis
        # Ensure axis is handled correctly (e.g., negative indexing)
        actual_axis = axis if axis >= 0 else tensor_array.ndim + axis
        if 0 <= actual_axis < tensor_array.ndim:
             slices[actual_axis] = slice(None, None, -1)
             sorted_array = sorted_array[tuple(slices)]
        else:
             raise ValueError(f"Invalid axis {axis} for tensor with {tensor_array.ndim} dimensions")


    return sorted_array


def argsort(data: TensorLike, axis: int = -1, descending: bool = False) -> mx.array:
    """
    Return the indices that would sort a tensor along the given axis.

    Args:
        data: Input array
        axis: Axis along which to sort
        descending: Whether to sort in descending order

    Returns:
        Indices that would sort the array
    """
    tensor_array = _convert_input(data)

    indices = mx.argsort(tensor_array, axis=axis)

    if descending:
        # Create a list of slice objects for each dimension
        slices = [slice(None)] * indices.ndim
        # Reverse the indices along the specified axis
        # Ensure axis is handled correctly
        actual_axis = axis if axis >= 0 else indices.ndim + axis
        if 0 <= actual_axis < indices.ndim:
            slices[actual_axis] = slice(None, None, -1)
            indices = indices[tuple(slices)]
        else:
            raise ValueError(f"Invalid axis {axis} for tensor with {indices.ndim} dimensions")

    return indices


def maximum(data1: TensorLike, data2: TensorLike) -> mx.array:
    """
    Element-wise maximum of two arrays.

    Args:
        data1: First input array
        data2: Second input array

    Returns:
        Element-wise maximum
    """
    data1_array = _convert_input(data1)
    data2_array = _convert_input(data2)
    return mx.maximum(data1_array, data2_array)


def _create_new_tensor(creation_func: callable, dtype: Optional[Any] = None, device: Optional[str] = None, **kwargs) -> mx.array:
    """
    Internal helper to create a new MLX tensor, handling dtype resolution and defaults.
    MLX handles device implicitly based on default device setting.

    Args:
        creation_func: The underlying MLX creation function (e.g., mx.zeros, mx.random.uniform).
        dtype: Optional desired dtype (EmberDType, string, mx.Dtype, None).
        device: Ignored for MLX backend.
        **kwargs: Function-specific arguments (e.g., shape, low, high, loc, scale).

    Returns:
        A new mx.array.
    """
    # Resolve dtype first
    target_mlx_dtype = _validate_and_get_mlx_dtype(dtype)

    # Apply default dtype if none resolved, considering kwargs context
    if target_mlx_dtype is None:
        from ember_ml.backend.mlx.types import default_float, default_int # Import defaults
        # Heuristic based on typical function return types
        if creation_func in [mx.random.normal, mx.random.uniform, mx.full] and not isinstance(kwargs.get('fill_value', 0.0), int):
             target_mlx_dtype = default_float
        elif creation_func in [mx.zeros, mx.ones, mx.full] and isinstance(kwargs.get('fill_value', 0), int):
             target_mlx_dtype = default_int
        else: # Default fallback
             target_mlx_dtype = default_float


    # Ensure shape is a tuple if present in kwargs
    if 'shape' in kwargs:
        shape_arg = kwargs['shape']
        if isinstance(shape_arg, int):
            # Special case: if shape is 0, treat it as a scalar tensor
            if shape_arg == 0:
                kwargs['shape'] = (1,)  # Create a 1D tensor with a single element
            else:
                kwargs['shape'] = (shape_arg,)
        elif not isinstance(shape_arg, tuple):
            kwargs['shape'] = tuple(shape_arg)

    # Call the actual MLX creation function
    try:
        # Pass resolved dtype and all other kwargs
        # Remove device kwarg as MLX doesn't explicitly take it in these functions
        kwargs.pop('device', None)
        return creation_func(dtype=target_mlx_dtype, **kwargs)
    except TypeError as e:
        # Provide more context on failure
        raise TypeError(
            f"{creation_func.__name__} failed. "
            f"Input dtype: {dtype}, Resolved native dtype: {target_mlx_dtype}, "
            f"Type: {type(target_mlx_dtype)}, Kwargs: {kwargs}. Error: {e}"
        )


# Expose necessary functions
__all__ = [
    "_convert_input",
    "_convert_to_tensor",
    "to_numpy",
    "item",
    "shape",
    "dtype", # Now returns string
    "copy",
    "var",
    "sort",
    "argsort",
    "maximum",
    "_validate_and_get_mlx_dtype",
    "_create_new_tensor", # Export the new helper
]