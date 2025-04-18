"""PyTorch tensor utility operations."""

import torch
import numpy as np
from typing import Union, Optional, Sequence, Any, List, Tuple

# Assume types are defined correctly
from ember_ml.backend.torch.types import TensorLike, DType, default_float, default_int, Shape
from ember_ml.backend.torch.tensor.dtype import TorchDType # Keep for from_dtype_str if needed


# --- NEW DTYPE VALIDATION FUNCTION (Ported from MLX) ---
def _validate_and_get_torch_dtype(dtype: Optional[Any]) -> Optional[torch.dtype]:
    """
    Validate and convert input dtype to a torch.dtype.

    Args:
        dtype: Input dtype (string, EmberDType, torch.dtype, None)

    Returns:
        Validated torch.dtype or None
    """
    if dtype is None:
        return None

    # If it's already a torch.dtype, return it
    if isinstance(dtype, torch.dtype):
        return dtype

    # Handle string dtypes or objects with a 'name' attribute
    dtype_name = None
    if isinstance(dtype, str):
        dtype_name = dtype
    elif hasattr(dtype, 'name'): # Handles EmberDType
        dtype_name = str(dtype.name)

    if dtype_name:
        # Map dtype names to torch dtypes
        try:
            # Attempt to get dtype directly from torch using the name
            # Covers float32, float64, int32, int64, bool, int8, int16, float16, complex64 etc.
            # Handle 'bool_' alias if necessary
            if dtype_name == 'bool_':
                 dtype_name = 'bool'
            return getattr(torch, dtype_name)
        except AttributeError:
            raise ValueError(f"Unknown or unsupported data type name for PyTorch backend: {dtype_name}")
        except Exception as e:
            raise ValueError(f"Error converting dtype name '{dtype_name}' to torch.dtype: {e}")

    # If it's not a string, EmberDType, or torch.dtype, it's invalid
    raise ValueError(f"Invalid dtype: {dtype} of type {type(dtype)}")
# --- END NEW DTYPE VALIDATION FUNCTION ---


def _convert_input(x: TensorLike) -> Any:
    """
    Convert input to PyTorch tensor, handling nested structures and default types.
    (Ported logic from MLX _convert_input)
    """
    # Already a Torch tensor
    if isinstance(x, torch.Tensor):
        return x

    # Handle TorchTensor objects
    if (hasattr(x, '__class__') and
        hasattr(x.__class__, '__name__') and
        x.__class__.__name__ == 'TorchTensor'):
        if hasattr(x, '_tensor') and isinstance(x._tensor, torch.Tensor):
             return x._tensor
        else:
             raise ValueError(f"TorchTensor does not have a valid '_tensor' attribute: {x}")

    # Handle EmberTensor objects
    if (hasattr(x, '__class__') and
        hasattr(x.__class__, '__name__') and
        x.__class__.__name__ == 'EmberTensor'):
        if hasattr(x, '_tensor'):
             # Recursively call _convert_input to handle potential backend mismatch
             return _convert_input(x._tensor)
        else:
             raise ValueError(f"EmberTensor does not have a '_tensor' attribute: {x}")

    # Handle Parameter objects
    if (hasattr(x, '__class__') and
        hasattr(x.__class__, '__name__') and
        x.__class__.__name__ == 'Parameter'):
        if hasattr(x, 'data'):
            # Recursively convert the underlying data
            return _convert_input(x.data)
        else:
            raise ValueError(f"Parameter object does not have a 'data' attribute: {x}")

    # Handle NumPy arrays
    if (hasattr(x, '__class__') and
        x.__class__.__module__ == 'numpy' and
        x.__class__.__name__ == 'ndarray'):
        # Use x.copy() to avoid potential memory sharing issues
        # Let torch.from_numpy handle dtype conversion (e.g., float64 -> float32 if needed later)
        return torch.from_numpy(x.copy())

    # Handle NumPy scalar types
    if (hasattr(x, 'item') and
        hasattr(x, '__class__') and
        hasattr(x.__class__, '__module__') and
        x.__class__.__module__ == 'numpy'):
        try:
            # Convert NumPy scalar to its Python equivalent
            py_scalar = x.item()
            # Get the corresponding torch dtype, preserving precision where possible
            torch_dtype = _validate_and_get_torch_dtype(x.dtype)
            if torch_dtype:
                 return torch.tensor(py_scalar, dtype=torch_dtype)
            else:
                 # Fallback if dtype validation fails (should ideally not happen)
                 # Let torch infer from the python type
                 return torch.tensor(py_scalar)
        except Exception as e:
             raise ValueError(f"Cannot convert NumPy scalar {type(x)} to torch.Tensor: {e}")

    # Handle Python scalars (int, float, bool), EXCLUDING NumPy scalars handled above
    is_python_scalar = isinstance(x, (int, float, bool))
    is_numpy_scalar = (hasattr(x, 'item') and hasattr(x, '__class__') and hasattr(x.__class__, '__module__') and x.__class__.__module__ == 'numpy')

    if is_python_scalar and not is_numpy_scalar:
        try:
            # Map Python types to default PyTorch types (float32, int32)
            if isinstance(x, float):
                return torch.tensor(x, dtype=default_float) # Use default_float
            elif isinstance(x, int):
                # Use default_int (torch.int32) for consistency
                return torch.tensor(x, dtype=default_int)
            else:  # bool
                 # Revert to standard boolean tensor creation
                 return torch.tensor(x, dtype=torch.bool)
        except Exception as e:
            raise ValueError(f"Cannot convert Python scalar {type(x)} to torch.Tensor: {e}")

    # Handle Python sequences (potential 1D or higher tensors) recursively
    if isinstance(x, (list, tuple)):
       try:
            # Special case for nested lists (2D or higher)
            if len(x) > 0 and isinstance(x[0], (list, tuple)):
                # Convert to numpy array first, then to torch tensor
                import numpy as np
                return torch.tensor(np.array(x))
            else:
                # For 1D lists, convert items individually
                converted_items = [_convert_input(item) for item in x]
                # Let PyTorch determine the best dtype from the converted items
                # This might promote types (e.g., list of int32 tensors becomes float32 tensor)
                # Consider adding explicit dtype control here if needed later.
                return torch.tensor(converted_items)
       except Exception as e:
            item_types = [type(item) for item in x]
            raise ValueError(f"Cannot convert sequence {type(x)} with item types {item_types} to torch.Tensor: {e}")


    # For any other type, reject it
    raise ValueError(f"Cannot convert {type(x)} to torch.Tensor. Supported types: Python scalars/sequences, NumPy scalars/arrays, TorchTensor, EmberTensor, Parameter.")


def _convert_to_tensor(data: Any, dtype: Optional[Any] = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Convert input to PyTorch tensor with specific dtype and device handling.
    (Ported logic from MLX _convert_to_tensor)
    """
    # Determine the target device *before* conversion
    target_device = device
    if target_device is None:
        from ember_ml.backend.torch.device_ops import get_device # Local import
        target_device = get_device() # Use framework's default device

    # Initial conversion using the refined _convert_input
    # _convert_input handles basic type checks and numpy/scalar conversions
    # It does NOT handle final dtype or device placement yet
    tensor = _convert_input(data)
    current_torch_dtype = tensor.dtype
    current_device = str(tensor.device) # Get device as string

    # Validate and get the target torch dtype
    target_torch_dtype = _validate_and_get_torch_dtype(dtype) # Use the new validation function

    # Apply dtype conversion if necessary
    dtype_changed = False
    if target_torch_dtype is not None and target_torch_dtype != current_torch_dtype:
        try:
            tensor = tensor.to(dtype=target_torch_dtype)
            dtype_changed = True
        except Exception as e:
            raise TypeError(f"Failed to cast tensor from {current_torch_dtype} to {target_torch_dtype}: {e}")

    # Move to the target device if necessary
    # Check device string representation to avoid unnecessary moves
    if target_device and target_device != current_device:
        try:
            # Check if we're moving to MPS device and the tensor is float64
            if 'mps' in target_device and tensor.dtype == torch.float64:
                # Convert to float32 first before moving to MPS
                tensor = tensor.to(dtype=torch.float32)
            # Now move to target device
            tensor = tensor.to(device=target_device)
        except Exception as e:
            raise RuntimeError(f"Failed to move tensor to device '{target_device}': {e}")
    
    # If target_torch_dtype was None, we keep the original or inferred dtype.
    # If target_device was None, we keep the original or inferred device (or the default).
    return tensor


def to_numpy(data: TensorLike) -> Optional[np.ndarray]:
    """
    Convert a PyTorch tensor to a NumPy array.

    IMPORTANT: This function is provided ONLY for visualization/plotting libraries
    that specifically require NumPy arrays. It should NOT be used for general tensor
    conversions or operations. Ember ML has a zero backend design where EmberTensor
    relies entirely on the selected backend for representation.

    Args:
        data: The tensor to convert

    Returns:
        NumPy array
    """
    if data is None:
        return None

    # Convert to tensor first to handle various inputs
    tensor_torch = _convert_to_tensor(data)
    # Ensure tensor is on CPU before converting to NumPy
    return tensor_torch.detach().cpu().numpy()


def item(data: TensorLike) -> Union[int, float, bool]:
    """
    Get the value of a scalar tensor.

    Args:
        data: The tensor to get the value from

    Returns:
        The scalar value
    """
    # Ensure data is a torch tensor first
    tensor_torch = _convert_to_tensor(data)
    # Check if the tensor is scalar before proceeding
    if tensor_torch.numel() != 1:
         raise ValueError("item() can only be called on scalar tensors (tensors with one element)")

    # Handle boolean dtype specifically
    if tensor_torch.dtype == torch.bool:
        # Directly compare the tensor value to True/False tensors
        # This avoids potential issues with .item() returning int for bool
        if torch.equal(tensor_torch, torch.tensor(True, dtype=torch.bool, device=tensor_torch.device)):
            return True
        elif torch.equal(tensor_torch, torch.tensor(False, dtype=torch.bool, device=tensor_torch.device)):
            return False
        else:
            # Should not happen for a scalar boolean tensor
            raise ValueError("Unexpected boolean tensor value encountered in item()")
    else:
        # For other dtypes, use the standard item() method
        return tensor_torch.item()

def shape(data: TensorLike) -> Shape:
    """
    Get the shape of a tensor.

    Args:
        data: The tensor to get the shape of

    Returns:
        The shape of the tensor
    """
    # Convert to tensor first
    return tuple(_convert_to_tensor(data).shape)

def dtype(data: TensorLike) -> str: # Return type is now string
    """
    Get the string representation of a tensor's data type.

    Args:
        data: The tensor to get the data type of

    Returns:
        String representation of the tensor's data type (e.g., 'float32', 'int64').
    """
    from ember_ml.backend.torch.tensor.dtype import TorchDType # For converting native to string

    native_dtype = _convert_to_tensor(data).dtype

    # Convert native Torch dtype to string representation
    torch_dtype_helper = TorchDType()
    dtype_str = torch_dtype_helper.to_dtype_str(native_dtype)

    return dtype_str

def copy(data: TensorLike) -> torch.Tensor:
    """
    Create a copy of a tensor.

    Args:
        data: The tensor to copy

    Returns:
        Copy of the tensor
    """
    tensor_torch = _convert_to_tensor(data)
    return tensor_torch.clone()

def var(data: TensorLike, axis: Optional[Union[int, List[int]]] = None, keepdims: bool = False, ddof: int = 0) -> torch.Tensor: # Added ddof
    """
    Compute the variance of a tensor along specified axes.

    Args:
        data: Input tensor
        axis: Axis or axes along which to compute the variance
        keepdims: Whether to keep the reduced dimensions
        ddof: Delta degrees of freedom.

    Returns:
        Variance of the tensor
    """
    tensor_torch = _convert_to_tensor(data)

    # PyTorch var uses 'dim' and 'keepdim', and 'unbiased' (True means ddof=1, False means ddof=0)
    # We need to handle ddof manually if it's not 0 or 1
    if ddof == 1:
         unbiased = True
         return torch.var(tensor_torch, dim=axis, keepdim=keepdims, unbiased=unbiased)
    elif ddof == 0:
         unbiased = False
         return torch.var(tensor_torch, dim=axis, keepdim=keepdims, unbiased=unbiased)
    else:
         # Manual calculation for other ddof values
         if axis is None:
              numel = tensor_torch.numel()
              mean = torch.mean(tensor_torch)
              squared_diff = torch.square(tensor_torch - mean)
              variance = torch.sum(squared_diff) / (numel - ddof)
              if keepdims:
                   # Find original ndim and create appropriate shape
                   orig_ndim = tensor_torch.ndim
                   new_shape = (1,) * orig_ndim
                   variance = variance.view(new_shape)
              return variance
         else:
              # Ensure axis is a tuple for multi-axis reduction consistency
              if isinstance(axis, int):
                  axis = (axis,)
              numel = torch.prod(torch.tensor([tensor_torch.shape[i] for i in axis])).item()
              mean = torch.mean(tensor_torch, dim=axis, keepdim=True)
              squared_diff = torch.square(tensor_torch - mean)
              variance = torch.sum(squared_diff, dim=axis, keepdim=keepdims) / (numel - ddof) # This numel calculation is likely wrong for axis sum
              # Correct N calculation for axis reduction:
              dims_to_reduce = set(axis) if isinstance(axis, (list, tuple)) else {axis}
              n = 1
              for i, dim_size in enumerate(tensor_torch.shape):
                   if i in dims_to_reduce:
                        n *= dim_size
              corrected_variance = torch.sum(squared_diff, dim=axis, keepdim=keepdims) / (n - ddof)
              return corrected_variance


def sort(data: TensorLike, axis: int = -1, descending: bool = False) -> torch.Tensor:
    """
    Sort a tensor along a specified axis.

    Args:
        data: Input tensor
        axis: Axis along which to sort
        descending: Whether to sort in descending order

    Returns:
        Sorted tensor
    """
    tensor_torch = _convert_to_tensor(data)

    # PyTorch sort returns a tuple of (values, indices)
    values, _ = torch.sort(tensor_torch, dim=axis, descending=descending)

    return values

def argsort(data: TensorLike, axis: int = -1, descending: bool = False) -> torch.Tensor:
    """
    Return the indices that would sort a tensor along a specified axis.

    Args:
        data: Input tensor
        axis: Axis along which to sort
        descending: Whether to sort in descending order

    Returns:
        Indices that would sort the tensor
    """
    tensor_torch = _convert_to_tensor(data)

    # PyTorch sort returns a tuple of (values, indices)
    _, indices = torch.sort(tensor_torch, dim=axis, descending=descending)

    return indices

def maximum(data1: TensorLike, data2: TensorLike) -> torch.Tensor:
    """
    Element-wise maximum of two tensors.

    Args:
        data1: First input tensor
        data2: Second input tensor

    Returns:
        Element-wise maximum
    """
    x_torch = _convert_to_tensor(data1)
    y_torch = _convert_to_tensor(data2)
    return torch.maximum(x_torch, y_torch)


def _create_new_tensor(creation_func: callable, dtype: Optional[Any] = None, device: Optional[str] = None, requires_grad: bool = False, **kwargs) -> torch.Tensor:
    """
    Internal helper to create a new Torch tensor, handling dtype, device, and defaults.

    Args:
        creation_func: The underlying Torch creation function (e.g., torch.zeros, torch.randn).
        dtype: Optional desired dtype (EmberDType, string, torch.dtype, None).
        device: Optional device string.
        requires_grad: Whether the new tensor requires gradients.
        **kwargs: Function-specific arguments (e.g., shape, mean, std, low, high, fill_value).

    Returns:
        A new torch.Tensor.
    """
    # Resolve device first
    target_device = device
    if target_device is None:
        from ember_ml.backend.torch.device_ops import get_device # Local import
        target_device = get_device()

    # Resolve dtype
    target_torch_dtype = _validate_and_get_torch_dtype(dtype)

    # Apply default dtype if none resolved, considering kwargs context
    if target_torch_dtype is None:
        from ember_ml.backend.torch.types import default_float, default_int # Import defaults
        # Heuristic based on typical function return types
        if creation_func in [torch.randn, torch.rand, torch.rand_like, torch.randn_like, torch.full] and not isinstance(kwargs.get('fill_value', 0.0), int):
             target_torch_dtype = default_float
        elif creation_func in [torch.randint, torch.zeros, torch.ones, torch.full] and isinstance(kwargs.get('fill_value', 0), int):
             target_torch_dtype = default_int
        else: # Default fallback
             target_torch_dtype = default_float


    # Call the actual Torch creation function
    try:
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

        # Separate shape for functions that don't take it in kwargs (like eye, arange, linspace)
        shape_kwarg = kwargs.pop('shape', None) # Remove shape if it exists

        # Prepare args list dynamically for functions like eye, arange, linspace
        # This part is tricky as the helper needs to know which args are positional vs keyword
        # A simpler approach might be to NOT use this helper for eye, arange, linspace in Torch either.
        # Sticking to kwargs-based approach for now, assuming creation_func handles them.

        # Add device, dtype, requires_grad to kwargs for torch function
        kwargs['dtype'] = target_torch_dtype
        kwargs['device'] = target_device
        # Only add requires_grad if the function supports it (many creation ops don't)
        # This might need function-specific handling or inspection
        # For simplicity, we'll add it and let torch error if unsupported for a specific func
        kwargs['requires_grad'] = requires_grad

        # Special case for torch.normal which expects 'size' as a positional argument
        if creation_func == torch.normal:
             # Use the shape_kwarg that was extracted earlier
             if shape_kwarg is not None:
                 return creation_func(mean=kwargs.get('mean', 0.0), std=kwargs.get('std', 1.0), size=shape_kwarg,
                                     dtype=kwargs.get('dtype'), device=kwargs.get('device'),
                                     requires_grad=kwargs.get('requires_grad', False))
             else:
                 raise ValueError("torch.normal requires 'shape' parameter")
        # Handle functions that take shape as a positional argument
        elif shape_kwarg is not None and creation_func in [torch.zeros, torch.ones, torch.full, torch.randn, torch.rand, torch.randint]:
             return creation_func(shape_kwarg, **kwargs)
             # This code is now replaced by the earlier check
        # Special case for torch.arange which expects positional arguments
        elif creation_func == torch.arange:
             # Check which arange signature to use
             if 'start' in kwargs and 'end' in kwargs:
                 # arange(start, end, step)
                 start = kwargs.pop('start')
                 end = kwargs.pop('end')
                 step = kwargs.pop('step', 1)
                 # Extract only the supported kwargs
                 dtype = kwargs.pop('dtype', None)
                 device = kwargs.pop('device', None)
                 requires_grad = kwargs.pop('requires_grad', False)
                 # Call with positional and only supported keyword args
                 return creation_func(start, end, step, dtype=dtype, device=device, requires_grad=requires_grad)
             elif 'end' in kwargs:
                 # arange(end)
                 end = kwargs.pop('end')
                 # Extract only the supported kwargs
                 dtype = kwargs.pop('dtype', None)
                 device = kwargs.pop('device', None)
                 requires_grad = kwargs.pop('requires_grad', False)
                 # Call with positional and only supported keyword args
                 return creation_func(end, dtype=dtype, device=device, requires_grad=requires_grad)
             else:
                 # If neither start nor end is in kwargs, we can't proceed
                 raise ValueError(f"torch.arange requires either 'end' or 'start' and 'end' parameters")
        else:
            # Assume other functions take their primary args via kwargs (e.g., N, M for eye)
            # This might fail if function expects positional args not covered here.
             return creation_func(**kwargs)

    except TypeError as e:
        raise TypeError(
            f"{creation_func.__name__} failed. "
            f"Input dtype: {dtype}, Resolved native dtype: {target_torch_dtype}, "
            f"Device: {target_device}, Requires Grad: {requires_grad}, Kwargs: {kwargs}. Error: {e}"
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
    "_validate_and_get_torch_dtype",
    "_create_new_tensor", # Export the new helper
]