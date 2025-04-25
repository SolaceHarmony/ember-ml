"""PyTorch tensor utility operations."""

import torch
from typing import Union, Optional, Any, Sequence, Callable, Tuple

# Assume types are defined correctly
from ember_ml.backend.torch.types import DType, TensorLike, Shape, default_float, default_int


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
        # Map dtype names to Torch dtypes
        if dtype_name == 'float32':
            return torch.float32
        elif dtype_name == 'float64': # Map float64 to float32 for Compatibility
             return torch.float32
        elif dtype_name == 'int32':
            return torch.int32
        elif dtype_name == 'int64':
            return torch.int64
        elif dtype_name in ('bool', 'bool_'):
            return torch.bool # Use torch.bool
        elif dtype_name == 'int8':
            return torch.int8
        elif dtype_name == 'int16':
            return torch.int16
        elif dtype_name == 'uint8':
            return torch.uint8
        elif dtype_name == 'uint16':
            return torch.uint16
        elif dtype_name == 'uint32':
            return torch.uint32
        elif dtype_name == 'uint64':
            return torch.uint64
        elif dtype_name == 'float16':
            return torch.float16
        elif dtype_name == 'complex64':
             # Check if complex64 is actually supported by the installed Torch version
             if hasattr(torch, 'complex64'):
                 return torch.complex64
             else:
                 raise ValueError(f"Torch backend does not support complex64 dtype.")
        else:
            raise ValueError(f"Unknown data type name: {dtype_name}")

    # If it's not a string, EmberDType, or torch.dtype, it's invalid
    raise ValueError(f"Invalid dtype: {dtype} of type {type(dtype)}")


def _convert_input(x: TensorLike, dtype: Optional[DType] = None) -> Any:
    """
    Convert input to Torch array.

    Handles various input types:
    - Torch arrays (returned as-is)
    - NumPy arrays (converted to Torch arrays)
    - TorchTensor objects (extract underlying data)
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
        Torch array

    Raises:
        ValueError: If the input cannot be converted to an Torch tensor.
    """
    # Handle None values
    if x is None:
        # Create a default zero tensor (scalar)
        # Use the provided dtype if available, otherwise default to float
        target_torch_dtype = _validate_and_get_torch_dtype(dtype) if dtype is not None else default_float
        return torch.tensor(0.0, dtype=target_torch_dtype)
    # Already an Torch tensor - check by type and module
    # Using isinstance for potentially more robust check
    if isinstance(x, torch.Tensor):
        # If dtype is provided and different, convert. Otherwise return as is.
        # Let _convert_to_tensor handle the final conversion if needed.
        return x
    # Fallback check if isinstance fails for some reason
    elif (hasattr(x, '__class__') and
        hasattr(x.__class__, '__module__') and
        x.__class__.__module__ == 'torch' and
        x.__class__.__name__ == 'Tensor'):
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
            # The tensor within EmberTensor might be from another backend,
            # so we recursively call _convert_input to handle it properly.
            # Pass the original dtype down
            return _convert_input(x._tensor, dtype=dtype)
        else:
            raise ValueError(f"EmberTensor does not have a '_tensor' attribute: {x}")

    # Handle Parameter objects
    # Check by class name to avoid direct import which might cause circular dependencies
    if (hasattr(x, '__class__') and
        hasattr(x.__class__, '__name__') and
        x.__class__.__name__ == 'Parameter'):
        if hasattr(x, 'data'):
            # Recursively convert the underlying data
            # Pass the original dtype down
            return _convert_input(x.data, dtype=dtype)
        else:
            raise ValueError(f"Parameter object does not have a 'data' attribute: {x}")

    # Check for NumPy arrays by type name rather than direct import
    if (hasattr(x, '__class__') and
        x.__class__.__module__ == 'numpy' and
        x.__class__.__name__ == 'ndarray'):
        # Use x.copy() to avoid potential memory sharing issues
        # Let torch.from_numpy handle dtype conversion (e.g., float64 -> float32 if needed later)
        # Convert from numpy, let _convert_to_tensor handle final dtype if needed
        return torch.from_numpy(x.copy())

    # Handle NumPy scalar types using hasattr
    if (hasattr(x, 'item') and
        hasattr(x, '__class__') and
        hasattr(x.__class__, '__module__') and
        x.__class__.__module__ == 'numpy'):
        try:
            # Convert NumPy scalar to its Python equivalent, then to Torch array
            # Use the new validation function to get the appropriate Torch dtype
            py_scalar = x.item()
            # Get the corresponding torch dtype, preserving precision where possible
            # Determine target dtype: Use provided dtype first, else infer from numpy dtype
            target_torch_dtype = _validate_and_get_torch_dtype(dtype)
            if target_torch_dtype is None:
                target_torch_dtype = _validate_and_get_torch_dtype(x.dtype) # Infer from numpy

            if target_torch_dtype:
                 return torch.tensor(py_scalar, dtype=target_torch_dtype)
            else:
                 # Fallback if dtype validation fails or inference fails
                 # Let torch infer from the python type
                 return torch.tensor(py_scalar)
        except Exception as e:
             raise ValueError(f"Cannot convert NumPy scalar {type(x)} to torch.Tensor: {e}")

    # Handle Python scalars (int, float, bool), EXCLUDING NumPy scalars handled above
    is_python_scalar = isinstance(x, (int, float, bool))
    is_numpy_scalar = (hasattr(x, 'item') and hasattr(x, '__class__') and hasattr(x.__class__, '__module__') and x.__class__.__module__ == 'numpy')

    if is_python_scalar and not is_numpy_scalar:
        try:
            # Determine target dtype: Use provided dtype first, else infer from Python type
            target_torch_dtype = _validate_and_get_torch_dtype(dtype)

            if target_torch_dtype is not None:
                 return torch.tensor(x, dtype=target_torch_dtype)
            # If dtype not provided, infer from Python type
            elif isinstance(x, float):
                return torch.tensor(x, dtype=default_float) # Use default_float
            elif isinstance(x, int):
                # Use default_int (torch.int32) for consistency
                return torch.tensor(x, dtype=default_int)
            else:  # bool
                 # Revert to standard boolean tensor creation
                 return torch.tensor(x, dtype=torch.bool)
        except Exception as e:
            raise ValueError(f"Cannot convert Python scalar {type(x)} to torch.Tensor: {e}")
    # Handle Python sequences (potential 1D or higher tensors or lists of Parameters)
    # Handle lists of Parameter objects specifically
    if isinstance(x, list) and all(hasattr(item, 'data') for item in x):
        # If it's a list of objects with a 'data' attribute (assuming Parameters),
        # return a list of their underlying data (Torch tensors)
        return [item.data for item in x]

    # Handle other Python sequences (potential 1D or higher tensors) recursively
    if isinstance(x, (list, tuple)):
       try:
            # Convert items first before creating the final array
            # Convert items first, passing the original dtype down for potential scalar conversions
            converted_items = [_convert_input(item, dtype=dtype) for item in x]
            # Determine final dtype: Use provided dtype if available, else let torch infer
            target_torch_dtype = _validate_and_get_torch_dtype(dtype)
            if target_torch_dtype is not None:
                # Ensure items are tensors before creating the final tensor with specified dtype
                # This might involve an intermediate tensor creation if items aren't already tensors
                try:
                    # Attempt direct creation with dtype
                    return torch.tensor(converted_items, dtype=target_torch_dtype)
                except TypeError:
                    # If direct creation fails (e.g., mixed types not handled by torch.tensor with dtype),
                    # create an intermediate tensor first and then convert dtype.
                    intermediate_tensor = torch.tensor(converted_items)
                    return intermediate_tensor.to(dtype=target_torch_dtype)
            else:
                # Let torch infer the dtype from the list of converted items
                return torch.tensor(converted_items)
       except Exception as e:
            # Safely get item types, handling potential errors
            try:
                item_types = [type(item).__name__ for item in x[:10]]  # Limit to first 10 items for safety
                if len(x) > 10:
                    item_types.append("...")
            except Exception:
                item_types = ["<unknown>"]
            raise ValueError(f"Cannot convert sequence {type(x)} with item types {item_types} to Torch tensor: {str(e)}")


    # For any other type, reject it with a corrected list of supported types
    raise ValueError(f"Cannot convert {type(x)} to torch.Tensor. Supported types: Python scalars/sequences, NumPy scalars/arrays, TorchTensor, EmberTensor, Parameter.")

def _convert_to_tensor(data: Any, dtype: Optional[Any] = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Convert input to Torch tensor with specific dtype handling.

    Args:
        data: Input data
        dtype: Optional desired data type (string, EmberDType, Torch Dtype).
        device: GPU or cpu type

    Returns:
        Torch array
    """
    # Determine the target device *before* conversion
    target_device = device
    if target_device is None:
        from ember_ml.backend.torch.device_ops import get_device # Local import
        target_device = get_device() # Use framework's default device

    # Initial conversion using the refined _convert_input
    # _convert_input handles basic type checks and numpy/scalar conversions
    # It does NOT handle final dtype or device placement yet
    # Pass dtype to _convert_input for initial handling (scalars, sequences)
    tensor = _convert_input(data, dtype=dtype)
    current_torch_dtype = tensor.dtype

    # Validate and get the target torch dtype
    target_torch_dtype = _validate_and_get_torch_dtype(dtype) # Use the new validation function

    # Apply dtype conversion if necessary
    dtype_changed = False
    if target_torch_dtype is not None and target_torch_dtype != current_torch_dtype:
            tensor = tensor.to(dtype=target_torch_dtype)
            dtype_changed = True

    # Move to the target device if necessary
    # Check device string representation to avoid unnecessary moves
    current_device = str(tensor.device)
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
import numpy as np
def to_numpy(data: TensorLike) -> 'np.ndarray': 
    """
    Convert a PyTorch tensor to a NumPy array.

    IMPORTANT: This function is provided ONLY for visualization/plotting libraries
    that specifically require NumPy arrays. It should NOT be used for general tensor
    conversions or operations. Ember ML has a zero backend design where EmberTensor
    relies entirely on the selected backend for representation.

    Args:
        data: The tensor to convert

    Returns:
        NumPy array (empty array if input is None)
    """
    if data is None:
        return np.array([]) # Return empty array if input is None

    # Convert to tensor first to handle various inputs
    tensor_torch = _convert_to_tensor(data)
    
    # Check if the tensor is an integer type
    is_integer_dtype = tensor_torch.dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8]
    
    # Ensure tensor is on CPU before converting to NumPy
    numpy_array = tensor_torch.detach().cpu().numpy()
    
    # If the original tensor was an integer type, ensure the NumPy array is also an integer type
    if is_integer_dtype and numpy_array.dtype.kind == 'f':
        # Convert to the corresponding NumPy integer type
        if tensor_torch.dtype == torch.int8:
            return numpy_array.astype(np.int8)
        elif tensor_torch.dtype == torch.int16:
            return numpy_array.astype(np.int16)
        elif tensor_torch.dtype == torch.int32:
            return numpy_array.astype(np.int32)
        elif tensor_torch.dtype == torch.int64:
            return numpy_array.astype(np.int64)
        elif tensor_torch.dtype == torch.uint8:
            return numpy_array.astype(np.uint8)
    
    return numpy_array


def item(data: TensorLike) -> Union[int, float, bool]:
    """
    Extract the scalar value from a tensor.

    Args:
        data: Input tensor containing a single element

    Returns:
        Standard Python scalar (int, float, or bool)
    """
    tensor_torch = _convert_to_tensor(data)

    # Check if the tensor is a scalar (0D tensor)
    if tensor_torch.ndim != 0:
         raise ValueError("item() can only be called on scalar tensors (0D tensors)")
  
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
    # No need for TorchTensor instance, just convert and get shape
    return _convert_input(data).shape

def dtype(data: TensorLike) -> Optional[str]: # Return type can be Optional[str]
    """
    Get the string representation of a tensor's data type.

    Args:
        data: Input array

    Returns:
        String representation of the array's data type (e.g., 'float32', 'int64').
    """
    from ember_ml.backend.torch.tensor.dtype import TorchDType # For converting native to string

    # Convert input data to a native Torch tensor
    tensor_torch = _convert_input(data)

    # Get the native Torch dtype from the tensor
    native_dtype = tensor_torch.dtype

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
    tensor_array = _convert_input(data)
    return tensor_array.clone()

def var(data: TensorLike, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False, ddof: int = 0) -> torch.Tensor:
    """
    Compute the variance of a tensor along specified axes.

    Args:
        data: Input tensor
        axis: Axis or axes along which to compute the variance
        keepdims: Whether to keep the reduced dimensions
        ddof: Delta degrees of freedom. The divisor used in calculations is N - ddof,
              where N represents the number of elements. Default: 0.

    Returns:
        Variance of the tensor
    """
    tensor_torch = _convert_input(data)

    # Determine axis_seq once at the beginning
    axis_seq: Optional[Tuple[int, ...]] = None
    if isinstance(axis, int):
        axis_seq = (axis,)
    elif isinstance(axis, Sequence):
        axis_seq = tuple(axis)
    # If axis is None, axis_seq remains None

    # PyTorch var uses 'dim' and 'keepdim', and 'unbiased' (True means ddof=1, False means ddof=0)
    # We need to handle ddof manually if it's not 0 or 1
    if ddof == 1:
         unbiased = True
         return torch.var(tensor_torch, dim=axis_seq, keepdim=keepdims, unbiased=unbiased)
    elif ddof == 0:
         unbiased = False
         return torch.var(tensor_torch, dim=axis_seq, keepdim=keepdims, unbiased=unbiased)
    else:
         # Manual calculation for other ddof values
         if axis_seq is None:
              numel = tensor_torch.numel()
              mean = torch.mean(tensor_torch)
              # Use torch ops directly in backend code
              squared_diff = torch.square(torch.subtract(tensor_torch, mean))
              # Use Python subtraction for scalar divisor
              variance = torch.divide(torch.sum(squared_diff), (numel - ddof))
              if keepdims:
                   # Find original ndim and create appropriate shape
                   orig_ndim = tensor_torch.ndim
                   new_shape = (1,) * orig_ndim
                   variance = variance.view(new_shape)
              return variance
         else:
              # Ensure axis is a tuple for multi-axis reduction consistency
              # This check is now redundant due to the initial axis_seq determination
              # if isinstance(axis, int):
              #     axis_seq = (axis,)
              # elif isinstance(axis, Sequence):
              #     axis_seq = tuple(axis)
              # else:
              #      # This case should be covered by the outer if axis is None, but for type safety:
              #      raise ValueError("Axis must be an integer, a sequence of integers, or None")

              numel_axis: float = torch.prod(torch.tensor([tensor_torch.shape[i] for i in axis_seq])).item() # Renamed variable
              mean = torch.mean(tensor_torch, dim=axis_seq, keepdim=True)
              # Use torch ops directly in backend code
              squared_diff = torch.square(torch.subtract(tensor_torch, mean))
              # Correct N calculation for axis reduction:
              dims_to_reduce = set(axis_seq)
              n = 1
              for i, dim_size in enumerate(tensor_torch.shape):
                   if i in dims_to_reduce:
                        n *= dim_size
              # Use torch ops directly in backend code
              # Use Python subtraction for scalar divisor
              corrected_variance = torch.divide(torch.sum(squared_diff, dim=axis_seq, keepdim=keepdims), (numel_axis - ddof)) # Use Python subtraction and renamed variable
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
    tensor_torch = _convert_input(data)

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
    tensor_torch = _convert_input(data)

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
    x_torch = _convert_input(data1)
    y_torch = _convert_input(data2)
    return torch.maximum(x_torch, y_torch)


def _create_new_tensor(creation_func: Callable, dtype: Optional[Any] = None, device: Optional[str] = None, requires_grad: bool = False, **kwargs) -> torch.Tensor:
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
    # Resolve dtype first
    target_torch_dtype = _validate_and_get_torch_dtype(dtype)
    # Resolve device first
    target_device = device
    if target_device is None:
        from ember_ml.backend.torch.device_ops import get_device
        target_device = get_device()



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
    # Special case for torch.arange which expects positional arguments
    elif creation_func == torch.arange:
        print("torch.arange called with kwargs:", kwargs)
        # Check if 'start' and 'end' are provided in kwargs
        # Extract arange-specific arguments
        start = kwargs.pop('start', None)
        end = kwargs.pop('end', None)
        step = kwargs.pop('step', 1)

        # Build positional arguments list based on provided args
        pos_args = []
        if start is not None and end is not None:
            pos_args.append(start)
            pos_args.append(end)
            if step != 1:
                pos_args.append(step)
        elif end is not None:
            pos_args.append(end)
        elif start is not None:
            # If only start is provided, treat it as the end and set start to 0
            pos_args.append(start)
        else:
            raise ValueError("torch.arange requires 'start' and/or 'end' parameters")
        
        # Extract common arguments (already handled before this elif block)
        # Ensure only allowed keyword arguments are passed to torch.arange
        allowed_kwargs = {}
        if 'dtype' in kwargs:
            allowed_kwargs['dtype'] = kwargs.pop('dtype')
        if 'device' in kwargs:
            allowed_kwargs['device'] = kwargs.pop('device')
        if 'requires_grad' in kwargs:
            allowed_kwargs['requires_grad'] = kwargs.pop('requires_grad')
        # Note: 'out' and 'layout' are also keyword-only for torch.arange,
        # but are not currently handled by _create_new_tensor.
        # If they were, they would be extracted here as well.

        # Call torch.arange with positional and only supported keyword args
        return creation_func(*pos_args, **allowed_kwargs)
    else:
        # Assume other functions take their primary args via kwargs (e.g., N, M for eye)
        # This might fail if function expects positional args not covered here.
            return creation_func(**kwargs)
        

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