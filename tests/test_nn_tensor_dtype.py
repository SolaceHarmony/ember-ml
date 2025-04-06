import pytest
from ember_ml import ops
from ember_ml.nn import tensor

# Assume conftest.py provides 'backend' fixture

# List of expected dtype objects available directly in tensor module
EXPECTED_DTYPES = [
    tensor.float16, tensor.float32, tensor.float64,
    tensor.int8, tensor.int16, tensor.int32, tensor.int64,
    tensor.uint8, # uint types might not be supported by all backends (e.g., PyTorch historically)
    # tensor.uint16, tensor.uint32, tensor.uint64, # Comment out if not universally supported
    tensor.bool_
]
# Filter based on actual availability if needed, or mark tests appropriately
AVAILABLE_DTYPES = [dt for dt in EXPECTED_DTYPES if dt is not None] # Simple check if None signifies unavailability

@pytest.mark.parametrize("expected_dtype", AVAILABLE_DTYPES)
def test_dtype_objects_exist(expected_dtype):
    """Tests that the standard dtype objects exist and have the correct type."""
    assert isinstance(expected_dtype, tensor.EmberDType), f"{expected_dtype} is not an EmberDType instance"

@pytest.mark.parametrize("dtype_to_test", AVAILABLE_DTYPES)
def test_get_dtype(backend, dtype_to_test):
    """Tests tensor.get_dtype for various types."""
    ops.set_backend(backend)
    try:
        # Create a small tensor of the specified type
        # Use zeros for bool, 1 otherwise to avoid issues with bool(0)=False
        if dtype_to_test == tensor.bool_:
             t = tensor.zeros(1, dtype=dtype_to_test)
        else:
             t = tensor.ones(1, dtype=dtype_to_test)
        
        retrieved_dtype = tensor.get_dtype(t)
        assert retrieved_dtype == dtype_to_test, f"get_dtype failed for {dtype_to_test}"
    except Exception as e:
        # Some backends might not support all dtypes (e.g., uint, float16)
        pytest.skip(f"Skipping dtype {dtype_to_test} for backend {backend} due to error: {e}")


@pytest.mark.parametrize("dtype_obj", AVAILABLE_DTYPES)
def test_dtype_str_conversion(backend, dtype_obj):
    """Tests tensor.to_dtype_str and tensor.from_dtype_str."""
    ops.set_backend(backend)
    
    # Convert dtype object to string
    dtype_str = tensor.to_dtype_str(dtype_obj)
    assert isinstance(dtype_str, str), f"to_dtype_str did not return string for {dtype_obj}"
    
    # Convert string back to dtype object
    retrieved_dtype_obj = tensor.from_dtype_str(dtype_str)
    assert isinstance(retrieved_dtype_obj, tensor.EmberDType), f"from_dtype_str did not return EmberDType for '{dtype_str}'"
    
    # Check if the conversion round trip works
    assert retrieved_dtype_obj == dtype_obj, f"Dtype string conversion round trip failed for {dtype_obj} ('{dtype_str}')"

def test_from_dtype_str_invalid():
    """Tests tensor.from_dtype_str with an invalid string."""
    with pytest.raises(ValueError):
        tensor.from_dtype_str("invalid_dtype_string")

def test_dtype_equality(backend):
    """Tests equality comparison between dtype objects."""
    ops.set_backend(backend)
    assert tensor.float32 == tensor.float32, "Dtype equality failed (self)"
    assert tensor.float32 != tensor.int32, "Dtype inequality failed"
    
    # Test equality after string conversion
    f32_str = tensor.to_dtype_str(tensor.float32)
    f32_from_str = tensor.from_dtype_str(f32_str)
    assert tensor.float32 == f32_from_str, "Dtype equality failed after string conversion"

# TODO: Test tensor.dtype function alias if needed (it's often same as get_dtype)
# TODO: Test DType class directly if necessary
# TODO: Add checks for cross-backend dtype compatibility if needed