import pytest
from ember_ml import ops
from ember_ml.nn import tensor
# Removed numpy import to comply with emberlint rules

# Assume conftest.py provides 'backend' fixture

@pytest.fixture(params=[tensor.float32, tensor.int32, tensor.bool_])
def sample_dtype(request):
    """Fixture to provide different data types."""
    return request.param

def test_tensor_creation_array(backend):
    """Tests tensor.array creation."""
    ops.set_backend(backend)
    data = [[1, 2], [3, 4]]
    t = tensor.array(data)
    assert isinstance(t, tensor.EmberTensor), "Did not create EmberTensor"
    assert tensor.shape(t) == (2, 2), "Shape mismatch"
    # Default dtype might vary, check content roughly
    assert ops.allclose(t, tensor.convert_to_tensor(data)), "Content mismatch"

def test_tensor_creation_convert_to_tensor(backend):
    """Tests tensor.convert_to_tensor."""
    ops.set_backend(backend)
    data_list = [[1.0, 2.0], [3.0, 4.0]]
    # data_numpy removed

    # From list
    t_list = tensor.convert_to_tensor(data_list)
    assert isinstance(t_list, tensor.EmberTensor), "Convert list failed"
    assert tensor.shape(t_list) == (2, 2), "Convert list shape failed"
    assert ops.allclose(t_list, tensor.convert_to_tensor(data_list)), "Convert list content failed"

    # Test for converting numpy array removed due to emberlint rule violation
    # From existing EmberTensor (should return same object or equivalent)
    # Test conversion from existing tensor (using the one created from list)
    t_existing = tensor.convert_to_tensor(t_list)
    assert isinstance(t_existing, tensor.EmberTensor), "Convert existing failed"
    assert ops.allclose(t_existing, tensor.convert_to_tensor(data_list)), "Convert existing content failed"
    # Could potentially check object identity if required, but content/type check is usually sufficient

def test_tensor_creation_zeros_ones(backend):
    """Tests tensor.zeros and tensor.ones."""
    ops.set_backend(backend)
    shape = (2, 3)
    
    t_zeros = tensor.zeros(shape)
    assert isinstance(t_zeros, tensor.EmberTensor), "zeros failed type"
    assert tensor.shape(t_zeros) == shape, "zeros shape failed"
    assert ops.all(ops.equal(t_zeros, tensor.convert_to_tensor(0.0))), "zeros content failed"

    t_ones = tensor.ones(shape)
    assert isinstance(t_ones, tensor.EmberTensor), "ones failed type"
    assert tensor.shape(t_ones) == shape, "ones shape failed"
    assert ops.all(ops.equal(t_ones, tensor.convert_to_tensor(1.0))), "ones content failed"

def test_tensor_creation_eye(backend):
    """Tests tensor.eye."""
    ops.set_backend(backend)
    n = 3
    t_eye = tensor.eye(n)
    expected_list = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] # n=3
    assert isinstance(t_eye, tensor.EmberTensor), "eye failed type"
    assert tensor.shape(t_eye) == (n, n), "eye shape failed"
    assert ops.allclose(t_eye, tensor.convert_to_tensor(expected_list)), "eye content failed"

    # Rectangular eye
    t_eye_rect = tensor.eye(n, m=4)
    expected_rect_list = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]] # n=3, m=4
    assert tensor.shape(t_eye_rect) == (n, 4), "eye rect shape failed"
    assert ops.allclose(t_eye_rect, tensor.convert_to_tensor(expected_rect_list)), "eye rect content failed"

def test_tensor_creation_arange_linspace(backend):
    """Tests tensor.arange and tensor.linspace."""
    ops.set_backend(backend)

    # arange
    t_arange = tensor.arange(0, 5, 1) # 0, 1, 2, 3, 4
    expected_arange_list = [0, 1, 2, 3, 4]
    # Ensure dtype match for comparison, as arange might produce int/float depending on backend/args
    expected_tensor = tensor.convert_to_tensor(expected_arange_list, dtype=t_arange.dtype)
    assert ops.allclose(t_arange, expected_tensor), "arange failed"

    # linspace
    t_linspace = tensor.linspace(0.0, 1.0, 5) # 0.0, 0.25, 0.5, 0.75, 1.0
    expected_linspace_list = [0.0, 0.25, 0.5, 0.75, 1.0]
    assert ops.allclose(t_linspace, tensor.convert_to_tensor(expected_linspace_list)), "linspace failed"

def test_tensor_creation_like(backend):
    """Tests tensor.zeros_like, tensor.ones_like, tensor.full_like."""
    ops.set_backend(backend)
    shape = (2, 4)
    t_ref = tensor.zeros(shape) # Reference tensor

    # zeros_like
    t_zeros_like = tensor.zeros_like(t_ref)
    assert tensor.shape(t_zeros_like) == shape, "zeros_like shape failed"
    assert ops.all(ops.equal(t_zeros_like, tensor.convert_to_tensor(0.0))), "zeros_like content failed"

    # ones_like
    t_ones_like = tensor.ones_like(t_ref)
    assert tensor.shape(t_ones_like) == shape, "ones_like shape failed"
    assert ops.all(ops.equal(t_ones_like, tensor.convert_to_tensor(1.0))), "ones_like content failed"

    # full_like
    fill_val = 7.0
    t_full_like = tensor.full_like(t_ref, fill_val)
    assert tensor.shape(t_full_like) == shape, "full_like shape failed"
    assert ops.all(ops.equal(t_full_like, tensor.convert_to_tensor(fill_val))), "full_like content failed"

def test_tensor_creation_full(backend):
    """Tests tensor.full."""
    ops.set_backend(backend)
    shape = (3, 2)
    fill_val = 5.5
    t_full = tensor.full(shape, fill_val)
    assert tensor.shape(t_full) == shape, "full shape failed"
    assert ops.all(ops.equal(t_full, tensor.convert_to_tensor(fill_val))), "full content failed"

def test_tensor_properties_shape_dtype(backend, sample_dtype):
    """Tests tensor shape and dtype properties."""
    ops.set_backend(backend)
    shape = (2, 3, 4)
    
    # Create tensor with specific dtype
    # Note: Creating bool tensor might require specific values for some backends
    if sample_dtype == tensor.bool_:
        t = tensor.zeros(shape, dtype=sample_dtype) # Use zeros for bool
    else:
        t = tensor.ones(shape, dtype=sample_dtype)
        
    assert tensor.shape(t) == shape, f"Shape property failed for dtype {sample_dtype}"
    retrieved_dtype = t.dtype
    assert retrieved_dtype == sample_dtype, f"Dtype property mismatch: expected {sample_dtype}, got {retrieved_dtype}"

# TODO: Add tests for tensor manipulation (reshape, transpose, concatenate, stack, split, expand_dims, squeeze, tile, gather, scatter, slice, pad)
# TODO: Add tests for type conversion (cast, to_numpy, item)
# TODO: Add tests for random operations (random_uniform, normal, bernoulli, etc.)
# TODO: Add tests for dtype operations (get_dtype, to_dtype_str, from_dtype_str)