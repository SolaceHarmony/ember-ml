import pytest
from ember_ml import ops
from ember_ml.nn import tensor
# No NumPy import needed here

# Assume conftest.py provides 'backend' fixture

@pytest.fixture
def indexing_tensor(backend):
    """Fixture for indexing tests."""
    ops.set_backend(backend)
    # Shape (3, 4)
    t = tensor.arange(12).reshape((3, 4))
    return t

def test_tensor_gather(indexing_tensor, backend):
    """Tests tensor.gather."""
    ops.set_backend(backend)
    params = indexing_tensor # [[0,1,2,3], [4,5,6,7], [8,9,10,11]]
    indices_axis0 = tensor.convert_to_tensor([0, 2])
    indices_axis1 = tensor.convert_to_tensor([1, 3])

    # Gather rows (axis=0)
    result_axis0 = tensor.gather(params, indices_axis0, axis=0)
    expected_axis0 = tensor.convert_to_tensor([[0, 1, 2, 3], [8, 9, 10, 11]])
    assert tensor.shape(result_axis0) == (2, 4), "Gather axis=0 shape failed"
    assert ops.allclose(result_axis0, expected_axis0), f"{backend}: Gather axis=0 failed"

    # Gather columns (axis=1)
    result_axis1 = tensor.gather(params, indices_axis1, axis=1)
    expected_axis1 = tensor.convert_to_tensor([[1, 3], [5, 7], [9, 11]])
    assert tensor.shape(result_axis1) == (3, 2), "Gather axis=1 shape failed"
    assert ops.allclose(result_axis1, expected_axis1), f"{backend}: Gather axis=1 failed"

def test_tensor_scatter(backend):
    """Tests tensor.scatter."""
    # Creates a new tensor, scattering updates into zeros
    ops.set_backend(backend)
    indices = tensor.convert_to_tensor([[0, 1], [2, 2]]) # Indices for a (3, 4) tensor
    updates = tensor.convert_to_tensor([100, 200])
    shape = (3, 4)
    
    scattered = tensor.scatter(indices, updates, shape)
    
    assert tensor.shape(scattered) == shape, "tensor.scatter shape failed"
    # Construct expected tensor using zeros and updates
    expected_manual = tensor.convert_to_tensor([[0, 100, 0, 0],
                                                [0, 0,   0, 0],
                                                [0, 0, 200, 0]])
    expected_manual = tensor.cast(expected_manual, tensor.dtype(scattered)) # Match dtype
    assert ops.allclose(scattered, expected_manual), f"{backend}: Scatter content check failed"

def test_tensor_tensor_scatter_nd_update(indexing_tensor, backend):
    """Tests tensor.tensor_scatter_nd_update."""
    # Updates an existing tensor
    ops.set_backend(backend)
    # Ensure we have a mutable copy if the operation modifies in-place
    # or if we want to check against original later.
    t_to_update = tensor.copy(indexing_tensor) # [[0,1,2,3], [4,5,6,7], [8,9,10,11]]
    indices = tensor.convert_to_tensor([[0, 0], [1, 2]]) # Indices to update
    updates = tensor.convert_to_tensor([-50, -60])      # New values
    
    updated_tensor = tensor.tensor_scatter_nd_update(t_to_update, indices, updates)
    
    assert tensor.shape(updated_tensor) == tensor.shape(t_to_update), "tensor_scatter_nd_update shape failed"
    # Construct expected tensor
    expected = tensor.convert_to_tensor([[-50, 1,  2,  3],
                                         [  4, 5, -60, 7],
                                         [  8, 9, 10, 11]])
    expected = tensor.cast(expected, tensor.dtype(updated_tensor)) # Match dtype
    assert ops.allclose(updated_tensor, expected), f"{backend}: tensor_scatter_nd_update content check failed"
    # Verify original tensor (t_to_update) is not modified if the op should return a new tensor
    # This assertion depends on the expected behavior (in-place vs. new tensor)
    # assert not ops.allclose(t_to_update, updated_tensor), "Original tensor was modified by tensor_scatter_nd_update"

def test_tensor_slice(indexing_tensor, backend):
    """Tests tensor.slice."""
    ops.set_backend(backend)
    params = indexing_tensor # [[0,1,2,3], [4,5,6,7], [8,9,10,11]]

    # Extract slice params[1:3, 0:2] (rows 1 and 2, columns 0 and 1)
    # begin=[1, 0], size=[2, 2]
    sliced = tensor.slice(params, begin=[1, 0], size=[2, 2])
    expected_slice = tensor.convert_to_tensor([[4, 5], [8, 9]])
    
    assert tensor.shape(sliced) == (2, 2), "tensor.slice shape failed"
    expected_slice = tensor.cast(expected_slice, tensor.dtype(sliced)) # Match dtype
    assert ops.allclose(sliced, expected_slice), f"{backend}: tensor.slice content failed"

def test_tensor_pad(indexing_tensor, backend):
    """Tests tensor.pad."""
    ops.set_backend(backend)
    t = indexing_tensor # Shape (3, 4)
    paddings = [[1, 2], [0, 1]] # Pad axis 0 (1 top, 2 bottom), axis 1 (0 left, 1 right)
    constant_values = 99
    
    padded = tensor.pad(t, paddings, mode='constant', constant_values=constant_values)
    
    expected_pad_shape = (3 + 1 + 2, 4 + 0 + 1) # (6, 5)
    assert tensor.shape(padded) == expected_pad_shape, "tensor.pad shape failed"
    
    # Construct expected tensor manually for verification
    expected_manual = tensor.convert_to_tensor([
        [99, 99, 99, 99, 99],
        [ 0,  1,  2,  3, 99],
        [ 4,  5,  6,  7, 99],
        [ 8,  9, 10, 11, 99],
        [99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99]
    ])
    expected_manual = tensor.cast(expected_manual, tensor.dtype(padded))

    assert ops.allclose(padded, expected_manual), f"{backend}: tensor.pad content check failed"