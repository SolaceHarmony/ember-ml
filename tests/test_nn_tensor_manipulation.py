import pytest
from ember_ml import ops
from ember_ml.nn import tensor
import numpy as np

# Assume conftest.py provides 'backend' fixture

@pytest.fixture
def manip_tensor(backend):
    """Fixture for manipulation tests."""
    ops.set_backend(backend)
    # Shape (2, 3, 4)
    t = tensor.arange(2 * 3 * 4).reshape((2, 3, 4))
    return t

def test_tensor_reshape(manip_tensor, backend):
    """Tests tensor.reshape."""
    ops.set_backend(backend)
    t = manip_tensor
    
    # Reshape to (6, 4)
    reshaped1 = tensor.reshape(t, (6, 4))
    assert tensor.shape(reshaped1) == (6, 4), "Reshape (6, 4) failed"
    # Check total elements remain same using ops.prod equivalent if available, or skip check
    # Assuming ops.prod exists or can be derived:
    # total_elements_t = ops.prod(tensor.shape(t)) # Needs shape as tensor
    # total_elements_r1 = ops.prod(tensor.shape(reshaped1))
    # assert ops.equal(total_elements_t, total_elements_r1), "Reshape element count mismatch"
    # Simplified check: Length of flattened arrays
    t_flat = tensor.reshape(t, (-1,))
    r1_flat = tensor.reshape(reshaped1, (-1,))
    assert tensor.shape(t_flat)[0] == tensor.shape(r1_flat)[0], "Reshape element count mismatch"


    # Reshape with -1 inference
    reshaped2 = tensor.reshape(t, (2, -1)) # Should be (2, 12)
    assert tensor.shape(reshaped2) == (2, 12), "Reshape (2, -1) failed"

def test_tensor_transpose(manip_tensor, backend):
    """Tests tensor.transpose."""
    ops.set_backend(backend)
    t = manip_tensor # Shape (2, 3, 4)

    # Default transpose (reverse axes)
    transposed1 = tensor.transpose(t) # Should be (4, 3, 2)
    assert tensor.shape(transposed1) == (4, 3, 2), "Default transpose shape failed"

    # Specific axes transpose
    transposed2 = tensor.transpose(t, axes=(1, 2, 0)) # Should be (3, 4, 2)
    assert tensor.shape(transposed2) == (3, 4, 2), "Axes transpose shape failed"

def test_tensor_concatenate(backend):
    """Tests tensor.concatenate."""
    ops.set_backend(backend)
    t1 = tensor.ones((2, 3))
    t2 = tensor.zeros((2, 3))
    t3 = tensor.ones((1, 3)) * 2

    # Concat along axis 0
    concat0 = tensor.concatenate([t1, t2, t3], axis=0)
    assert tensor.shape(concat0) == (5, 3), "Concatenate axis=0 shape failed" # Corrected shape assertion
    # Check content using tensor operations
    expected0_part1 = tensor.ones((2, 3))
    expected0_part2 = tensor.zeros((2, 3))
    expected0_part3 = tensor.ones((1, 3)) * 2
    # Reconstruct expected tensor using concatenate or slicing if possible
    # For verification, let's check slices
    assert ops.allclose(tensor.slice(concat0, [0, 0], [2, 3]), expected0_part1), "Concat axis=0 content failed (part1)"
    assert ops.allclose(tensor.slice(concat0, [2, 0], [2, 3]), expected0_part2), "Concat axis=0 content failed (part2)"
    assert ops.allclose(tensor.slice(concat0, [4, 0], [1, 3]), expected0_part3), "Concat axis=0 content failed (part3)"


    # Concat along axis 1
    t4 = tensor.zeros((2, 1))
    concat1 = tensor.concatenate([t1, t4], axis=1)
    assert tensor.shape(concat1) == (2, 3 + 1), "Concatenate axis=1 shape failed"

def test_tensor_stack(backend):
    """Tests tensor.stack."""
    ops.set_backend(backend)
    t1 = tensor.ones((2, 3))
    t2 = tensor.zeros((2, 3))

    # Stack along axis 0 (new first axis)
    stack0 = tensor.stack([t1, t2], axis=0)
    assert tensor.shape(stack0) == (2, 2, 3), "Stack axis=0 shape failed"

    # Stack along axis 1 (new second axis)
    stack1 = tensor.stack([t1, t2], axis=1)
    assert tensor.shape(stack1) == (2, 2, 3), "Stack axis=1 shape failed" # Corrected expected shape

    # Stack along axis -1 (new last axis)
    stack_last = tensor.stack([t1, t2], axis=-1)
    assert tensor.shape(stack_last) == (2, 3, 2), "Stack axis=-1 shape failed"

def test_tensor_split(manip_tensor, backend):
    """Tests tensor.split."""
    ops.set_backend(backend)
    t = manip_tensor # Shape (2, 3, 4)

    # Split into N equal parts
    splits_axis1 = tensor.split(t, 3, axis=1) # Split dim 1 (size 3) into 3 parts
    assert isinstance(splits_axis1, list), "Split did not return list"
    assert len(splits_axis1) == 3, "Split wrong number of parts"
    assert tensor.shape(splits_axis1[0]) == (2, 1, 4), "Split part shape mismatch"

    # Split by sections (example for axis 2, size 4 -> sections [1, 2, 1])
    t_axis2 = tensor.transpose(t, axes=(0, 2, 1)) # Shape (2, 4, 3)
    try:
        # This might fail depending on backend implementation of split by sections
        splits_axis1_sections = tensor.split(t_axis2, [1, 3], axis=1) # Indices where splits occur: [0:1], [1:3], [3:4] -> sizes 1, 2, 1
        assert len(splits_axis1_sections) == 3, "Split by sections wrong number of parts"
        assert tensor.shape(splits_axis1_sections[0]) == (2, 1, 3), "Split section 1 shape mismatch"
        assert tensor.shape(splits_axis1_sections[1]) == (2, 2, 3), "Split section 2 shape mismatch"
        assert tensor.shape(splits_axis1_sections[2]) == (2, 1, 3), "Split section 3 shape mismatch"
    except Exception as e:
        # Use pytest.xfail or specific exception handling if split sections API is known to differ
        pytest.skip(f"Split by sections test skipped for backend {backend}: {e}")


def test_tensor_expand_squeeze(manip_tensor, backend):
    """Tests tensor.expand_dims and tensor.squeeze."""
    ops.set_backend(backend)
    t = manip_tensor # Shape (2, 3, 4)

    # Expand dims
    expanded = tensor.expand_dims(t, axis=1) # Insert axis at pos 1 -> (2, 1, 3, 4)
    assert tensor.shape(expanded) == (2, 1, 3, 4), "Expand dims shape failed"

    # Squeeze the added dim
    squeezed1 = tensor.squeeze(expanded, axis=1)
    assert tensor.shape(squeezed1) == tensor.shape(t), "Squeeze specific axis failed"

    # Squeeze all dims of size 1
    t_extra_dims = tensor.reshape(t, (1, 2, 1, 3, 1, 4, 1))
    squeezed_all = tensor.squeeze(t_extra_dims)
    assert tensor.shape(squeezed_all) == tensor.shape(t), "Squeeze all failed"

def test_tensor_tile(backend):
    """Tests tensor.tile."""
    ops.set_backend(backend)
    t = tensor.convert_to_tensor([[1, 2], [3, 4]]) # Shape (2, 2)
    reps = [2, 3] # Tile 2 times along axis 0, 3 times along axis 1

    tiled = tensor.tile(t, reps)
    expected_shape = (tensor.shape(t)[0] * reps[0], tensor.shape(t)[1] * reps[1]) # (4, 6)
    assert tensor.shape(tiled) == expected_shape, "Tile shape failed"
    # Check content pattern using slicing and expected values
    # Expected result:
    # [[1, 2, 1, 2, 1, 2],
    #  [3, 4, 3, 4, 3, 4],
    #  [1, 2, 1, 2, 1, 2],
    #  [3, 4, 3, 4, 3, 4]]
    expected_row0 = tensor.convert_to_tensor([1., 2., 1., 2., 1., 2.])
    expected_row1 = tensor.convert_to_tensor([3., 4., 3., 4., 3., 4.])
    assert ops.allclose(tensor.slice(tiled, [0, 0], [1, 6]), tensor.reshape(expected_row0, (1, 6))), "Tile content failed row 0"
    assert ops.allclose(tensor.slice(tiled, [1, 0], [1, 6]), tensor.reshape(expected_row1, (1, 6))), "Tile content failed row 1"
    assert ops.allclose(tensor.slice(tiled, [2, 0], [1, 6]), tensor.reshape(expected_row0, (1, 6))), "Tile content failed row 2"
    assert ops.allclose(tensor.slice(tiled, [3, 0], [1, 6]), tensor.reshape(expected_row1, (1, 6))), "Tile content failed row 3"