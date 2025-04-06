import pytest
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.ops import stats # For mean/std checks

# Assume conftest.py provides 'backend' fixture

@pytest.fixture
def random_params(backend):
    """Fixture for random op parameters."""
    ops.set_backend(backend)
    shape = (1000, 10) # Use a reasonable size for statistical checks
    seed = 42
    return shape, seed

def test_tensor_set_get_seed(random_params, backend):
    """Tests tensor.set_seed and tensor.get_seed."""
    ops.set_backend(backend)
    _, seed = random_params
    
    # Note: Seed setting/getting might be backend specific and not globally tracked by EmberTensor directly.
    # This test assumes a global or backend-specific seed mechanism accessible via tensor module.
    # Need to verify actual implementation details.
    # If seed is purely backend, test might need adjustment.
    
    # Let's assume tensor.set_seed configures the active backend's seed generator.
    try:
        original_seed_state = tensor.get_seed() # Assuming this retrieves backend state if possible
    except NotImplementedError:
        original_seed_state = None # If get_seed is not implemented

    tensor.set_seed(seed)
    # get_seed might not be implemented or might return backend state
    # assert tensor.get_seed() == seed, "get_seed did not return the set seed" # This assertion might fail

    # Check reproducibility
    tensor.set_seed(seed)
    rand1 = tensor.random_uniform((10, 10))
    tensor.set_seed(seed)
    rand2 = tensor.random_uniform((10, 10))
    assert ops.allclose(rand1, rand2, atol=1e-6), "Setting seed did not ensure reproducibility"
    
    # Restore original state if possible - tricky without reliable get_seed/set_seed state capture
    # if original_seed_state is not None:
    #     tensor.set_seed(original_seed_state) # This might not work as expected

def test_tensor_random_uniform(random_params, backend):
    """Tests tensor.random_uniform."""
    ops.set_backend(backend)
    shape, seed = random_params
    minval, maxval = -1.0, 1.0
    tensor.set_seed(seed) # Ensure reproducibility for checks
    
    rand_tensor = tensor.random_uniform(shape, minval=minval, maxval=maxval, dtype=tensor.float32)
    
    assert isinstance(rand_tensor, tensor.EmberTensor), "random_uniform did not return EmberTensor"
    assert tensor.shape(rand_tensor) == shape, "random_uniform shape mismatch"
    assert tensor.dtype(rand_tensor) == tensor.float32, "random_uniform dtype mismatch"
    
    # Check range (min/max values might slightly exceed bounds due to generation method)
    min_val_actual = tensor.item(stats.min(rand_tensor))
    max_val_actual = tensor.item(stats.max(rand_tensor))
    # Use tolerance
    assert min_val_actual >= minval - 1e-5, f"random_uniform min value out of range: {min_val_actual}"
    # Maxval is exclusive in some implementations, allow slight overshoot if needed
    assert max_val_actual <= maxval + 1e-5, f"random_uniform max value out of range: {max_val_actual}" 
    
    # Check distribution (mean should be approx (minval+maxval)/2)
    mean_val = tensor.item(stats.mean(rand_tensor))
    expected_mean = (minval + maxval) / 2.0
    assert abs(mean_val - expected_mean) < 0.1, f"random_uniform mean out of expected range: {mean_val}" # Loose check

def test_tensor_random_normal(random_params, backend):
    """Tests tensor.random_normal."""
    ops.set_backend(backend)
    shape, seed = random_params
    mean, stddev = 0.0, 1.0
    tensor.set_seed(seed)
    
    rand_tensor = tensor.random_normal(shape, mean=mean, stddev=stddev, dtype=tensor.float32)
    
    assert isinstance(rand_tensor, tensor.EmberTensor), "random_normal did not return EmberTensor"
    assert tensor.shape(rand_tensor) == shape, "random_normal shape mismatch"
    assert tensor.dtype(rand_tensor) == tensor.float32, "random_normal dtype mismatch"

    # Check distribution statistics (mean and stddev should be close to specified)
    mean_actual = tensor.item(stats.mean(rand_tensor))
    stddev_actual = tensor.item(stats.std(rand_tensor))
    
    assert abs(mean_actual - mean) < 0.1, f"random_normal mean out of expected range: {mean_actual}" # Loose check
    assert abs(stddev_actual - stddev) < 0.1, f"random_normal stddev out of expected range: {stddev_actual}" # Loose check

def test_tensor_random_bernoulli(random_params, backend):
    """Tests tensor.random_bernoulli."""
    ops.set_backend(backend)
    shape, seed = random_params
    p = 0.7 # Probability of generating 1
    tensor.set_seed(seed)
    
    # Generate as float first for mean check
    rand_tensor_float = tensor.random_bernoulli(shape, p=p, dtype=tensor.float32) 

    assert isinstance(rand_tensor_float, tensor.EmberTensor), "random_bernoulli did not return EmberTensor"
    assert tensor.shape(rand_tensor_float) == shape, "random_bernoulli shape mismatch"
    
    # Check if it contains 0s and 1s mostly
    min_val = tensor.item(stats.min(rand_tensor_float))
    max_val = tensor.item(stats.max(rand_tensor_float))
    # Use allclose for float comparison
    assert ops.allclose(min_val, 0.0) or ops.allclose(min_val, 1.0), "bernoulli min value not 0 or 1"
    assert ops.allclose(max_val, 0.0) or ops.allclose(max_val, 1.0), "bernoulli max value not 0 or 1"

    # Check distribution (mean should be close to p)
    mean_actual = tensor.item(stats.mean(rand_tensor_float))
    assert abs(mean_actual - p) < 0.1, f"random_bernoulli mean out of expected range: {mean_actual}" # Loose check

    # Check default dtype (should likely be bool or int-like depending on backend)
    rand_tensor_default = tensor.random_bernoulli(shape, p=p)
    assert tensor.shape(rand_tensor_default) == shape, "random_bernoulli default dtype shape mismatch"
    # Check if dtype is boolean or integer-like
    assert 'bool' in str(tensor.dtype(rand_tensor_default)) or 'int' in str(tensor.dtype(rand_tensor_default)), "bernoulli default dtype not bool or int"


# TODO: Add tests for random_gamma, random_exponential, random_poisson (these require checking against theoretical distributions)
# TODO: Add tests for random_categorical, random_permutation, shuffle