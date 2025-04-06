import pytest
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn import features
# No NumPy import needed here (removed np)

# Assume conftest.py provides 'backend' fixture

@pytest.fixture
def pca_data(backend):
    """Fixture for PCA tests."""
    ops.set_backend(backend)
    # Create data using tensor random functions
    tensor.set_seed(42)
    base_data = tensor.random_uniform((100, 5), dtype=tensor.float32) # Uniform [0, 1)
    # Scale columns
    scale_factors = tensor.convert_to_tensor([10.0, 5.0, 1.0, 0.5, 0.1])
    offset = tensor.convert_to_tensor([1.0, 0.0, -1.0, 0.5, -0.5])
    # Ensure broadcasting works: base_data (100, 5), scale_factors (5,), offset (5,)
    data = ops.add(ops.multiply(base_data, scale_factors), offset)
    return data

def test_features_pca_fit_transform(pca_data, backend):
    """Tests PCA fit_transform using the features.PCA class."""
    ops.set_backend(backend)
    data = pca_data
    n_components = 3

    # Use the PCA class explicitly
    pca_instance = features.PCA()
    transformed = pca_instance.fit_transform(data, n_components=n_components)

    assert isinstance(transformed, tensor.EmberTensor), "PCA transform did not return EmberTensor"
    assert tensor.shape(transformed) == (100, n_components), "PCA transformed shape is incorrect"
    
    # Basic check: variance should decrease with component index (roughly)
    from ember_ml.ops import stats
    variances = stats.var(transformed, axis=0)
    # Check if variances are generally decreasing 
    if tensor.shape(variances)[0] > 1:
        var_0 = variances[0]
        var_1 = variances[1]
        # Use ops.greater_equal with tolerance
        is_decreasing = ops.greater_equal(var_0, ops.subtract(var_1, 1e-5))
        assert tensor.item(is_decreasing), "PCA component variances not decreasing (1st vs 2nd)"

def test_features_pca_inverse_transform(pca_data, backend):
    """Tests PCA inverse_transform using the features.PCA class."""
    ops.set_backend(backend)
    data = pca_data
    n_components = 3

    # Need to fit first to get components/mean for inverse_transform
    pca_instance = features.PCA()
    transformed = pca_instance.fit_transform(data, n_components=n_components)
    
    reconstructed = pca_instance.inverse_transform(transformed)

    assert isinstance(reconstructed, tensor.EmberTensor), "PCA inverse transform did not return EmberTensor"
    assert tensor.shape(reconstructed) == tensor.shape(data), "PCA reconstructed shape is incorrect"
    
    # Check reconstruction error is reasonable
    mean_diff = ops.mean(ops.abs(ops.subtract(data, reconstructed)))
    assert tensor.item(mean_diff) < 5.0, f"PCA reconstruction error seems too high: {tensor.item(mean_diff)}"

def test_features_one_hot(backend):
    """Tests features.one_hot."""
    ops.set_backend(backend)
    indices = tensor.convert_to_tensor([0, 2, 1, 0])
    depth = 3

    one_hot_result = features.one_hot(indices, depth=depth)
    # Construct expected tensor using Ember operations
    expected = tensor.convert_to_tensor([[1., 0., 0.], 
                                         [0., 0., 1.], 
                                         [0., 1., 0.], 
                                         [1., 0., 0.]])

    assert isinstance(one_hot_result, tensor.EmberTensor), "one_hot did not return EmberTensor"
    # Cast expected to result dtype for comparison if necessary
    if tensor.dtype(one_hot_result) != tensor.dtype(expected):
        expected = tensor.cast(expected, tensor.dtype(one_hot_result))
    assert ops.all(ops.equal(one_hot_result, expected)), "one_hot result mismatch"

def test_features_scatter(backend):
    """Tests features.scatter."""
    # Note: This is similar to tensor.scatter, testing it via features namespace.
    ops.set_backend(backend)
    indices = tensor.convert_to_tensor([[0, 1], [2, 2]]) # Indices for a (3, 4) tensor
    updates = tensor.convert_to_tensor([100, 200])
    shape = (3, 4)
    
    scattered = features.scatter(indices, updates, shape)
    
    assert tensor.shape(scattered) == shape, "features.scatter shape failed"
    # Construct expected tensor using zeros and updates
    expected_manual = tensor.convert_to_tensor([[0, 100, 0, 0],
                                                [0, 0,   0, 0],
                                                [0, 0, 200, 0]])
    expected_manual = tensor.cast(expected_manual, tensor.dtype(scattered)) # Match dtype
    assert ops.allclose(scattered, expected_manual), f"{backend}: features.scatter content check failed"

# TODO: Add tests for standardize, normalize if interfaces/functions are exposed directly
# TODO: Add tests for specific extractor classes (Terabyte, TemporalStride, Column) likely in separate files