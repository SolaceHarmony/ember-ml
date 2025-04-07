# tests/mlx_tests/test_nn_features.py
import pytest
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn import features
from ember_ml.ops import stats # Import stats needed for PCA check

# Note: Assumes conftest.py provides the mlx_backend fixture

# Helper function to generate PCA data for a specific backend
def _generate_pca_data():
    """Generates consistent PCA test data for the currently set backend."""
    tensor.set_seed(42)
    base_data = tensor.random_uniform((100, 5), dtype=tensor.float32)
    scale_factors = tensor.convert_to_tensor([10.0, 5.0, 1.0, 0.5, 0.1])
    offset = tensor.convert_to_tensor([1.0, 0.0, -1.0, 0.5, -0.5])
    data = ops.add(ops.multiply(base_data, scale_factors), offset)
    return data

def test_features_pca_fit_transform_mlx(mlx_backend): # Use fixture
    """Tests PCA fit_transform with MLX backend."""
    data = _generate_pca_data()
    n_components = 3
    pca_instance = features.PCA()
    transformed = pca_instance.fit_transform(data, n_components=n_components)
    assert isinstance(transformed, tensor.EmberTensor), "PCA transform did not return EmberTensor"
    assert tensor.shape(transformed) == (100, n_components), "PCA transformed shape is incorrect"
    variances = stats.var(transformed, axis=0)
    if tensor.shape(variances)[0] > 1:
        var_0 = variances[0]; var_1 = variances[1]
        is_decreasing = ops.greater_equal(var_0, ops.subtract(var_1, 1e-5))
        assert tensor.item(is_decreasing), "PCA component variances not decreasing (1st vs 2nd)"

def test_features_pca_inverse_transform_mlx(mlx_backend): # Use fixture
    """Tests PCA inverse_transform with MLX backend."""
    data = _generate_pca_data()
    n_components = 3
    pca_instance = features.PCA()
    transformed = pca_instance.fit_transform(data, n_components=n_components)
    reconstructed = pca_instance.inverse_transform(transformed)
    assert isinstance(reconstructed, tensor.EmberTensor), "PCA inverse transform did not return EmberTensor"
    assert tensor.shape(reconstructed) == tensor.shape(data), "PCA reconstructed shape is incorrect"
    mean_diff = ops.stats.mean(ops.abs(ops.subtract(data, reconstructed)))
    assert tensor.item(mean_diff) < 5.0, f"PCA reconstruction error seems too high: {tensor.item(mean_diff)}"

def test_features_one_hot_mlx(mlx_backend): # Use fixture
    """Tests features.one_hot with MLX backend."""
    indices = tensor.convert_to_tensor([0, 2, 1, 0])
    depth = 3
    one_hot_result = features.one_hot(indices, depth=depth)
    expected = tensor.convert_to_tensor([[1., 0., 0.], [0., 0., 1.], [0., 1., 0.], [1., 0., 0.]])
    assert isinstance(one_hot_result, tensor.EmberTensor), "one_hot did not return EmberTensor"
    if tensor.dtype(one_hot_result) != tensor.dtype(expected):
        expected = tensor.cast(expected, tensor.dtype(one_hot_result))
    assert ops.all(ops.equal(one_hot_result, expected)), "one_hot result mismatch"

def test_features_scatter_mlx(mlx_backend): # Use fixture
    """Tests features.scatter with MLX backend."""
    indices = tensor.convert_to_tensor([[0, 1], [2, 2]])
    updates = tensor.convert_to_tensor([100, 200])
    shape = (3, 4)
    scattered = features.scatter(indices, updates, shape)
    assert tensor.shape(scattered) == shape, "features.scatter shape failed"
    expected_manual = tensor.convert_to_tensor([[0, 100, 0, 0],[0, 0,   0, 0],[0, 0, 200, 0]])
    expected_manual = tensor.cast(expected_manual, tensor.dtype(scattered))
    assert ops.allclose(scattered, expected_manual), f"features.scatter content check failed"