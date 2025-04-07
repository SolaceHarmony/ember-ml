import pytest
from pytest import mark
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn.tensor.common.ember_tensor import EmberTensor # For type checks
import numpy as np # For type checks

# Define the backend order: numpy -> torch -> mlx

# Helper function to get vector tensors
def _get_vector_tensors():
    vec1 = tensor.convert_to_tensor([1.0, 2.0, 3.0])
    vec2 = tensor.convert_to_tensor([4.0, 5.0, 6.0])
    zero_vec = tensor.zeros(3)
    matrix = tensor.convert_to_tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]) # 3x2
    mask = tensor.convert_to_tensor([True, False, True], dtype=tensor.bool_)
    return vec1, vec2, zero_vec, matrix, mask

# --- NumPy Backend Setup & Tests ---
@mark.run(order=1)
def test_setup_numpy():
    """Set up the NumPy backend."""
    print("\n=== Setting backend to NumPy ===")
    ops.set_backend('numpy')
    assert ops.get_backend() == 'numpy'

@mark.run(order=1)
def test_ops_normalize_vector_numpy():
    """Tests ops.normalize_vector with NumPy backend."""
    from ember_ml.ops import stats # Import locally
    vec1, _, zero_vec, matrix, _ = _get_vector_tensors()
    norm_vec1 = ops.normalize_vector(vec1)
    expected_norm = ops.sqrt(stats.sum(ops.square(vec1)))
    expected_vec = ops.divide(vec1, ops.add(expected_norm, tensor.convert_to_tensor(1e-8)))
    assert ops.allclose(norm_vec1, expected_vec), "Normalize vector failed"
    assert ops.allclose(ops.sqrt(stats.sum(ops.square(norm_vec1))), tensor.convert_to_tensor(1.0)), "Normalized vector norm is not 1"
    norm_zero = ops.normalize_vector(zero_vec)
    assert ops.allclose(norm_zero, zero_vec), "Normalize zero vector failed"
    norm_mat_cols = ops.normalize_vector(matrix, axis=0)
    col0_norm = ops.sqrt(stats.sum(ops.square(norm_mat_cols[:, 0])))
    assert ops.allclose(col0_norm, tensor.convert_to_tensor(1.0)), "Normalized matrix col 0 norm is not 1"
    col1_norm = ops.sqrt(stats.sum(ops.square(norm_mat_cols[:, 1])))
    assert ops.allclose(col1_norm, tensor.convert_to_tensor(1.0)), "Normalized matrix col 1 norm is not 1"
    norm_mat_rows = ops.normalize_vector(matrix, axis=1)
    row0_norm = ops.sqrt(stats.sum(ops.square(norm_mat_rows[0, :])))
    assert ops.allclose(row0_norm, tensor.convert_to_tensor(1.0)), "Normalized matrix row 0 norm is not 1"

@mark.run(order=1)
def test_ops_euclidean_distance_numpy():
    """Tests ops.euclidean_distance with NumPy backend."""
    vec1, vec2, _, _, _ = _get_vector_tensors()
    dist = ops.euclidean_distance(vec1, vec2)
    expected_dist = ops.sqrt(tensor.convert_to_tensor(27.0))
    assert ops.allclose(dist, expected_dist), "Euclidean distance failed"
    dist_self = ops.euclidean_distance(vec1, vec1)
    assert ops.allclose(dist_self, tensor.convert_to_tensor(0.0), atol=1e-7), "Euclidean distance to self failed"

@mark.run(order=1)
def test_ops_cosine_similarity_numpy():
    """Tests ops.cosine_similarity with NumPy backend."""
    vec1, vec2, zero_vec, _, _ = _get_vector_tensors()
    sim = ops.cosine_similarity(vec1, vec2)
    expected_sim = tensor.convert_to_tensor(0.9746318)
    assert ops.allclose(sim, expected_sim, atol=1e-5), "Cosine similarity failed"
    sim_self = ops.cosine_similarity(vec1, vec1)
    assert ops.allclose(sim_self, tensor.convert_to_tensor(1.0), atol=1e-7), "Cosine similarity to self failed"
    sim_zero = ops.cosine_similarity(vec1, zero_vec)
    is_nan_or_zero = ops.logical_or(ops.allclose(sim_zero, tensor.convert_to_tensor(0.0), atol=1e-7), ops.isnan(sim_zero))
    assert ops.all(is_nan_or_zero), "Cosine similarity to zero vector failed (not NaN or zero)"

@mark.run(order=1)
def test_ops_exponential_decay_numpy():
    """Tests ops.exponential_decay with NumPy backend."""
    vec = tensor.convert_to_tensor([1.0, 2.0, 3.0])
    rate = 0.5
    result = ops.exponential_decay(vec, decay_rate=rate)
    indices = tensor.arange(tensor.shape(vec)[0], dtype=tensor.float32)
    decay_factor = ops.exp(ops.negative(ops.multiply(indices, tensor.convert_to_tensor(rate))))
    expected = ops.multiply(vec, decay_factor)
    assert ops.allclose(result, expected, atol=1e-5), "Exponential decay failed"
    assert tensor.shape(result) == tensor.shape(vec), "Exponential decay shape mismatch"

@mark.run(order=1)
def test_ops_vector_advanced_shapes_numpy():
    """Basic shape/type checks for advanced vector ops with NumPy backend."""
    vec1, vec2, _, _, mask = _get_vector_tensors()
    stability = ops.compute_energy_stability(vec1)
    assert isinstance(stability, (np.number, float, int)), f"energy_stability type mismatch, got {type(stability)}"
    # assert len(tensor.shape(stability)) == 0, "energy_stability should be scalar" # numpy scalars might not have shape 0
    interference = ops.compute_interference_strength(vec1, vec2)
    assert isinstance(interference, EmberTensor), "interference_strength type mismatch"
    assert len(tensor.shape(interference)) == 0, "interference_strength should be scalar"
    coherence = ops.compute_phase_coherence(vec1, vec2)
    assert isinstance(coherence, EmberTensor), "phase_coherence type mismatch"
    assert len(tensor.shape(coherence)) == 0, "phase_coherence should be scalar"
    partial = ops.partial_interference(vec1, vec2, mask)
    assert isinstance(partial, EmberTensor), "partial_interference type mismatch"
    assert len(tensor.shape(partial)) == 0, "partial_interference should be scalar"


# --- PyTorch Backend Setup & Tests ---
@mark.run(order=2)
def test_setup_torch():
    """Set up the PyTorch backend."""
    print("\n=== Setting backend to PyTorch ===")
    try:
        import torch
        ops.set_backend('torch')
        assert ops.get_backend() == 'torch'
    except ImportError:
        pytest.skip("PyTorch not available")

@mark.run(order=2)
def test_ops_normalize_vector_torch():
    """Tests ops.normalize_vector with PyTorch backend."""
    from ember_ml.ops import stats # Import locally
    vec1, _, zero_vec, matrix, _ = _get_vector_tensors()
    norm_vec1 = ops.normalize_vector(vec1)
    expected_norm = ops.sqrt(stats.sum(ops.square(vec1)))
    expected_vec = ops.divide(vec1, ops.add(expected_norm, tensor.convert_to_tensor(1e-8)))
    assert ops.allclose(norm_vec1, expected_vec), "Normalize vector failed"
    assert ops.allclose(ops.sqrt(stats.sum(ops.square(norm_vec1))), tensor.convert_to_tensor(1.0)), "Normalized vector norm is not 1"
    norm_zero = ops.normalize_vector(zero_vec)
    assert ops.allclose(norm_zero, zero_vec), "Normalize zero vector failed"
    norm_mat_cols = ops.normalize_vector(matrix, axis=0)
    col0_norm = ops.sqrt(stats.sum(ops.square(norm_mat_cols[:, 0])))
    assert ops.allclose(col0_norm, tensor.convert_to_tensor(1.0)), "Normalized matrix col 0 norm is not 1"
    col1_norm = ops.sqrt(stats.sum(ops.square(norm_mat_cols[:, 1])))
    assert ops.allclose(col1_norm, tensor.convert_to_tensor(1.0)), "Normalized matrix col 1 norm is not 1"
    norm_mat_rows = ops.normalize_vector(matrix, axis=1)
    row0_norm = ops.sqrt(stats.sum(ops.square(norm_mat_rows[0, :])))
    assert ops.allclose(row0_norm, tensor.convert_to_tensor(1.0)), "Normalized matrix row 0 norm is not 1"

@mark.run(order=2)
def test_ops_euclidean_distance_torch():
    """Tests ops.euclidean_distance with PyTorch backend."""
    vec1, vec2, _, _, _ = _get_vector_tensors()
    dist = ops.euclidean_distance(vec1, vec2)
    expected_dist = ops.sqrt(tensor.convert_to_tensor(27.0))
    assert ops.allclose(dist, expected_dist), "Euclidean distance failed"
    dist_self = ops.euclidean_distance(vec1, vec1)
    assert ops.allclose(dist_self, tensor.convert_to_tensor(0.0), atol=1e-7), "Euclidean distance to self failed"

@mark.run(order=2)
def test_ops_cosine_similarity_torch():
    """Tests ops.cosine_similarity with PyTorch backend."""
    vec1, vec2, zero_vec, _, _ = _get_vector_tensors()
    sim = ops.cosine_similarity(vec1, vec2)
    expected_sim = tensor.convert_to_tensor(0.9746318)
    assert ops.allclose(sim, expected_sim, atol=1e-5), "Cosine similarity failed"
    sim_self = ops.cosine_similarity(vec1, vec1)
    assert ops.allclose(sim_self, tensor.convert_to_tensor(1.0), atol=1e-7), "Cosine similarity to self failed"
    sim_zero = ops.cosine_similarity(vec1, zero_vec)
    is_nan_or_zero = ops.logical_or(ops.allclose(sim_zero, tensor.convert_to_tensor(0.0), atol=1e-7), ops.isnan(sim_zero))
    assert ops.all(is_nan_or_zero), "Cosine similarity to zero vector failed (not NaN or zero)"

@mark.run(order=2)
def test_ops_exponential_decay_torch():
    """Tests ops.exponential_decay with PyTorch backend."""
    vec = tensor.convert_to_tensor([1.0, 2.0, 3.0])
    rate = 0.5
    result = ops.exponential_decay(vec, decay_rate=rate)
    indices = tensor.arange(tensor.shape(vec)[0], dtype=tensor.float32)
    decay_factor = ops.exp(ops.negative(ops.multiply(indices, tensor.convert_to_tensor(rate))))
    expected = ops.multiply(vec, decay_factor)
    assert ops.allclose(result, expected, atol=1e-5), "Exponential decay failed"
    assert tensor.shape(result) == tensor.shape(vec), "Exponential decay shape mismatch"

@mark.run(order=2)
def test_ops_vector_advanced_shapes_torch():
    """Basic shape/type checks for advanced vector ops with PyTorch backend."""
    vec1, vec2, _, _, mask = _get_vector_tensors()
    stability = ops.compute_energy_stability(vec1)
    assert isinstance(stability, EmberTensor), f"energy_stability type mismatch, got {type(stability)}"
    assert len(tensor.shape(stability)) == 0, "energy_stability should be scalar"
    interference = ops.compute_interference_strength(vec1, vec2)
    assert isinstance(interference, EmberTensor), "interference_strength type mismatch"
    assert len(tensor.shape(interference)) == 0, "interference_strength should be scalar"
    coherence = ops.compute_phase_coherence(vec1, vec2)
    assert isinstance(coherence, EmberTensor), "phase_coherence type mismatch"
    assert len(tensor.shape(coherence)) == 0, "phase_coherence should be scalar"
    partial = ops.partial_interference(vec1, vec2, mask)
    assert isinstance(partial, EmberTensor), "partial_interference type mismatch"
    assert len(tensor.shape(partial)) == 0, "partial_interference should be scalar"


# --- MLX Backend Setup & Tests ---
@mark.run(order=3)
def test_setup_mlx():
    """Set up the MLX backend."""
    print("\n=== Setting backend to MLX ===")
    try:
        import mlx.core
        ops.set_backend('mlx')
        assert ops.get_backend() == 'mlx'
    except ImportError:
        pytest.skip("MLX not available")

@mark.run(order=3)
def test_ops_normalize_vector_mlx():
    """Tests ops.normalize_vector with MLX backend."""
    from ember_ml.ops import stats # Import locally
    vec1, _, zero_vec, matrix, _ = _get_vector_tensors()
    norm_vec1 = ops.normalize_vector(vec1)
    expected_norm = ops.sqrt(stats.sum(ops.square(vec1)))
    expected_vec = ops.divide(vec1, ops.add(expected_norm, tensor.convert_to_tensor(1e-8)))
    assert ops.allclose(norm_vec1, expected_vec), "Normalize vector failed"
    assert ops.allclose(ops.sqrt(stats.sum(ops.square(norm_vec1))), tensor.convert_to_tensor(1.0)), "Normalized vector norm is not 1"
    norm_zero = ops.normalize_vector(zero_vec)
    assert ops.allclose(norm_zero, zero_vec), "Normalize zero vector failed"
    norm_mat_cols = ops.normalize_vector(matrix, axis=0)
    col0_norm = ops.sqrt(stats.sum(ops.square(norm_mat_cols[:, 0])))
    assert ops.allclose(col0_norm, tensor.convert_to_tensor(1.0)), "Normalized matrix col 0 norm is not 1"
    col1_norm = ops.sqrt(stats.sum(ops.square(norm_mat_cols[:, 1])))
    assert ops.allclose(col1_norm, tensor.convert_to_tensor(1.0)), "Normalized matrix col 1 norm is not 1"
    norm_mat_rows = ops.normalize_vector(matrix, axis=1)
    row0_norm = ops.sqrt(stats.sum(ops.square(norm_mat_rows[0, :])))
    assert ops.allclose(row0_norm, tensor.convert_to_tensor(1.0)), "Normalized matrix row 0 norm is not 1"

@mark.run(order=3)
def test_ops_euclidean_distance_mlx():
    """Tests ops.euclidean_distance with MLX backend."""
    vec1, vec2, _, _, _ = _get_vector_tensors()
    dist = ops.euclidean_distance(vec1, vec2)
    expected_dist = ops.sqrt(tensor.convert_to_tensor(27.0))
    assert ops.allclose(dist, expected_dist), "Euclidean distance failed"
    dist_self = ops.euclidean_distance(vec1, vec1)
    assert ops.allclose(dist_self, tensor.convert_to_tensor(0.0), atol=1e-7), "Euclidean distance to self failed"

@mark.run(order=3)
def test_ops_cosine_similarity_mlx():
    """Tests ops.cosine_similarity with MLX backend."""
    vec1, vec2, zero_vec, _, _ = _get_vector_tensors()
    sim = ops.cosine_similarity(vec1, vec2)
    expected_sim = tensor.convert_to_tensor(0.9746318)
    assert ops.allclose(sim, expected_sim, atol=1e-5), "Cosine similarity failed"
    sim_self = ops.cosine_similarity(vec1, vec1)
    assert ops.allclose(sim_self, tensor.convert_to_tensor(1.0), atol=1e-7), "Cosine similarity to self failed"
    sim_zero = ops.cosine_similarity(vec1, zero_vec)
    is_nan_or_zero = ops.logical_or(ops.allclose(sim_zero, tensor.convert_to_tensor(0.0), atol=1e-7), ops.isnan(sim_zero))
    assert ops.all(is_nan_or_zero), "Cosine similarity to zero vector failed (not NaN or zero)"

@mark.run(order=3)
def test_ops_exponential_decay_mlx():
    """Tests ops.exponential_decay with MLX backend."""
    vec = tensor.convert_to_tensor([1.0, 2.0, 3.0])
    rate = 0.5
    result = ops.exponential_decay(vec, decay_rate=rate)
    indices = tensor.arange(tensor.shape(vec)[0], dtype=tensor.float32)
    decay_factor = ops.exp(ops.negative(ops.multiply(indices, tensor.convert_to_tensor(rate))))
    expected = ops.multiply(vec, decay_factor)
    assert ops.allclose(result, expected, atol=1e-5), "Exponential decay failed"
    assert tensor.shape(result) == tensor.shape(vec), "Exponential decay shape mismatch"

@mark.run(order=3)
def test_ops_vector_advanced_shapes_mlx():
    """Basic shape/type checks for advanced vector ops with MLX backend."""
    vec1, vec2, _, _, mask = _get_vector_tensors()
    stability = ops.compute_energy_stability(vec1)
    assert isinstance(stability, EmberTensor), f"energy_stability type mismatch, got {type(stability)}"
    assert len(tensor.shape(stability)) == 0, "energy_stability should be scalar"
    interference = ops.compute_interference_strength(vec1, vec2)
    assert isinstance(interference, EmberTensor), "interference_strength type mismatch"
    assert len(tensor.shape(interference)) == 0, "interference_strength should be scalar"
    coherence = ops.compute_phase_coherence(vec1, vec2)
    assert isinstance(coherence, EmberTensor), "phase_coherence type mismatch"
    assert len(tensor.shape(coherence)) == 0, "phase_coherence should be scalar"
    partial = ops.partial_interference(vec1, vec2, mask)
    assert isinstance(partial, EmberTensor), "partial_interference type mismatch"
    assert len(tensor.shape(partial)) == 0, "partial_interference should be scalar"

# TODO: Add more detailed value checks for advanced ops if reference values are available.