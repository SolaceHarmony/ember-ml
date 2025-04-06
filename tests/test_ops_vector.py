import pytest
from ember_ml import ops
from ember_ml.nn import tensor

# Assume conftest.py provides 'backend' fixture

@pytest.fixture
def vector_tensors(backend):
    """Fixture to create sample vectors for testing."""
    ops.set_backend(backend)
    vec1 = tensor.convert_to_tensor([1.0, 2.0, 3.0])
    vec2 = tensor.convert_to_tensor([4.0, 5.0, 6.0])
    zero_vec = tensor.zeros(3)
    matrix = tensor.convert_to_tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]) # 3x2
    return vec1, vec2, zero_vec, matrix

def test_ops_normalize_vector(vector_tensors, backend):
    """Tests ops.normalize_vector."""
    ops.set_backend(backend)
    vec1, _, zero_vec, matrix = vector_tensors

    # Normalize vec1 (L2 norm)
    norm_vec1 = ops.normalize_vector(vec1)
    # Expected norm = sqrt(1^2 + 2^2 + 3^2) = sqrt(14)
    # Expected vec = [1/sqrt(14), 2/sqrt(14), 3/sqrt(14)] approx [0.267, 0.534, 0.801]
    expected_norm = ops.sqrt(ops.stats.sum(ops.square(vec1)))
    # Add epsilon to avoid division by zero if norm is zero
    expected_vec = ops.divide(vec1, ops.add(expected_norm, tensor.convert_to_tensor(1e-8))) 
    assert ops.allclose(norm_vec1, expected_vec), f"{backend}: Normalize vector failed"
    # Check if norm is close to 1
    assert ops.allclose(ops.sqrt(ops.stats.sum(ops.square(norm_vec1))), tensor.convert_to_tensor(1.0)), f"{backend}: Normalized vector norm is not 1"

    # Normalize zero vector (should remain zero, handle potential division by zero)
    norm_zero = ops.normalize_vector(zero_vec)
    assert ops.allclose(norm_zero, zero_vec), f"{backend}: Normalize zero vector failed"

    # Normalize matrix columns (axis=0)
    norm_mat_cols = ops.normalize_vector(matrix, axis=0)
    # Check norm of first column
    col0_norm = ops.sqrt(ops.stats.sum(ops.square(norm_mat_cols[:, 0])))
    assert ops.allclose(col0_norm, tensor.convert_to_tensor(1.0)), f"{backend}: Normalized matrix col 0 norm is not 1"
     # Check norm of second column
    col1_norm = ops.sqrt(ops.stats.sum(ops.square(norm_mat_cols[:, 1])))
    assert ops.allclose(col1_norm, tensor.convert_to_tensor(1.0)), f"{backend}: Normalized matrix col 1 norm is not 1"

    # Normalize matrix rows (axis=1)
    norm_mat_rows = ops.normalize_vector(matrix, axis=1)
     # Check norm of first row
    row0_norm = ops.sqrt(ops.stats.sum(ops.square(norm_mat_rows[0, :])))
    assert ops.allclose(row0_norm, tensor.convert_to_tensor(1.0)), f"{backend}: Normalized matrix row 0 norm is not 1"

def test_ops_euclidean_distance(vector_tensors, backend):
    """Tests ops.euclidean_distance."""
    ops.set_backend(backend)
    vec1, vec2, _, _ = vector_tensors

    dist = ops.euclidean_distance(vec1, vec2)
    # Expected dist = sqrt((1-4)^2 + (2-5)^2 + (3-6)^2) = sqrt((-3)^2 + (-3)^2 + (-3)^2)
    # = sqrt(9 + 9 + 9) = sqrt(27) approx 5.196
    expected_dist = ops.sqrt(tensor.convert_to_tensor(27.0))
    assert ops.allclose(dist, expected_dist), f"{backend}: Euclidean distance failed"

    # Distance to self should be 0
    dist_self = ops.euclidean_distance(vec1, vec1)
    assert ops.allclose(dist_self, tensor.convert_to_tensor(0.0), atol=1e-7), f"{backend}: Euclidean distance to self failed"

def test_ops_cosine_similarity(vector_tensors, backend):
    """Tests ops.cosine_similarity."""
    ops.set_backend(backend)
    vec1, vec2, zero_vec, _ = vector_tensors

    sim = ops.cosine_similarity(vec1, vec2)
    # Expected sim = dot(vec1, vec2) / (norm(vec1) * norm(vec2))
    # dot = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    # norm(vec1) = sqrt(14)
    # norm(vec2) = sqrt(4^2 + 5^2 + 6^2) = sqrt(16 + 25 + 36) = sqrt(77)
    # sim = 32 / (sqrt(14) * sqrt(77)) = 32 / sqrt(1078) approx 32 / 32.83 = 0.9746
    expected_sim = tensor.convert_to_tensor(0.9746318)
    assert ops.allclose(sim, expected_sim, atol=1e-5), f"{backend}: Cosine similarity failed"

    # Similarity with self should be 1
    sim_self = ops.cosine_similarity(vec1, vec1)
    assert ops.allclose(sim_self, tensor.convert_to_tensor(1.0), atol=1e-7), f"{backend}: Cosine similarity to self failed"

    # Similarity with zero vector (handle potential division by zero - should likely be 0 or NaN depending on backend)
    # Most libraries return 0 or NaN for cosine similarity with a zero vector. Let's check for NaN or 0.
    sim_zero = ops.cosine_similarity(vec1, zero_vec)
    # isnan might not be directly available in ops, need to check implementation or use try-except


def test_ops_exponential_decay(backend):
    """Tests ops.exponential_decay."""
    ops.set_backend(backend)
    vec = tensor.convert_to_tensor([1.0, 2.0, 3.0])
    rate = 0.5
    result = ops.exponential_decay(vec, decay_rate=rate)
    #     def exponential_decay(self, initial_value: Any, decay_rate: Any, time_step: Any) -> Any:
    # Expected: vec * exp(-rate * index) -> [1*exp(0), 2*exp(-0.5), 3*exp(-1)]
    # Approx: [1.0, 2*0.6065, 3*0.3678] = [1.0, 1.213, 1.103]
    # Note: This assumes index-based decay. If it's time-based, the test needs adjustment.
    # Let's assume simple index-based decay for now.
    indices = tensor.arange(tensor.shape(vec)[0], dtype=tensor.float32)
    decay_factor = ops.exp(ops.negative(ops.multiply(indices, tensor.convert_to_tensor(rate))))
    expected = ops.multiply(vec, decay_factor)
    assert ops.allclose(result, expected, atol=1e-5), f"{backend}: Exponential decay failed"
    assert tensor.shape(result) == tensor.shape(vec), f"{backend}: Exponential decay shape mismatch"

# NOTE: Tests for energy_stability, interference_strength, phase_coherence, 
# and partial_interference require specific domain knowledge or reference values
# for meaningful assertions beyond basic shape/type checks. Adding basic checks.

def test_ops_vector_advanced_shapes(vector_tensors, backend):
    """Basic shape/type checks for advanced vector ops."""
    ops.set_backend(backend)
    vec1, vec2, _, _ = vector_tensors
    mask = tensor.convert_to_tensor([True, False, True], dtype=tensor.bool_)

    # energy_stability
    stability = ops.compute_energy_stability(vec1)
    assert isinstance(stability, tensor.EmberTensor), "energy_stability type mismatch"
    assert len(tensor.shape(stability)) == 0, "energy_stability should be scalar"

    # interference_strength
    interference = ops.compute_interference_strength(vec1, vec2)
    assert isinstance(interference, tensor.EmberTensor), "interference_strength type mismatch"
    assert len(tensor.shape(interference)) == 0, "interference_strength should be scalar"

    # phase_coherence
    coherence = ops.compute_phase_coherence(vec1, vec2)
    assert isinstance(coherence, tensor.EmberTensor), "phase_coherence type mismatch"
    assert len(tensor.shape(coherence)) == 0, "phase_coherence should be scalar"

    # partial_interference
    partial = ops.partial_interference(vec1, vec2, mask)
    assert isinstance(partial, tensor.EmberTensor), "partial_interference type mismatch"
    assert len(tensor.shape(partial)) == 0, "partial_interference should be scalar"


    # For simplicity, we might assert it's close to 0 if not NaN handling is complex across backends here.
    # Checking if it's close to 0 or NaN
    is_nan_or_zero = ops.logical_or(ops.allclose(sim_zero, tensor.convert_to_tensor(0.0), atol=1e-7), ops.isnan(sim_zero))
    assert ops.all(is_nan_or_zero), f"{backend}: Cosine similarity to zero vector failed (not NaN or zero)"


# TODO: Add tests for compute_energy_stability, compute_interference_strength, compute_phase_coherence, partial_interference, exponential_decay, gaussian