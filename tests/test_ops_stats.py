import pytest
from pytest import mark
from ember_ml import ops
from ember_ml.ops import stats # Import the stats submodule
from ember_ml.nn import tensor

# Define the backend order: numpy -> torch -> mlx

# Helper function to get stats tensor
def _get_stats_tensor():
    # Using integers first to test type handling, will cast later if needed
    return tensor.convert_to_tensor([[1, 2, 3], [4, 5, 6]])

# --- NumPy Backend Setup & Tests ---
@mark.run(order=1)
def test_setup_numpy():
    """Set up the NumPy backend."""
    print("\n=== Setting backend to NumPy ===")
    ops.set_backend('numpy')
    assert ops.get_backend() == 'numpy'

@mark.run(order=1)
def test_stats_mean_numpy():
    """Tests stats.mean with NumPy backend."""
    t = _get_stats_tensor()
    t_float = tensor.cast(t, tensor.float32)
    mean_all = stats.mean(t_float)
    assert ops.allclose(mean_all, tensor.convert_to_tensor(3.5)), "Mean all failed"
    mean_cols = stats.mean(t_float, axis=0)
    assert ops.allclose(mean_cols, tensor.convert_to_tensor([2.5, 3.5, 4.5])), "Mean axis=0 failed"
    mean_rows = stats.mean(t_float, axis=1)
    assert ops.allclose(mean_rows, tensor.convert_to_tensor([2.0, 5.0])), "Mean axis=1 failed"

@mark.run(order=1)
def test_stats_sum_numpy():
    """Tests stats.sum with NumPy backend."""
    t = _get_stats_tensor()
    sum_all = stats.sum(t)
    assert ops.allclose(sum_all, tensor.convert_to_tensor(21)), "Sum all failed"
    sum_cols = stats.sum(t, axis=0)
    assert ops.allclose(sum_cols, tensor.convert_to_tensor([5, 7, 9])), "Sum axis=0 failed"
    sum_rows = stats.sum(t, axis=1)
    assert ops.allclose(sum_rows, tensor.convert_to_tensor([6, 15])), "Sum axis=1 failed"

@mark.run(order=1)
def test_stats_var_numpy():
    """Tests stats.var with NumPy backend."""
    t = _get_stats_tensor()
    t_float = tensor.cast(t, tensor.float32)
    var_all = stats.var(t_float) # ddof=0 default
    assert ops.allclose(var_all, tensor.convert_to_tensor(2.916666), atol=1e-5), "Var all (ddof=0) failed"
    var_all_sample = stats.var(t_float, ddof=1)
    assert ops.allclose(var_all_sample, tensor.convert_to_tensor(3.5), atol=1e-5), "Var all (ddof=1) failed"

@mark.run(order=1)
def test_stats_std_numpy():
    """Tests stats.std with NumPy backend."""
    t = _get_stats_tensor()
    t_float = tensor.cast(t, tensor.float32)
    std_all = stats.std(t_float) # ddof=0 default
    assert ops.allclose(std_all, tensor.convert_to_tensor(1.707825), atol=1e-5), "Std all (ddof=0) failed"
    std_all_sample = stats.std(t_float, ddof=1)
    assert ops.allclose(std_all_sample, tensor.convert_to_tensor(1.870828), atol=1e-5), "Std all (ddof=1) failed"

@mark.run(order=1)
def test_stats_min_max_numpy():
    """Tests stats.min and stats.max with NumPy backend."""
    t = _get_stats_tensor()
    min_all = stats.min(t); max_all = stats.max(t)
    assert ops.allclose(min_all, tensor.convert_to_tensor(1)), "Min all failed"
    assert ops.allclose(max_all, tensor.convert_to_tensor(6)), "Max all failed"
    min_cols = stats.min(t, axis=0); max_cols = stats.max(t, axis=0)
    assert ops.allclose(min_cols, tensor.convert_to_tensor([1, 2, 3])), "Min axis=0 failed"
    assert ops.allclose(max_cols, tensor.convert_to_tensor([4, 5, 6])), "Max axis=0 failed"
    min_rows = stats.min(t, axis=1); max_rows = stats.max(t, axis=1)
    assert ops.allclose(min_rows, tensor.convert_to_tensor([1, 4])), "Min axis=1 failed"
    assert ops.allclose(max_rows, tensor.convert_to_tensor([3, 6])), "Max axis=1 failed"

@mark.run(order=1)
def test_stats_median_numpy():
    """Tests stats.median with NumPy backend."""
    t_odd = tensor.convert_to_tensor([1, 5, 2, 8, 7])
    median_odd = stats.median(t_odd)
    assert ops.allclose(median_odd, tensor.convert_to_tensor(5)), "Median odd failed"
    t_even = tensor.convert_to_tensor([1, 5, 2, 8, 7, 3])
    t_even_float = tensor.cast(t_even, tensor.float32)
    median_even = stats.median(t_even_float)
    assert ops.allclose(median_even, tensor.convert_to_tensor(4.0)), "Median even failed"
    t_matrix = tensor.convert_to_tensor([[1, 5, 2], [8, 7, 3]])
    t_matrix_float = tensor.cast(t_matrix, tensor.float32)
    median_cols = stats.median(t_matrix_float, axis=0)
    assert ops.allclose(median_cols, tensor.convert_to_tensor([4.5, 6.0, 2.5])), "Median axis=0 failed"

@mark.run(order=1)
def test_stats_cumsum_numpy():
    """Tests stats.cumsum with NumPy backend."""
    t = _get_stats_tensor()
    cumsum_flat = stats.cumsum(t)
    assert ops.allclose(cumsum_flat, tensor.convert_to_tensor([1, 3, 6, 10, 15, 21])), "Cumsum flat failed"
    cumsum_cols = stats.cumsum(t, axis=0)
    assert ops.allclose(cumsum_cols, tensor.convert_to_tensor([[1, 2, 3], [5, 7, 9]])), "Cumsum axis=0 failed"
    cumsum_rows = stats.cumsum(t, axis=1)
    assert ops.allclose(cumsum_rows, tensor.convert_to_tensor([[1, 3, 6], [4, 9, 15]])), "Cumsum axis=1 failed"

@mark.run(order=1)
def test_stats_argmax_numpy():
    """Tests stats.argmax with NumPy backend."""
    t = tensor.convert_to_tensor([[1, 5, 3], [4, 2, 6]])
    argmax_flat = stats.argmax(t)
    assert ops.allclose(argmax_flat, tensor.convert_to_tensor(5)), "Argmax flat failed"
    argmax_cols = stats.argmax(t, axis=0)
    assert ops.allclose(argmax_cols, tensor.convert_to_tensor([1, 0, 1])), "Argmax axis=0 failed"
    argmax_rows = stats.argmax(t, axis=1)
    assert ops.allclose(argmax_rows, tensor.convert_to_tensor([1, 2])), "Argmax axis=1 failed"

@mark.run(order=1)
def test_stats_sort_argsort_numpy():
    """Tests stats.sort and stats.argsort with NumPy backend."""
    t = tensor.convert_to_tensor([[3, 1, 2], [6, 5, 4]])
    sorted_rows = stats.sort(t, axis=1)
    expected_sorted = tensor.convert_to_tensor([[1, 2, 3], [4, 5, 6]])
    assert ops.allclose(sorted_rows, expected_sorted), "Sort axis=1 failed"
    argsort_rows = stats.argsort(t, axis=1)
    expected_argsort = tensor.convert_to_tensor([[1, 2, 0], [2, 1, 0]]) # Corrected expectation
    assert ops.allclose(argsort_rows, expected_argsort), "Argsort axis=1 failed"
    sorted_desc = stats.sort(t, axis=1, descending=True)
    expected_sorted_desc = tensor.convert_to_tensor([[3, 2, 1], [6, 5, 4]])
    assert ops.allclose(sorted_desc, expected_sorted_desc), "Sort descending failed"
    argsort_desc = stats.argsort(t, axis=1, descending=True)
    expected_argsort_desc = tensor.convert_to_tensor([[0, 2, 1], [0, 1, 2]]) # Corrected expectation
    assert ops.allclose(argsort_desc, expected_argsort_desc), "Argsort descending failed"

@mark.run(order=1)
def test_stats_percentile_numpy():
    """Tests stats.percentile with NumPy backend."""
    t = tensor.arange(1, 11); t_float = tensor.cast(t, tensor.float32)
    p0 = stats.percentile(t_float, 0); assert ops.allclose(p0, tensor.convert_to_tensor(1.0)), "Percentile q=0 failed"
    p100 = stats.percentile(t_float, 100); assert ops.allclose(p100, tensor.convert_to_tensor(10.0)), "Percentile q=100 failed"
    p50 = stats.percentile(t_float, 50); assert ops.allclose(p50, tensor.convert_to_tensor(5.5)), "Percentile q=50 failed"
    p25 = stats.percentile(t_float, 25); assert ops.allclose(p25, tensor.convert_to_tensor(3.25), atol=0.01), "Percentile q=25 failed"
    p75 = stats.percentile(t_float, 75); assert ops.allclose(p75, tensor.convert_to_tensor(7.75), atol=0.01), "Percentile q=75 failed"
    qs = [10, 90]; p_multi = stats.percentile(t_float, qs)
    expected_multi = tensor.convert_to_tensor([1.9, 9.1])
    assert ops.allclose(p_multi, expected_multi, atol=0.01), "Percentile multiple qs failed"
    assert tensor.shape(p_multi) == (len(qs),), "Percentile multiple qs shape failed"

@mark.run(order=1)
def test_stats_gaussian_numpy():
    """Tests stats.gaussian with NumPy backend."""
    vec = tensor.convert_to_tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    mean, std = 0.0, 1.0
    result = stats.gaussian(vec, mu=mean, sigma=std)
    mean_tensor = tensor.convert_to_tensor(mean, dtype=tensor.get_dtype(vec))
    std_tensor = tensor.convert_to_tensor(std, dtype=tensor.get_dtype(vec))
    term = ops.divide(ops.subtract(vec, mean_tensor), std_tensor)
    exponent = ops.multiply(tensor.convert_to_tensor(-0.5, dtype=tensor.get_dtype(vec)), ops.square(term))
    exp_part = ops.exp(exponent)
    pi_tensor = tensor.convert_to_tensor(ops.pi, dtype=tensor.get_dtype(vec))
    two_tensor = tensor.convert_to_tensor(2.0, dtype=tensor.get_dtype(vec))
    sqrt_two_pi = ops.sqrt(ops.multiply(two_tensor, pi_tensor))
    denominator_factor = ops.multiply(std_tensor, sqrt_two_pi)
    inv_denominator = ops.divide(tensor.convert_to_tensor(1.0, dtype=tensor.get_dtype(vec)), ops.add(denominator_factor, tensor.convert_to_tensor(1e-12)))
    expected = ops.multiply(inv_denominator, exp_part)
    assert ops.allclose(result, expected, atol=1e-5), "Gaussian function failed"
    assert tensor.shape(result) == tensor.shape(vec), "Gaussian function shape mismatch"


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
def test_stats_mean_torch():
    """Tests stats.mean with PyTorch backend."""
    t = _get_stats_tensor()
    t_float = tensor.cast(t, tensor.float32)
    mean_all = stats.mean(t_float)
    assert ops.allclose(mean_all, tensor.convert_to_tensor(3.5)), "Mean all failed"
    mean_cols = stats.mean(t_float, axis=0)
    assert ops.allclose(mean_cols, tensor.convert_to_tensor([2.5, 3.5, 4.5])), "Mean axis=0 failed"
    mean_rows = stats.mean(t_float, axis=1)
    assert ops.allclose(mean_rows, tensor.convert_to_tensor([2.0, 5.0])), "Mean axis=1 failed"

@mark.run(order=2)
def test_stats_sum_torch():
    """Tests stats.sum with PyTorch backend."""
    t = _get_stats_tensor()
    sum_all = stats.sum(t)
    assert ops.allclose(sum_all, tensor.convert_to_tensor(21)), "Sum all failed"
    sum_cols = stats.sum(t, axis=0)
    assert ops.allclose(sum_cols, tensor.convert_to_tensor([5, 7, 9])), "Sum axis=0 failed"
    sum_rows = stats.sum(t, axis=1)
    assert ops.allclose(sum_rows, tensor.convert_to_tensor([6, 15])), "Sum axis=1 failed"

@mark.run(order=2)
def test_stats_var_torch():
    """Tests stats.var with PyTorch backend."""
    t = _get_stats_tensor()
    t_float = tensor.cast(t, tensor.float32)
    var_all = stats.var(t_float)
    assert ops.allclose(var_all, tensor.convert_to_tensor(2.916666), atol=1e-5), "Var all (ddof=0) failed"
    var_all_sample = stats.var(t_float, ddof=1)
    assert ops.allclose(var_all_sample, tensor.convert_to_tensor(3.5), atol=1e-5), "Var all (ddof=1) failed"

@mark.run(order=2)
def test_stats_std_torch():
    """Tests stats.std with PyTorch backend."""
    t = _get_stats_tensor()
    t_float = tensor.cast(t, tensor.float32)
    std_all = stats.std(t_float)
    assert ops.allclose(std_all, tensor.convert_to_tensor(1.707825), atol=1e-5), "Std all (ddof=0) failed"
    std_all_sample = stats.std(t_float, ddof=1)
    assert ops.allclose(std_all_sample, tensor.convert_to_tensor(1.870828), atol=1e-5), "Std all (ddof=1) failed"

@mark.run(order=2)
def test_stats_min_max_torch():
    """Tests stats.min and stats.max with PyTorch backend."""
    t = _get_stats_tensor()
    min_all = stats.min(t); max_all = stats.max(t)
    assert ops.allclose(min_all, tensor.convert_to_tensor(1)), "Min all failed"
    assert ops.allclose(max_all, tensor.convert_to_tensor(6)), "Max all failed"
    min_cols = stats.min(t, axis=0); max_cols = stats.max(t, axis=0)
    assert ops.allclose(min_cols, tensor.convert_to_tensor([1, 2, 3])), "Min axis=0 failed"
    assert ops.allclose(max_cols, tensor.convert_to_tensor([4, 5, 6])), "Max axis=0 failed"
    min_rows = stats.min(t, axis=1); max_rows = stats.max(t, axis=1)
    assert ops.allclose(min_rows, tensor.convert_to_tensor([1, 4])), "Min axis=1 failed"
    assert ops.allclose(max_rows, tensor.convert_to_tensor([3, 6])), "Max axis=1 failed"

@mark.run(order=2)
def test_stats_median_torch():
    """Tests stats.median with PyTorch backend."""
    t_odd = tensor.convert_to_tensor([1, 5, 2, 8, 7])
    median_odd = stats.median(t_odd)
    assert ops.allclose(median_odd, tensor.convert_to_tensor(5)), "Median odd failed"
    t_even = tensor.convert_to_tensor([1, 5, 2, 8, 7, 3])
    t_even_float = tensor.cast(t_even, tensor.float32)
    median_even = stats.median(t_even_float)
    assert ops.allclose(median_even, tensor.convert_to_tensor(4.0)), "Median even failed"
    t_matrix = tensor.convert_to_tensor([[1, 5, 2], [8, 7, 3]])
    t_matrix_float = tensor.cast(t_matrix, tensor.float32)
    median_cols = stats.median(t_matrix_float, axis=0)
    assert ops.allclose(median_cols, tensor.convert_to_tensor([4.5, 6.0, 2.5])), "Median axis=0 failed"

@mark.run(order=2)
def test_stats_cumsum_torch():
    """Tests stats.cumsum with PyTorch backend."""
    t = _get_stats_tensor()
    cumsum_flat = stats.cumsum(t)
    assert ops.allclose(cumsum_flat, tensor.convert_to_tensor([1, 3, 6, 10, 15, 21])), "Cumsum flat failed"
    cumsum_cols = stats.cumsum(t, axis=0)
    assert ops.allclose(cumsum_cols, tensor.convert_to_tensor([[1, 2, 3], [5, 7, 9]])), "Cumsum axis=0 failed"
    cumsum_rows = stats.cumsum(t, axis=1)
    assert ops.allclose(cumsum_rows, tensor.convert_to_tensor([[1, 3, 6], [4, 9, 15]])), "Cumsum axis=1 failed"

@mark.run(order=2)
def test_stats_argmax_torch():
    """Tests stats.argmax with PyTorch backend."""
    t = tensor.convert_to_tensor([[1, 5, 3], [4, 2, 6]])
    argmax_flat = stats.argmax(t)
    assert ops.allclose(argmax_flat, tensor.convert_to_tensor(5)), "Argmax flat failed"
    argmax_cols = stats.argmax(t, axis=0)
    assert ops.allclose(argmax_cols, tensor.convert_to_tensor([1, 0, 1])), "Argmax axis=0 failed"
    argmax_rows = stats.argmax(t, axis=1)
    assert ops.allclose(argmax_rows, tensor.convert_to_tensor([1, 2])), "Argmax axis=1 failed"

@mark.run(order=2)
def test_stats_sort_argsort_torch():
    """Tests stats.sort and stats.argsort with PyTorch backend."""
    t = tensor.convert_to_tensor([[3, 1, 2], [6, 5, 4]])
    sorted_rows = stats.sort(t, axis=1)
    expected_sorted = tensor.convert_to_tensor([[1, 2, 3], [4, 5, 6]])
    assert ops.allclose(sorted_rows, expected_sorted), "Sort axis=1 failed"
    argsort_rows = stats.argsort(t, axis=1)
    expected_argsort = tensor.convert_to_tensor([[1, 2, 0], [2, 1, 0]])
    assert ops.allclose(argsort_rows, expected_argsort), "Argsort axis=1 failed"
    sorted_desc = stats.sort(t, axis=1, descending=True)
    expected_sorted_desc = tensor.convert_to_tensor([[3, 2, 1], [6, 5, 4]])
    assert ops.allclose(sorted_desc, expected_sorted_desc), "Sort descending failed"
    argsort_desc = stats.argsort(t, axis=1, descending=True)
    expected_argsort_desc = tensor.convert_to_tensor([[0, 2, 1], [0, 1, 2]])
    assert ops.allclose(argsort_desc, expected_argsort_desc), "Argsort descending failed"

@mark.run(order=2)
def test_stats_percentile_torch():
    """Tests stats.percentile with PyTorch backend."""
    t = tensor.arange(1, 11); t_float = tensor.cast(t, tensor.float32)
    p0 = stats.percentile(t_float, 0); assert ops.allclose(p0, tensor.convert_to_tensor(1.0)), "Percentile q=0 failed"
    p100 = stats.percentile(t_float, 100); assert ops.allclose(p100, tensor.convert_to_tensor(10.0)), "Percentile q=100 failed"
    p50 = stats.percentile(t_float, 50); assert ops.allclose(p50, tensor.convert_to_tensor(5.5)), "Percentile q=50 failed"
    p25 = stats.percentile(t_float, 25); assert ops.allclose(p25, tensor.convert_to_tensor(3.25), atol=0.01), "Percentile q=25 failed"
    p75 = stats.percentile(t_float, 75); assert ops.allclose(p75, tensor.convert_to_tensor(7.75), atol=0.01), "Percentile q=75 failed"
    qs = [10, 90]; p_multi = stats.percentile(t_float, qs)
    expected_multi = tensor.convert_to_tensor([1.9, 9.1])
    assert ops.allclose(p_multi, expected_multi, atol=0.01), "Percentile multiple qs failed"
    assert tensor.shape(p_multi) == (len(qs),), "Percentile multiple qs shape failed"

@mark.run(order=2)
def test_stats_gaussian_torch():
    """Tests stats.gaussian with PyTorch backend."""
    vec = tensor.convert_to_tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    mean, std = 0.0, 1.0
    result = stats.gaussian(vec, mu=mean, sigma=std)
    mean_tensor = tensor.convert_to_tensor(mean, dtype=tensor.get_dtype(vec))
    std_tensor = tensor.convert_to_tensor(std, dtype=tensor.get_dtype(vec))
    term = ops.divide(ops.subtract(vec, mean_tensor), std_tensor)
    exponent = ops.multiply(tensor.convert_to_tensor(-0.5, dtype=tensor.get_dtype(vec)), ops.square(term))
    exp_part = ops.exp(exponent)
    pi_tensor = tensor.convert_to_tensor(ops.pi, dtype=tensor.get_dtype(vec))
    two_tensor = tensor.convert_to_tensor(2.0, dtype=tensor.get_dtype(vec))
    sqrt_two_pi = ops.sqrt(ops.multiply(two_tensor, pi_tensor))
    denominator_factor = ops.multiply(std_tensor, sqrt_two_pi)
    inv_denominator = ops.divide(tensor.convert_to_tensor(1.0, dtype=tensor.get_dtype(vec)), ops.add(denominator_factor, tensor.convert_to_tensor(1e-12)))
    expected = ops.multiply(inv_denominator, exp_part)
    assert ops.allclose(result, expected, atol=1e-5), "Gaussian function failed"
    assert tensor.shape(result) == tensor.shape(vec), "Gaussian function shape mismatch"


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
def test_stats_mean_mlx():
    """Tests stats.mean with MLX backend."""
    t = _get_stats_tensor()
    t_float = tensor.cast(t, tensor.float32)
    mean_all = stats.mean(t_float)
    assert ops.allclose(mean_all, tensor.convert_to_tensor(3.5)), "Mean all failed"
    mean_cols = stats.mean(t_float, axis=0)
    assert ops.allclose(mean_cols, tensor.convert_to_tensor([2.5, 3.5, 4.5])), "Mean axis=0 failed"
    mean_rows = stats.mean(t_float, axis=1)
    assert ops.allclose(mean_rows, tensor.convert_to_tensor([2.0, 5.0])), "Mean axis=1 failed"

@mark.run(order=3)
def test_stats_sum_mlx():
    """Tests stats.sum with MLX backend."""
    t = _get_stats_tensor()
    sum_all = stats.sum(t)
    assert ops.allclose(sum_all, tensor.convert_to_tensor(21)), "Sum all failed"
    sum_cols = stats.sum(t, axis=0)
    assert ops.allclose(sum_cols, tensor.convert_to_tensor([5, 7, 9])), "Sum axis=0 failed"
    sum_rows = stats.sum(t, axis=1)
    assert ops.allclose(sum_rows, tensor.convert_to_tensor([6, 15])), "Sum axis=1 failed"

@mark.run(order=3)
def test_stats_var_mlx():
    """Tests stats.var with MLX backend."""
    t = _get_stats_tensor()
    t_float = tensor.cast(t, tensor.float32)
    var_all = stats.var(t_float)
    assert ops.allclose(var_all, tensor.convert_to_tensor(2.916666), atol=1e-5), "Var all (ddof=0) failed"
    var_all_sample = stats.var(t_float, ddof=1)
    assert ops.allclose(var_all_sample, tensor.convert_to_tensor(3.5), atol=1e-5), "Var all (ddof=1) failed"

@mark.run(order=3)
def test_stats_std_mlx():
    """Tests stats.std with MLX backend."""
    t = _get_stats_tensor()
    t_float = tensor.cast(t, tensor.float32)
    std_all = stats.std(t_float)
    assert ops.allclose(std_all, tensor.convert_to_tensor(1.707825), atol=1e-5), "Std all (ddof=0) failed"
    std_all_sample = stats.std(t_float, ddof=1)
    assert ops.allclose(std_all_sample, tensor.convert_to_tensor(1.870828), atol=1e-5), "Std all (ddof=1) failed"

@mark.run(order=3)
def test_stats_min_max_mlx():
    """Tests stats.min and stats.max with MLX backend."""
    t = _get_stats_tensor()
    min_all = stats.min(t); max_all = stats.max(t)
    assert ops.allclose(min_all, tensor.convert_to_tensor(1)), "Min all failed"
    assert ops.allclose(max_all, tensor.convert_to_tensor(6)), "Max all failed"
    min_cols = stats.min(t, axis=0); max_cols = stats.max(t, axis=0)
    assert ops.allclose(min_cols, tensor.convert_to_tensor([1, 2, 3])), "Min axis=0 failed"
    assert ops.allclose(max_cols, tensor.convert_to_tensor([4, 5, 6])), "Max axis=0 failed"
    min_rows = stats.min(t, axis=1); max_rows = stats.max(t, axis=1)
    assert ops.allclose(min_rows, tensor.convert_to_tensor([1, 4])), "Min axis=1 failed"
    assert ops.allclose(max_rows, tensor.convert_to_tensor([3, 6])), "Max axis=1 failed"

@mark.run(order=3)
def test_stats_median_mlx():
    """Tests stats.median with MLX backend."""
    t_odd = tensor.convert_to_tensor([1, 5, 2, 8, 7])
    median_odd = stats.median(t_odd)
    assert ops.allclose(median_odd, tensor.convert_to_tensor(5)), "Median odd failed"
    t_even = tensor.convert_to_tensor([1, 5, 2, 8, 7, 3])
    t_even_float = tensor.cast(t_even, tensor.float32)
    median_even = stats.median(t_even_float)
    assert ops.allclose(median_even, tensor.convert_to_tensor(4.0)), "Median even failed"
    t_matrix = tensor.convert_to_tensor([[1, 5, 2], [8, 7, 3]])
    t_matrix_float = tensor.cast(t_matrix, tensor.float32)
    median_cols = stats.median(t_matrix_float, axis=0)
    assert ops.allclose(median_cols, tensor.convert_to_tensor([4.5, 6.0, 2.5])), "Median axis=0 failed"

@mark.run(order=3)
def test_stats_cumsum_mlx():
    """Tests stats.cumsum with MLX backend."""
    t = _get_stats_tensor()
    cumsum_flat = stats.cumsum(t)
    assert ops.allclose(cumsum_flat, tensor.convert_to_tensor([1, 3, 6, 10, 15, 21])), "Cumsum flat failed"
    cumsum_cols = stats.cumsum(t, axis=0)
    assert ops.allclose(cumsum_cols, tensor.convert_to_tensor([[1, 2, 3], [5, 7, 9]])), "Cumsum axis=0 failed"
    cumsum_rows = stats.cumsum(t, axis=1)
    assert ops.allclose(cumsum_rows, tensor.convert_to_tensor([[1, 3, 6], [4, 9, 15]])), "Cumsum axis=1 failed"

@mark.run(order=3)
def test_stats_argmax_mlx():
    """Tests stats.argmax with MLX backend."""
    t = tensor.convert_to_tensor([[1, 5, 3], [4, 2, 6]])
    argmax_flat = stats.argmax(t)
    assert ops.allclose(argmax_flat, tensor.convert_to_tensor(5)), "Argmax flat failed"
    argmax_cols = stats.argmax(t, axis=0)
    assert ops.allclose(argmax_cols, tensor.convert_to_tensor([1, 0, 1])), "Argmax axis=0 failed"
    argmax_rows = stats.argmax(t, axis=1)
    assert ops.allclose(argmax_rows, tensor.convert_to_tensor([1, 2])), "Argmax axis=1 failed"

@mark.run(order=3)
def test_stats_sort_argsort_mlx():
    """Tests stats.sort and stats.argsort with MLX backend."""
    t = tensor.convert_to_tensor([[3, 1, 2], [6, 5, 4]])
    sorted_rows = stats.sort(t, axis=1)
    expected_sorted = tensor.convert_to_tensor([[1, 2, 3], [4, 5, 6]])
    assert ops.allclose(sorted_rows, expected_sorted), "Sort axis=1 failed"
    argsort_rows = stats.argsort(t, axis=1)
    expected_argsort = tensor.convert_to_tensor([[1, 2, 0], [2, 1, 0]])
    assert ops.allclose(argsort_rows, expected_argsort), "Argsort axis=1 failed"
    sorted_desc = stats.sort(t, axis=1, descending=True)
    expected_sorted_desc = tensor.convert_to_tensor([[3, 2, 1], [6, 5, 4]])
    assert ops.allclose(sorted_desc, expected_sorted_desc), "Sort descending failed"
    argsort_desc = stats.argsort(t, axis=1, descending=True)
    expected_argsort_desc = tensor.convert_to_tensor([[0, 2, 1], [0, 1, 2]])
    assert ops.allclose(argsort_desc, expected_argsort_desc), "Argsort descending failed"

@mark.run(order=3)
def test_stats_percentile_mlx():
    """Tests stats.percentile with MLX backend."""
    t = tensor.arange(1, 11); t_float = tensor.cast(t, tensor.float32)
    p0 = stats.percentile(t_float, 0); assert ops.allclose(p0, tensor.convert_to_tensor(1.0)), "Percentile q=0 failed"
    p100 = stats.percentile(t_float, 100); assert ops.allclose(p100, tensor.convert_to_tensor(10.0)), "Percentile q=100 failed"
    p50 = stats.percentile(t_float, 50); assert ops.allclose(p50, tensor.convert_to_tensor(5.5)), "Percentile q=50 failed"
    p25 = stats.percentile(t_float, 25); assert ops.allclose(p25, tensor.convert_to_tensor(3.25), atol=0.01), "Percentile q=25 failed"
    p75 = stats.percentile(t_float, 75); assert ops.allclose(p75, tensor.convert_to_tensor(7.75), atol=0.01), "Percentile q=75 failed"
    qs = [10, 90]; p_multi = stats.percentile(t_float, qs)
    expected_multi = tensor.convert_to_tensor([1.9, 9.1])
    assert ops.allclose(p_multi, expected_multi, atol=0.01), "Percentile multiple qs failed"
    assert tensor.shape(p_multi) == (len(qs),), "Percentile multiple qs shape failed"

@mark.run(order=3)
def test_stats_gaussian_mlx():
    """Tests stats.gaussian with MLX backend."""
    vec = tensor.convert_to_tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    mean, std = 0.0, 1.0
    result = stats.gaussian(vec, mu=mean, sigma=std)
    mean_tensor = tensor.convert_to_tensor(mean, dtype=tensor.get_dtype(vec))
    std_tensor = tensor.convert_to_tensor(std, dtype=tensor.get_dtype(vec))
    term = ops.divide(ops.subtract(vec, mean_tensor), std_tensor)
    exponent = ops.multiply(tensor.convert_to_tensor(-0.5, dtype=tensor.get_dtype(vec)), ops.square(term))
    exp_part = ops.exp(exponent)
    pi_tensor = tensor.convert_to_tensor(ops.pi, dtype=tensor.get_dtype(vec))
    two_tensor = tensor.convert_to_tensor(2.0, dtype=tensor.get_dtype(vec))
    sqrt_two_pi = ops.sqrt(ops.multiply(two_tensor, pi_tensor))
    denominator_factor = ops.multiply(std_tensor, sqrt_two_pi)
    inv_denominator = ops.divide(tensor.convert_to_tensor(1.0, dtype=tensor.get_dtype(vec)), ops.add(denominator_factor, tensor.convert_to_tensor(1e-12)))
    expected = ops.multiply(inv_denominator, exp_part)
    assert ops.allclose(result, expected, atol=1e-5), "Gaussian function failed"
    assert tensor.shape(result) == tensor.shape(vec), "Gaussian function shape mismatch"
