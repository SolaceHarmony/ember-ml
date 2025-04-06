import pytest
from ember_ml import ops
from ember_ml.ops import stats # Import the stats submodule
from ember_ml.nn import tensor

# Assume conftest.py provides 'backend' fixture

@pytest.fixture
def stats_tensor(backend):
    """Fixture for statistical tests."""
    ops.set_backend(backend)
    # Using integers first to test type handling, will cast later if needed
    t = tensor.convert_to_tensor([[1, 2, 3], [4, 5, 6]])
    return t

def test_stats_mean(stats_tensor, backend):
    """Tests stats.mean."""
    ops.set_backend(backend)
    t = stats_tensor
    t_float = tensor.cast(t, tensor.float32) # Cast for mean

    # Mean of all elements
    mean_all = stats.mean(t_float)
    # (1+2+3+4+5+6)/6 = 21/6 = 3.5
    assert ops.allclose(mean_all, tensor.convert_to_tensor(3.5)), f"{backend}: Mean all failed"

    # Mean along axis 0 (columns)
    mean_cols = stats.mean(t_float, axis=0)
    # [(1+4)/2, (2+5)/2, (3+6)/2] = [2.5, 3.5, 4.5]
    assert ops.allclose(mean_cols, tensor.convert_to_tensor([2.5, 3.5, 4.5])), f"{backend}: Mean axis=0 failed"

    # Mean along axis 1 (rows)
    mean_rows = stats.mean(t_float, axis=1)
    # [(1+2+3)/3, (4+5+6)/3] = [2.0, 5.0]
    assert ops.allclose(mean_rows, tensor.convert_to_tensor([2.0, 5.0])), f"{backend}: Mean axis=1 failed"

def test_stats_sum(stats_tensor, backend):
    """Tests stats.sum."""
    ops.set_backend(backend)
    t = stats_tensor

    # Sum of all elements
    sum_all = stats.sum(t)
    # 1+2+3+4+5+6 = 21
    assert ops.allclose(sum_all, tensor.convert_to_tensor(21)), f"{backend}: Sum all failed"

    # Sum along axis 0 (columns)
    sum_cols = stats.sum(t, axis=0)
    # [1+4, 2+5, 3+6] = [5, 7, 9]
    assert ops.allclose(sum_cols, tensor.convert_to_tensor([5, 7, 9])), f"{backend}: Sum axis=0 failed"

    # Sum along axis 1 (rows)
    sum_rows = stats.sum(t, axis=1)
    # [1+2+3, 4+5+6] = [6, 15]
    assert ops.allclose(sum_rows, tensor.convert_to_tensor([6, 15])), f"{backend}: Sum axis=1 failed"

def test_stats_var(stats_tensor, backend):
    """Tests stats.var."""
    ops.set_backend(backend)
    t = stats_tensor
    t_float = tensor.cast(t, tensor.float32)

    # Variance of all elements (ddof=0, population variance)
    var_all = stats.var(t_float)
    # mean=3.5, values=[1,2,3,4,5,6]
    # sq_diff = [(-2.5)^2, (-1.5)^2, (-0.5)^2, (0.5)^2, (1.5)^2, (2.5)^2]
    # = [6.25, 2.25, 0.25, 0.25, 2.25, 6.25]
    # sum = 17.5, mean = 17.5 / 6 = 2.91666...
    assert ops.allclose(var_all, tensor.convert_to_tensor(2.916666), atol=1e-5), f"{backend}: Var all (ddof=0) failed"

    # Variance with ddof=1 (sample variance)
    var_all_sample = stats.var(t_float, ddof=1)
    # mean = 17.5 / (6-1) = 17.5 / 5 = 3.5
    assert ops.allclose(var_all_sample, tensor.convert_to_tensor(3.5), atol=1e-5), f"{backend}: Var all (ddof=1) failed"

def test_stats_std(stats_tensor, backend):
    """Tests stats.std."""
    ops.set_backend(backend)
    t = stats_tensor
    t_float = tensor.cast(t, tensor.float32)

    # Std dev of all elements (ddof=0)
    std_all = stats.std(t_float)
    # sqrt(variance) = sqrt(2.916666) approx 1.7078
    assert ops.allclose(std_all, tensor.convert_to_tensor(1.707825), atol=1e-5), f"{backend}: Std all (ddof=0) failed"

    # Std dev with ddof=1
    std_all_sample = stats.std(t_float, ddof=1)
    # sqrt(variance) = sqrt(3.5) approx 1.8708
    assert ops.allclose(std_all_sample, tensor.convert_to_tensor(1.870828), atol=1e-5), f"{backend}: Std all (ddof=1) failed"

def test_stats_min_max(stats_tensor, backend):
    """Tests stats.min and stats.max."""
    ops.set_backend(backend)
    t = stats_tensor

    # Min/Max of all elements
    min_all = stats.min(t)
    max_all = stats.max(t)
    assert ops.allclose(min_all, tensor.convert_to_tensor(1)), f"{backend}: Min all failed"
    assert ops.allclose(max_all, tensor.convert_to_tensor(6)), f"{backend}: Max all failed"

    # Min/Max along axis 0 (columns)
    min_cols = stats.min(t, axis=0)
    max_cols = stats.max(t, axis=0)
    assert ops.allclose(min_cols, tensor.convert_to_tensor([1, 2, 3])), f"{backend}: Min axis=0 failed"
    assert ops.allclose(max_cols, tensor.convert_to_tensor([4, 5, 6])), f"{backend}: Max axis=0 failed"

    # Min/Max along axis 1 (rows)
    min_rows = stats.min(t, axis=1)
    max_rows = stats.max(t, axis=1)
    assert ops.allclose(min_rows, tensor.convert_to_tensor([1, 4])), f"{backend}: Min axis=1 failed"
    assert ops.allclose(max_rows, tensor.convert_to_tensor([3, 6])), f"{backend}: Max axis=1 failed"

def test_stats_median(backend):
    """Tests stats.median."""
    ops.set_backend(backend)
    # Odd number of elements
    t_odd = tensor.convert_to_tensor([1, 5, 2, 8, 7])
    median_odd = stats.median(t_odd) # Sorted: [1, 2, 5, 7, 8] -> Median 5
    assert ops.allclose(median_odd, tensor.convert_to_tensor(5)), f"{backend}: Median odd failed"

    # Even number of elements
    t_even = tensor.convert_to_tensor([1, 5, 2, 8, 7, 3])
    t_even_float = tensor.cast(t_even, tensor.float32) # Cast for mean calculation in median
    median_even = stats.median(t_even_float) # Sorted: [1, 2, 3, 5, 7, 8] -> Median (3+5)/2 = 4
    assert ops.allclose(median_even, tensor.convert_to_tensor(4.0)), f"{backend}: Median even failed"

    # Median along axis
    t_matrix = tensor.convert_to_tensor([[1, 5, 2], [8, 7, 3]])
    t_matrix_float = tensor.cast(t_matrix, tensor.float32)
    median_cols = stats.median(t_matrix_float, axis=0) # Cols: [1,8]->4.5, [5,7]->6, [2,3]->2.5
    assert ops.allclose(median_cols, tensor.convert_to_tensor([4.5, 6.0, 2.5])), f"{backend}: Median axis=0 failed"

def test_stats_cumsum(stats_tensor, backend):
    """Tests stats.cumsum."""
    ops.set_backend(backend)
    t = stats_tensor

    # Cumsum flattened
    cumsum_flat = stats.cumsum(t)
    # [1, 1+2, 1+2+3, 1+2+3+4, 1+2+3+4+5, 1+2+3+4+5+6] = [1, 3, 6, 10, 15, 21]
    assert ops.allclose(cumsum_flat, tensor.convert_to_tensor([1, 3, 6, 10, 15, 21])), f"{backend}: Cumsum flat failed"

    # Cumsum along axis 0
    cumsum_cols = stats.cumsum(t, axis=0)
    # [[1, 2, 3], [1+4, 2+5, 3+6]] = [[1, 2, 3], [5, 7, 9]]
    assert ops.allclose(cumsum_cols, tensor.convert_to_tensor([[1, 2, 3], [5, 7, 9]])), f"{backend}: Cumsum axis=0 failed"

    # Cumsum along axis 1
    cumsum_rows = stats.cumsum(t, axis=1)
    # [[1, 1+2, 1+2+3], [4, 4+5, 4+5+6]] = [[1, 3, 6], [4, 9, 15]]
    assert ops.allclose(cumsum_rows, tensor.convert_to_tensor([[1, 3, 6], [4, 9, 15]])), f"{backend}: Cumsum axis=1 failed"

def test_stats_argmax(backend):
    """Tests stats.argmax."""
    ops.set_backend(backend)
    t = tensor.convert_to_tensor([[1, 5, 3], [4, 2, 6]])

    # Argmax flattened
    argmax_flat = stats.argmax(t) # Max is 6 at index 5
    assert ops.allclose(argmax_flat, tensor.convert_to_tensor(5)), f"{backend}: Argmax flat failed"

    # Argmax along axis 0 (columns)
    argmax_cols = stats.argmax(t, axis=0) # Max indices: [1, 0, 1]
    assert ops.allclose(argmax_cols, tensor.convert_to_tensor([1, 0, 1])), f"{backend}: Argmax axis=0 failed"

    # Argmax along axis 1 (rows)
    argmax_rows = stats.argmax(t, axis=1) # Max indices: [1, 2]
    assert ops.allclose(argmax_rows, tensor.convert_to_tensor([1, 2])), f"{backend}: Argmax axis=1 failed"

def test_stats_sort_argsort(backend):
    """Tests stats.sort and stats.argsort."""
    ops.set_backend(backend)
    t = tensor.convert_to_tensor([[3, 1, 2], [6, 5, 4]])

    # Sort along axis 1 (rows)

def test_stats_percentile(backend):
    """Tests stats.percentile."""
    ops.set_backend(backend)
    t = tensor.arange(1, 11) # Tensor [1, 2, ..., 10]
    t_float = tensor.cast(t, tensor.float32)

    # 0th percentile (min)
    p0 = stats.percentile(t_float, 0)
    assert ops.allclose(p0, tensor.convert_to_tensor(1.0)), f"{backend}: Percentile q=0 failed"

    # 100th percentile (max)
    p100 = stats.percentile(t_float, 100)
    assert ops.allclose(p100, tensor.convert_to_tensor(10.0)), f"{backend}: Percentile q=100 failed"

    # 50th percentile (median)
    p50 = stats.percentile(t_float, 50)
    # For [1..10], median is (5+6)/2 = 5.5
    assert ops.allclose(p50, tensor.convert_to_tensor(5.5)), f"{backend}: Percentile q=50 failed"

    # 25th percentile
    p25 = stats.percentile(t_float, 25)
    # Interpolation might differ. Numpy('linear') gives 3.25
    # Expected value might need adjustment based on backend default interpolation
    # Using a tolerance
    assert ops.allclose(p25, tensor.convert_to_tensor(3.25), atol=0.01), f"{backend}: Percentile q=25 failed"

    # 75th percentile
    p75 = stats.percentile(t_float, 75)
    # Interpolation might differ. Numpy('linear') gives 7.75
    assert ops.allclose(p75, tensor.convert_to_tensor(7.75), atol=0.01), f"{backend}: Percentile q=75 failed"

    # Multiple percentiles
    qs = [10, 90]
    p_multi = stats.percentile(t_float, qs)
    # Numpy('linear'): p10=1.9, p90=9.1
    expected_multi = tensor.convert_to_tensor([1.9, 9.1])
    assert ops.allclose(p_multi, expected_multi, atol=0.01), f"{backend}: Percentile multiple qs failed"
    assert tensor.shape(p_multi) == (len(qs),), f"{backend}: Percentile multiple qs shape failed"


    sorted_rows = stats.sort(t, axis=1)
    expected_sorted = tensor.convert_to_tensor([[1, 2, 3], [4, 5, 6]])
    assert ops.allclose(sorted_rows, expected_sorted), f"{backend}: Sort axis=1 failed"

    # Argsort along axis 1 (rows)
    argsort_rows = stats.argsort(t, axis=1)
    expected_argsort = tensor.convert_to_tensor([[1, 2, 0], [2, 1, 0]]) # Indices to get sorted array
    assert ops.allclose(argsort_rows, expected_argsort), f"{backend}: Argsort axis=1 failed"

    # Sort descending
    sorted_desc = stats.sort(t, axis=1, descending=True)
    expected_sorted_desc = tensor.convert_to_tensor([[3, 2, 1], [6, 5, 4]])
    assert ops.allclose(sorted_desc, expected_sorted_desc), f"{backend}: Sort descending failed"

    # Argsort descending
    argsort_desc = stats.argsort(t, axis=1, descending=True)
    expected_argsort_desc = tensor.convert_to_tensor([[0, 2, 1], [0, 1, 2]])
    assert ops.allclose(argsort_desc, expected_argsort_desc), f"{backend}: Argsort descending failed"

# TODO: Add tests for percentile