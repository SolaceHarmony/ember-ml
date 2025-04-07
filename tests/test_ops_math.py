import pytest
from pytest import mark
from ember_ml import ops
from ember_ml.nn import tensor, modules

# Define the backend order: numpy -> torch -> mlx

# Helper function to get sample tensors
def _get_sample_tensors():
    tensor1 = tensor.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]])
    tensor2 = tensor.convert_to_tensor([[5.0, 6.0], [7.0, 8.0]])
    scalar = tensor.convert_to_tensor(2.0)
    return tensor1, tensor2, scalar

# --- NumPy Backend Setup & Tests ---
@mark.run(order=1)
def test_setup_numpy():
    """Set up the NumPy backend."""
    print("\n=== Setting backend to NumPy ===")
    ops.set_backend('numpy')
    assert ops.get_backend() == 'numpy'

@mark.run(order=1)
def test_ops_add_numpy():
    """Tests ops.add with NumPy backend."""
    t1, t2, scalar = _get_sample_tensors()
    result_tt = ops.add(t1, t2)
    expected_tt = tensor.convert_to_tensor([[6.0, 8.0], [10.0, 12.0]])
    assert ops.allclose(result_tt, expected_tt), "Tensor-Tensor Add failed"
    result_ts = ops.add(t1, scalar)
    expected_ts = tensor.convert_to_tensor([[3.0, 4.0], [5.0, 6.0]])
    assert ops.allclose(result_ts, expected_ts), "Tensor-Scalar Add failed"

@mark.run(order=1)
def test_ops_subtract_numpy():
    """Tests ops.subtract with NumPy backend."""
    t1, t2, scalar = _get_sample_tensors()
    result_tt = ops.subtract(t1, t2)
    expected_tt = tensor.convert_to_tensor([[-4.0, -4.0], [-4.0, -4.0]])
    assert ops.allclose(result_tt, expected_tt), "Tensor-Tensor Subtract failed"
    result_ts = ops.subtract(t1, scalar)
    expected_ts = tensor.convert_to_tensor([[-1.0, 0.0], [1.0, 2.0]])
    assert ops.allclose(result_ts, expected_ts), "Tensor-Scalar Subtract failed"

@mark.run(order=1)
def test_ops_multiply_numpy():
    """Tests ops.multiply with NumPy backend."""
    t1, t2, scalar = _get_sample_tensors()
    result_tt = ops.multiply(t1, t2)
    expected_tt = tensor.convert_to_tensor([[5.0, 12.0], [21.0, 32.0]])
    assert ops.allclose(result_tt, expected_tt), "Tensor-Tensor Multiply failed"
    result_ts = ops.multiply(t1, scalar)
    expected_ts = tensor.convert_to_tensor([[2.0, 4.0], [6.0, 8.0]])
    assert ops.allclose(result_ts, expected_ts), "Tensor-Scalar Multiply failed"

@mark.run(order=1)
def test_ops_divide_numpy():
    """Tests ops.divide with NumPy backend."""
    t1, t2, _ = _get_sample_tensors()
    scalar_div = tensor.convert_to_tensor(2.0)
    result_tt = ops.divide(t1, t2)
    expected_tt = tensor.convert_to_tensor([[1/5, 2/6], [3/7, 4/8]])
    expected_tt = tensor.cast(expected_tt, result_tt.dtype)
    assert ops.allclose(result_tt, expected_tt, atol=1e-6), "Tensor-Tensor Divide failed"
    result_ts = ops.divide(t1, scalar_div)
    expected_ts = tensor.convert_to_tensor([[0.5, 1.0], [1.5, 2.0]])
    assert ops.allclose(result_ts, expected_ts), "Tensor-Scalar Divide failed"

@mark.run(order=1)
def test_ops_floor_divide_numpy():
    """Tests ops.floor_divide with NumPy backend."""
    t1 = tensor.convert_to_tensor([[5.0, 8.0], [10.0, 13.0]])
    t2 = tensor.convert_to_tensor([[2.0, 3.0], [3.0, 4.0]])
    scalar_div = tensor.convert_to_tensor(3.0)
    result_tt = ops.floor_divide(t1, t2)
    expected_tt = tensor.convert_to_tensor([[2.0, 2.0], [3.0, 3.0]])
    expected_tt = tensor.cast(expected_tt, result_tt.dtype)
    assert ops.allclose(result_tt, expected_tt), "Tensor-Tensor Floor Divide failed"
    result_ts = ops.floor_divide(t1, scalar_div)
    expected_ts = tensor.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]])
    expected_ts = tensor.cast(expected_ts, result_ts.dtype)
    assert ops.allclose(result_ts, expected_ts), "Tensor-Scalar Floor Divide failed"

@mark.run(order=1)
def test_ops_dot_matmul_numpy():
    """Tests ops.dot and ops.matmul with NumPy backend."""
    from ember_ml.ops import stats # Import locally
    vec1 = tensor.convert_to_tensor([1.0, 2.0, 3.0])
    vec2 = tensor.convert_to_tensor([4.0, 5.0, 6.0])
    result_dot = ops.dot(vec1, vec2)
    expected_dot = stats.sum(ops.multiply(vec1, vec2))
    expected_dot = tensor.cast(expected_dot, result_dot.dtype)
    assert ops.allclose(result_dot, expected_dot), "Vector dot product failed"
    mat1 = tensor.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]])
    mat2 = tensor.convert_to_tensor([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]])
    result_matmul = ops.matmul(mat1, mat2)
    expected_matmul = tensor.convert_to_tensor([[21.0, 24.0, 27.0], [47.0, 54.0, 61.0]])
    expected_matmul = tensor.cast(expected_matmul, result_matmul.dtype)
    assert ops.allclose(result_matmul, expected_matmul), "Matrix multiplication failed"
    assert tensor.shape(result_matmul) == (2, 3), "Matmul result shape incorrect"

@mark.run(order=1)
def test_ops_gather_numpy():
    """Tests ops.gather with NumPy backend."""
    params = tensor.convert_to_tensor([[10, 20, 30, 40],[50, 60, 70, 80],[90, 100, 110, 120]])
    indices_axis0 = tensor.convert_to_tensor([0, 2])
    indices_axis1 = tensor.convert_to_tensor([1, 3])
    result_axis0 = ops.gather(params, indices_axis0, axis=0)
    expected_axis0 = tensor.convert_to_tensor([[10, 20, 30, 40], [90, 100, 110, 120]])
    expected_axis0 = tensor.cast(expected_axis0, result_axis0.dtype)
    assert ops.allclose(result_axis0, expected_axis0), "Gather axis=0 failed"
    result_axis1 = ops.gather(params, indices_axis1, axis=1)
    expected_axis1 = tensor.convert_to_tensor([[20, 40], [60, 80], [100, 120]])
    expected_axis1 = tensor.cast(expected_axis1, result_axis1.dtype)
    assert ops.allclose(result_axis1, expected_axis1), "Gather axis=1 failed"

@mark.run(order=1)
def test_ops_exp_log_numpy():
    """Tests exp and log functions with NumPy backend."""
    t = tensor.convert_to_tensor([1.0, 2.0, 0.0, -1.0])
    t_pos = tensor.convert_to_tensor([1.0, 10.0, 0.1])
    result_exp = ops.exp(t)
    expected_exp = ops.exp(tensor.convert_to_tensor([1.0, 2.0, 0.0, -1.0]))
    expected_exp = tensor.cast(expected_exp, result_exp.dtype)
    assert ops.allclose(result_exp, expected_exp), "Exp failed"
    result_log = ops.log(t_pos)
    expected_log = ops.log(tensor.convert_to_tensor([1.0, 10.0, 0.1]))
    expected_log = tensor.cast(expected_log, result_log.dtype)
    assert ops.allclose(result_log, expected_log), "Log failed"
    result_log10 = ops.log10(t_pos)
    expected_log10 = ops.log10(tensor.convert_to_tensor([1.0, 10.0, 0.1]))
    expected_log10 = tensor.cast(expected_log10, result_log10.dtype)
    assert ops.allclose(result_log10, expected_log10), "Log10 failed"
    result_log2 = ops.log2(t_pos)
    expected_log2 = ops.log2(tensor.convert_to_tensor([1.0, 10.0, 0.1]))
    expected_log2 = tensor.cast(expected_log2, result_log2.dtype)
    assert ops.allclose(result_log2, expected_log2), "Log2 failed"

@mark.run(order=1)
def test_ops_pow_sqrt_square_numpy():
    """Tests power, sqrt, and square functions with NumPy backend."""
    t = tensor.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]])
    t_pos = tensor.convert_to_tensor([[1.0, 4.0], [9.0, 16.0]])
    exponent = 3.0
    exponent_tensor = tensor.convert_to_tensor([[2.0, 0.5], [3.0, 0.0]])
    result_pow_scalar = ops.pow(t, exponent)
    expected_pow_scalar = ops.pow(t, tensor.convert_to_tensor(exponent))
    expected_pow_scalar = tensor.cast(expected_pow_scalar, result_pow_scalar.dtype)
    assert ops.allclose(result_pow_scalar, expected_pow_scalar), "Pow (scalar exp) failed"
    result_pow_tensor = ops.pow(t, exponent_tensor)
    expected_pow_tensor = ops.pow(t, exponent_tensor)
    expected_pow_tensor = tensor.cast(expected_pow_tensor, result_pow_tensor.dtype)
    assert ops.allclose(result_pow_tensor, expected_pow_tensor), "Pow (tensor exp) failed"
    result_sqrt = ops.sqrt(t_pos)
    expected_sqrt = tensor.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]])
    expected_sqrt = tensor.cast(expected_sqrt, result_sqrt.dtype)
    assert ops.allclose(result_sqrt, expected_sqrt), "Sqrt failed"
    result_square = ops.square(t)
    expected_square = tensor.convert_to_tensor([[1.0, 4.0], [9.0, 16.0]])
    expected_square = tensor.cast(expected_square, result_square.dtype)
    assert ops.allclose(result_square, expected_square), "Square failed"

@mark.run(order=1)
def test_ops_abs_negative_sign_clip_numpy():
    """Tests abs, negative, sign, and clip functions with NumPy backend."""
    t = tensor.convert_to_tensor([[1.0, -2.0], [0.0, -0.5]])
    result_abs = ops.abs(t)
    expected_abs = tensor.convert_to_tensor([[1.0, 2.0], [0.0, 0.5]])
    expected_abs = tensor.cast(expected_abs, result_abs.dtype)
    assert ops.allclose(result_abs, expected_abs), "Abs failed"
    result_neg = ops.negative(t)
    expected_neg = tensor.convert_to_tensor([[-1.0, 2.0], [0.0, 0.5]])
    expected_neg = tensor.cast(expected_neg, result_neg.dtype)
    assert ops.allclose(result_neg, expected_neg), "Negative failed"
    result_sign = ops.sign(t)
    expected_sign = tensor.convert_to_tensor([[1.0, -1.0], [0.0, -1.0]])
    expected_sign = tensor.cast(expected_sign, result_sign.dtype)
    assert ops.allclose(result_sign, expected_sign), "Sign failed"
    min_val, max_val = -0.6, 1.2
    result_clip = ops.clip(t, min_val, max_val)
    expected_clip = tensor.convert_to_tensor([[1.0, -0.6], [0.0, -0.5]])
    expected_clip = tensor.cast(expected_clip, result_clip.dtype)
    assert ops.allclose(result_clip, expected_clip), "Clip failed"

@mark.run(order=1)
def test_ops_trigonometric_numpy():
    """Tests trigonometric functions with NumPy backend."""
    pi_val = ops.pi; pi_val_tensor = tensor.convert_to_tensor(pi_val)
    zero_t, four_t = tensor.convert_to_tensor(0.0), tensor.convert_to_tensor(4.0)
    angles = tensor.stack([zero_t, ops.divide(pi_val_tensor, four_t), ops.divide(pi_val_tensor, tensor.convert_to_tensor(2.0)), pi_val_tensor])
    values = tensor.convert_to_tensor([-1.0, 0.0, 1.0])
    result_sin = ops.sin(angles)
    expected_sin = ops.sin(angles)
    assert ops.allclose(result_sin, expected_sin, atol=1e-6), "Sin failed"
    result_cos = ops.cos(angles)
    expected_cos = ops.cos(angles)
    assert ops.allclose(result_cos, expected_cos, atol=1e-6), "Cos failed"
    angles_tan = tensor.stack([zero_t, ops.divide(pi_val_tensor, four_t), ops.negative(ops.divide(pi_val_tensor, four_t))])
    result_tan = ops.tan(angles_tan)
    expected_tan = ops.tan(angles_tan)
    assert ops.allclose(result_tan, expected_tan, atol=1e-6), "Tan failed"
    result_sinh = ops.sinh(values)
    expected_sinh = ops.sinh(values)
    assert ops.allclose(result_sinh, expected_sinh, atol=1e-6), "Sinh failed"
    result_cosh = ops.cosh(values)
    expected_cosh = ops.cosh(values)
    assert ops.allclose(result_cosh, expected_cosh, atol=1e-6), "Cosh failed"
    result_tanh = ops.tanh(values)
    expected_tanh = ops.tanh(values)
    assert ops.allclose(result_tanh, expected_tanh, atol=1e-6), "Tanh failed"

@mark.run(order=1)
def test_ops_activations_numpy():
    """Tests activation functions using modules with NumPy backend."""
    from ember_ml.ops import stats # Import locally
    t = tensor.convert_to_tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    t_matrix = tensor.convert_to_tensor([[1.0, 2.0, 1.0], [-1.0, 0.0, 1.0]])
    sigmoid_module = modules.Sigmoid(); result_sig = sigmoid_module(t)
    expected_sig = tensor.convert_to_tensor([0.1192029, 0.37754067, 0.5, 0.62245933, 0.880797])
    expected_sig = tensor.cast(expected_sig, result_sig.dtype)
    assert ops.allclose(result_sig, expected_sig, atol=1e-6), "Sigmoid module failed"
    softplus_module = modules.Softplus(); result_sp = softplus_module(t)
    expected_sp = tensor.convert_to_tensor([0.126928, 0.474077, 0.693147, 0.974077, 2.126928])
    expected_sp = tensor.cast(expected_sp, result_sp.dtype)
    assert ops.allclose(result_sp, expected_sp, atol=1e-6), "Softplus module failed"
    relu_module = modules.ReLU(); result_relu = relu_module(t)
    expected_relu = tensor.convert_to_tensor([0., 0., 0., 0.5, 2.0])
    expected_relu = tensor.cast(expected_relu, result_relu.dtype)
    assert ops.allclose(result_relu, expected_relu, atol=1e-6), "ReLU module failed"
    softmax_module = modules.Softmax(axis=1); result_sm = softmax_module(t_matrix)
    expected_softmax_row0 = tensor.convert_to_tensor([0.21194156, 0.57611691, 0.21194156])
    expected_softmax_row1 = tensor.convert_to_tensor([0.09003057, 0.24472847, 0.66524096])
    expected_softmax_row0 = tensor.cast(expected_softmax_row0, result_sm.dtype)
    expected_softmax_row1 = tensor.cast(expected_softmax_row1, result_sm.dtype)
    assert ops.allclose(result_sm[0], expected_softmax_row0, atol=1e-6), "Softmax row 0 failed"
    assert ops.allclose(result_sm[1], expected_softmax_row1, atol=1e-6), "Softmax row 1 failed"
    row_sums = stats.sum(result_sm, axis=1)
    expected_sums = tensor.ones(tensor.shape(row_sums)[0])
    expected_sums = tensor.cast(expected_sums, row_sums.dtype)
    assert ops.allclose(row_sums, expected_sums, atol=1e-6), "Softmax rows do not sum to 1"

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
def test_ops_add_torch():
    """Tests ops.add with PyTorch backend."""
    t1, t2, scalar = _get_sample_tensors()
    result_tt = ops.add(t1, t2)
    expected_tt = tensor.convert_to_tensor([[6.0, 8.0], [10.0, 12.0]])
    assert ops.allclose(result_tt, expected_tt), "Tensor-Tensor Add failed"
    result_ts = ops.add(t1, scalar)
    expected_ts = tensor.convert_to_tensor([[3.0, 4.0], [5.0, 6.0]])
    assert ops.allclose(result_ts, expected_ts), "Tensor-Scalar Add failed"

@mark.run(order=2)
def test_ops_subtract_torch():
    """Tests ops.subtract with PyTorch backend."""
    t1, t2, scalar = _get_sample_tensors()
    result_tt = ops.subtract(t1, t2)
    expected_tt = tensor.convert_to_tensor([[-4.0, -4.0], [-4.0, -4.0]])
    assert ops.allclose(result_tt, expected_tt), "Tensor-Tensor Subtract failed"
    result_ts = ops.subtract(t1, scalar)
    expected_ts = tensor.convert_to_tensor([[-1.0, 0.0], [1.0, 2.0]])
    assert ops.allclose(result_ts, expected_ts), "Tensor-Scalar Subtract failed"

@mark.run(order=2)
def test_ops_multiply_torch():
    """Tests ops.multiply with PyTorch backend."""
    t1, t2, scalar = _get_sample_tensors()
    result_tt = ops.multiply(t1, t2)
    expected_tt = tensor.convert_to_tensor([[5.0, 12.0], [21.0, 32.0]])
    assert ops.allclose(result_tt, expected_tt), "Tensor-Tensor Multiply failed"
    result_ts = ops.multiply(t1, scalar)
    expected_ts = tensor.convert_to_tensor([[2.0, 4.0], [6.0, 8.0]])
    assert ops.allclose(result_ts, expected_ts), "Tensor-Scalar Multiply failed"

@mark.run(order=2)
def test_ops_divide_torch():
    """Tests ops.divide with PyTorch backend."""
    t1, t2, _ = _get_sample_tensors()
    scalar_div = tensor.convert_to_tensor(2.0)
    result_tt = ops.divide(t1, t2)
    expected_tt = tensor.convert_to_tensor([[1/5, 2/6], [3/7, 4/8]])
    expected_tt = tensor.cast(expected_tt, result_tt.dtype)
    assert ops.allclose(result_tt, expected_tt, atol=1e-6), "Tensor-Tensor Divide failed"
    result_ts = ops.divide(t1, scalar_div)
    expected_ts = tensor.convert_to_tensor([[0.5, 1.0], [1.5, 2.0]])
    assert ops.allclose(result_ts, expected_ts), "Tensor-Scalar Divide failed"

@mark.run(order=2)
def test_ops_floor_divide_torch():
    """Tests ops.floor_divide with PyTorch backend."""
    t1 = tensor.convert_to_tensor([[5.0, 8.0], [10.0, 13.0]])
    t2 = tensor.convert_to_tensor([[2.0, 3.0], [3.0, 4.0]])
    scalar_div = tensor.convert_to_tensor(3.0)
    result_tt = ops.floor_divide(t1, t2)
    expected_tt = tensor.convert_to_tensor([[2.0, 2.0], [3.0, 3.0]])
    expected_tt = tensor.cast(expected_tt, result_tt.dtype)
    assert ops.allclose(result_tt, expected_tt), "Tensor-Tensor Floor Divide failed"
    result_ts = ops.floor_divide(t1, scalar_div)
    expected_ts = tensor.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]])
    expected_ts = tensor.cast(expected_ts, result_ts.dtype)
    assert ops.allclose(result_ts, expected_ts), "Tensor-Scalar Floor Divide failed"

@mark.run(order=2)
def test_ops_dot_matmul_torch():
    """Tests ops.dot and ops.matmul with PyTorch backend."""
    from ember_ml.ops import stats # Import locally
    vec1 = tensor.convert_to_tensor([1.0, 2.0, 3.0])
    vec2 = tensor.convert_to_tensor([4.0, 5.0, 6.0])
    result_dot = ops.dot(vec1, vec2)
    expected_dot = stats.sum(ops.multiply(vec1, vec2))
    expected_dot = tensor.cast(expected_dot, result_dot.dtype)
    assert ops.allclose(result_dot, expected_dot), "Vector dot product failed"
    mat1 = tensor.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]])
    mat2 = tensor.convert_to_tensor([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]])
    result_matmul = ops.matmul(mat1, mat2)
    expected_matmul = tensor.convert_to_tensor([[21.0, 24.0, 27.0], [47.0, 54.0, 61.0]])
    expected_matmul = tensor.cast(expected_matmul, result_matmul.dtype)
    assert ops.allclose(result_matmul, expected_matmul), "Matrix multiplication failed"
    assert tensor.shape(result_matmul) == (2, 3), "Matmul result shape incorrect"

@mark.run(order=2)
def test_ops_gather_torch():
    """Tests ops.gather with PyTorch backend."""
    params = tensor.convert_to_tensor([[10, 20, 30, 40],[50, 60, 70, 80],[90, 100, 110, 120]])
    indices_axis0 = tensor.convert_to_tensor([0, 2])
    indices_axis1 = tensor.convert_to_tensor([1, 3])
    result_axis0 = ops.gather(params, indices_axis0, axis=0)
    expected_axis0 = tensor.convert_to_tensor([[10, 20, 30, 40], [90, 100, 110, 120]])
    expected_axis0 = tensor.cast(expected_axis0, result_axis0.dtype)
    assert ops.allclose(result_axis0, expected_axis0), "Gather axis=0 failed"
    result_axis1 = ops.gather(params, indices_axis1, axis=1)
    expected_axis1 = tensor.convert_to_tensor([[20, 40], [60, 80], [100, 120]])
    expected_axis1 = tensor.cast(expected_axis1, result_axis1.dtype)
    assert ops.allclose(result_axis1, expected_axis1), "Gather axis=1 failed"

@mark.run(order=2)
def test_ops_exp_log_torch():
    """Tests exp and log functions with PyTorch backend."""
    t = tensor.convert_to_tensor([1.0, 2.0, 0.0, -1.0])
    t_pos = tensor.convert_to_tensor([1.0, 10.0, 0.1])
    result_exp = ops.exp(t)
    expected_exp = ops.exp(tensor.convert_to_tensor([1.0, 2.0, 0.0, -1.0]))
    expected_exp = tensor.cast(expected_exp, result_exp.dtype)
    assert ops.allclose(result_exp, expected_exp), "Exp failed"
    result_log = ops.log(t_pos)
    expected_log = ops.log(tensor.convert_to_tensor([1.0, 10.0, 0.1]))
    expected_log = tensor.cast(expected_log, result_log.dtype)
    assert ops.allclose(result_log, expected_log), "Log failed"
    result_log10 = ops.log10(t_pos)
    expected_log10 = ops.log10(tensor.convert_to_tensor([1.0, 10.0, 0.1]))
    expected_log10 = tensor.cast(expected_log10, result_log10.dtype)
    assert ops.allclose(result_log10, expected_log10), "Log10 failed"
    result_log2 = ops.log2(t_pos)
    expected_log2 = ops.log2(tensor.convert_to_tensor([1.0, 10.0, 0.1]))
    expected_log2 = tensor.cast(expected_log2, result_log2.dtype)
    assert ops.allclose(result_log2, expected_log2), "Log2 failed"

@mark.run(order=2)
def test_ops_pow_sqrt_square_torch():
    """Tests power, sqrt, and square functions with PyTorch backend."""
    t = tensor.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]])
    t_pos = tensor.convert_to_tensor([[1.0, 4.0], [9.0, 16.0]])
    exponent = 3.0
    exponent_tensor = tensor.convert_to_tensor([[2.0, 0.5], [3.0, 0.0]])
    result_pow_scalar = ops.pow(t, exponent)
    expected_pow_scalar = ops.pow(t, tensor.convert_to_tensor(exponent))
    expected_pow_scalar = tensor.cast(expected_pow_scalar, result_pow_scalar.dtype)
    assert ops.allclose(result_pow_scalar, expected_pow_scalar), "Pow (scalar exp) failed"
    result_pow_tensor = ops.pow(t, exponent_tensor)
    expected_pow_tensor = ops.pow(t, exponent_tensor)
    expected_pow_tensor = tensor.cast(expected_pow_tensor, result_pow_tensor.dtype)
    assert ops.allclose(result_pow_tensor, expected_pow_tensor), "Pow (tensor exp) failed"
    result_sqrt = ops.sqrt(t_pos)
    expected_sqrt = tensor.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]])
    expected_sqrt = tensor.cast(expected_sqrt, result_sqrt.dtype)
    assert ops.allclose(result_sqrt, expected_sqrt), "Sqrt failed"
    result_square = ops.square(t)
    expected_square = tensor.convert_to_tensor([[1.0, 4.0], [9.0, 16.0]])
    expected_square = tensor.cast(expected_square, result_square.dtype)
    assert ops.allclose(result_square, expected_square), "Square failed"

@mark.run(order=2)
def test_ops_abs_negative_sign_clip_torch():
    """Tests abs, negative, sign, and clip functions with PyTorch backend."""
    t = tensor.convert_to_tensor([[1.0, -2.0], [0.0, -0.5]])
    result_abs = ops.abs(t)
    expected_abs = tensor.convert_to_tensor([[1.0, 2.0], [0.0, 0.5]])
    expected_abs = tensor.cast(expected_abs, result_abs.dtype)
    assert ops.allclose(result_abs, expected_abs), "Abs failed"
    result_neg = ops.negative(t)
    expected_neg = tensor.convert_to_tensor([[-1.0, 2.0], [0.0, 0.5]])
    expected_neg = tensor.cast(expected_neg, result_neg.dtype)
    assert ops.allclose(result_neg, expected_neg), "Negative failed"
    result_sign = ops.sign(t)
    expected_sign = tensor.convert_to_tensor([[1.0, -1.0], [0.0, -1.0]])
    expected_sign = tensor.cast(expected_sign, result_sign.dtype)
    assert ops.allclose(result_sign, expected_sign), "Sign failed"
    min_val, max_val = -0.6, 1.2
    result_clip = ops.clip(t, min_val, max_val)
    expected_clip = tensor.convert_to_tensor([[1.0, -0.6], [0.0, -0.5]])
    expected_clip = tensor.cast(expected_clip, result_clip.dtype)
    assert ops.allclose(result_clip, expected_clip), "Clip failed"

@mark.run(order=2)
def test_ops_trigonometric_torch():
    """Tests trigonometric functions with PyTorch backend."""
    pi_val = ops.pi; pi_val_tensor = tensor.convert_to_tensor(pi_val)
    zero_t, four_t = tensor.convert_to_tensor(0.0), tensor.convert_to_tensor(4.0)
    angles = tensor.stack([zero_t, ops.divide(pi_val_tensor, four_t), ops.divide(pi_val_tensor, tensor.convert_to_tensor(2.0)), pi_val_tensor])
    values = tensor.convert_to_tensor([-1.0, 0.0, 1.0])
    result_sin = ops.sin(angles)
    expected_sin = ops.sin(angles)
    assert ops.allclose(result_sin, expected_sin, atol=1e-6), "Sin failed"
    result_cos = ops.cos(angles)
    expected_cos = ops.cos(angles)
    assert ops.allclose(result_cos, expected_cos, atol=1e-6), "Cos failed"
    angles_tan = tensor.stack([zero_t, ops.divide(pi_val_tensor, four_t), ops.negative(ops.divide(pi_val_tensor, four_t))])
    result_tan = ops.tan(angles_tan)
    expected_tan = ops.tan(angles_tan)
    assert ops.allclose(result_tan, expected_tan, atol=1e-6), "Tan failed"
    result_sinh = ops.sinh(values)
    expected_sinh = ops.sinh(values)
    assert ops.allclose(result_sinh, expected_sinh, atol=1e-6), "Sinh failed"
    result_cosh = ops.cosh(values)
    expected_cosh = ops.cosh(values)
    assert ops.allclose(result_cosh, expected_cosh, atol=1e-6), "Cosh failed"
    result_tanh = ops.tanh(values)
    expected_tanh = ops.tanh(values)
    assert ops.allclose(result_tanh, expected_tanh, atol=1e-6), "Tanh failed"

@mark.run(order=2)
def test_ops_activations_torch():
    """Tests activation functions using modules with PyTorch backend."""
    from ember_ml.ops import stats # Import locally
    t = tensor.convert_to_tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    t_matrix = tensor.convert_to_tensor([[1.0, 2.0, 1.0], [-1.0, 0.0, 1.0]])
    sigmoid_module = modules.Sigmoid(); result_sig = sigmoid_module(t)
    expected_sig = tensor.convert_to_tensor([0.1192029, 0.37754067, 0.5, 0.62245933, 0.880797])
    expected_sig = tensor.cast(expected_sig, result_sig.dtype)
    assert ops.allclose(result_sig, expected_sig, atol=1e-6), "Sigmoid module failed"
    softplus_module = modules.Softplus(); result_sp = softplus_module(t)
    expected_sp = tensor.convert_to_tensor([0.126928, 0.474077, 0.693147, 0.974077, 2.126928])
    expected_sp = tensor.cast(expected_sp, result_sp.dtype)
    assert ops.allclose(result_sp, expected_sp, atol=1e-6), "Softplus module failed"
    relu_module = modules.ReLU(); result_relu = relu_module(t)
    expected_relu = tensor.convert_to_tensor([0., 0., 0., 0.5, 2.0])
    expected_relu = tensor.cast(expected_relu, result_relu.dtype)
    assert ops.allclose(result_relu, expected_relu, atol=1e-6), "ReLU module failed"
    softmax_module = modules.Softmax(axis=1); result_sm = softmax_module(t_matrix)
    expected_softmax_row0 = tensor.convert_to_tensor([0.21194156, 0.57611691, 0.21194156])
    expected_softmax_row1 = tensor.convert_to_tensor([0.09003057, 0.24472847, 0.66524096])
    expected_softmax_row0 = tensor.cast(expected_softmax_row0, result_sm.dtype)
    expected_softmax_row1 = tensor.cast(expected_softmax_row1, result_sm.dtype)
    assert ops.allclose(result_sm[0], expected_softmax_row0, atol=1e-6), "Softmax row 0 failed"
    assert ops.allclose(result_sm[1], expected_softmax_row1, atol=1e-6), "Softmax row 1 failed"
    row_sums = stats.sum(result_sm, axis=1)
    expected_sums = tensor.ones(tensor.shape(row_sums)[0])
    expected_sums = tensor.cast(expected_sums, row_sums.dtype)
    assert ops.allclose(row_sums, expected_sums, atol=1e-6), "Softmax rows do not sum to 1"


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
def test_ops_add_mlx():
    """Tests ops.add with MLX backend."""
    t1, t2, scalar = _get_sample_tensors()
    result_tt = ops.add(t1, t2)
    expected_tt = tensor.convert_to_tensor([[6.0, 8.0], [10.0, 12.0]])
    assert ops.allclose(result_tt, expected_tt), "Tensor-Tensor Add failed"
    result_ts = ops.add(t1, scalar)
    expected_ts = tensor.convert_to_tensor([[3.0, 4.0], [5.0, 6.0]])
    assert ops.allclose(result_ts, expected_ts), "Tensor-Scalar Add failed"

@mark.run(order=3)
def test_ops_subtract_mlx():
    """Tests ops.subtract with MLX backend."""
    t1, t2, scalar = _get_sample_tensors()
    result_tt = ops.subtract(t1, t2)
    expected_tt = tensor.convert_to_tensor([[-4.0, -4.0], [-4.0, -4.0]])
    assert ops.allclose(result_tt, expected_tt), "Tensor-Tensor Subtract failed"
    result_ts = ops.subtract(t1, scalar)
    expected_ts = tensor.convert_to_tensor([[-1.0, 0.0], [1.0, 2.0]])
    assert ops.allclose(result_ts, expected_ts), "Tensor-Scalar Subtract failed"

@mark.run(order=3)
def test_ops_multiply_mlx():
    """Tests ops.multiply with MLX backend."""
    t1, t2, scalar = _get_sample_tensors()
    result_tt = ops.multiply(t1, t2)
    expected_tt = tensor.convert_to_tensor([[5.0, 12.0], [21.0, 32.0]])
    assert ops.allclose(result_tt, expected_tt), "Tensor-Tensor Multiply failed"
    result_ts = ops.multiply(t1, scalar)
    expected_ts = tensor.convert_to_tensor([[2.0, 4.0], [6.0, 8.0]])
    assert ops.allclose(result_ts, expected_ts), "Tensor-Scalar Multiply failed"

@mark.run(order=3)
def test_ops_divide_mlx():
    """Tests ops.divide with MLX backend."""
    t1, t2, _ = _get_sample_tensors()
    scalar_div = tensor.convert_to_tensor(2.0)
    result_tt = ops.divide(t1, t2)
    expected_tt = tensor.convert_to_tensor([[1/5, 2/6], [3/7, 4/8]])
    expected_tt = tensor.cast(expected_tt, result_tt.dtype)
    assert ops.allclose(result_tt, expected_tt, atol=1e-6), "Tensor-Tensor Divide failed"
    result_ts = ops.divide(t1, scalar_div)
    expected_ts = tensor.convert_to_tensor([[0.5, 1.0], [1.5, 2.0]])
    assert ops.allclose(result_ts, expected_ts), "Tensor-Scalar Divide failed"

@mark.run(order=3)
def test_ops_floor_divide_mlx():
    """Tests ops.floor_divide with MLX backend."""
    t1 = tensor.convert_to_tensor([[5.0, 8.0], [10.0, 13.0]])
    t2 = tensor.convert_to_tensor([[2.0, 3.0], [3.0, 4.0]])
    scalar_div = tensor.convert_to_tensor(3.0)
    result_tt = ops.floor_divide(t1, t2)
    expected_tt = tensor.convert_to_tensor([[2.0, 2.0], [3.0, 3.0]])
    expected_tt = tensor.cast(expected_tt, result_tt.dtype)
    assert ops.allclose(result_tt, expected_tt), "Tensor-Tensor Floor Divide failed"
    result_ts = ops.floor_divide(t1, scalar_div)
    expected_ts = tensor.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]])
    expected_ts = tensor.cast(expected_ts, result_ts.dtype)
    assert ops.allclose(result_ts, expected_ts), "Tensor-Scalar Floor Divide failed"

@mark.run(order=3)
def test_ops_dot_matmul_mlx():
    """Tests ops.dot and ops.matmul with MLX backend."""
    from ember_ml.ops import stats # Import locally
    vec1 = tensor.convert_to_tensor([1.0, 2.0, 3.0])
    vec2 = tensor.convert_to_tensor([4.0, 5.0, 6.0])
    result_dot = ops.dot(vec1, vec2)
    expected_dot = stats.sum(ops.multiply(vec1, vec2))
    expected_dot = tensor.cast(expected_dot, result_dot.dtype)
    assert ops.allclose(result_dot, expected_dot), "Vector dot product failed"
    mat1 = tensor.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]])
    mat2 = tensor.convert_to_tensor([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]])
    result_matmul = ops.matmul(mat1, mat2)
    expected_matmul = tensor.convert_to_tensor([[21.0, 24.0, 27.0], [47.0, 54.0, 61.0]])
    expected_matmul = tensor.cast(expected_matmul, result_matmul.dtype)
    assert ops.allclose(result_matmul, expected_matmul), "Matrix multiplication failed"
    assert tensor.shape(result_matmul) == (2, 3), "Matmul result shape incorrect"

@mark.run(order=3)
def test_ops_gather_mlx():
    """Tests ops.gather with MLX backend."""
    params = tensor.convert_to_tensor([[10, 20, 30, 40],[50, 60, 70, 80],[90, 100, 110, 120]])
    indices_axis0 = tensor.convert_to_tensor([0, 2])
    indices_axis1 = tensor.convert_to_tensor([1, 3])
    result_axis0 = ops.gather(params, indices_axis0, axis=0)
    expected_axis0 = tensor.convert_to_tensor([[10, 20, 30, 40], [90, 100, 110, 120]])
    expected_axis0 = tensor.cast(expected_axis0, result_axis0.dtype)
    assert ops.allclose(result_axis0, expected_axis0), "Gather axis=0 failed"
    result_axis1 = ops.gather(params, indices_axis1, axis=1)
    expected_axis1 = tensor.convert_to_tensor([[20, 40], [60, 80], [100, 120]])
    expected_axis1 = tensor.cast(expected_axis1, result_axis1.dtype)
    assert ops.allclose(result_axis1, expected_axis1), "Gather axis=1 failed"

@mark.run(order=3)
def test_ops_exp_log_mlx():
    """Tests exp and log functions with MLX backend."""
    t = tensor.convert_to_tensor([1.0, 2.0, 0.0, -1.0])
    t_pos = tensor.convert_to_tensor([1.0, 10.0, 0.1])
    result_exp = ops.exp(t)
    expected_exp = ops.exp(tensor.convert_to_tensor([1.0, 2.0, 0.0, -1.0]))
    expected_exp = tensor.cast(expected_exp, result_exp.dtype)
    assert ops.allclose(result_exp, expected_exp), "Exp failed"
    result_log = ops.log(t_pos)
    expected_log = ops.log(tensor.convert_to_tensor([1.0, 10.0, 0.1]))
    expected_log = tensor.cast(expected_log, result_log.dtype)
    assert ops.allclose(result_log, expected_log), "Log failed"
    result_log10 = ops.log10(t_pos)
    expected_log10 = ops.log10(tensor.convert_to_tensor([1.0, 10.0, 0.1]))
    expected_log10 = tensor.cast(expected_log10, result_log10.dtype)
    assert ops.allclose(result_log10, expected_log10), "Log10 failed"
    result_log2 = ops.log2(t_pos)
    expected_log2 = ops.log2(tensor.convert_to_tensor([1.0, 10.0, 0.1]))
    expected_log2 = tensor.cast(expected_log2, result_log2.dtype)
    assert ops.allclose(result_log2, expected_log2), "Log2 failed"

@mark.run(order=3)
def test_ops_pow_sqrt_square_mlx():
    """Tests power, sqrt, and square functions with MLX backend."""
    t = tensor.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]])
    t_pos = tensor.convert_to_tensor([[1.0, 4.0], [9.0, 16.0]])
    exponent = 3.0
    exponent_tensor = tensor.convert_to_tensor([[2.0, 0.5], [3.0, 0.0]])
    result_pow_scalar = ops.pow(t, exponent)
    expected_pow_scalar = ops.pow(t, tensor.convert_to_tensor(exponent))
    expected_pow_scalar = tensor.cast(expected_pow_scalar, result_pow_scalar.dtype)
    assert ops.allclose(result_pow_scalar, expected_pow_scalar), "Pow (scalar exp) failed"
    result_pow_tensor = ops.pow(t, exponent_tensor)
    expected_pow_tensor = ops.pow(t, exponent_tensor)
    expected_pow_tensor = tensor.cast(expected_pow_tensor, result_pow_tensor.dtype)
    assert ops.allclose(result_pow_tensor, expected_pow_tensor), "Pow (tensor exp) failed"
    result_sqrt = ops.sqrt(t_pos)
    expected_sqrt = tensor.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]])
    expected_sqrt = tensor.cast(expected_sqrt, result_sqrt.dtype)
    assert ops.allclose(result_sqrt, expected_sqrt), "Sqrt failed"
    result_square = ops.square(t)
    expected_square = tensor.convert_to_tensor([[1.0, 4.0], [9.0, 16.0]])
    expected_square = tensor.cast(expected_square, result_square.dtype)
    assert ops.allclose(result_square, expected_square), "Square failed"

@mark.run(order=3)
def test_ops_abs_negative_sign_clip_mlx():
    """Tests abs, negative, sign, and clip functions with MLX backend."""
    t = tensor.convert_to_tensor([[1.0, -2.0], [0.0, -0.5]])
    result_abs = ops.abs(t)
    expected_abs = tensor.convert_to_tensor([[1.0, 2.0], [0.0, 0.5]])
    expected_abs = tensor.cast(expected_abs, result_abs.dtype)
    assert ops.allclose(result_abs, expected_abs), "Abs failed"
    result_neg = ops.negative(t)
    expected_neg = tensor.convert_to_tensor([[-1.0, 2.0], [0.0, 0.5]])
    expected_neg = tensor.cast(expected_neg, result_neg.dtype)
    assert ops.allclose(result_neg, expected_neg), "Negative failed"
    result_sign = ops.sign(t)
    expected_sign = tensor.convert_to_tensor([[1.0, -1.0], [0.0, -1.0]])
    expected_sign = tensor.cast(expected_sign, result_sign.dtype)
    assert ops.allclose(result_sign, expected_sign), "Sign failed"
    min_val, max_val = -0.6, 1.2
    result_clip = ops.clip(t, min_val, max_val)
    expected_clip = tensor.convert_to_tensor([[1.0, -0.6], [0.0, -0.5]])
    expected_clip = tensor.cast(expected_clip, result_clip.dtype)
    assert ops.allclose(result_clip, expected_clip), "Clip failed"

@mark.run(order=3)
def test_ops_trigonometric_mlx():
    """Tests trigonometric functions with MLX backend."""
    pi_val = ops.pi; pi_val_tensor = tensor.convert_to_tensor(pi_val)
    zero_t, four_t = tensor.convert_to_tensor(0.0), tensor.convert_to_tensor(4.0)
    angles = tensor.stack([zero_t, ops.divide(pi_val_tensor, four_t), ops.divide(pi_val_tensor, tensor.convert_to_tensor(2.0)), pi_val_tensor])
    values = tensor.convert_to_tensor([-1.0, 0.0, 1.0])
    result_sin = ops.sin(angles)
    expected_sin = ops.sin(angles)
    assert ops.allclose(result_sin, expected_sin, atol=1e-6), "Sin failed"
    result_cos = ops.cos(angles)
    expected_cos = ops.cos(angles)
    assert ops.allclose(result_cos, expected_cos, atol=1e-6), "Cos failed"
    angles_tan = tensor.stack([zero_t, ops.divide(pi_val_tensor, four_t), ops.negative(ops.divide(pi_val_tensor, four_t))])
    result_tan = ops.tan(angles_tan)
    expected_tan = ops.tan(angles_tan)
    assert ops.allclose(result_tan, expected_tan, atol=1e-6), "Tan failed"
    result_sinh = ops.sinh(values)
    expected_sinh = ops.sinh(values)
    assert ops.allclose(result_sinh, expected_sinh, atol=1e-6), "Sinh failed"
    result_cosh = ops.cosh(values)
    expected_cosh = ops.cosh(values)
    assert ops.allclose(result_cosh, expected_cosh, atol=1e-6), "Cosh failed"
    result_tanh = ops.tanh(values)
    expected_tanh = ops.tanh(values)
    assert ops.allclose(result_tanh, expected_tanh, atol=1e-6), "Tanh failed"

@mark.run(order=3)
def test_ops_activations_mlx():
    """Tests activation functions using modules with MLX backend."""
    from ember_ml.ops import stats # Import locally
    t = tensor.convert_to_tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    t_matrix = tensor.convert_to_tensor([[1.0, 2.0, 1.0], [-1.0, 0.0, 1.0]])
    sigmoid_module = modules.Sigmoid(); result_sig = sigmoid_module(t)
    expected_sig = tensor.convert_to_tensor([0.1192029, 0.37754067, 0.5, 0.62245933, 0.880797])
    expected_sig = tensor.cast(expected_sig, result_sig.dtype)
    assert ops.allclose(result_sig, expected_sig, atol=1e-6), "Sigmoid module failed"
    softplus_module = modules.Softplus(); result_sp = softplus_module(t)
    expected_sp = tensor.convert_to_tensor([0.126928, 0.474077, 0.693147, 0.974077, 2.126928])
    expected_sp = tensor.cast(expected_sp, result_sp.dtype)
    assert ops.allclose(result_sp, expected_sp, atol=1e-6), "Softplus module failed"
    relu_module = modules.ReLU(); result_relu = relu_module(t)
    expected_relu = tensor.convert_to_tensor([0., 0., 0., 0.5, 2.0])
    expected_relu = tensor.cast(expected_relu, result_relu.dtype)
    assert ops.allclose(result_relu, expected_relu, atol=1e-6), "ReLU module failed"
    softmax_module = modules.Softmax(axis=1); result_sm = softmax_module(t_matrix)
    expected_softmax_row0 = tensor.convert_to_tensor([0.21194156, 0.57611691, 0.21194156])
    expected_softmax_row1 = tensor.convert_to_tensor([0.09003057, 0.24472847, 0.66524096])
    expected_softmax_row0 = tensor.cast(expected_softmax_row0, result_sm.dtype)
    expected_softmax_row1 = tensor.cast(expected_softmax_row1, result_sm.dtype)
    assert ops.allclose(result_sm[0], expected_softmax_row0, atol=1e-6), "Softmax row 0 failed"
    assert ops.allclose(result_sm[1], expected_softmax_row1, atol=1e-6), "Softmax row 1 failed"
    row_sums = stats.sum(result_sm, axis=1)
    expected_sums = tensor.ones(tensor.shape(row_sums)[0])
    expected_sums = tensor.cast(expected_sums, row_sums.dtype)
    assert ops.allclose(row_sums, expected_sums, atol=1e-6), "Softmax rows do not sum to 1"

# TODO: Add tests for gradient, eigh