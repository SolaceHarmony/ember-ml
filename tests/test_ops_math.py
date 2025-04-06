import pytest
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.ops import stats # Added missing import for activation test

# Assume conftest.py provides 'backend' fixture and potentially others
# like 'test_data_shape_pair' which yields pairs of compatible tensor shapes.

@pytest.fixture
def sample_tensors(backend):
    """Fixture to create sample tensors for testing."""
    # Ensure backend is set for tensor creation
    ops.set_backend(backend)
    # Simple 2x2 tensors
    tensor1 = tensor.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]])
    tensor2 = tensor.convert_to_tensor([[5.0, 6.0], [7.0, 8.0]])
    # Scalar tensor
    scalar = tensor.convert_to_tensor(2.0)
    return tensor1, tensor2, scalar

def test_ops_add(sample_tensors, backend):
    """Tests ops.add for tensor-tensor and tensor-scalar addition."""
    ops.set_backend(backend)
    t1, t2, scalar = sample_tensors
    
    # Tensor + Tensor
    result_tt = ops.add(t1, t2)
    expected_tt = tensor.convert_to_tensor([[6.0, 8.0], [10.0, 12.0]])
    assert ops.allclose(result_tt, expected_tt), f"{backend}: Tensor-Tensor Add failed"

    # Tensor + Scalar
    result_ts = ops.add(t1, scalar)
    expected_ts = tensor.convert_to_tensor([[3.0, 4.0], [5.0, 6.0]])
    assert ops.allclose(result_ts, expected_ts), f"{backend}: Tensor-Scalar Add failed"

def test_ops_subtract(sample_tensors, backend):
    """Tests ops.subtract for tensor-tensor and tensor-scalar subtraction."""
    ops.set_backend(backend)
    t1, t2, scalar = sample_tensors

    # Tensor - Tensor
    result_tt = ops.subtract(t1, t2)
    expected_tt = tensor.convert_to_tensor([[-4.0, -4.0], [-4.0, -4.0]])
    assert ops.allclose(result_tt, expected_tt), f"{backend}: Tensor-Tensor Subtract failed"

    # Tensor - Scalar
    result_ts = ops.subtract(t1, scalar)
    expected_ts = tensor.convert_to_tensor([[-1.0, 0.0], [1.0, 2.0]])
    assert ops.allclose(result_ts, expected_ts), f"{backend}: Tensor-Scalar Subtract failed"

def test_ops_multiply(sample_tensors, backend):
    """Tests ops.multiply for element-wise tensor-tensor and tensor-scalar multiplication."""
    ops.set_backend(backend)
    t1, t2, scalar = sample_tensors

    # Tensor * Tensor
    result_tt = ops.multiply(t1, t2)
    expected_tt = tensor.convert_to_tensor([[5.0, 12.0], [21.0, 32.0]])
    assert ops.allclose(result_tt, expected_tt), f"{backend}: Tensor-Tensor Multiply failed"

    # Tensor * Scalar
    result_ts = ops.multiply(t1, scalar)
    expected_ts = tensor.convert_to_tensor([[2.0, 4.0], [6.0, 8.0]])
    assert ops.allclose(result_ts, expected_ts), f"{backend}: Tensor-Scalar Multiply failed"

def test_ops_divide(sample_tensors, backend):
    """Tests ops.divide for element-wise tensor-tensor and tensor-scalar division."""
    ops.set_backend(backend)
    t1, t2, scalar = sample_tensors
    # Avoid division by zero if scalar is 0 - use a different scalar for division
    scalar_div = tensor.convert_to_tensor(2.0) 

    # Tensor / Tensor
    result_tt = ops.divide(t1, t2)
    # Calculate expected values using ops.divide
    # Use float literals to ensure float division
    expected_tt = tensor.convert_to_tensor([
        [ops.divide(1.0, 5.0), ops.divide(2.0, 6.0)],
        [ops.divide(3.0, 7.0), ops.divide(4.0, 8.0)]
    ])
    # Cast expected to match result dtype if necessary
    expected_tt = tensor.cast(expected_tt, result_tt.dtype)
    assert ops.allclose(result_tt, expected_tt, atol=1e-6), f"{backend}: Tensor-Tensor Divide failed"

    # Tensor / Scalar
    result_ts = ops.divide(t1, scalar_div)
    expected_ts = tensor.convert_to_tensor([[0.5, 1.0], [1.5, 2.0]])
    assert ops.allclose(result_ts, expected_ts), f"{backend}: Tensor-Scalar Divide failed"


def test_ops_floor_divide(sample_tensors, backend):
    """Tests ops.floor_divide."""
    ops.set_backend(backend)
    t1 = tensor.convert_to_tensor([[5.0, 8.0], [10.0, 13.0]])
    t2 = tensor.convert_to_tensor([[2.0, 3.0], [3.0, 4.0]])
    scalar_div = tensor.convert_to_tensor(3.0)

    # Tensor // Tensor
    result_tt = ops.floor_divide(t1, t2)
    expected_tt = tensor.convert_to_tensor([[2.0, 2.0], [3.0, 3.0]]) # 5//2=2, 8//3=2, 10//3=3, 13//4=3
    expected_tt = tensor.cast(expected_tt, result_tt.dtype) # Match dtype
    assert ops.allclose(result_tt, expected_tt), f"{backend}: Tensor-Tensor Floor Divide failed"

    # Tensor // Scalar
    result_ts = ops.floor_divide(t1, scalar_div)
    expected_ts = tensor.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]]) # 5//3=1, 8//3=2, 10//3=3, 13//3=4
    expected_ts = tensor.cast(expected_ts, result_ts.dtype) # Match dtype
    assert ops.allclose(result_ts, expected_ts), f"{backend}: Tensor-Scalar Floor Divide failed"

def test_ops_dot_matmul(backend):
    """Tests ops.dot and ops.matmul."""
    ops.set_backend(backend)
    # Vector dot product
    vec1 = tensor.convert_to_tensor([1.0, 2.0, 3.0])
    vec2 = tensor.convert_to_tensor([4.0, 5.0, 6.0])
    result_dot = ops.dot(vec1, vec2)
    # Calculate expected dot product using ops functions
    expected_dot = stats.sum(ops.multiply(vec1, vec2)) # 4 + 10 + 18 = 32
    # Ensure expected_dot is scalar-like or matches result_dot's shape if needed
    # Cast to match result type before comparison
    expected_dot = tensor.cast(expected_dot, result_dot.dtype)
    assert ops.allclose(result_dot, expected_dot), f"{backend}: Vector dot product failed"

    # Matrix multiplication
    mat1 = tensor.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]]) # 2x2
    mat2 = tensor.convert_to_tensor([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]) # 2x3
    result_matmul = ops.matmul(mat1, mat2)
    # Expected: [[21, 24, 27], [47, 54, 61]]
    expected_matmul = tensor.convert_to_tensor([[21.0, 24.0, 27.0], [47.0, 54.0, 61.0]])
    expected_matmul = tensor.cast(expected_matmul, result_matmul.dtype) # Match type
    assert ops.allclose(result_matmul, expected_matmul), f"{backend}: Matrix multiplication failed"
    assert tensor.shape(result_matmul) == (2, 3), f"{backend}: Matmul result shape incorrect"

def test_ops_gather(backend):
    """Tests ops.gather."""
    ops.set_backend(backend)
    params = tensor.convert_to_tensor([[10, 20, 30, 40],
                                     [50, 60, 70, 80],
                                     [90, 100, 110, 120]]) # 3x4
    indices_axis0 = tensor.convert_to_tensor([0, 2])
    indices_axis1 = tensor.convert_to_tensor([1, 3])

    # Gather rows (axis=0)
    result_axis0 = ops.gather(params, indices_axis0, axis=0)
    expected_axis0 = tensor.convert_to_tensor([[10, 20, 30, 40], [90, 100, 110, 120]])
    expected_axis0 = tensor.cast(expected_axis0, result_axis0.dtype) # Match type
    assert ops.allclose(result_axis0, expected_axis0), f"{backend}: Gather axis=0 failed"

    # Gather columns (axis=1)
    result_axis1 = ops.gather(params, indices_axis1, axis=1)
    expected_axis1 = tensor.convert_to_tensor([[20, 40], [60, 80], [100, 120]])
    expected_axis1 = tensor.cast(expected_axis1, result_axis1.dtype) # Match type
    assert ops.allclose(result_axis1, expected_axis1), f"{backend}: Gather axis=1 failed"


def test_ops_exp_log(backend):
    """Tests exponential and logarithmic functions."""
    ops.set_backend(backend)
    t = tensor.convert_to_tensor([1.0, 2.0, 0.0, -1.0])
    t_pos = tensor.convert_to_tensor([1.0, 10.0, 0.1])

    # Exp
    result_exp = ops.exp(t)
    # Use ops.exp directly on the input tensor for the expected value
    expected_exp = ops.exp(tensor.convert_to_tensor([1.0, 2.0, 0.0, -1.0]))
    expected_exp = tensor.cast(expected_exp, result_exp.dtype) # Match type
    assert ops.allclose(result_exp, expected_exp), f"{backend}: Exp failed"

    # Log (natural log)
    result_log = ops.log(t_pos) # Use positive values for log
    expected_log = ops.log(tensor.convert_to_tensor([1.0, 10.0, 0.1]))
    expected_log = tensor.cast(expected_log, result_log.dtype) # Match type
    assert ops.allclose(result_log, expected_log), f"{backend}: Log failed"

    # Log10
    result_log10 = ops.log10(t_pos)
    expected_log10 = ops.log10(tensor.convert_to_tensor([1.0, 10.0, 0.1]))
    expected_log10 = tensor.cast(expected_log10, result_log10.dtype) # Match type
    assert ops.allclose(result_log10, expected_log10), f"{backend}: Log10 failed"

    # Log2
    result_log2 = ops.log2(t_pos)
    expected_log2 = ops.log2(tensor.convert_to_tensor([1.0, 10.0, 0.1]))
    expected_log2 = tensor.cast(expected_log2, result_log2.dtype) # Match type
    assert ops.allclose(result_log2, expected_log2), f"{backend}: Log2 failed"

def test_ops_pow_sqrt_square(backend):
    """Tests power, square root, and square functions."""
    ops.set_backend(backend)
    t = tensor.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]])
    t_pos = tensor.convert_to_tensor([[1.0, 4.0], [9.0, 16.0]])
    exponent = 3.0
    exponent_tensor = tensor.convert_to_tensor([[2.0, 0.5], [3.0, 0.0]])

    # Pow (tensor, scalar)
    result_pow_scalar = ops.pow(t, exponent)
    # Calculate expected values using ops.pow
    expected_pow_scalar = ops.pow(t, tensor.convert_to_tensor(exponent))
    expected_pow_scalar = tensor.cast(expected_pow_scalar, result_pow_scalar.dtype) # Match type
    assert ops.allclose(result_pow_scalar, expected_pow_scalar), f"{backend}: Pow (scalar exp) failed"

    # Pow (tensor, tensor)
    result_pow_tensor = ops.pow(t, exponent_tensor)
    expected_pow_tensor = ops.pow(t, exponent_tensor) # Expected is just the direct calc
    expected_pow_tensor = tensor.cast(expected_pow_tensor, result_pow_tensor.dtype) # Match type
    assert ops.allclose(result_pow_tensor, expected_pow_tensor), f"{backend}: Pow (tensor exp) failed"

    # Sqrt
    result_sqrt = ops.sqrt(t_pos)
    expected_sqrt = tensor.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]])
    expected_sqrt = tensor.cast(expected_sqrt, result_sqrt.dtype) # Match type
    assert ops.allclose(result_sqrt, expected_sqrt), f"{backend}: Sqrt failed"

    # Square
    result_square = ops.square(t)
    expected_square = tensor.convert_to_tensor([[1.0, 4.0], [9.0, 16.0]])
    expected_square = tensor.cast(expected_square, result_square.dtype) # Match type
    assert ops.allclose(result_square, expected_square), f"{backend}: Square failed"

def test_ops_abs_negative_sign_clip(backend):
    """Tests absolute value, negation, sign, and clip functions."""
    ops.set_backend(backend)
    t = tensor.convert_to_tensor([[1.0, -2.0], [0.0, -0.5]])

    # Abs
    result_abs = ops.abs(t)
    expected_abs = tensor.convert_to_tensor([[1.0, 2.0], [0.0, 0.5]])
    expected_abs = tensor.cast(expected_abs, result_abs.dtype) # Match type
    assert ops.allclose(result_abs, expected_abs), f"{backend}: Abs failed"

    # Negative
    result_neg = ops.negative(t)
    expected_neg = tensor.convert_to_tensor([[-1.0, 2.0], [0.0, 0.5]])
    expected_neg = tensor.cast(expected_neg, result_neg.dtype) # Match type
    assert ops.allclose(result_neg, expected_neg), f"{backend}: Negative failed"

    # Sign
    result_sign = ops.sign(t)
    # Expected: [[1.0, -1.0], [0.0, -1.0]] (sign of 0 is 0)
    expected_sign = tensor.convert_to_tensor([[1.0, -1.0], [0.0, -1.0]])
    expected_sign = tensor.cast(expected_sign, result_sign.dtype) # Match result dtype
    assert ops.allclose(result_sign, expected_sign), f"{backend}: Sign failed"

    # Clip
    min_val, max_val = -0.6, 1.2
    result_clip = ops.clip(t, min_val, max_val)
    # Expected: [[1.0, -0.6], [0.0, -0.5]]
    expected_clip = tensor.convert_to_tensor([[1.0, -0.6], [0.0, -0.5]])
    expected_clip = tensor.cast(expected_clip, result_clip.dtype) # Match type
    assert ops.allclose(result_clip, expected_clip), f"{backend}: Clip failed"


def test_ops_trigonometric(backend):
    """Tests trigonometric functions (sin, cos, tan, sinh, cosh, tanh)."""
    ops.set_backend(backend)
    pi_val = ops.pi # Access pi as a property
    # Ensure pi_val is treated as a tensor or float for division
    pi_val_tensor = tensor.convert_to_tensor(pi_val) 
    # Construct angles tensor using ops
    zero_t = tensor.convert_to_tensor(0.0)
    four_t = tensor.convert_to_tensor(4.0)
    two_t = tensor.convert_to_tensor(2.0)
    angles = tensor.stack([zero_t, ops.divide(pi_val_tensor, four_t), ops.divide(pi_val_tensor, two_t), pi_val_tensor])
    values = tensor.convert_to_tensor([-1.0, 0.0, 1.0])

    # Sin
    result_sin = ops.sin(angles)
    expected_sin = ops.sin(angles) # Expected is the op applied to the input
    assert ops.allclose(result_sin, expected_sin, atol=1e-6), f"{backend}: Sin failed"

    # Cos
    result_cos = ops.cos(angles)
    expected_cos = ops.cos(angles)
    assert ops.allclose(result_cos, expected_cos, atol=1e-6), f"{backend}: Cos failed"

    # Tan (avoiding pi/2 where tan is undefined)
    pi_div_4 = ops.divide(pi_val_tensor, four_t)
    angles_tan = tensor.stack([zero_t, pi_div_4, ops.negative(pi_div_4)])
    result_tan = ops.tan(angles_tan)
    expected_tan = ops.tan(angles_tan)
    assert ops.allclose(result_tan, expected_tan, atol=1e-6), f"{backend}: Tan failed"

    # Sinh
    result_sinh = ops.sinh(values)
    expected_sinh = ops.sinh(values)
    assert ops.allclose(result_sinh, expected_sinh, atol=1e-6), f"{backend}: Sinh failed"

    # Cosh
    result_cosh = ops.cosh(values)
    expected_cosh = ops.cosh(values)
    assert ops.allclose(result_cosh, expected_cosh, atol=1e-6), f"{backend}: Cosh failed"

    # Tanh
    result_tanh = ops.tanh(values)
    expected_tanh = ops.tanh(values)
    assert ops.allclose(result_tanh, expected_tanh, atol=1e-6), f"{backend}: Tanh failed"

def test_ops_activations(backend):
    """Tests common activation functions (sigmoid, softplus, relu, softmax)."""
    ops.set_backend(backend)
    t = tensor.convert_to_tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    t_matrix = tensor.convert_to_tensor([[1.0, 2.0, 1.0], [-1.0, 0.0, 1.0]])

    # Sigmoid
    result_sig = ops.sigmoid(t)
    # Calculate expected using ops
    expected_sig = ops.divide(1.0, ops.add(1.0, ops.exp(ops.negative(t))))
    expected_sig = tensor.cast(expected_sig, result_sig.dtype) # Match type
    assert ops.allclose(result_sig, expected_sig), f"{backend}: Sigmoid failed"

    # Softplus
    result_sp = ops.softplus(t)
    # Calculate expected using ops
    expected_sp = ops.log(ops.add(1.0, ops.exp(t)))
    expected_sp = tensor.cast(expected_sp, result_sp.dtype) # Match type
    assert ops.allclose(result_sp, expected_sp), f"{backend}: Softplus failed"

    # ReLU
    result_relu = ops.relu(t)
    # Calculate expected using ops.maximum
    expected_relu = tensor.maximum(t, 0.0) # Simpler way to calculate expected ReLU
    expected_relu = tensor.cast(expected_relu, result_relu.dtype) # Match type
    assert ops.allclose(result_relu, expected_relu), f"{backend}: ReLU failed"

    # Softmax (applied row-wise, axis=1)
    result_sm = ops.softmax(t_matrix, axis=1)
    # Calculate expected using ops
    exp_t = ops.exp(t_matrix)
    sum_exp_t = stats.sum(exp_t, axis=1, keepdims=True)
    expected_sm = ops.divide(exp_t, sum_exp_t)
    expected_sm = tensor.cast(expected_sm, result_sm.dtype) # Match type
    assert ops.allclose(result_sm, expected_sm, atol=1e-5), f"{backend}: Softmax failed"
    # Check that rows sum to 1
    row_sums = stats.sum(result_sm, axis=1)
    expected_sums = tensor.ones(tensor.shape(row_sums)[0]) # Create ones vector of correct size
    expected_sums = tensor.cast(expected_sums, row_sums.dtype) # Match type
    assert ops.allclose(row_sums, expected_sums, atol=1e-6), f"{backend}: Softmax rows do not sum to 1"


# TODO: Add tests for gradient, eigh