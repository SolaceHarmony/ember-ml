import pytest
from pytest import mark
import os
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn.tensor.common.ember_tensor import EmberTensor # For isinstance checks

# Define the backend order: numpy -> torch -> mlx

# Helper function to get sample tensor for I/O
def _get_sample_tensor_io(tmp_path, backend_name):
    original_tensor = tensor.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]])
    file_path = tmp_path / f"test_tensor_{backend_name}.pkl"
    return original_tensor, file_path

# Helper function to get loss data
def _get_loss_data():
    y_true = tensor.convert_to_tensor([[0.0, 1.0], [1.0, 0.0]])
    y_pred = tensor.convert_to_tensor([[0.1, 0.9], [0.8, 0.2]])
    return y_true, y_pred

# Helper function to get cross-entropy loss data
def _get_ce_loss_data():
    y_true_cat = tensor.convert_to_tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    y_pred_cat = tensor.convert_to_tensor([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]])
    y_true_sparse = tensor.convert_to_tensor([1, 0])
    return y_true_cat, y_pred_cat, y_true_sparse

# --- NumPy Backend Setup & Tests ---
@mark.run(order=1)
def test_setup_numpy():
    """Set up the NumPy backend."""
    print("\n=== Setting backend to NumPy ===")
    ops.set_backend('numpy')
    assert ops.get_backend() == 'numpy'

@mark.run(order=1)
def test_ops_save_load_numpy(tmp_path):
    """Tests ops.save and ops.load with NumPy backend."""
    original_tensor, file_path = _get_sample_tensor_io(tmp_path, 'numpy')
    ops.save(original_tensor, str(file_path))
    assert os.path.exists(file_path), f"File not saved at {file_path}"
    loaded_tensor = ops.load(str(file_path))
    assert isinstance(loaded_tensor, EmberTensor), "Loaded object not EmberTensor"
    assert ops.allclose(original_tensor, loaded_tensor), "Loaded tensor mismatch"

@mark.run(order=1)
def test_ops_mean_squared_error_numpy():
    """Tests ops.mean_squared_error with NumPy backend."""
    y_true, y_pred = _get_loss_data()
    mse = ops.mean_squared_error(y_true, y_pred)
    expected_mse = tensor.convert_to_tensor(0.025)
    assert ops.allclose(mse, expected_mse, atol=1e-6), "MSE calculation failed"

@mark.run(order=1)
def test_ops_mean_absolute_error_numpy():
    """Tests ops.mean_absolute_error with NumPy backend."""
    y_true, y_pred = _get_loss_data()
    mae = ops.mean_absolute_error(y_true, y_pred)
    expected_mae = tensor.convert_to_tensor(0.15)
    assert ops.allclose(mae, expected_mae, atol=1e-6), "MAE calculation failed"

@mark.run(order=1)
def test_ops_binary_crossentropy_numpy():
    """Tests ops.binary_crossentropy with NumPy backend."""
    y_true, y_pred = _get_loss_data()
    bce = ops.binary_crossentropy(y_true, y_pred)
    expected_bce = tensor.convert_to_tensor(0.16425)
    assert ops.allclose(bce, expected_bce, atol=1e-4), "BCE calculation failed"

@mark.run(order=1)
def test_ops_categorical_crossentropy_numpy():
    """Tests ops.categorical_crossentropy with NumPy backend."""
    y_true_cat, y_pred_cat, _ = _get_ce_loss_data()
    cce = ops.categorical_crossentropy(y_true_cat, y_pred_cat)
    expected_cce = tensor.convert_to_tensor(0.289905)
    assert ops.allclose(cce, expected_cce, atol=1e-5), "CCE calculation failed"

@mark.run(order=1)
def test_ops_sparse_categorical_crossentropy_numpy():
    """Tests ops.sparse_categorical_crossentropy with NumPy backend."""
    _, y_pred_cat, y_true_sparse = _get_ce_loss_data()
    scce = ops.sparse_categorical_crossentropy(y_true_sparse, y_pred_cat)
    expected_scce = tensor.convert_to_tensor(0.289905)
    assert ops.allclose(scce, expected_scce, atol=1e-5), "SCCE calculation failed"

@mark.run(order=1)
def test_ops_huber_loss_numpy():
    """Tests ops.huber_loss with NumPy backend."""
    y_true, y_pred = _get_loss_data()
    delta = 1.0
    huber = ops.huber_loss(y_true, y_pred, delta=delta)
    expected_huber = tensor.convert_to_tensor(0.0125)
    assert ops.allclose(huber, expected_huber, atol=1e-6), "Huber loss calculation failed"

@mark.run(order=1)
def test_ops_log_cosh_loss_numpy():
    """Tests ops.log_cosh_loss with NumPy backend."""
    y_true, y_pred = _get_loss_data()
    logcosh = ops.log_cosh_loss(y_true, y_pred)
    expected_logcosh = tensor.convert_to_tensor(0.0123936)
    assert ops.allclose(logcosh, expected_logcosh, atol=1e-5), "LogCosh loss calculation failed"


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
def test_ops_save_load_torch(tmp_path):
    """Tests ops.save and ops.load with PyTorch backend."""
    original_tensor, file_path = _get_sample_tensor_io(tmp_path, 'torch')
    ops.save(original_tensor, str(file_path))
    assert os.path.exists(file_path), f"File not saved at {file_path}"
    loaded_tensor = ops.load(str(file_path))
    assert isinstance(loaded_tensor, EmberTensor), "Loaded object not EmberTensor"
    assert ops.allclose(original_tensor, loaded_tensor), "Loaded tensor mismatch"

@mark.run(order=2)
def test_ops_mean_squared_error_torch():
    """Tests ops.mean_squared_error with PyTorch backend."""
    y_true, y_pred = _get_loss_data()
    mse = ops.mean_squared_error(y_true, y_pred)
    expected_mse = tensor.convert_to_tensor(0.025)
    assert ops.allclose(mse, expected_mse, atol=1e-6), "MSE calculation failed"

@mark.run(order=2)
def test_ops_mean_absolute_error_torch():
    """Tests ops.mean_absolute_error with PyTorch backend."""
    y_true, y_pred = _get_loss_data()
    mae = ops.mean_absolute_error(y_true, y_pred)
    expected_mae = tensor.convert_to_tensor(0.15)
    assert ops.allclose(mae, expected_mae, atol=1e-6), "MAE calculation failed"

@mark.run(order=2)
def test_ops_binary_crossentropy_torch():
    """Tests ops.binary_crossentropy with PyTorch backend."""
    y_true, y_pred = _get_loss_data()
    bce = ops.binary_crossentropy(y_true, y_pred)
    expected_bce = tensor.convert_to_tensor(0.16425)
    assert ops.allclose(bce, expected_bce, atol=1e-4), "BCE calculation failed"

@mark.run(order=2)
def test_ops_categorical_crossentropy_torch():
    """Tests ops.categorical_crossentropy with PyTorch backend."""
    y_true_cat, y_pred_cat, _ = _get_ce_loss_data()
    cce = ops.categorical_crossentropy(y_true_cat, y_pred_cat)
    expected_cce = tensor.convert_to_tensor(0.289905)
    assert ops.allclose(cce, expected_cce, atol=1e-5), "CCE calculation failed"

@mark.run(order=2)
def test_ops_sparse_categorical_crossentropy_torch():
    """Tests ops.sparse_categorical_crossentropy with PyTorch backend."""
    _, y_pred_cat, y_true_sparse = _get_ce_loss_data()
    scce = ops.sparse_categorical_crossentropy(y_true_sparse, y_pred_cat)
    expected_scce = tensor.convert_to_tensor(0.289905)
    assert ops.allclose(scce, expected_scce, atol=1e-5), "SCCE calculation failed"

@mark.run(order=2)
def test_ops_huber_loss_torch():
    """Tests ops.huber_loss with PyTorch backend."""
    y_true, y_pred = _get_loss_data()
    delta = 1.0
    huber = ops.huber_loss(y_true, y_pred, delta=delta)
    expected_huber = tensor.convert_to_tensor(0.0125)
    assert ops.allclose(huber, expected_huber, atol=1e-6), "Huber loss calculation failed"

@mark.run(order=2)
def test_ops_log_cosh_loss_torch():
    """Tests ops.log_cosh_loss with PyTorch backend."""
    y_true, y_pred = _get_loss_data()
    logcosh = ops.log_cosh_loss(y_true, y_pred)
    expected_logcosh = tensor.convert_to_tensor(0.0123936)
    assert ops.allclose(logcosh, expected_logcosh, atol=1e-5), "LogCosh loss calculation failed"


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
def test_ops_save_load_mlx(tmp_path):
    """Tests ops.save and ops.load with MLX backend."""
    original_tensor, file_path = _get_sample_tensor_io(tmp_path, 'mlx')
    ops.save(original_tensor, str(file_path))
    assert os.path.exists(file_path), f"File not saved at {file_path}"
    loaded_tensor = ops.load(str(file_path))
    assert isinstance(loaded_tensor, EmberTensor), "Loaded object not EmberTensor"
    assert ops.allclose(original_tensor, loaded_tensor), "Loaded tensor mismatch"

@mark.run(order=3)
def test_ops_mean_squared_error_mlx():
    """Tests ops.mean_squared_error with MLX backend."""
    y_true, y_pred = _get_loss_data()
    mse = ops.mean_squared_error(y_true, y_pred)
    expected_mse = tensor.convert_to_tensor(0.025)
    assert ops.allclose(mse, expected_mse, atol=1e-6), "MSE calculation failed"

@mark.run(order=3)
def test_ops_mean_absolute_error_mlx():
    """Tests ops.mean_absolute_error with MLX backend."""
    y_true, y_pred = _get_loss_data()
    mae = ops.mean_absolute_error(y_true, y_pred)
    expected_mae = tensor.convert_to_tensor(0.15)
    assert ops.allclose(mae, expected_mae, atol=1e-6), "MAE calculation failed"

@mark.run(order=3)
def test_ops_binary_crossentropy_mlx():
    """Tests ops.binary_crossentropy with MLX backend."""
    y_true, y_pred = _get_loss_data()
    bce = ops.binary_crossentropy(y_true, y_pred)
    expected_bce = tensor.convert_to_tensor(0.16425)
    assert ops.allclose(bce, expected_bce, atol=1e-4), "BCE calculation failed"

@mark.run(order=3)
def test_ops_categorical_crossentropy_mlx():
    """Tests ops.categorical_crossentropy with MLX backend."""
    y_true_cat, y_pred_cat, _ = _get_ce_loss_data()
    cce = ops.categorical_crossentropy(y_true_cat, y_pred_cat)
    expected_cce = tensor.convert_to_tensor(0.289905)
    assert ops.allclose(cce, expected_cce, atol=1e-5), "CCE calculation failed"

@mark.run(order=3)
def test_ops_sparse_categorical_crossentropy_mlx():
    """Tests ops.sparse_categorical_crossentropy with MLX backend."""
    _, y_pred_cat, y_true_sparse = _get_ce_loss_data()
    scce = ops.sparse_categorical_crossentropy(y_true_sparse, y_pred_cat)
    expected_scce = tensor.convert_to_tensor(0.289905)
    assert ops.allclose(scce, expected_scce, atol=1e-5), "SCCE calculation failed"

@mark.run(order=3)
def test_ops_huber_loss_mlx():
    """Tests ops.huber_loss with MLX backend."""
    y_true, y_pred = _get_loss_data()
    delta = 1.0
    huber = ops.huber_loss(y_true, y_pred, delta=delta)
    expected_huber = tensor.convert_to_tensor(0.0125)
    assert ops.allclose(huber, expected_huber, atol=1e-6), "Huber loss calculation failed"

@mark.run(order=3)
def test_ops_log_cosh_loss_mlx():
    """Tests ops.log_cosh_loss with MLX backend."""
    y_true, y_pred = _get_loss_data()
    logcosh = ops.log_cosh_loss(y_true, y_pred)
    expected_logcosh = tensor.convert_to_tensor(0.0123936)
    assert ops.allclose(logcosh, expected_logcosh, atol=1e-5), "LogCosh loss calculation failed"