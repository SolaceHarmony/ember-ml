import pytest
import os
from ember_ml import ops
from ember_ml.nn import tensor

# Assume conftest.py provides 'backend' fixture

@pytest.fixture
def sample_tensor_io(backend, tmp_path):
    """Fixture for I/O tests."""
    ops.set_backend(backend)
    original_tensor = tensor.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]])
    file_path = tmp_path / f"test_tensor_{backend}.pkl" # Use .pkl for generic saving
    return original_tensor, file_path

def test_ops_save_load(sample_tensor_io, backend):
    """Tests ops.save and ops.load."""
    original_tensor, file_path = sample_tensor_io
    ops.set_backend(backend) # Ensure backend is set

    # Save the tensor
    ops.save(original_tensor, str(file_path))
    assert os.path.exists(file_path), f"File not saved at {file_path}"

    # Load the tensor
    loaded_tensor = ops.load(str(file_path))

    # Verify loaded tensor content and type
    assert isinstance(loaded_tensor, tensor.EmberTensor), "Loaded object is not an EmberTensor"
    assert ops.allclose(original_tensor, loaded_tensor), "Loaded tensor does not match original"

# --- Loss Function Tests ---

@pytest.fixture
def loss_data(backend):
    """Fixture for loss function tests."""
    ops.set_backend(backend)
    y_true = tensor.convert_to_tensor([[0.0, 1.0], [1.0, 0.0]])
    y_pred = tensor.convert_to_tensor([[0.1, 0.9], [0.8, 0.2]])
    return y_true, y_pred

def test_ops_mean_squared_error(loss_data, backend):
    """Tests ops.mean_squared_error."""
    ops.set_backend(backend)
    y_true, y_pred = loss_data
    mse = ops.mean_squared_error(y_true, y_pred)
    # Expected MSE: mean([(0-0.1)^2, (1-0.9)^2, (1-0.8)^2, (0-0.2)^2])
    # = mean([0.01, 0.01, 0.04, 0.04]) = mean([0.1]) = 0.025 # Correction: mean(0.01, 0.01, 0.04, 0.04) = 0.1/4 = 0.025
    expected_mse = tensor.convert_to_tensor(0.025)
    assert ops.allclose(mse, expected_mse, atol=1e-6), f"{backend}: MSE calculation failed"

def test_ops_mean_absolute_error(loss_data, backend):
    """Tests ops.mean_absolute_error."""
    ops.set_backend(backend)
    y_true, y_pred = loss_data
    mae = ops.mean_absolute_error(y_true, y_pred)
    # Expected MAE: mean([|0-0.1|, |1-0.9|, |1-0.8|, |0-0.2|])
    # = mean([0.1, 0.1, 0.2, 0.2]) = 0.6 / 4 = 0.15
    expected_mae = tensor.convert_to_tensor(0.15)
    assert ops.allclose(mae, expected_mae, atol=1e-6), f"{backend}: MAE calculation failed"

def test_ops_binary_crossentropy(loss_data, backend):
    """Tests ops.binary_crossentropy."""
    ops.set_backend(backend)
    y_true, y_pred = loss_data
    bce = ops.binary_crossentropy(y_true, y_pred)
    # Expected BCE: mean(-(y_true*log(y_pred) + (1-y_true)*log(1-y_pred)))
    # -(0*log(0.1) + 1*log(0.9)) = -log(0.9) approx 0.10536


@pytest.fixture
def ce_loss_data(backend):
    """Fixture for cross-entropy loss tests."""
    ops.set_backend(backend)
    # Categorical: One-hot true labels, probability predictions
    y_true_cat = tensor.convert_to_tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    y_pred_cat = tensor.convert_to_tensor([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]]) # Probabilities
    # Sparse Categorical: Integer true labels
    y_true_sparse = tensor.convert_to_tensor([1, 0]) # Indices corresponding to y_true_cat
    return y_true_cat, y_pred_cat, y_true_sparse

def test_ops_categorical_crossentropy(ce_loss_data, backend):
    """Tests ops.categorical_crossentropy."""
    ops.set_backend(backend)
    y_true_cat, y_pred_cat, _ = ce_loss_data
    cce = ops.categorical_crossentropy(y_true_cat, y_pred_cat)
    # Expected CCE: mean(-sum(y_true * log(y_pred), axis=-1))
    # Sample 1: -(0*log(0.1) + 1*log(0.8) + 0*log(0.1)) = -log(0.8) approx 0.22314
    # Sample 2: -(1*log(0.7) + 0*log(0.2) + 0*log(0.1)) = -log(0.7) approx 0.35667
    # Mean = (0.22314 + 0.35667) / 2 = 0.57981 / 2 = 0.2899
    expected_cce = tensor.convert_to_tensor(0.289905)
    assert ops.allclose(cce, expected_cce, atol=1e-5), f"{backend}: CCE calculation failed"

def test_ops_sparse_categorical_crossentropy(ce_loss_data, backend):
    """Tests ops.sparse_categorical_crossentropy."""
    ops.set_backend(backend)
    _, y_pred_cat, y_true_sparse = ce_loss_data
    scce = ops.sparse_categorical_crossentropy(y_true_sparse, y_pred_cat)
    # Should give the same result as CCE with one-hot labels
    # Sample 1: True label 1 -> uses log(y_pred[0,1]) = log(0.8) -> -log(0.8) approx 0.22314
    # Sample 2: True label 0 -> uses log(y_pred[1,0]) = log(0.7) -> -log(0.7) approx 0.35667
    # Mean = (0.22314 + 0.35667) / 2 = 0.2899
    expected_scce = tensor.convert_to_tensor(0.289905)
    assert ops.allclose(scce, expected_scce, atol=1e-5), f"{backend}: SCCE calculation failed"

def test_ops_huber_loss(loss_data, backend):
    """Tests ops.huber_loss."""
    ops.set_backend(backend)
    y_true, y_pred = loss_data
    delta = 1.0
    huber = ops.huber_loss(y_true, y_pred, delta=delta)
    # Error = y_true - y_pred = [[-0.1, 0.1], [0.2, -0.2]]
    # Abs Error = [[0.1, 0.1], [0.2, 0.2]]
    # Huber: 0.5 * error^2 if |error| <= delta, else delta * (|error| - 0.5 * delta)
    # All errors are <= 1.0, so use 0.5 * error^2
    # Huber values = [[0.5*0.01, 0.5*0.01], [0.5*0.04, 0.5*0.04]] = [[0.005, 0.005], [0.02, 0.02]]
    # Mean = (0.005 + 0.005 + 0.02 + 0.02) / 4 = 0.05 / 4 = 0.0125
    expected_huber = tensor.convert_to_tensor(0.0125)
    assert ops.allclose(huber, expected_huber, atol=1e-6), f"{backend}: Huber loss calculation failed"

def test_ops_log_cosh_loss(loss_data, backend):
    """Tests ops.log_cosh_loss."""
    ops.set_backend(backend)
    y_true, y_pred = loss_data
    logcosh = ops.log_cosh_loss(y_true, y_pred)
    # Error = y_pred - y_true = [[0.1, -0.1], [-0.2, 0.2]]
    # log(cosh(error))
    # cosh(0.1) approx 1.005, log(1.005) approx 0.004987
    # cosh(-0.1) approx 1.005, log(1.005) approx 0.004987
    # cosh(-0.2) approx 1.020, log(1.020) approx 0.01980
    # cosh(0.2) approx 1.020, log(1.020) approx 0.01980
    # Mean = (0.004987 * 2 + 0.01980 * 2) / 4 = (0.009974 + 0.0396) / 4 = 0.049574 / 4 = 0.01239
    expected_logcosh = tensor.convert_to_tensor(0.0123936)
    assert ops.allclose(logcosh, expected_logcosh, atol=1e-5), f"{backend}: LogCosh loss calculation failed"


    # -(1*log(0.9) + 0*log(0.1)) = -log(0.9) approx 0.10536
    # -(1*log(0.8) + 0*log(0.2)) = -log(0.8) approx 0.22314
    # -(0*log(0.2) + 1*log(0.8)) = -log(0.8) approx 0.22314
    # Mean = (0.10536 + 0.10536 + 0.22314 + 0.22314) / 4 = 0.657 / 4 = 0.16425
    expected_bce = tensor.convert_to_tensor(0.16425)
    assert ops.allclose(bce, expected_bce, atol=1e-4), f"{backend}: BCE calculation failed"

# TODO: Add tests for categorical_crossentropy, sparse_categorical_crossentropy, huber_loss, log_cosh_loss
# TODO: Add tests for vector operations (normalize_vector, etc.) - potentially in a separate file?