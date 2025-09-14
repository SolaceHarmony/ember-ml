import pytest
import numpy as np

from ember_ml.utils import backend_utils
from ember_ml import tensor, ops
from ember_ml.ops import set_backend, get_backend


@pytest.mark.parametrize("backend", ["numpy", "torch", "mlx"])
def test_sin_cos_transform_across_backends(backend):
    """Ensure sin_cos_transform works for multiple backends."""
    if backend == "mlx":
        pytest.importorskip("mlx.core")
    original_backend = get_backend()
    set_backend(backend)
    try:
        values = [0.0, 0.25, 0.5, 0.75]
        sin_vals, cos_vals = backend_utils.sin_cos_transform(values, period=1.0)

        expected_sin = np.sin(2 * np.pi * np.array(values))
        expected_cos = np.cos(2 * np.pi * np.array(values))

        assert np.allclose(tensor.to_numpy(sin_vals), expected_sin, atol=1e-6)
        assert np.allclose(tensor.to_numpy(cos_vals), expected_cos, atol=1e-6)

        input_tensor = tensor.convert_to_tensor(values)
        assert ops.get_device(sin_vals) == ops.get_device(input_tensor)
        assert ops.get_device(cos_vals) == ops.get_device(input_tensor)
        assert ops.dtype(sin_vals) == ops.dtype(input_tensor)
        assert ops.dtype(cos_vals) == ops.dtype(input_tensor)
    finally:
        set_backend(original_backend)
