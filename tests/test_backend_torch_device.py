import pytest

pytest.skip('Torch backend example script, not a real test', allow_module_level=True)

from ember_ml.backend import get_backend, set_backend
set_backend('torch')
from ember_ml import tensor
t = tensor([[1, 2, 3], [4, 5, 6]])
print(f'Tensor shape: {t.shape}, Tensor dtype: {t.dtype}, Tensor device: {t.device}')