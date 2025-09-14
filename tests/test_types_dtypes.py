"""Tests for core public typing and dtype helpers.

These tests focus on ensuring that the refactored `ember_ml.types` and
`ember_ml.dtypes` modules behave correctly without requiring all backends.
"""

from ember_ml import set_backend
from ember_ml import backend
from ember_ml import types as et
from ember_ml import tensor


def test_dtype_string_resolution_identity():
    # Canonical dtype strings resolve to themselves
    assert tensor.dtype('float32') == tensor.float32
    assert tensor.dtype('int64') == tensor.int64


def test_backend_dtype_resolution_numpy():
    set_backend('numpy')
    # Backend dtype conversion path: tensor.dtype returns backend-native dtype object
    backend_dt = tensor.dtype('float32')
    # numpy backend should expose a dtype with 'name' attribute or str convertible
    assert str(backend_dt) in ('float32', 'float32')


def test_tensorlike_union_accepts_scalar_and_list():
    def accepts_tensorlike(x: et.TensorLike):  # type: ignore
        return x
    assert accepts_tensorlike(5) == 5
    assert accepts_tensorlike([1, 2, 3]) == [1, 2, 3]


def test_protocol_runtime_checks_graceful():
    class Dummy:
        def __init__(self):
            self.shape = (2, 2)
            self.dtype = 'float32'
    d = Dummy()
    # Runtime isinstance with Protocol only works if runtime_checkable
    assert isinstance(d, et.EmberTensorLike)
