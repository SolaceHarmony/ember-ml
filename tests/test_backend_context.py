import pytest
from ember_ml.backend import (
    get_backend,
    set_backend,
    get_available_backends,
    using_backend,
)


def test_get_available_backends_returns_list():
    backends = get_available_backends()
    assert isinstance(backends, list)
    assert 'numpy' in backends


def test_using_backend_switches_and_restores():
    original = get_backend()
    set_backend('numpy')
    backends = get_available_backends()
    target = next((b for b in backends if b != 'numpy'), 'numpy')
    with using_backend(target):
        assert get_backend() == target
    assert get_backend() == 'numpy'
    if original and original != 'numpy':
        set_backend(original)
