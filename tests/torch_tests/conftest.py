import pytest
from ember_ml.backend import load_backend_config

config = load_backend_config()
if not config.get('torch', False):
    pytest.skip('PyTorch backend disabled', allow_module_level=True)

try:
    import torch  # noqa: F401
except ImportError:
    pytest.skip('PyTorch not installed', allow_module_level=True)
