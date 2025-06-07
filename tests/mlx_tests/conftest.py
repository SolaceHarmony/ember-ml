import pytest
from ember_ml.backend import load_backend_config

config = load_backend_config()
if not config.get('mlx', False):
    pytest.skip('MLX backend disabled', allow_module_level=True)

try:
    import mlx.core  # noqa: F401
except ImportError:
    pytest.skip('MLX not installed', allow_module_level=True)
