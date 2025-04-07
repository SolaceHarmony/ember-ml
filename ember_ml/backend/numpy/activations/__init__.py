"""NumPy activation operations for ember_ml."""

# Removed NumpyActivationOps import
# from ember_ml.backend.numpy.activations.activations_ops import NumpyActivationOps
from ember_ml.backend.numpy.activations.ops import ( # Import from ops.py
    relu,
    sigmoid,
    tanh,
    softmax,
    softplus
)

__all__ = [
    # "NumpyActivationOps", # Removed class export
    "relu",
    "sigmoid",
    "tanh",
    "softmax",
    "softplus"
]