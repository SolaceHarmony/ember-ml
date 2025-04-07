"""PyTorch activation operations for ember_ml."""

# Removed TorchActivationOps import
# from ember_ml.backend.torch.activations.activations_ops import TorchActivationOps
from ember_ml.backend.torch.activations.ops import ( # Import from ops.py
    relu,
    sigmoid,
    tanh,
    softmax,
    softplus
)

__all__ = [
    # "TorchActivationOps", # Removed class export
    "relu",
    "sigmoid",
    "tanh",
    "softmax",
    "softplus"
]