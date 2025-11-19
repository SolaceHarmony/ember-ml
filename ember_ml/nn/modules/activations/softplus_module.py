# ember_ml/nn/modules/activations/softplus.py
"""
Softplus activation module.
"""
from ember_ml.nn.modules import Module
from ember_ml.types import TensorLike

class Softplus(Module):
    """
    Applies the Softplus function element-wise.
    Softplus(x) = log(exp(x) + 1)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: TensorLike) -> TensorLike:
        """Forward pass of Softplus."""
        # Import lazily and call the backend-agnostic activation op
        from ember_ml.nn.modules.activations import softplus # Import from parent __init__
        return softplus(x)

    def __repr__(self) -> str:
        return "Softplus()"