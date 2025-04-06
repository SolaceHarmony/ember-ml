# ember_ml/nn/modules/activations/softplus.py
"""
Softplus activation module.
"""
from ember_ml import ops
from ember_ml.nn.modules import Module
from ember_ml.nn import tensor

class Softplus(Module):
    """
    Applies the Softplus function element-wise.
    Softplus(x) = log(exp(x) + 1)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: tensor.EmberTensor) -> tensor.EmberTensor:
        """Forward pass of Softplus."""
        return ops.softplus(x)

    def __repr__(self) -> str:
        return "Softplus()"