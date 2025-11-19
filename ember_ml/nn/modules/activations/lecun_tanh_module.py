# ember_ml/nn/modules/activations/lecun_tanh.py
"""
LeCun Tanh activation module.
"""
from ember_ml import ops
from ember_ml import tensor
from ember_ml.nn.modules import Module
from ember_ml.types import TensorLike


class LeCunTanh(Module):
    """
    Applies the LeCun improved Tanh function element-wise.
    LeCunTanh(x) = 1.7159 * tanh(0.6667 * x)
    """
    def __init__(self):
        super().__init__()
        # Precompute constants as tensors if ops require tensor inputs
        self.scale_factor = tensor.convert_to_tensor(0.66666667)
        self.amplitude = tensor.convert_to_tensor(1.7159)

    def forward(self, x: TensorLike) -> TensorLike:
        """Forward pass of LeCunTanh."""
        scaled_x = ops.multiply(self.scale_factor, x)
        from ember_ml.nn.modules.activations import tanh
        tanh_x = tanh(scaled_x)
        return ops.multiply(self.amplitude, tanh_x)

    def __repr__(self) -> str:
        return "LeCunTanh()"