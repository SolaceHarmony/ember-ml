# ember_ml/nn/modules/activations/__init__.py
"""
Activation function modules for ember_ml.

These modules wrap backend-agnostic activation functions from `ember_ml.ops`
into `ember_ml.nn.modules.Module` subclasses, allowing them to be used
seamlessly within `Sequential` containers and other module compositions.
"""

from .relu import ReLU
from .tanh import Tanh
from .sigmoid import Sigmoid
from .softmax import Softmax
from .softplus import Softplus
from .lecun_tanh import LeCunTanh
from .dropout import Dropout # Include Dropout here as well

__all__ = [
    "ReLU",
    "Tanh",
    "Sigmoid",
    "Softmax",
    "Softplus",
    "LeCunTanh",
    "Dropout",
]