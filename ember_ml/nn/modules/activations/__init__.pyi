"""
Type stub file for ember_ml.nn.modules.activations module.

This provides explicit type hints for all dynamically aliased activation operations,
allowing type checkers to recognize the proper signatures of activation functions.
"""

from typing import Optional, Callable, Any, Union, Tuple, List

from ember_ml.nn.tensor.types import TensorLike

# Type alias for Tensor (similar to ops/__init__.pyi)
type Tensor = Any

# Module Classes
class ReLU:
    def __init__(self, inplace: bool = False) -> None: ...
    def __call__(self, x: TensorLike) -> Tensor: ...
    def forward(self, x: TensorLike) -> Tensor: ...

class Sigmoid:
    def __init__(self) -> None: ...
    def __call__(self, x: TensorLike) -> Tensor: ...
    def forward(self, x: TensorLike) -> Tensor: ...

class Tanh:
    def __init__(self) -> None: ...
    def __call__(self, x: TensorLike) -> Tensor: ...
    def forward(self, x: TensorLike) -> Tensor: ...

class Softmax:
    def __init__(self, axis: int = -1) -> None: ...
    def __call__(self, x: TensorLike) -> Tensor: ...
    def forward(self, x: TensorLike) -> Tensor: ...

class Softplus:
    def __init__(self) -> None: ...
    def __call__(self, x: TensorLike) -> Tensor: ...
    def forward(self, x: TensorLike) -> Tensor: ...

class LeCunTanh:
    def __init__(self) -> None: ...
    def __call__(self, x: TensorLike) -> Tensor: ...
    def forward(self, x: TensorLike) -> Tensor: ...

class Dropout:
    def __init__(self, rate: float = 0.5, training: bool = True) -> None: ...
    def __call__(self, x: TensorLike) -> Tensor: ...
    def forward(self, x: TensorLike) -> Tensor: ...

# Activation functions
def relu(x: TensorLike) -> Tensor: ...
def sigmoid(x: TensorLike) -> Tensor: ...
def tanh(x: TensorLike) -> Tensor: ...
def softmax(x: TensorLike, axis: int = -1) -> Tensor: ...
def softplus(x: TensorLike) -> Tensor: ...
def lecun_tanh(x: TensorLike) -> Tensor: ...

# Helper function
def get_activation(name: str) -> Callable[[TensorLike], Tensor]: ...

def _update_activation_aliases() -> None:
    """Dynamically updates this module's namespace with backend activation functions."""
    # This function is not used in this module but can be used for testing purposes
    # or to ensure that the backend module is imported correctly.
    pass