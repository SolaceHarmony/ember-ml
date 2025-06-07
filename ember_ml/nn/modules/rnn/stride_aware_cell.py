"""Basic stride-aware recurrent cell."""

from typing import Optional, Tuple

from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn.modules import Module, Parameter
from ember_ml.nn.modules.activations import get_activation
from ember_ml.nn.initializers import glorot_uniform


class StrideAwareCell(Module):
    """Simple recurrent cell that processes inputs at a fixed stride."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        stride_length: int = 1,
        time_scale_factor: float = 1.0,
        activation: str = "tanh",
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.stride_length = stride_length
        self.time_scale_factor = time_scale_factor
        self.activation = activation
        self.use_bias = use_bias

        self.input_kernel = Parameter(glorot_uniform((input_size, hidden_size)))
        self.recurrent_kernel = Parameter(glorot_uniform((hidden_size, hidden_size)))
        if use_bias:
            self.bias = Parameter(tensor.zeros((hidden_size,)))
        else:
            self.bias = None

    def forward(self, inputs, state: Optional[tensor.EmberTensor] = None) -> Tuple[tensor.EmberTensor, tensor.EmberTensor]:
        if state is None:
            batch_size = tensor.shape(inputs)[0]
            state = tensor.zeros((batch_size, self.hidden_size))

        output = ops.matmul(inputs, self.input_kernel.data)
        output = ops.add(output, ops.matmul(state, self.recurrent_kernel.data))
        if self.bias is not None:
            output = ops.add(output, self.bias.data)

        activation_fn = get_activation(self.activation)
        new_state = activation_fn(output)
        return new_state, new_state
