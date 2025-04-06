# Activation Modules (nn.modules.activations)

The `ember_ml.nn.modules.activations` package provides a set of backend-agnostic activation function modules. These modules inherit from `ember_ml.nn.modules.Module` and wrap the corresponding functions from the `ember_ml.ops` module, allowing them to be easily integrated into neural network architectures, especially within `Sequential` containers.

## Importing

```python
from ember_ml.nn.modules.activations import ReLU, Tanh, Sigmoid, Softmax, Softplus, LeCunTanh, Dropout
# Or import the specific module needed
from ember_ml.nn.modules.activations import ReLU
```

## Available Activation Modules

### ReLU

`ReLU` implements the Rectified Linear Unit activation function.

```python
from ember_ml.nn.modules.activations import ReLU
from ember_ml.nn import tensor

relu_activation = ReLU()
input_tensor = tensor.convert_to_tensor([-1.0, 0.0, 1.0, 2.0])
output = relu_activation(input_tensor) # Output: [0., 0., 1., 2.]
```

### Tanh

`Tanh` implements the Hyperbolic Tangent activation function.

```python
from ember_ml.nn.modules.activations import Tanh
from ember_ml.nn import tensor

tanh_activation = Tanh()
input_tensor = tensor.convert_to_tensor([-1.0, 0.0, 1.0])
output = tanh_activation(input_tensor) # Output: [-0.76159, 0.        , 0.76159]
```

### Sigmoid

`Sigmoid` implements the Sigmoid activation function.

```python
from ember_ml.nn.modules.activations import Sigmoid
from ember_ml.nn import tensor

sigmoid_activation = Sigmoid()
input_tensor = tensor.convert_to_tensor([-1.0, 0.0, 1.0])
output = sigmoid_activation(input_tensor) # Output: [0.26894, 0.5      , 0.73105]
```

### Softmax

`Softmax` implements the Softmax activation function, typically used for multi-class classification output layers.

```python
from ember_ml.nn.modules.activations import Softmax
from ember_ml.nn import tensor

softmax_activation = Softmax(axis=-1)
input_tensor = tensor.convert_to_tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
output = softmax_activation(input_tensor)
# Output: [[0.09003, 0.24472, 0.66524],
#          [0.33333, 0.33333, 0.33333]]
```

**Arguments:**
*   `axis` (int): The axis along which the softmax normalization is applied. Defaults to -1.

### Softplus

`Softplus` implements the Softplus activation function: `log(exp(x) + 1)`.

```python
from ember_ml.nn.modules.activations import Softplus
from ember_ml.nn import tensor

softplus_activation = Softplus()
input_tensor = tensor.convert_to_tensor([-1.0, 0.0, 1.0])
output = softplus_activation(input_tensor) # Output: [0.31326, 0.69314, 1.31326]
```

### LeCunTanh

`LeCunTanh` implements the scaled hyperbolic tangent activation function proposed by LeCun et al. (1998): `1.7159 * tanh(2/3 * x)`.

```python
from ember_ml.nn.modules.activations import LeCunTanh
from ember_ml.nn import tensor

lecun_tanh_activation = LeCunTanh()
input_tensor = tensor.convert_to_tensor([-1.0, 0.0, 1.0])
output = lecun_tanh_activation(input_tensor) # Output: [-1.000..., 0.       , 1.000...] (approx)
```

### Dropout

`Dropout` implements the dropout regularization technique.

Dropout randomly sets a fraction `rate` of input units to 0 during training updates, helping prevent overfitting. Units not zeroed are scaled up by `1 / (1 - rate)` to maintain the expected sum. Dropout is only active when the `training=True` argument is passed to the `forward` call.

```python
from ember_ml.nn.modules.activations import Dropout
from ember_ml.nn import tensor

dropout_layer = Dropout(rate=0.5, seed=42)
input_tensor = tensor.ones((2, 2))

# During training
output_train = dropout_layer(input_tensor, training=True)
# Example Output (stochastic): [[2., 0.], [0., 2.]]

# During inference
output_eval = dropout_layer(input_tensor, training=False)
# Output: [[1., 1.], [1., 1.]]
```

**Arguments:**
*   `rate` (float): Fraction of input units to drop (0 <= rate < 1).
*   `seed` (Optional[int]): Seed for the random number generator for reproducibility.

## Usage with Sequential

Activation modules are commonly used within `Sequential` containers:

```python
from ember_ml.nn.container import Sequential
from ember_ml.nn.modules import Dense
from ember_ml.nn.modules.activations import ReLU, Dropout

model = Sequential([
    Dense(units=128),
    ReLU(),
    Dropout(0.2),
    Dense(units=10)
])

# Forward pass (training=True implicitly passed to Dropout within Sequential)
output = model(input_tensor, training=True)
```

## Backend Support

All activation modules are backend-agnostic and work with any backend (NumPy, PyTorch, MLX) selected via `ember_ml.backend.set_backend`.

## See Also

*   [Core NN Modules](nn_modules.md): Documentation on base modules and other core layers.
*   [Operations (ops)](ops.md): Documentation on the underlying backend-agnostic operations used by activations.