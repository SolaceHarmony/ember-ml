import numpy as np
from ember_ml import ops
from ember_ml.nn.modules import NCP
from ember_ml.nn.wirings import NCPWiring

# Create a wiring
wiring = NCPWiring(
    inter_neurons=10,
    motor_neurons=5,
    sensory_neurons=0,
    sparsity_level=0.5,
    input_dim=15
)

# Create an NCP module
ncp = NCP(
    wiring=wiring,
    activation="tanh",
    use_bias=True
)

# Print the NCP's parameters
print("NCP parameters:")
for name, param in ncp._parameters.items():
    print(f"- {name}: {param}")

# Try to access parameters directly
print("\nAccessing parameters directly:")
print(f"- kernel: {ncp.kernel}")
print(f"- recurrent_kernel: {ncp.recurrent_kernel}")
print(f"- bias: {ncp.bias}")

# Create input
batch_size = 2
inputs = ops.random_normal((batch_size, 15))

# Forward pass
output = ncp(inputs)

print(f"\nInput shape: {ops.shape(inputs)}")
print(f"Output shape: {ops.shape(output)}")